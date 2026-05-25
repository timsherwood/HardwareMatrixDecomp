"""Simulation 2: Quantized delay model.

Models the accuracy impact of two hardware quantization effects:

  n_levels  — number of discrete conductance states → delay bins in [d_min, d_max]
              (e.g. 16, 32, 64, 128 levels)
  tdc_res   — TDC measurement resolution in ns (e.g. 0.5, 1.0, 2.0 ns)

Both are simulated as post-training quantization: train with continuous
delays, snap each delay cell to its nearest hardware-realizable level,
then re-evaluate.  This gives the worst-case accuracy floor; quantization-
aware retraining would recover some of the gap.

Conductance quantization is applied before TDC quantization when both
are specified, since the TDC resolution determines how precisely we can
hit the conductance target during program-and-verify.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from memristor.network import ComplementaryDelayLayer, MemristorNet
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def quantize_delays(
    d: torch.Tensor,
    d_min: float,
    d_max: float,
    n_levels: int | None = None,
    tdc_res: float | None = None,
) -> torch.Tensor:
    """Snap continuous delays to hardware-realizable levels.

    Parameters
    ----------
    d:
        Delay tensor (ns), expected in [d_min, d_max].
    n_levels:
        Number of uniform delay bins.  step = (d_max - d_min) / (n_levels - 1).
        None means no conductance quantization.
    tdc_res:
        TDC bin width (ns).  Delays are snapped to multiples of tdc_res.
        None means no TDC quantization.

    Conductance quantization is applied first; result is clamped to [d_min, d_max].
    """
    result = d.clone().float()
    if n_levels is not None and n_levels > 1:
        step = (d_max - d_min) / (n_levels - 1)
        result = torch.round((result - d_min) / step) * step + d_min
        result = torch.clamp(result, d_min, d_max)
    if tdc_res is not None and tdc_res > 0.0:
        result = torch.round(result / tdc_res) * tdc_res
        result = torch.clamp(result, d_min, d_max)
    return result


def quantize_complementary(
    d_pos: torch.Tensor,
    d_min: float,
    d_max: float,
    n_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap d_pos to its nearest level and derive d_neg from the complementary constraint.

    For a complementary pair the level indices satisfy:

        level_pos + level_neg = n_levels - 1

    so:

        d_pos_q[k]  = d_min + k * step           (k = 0 … n_levels-1)
        d_neg_q[k]  = d_min + (n_levels-1-k) * step  = d_max - k * step

    which is equivalent to ``d_neg_q = d_min + d_max - d_pos_q``.

    Parameters
    ----------
    d_pos:
        Continuous positive-branch delays (ns).
    d_min, d_max:
        Delay range.
    n_levels:
        Number of discrete levels (must be >= 2).

    Returns
    -------
    (d_pos_q, d_neg_q) — quantized pair satisfying d_pos_q + d_neg_q = d_min + d_max.
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    step = (d_max - d_min) / (n_levels - 1)
    # Snap d_pos to nearest level
    level_pos = torch.round((d_pos.clamp(d_min, d_max) - d_min) / step).long()
    level_pos = level_pos.clamp(0, n_levels - 1)
    # Complementary: level_neg = n_levels - 1 - level_pos
    level_neg = (n_levels - 1) - level_pos
    d_pos_q = d_min + level_pos.float() * step
    d_neg_q = d_min + level_neg.float() * step
    return d_pos_q, d_neg_q


def make_quantized_net(
    net: MemristorNet,
    n_levels: int | None = None,
    tdc_res: float | None = None,
) -> MemristorNet:
    """Return a deep copy of net with all delay parameters quantized.

    For DelayLayer: u_pos/u_neg are set so that delays() returns the
    quantized delay values independently.

    For ComplementaryDelayLayer: d_pos is quantized with the complementary
    constraint (``quantize_complementary``), and u is set from d_pos_q.

    No gradients are required in the returned copy.
    """
    q_net = copy.deepcopy(net)
    with torch.no_grad():
        for orig_layer, q_layer in zip(net.layers, q_net.layers, strict=True):
            d_pos, d_neg = orig_layer.delays()
            kappa = torch.tensor(orig_layer.kappa, dtype=torch.float32)

            if isinstance(orig_layer, ComplementaryDelayLayer):
                # Use complementary quantization if n_levels is given
                if n_levels is not None and n_levels >= 2:
                    d_pos_q, _d_neg_q = quantize_complementary(
                        d_pos, orig_layer.d_min, orig_layer.d_max, n_levels
                    )
                else:
                    d_pos_q = d_pos.clone()
                if tdc_res is not None and tdc_res > 0.0:
                    d_pos_q = torch.round(d_pos_q / tdc_res) * tdc_res
                    d_pos_q = d_pos_q.clamp(orig_layer.d_min, orig_layer.d_max)
                # u = ln(kappa / d_pos)
                q_layer.u.data = torch.log(kappa / d_pos_q)  # type: ignore[attr-defined]
            else:
                # Standard independent quantization for DelayLayer
                d_pos_q = quantize_delays(
                    d_pos, orig_layer.d_min, orig_layer.d_max, n_levels, tdc_res
                )
                d_neg_q = quantize_delays(
                    d_neg, orig_layer.d_min, orig_layer.d_max, n_levels, tdc_res
                )
                # Invert d = kappa * exp(-u)  →  u = ln(kappa / d)
                q_layer.u_pos.data = torch.log(kappa / d_pos_q)
                q_layer.u_neg.data = torch.log(kappa / d_neg_q)
    return q_net


@dataclass
class QuantizationSweepResult:
    """Results from a post-training quantization accuracy sweep."""

    n_trials: int
    n_converged: int
    baseline_rate: float
    # grid[(n_levels, tdc_res)] = fraction of converged seeds still 100% accurate
    accuracy_grid: dict[tuple[int | None, float | None], float] = field(default_factory=dict)
    # mean fraction of XOR patterns correct (0–1) per config
    mean_xor_grid: dict[tuple[int | None, float | None], float] = field(default_factory=dict)


def quantization_accuracy_sweep(
    n_trials: int = 20,
    n_epochs: int = 1500,
    eta: float = 0.06,
    levels_grid: tuple[int, ...] = (16, 32, 64, 128),
    tdc_grid: tuple[float, ...] = (0.5, 1.0, 2.0),
    seed_offset: int = 0,
    verbose: bool = False,
) -> QuantizationSweepResult:
    """Train n_trials seeds; evaluate post-quantization XOR accuracy.

    For each converged seed, snaps all delays to each (n_levels, tdc_res)
    combination and measures accuracy without any retraining.

    Parameters
    ----------
    n_trials:
        Number of random seeds to train.
    n_epochs:
        Training epochs per seed.
    eta:
        Learning rate for MemristorTrainer.
    levels_grid:
        Conductance quantization levels to sweep.
    tdc_grid:
        TDC resolution values (ns) to sweep.
    seed_offset:
        First seed index (useful for reproducibility across scripts).
    verbose:
        Print per-seed training progress.
    """
    converged_nets: list[MemristorNet] = []

    for i in range(n_trials):
        seed = seed_offset + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
        trainer = MemristorTrainer(net, eta=eta)
        trainer.fit(XOR_X, XOR_Y, n_epochs=n_epochs)
        acc = trainer.accuracy(XOR_X, XOR_Y)
        converged = acc == 1.0
        if verbose:
            print(f"  seed {seed:3d}  {'CONVERGED' if converged else f'acc={acc:.0%}'}")
        if converged:
            converged_nets.append(net)

    n_converged = len(converged_nets)
    baseline_rate = n_converged / n_trials

    result = QuantizationSweepResult(
        n_trials=n_trials,
        n_converged=n_converged,
        baseline_rate=baseline_rate,
    )

    if n_converged == 0:
        return result

    for n_levels in levels_grid:
        for tdc_res in tdc_grid:
            key = (n_levels, tdc_res)
            full_acc_count = 0
            xor_fracs = []
            for net in converged_nets:
                q_net = make_quantized_net(net, n_levels=n_levels, tdc_res=tdc_res)
                with torch.no_grad():
                    preds = [int(float(q_net.predict(x)[0]) > 0.5) for x in XOR_X]
                correct = sum(p == int(y) for p, y in zip(preds, XOR_Y, strict=True))
                xor_fracs.append(correct / len(XOR_Y))
                if correct == len(XOR_Y):
                    full_acc_count += 1
            result.accuracy_grid[key] = full_acc_count / n_converged
            result.mean_xor_grid[key] = float(np.mean(xor_fracs))

    return result


def print_sweep_table(result: QuantizationSweepResult) -> None:
    """Print a formatted table of post-quantization accuracy."""
    levels_seen = sorted({k[0] for k in result.accuracy_grid if k[0] is not None})
    tdc_seen = sorted({k[1] for k in result.accuracy_grid if k[1] is not None})

    print()
    print("Post-Training Delay Quantization — XOR Accuracy")
    print("=" * 62)
    print(
        f"Baseline: {result.n_converged}/{result.n_trials} seeds converged"
        f" ({result.baseline_rate:.0%})"
    )
    print()
    print("Fraction of converged seeds still 100% accurate after quantization:")
    print()

    col_w = 14
    header = f"{'':12s}" + "".join(f"{'n=' + str(n):>{col_w}}" for n in levels_seen)
    print(header)
    print("-" * len(header))

    for tdc_res in tdc_seen:
        row = f"tdc={tdc_res:.1f}ns  "
        for n_levels in levels_seen:
            val = result.accuracy_grid.get((n_levels, tdc_res), float("nan"))
            row += f"{val:>{col_w}.0%}"
        print(row)

    print()
    print("Mean XOR pattern accuracy (0–4 correct / 4):")
    print()
    print(header)
    print("-" * len(header))

    for tdc_res in tdc_seen:
        row = f"tdc={tdc_res:.1f}ns  "
        for n_levels in levels_seen:
            val = result.mean_xor_grid.get((n_levels, tdc_res), float("nan"))
            row += f"{val:>{col_w}.2f}"
        print(row)

    # Effective delay resolution for reference
    print()
    print("Reference: effective delay step per config (ns)")
    print(f"  d range = [5, 50] ns  ({45:.0f} ns span)")
    d_span = 50.0 - 5.0
    for n_levels in levels_seen:
        step = d_span / (n_levels - 1)
        print(f"  n_levels={n_levels:3d}  conductance step = {step:.2f} ns")
