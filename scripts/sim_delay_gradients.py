"""Sim: Delay-space gradient analysis and complementary encoding comparison.

Trains an XOR network (seed=0, 2000 epochs), snapshots gradients at representative
epochs, and compares standard vs. complementary encoding convergence rates.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from memristor.gradient_analysis import (
    extract_delay_gradients,
    gradient_active_fraction,
    gradient_summary,
)
from memristor.network import MemristorNet
from memristor.training import MemristorTrainer

# ---------------------------------------------------------------------------
# XOR dataset
# ---------------------------------------------------------------------------
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

SNAPSHOT_EPOCHS = [1, 100, 500, 2000]


def gradient_half_width(grads: list[dict[str, np.ndarray]], d_bins: int = 100) -> float:
    """Return the delay range (ns) that captures 80% of total |delta_d| magnitude.

    Bins all delay cells by their d value (across pos and neg) and accumulates
    |delta_d| per bin.  Returns the width of the smallest contiguous delay interval
    that contains 80% of the total |delta_d| mass.
    """
    all_d: list[float] = []
    all_delta: list[float] = []
    for g in grads:
        all_d.extend(g["d_pos"].ravel().tolist())
        all_d.extend(g["d_neg"].ravel().tolist())
        all_delta.extend(np.abs(g["delta_d_pos"]).ravel().tolist())
        all_delta.extend(np.abs(g["delta_d_neg"]).ravel().tolist())

    if not all_d:
        return 0.0

    d_arr = np.array(all_d)
    delta_arr = np.array(all_delta)
    total = float(delta_arr.sum())
    if total == 0.0:
        return 0.0

    d_min_v, d_max_v = float(d_arr.min()), float(d_arr.max())
    if d_min_v == d_max_v:
        return 0.0

    bins = np.linspace(d_min_v, d_max_v, d_bins + 1)
    bin_idx = np.digitize(d_arr, bins) - 1
    bin_idx = np.clip(bin_idx, 0, d_bins - 1)
    bin_mass = np.zeros(d_bins)
    for i, m in zip(bin_idx, delta_arr, strict=True):
        bin_mass[i] += m

    target = 0.80 * total
    best_width = d_max_v - d_min_v
    cumsum = np.cumsum(bin_mass)
    for lo in range(d_bins):
        for hi in range(lo, d_bins):
            mass = cumsum[hi] - (cumsum[lo - 1] if lo > 0 else 0.0)
            if mass >= target:
                width = bins[hi + 1] - bins[lo]
                if width < best_width:
                    best_width = width
                break

    return float(best_width)


def d_range_active(
    grads: list[dict[str, np.ndarray]], threshold: float = 0.01
) -> tuple[float, float]:
    """Return (d_min, d_max) of delay cells with |delta_d| > threshold * peak."""
    all_d: list[float] = []
    all_delta: list[float] = []
    for g in grads:
        all_d.extend(g["d_pos"].ravel().tolist())
        all_d.extend(g["d_neg"].ravel().tolist())
        all_delta.extend(np.abs(g["delta_d_pos"]).ravel().tolist())
        all_delta.extend(np.abs(g["delta_d_neg"]).ravel().tolist())

    d_arr = np.array(all_d)
    delta_arr = np.array(all_delta)
    peak = float(delta_arr.max())
    if peak == 0.0:
        return 0.0, 0.0

    mask = delta_arr > threshold * peak
    if not mask.any():
        return 0.0, 0.0

    active_d = d_arr[mask]
    return float(active_d.min()), float(active_d.max())


def run_xor_with_snapshots(seed: int = 0, total_epochs: int = 2000) -> None:
    """Train XOR and print gradient analysis at snapshot epochs."""
    print("=" * 70)
    print("Delay-Space Gradient Analysis — XOR (seed=0, 2000 epochs)")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
    trainer = MemristorTrainer(net, eta=0.06)

    snap_set = set(SNAPSHOT_EPOCHS)
    epoch = 0

    for target_epoch in sorted(snap_set):
        epochs_to_run = target_epoch - epoch
        if epochs_to_run > 0:
            trainer.fit(XOR_X, XOR_Y, n_epochs=epochs_to_run, verbose=False)
        epoch = target_epoch

        # Take a snapshot: do one gradient pass on the full XOR dataset
        snap_net = copy.deepcopy(net)
        summary = gradient_summary(snap_net, XOR_X, XOR_Y)
        grads = extract_delay_gradients(snap_net)
        acc = trainer.accuracy(XOR_X, XOR_Y)
        half_w = gradient_half_width(grads)
        d_lo, d_hi = d_range_active(grads)
        active_frac = gradient_active_fraction(grads)

        print(f"\n--- Epoch {epoch:5d}  |  loss={summary['loss']:.4f}  |  acc={acc:.0%}  ---")
        print(f"  Global active fraction : {active_frac:.1%}")
        print(f"  Active d range         : [{d_lo:.1f}, {d_hi:.1f}] ns")
        print(f"  Gradient half-width    : {half_w:.2f} ns (captures 80% |delta_d|)")
        print()
        for i, layer_info in enumerate(summary["layers"]):
            print(
                f"  Layer {i}: mean|∂L/∂d|={layer_info['mean_abs_grad_d']:.4f}  "
                f"std={layer_info['std_abs_grad_d']:.4f}  "
                f"active={layer_info['active_fraction']:.1%}"
            )

    print()


def run_convergence_comparison(n_seeds: int = 20, n_epochs: int = 2000) -> None:
    """Compare standard vs. complementary XOR convergence rates."""
    print("=" * 70)
    print("Convergence Comparison: Standard vs. Complementary Encoding")
    print(f"({n_seeds} seeds, {n_epochs} epochs each)")
    print("=" * 70)

    std_converged = 0
    comp_converged = 0

    for seed in range(n_seeds):
        # Standard
        torch.manual_seed(seed)
        np.random.seed(seed)
        std_net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=False)
        std_trainer = MemristorTrainer(std_net, eta=0.06)
        std_trainer.fit(XOR_X, XOR_Y, n_epochs=n_epochs)
        if std_trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            std_converged += 1

        # Complementary
        torch.manual_seed(seed)
        np.random.seed(seed)
        comp_net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)
        comp_trainer = MemristorTrainer(comp_net, eta=0.06)
        comp_trainer.fit(XOR_X, XOR_Y, n_epochs=n_epochs)
        if comp_trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            comp_converged += 1

    std_rate = std_converged / n_seeds
    comp_rate = comp_converged / n_seeds
    print(f"\n  Standard    : {std_converged}/{n_seeds} = {std_rate:.0%} converged")
    print(f"  Complementary: {comp_converged}/{n_seeds} = {comp_rate:.0%} converged")
    print()
    if comp_rate >= std_rate:
        print("  >> Complementary encoding matches or exceeds standard convergence rate.")
    else:
        print(
            f"  >> Complementary encoding converged {std_rate - comp_rate:.0%} less than standard."
        )
    print()


if __name__ == "__main__":
    run_xor_with_snapshots(seed=0, total_epochs=2000)
    run_convergence_comparison(n_seeds=20, n_epochs=2000)
