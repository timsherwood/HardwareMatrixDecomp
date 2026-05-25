"""Simulation 3: Noisy timing model.

Adds Gaussian jitter to all arrival times in the forward pass, modelling:

  - Clock-to-Q and setup/hold variation in digital delay elements
  - Substrate coupling and power-supply noise in analog delay lines
  - TDC aperture jitter when reading memristor state during inference

Jitter is injected at every layer input, so errors accumulate through depth.
Training is done without noise; inference is evaluated at increasing sigma_j.

Key finding from the spec (Section 6):
  sigma_j < tau / 3  → accuracy is robust (jitter is smaller than nLSE
                       integration window)
  sigma_j ~ tau      → noticeable degradation
  sigma_j > tau      → may cause race inversions

With tau=10 ns, the operating envelope is sigma_j << 10 ns.  The spec
target is sigma_j <= 1.0 ns (Simulation 3 sweep).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from memristor.network import MemristorNet
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


class NoisyMemristorNet(nn.Module):
    """Wraps MemristorNet and injects Gaussian jitter on arrival times.

    Jitter is added to every arrival-time vector before each layer's
    delay computation, modelling temporal uncertainty that accumulates
    through the network.  At sigma_j=0 this is identical to MemristorNet.
    """

    def __init__(self, net: MemristorNet, sigma_j: float = 0.0) -> None:
        super().__init__()
        self.net = net
        self.sigma_j = sigma_j

    def _jitter(self, T: torch.Tensor, rng: np.random.Generator | None) -> torch.Tensor:
        if self.sigma_j <= 0.0:
            return T
        noise_np = (rng or np.random.default_rng()).normal(0.0, self.sigma_j, size=T.shape)
        return T + torch.tensor(noise_np, dtype=T.dtype)

    def forward(
        self, T_in: torch.Tensor, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        """Forward pass with per-layer arrival-time jitter."""
        T_current = self._jitter(T_in, rng)
        n_layers = len(self.net.layers)

        for idx, layer in enumerate(self.net.layers):
            is_last = idx == n_layers - 1
            d_pos, d_neg = layer.delays()

            A_pos = T_current.unsqueeze(1) + d_pos
            A_neg = T_current.unsqueeze(1) + d_neg

            tau = self.net.tau
            T_plus = -tau * torch.logsumexp(-A_pos / tau, dim=0)
            T_minus = -tau * torch.logsumexp(-A_neg / tau, dim=0)
            margin = T_minus - T_plus

            if is_last:
                return torch.sigmoid(margin / self.net.tau_d)

            p = torch.sigmoid(margin / self.net.tau_d)
            T_h = p * T_plus + (1.0 - p) * T_minus
            T_current = self._jitter(
                torch.cat([T_h, torch.zeros(1)]), rng
            )

        raise RuntimeError("No layers defined")  # pragma: no cover

    def predict_noisy(
        self,
        x: np.ndarray,
        n_trials: int = 100,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Return (mean probability, std) over n_trials jitter realisations."""
        rng = np.random.default_rng(seed)
        T_in = self.net.encode_binary(x)
        probs = []
        with torch.no_grad():
            for _ in range(n_trials):
                p = float(self.forward(T_in, rng=rng)[0])
                probs.append(p)
        return float(np.mean(probs)), float(np.std(probs))

    def accuracy_noisy(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_trials: int = 50,
        seed: int = 0,
    ) -> float:
        """Majority-vote accuracy over n_trials jitter realisations per sample."""
        rng = np.random.default_rng(seed)
        correct = 0
        T_ins = [self.net.encode_binary(x) for x in X]
        with torch.no_grad():
            for T_in, y in zip(T_ins, Y, strict=True):
                votes = [
                    int(float(self.forward(T_in, rng=rng)[0]) > 0.5)
                    for _ in range(n_trials)
                ]
                pred = int(sum(votes) > n_trials / 2)
                if pred == int(y):
                    correct += 1
        return correct / len(Y)


@dataclass
class NoiseSweepResult:
    """Results from the jitter tolerance sweep."""

    n_trials: int
    n_converged: int
    baseline_rate: float
    sigma_grid: tuple[float, ...]
    # accuracy_grid[sigma_j] = fraction of converged seeds still 100% correct (majority vote)
    accuracy_grid: dict[float, float] = field(default_factory=dict)
    # mean_xor_grid[sigma_j] = mean fraction of XOR patterns correct
    mean_xor_grid: dict[float, float] = field(default_factory=dict)


def noise_accuracy_sweep(
    n_trials: int = 20,
    n_epochs: int = 1500,
    eta: float = 0.06,
    sigma_grid: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0),
    n_eval_trials: int = 100,
    seed_offset: int = 0,
    verbose: bool = False,
) -> NoiseSweepResult:
    """Train n_trials seeds; evaluate XOR accuracy at each jitter level.

    Parameters
    ----------
    sigma_grid:
        Jitter standard deviations to sweep (ns).
    n_eval_trials:
        Stochastic forward-pass repetitions per XOR pattern for majority-vote
        accuracy estimation.
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
            print(f"  seed {seed_offset+i:3d}  {'CONVERGED' if converged else f'acc={acc:.0%}'}")
        if converged:
            converged_nets.append(net)

    n_converged = len(converged_nets)

    result = NoiseSweepResult(
        n_trials=n_trials,
        n_converged=n_converged,
        baseline_rate=n_converged / n_trials,
        sigma_grid=sigma_grid,
    )

    if n_converged == 0:
        return result

    for sigma_j in sigma_grid:
        full_acc_count = 0
        xor_fracs = []
        eval_rng_seed = 99  # fixed eval seed for reproducibility

        for net in converged_nets:
            noisy = NoisyMemristorNet(net, sigma_j=sigma_j)
            if sigma_j == 0.0:
                # Deterministic: single evaluation is sufficient
                preds = [int(float(net.predict(x)[0]) > 0.5) for x in XOR_X]
            else:
                # Majority vote over n_eval_trials stochastic runs
                preds = []
                for x in XOR_X:
                    T_x = net.encode_binary(x)
                    votes = 0
                    for j in range(n_eval_trials):
                        rng_j = np.random.default_rng(eval_rng_seed + j)
                        p_j = float(noisy.forward(T_x, rng=rng_j)[0])
                        votes += int(p_j > 0.5)
                    preds.append(int(votes > n_eval_trials / 2))
            correct = sum(p == int(y) for p, y in zip(preds, XOR_Y, strict=True))
            xor_fracs.append(correct / len(XOR_Y))
            if correct == len(XOR_Y):
                full_acc_count += 1

        result.accuracy_grid[sigma_j] = full_acc_count / n_converged
        result.mean_xor_grid[sigma_j] = float(np.mean(xor_fracs))

    return result


def print_noise_table(result: NoiseSweepResult) -> None:
    """Print a formatted table of jitter-tolerance results."""
    print()
    print("Jitter Tolerance Sweep — XOR Accuracy vs sigma_j")
    print("=" * 58)
    print(
        f"Baseline: {result.n_converged}/{result.n_trials} seeds converged"
        f" ({result.baseline_rate:.0%})"
    )
    print("tau (nLSE) = 10 ns  tau_d (decision) = 5 ns")
    print()
    print(f"{'sigma_j (ns)':>14}  {'100% acc':>10}  {'mean acc':>10}  {'note':>20}")
    print("-" * 58)

    for sigma_j in result.sigma_grid:
        full = result.accuracy_grid.get(sigma_j, float("nan"))
        mean = result.mean_xor_grid.get(sigma_j, float("nan"))
        note = ""
        if sigma_j == 0.0:
            note = "baseline"
        elif sigma_j <= 1.0:
            note = "< tau/10" if sigma_j <= 1.0 else ""
        ratio = sigma_j / 10.0  # relative to tau=10
        if sigma_j > 0:
            note = f"sigma/tau={ratio:.1f}"
        print(f"{sigma_j:>14.1f}  {full:>10.0%}  {mean:>10.2f}  {note:>20}")
