"""XOR training demo for the memristor-calibrated log-delay network.

Spec Milestone 1 (Section 22):  behavioral XOR model.

Network:
  3 inputs (x0, x1, bias) → 2 hidden neurons → 1 output
  18 individual delay cells (9 signed differential branch pairs × 2 signs)

Physical interpretation:
  Each delay cell is a memristor-controlled RC delay: d = kappa / exp(u).
  Training adjusts u (log-conductance) so that the race between positive
  and negative delay branches implements the XOR decision boundary.

Update rule (spec Section 4):
  d_target = d * exp(-eta * lambda * d)
  This is equivalent to SGD on u (see memristor/training.py for the proof).

Usage:
    uv run python scripts/train_xor.py
    uv run python scripts/train_xor.py --n-seeds 20 --epochs 2000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from memristor.network import MemristorNet
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
XOR_LABELS = ["00→0", "01→1", "10→1", "11→0"]


def train_one_seed(
    seed: int,
    n_epochs: int,
    eta: float,
    verbose: bool = False,
) -> tuple[float, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
    trainer = MemristorTrainer(net, eta=eta)
    losses = trainer.fit(XOR_X, XOR_Y, n_epochs=n_epochs, verbose=verbose, print_every=200)
    acc = trainer.accuracy(XOR_X, XOR_Y)
    return acc, losses


def print_convergence_report(net: MemristorNet) -> None:
    """Show final predictions and branch target delays."""
    print()
    print("Final predictions:")
    for (x0, x1), label in zip(XOR_X, XOR_LABELS, strict=True):
        p = float(net.predict(np.array([x0, x1]))[0])
        correct = "✓" if (round(p) == (1 if "1" in label[-1] else 0)) else "✗"
        print(f"  XOR({x0:.0f},{x1:.0f}) = {label[-1]}  →  p={p:.4f}  {correct}")

    print()
    print("Delay summary after training (ns):")
    for info in net.delay_summary():
        print(
            f"  Layer {info['layer']}  shape={info['shape']}"
            f"  d_pos={info['d_pos_mean']:.1f}ns  d_neg={info['d_neg_mean']:.1f}ns"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="XOR training on memristive delay network")
    parser.add_argument("--n-seeds", type=int, default=10, help="Seeds to test")
    parser.add_argument("--epochs", type=int, default=2000, help="Epochs per seed")
    parser.add_argument("--eta", type=float, default=0.06, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch loss")
    args = parser.parse_args()

    print("=" * 60)
    print("MEMRISTOR LOG-DELAY NETWORK — XOR TRAINING (Milestone 1)")
    print("=" * 60)
    print("Architecture: 3→2→1  (18 delay cells, 9 signed branch pairs)")
    print(f"Delay range:  5–50 ns,  kappa={15.81:.2f} ns (geometric midpoint)")
    print("Training:     tau_nLSE=10 ns,  tau_decision=5 ns")
    print(f"              eta={args.eta},  {args.epochs} epochs per seed")
    print()

    successes = 0
    converged_trainer = None

    for seed in range(args.n_seeds):
        acc, losses = train_one_seed(seed, args.epochs, args.eta, verbose=args.verbose)
        status = "CONVERGED" if acc == 1.0 else f"acc={acc:.0%}"
        print(f"  seed {seed:3d}  final_loss={losses[-1]:.4f}  {status}")
        if acc == 1.0:
            successes += 1
            if converged_trainer is None:
                # Keep the first converged net for display
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
                trainer = MemristorTrainer(net, eta=args.eta)
                trainer.fit(XOR_X, XOR_Y, n_epochs=args.epochs)
                converged_trainer = trainer

    rate = successes / args.n_seeds
    print()
    print(f"Convergence rate: {successes}/{args.n_seeds} = {rate:.0%}")
    print(f"Spec target: ≥90%  {'PASS ✓' if rate >= 0.90 else 'FAIL (< 90%)'}")

    if converged_trainer is not None:
        print_convergence_report(converged_trainer.net)

    print()
    print("Hardware program-and-verify targets (sample branch):")
    if converged_trainer is not None:
        targets = converged_trainer.compute_branch_targets()
        for t in targets:
            d_pos = t["d_pos_target"]
            d_neg = t["d_neg_target"]
            print(
                f"  Layer {t['layer']}  d_pos range: [{d_pos.min():.1f}, {d_pos.max():.1f}] ns"
                f"  d_neg range: [{d_neg.min():.1f}, {d_neg.max():.1f}] ns"
            )

    sys.exit(0 if rate >= 0.80 else 1)


if __name__ == "__main__":
    main()
