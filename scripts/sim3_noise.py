"""Simulation 3: Jitter tolerance sweep.

Trains XOR networks with noiseless delays, then evaluates accuracy at
increasing levels of Gaussian arrival-time jitter.

The key question: what is the maximum sigma_j (ns) before XOR accuracy
degrades significantly?  Spec target: robust for sigma_j <= 1.0 ns.

Usage:
    uv run python scripts/sim3_noise.py
    uv run python scripts/sim3_noise.py --n-seeds 20 --epochs 2000
"""

from __future__ import annotations

import argparse
import sys

from memristor.noise import NoiseSweepResult, noise_accuracy_sweep, print_noise_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim 3: jitter tolerance sweep")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--eta", type=float, default=0.06)
    parser.add_argument("--n-eval", type=int, default=100, help="Majority-vote trials per sample")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 58)
    print("MEMRISTOR LOG-DELAY NETWORK — SIMULATION 3: JITTER")
    print("=" * 58)
    print(f"Seeds: {args.n_seeds}  |  Epochs: {args.epochs}  |  eta: {args.eta}")
    print(f"Eval: majority vote over {args.n_eval} trials per XOR pattern")
    print()

    result: NoiseSweepResult = noise_accuracy_sweep(
        n_trials=args.n_seeds,
        n_epochs=args.epochs,
        eta=args.eta,
        sigma_grid=(0.0, 0.1, 0.5, 1.0, 2.0, 5.0),
        n_eval_trials=args.n_eval,
        verbose=args.verbose,
    )

    print_noise_table(result)

    # Spec target: sigma_j <= 1.0 ns should preserve ≥80% of converged seeds
    acc_1ns = result.accuracy_grid.get(1.0, 0.0)
    baseline = result.baseline_rate
    print()
    print("Key result (sigma_j = 1.0 ns):")
    if baseline > 0:
        retention = acc_1ns / baseline if baseline > 0 else 0.0
        threshold = 0.80
        status = "PASS" if retention >= threshold else "FAIL"
        print(
            f"  {acc_1ns:.0%} of converged seeds remain accurate"
            f"  (retention={retention:.0%}, threshold={threshold:.0%})  {status}"
        )
    else:
        print("  No seeds converged; cannot evaluate.")

    sys.exit(0 if baseline > 0 else 1)


if __name__ == "__main__":
    main()
