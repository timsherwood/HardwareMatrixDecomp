"""Simulation 2: Post-training delay quantization sweep.

Trains XOR networks with continuous delays, then evaluates how much
accuracy degrades when delays are snapped to finite hardware levels.

Two quantization axes:
  n_levels  — conductance states (16 / 32 / 64 / 128)
  tdc_res   — TDC measurement resolution in ns (0.5 / 1.0 / 2.0)

Usage:
    uv run python scripts/sim2_quantization.py
    uv run python scripts/sim2_quantization.py --n-seeds 20 --epochs 2000
"""

from __future__ import annotations

import argparse
import sys

from memristor.quantization import (
    QuantizationSweepResult,
    print_sweep_table,
    quantization_accuracy_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim 2: post-training quantization sweep")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--eta", type=float, default=0.06)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("MEMRISTOR LOG-DELAY NETWORK — SIMULATION 2: QUANTIZATION")
    print("=" * 62)
    print(f"Seeds: {args.n_seeds}  |  Epochs: {args.epochs}  |  eta: {args.eta}")
    print("Quantization: post-training (no retraining after snapping)")
    print()

    result: QuantizationSweepResult = quantization_accuracy_sweep(
        n_trials=args.n_seeds,
        n_epochs=args.epochs,
        eta=args.eta,
        levels_grid=(16, 32, 64, 128),
        tdc_grid=(0.5, 1.0, 2.0),
        verbose=args.verbose,
    )

    print_sweep_table(result)

    # Determine pass/fail: at 64+ levels with 1ns TDC, should preserve ≥80% of baseline
    baseline = result.baseline_rate
    key_64_1 = (64, 1.0)
    acc_64_1 = result.accuracy_grid.get(key_64_1, 0.0)

    print()
    print("Key result (n=64, tdc=1.0ns):")
    if baseline > 0:
        retention = acc_64_1 / baseline if baseline > 0 else 0.0
        threshold = 0.80
        status = "PASS" if retention >= threshold else "FAIL"
        print(
            f"  {acc_64_1:.0%} of converged seeds remain accurate"
            f"  (retention={retention:.0%}, threshold={threshold:.0%})  {status}"
        )
    else:
        print("  No seeds converged; cannot evaluate.")

    sys.exit(0 if baseline > 0 else 1)


if __name__ == "__main__":
    main()
