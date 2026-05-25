"""Simulation 4: Stochastic memristor programming sweep.

Evaluates closed-loop program-and-verify (P&V) yield under realistic
memristor non-idealities:

  Asymmetry   — SET step ≠ RESET step (ratio up to 3×)
  Noise       — cycle-to-cycle pulse size variability (up to 20%)
  Read noise  — TDC measurement uncertainty (up to 2 ns)

Usage:
    uv run python scripts/sim4_programming.py
    uv run python scripts/sim4_programming.py --n-targets 1000 --tol 0.5
"""

from __future__ import annotations

import argparse
import sys

from memristor.programming import ProgrammingSweepResult, print_programming_table, programming_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim 4: stochastic P&V sweep")
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--tol", type=float, default=0.5, help="Success tolerance (ns)")
    parser.add_argument("--max-pulses", type=int, default=200)
    parser.add_argument("--base-step", type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 68)
    print("MEMRISTOR LOG-DELAY NETWORK — SIMULATION 4: STOCHASTIC P&V")
    print("=" * 68)
    print(f"n_targets={args.n_targets}  tol={args.tol} ns  max_pulses={args.max_pulses}")
    print(f"base_step={args.base_step} (log-conductance units per pulse)")
    print()

    result: ProgrammingSweepResult = programming_sweep(
        n_targets=args.n_targets,
        tol=args.tol,
        max_pulses=args.max_pulses,
        base_step=args.base_step,
        asymmetry_grid=(1.0, 1.5, 2.0, 3.0),
        noise_frac_grid=(0.0, 0.05, 0.10, 0.20),
        read_noise_grid=(0.0, 0.5, 1.0, 2.0),
    )

    print_programming_table(result)

    # Key spec target: ideal P&V (asymmetry=1, no noise) should achieve ≥99% yield
    ideal_yield = result.success_rate.get((1.0, 0.0, 0.0), 0.0)
    print()
    print("Key result (ideal P&V, asymmetry=1, no noise):")
    status = "PASS" if ideal_yield >= 0.99 else "FAIL (< 99%)"
    print(f"  yield={ideal_yield:.1%}  {status}")

    sys.exit(0 if ideal_yield >= 0.99 else 1)


if __name__ == "__main__":
    main()
