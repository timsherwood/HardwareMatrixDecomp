"""Simulation 4: Stochastic memristor programming model.

Models the non-idealities in closed-loop memristor program-and-verify (P&V):

  SET asymmetry  — SET (HRS→LRS, d↓) and RESET (LRS→HRS, d↑) pulses have
                   different mean step sizes (ratio = reset_step / set_step)
  Cycle noise    — each pulse overshoots/undershoots by a Gaussian fraction
                   of the nominal step size (cycle-to-cycle variability)
  Read noise     — TDC measurement uncertainty adds Gaussian noise to the
                   observed delay (determines when P&V declares success)
  State dependence — pulse effectiveness scales with current conductance:
                   delta_u *= (1 + gamma * abs(u)); real devices show
                   smaller incremental change at extremes of the range
  Drift          — conductance relaxes toward u=0 at rate `drift_rate` per
                   P&V iteration (logarithmic creep observed in oxide devices)

Sweep structure (scripts/sim4_programming.py):
  For a grid of (asymmetry, noise_frac, read_noise) parameters, programs
  N_TARGETS random target delays in [d_min, d_max] and reports:
    - success_rate : fraction hitting tol within max_pulses
    - mean_abs_error: RMS delay error at convergence
    - mean_n_pulses : average pulses needed for successful cells
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StochasticProgrammer:
    """Stochastic memristor programming model for a single delay cell.

    Parameters
    ----------
    set_step:
        Mean log-conductance increment per SET pulse (u increases → d decreases).
    reset_step:
        Mean log-conductance decrement per RESET pulse (u decreases → d increases).
        Asymmetry ratio = reset_step / set_step.
    noise_frac:
        Cycle-to-cycle pulse noise: sigma = noise_frac * step_size.
    read_noise:
        TDC measurement noise std (ns); added to every delay measurement.
    state_dependence:
        State-dependent gain coefficient gamma.  Effective step is multiplied
        by (1 + gamma * |u|).  Zero means constant pulse step.
    drift_rate:
        Per-iteration relaxation: u ← u * (1 - drift_rate).  Models logarithmic
        conductance drift toward the neutral state u=0 (d=kappa).
    kappa, d_min, d_max:
        Physical delay cell parameters.
    """

    set_step: float = 0.01
    reset_step: float = 0.01
    noise_frac: float = 0.0
    read_noise: float = 0.0
    state_dependence: float = 0.0
    drift_rate: float = 0.0
    kappa: float = 15.81
    d_min: float = 5.0
    d_max: float = 50.0

    def _delay(self, u: float) -> float:
        return float(np.clip(self.kappa * np.exp(-u), self.d_min, self.d_max))

    def _measure(self, u: float, rng: np.random.Generator) -> float:
        d_true = self._delay(u)
        noise = rng.normal(0.0, self.read_noise) if self.read_noise > 0.0 else 0.0
        return float(np.clip(d_true + noise, self.d_min, self.d_max))

    def _apply_pulse(
        self, u: float, polarity: str, rng: np.random.Generator
    ) -> float:
        """Apply one SET or RESET pulse and return updated u."""
        base_step = self.set_step if polarity == "SET" else self.reset_step
        # State-dependent gain: larger effect when conductance is extreme
        gain = 1.0 + self.state_dependence * abs(u)
        effective_step = base_step * gain
        # Cycle-to-cycle noise
        noise = rng.normal(0.0, self.noise_frac * effective_step) if self.noise_frac > 0 else 0.0
        delta = (effective_step + noise)
        u_new = u + delta if polarity == "SET" else u - delta
        # Drift after each pulse
        if self.drift_rate > 0.0:
            u_new = u_new * (1.0 - self.drift_rate)
        return u_new

    def program(
        self,
        u_init: float,
        d_target: float,
        tol: float = 0.5,
        max_pulses: int = 200,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Run P&V loop to hit d_target from u_init.

        Returns
        -------
        dict with keys:
          u_final, d_final, n_pulses, polarity, success, saturated
        """
        rng = rng or np.random.default_rng()
        d_target_clamped = float(np.clip(d_target, self.d_min, self.d_max))
        u = u_init
        n_pulses = 0
        polarity = "NONE"

        d_meas = self._measure(u, rng)
        if abs(d_meas - d_target_clamped) <= tol:
            return {
                "u_final": u,
                "d_final": self._delay(u),
                "n_pulses": 0,
                "polarity": "NONE",
                "success": True,
                "saturated": False,
            }

        polarity = "SET" if d_meas > d_target_clamped else "RESET"

        for _ in range(max_pulses):
            u = self._apply_pulse(u, polarity, rng)
            n_pulses += 1
            d_meas = self._measure(u, rng)
            if abs(d_meas - d_target_clamped) <= tol:
                return {
                    "u_final": u,
                    "d_final": self._delay(u),
                    "n_pulses": n_pulses,
                    "polarity": polarity,
                    "success": True,
                    "saturated": False,
                }
            # Check for saturation (overshot to limit)
            d_true = self._delay(u)
            if d_true <= self.d_min + 1e-3 or d_true >= self.d_max - 1e-3:
                return {
                    "u_final": u,
                    "d_final": d_true,
                    "n_pulses": n_pulses,
                    "polarity": polarity,
                    "success": False,
                    "saturated": True,
                }
            # Flip polarity if overshot
            new_polarity = "SET" if d_meas > d_target_clamped else "RESET"
            polarity = new_polarity

        return {
            "u_final": u,
            "d_final": self._delay(u),
            "n_pulses": n_pulses,
            "polarity": polarity,
            "success": False,
            "saturated": False,
        }


@dataclass
class ProgrammingSweepResult:
    """Results from a stochastic P&V parameter sweep."""

    n_targets: int
    tol: float
    # grid key: (asymmetry_ratio, noise_frac, read_noise_ns)
    success_rate: dict[tuple[float, float, float], float] = field(default_factory=dict)
    mean_abs_error: dict[tuple[float, float, float], float] = field(default_factory=dict)
    mean_n_pulses: dict[tuple[float, float, float], float] = field(default_factory=dict)


def programming_sweep(
    n_targets: int = 500,
    tol: float = 0.5,
    max_pulses: int = 200,
    asymmetry_grid: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    noise_frac_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    read_noise_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    base_step: float = 0.01,
    state_dependence: float = 0.0,
    drift_rate: float = 0.0,
    seed: int = 0,
) -> ProgrammingSweepResult:
    """Sweep stochastic P&V parameters and measure programming yield.

    For each parameter combination, programs n_targets random cells
    starting from u=0 (d=kappa, the geometric midpoint) to uniformly
    distributed target delays in [d_min+1, d_max-1].

    One sweep axis is varied at a time while others are held at their
    ideal (zero) value:
      - asymmetry: vary asymmetry_ratio; noise_frac=0, read_noise=0
      - noise: vary noise_frac; asymmetry=1, read_noise=0
      - read noise: vary read_noise; asymmetry=1, noise_frac=0
    """
    rng = np.random.default_rng(seed)
    # Fixed target delays (same across all configs for fair comparison)
    d_targets = rng.uniform(6.0, 49.0, size=n_targets)
    u_init = 0.0  # start from kappa (midpoint)

    result = ProgrammingSweepResult(n_targets=n_targets, tol=tol)

    def _run_config(
        asym: float, nf: float, rn: float
    ) -> tuple[float, float, float]:
        key_rng = np.random.default_rng(seed + 1)
        successes, abs_errors, pulse_counts = [], [], []
        for d_t in d_targets:
            prog = StochasticProgrammer(
                set_step=base_step,
                reset_step=base_step * asym,
                noise_frac=nf,
                read_noise=rn,
                state_dependence=state_dependence,
                drift_rate=drift_rate,
            )
            res = prog.program(u_init, d_t, tol=tol, max_pulses=max_pulses, rng=key_rng)
            successes.append(int(res["success"]))
            abs_errors.append(abs(float(res["d_final"]) - d_t))
            if res["success"]:
                pulse_counts.append(int(res["n_pulses"]))
        return (
            float(np.mean(successes)),
            float(np.mean(abs_errors)),
            float(np.mean(pulse_counts)) if pulse_counts else float("nan"),
        )

    # Asymmetry sweep (noise_frac=0, read_noise=0)
    for asym in asymmetry_grid:
        key = (asym, 0.0, 0.0)
        sr, mae, mp = _run_config(asym, 0.0, 0.0)
        result.success_rate[key] = sr
        result.mean_abs_error[key] = mae
        result.mean_n_pulses[key] = mp

    # Noise fraction sweep (asymmetry=1, read_noise=0)
    for nf in noise_frac_grid:
        key = (1.0, nf, 0.0)
        sr, mae, mp = _run_config(1.0, nf, 0.0)
        result.success_rate[key] = sr
        result.mean_abs_error[key] = mae
        result.mean_n_pulses[key] = mp

    # Read noise sweep (asymmetry=1, noise_frac=0)
    for rn in read_noise_grid:
        key = (1.0, 0.0, rn)
        sr, mae, mp = _run_config(1.0, 0.0, rn)
        result.success_rate[key] = sr
        result.mean_abs_error[key] = mae
        result.mean_n_pulses[key] = mp

    return result


def print_programming_table(result: ProgrammingSweepResult) -> None:
    """Print a compact summary of P&V sweep results."""
    print()
    print("Stochastic P&V Sweep — Programming Yield")
    print("=" * 68)
    print(
        f"n_targets={result.n_targets}  tol={result.tol} ns"
        "  (success: |d_final - d_target| ≤ tol)"
    )
    print()

    def _row(label: str, key: tuple[float, float, float]) -> str:
        sr = result.success_rate.get(key, float("nan"))
        mae = result.mean_abs_error.get(key, float("nan"))
        mp = result.mean_n_pulses.get(key, float("nan"))
        mp_str = f"{mp:.1f}" if not np.isnan(mp) else "  —"
        return f"  {label:<30}  {sr:>8.1%}  {mae:>10.3f} ns  {mp_str:>8} pulses"

    asym_keys = sorted({k for k in result.success_rate if k[1] == 0.0 and k[2] == 0.0})
    noise_keys = sorted({k for k in result.success_rate if k[0] == 1.0 and k[2] == 0.0})
    rn_keys = sorted({k for k in result.success_rate if k[0] == 1.0 and k[1] == 0.0})

    print(f"  {'Config':<30}  {'yield':>8}  {'mean |err|':>12}  {'mean pulses':>12}")
    print("  " + "-" * 64)

    print("  [Asymmetry: reset_step / set_step]")
    for k in asym_keys:
        print(_row(f"asymmetry={k[0]:.1f}", k))

    print("  [Cycle noise: sigma = frac * step]")
    for k in noise_keys:
        if k not in asym_keys:
            print(_row(f"noise_frac={k[1]:.2f}", k))

    print("  [TDC read noise (ns)]")
    for k in rn_keys:
        if k not in asym_keys and k not in noise_keys:
            print(_row(f"read_noise={k[2]:.1f} ns", k))
