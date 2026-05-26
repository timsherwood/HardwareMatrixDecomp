"""Knowm SDC memristor device model for Tier-2 PCB prototype.

Models the Knowm Self-Directed Channel (SDC) memristor:
  - Bounded resistance range [R_min, R_max]
  - Stochastic switching: ΔlnR per pulse ~ Normal(±step_mean_ln, σ)
  - Closed-loop P&V programming to a target delay d = R × C_cell

Reference: Nugent & Molter (2014) "AHaH Computing–From Metastable Switches
to Attractors to Machine Learning". PLOS ONE.
Knowm SDC operating range: 50 kΩ – 500 kΩ (HRS/intermediate state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnowmSDC:
    """Stochastic Knowm SDC memristor model.

    Parameters
    ----------
    R:
        Current resistance (Ω).  Default: geometric mean of R_min/R_max.
    R_min:
        Minimum (highest-conductance) state used: 50 kΩ → d_min = 50 µs.
    R_max:
        Maximum (lowest-conductance) state used: 500 kΩ → d_max = 500 µs.
    step_mean_ln:
        Mean |ΔlnR| per pulse in log-resistance space.  ~10% per pulse.
    noise_frac:
        Cycle-to-cycle variability: σ = noise_frac × step_mean_ln.
        Typical Knowm SDC value: 0.10–0.20.
    """

    R: float = 158_114.0   # Ω  (= sqrt(50k × 500k), geometric midpoint)
    R_min: float = 50_000.0
    R_max: float = 500_000.0
    step_mean_ln: float = 0.10
    noise_frac: float = 0.15

    # ------------------------------------------------------------------ pulses

    def set_pulse(self, rng: np.random.Generator | None = None) -> None:
        """Apply one SET pulse (increases conductance → decreases R)."""
        _rng = rng or np.random.default_rng()
        step = self.step_mean_ln * (1.0 + _rng.normal(0.0, self.noise_frac))
        lnR = np.clip(np.log(self.R) - step, np.log(self.R_min), np.log(self.R_max))
        self.R = float(np.exp(lnR))

    def reset_pulse(self, rng: np.random.Generator | None = None) -> None:
        """Apply one RESET pulse (decreases conductance → increases R)."""
        _rng = rng or np.random.default_rng()
        step = self.step_mean_ln * (1.0 + _rng.normal(0.0, self.noise_frac))
        lnR = np.clip(np.log(self.R) + step, np.log(self.R_min), np.log(self.R_max))
        self.R = float(np.exp(lnR))

    # ------------------------------------------------------------------ delay

    def delay_us(self, C_cell_F: float = 1e-9) -> float:
        """Current delay in µs: d = R × C_cell × 1e6."""
        return self.R * C_cell_F * 1e6

    # ----------------------------------------------------------------- P&V

    def program_to_delay(
        self,
        d_target_us: float,
        C_cell_F: float = 1e-9,
        tol_us: float = 5.0,
        max_pulses: int = 100,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Closed-loop P&V: drive device to d_target_us within tol_us.

        Each iteration applies one SET or RESET pulse, measures the resulting
        delay, and flips polarity if needed.  Stops on convergence or exhaustion.

        Returns
        -------
        dict with keys:
            success     – bool: |d_final - d_target| ≤ tol_us
            d_final_us  – float: delay after last pulse
            n_pulses    – int: pulses applied
            polarity    – str: "SET", "RESET", or "NONE"
        """
        _rng = rng or np.random.default_rng()
        d_meas = self.delay_us(C_cell_F)
        polarity = "NONE"

        if abs(d_meas - d_target_us) <= tol_us:
            return {"success": True, "d_final_us": d_meas, "n_pulses": 0, "polarity": "NONE"}

        polarity = "SET" if d_meas > d_target_us else "RESET"

        for n in range(1, max_pulses + 1):
            if polarity == "SET":
                self.set_pulse(rng=_rng)
            else:
                self.reset_pulse(rng=_rng)
            d_meas = self.delay_us(C_cell_F)
            if abs(d_meas - d_target_us) <= tol_us:
                return {"success": True, "d_final_us": d_meas, "n_pulses": n, "polarity": polarity}
            polarity = "SET" if d_meas > d_target_us else "RESET"

        return {
            "success": False,
            "d_final_us": d_meas,
            "n_pulses": max_pulses,
            "polarity": polarity,
        }
