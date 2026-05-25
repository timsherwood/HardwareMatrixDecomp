"""Single memristive delay cell and closed-loop program-and-verify.

Physical model:
    d = kappa * exp(-u),    u = ln(G)

Trainable variable is u (log-conductance).  Increasing u increases G and
reduces d (SET pulse); decreasing u reduces G and increases d (RESET pulse).

Training rule (spec Section 4):
    d_target = d * exp(-eta * lambda * d)
    u_new    = ln(kappa / d_target)  =  u + eta * lambda * d

This equals standard gradient descent on u since:
    grad_u = dL/du = lambda * (dd/du) = lambda * (-d)
    → u_new = u - eta * grad_u = u + eta * lambda * d  ✓

Hardware program-and-verify (spec Section 17):
    if d_measured > d_target: SET pulses (increase G, decrease d)
    if d_measured < d_target: RESET pulses (decrease G, increase d)
    Repeat until |d_measured - d_target| < tolerance or max pulses hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DelayCell:
    """Single memristive delay branch: d = kappa * exp(-u).

    Parameters
    ----------
    u:
        Log-conductance (trainable).  u = 0 → d = kappa (midpoint delay).
    kappa:
        Calibration constant in ns.  Set to geometric-mean of d_min*d_max.
    d_min, d_max:
        Valid delay range in ns.
    """

    u: float = 0.0
    kappa: float = 15.81  # sqrt(5*50) so u=0 is the geometric midpoint
    d_min: float = 5.0
    d_max: float = 50.0

    @property
    def conductance(self) -> float:
        return float(np.exp(self.u))

    @property
    def delay(self) -> float:
        """Current ideal delay in ns (clamped to [d_min, d_max])."""
        return float(np.clip(self.kappa * np.exp(-self.u), self.d_min, self.d_max))

    def _u_bounds(self) -> tuple[float, float]:
        return float(np.log(self.kappa / self.d_max)), float(np.log(self.kappa / self.d_min))

    def _clamp_u(self) -> None:
        lo, hi = self._u_bounds()
        self.u = float(np.clip(self.u, lo, hi))

    def apply_local_update(self, lam: float, eta: float) -> None:
        """Apply spec's update rule.  Equivalent to u -= eta * grad_u."""
        d = self.delay
        d_target = float(np.clip(d * np.exp(-eta * lam * d), self.d_min, self.d_max))
        self.u = float(np.log(self.kappa / d_target))

    def set_target_delay(self, d_target: float) -> None:
        """Directly program to a target delay (ideal, no noise)."""
        d_clamped = float(np.clip(d_target, self.d_min, self.d_max))
        self.u = float(np.log(self.kappa / d_clamped))

    def measure(
        self,
        jitter_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Read delay with optional Gaussian timing jitter."""
        d = self.delay
        if jitter_std > 0.0:
            _rng = rng or np.random.default_rng()
            d = float(d + _rng.normal(0.0, jitter_std))
        return float(np.clip(d, self.d_min, self.d_max))

    def program_and_verify(
        self,
        d_target: float,
        tol: float = 0.5,
        max_pulses: int = 200,
        pulse_step: float = 0.01,
        jitter_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Simulated closed-loop SET/RESET loop (spec Section 17).

        Each iteration applies one incremental pulse (±pulse_step in u-space)
        then re-measures.  Stops when within tolerance or max_pulses reached.

        Returns a log dict with d_final, n_pulses, polarity, success.
        """
        _rng = rng or np.random.default_rng()
        lo, hi = self._u_bounds()
        d_target = float(np.clip(d_target, self.d_min, self.d_max))

        d_meas = self.measure(jitter_std, _rng)
        n_pulses = 0
        polarity = "none"

        while abs(d_meas - d_target) > tol and n_pulses < max_pulses:
            if d_meas > d_target:
                self.u = float(np.clip(self.u + pulse_step, lo, hi))
                polarity = "SET"
            else:
                self.u = float(np.clip(self.u - pulse_step, lo, hi))
                polarity = "RESET"
            d_meas = self.measure(jitter_std, _rng)
            n_pulses += 1

        return {
            "d_final": d_meas,
            "n_pulses": n_pulses,
            "polarity": polarity,
            "success": bool(abs(d_meas - d_target) <= tol),
            "saturated": n_pulses >= max_pulses,
        }
