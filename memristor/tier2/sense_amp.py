"""BJT exponential-transconductance nLSE sense amplifier simulation.

Physical model
--------------
Each delay branch drives a BJT base: V_BE_i(t) = V_RC_i(t).
In the exponential conduction region, the collector current is:

    I_branch_i(t) = exp((V_BE_i(t) − V_th) / (gain_A × V_T))

where gain_A is a translinear current-mirror gain that scales the effective
integration time constant from τ_branch up to τ_sense:

    τ_branch_i = V_T × d_i / (V_DD − V_th)
    τ_sense_i  = gain_A × τ_branch_i

The sum of branch currents is compared against a threshold.  By normalisation
(I_threshold = 1.0), a single branch fires exactly at its analytic t_cross.
For N simultaneous branches the sense amp fires τ_sense × ln(N) earlier —
the nLSE soft-min behaviour.

The numerical simulation integrates the vectorised current model:
    I_total(t) = Σ_i exp(clip((V_BE_i(t) − V_th) / (gain_A × V_T), −50, 50))
and reports the first t where I_total ≥ 1.0.

Helper
------
nLSE_us(t_cross_list, tau_us): analytical nLSE formula (used in Tier2Network
    forward pass and in margin-cancellation tests).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp  # type: ignore[import-untyped]

# ─── analytical nLSE helper ───────────────────────────────────────────────────

def nLSE_us(t_cross_list: list[float], tau_us: float) -> float:  # noqa: N802
    """Negative log-sum-exp soft-min over threshold crossing times (µs).

    nLSE(tc; τ) = −τ · log Σ_i exp(−tc_i / τ)

    Numerically stable via the log-sum-exp identity:
        log Σ exp(x_i) = max(x_i) + log Σ exp(x_i − max(x_i))
    """
    tcs = np.asarray(t_cross_list, dtype=float)
    log_terms = -tcs / tau_us
    lse = np.max(log_terms) + np.log(np.sum(np.exp(log_terms - np.max(log_terms))))
    return float(-tau_us * lse)


# ─── BJT sense amplifier ──────────────────────────────────────────────────────

class BJTSenseAmp:
    """Numerical simulation of BJT nLSE sense amplifier at PCB scale.

    Parameters
    ----------
    V_DD:
        RC ramp supply voltage (V).  Default 1.5 V.
    V_th:
        Voltage threshold (V).  Default V_DD/2 = 0.75 V.
        Single branch fires when V_BE reaches V_th (by normalisation).
    V_T:
        BJT thermal voltage (V).  26 mV at 25 °C.
    gain_A:
        Translinear current-mirror gain.  Sets τ_sense = gain_A × τ_branch.
        Use ≥ 20 for PCB: gives τ_sense ≈ 110 µs at d_nom with V_DD=1.5 V.
    dt_us:
        Numerical integration time step (µs).  Smaller → more accurate but slower.
    """

    def __init__(
        self,
        V_DD: float = 1.5,
        V_th: float | None = None,
        V_T: float = 0.026,
        gain_A: float = 20.0,
        dt_us: float = 0.5,
    ) -> None:
        self.V_DD = V_DD
        self.V_th = V_th if V_th is not None else V_DD / 2.0
        self.V_T = V_T
        self.gain_A = gain_A
        self.dt_us = dt_us

    # ---------------------------------------------------------------- helpers

    def tau_sense_us(self, d_us: float) -> float:
        """Effective sense-amp time constant (µs) for delay d_us.

        τ_sense = gain_A × V_T × d / (V_DD − V_th)
        """
        return self.gain_A * self.V_T * d_us / (self.V_DD - self.V_th)

    # --------------------------------------------------------- fire time

    def fire_time_us(
        self,
        delays_us: list[float],
        T_in_list: list[float],
        t_max_us: float | None = None,
    ) -> float:
        """Numerically simulate BJT sense-amp firing time (µs).

        Returns the first time t where Σ_i I_branch_i(t) ≥ 1.0.
        If threshold is not crossed within t_max_us, returns t_max_us.

        Parameters
        ----------
        delays_us:
            RC time constants d_i for each branch (µs).
        T_in_list:
            Input arrival times T_in_i for each branch (µs).
        t_max_us:
            Upper bound on simulation time.  Auto-set if None.
        """
        delays = np.asarray(delays_us, dtype=float)    # (N,)
        T_ins = np.asarray(T_in_list, dtype=float)     # (N,)

        # Reference t_cross for the fastest branch (V_th = V_DD/2 → ln(2))
        ln_factor = math.log(self.V_DD / (self.V_DD - self.V_th))
        t_cross_ref = float(np.min(T_ins + delays * ln_factor))
        if t_max_us is None:
            t_max_us = t_cross_ref + 5.0 * float(np.max(delays))

        t_start = float(np.min(T_ins))
        t_arr = np.arange(t_start, t_max_us + self.dt_us, self.dt_us)  # (T,)

        # V_BE_i(t) = V_DD × (1 − exp(−(t − T_in_i) / d_i))  for t > T_in_i
        dt_grid = t_arr[np.newaxis, :] - T_ins[:, np.newaxis]  # (N, T)
        V_BE = np.where(
            dt_grid > 0.0,
            self.V_DD * (1.0 - np.exp(-dt_grid / delays[:, np.newaxis])),
            0.0,
        )  # (N, T)

        # Normalised log-current: log_I = (V_BE − V_th) / (gain_A × V_T).
        # Branches that have not yet arrived (dt ≤ 0) are modelled as completely
        # off (active-pulldown reset holds V_BE << V_th before T_in_i).
        log_I = np.where(
            dt_grid > 0.0,
            np.clip(
                (V_BE - self.V_th) / (self.gain_A * self.V_T),
                -50.0,
                50.0,
            ),
            -50.0,
        )  # (N, T)

        # log(I_total) = log Σ_i exp(log_I_i) — use logsumexp for stability
        log_I_total = logsumexp(log_I, axis=0)  # (T,)

        # Fire when I_total ≥ 1.0, i.e. log_I_total ≥ 0
        fired = log_I_total >= 0.0
        if not np.any(fired):
            return float(t_arr[-1])
        return float(t_arr[int(np.argmax(fired))])
