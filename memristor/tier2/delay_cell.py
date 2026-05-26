"""RC delay cell model at PCB µs scale (Tier-2 prototype).

Physical model:
    V_RC(t) = V_DD × (1 − exp(−(t − T_in) / d))   for t > T_in, else 0
    d       = R_ohm × C_cell_nF × 1e-3  [µs]

Threshold crossing (when V_RC reaches V_th):
    t_cross = T_in + d × ln(V_DD / (V_DD − V_th))
    At V_th = V_DD/2:  t_cross = T_in + d × ln(2)

PCB nominal parameters:
    R_ohm   ∈ [50 kΩ, 500 kΩ]  (Knowm SDC operating range)
    C_cell  = 1 nF              (fixed load capacitor)
    d       ∈ [50, 500] µs
    V_DD    = 1.5 V             (RC ramp supply, matches BJT V_BE range)
    V_th    = 0.75 V = V_DD/2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RCDelayCell:
    """RC delay cell at PCB µs scale.

    Parameters
    ----------
    R_ohm:
        Memristor resistance (Ω).  Set by KnowmSDC.R.
    C_cell_nF:
        Fixed load capacitance (nF).  Use 1.0 nF for PCB prototype.
    V_DD:
        RC ramp supply voltage (V).  Use 1.5 V to match BJT threshold range.
    V_th:
        Voltage threshold for crossing-time calculation (V).
        Default V_DD/2 so t_cross = T_in + d × ln(2).
    """

    R_ohm: float = 158_114.0
    C_cell_nF: float = 1.0
    V_DD: float = 1.5
    V_th: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.V_th == 0.0:
            self.V_th = self.V_DD / 2.0

    # -------------------------------------------------------------- properties

    @property
    def delay_us(self) -> float:
        """Delay in µs: d = R_ohm × C_cell_nF × 1e-3."""
        return self.R_ohm * self.C_cell_nF * 1e-3

    # ---------------------------------------------------------------- methods

    def threshold_crossing_us(self, T_in_us: float = 0.0) -> float:
        """Analytic threshold crossing time (µs).

        t_cross = T_in + d × ln(V_DD / (V_DD − V_th))
        """
        return T_in_us + self.delay_us * math.log(self.V_DD / (self.V_DD - self.V_th))

    def waveform(self, t_us: np.ndarray, T_in_us: float = 0.0) -> np.ndarray:
        """Compute V_RC(t) over time array t_us (µs).

        Returns an array the same shape as t_us.
        """
        t = np.asarray(t_us, dtype=float)
        V = np.zeros_like(t)
        mask = t > T_in_us
        V[mask] = self.V_DD * (1.0 - np.exp(-(t[mask] - T_in_us) / self.delay_us))
        return V
