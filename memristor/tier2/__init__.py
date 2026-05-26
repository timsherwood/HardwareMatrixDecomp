"""Tier-2 PCB prototype simulation: Knowm SDC memristors + BJT nLSE sense amp.

Scales the ASIC design to bench-measurable µs timing (×10,000 from ns) using:
  - 1 nF load capacitors (vs 100 fF on-chip)
  - Same 50–500 kΩ RRAM resistance range → 50–500 µs delays
  - BJT exponential-transconductance sense amp with translinear gain stage
  - Knowm SDC memristor device model with cycle-to-cycle variability

PCB parameters:
  C_cell     = 1 nF
  d range    = 50–500 µs  (κ = 158 µs)
  V_DD_ramp  = 1.5 V
  V_th       = 0.75 V = V_DD/2
  gain_A     = 20  →  τ_sense ≈ 110 µs; must be < 26.3 to avoid premature BJT firing
  T_inactive = 1500 µs
"""
