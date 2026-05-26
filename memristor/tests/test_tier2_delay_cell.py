"""Tests for RCDelayCell at PCB µs scale (Tier-2 prototype)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from memristor.tier2.delay_cell import RCDelayCell

# Nominal PCB parameters
V_DD = 1.5      # V  (RC ramp supply — matches BJT V_BE operating range)
V_TH = 0.75     # V  (= V_DD/2; threshold for 50% crossing)
C_NF = 1.0      # nF
R_NOM = 158_114  # Ω


class TestRCDelayCellDelay:
    def test_delay_formula_nominal(self):
        """d_us = R_ohm × C_cell_nF × 1e-3 at nominal parameters."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        expected = R_NOM * C_NF * 1e-3  # µs
        assert cell.delay_us == pytest.approx(expected, rel=1e-6)

    def test_delay_at_r_min(self):
        """R=50 kΩ, C=1 nF → d = 50 µs."""
        cell = RCDelayCell(R_ohm=50_000, C_cell_nF=1.0, V_DD=V_DD, V_th=V_TH)
        assert cell.delay_us == pytest.approx(50.0, rel=1e-4)

    def test_delay_at_r_max(self):
        """R=500 kΩ, C=1 nF → d = 500 µs."""
        cell = RCDelayCell(R_ohm=500_000, C_cell_nF=1.0, V_DD=V_DD, V_th=V_TH)
        assert cell.delay_us == pytest.approx(500.0, rel=1e-4)

    def test_delay_proportional_to_r(self):
        """Delay doubles when R doubles (C fixed)."""
        cell_1x = RCDelayCell(R_ohm=100_000, C_cell_nF=1.0, V_DD=V_DD, V_th=V_TH)
        cell_2x = RCDelayCell(R_ohm=200_000, C_cell_nF=1.0, V_DD=V_DD, V_th=V_TH)
        assert cell_2x.delay_us == pytest.approx(2 * cell_1x.delay_us, rel=1e-6)

    def test_delay_proportional_to_c(self):
        """Delay doubles when C doubles (R fixed)."""
        cell_1x = RCDelayCell(R_ohm=R_NOM, C_cell_nF=1.0, V_DD=V_DD, V_th=V_TH)
        cell_2x = RCDelayCell(R_ohm=R_NOM, C_cell_nF=2.0, V_DD=V_DD, V_th=V_TH)
        assert cell_2x.delay_us == pytest.approx(2 * cell_1x.delay_us, rel=1e-6)


class TestRCDelayCellThresholdCrossing:
    def test_crossing_analytic_formula(self):
        """t_cross = T_in + d × ln(V_DD / (V_DD − V_th))."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 100.0
        t_cross = cell.threshold_crossing_us(T_in_us=T_in)
        expected = T_in + cell.delay_us * math.log(V_DD / (V_DD - V_TH))
        assert t_cross == pytest.approx(expected, rel=1e-6)

    def test_crossing_at_half_vdd_is_d_ln2(self):
        """At V_th = V_DD/2, t_cross = T_in + d × ln(2)."""
        d_us = 200.0
        R_ohm = int(d_us * 1e3)  # C=1 nF → R = d_us × 1e3 Ω
        cell = RCDelayCell(R_ohm=R_ohm, C_cell_nF=1.0, V_DD=V_DD, V_th=V_DD / 2)
        T_in = 0.0
        t_cross = cell.threshold_crossing_us(T_in_us=T_in)
        assert t_cross == pytest.approx(d_us * math.log(2), rel=1e-4)

    def test_crossing_shifts_with_t_in(self):
        """Crossing time shifts exactly by ΔT when T_in shifts by ΔT."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        t1 = cell.threshold_crossing_us(T_in_us=0.0)
        t2 = cell.threshold_crossing_us(T_in_us=50.0)
        assert (t2 - t1) == pytest.approx(50.0, rel=1e-6)


class TestRCDelayCellWaveform:
    def test_waveform_zero_before_t_in(self):
        """V_RC(t) = 0 for all t ≤ T_in."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 200.0
        t_arr = np.linspace(0.0, T_in - 0.01, 100)
        V = cell.waveform(t_arr, T_in_us=T_in)
        assert np.all(V == 0.0)

    def test_waveform_at_threshold_crossing(self):
        """V_RC(t_cross) == V_th."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 50.0
        t_cross = cell.threshold_crossing_us(T_in_us=T_in)
        V = cell.waveform(np.array([t_cross]), T_in_us=T_in)
        assert V[0] == pytest.approx(V_TH, rel=1e-4)

    def test_waveform_approaches_vdd_asymptote(self):
        """V_RC(T_in + 10d) ≈ V_DD (within 0.01%)."""
        cell = RCDelayCell(R_ohm=50_000, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 0.0
        t_far = np.array([T_in + 10 * cell.delay_us])
        V = cell.waveform(t_far, T_in_us=T_in)
        assert V[0] == pytest.approx(V_DD, rel=1e-4)

    def test_waveform_monotone_increasing(self):
        """V_RC(t) is non-decreasing for t ≥ T_in."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 0.0
        t_arr = np.linspace(T_in, T_in + 5 * cell.delay_us, 500)
        V = cell.waveform(t_arr, T_in_us=T_in)
        diffs = np.diff(V)
        assert np.all(diffs >= -1e-12)

    def test_numerical_crossing_matches_analytic(self):
        """Numerical threshold detection agrees with analytic t_cross (± 1 µs)."""
        cell = RCDelayCell(R_ohm=R_NOM, C_cell_nF=C_NF, V_DD=V_DD, V_th=V_TH)
        T_in = 50.0
        t_analytic = cell.threshold_crossing_us(T_in_us=T_in)
        t_arr = np.linspace(T_in, T_in + 5 * cell.delay_us, 500_000)
        V = cell.waveform(t_arr, T_in_us=T_in)
        idx = int(np.argmax(cell.V_th <= V))
        t_numeric = t_arr[idx]
        assert abs(t_numeric - t_analytic) < 1.0  # within 1 µs
