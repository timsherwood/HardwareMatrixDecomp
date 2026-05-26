"""Tests for BJTSenseAmp: exponential-transconductance nLSE sense amplifier.

Circuit physics being tested:
  Each delay branch drives a BJT base (V_BE = V_RC ramp).
  I_branch(t) = exp((V_BE(t) - V_th) / (gain_A × V_T))
  I_total = Σ I_branch_i fires at t* where I_total = 1.0.

Key properties:
  - Single branch fires at t_cross (by normalization)
  - N simultaneous branches fire τ_sense × ln(N) earlier than single branch
  - This is the nLSE soft-min with temperature τ_sense = gain_A × V_T × d / (V_DD − V_th)
  - A circuit offset C cancels in margin (T_minus − T_plus), so classification is unaffected
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from memristor.tier2.sense_amp import BJTSenseAmp, nLSE_us

# Shared PCB parameters
V_DD = 1.5    # V
V_TH = 0.75   # V = V_DD/2
V_T = 0.026   # V (room temperature)
GAIN_A = 20.0
D_NOM_US = 158.0


def _tau_s(d_us: float, gain_A: float = GAIN_A) -> float:
    """Expected τ_sense = gain_A × V_T × d / (V_DD − V_th)."""
    return gain_A * V_T * d_us / (V_DD - V_TH)


class TestNLSEHelper:
    def test_single_value(self):
        """nLSE of a single crossing time equals that time."""
        assert nLSE_us([100.0], tau_us=10.0) == pytest.approx(100.0, rel=1e-6)

    def test_two_equal_values(self):
        """nLSE([t, t], τ) = t − τ·ln(2)."""
        t, tau = 100.0, 10.0
        assert nLSE_us([t, t], tau) == pytest.approx(t - tau * math.log(2), rel=1e-6)

    def test_nLSE_le_min(self):
        """nLSE ≤ min(t_cross_i) for any inputs."""
        tcs = [50.0, 80.0, 120.0]
        tau = 15.0
        result = nLSE_us(tcs, tau)
        assert result <= min(tcs) + 1e-9

    def test_approaches_min_as_tau_to_zero(self):
        """nLSE → min as τ → 0."""
        tcs = [50.0, 100.0, 200.0]
        for tau in [1.0, 0.1, 0.01]:
            result = nLSE_us(tcs, tau)
            assert abs(result - min(tcs)) < tau * 2


class TestBJTSenseAmpBasic:
    def test_tau_sense_formula(self):
        """τ_sense = gain_A × V_T × d / (V_DD − V_th)."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A)
        expected = _tau_s(D_NOM_US)
        assert amp.tau_sense_us(D_NOM_US) == pytest.approx(expected, rel=1e-6)

    def test_tau_sense_proportional_to_gain(self):
        """τ_sense doubles when gain_A doubles."""
        amp_lo = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=10.0)
        amp_hi = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=20.0)
        ratio = amp_hi.tau_sense_us(D_NOM_US) / amp_lo.tau_sense_us(D_NOM_US)
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_tau_sense_proportional_to_d(self):
        """τ_sense is proportional to delay d."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A)
        tau_1 = amp.tau_sense_us(100.0)
        tau_2 = amp.tau_sense_us(200.0)
        assert tau_2 == pytest.approx(2 * tau_1, rel=1e-6)


class TestBJTSenseAmpFiring:
    def test_single_branch_fires_near_t_cross(self):
        """N=1 branch: fire time is within ±2τ_sense of analytic t_cross."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.2)
        T_in = 0.0
        t_cross = T_in + D_NOM_US * math.log(V_DD / (V_DD - V_TH))
        tau_s = _tau_s(D_NOM_US)
        t_fire = amp.fire_time_us(delays_us=[D_NOM_US], T_in_list=[T_in])
        assert abs(t_fire - t_cross) < 2.0 * tau_s

    def test_two_simultaneous_branches_fire_earlier_than_one(self):
        """N=2, identical branches → fires before N=1."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.2)
        fire_1 = amp.fire_time_us([D_NOM_US], [0.0])
        fire_2 = amp.fire_time_us([D_NOM_US, D_NOM_US], [0.0, 0.0])
        assert fire_2 < fire_1

    def test_multiplicity_advance_approx_tau_ln_N(self):
        """Advance for N simultaneous branches matches exact BJT formula.

        Exact (from RC ramp inversion):
            advance = d × ln(1 + (gain_A × V_T / (V_DD − V_th)) × ln(N))
                    = d × ln(1 + (τ_sense/d) × ln(N))

        The linear approximation τ_sense × ln(N) holds only when τ_sense << d.
        Here τ_sense/d ≈ 0.69 so the approximation over-estimates by 20–30%.
        The simulation is compared against the exact formula (within dt tolerance).
        """
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.2)
        fire_1 = amp.fire_time_us([D_NOM_US], [0.0])
        for N in [2, 3, 4]:
            fire_N = amp.fire_time_us([D_NOM_US] * N, [0.0] * N)
            advance = fire_1 - fire_N
            # Exact advance from RC ramp inversion (not the linear nLSE approx)
            gain_factor = GAIN_A * V_T / (V_DD - V_TH)   # = τ_sense / d
            advance_exact = D_NOM_US * math.log(1.0 + gain_factor * math.log(N))
            # Allow ±1 µs for dt_us = 0.2 µs discretisation
            assert advance == pytest.approx(advance_exact, abs=1.0), (
                f"N={N}: simulation advance={advance:.2f} µs, "
                f"exact={advance_exact:.2f} µs"
            )

    def test_earlier_arrival_dominates(self):
        """Fast branch (small d or early T_in) dominates sense-amp firing."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.2)
        t_cross_fast = D_NOM_US / 3 * math.log(V_DD / (V_DD - V_TH))  # d = d_nom/3
        t_cross_slow = D_NOM_US * 3 * math.log(V_DD / (V_DD - V_TH))  # d = d_nom×3
        fire = amp.fire_time_us(
            delays_us=[D_NOM_US / 3, D_NOM_US * 3],
            T_in_list=[0.0, 0.0],
        )
        assert fire < (t_cross_fast + t_cross_slow) / 2

    def test_three_branches_ordered_correctly(self):
        """Fire time with 3 branches is less than with 2 which is less than with 1."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.2)
        f1 = amp.fire_time_us([D_NOM_US], [0.0])
        f2 = amp.fire_time_us([D_NOM_US, D_NOM_US], [0.0, 0.0])
        f3 = amp.fire_time_us([D_NOM_US, D_NOM_US, D_NOM_US], [0.0, 0.0, 0.0])
        assert f3 < f2 < f1


class TestBJTSenseAmpNLSEApproximation:
    def test_fire_time_approximates_nLSE(self):
        """BJT fire time ≈ nLSE(t_cross_i; τ_sense) for different branch configs."""
        amp = BJTSenseAmp(V_DD=V_DD, V_th=V_TH, V_T=V_T, gain_A=GAIN_A, dt_us=0.1)

        # Three test cases: single, simultaneous pair, spread pair
        cases = [
            ([D_NOM_US], [0.0]),
            ([D_NOM_US, D_NOM_US], [0.0, 0.0]),
            ([D_NOM_US, D_NOM_US * 2], [0.0, 0.0]),
        ]
        for delays, T_ins in cases:
            t_crosses = [
                T + d * math.log(V_DD / (V_DD - V_TH))
                for d, T in zip(delays, T_ins)
            ]
            # Use average τ_sense for the analytical nLSE estimate
            tau_avg = float(np.mean([_tau_s(d) for d in delays]))
            nLSE_pred = nLSE_us(t_crosses, tau_avg)
            t_fire = amp.fire_time_us(delays, T_ins)
            # Allow ±3τ_avg tolerance (BJT approximation, not exact)
            assert abs(t_fire - nLSE_pred) < 3.0 * tau_avg, (
                f"delays={delays}: t_fire={t_fire:.2f}, nLSE={nLSE_pred:.2f}, "
                f"diff={abs(t_fire-nLSE_pred):.2f} > 3τ={3*tau_avg:.2f}"
            )


class TestMarginCancellation:
    def test_C_offset_cancels_in_margin(self):
        """Circuit offset C cancels in margin: margin = nLSE_neg − nLSE_pos."""
        tau = 50.0  # µs — representative τ_sense
        # Positive race: one fast branch
        t_cross_pos = [109.5]   # µs (≈ d_nom × ln2 for d_nom=158 µs)
        # Negative race: two simultaneous branches
        t_cross_neg = [120.0, 120.0]

        margin_no_C = nLSE_us(t_cross_neg, tau) - nLSE_us(t_cross_pos, tau)

        # With equal C added to both T_fire values (same sense-amp circuit)
        for C in [0.0, 20.0, 50.0, 100.0]:
            T_plus_C = C + nLSE_us(t_cross_pos, tau)
            T_minus_C = C + nLSE_us(t_cross_neg, tau)
            margin_with_C = T_minus_C - T_plus_C
            assert margin_with_C == pytest.approx(margin_no_C, rel=1e-9)

    def test_margin_sign_is_correct_for_simple_race(self):
        """Positive branch clearly faster → positive margin → p > 0.5."""
        tau = 50.0
        t_cross_pos = [80.0]    # fast
        t_cross_neg = [130.0]   # slow
        margin = nLSE_us(t_cross_neg, tau) - nLSE_us(t_cross_pos, tau)
        assert margin > 0.0  # T_minus > T_plus → positive → correct
