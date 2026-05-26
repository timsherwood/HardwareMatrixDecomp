"""Tests for KnowmSDC stochastic memristor device model (Tier-2 PCB)."""

from __future__ import annotations

import numpy as np
import pytest

from memristor.tier2.device import KnowmSDC

R_MIN = 50_000.0    # Ω
R_NOM = 158_114.0   # Ω  (geometric mean: sqrt(50k × 500k))
R_MAX = 500_000.0   # Ω
C_CELL_F = 1e-9     # 1 nF


class TestKnowmSDCBasic:
    def test_default_resistance_in_range(self):
        """Default device starts within [R_min, R_max]."""
        dev = KnowmSDC()
        assert R_MIN <= dev.R <= R_MAX

    def test_nominal_resistance_set_correctly(self):
        """R=R_NOM is stored exactly."""
        dev = KnowmSDC(R=R_NOM)
        assert pytest.approx(R_NOM) == dev.R

    def test_delay_us_equals_r_times_c(self):
        """delay_us = R × C_cell × 1e6."""
        dev = KnowmSDC(R=R_NOM)
        expected_us = R_NOM * C_CELL_F * 1e6
        assert dev.delay_us(C_CELL_F) == pytest.approx(expected_us, rel=1e-6)

    def test_delay_us_at_r_min(self):
        """R_min → d = 50 µs."""
        dev = KnowmSDC(R=R_MIN)
        assert dev.delay_us(C_CELL_F) == pytest.approx(50.0, rel=1e-4)

    def test_delay_us_at_r_max(self):
        """R_max → d = 500 µs."""
        dev = KnowmSDC(R=R_MAX)
        assert dev.delay_us(C_CELL_F) == pytest.approx(500.0, rel=1e-4)


class TestKnowmSDCSwitching:
    def test_set_decreases_resistance_on_average(self):
        """Repeated SET pulses from high R → mean R decreases."""
        R_start = R_MAX * 0.8
        R_after = []
        for seed in range(60):
            dev = KnowmSDC(R=R_start, noise_frac=0.15)
            dev.set_pulse(rng=np.random.default_rng(seed))
            R_after.append(dev.R)
        assert np.mean(R_after) < R_start

    def test_reset_increases_resistance_on_average(self):
        """Repeated RESET pulses from low R → mean R increases."""
        R_start = R_MIN * 1.5
        R_after = []
        for seed in range(60):
            dev = KnowmSDC(R=R_start, noise_frac=0.15)
            dev.reset_pulse(rng=np.random.default_rng(seed))
            R_after.append(dev.R)
        assert np.mean(R_after) > R_start

    def test_resistance_stays_in_bounds_after_many_pulses(self):
        """After 200 random pulses R stays within [R_min, R_max]."""
        rng = np.random.default_rng(42)
        dev = KnowmSDC(R=R_NOM, noise_frac=0.15)
        for _ in range(200):
            if rng.random() > 0.5:
                dev.set_pulse(rng=rng)
            else:
                dev.reset_pulse(rng=rng)
            assert R_MIN <= dev.R <= R_MAX, f"R={dev.R:.0f} exited bounds"

    def test_cycle_noise_gives_variability(self):
        """With noise_frac > 0, same starting R gives different results across seeds."""
        R_results = []
        for seed in range(20):
            dev = KnowmSDC(R=R_NOM, noise_frac=0.15)
            dev.set_pulse(rng=np.random.default_rng(seed))
            R_results.append(dev.R)
        assert np.std(R_results) > 0.0

    def test_zero_noise_is_deterministic(self):
        """noise_frac=0 → every SET pulse from same R produces identical result."""
        R_vals = []
        for seed in range(5):
            dev = KnowmSDC(R=R_NOM, noise_frac=0.0)
            dev.set_pulse(rng=np.random.default_rng(seed))
            R_vals.append(dev.R)
        assert all(r == pytest.approx(R_vals[0]) for r in R_vals)

    def test_set_and_reset_are_approximately_symmetric(self):
        """SET from R_NOM then RESET returns close to R_NOM (zero noise)."""
        dev = KnowmSDC(R=R_NOM, noise_frac=0.0)
        dev.set_pulse(rng=np.random.default_rng(0))
        dev.reset_pulse(rng=np.random.default_rng(0))
        # Should be back near R_NOM (within one step of variability)
        assert abs(dev.R - R_NOM) < R_NOM * 0.15


class TestKnowmSDCProgramming:
    def test_pv_converges_to_target_delay(self):
        """P&V loop reaches d_target within tolerance."""
        dev = KnowmSDC(R=R_NOM, noise_frac=0.10)
        rng = np.random.default_rng(7)
        result = dev.program_to_delay(
            d_target_us=100.0, C_cell_F=C_CELL_F, tol_us=5.0, max_pulses=100, rng=rng
        )
        assert result["success"], f"P&V failed after {result['n_pulses']} pulses"
        assert abs(result["d_final_us"] - 100.0) <= 5.0

    def test_pv_success_rate_high(self):
        """≥ 90% of random target delays converge within 100 pulses."""
        rng = np.random.default_rng(0)
        n_targets = 50
        successes = 0
        for _ in range(n_targets):
            R_init = float(rng.uniform(R_MIN, R_MAX))
            d_target = float(rng.uniform(60.0, 450.0))
            dev = KnowmSDC(R=R_init, noise_frac=0.10)
            result = dev.program_to_delay(
                d_target_us=d_target, C_cell_F=C_CELL_F,
                tol_us=5.0, max_pulses=100, rng=rng,
            )
            if result["success"]:
                successes += 1
        assert successes / n_targets >= 0.90, f"Success rate {successes}/{n_targets} < 90%"

    def test_pv_result_keys(self):
        """program_to_delay returns expected keys."""
        dev = KnowmSDC(R=R_NOM)
        result = dev.program_to_delay(150.0, C_CELL_F, rng=np.random.default_rng(0))
        assert set(result.keys()) == {"success", "d_final_us", "n_pulses", "polarity"}

    def test_pv_polarity_set_when_delay_too_long(self):
        """Programming to shorter delay → SET polarity."""
        dev = KnowmSDC(R=R_MAX, noise_frac=0.0)  # start at max delay (500 µs)
        result = dev.program_to_delay(100.0, C_CELL_F, rng=np.random.default_rng(0))
        assert result["polarity"] == "SET"

    def test_pv_polarity_reset_when_delay_too_short(self):
        """Programming to longer delay → RESET polarity."""
        dev = KnowmSDC(R=R_MIN, noise_frac=0.0)  # start at min delay (50 µs)
        result = dev.program_to_delay(400.0, C_CELL_F, rng=np.random.default_rng(0))
        assert result["polarity"] == "RESET"
