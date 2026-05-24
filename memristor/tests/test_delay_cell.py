"""Tests for DelayCell and program-and-verify."""

from __future__ import annotations

import numpy as np
import pytest

from memristor.delay_cell import DelayCell


class TestDelayCell:
    def test_default_delay_is_kappa(self) -> None:
        """u=0 → d = kappa."""
        cell = DelayCell(u=0.0, kappa=15.81)
        assert cell.delay == pytest.approx(15.81, rel=1e-4)

    def test_delay_clamped_to_dmin(self) -> None:
        """Very large u should clamp to d_min."""
        cell = DelayCell(u=100.0, d_min=5.0, d_max=50.0)
        assert cell.delay == pytest.approx(5.0)

    def test_delay_clamped_to_dmax(self) -> None:
        """Very negative u should clamp to d_max."""
        cell = DelayCell(u=-100.0, d_min=5.0, d_max=50.0)
        assert cell.delay == pytest.approx(50.0)

    def test_delay_decreases_with_u(self) -> None:
        """Higher u (more conductance) → shorter delay."""
        cell = DelayCell()
        d0 = cell.delay
        cell.u += 1.0
        assert cell.delay < d0

    def test_conductance_is_exp_u(self) -> None:
        cell = DelayCell(u=2.0)
        assert cell.conductance == pytest.approx(np.exp(2.0), rel=1e-6)

    def test_set_target_delay_roundtrip(self) -> None:
        """set_target_delay then read delay should recover the target."""
        cell = DelayCell()
        for d_t in [5.5, 15.0, 30.0, 49.0]:
            cell.set_target_delay(d_t)
            assert cell.delay == pytest.approx(d_t, rel=1e-6)

    def test_set_target_delay_clamps(self) -> None:
        cell = DelayCell(d_min=5.0, d_max=50.0)
        cell.set_target_delay(1.0)  # below d_min
        assert cell.delay == pytest.approx(5.0)
        cell.set_target_delay(200.0)  # above d_max
        assert cell.delay == pytest.approx(50.0)

    def test_local_update_positive_lambda_decreases_delay(self) -> None:
        """lambda > 0 → loss increases with delay → delay should decrease."""
        cell = DelayCell(u=0.0)
        d_before = cell.delay
        cell.apply_local_update(lam=1.0, eta=0.01)
        assert cell.delay < d_before

    def test_local_update_negative_lambda_increases_delay(self) -> None:
        cell = DelayCell(u=0.0)
        d_before = cell.delay
        cell.apply_local_update(lam=-1.0, eta=0.01)
        assert cell.delay > d_before

    def test_local_update_zero_lambda_no_change(self) -> None:
        cell = DelayCell(u=0.5)
        d_before = cell.delay
        cell.apply_local_update(lam=0.0, eta=0.1)
        assert cell.delay == pytest.approx(d_before)

    def test_measure_noiseless_equals_delay(self) -> None:
        cell = DelayCell()
        assert cell.measure(jitter_std=0.0) == pytest.approx(cell.delay)

    def test_measure_jitter_adds_noise(self) -> None:
        rng = np.random.default_rng(7)
        cell = DelayCell(u=0.0)
        measurements = [cell.measure(jitter_std=1.0, rng=rng) for _ in range(500)]
        assert abs(np.mean(measurements) - cell.delay) < 0.2
        assert abs(np.std(measurements) - 1.0) < 0.2


class TestProgramAndVerify:
    def test_can_hit_target_within_tolerance(self) -> None:
        """Ideal (no jitter) P&V should converge to any target in range."""
        cell = DelayCell()
        for d_target in [7.0, 15.0, 25.0, 40.0]:
            result = cell.program_and_verify(d_target, tol=0.5, jitter_std=0.0)
            assert result["success"], f"Failed for d_target={d_target}"
            assert abs(float(result["d_final"]) - d_target) <= 0.5

    def test_set_polarity_for_lower_target(self) -> None:
        """When current delay > target, polarity should be SET."""
        cell = DelayCell(u=-2.0)  # large delay
        result = cell.program_and_verify(d_target=10.0, tol=0.5, jitter_std=0.0)
        assert result["polarity"] == "SET"

    def test_reset_polarity_for_higher_target(self) -> None:
        """When current delay < target, polarity should be RESET."""
        cell = DelayCell(u=2.0)  # small delay
        result = cell.program_and_verify(d_target=40.0, tol=0.5, jitter_std=0.0)
        assert result["polarity"] == "RESET"

    def test_no_pulses_if_already_at_target(self) -> None:
        cell = DelayCell()
        d = cell.delay
        result = cell.program_and_verify(d_target=d, tol=1.0, jitter_std=0.0)
        assert int(result["n_pulses"]) == 0
        assert result["success"]

    def test_returns_required_keys(self) -> None:
        cell = DelayCell()
        result = cell.program_and_verify(20.0)
        assert set(result.keys()) == {"d_final", "n_pulses", "polarity", "success", "saturated"}
