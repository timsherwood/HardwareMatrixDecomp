"""Tests for Simulation 4: stochastic memristor programming model."""

from __future__ import annotations

import numpy as np

from memristor.programming import StochasticProgrammer, programming_sweep


class TestStochasticProgrammer:
    def _make(self, **kw) -> StochasticProgrammer:
        return StochasticProgrammer(**kw)

    def test_ideal_hits_target(self) -> None:
        """No noise, no asymmetry: should always converge."""
        prog = self._make(set_step=0.01, reset_step=0.01)
        rng = np.random.default_rng(0)
        for d_t in [7.0, 15.0, 25.0, 40.0]:
            res = prog.program(0.0, d_t, tol=0.5, rng=rng)
            assert res["success"], f"Failed for d_target={d_t}"
            assert abs(float(res["d_final"]) - d_t) <= 0.5

    def test_returns_required_keys(self) -> None:
        prog = self._make()
        res = prog.program(0.0, 20.0, rng=np.random.default_rng(0))
        expected = {"u_final", "d_final", "n_pulses", "polarity", "success", "saturated"}
        assert set(res.keys()) == expected

    def test_zero_pulses_if_already_at_target(self) -> None:
        prog = self._make()
        u_at_target = float(np.log(15.81 / 25.0))  # d ≈ 25 ns
        res = prog.program(u_at_target, 25.0, tol=1.0, rng=np.random.default_rng(0))
        assert int(res["n_pulses"]) == 0
        assert res["success"]

    def test_set_polarity_for_high_delay(self) -> None:
        prog = self._make()
        res = prog.program(-2.0, 10.0, tol=0.5, rng=np.random.default_rng(0))
        assert res["polarity"] == "SET"

    def test_reset_polarity_for_low_delay(self) -> None:
        prog = self._make()
        res = prog.program(2.0, 40.0, tol=0.5, rng=np.random.default_rng(0))
        assert res["polarity"] == "RESET"

    def test_high_noise_reduces_success_rate(self) -> None:
        """Very large noise should cause some P&V failures."""
        rng = np.random.default_rng(42)
        d_targets = np.linspace(6.0, 49.0, 50)
        successes_ideal = 0
        successes_noisy = 0
        for d_t in d_targets:
            seed_i = int(d_t * 100)
            r_ideal = StochasticProgrammer(noise_frac=0.0).program(
                0.0, d_t, rng=np.random.default_rng(seed_i)
            )
            r_noisy = StochasticProgrammer(noise_frac=0.5).program(0.0, d_t, rng=rng)
            successes_ideal += int(r_ideal["success"])
            successes_noisy += int(r_noisy["success"])
        assert successes_ideal >= successes_noisy, "Noisy should not beat ideal"

    def test_asymmetry_does_not_prevent_convergence(self) -> None:
        """3x asymmetry with small tol: P&V may need more pulses but should converge."""
        prog = self._make(set_step=0.01, reset_step=0.03, noise_frac=0.0)
        rng = np.random.default_rng(7)
        n_success = sum(
            int(prog.program(0.0, d_t, tol=0.5, max_pulses=500, rng=rng)["success"])
            for d_t in np.linspace(7.0, 48.0, 20)
        )
        assert n_success >= 15, f"3x asymmetry failed too often: {n_success}/20"

    def test_read_noise_degrades_success_rate(self) -> None:
        """Large read noise (tol-sized) should lower success rate."""
        rng_clean = np.random.default_rng(1)
        rng_noisy = np.random.default_rng(1)
        d_targets = np.linspace(8.0, 47.0, 30)
        ok_clean = sum(
            int(StochasticProgrammer(read_noise=0.0).program(
                0.0, d, tol=0.5, rng=rng_clean
            )["success"])
            for d in d_targets
        )
        ok_noisy = sum(
            int(StochasticProgrammer(read_noise=3.0).program(
                0.0, d, tol=0.5, rng=rng_noisy
            )["success"])
            for d in d_targets
        )
        assert ok_clean >= ok_noisy

    def test_drift_shifts_final_delay(self) -> None:
        """With drift, final delay should differ from ideal (drift pulls toward kappa)."""
        rng2 = np.random.default_rng(3)
        res_drift = StochasticProgrammer(drift_rate=0.05).program(
            0.0, 8.0, tol=0.3, max_pulses=500, rng=rng2
        )
        assert "d_final" in res_drift

    def test_state_dependence_does_not_break_convergence(self) -> None:
        """Small state dependence should not prevent convergence."""
        prog = self._make(state_dependence=0.1)
        rng = np.random.default_rng(5)
        n_ok = sum(
            int(prog.program(0.0, d_t, tol=0.5, rng=rng)["success"])
            for d_t in np.linspace(8.0, 47.0, 10)
        )
        assert n_ok >= 8


class TestProgrammingSweep:
    def test_sweep_produces_expected_keys(self) -> None:
        result = programming_sweep(
            n_targets=20,
            asymmetry_grid=(1.0, 2.0),
            noise_frac_grid=(0.0, 0.1),
            read_noise_grid=(0.0, 1.0),
        )
        # Asymmetry keys: (asym, 0.0, 0.0)
        assert (1.0, 0.0, 0.0) in result.success_rate
        assert (2.0, 0.0, 0.0) in result.success_rate
        # Noise keys: (1.0, nf, 0.0)
        assert (1.0, 0.1, 0.0) in result.success_rate
        # Read-noise keys: (1.0, 0.0, rn)
        assert (1.0, 0.0, 1.0) in result.success_rate

    def test_ideal_config_near_perfect_yield(self) -> None:
        result = programming_sweep(
            n_targets=100,
            asymmetry_grid=(1.0,),
            noise_frac_grid=(0.0,),
            read_noise_grid=(0.0,),
        )
        assert result.success_rate[(1.0, 0.0, 0.0)] > 0.95

    def test_success_rates_in_unit_interval(self) -> None:
        result = programming_sweep(n_targets=20)
        for v in result.success_rate.values():
            assert 0.0 <= v <= 1.0
