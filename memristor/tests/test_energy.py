"""Tests for the hardware energy model."""

from __future__ import annotations

import pytest
import torch

from memristor.energy import (
    HardwareSpec,
    InferenceEnergy,
    TrainingEnergy,
    compare_to_digital,
    estimate_inference_energy,
    estimate_training_energy,
)
from memristor.network import MemristorNet


def _xor_net() -> MemristorNet:
    torch.manual_seed(0)
    return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)


def _mnist_net() -> MemristorNet:
    torch.manual_seed(0)
    return MemristorNet(n_inputs=64, hidden_sizes=[32], n_outputs=10)


class TestHardwareSpec:
    def test_e_cell_formula(self) -> None:
        spec = HardwareSpec(C_cell_fF=100.0, V_dd_V=0.8)
        assert spec.E_cell_fJ == pytest.approx(0.5 * 100.0 * 0.64, rel=1e-9)

    def test_e_cell_default(self) -> None:
        spec = HardwareSpec()
        assert spec.E_cell_fJ == pytest.approx(32.0, rel=1e-6)

    def test_driver_defaults_to_e_cell(self) -> None:
        spec = HardwareSpec()
        assert spec.E_driver_fJ is None
        assert spec.E_driver_effective_fJ == spec.E_cell_fJ

    def test_driver_override(self) -> None:
        spec = HardwareSpec(E_driver_fJ=50.0)
        assert spec.E_driver_effective_fJ == 50.0

    def test_e_pv_cell(self) -> None:
        spec = HardwareSpec(E_pv_pulse_fJ=10.0, n_pv_pulses_mean=20.0)
        assert spec.E_pv_cell_fJ == pytest.approx(200.0)


class TestInferenceEnergy:
    def test_all_components_positive(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        infer = estimate_inference_energy(net, spec)
        assert infer.E_input_fJ > 0
        assert infer.E_delay_fJ > 0
        assert infer.E_race_fJ > 0
        assert infer.E_total_fJ > 0

    def test_total_equals_sum(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        infer = estimate_inference_energy(net, spec)
        expected = infer.E_input_fJ + infer.E_delay_fJ + infer.E_race_fJ
        assert infer.E_total_fJ == pytest.approx(expected, rel=1e-9)

    def test_returns_inference_energy_type(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        result = estimate_inference_energy(net, spec)
        assert isinstance(result, InferenceEnergy)

    def test_e_total_pJ_conversion(self) -> None:  # noqa: N802
        net = _xor_net()
        spec = HardwareSpec()
        infer = estimate_inference_energy(net, spec)
        assert infer.E_total_pJ == pytest.approx(infer.E_total_fJ / 1000.0)

    def test_explicit_n_inputs_active(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        infer_all = estimate_inference_energy(net, spec, n_inputs_active=net.n_inputs)
        infer_half = estimate_inference_energy(net, spec, n_inputs_active=net.n_inputs // 2)
        assert infer_all.E_input_fJ > infer_half.E_input_fJ

    def test_larger_capacitance_more_energy(self) -> None:
        net = _mnist_net()
        spec_small = HardwareSpec(C_cell_fF=10.0)
        spec_large = HardwareSpec(C_cell_fF=500.0)
        infer_small = estimate_inference_energy(net, spec_small)
        infer_large = estimate_inference_energy(net, spec_large)
        assert infer_large.E_total_fJ > infer_small.E_total_fJ

    def test_mnist_net_less_energy_than_digital(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        cmp = compare_to_digital(net, spec)
        assert cmp["speedup"] > 1.0, "Timing net should use less energy than digital MACs"

    def test_str_representation(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        infer = estimate_inference_energy(net, spec)
        s = str(infer)
        assert "Total" in s
        assert "fJ" in s


class TestTrainingEnergy:
    def test_returns_training_energy_type(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        result = estimate_training_energy(net, spec, n_epochs=1, n_samples=100, batch_size=32)
        assert isinstance(result, TrainingEnergy)

    def test_training_energy_exceeds_inference(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        infer = estimate_inference_energy(net, spec)
        train = estimate_training_energy(net, spec, n_epochs=1, n_samples=6000, batch_size=128)
        assert train.E_total_fJ > infer.E_total_fJ

    def test_total_equals_sum(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        train = estimate_training_energy(net, spec, n_epochs=5, n_samples=500, batch_size=50)
        expected = train.E_inference_fJ + train.E_pv_fJ + train.E_tdc_fJ
        assert train.E_total_fJ == pytest.approx(expected, rel=1e-9)

    def test_more_epochs_more_energy(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        train1 = estimate_training_energy(net, spec, n_epochs=1, n_samples=1000, batch_size=100)
        train10 = estimate_training_energy(net, spec, n_epochs=10, n_samples=1000, batch_size=100)
        assert train10.E_total_fJ == pytest.approx(10 * train1.E_total_fJ, rel=1e-6)

    def test_n_batches_correct(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        train = estimate_training_energy(net, spec, n_epochs=1, n_samples=1000, batch_size=100)
        assert train.n_batches == 10

    def test_e_total_nJ_conversion(self) -> None:  # noqa: N802
        net = _xor_net()
        spec = HardwareSpec()
        train = estimate_training_energy(net, spec, n_epochs=1, n_samples=100, batch_size=10)
        assert train.E_total_nJ == pytest.approx(train.E_total_fJ / 1e6)

    def test_str_representation(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        train = estimate_training_energy(net, spec, n_epochs=1, n_samples=100, batch_size=10)
        s = str(train)
        assert "nJ/epoch" in s


class TestCompareToDigital:
    def test_returns_required_keys(self) -> None:
        net = _xor_net()
        spec = HardwareSpec()
        result = compare_to_digital(net, spec)
        for key in ("n_mac", "E_digital_fJ", "E_ours_fJ", "speedup", "E_per_mac_ours_fJ"):
            assert key in result

    def test_speedup_positive(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        result = compare_to_digital(net, spec)
        assert result["speedup"] > 0

    def test_digital_energy_proportional_to_mac_count(self) -> None:
        net = _mnist_net()
        spec = HardwareSpec()
        result = compare_to_digital(net, spec)
        assert result["E_digital_fJ"] == pytest.approx(
            result["n_mac"] * spec.E_mac_digital_fJ, rel=1e-9
        )

    def test_higher_digital_mac_energy_higher_speedup(self) -> None:
        net = _mnist_net()
        spec_lo = HardwareSpec(E_mac_digital_fJ=500.0)
        spec_hi = HardwareSpec(E_mac_digital_fJ=2000.0)
        assert (
            compare_to_digital(net, spec_hi)["speedup"]
            > compare_to_digital(net, spec_lo)["speedup"]
        )
