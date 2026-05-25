import numpy as np
import pytest
import torch
import torch.nn as nn

from hardware_matrix_decomp.analysis import (
    SweepConfig,
    forward_with_taps,
    per_layer_relative_error,
    run_sweep,
)
from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.network import DecomposedMLP


def _build_mlp() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))


def _decomposed(model: nn.Sequential, noise_std: float = 0.0) -> DecomposedMLP:
    error_model = GaussianErrorModel(noise_std=noise_std) if noise_std > 0 else None
    return DecomposedMLP.from_pytorch_sequential(
        model, decomp=SVDDecomposition(rank=16), error_model=error_model
    )


# --- forward_with_taps ---


def test_taps_count_equals_layer_count() -> None:
    model = _build_mlp()
    net = _decomposed(model)
    x = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    _, taps = forward_with_taps(net, x)
    assert len(taps) == len(net.layers)


def test_final_tap_equals_forward_output() -> None:
    model = _build_mlp()
    net = _decomposed(model)
    x = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    out, taps = forward_with_taps(net, x)
    np.testing.assert_array_equal(out, taps[-1])


def test_taps_have_correct_shapes() -> None:
    model = _build_mlp()
    net = _decomposed(model)
    x = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    _, taps = forward_with_taps(net, x)
    # MLP: 32→16→8→4, 3 linear layers → 3 taps
    assert taps[0].shape == (8, 16)
    assert taps[1].shape == (8, 8)
    assert taps[2].shape == (8, 4)


def test_taps_with_rng_differs_from_no_rng_under_noise() -> None:
    model = _build_mlp()
    net = _decomposed(model, noise_std=0.5)
    x = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    _, taps_noisy = forward_with_taps(net, x, rng=np.random.default_rng(1))
    _, taps_clean = forward_with_taps(_decomposed(model, noise_std=0.0), x)
    assert not np.allclose(taps_noisy[0], taps_clean[0])


# --- per_layer_relative_error ---


def test_per_layer_error_zero_for_zero_noise() -> None:
    model = _build_mlp()
    clean = _decomposed(model, noise_std=0.0)
    also_clean = _decomposed(model, noise_std=0.0)
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    errors = per_layer_relative_error(clean, also_clean, x, n_samples=5, seed=0)
    for e in errors:
        assert e == pytest.approx(0.0, abs=1e-6)


def test_per_layer_error_nonzero_for_nonzero_noise() -> None:
    model = _build_mlp()
    clean = _decomposed(model, noise_std=0.0)
    noisy = _decomposed(model, noise_std=0.2)
    x = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float32)
    errors = per_layer_relative_error(clean, noisy, x, n_samples=10, seed=0)
    assert all(e > 0 for e in errors)


def test_per_layer_error_increases_with_noise_level() -> None:
    model = _build_mlp()
    clean = _decomposed(model, noise_std=0.0)
    noisy_low = _decomposed(model, noise_std=0.01)
    noisy_high = _decomposed(model, noise_std=0.3)
    x = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    errors_low = per_layer_relative_error(clean, noisy_low, x, n_samples=20, seed=0)
    errors_high = per_layer_relative_error(clean, noisy_high, x, n_samples=20, seed=0)
    # Mean error across all layers should be larger for higher noise
    assert np.mean(errors_high) > np.mean(errors_low)


def test_per_layer_error_returns_one_value_per_layer() -> None:
    model = _build_mlp()
    clean = _decomposed(model)
    noisy = _decomposed(model, noise_std=0.1)
    x = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float32)
    errors = per_layer_relative_error(clean, noisy, x, n_samples=5, seed=0)
    assert len(errors) == len(clean.layers)


# --- run_sweep ---


def test_sweep_total_result_count() -> None:
    model = _build_mlp()
    x = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 4, size=50)
    config = SweepConfig(ranks=[8, 16], noise_stds=[0.0, 0.01, 0.1], seeds=[0, 1])
    results = run_sweep(model, x, y, config)
    assert len(results) == 2 * 3 * 2  # 12


def test_sweep_zero_noise_baseline_equals_noisy() -> None:
    model = _build_mlp()
    x = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 4, size=50)
    config = SweepConfig(ranks=[16], noise_stds=[0.0], seeds=[0])
    results = run_sweep(model, x, y, config)
    assert len(results) == 1
    assert results[0].baseline_acc == results[0].noisy_acc


def test_sweep_result_metadata_correct() -> None:
    model = _build_mlp()
    x = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 4, size=50)
    config = SweepConfig(ranks=[12], noise_stds=[0.05], seeds=[3])
    results = run_sweep(model, x, y, config)
    r = results[0]
    assert r.rank == 12
    assert r.noise_std == pytest.approx(0.05)
    assert r.seed == 3


def test_sweep_accuracy_in_valid_range() -> None:
    model = _build_mlp()
    x = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 4, size=50)
    config = SweepConfig(ranks=[8, 16], noise_stds=[0.0, 0.1], seeds=[0])
    results = run_sweep(model, x, y, config)
    for r in results:
        assert 0.0 <= r.baseline_acc <= 1.0
        assert 0.0 <= r.noisy_acc <= 1.0
