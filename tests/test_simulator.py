import numpy as np
import pytest
import torch
import torch.nn as nn

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.simulator import run_simulation


def _build_mlp(in_dim: int = 64, hidden: int = 32, out_dim: int = 10) -> nn.Sequential:
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def _random_data(
    n: int = 100, in_dim: int = 64, n_classes: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return x, y


def test_result_has_valid_accuracy_range() -> None:
    model = _build_mlp()
    x, y = _random_data()
    result = run_simulation(model, x, y)
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert 0.0 <= result.noisy_accuracy <= 1.0


def test_no_error_model_baseline_equals_noisy() -> None:
    model = _build_mlp()
    x, y = _random_data()
    result = run_simulation(model, x, y, error_model=None)
    assert result.baseline_accuracy == result.noisy_accuracy


def test_extreme_noise_does_not_improve_accuracy() -> None:
    model = _build_mlp()
    x, y = _random_data(n=200)
    result = run_simulation(
        model,
        x,
        y,
        decomp=SVDDecomposition(rank=32),
        error_model=GaussianErrorModel(noise_std=10.0),
    )
    # Extreme noise may scramble predictions — accuracy should be ≤ baseline + small tolerance
    assert result.noisy_accuracy <= result.baseline_accuracy + 0.05


def test_tiny_noise_preserves_accuracy() -> None:
    model = _build_mlp()
    x, y = _random_data(n=200)
    result = run_simulation(
        model,
        x,
        y,
        decomp=SVDDecomposition(rank=32),
        error_model=GaussianErrorModel(noise_std=0.001),
    )
    assert abs(result.noisy_accuracy - result.baseline_accuracy) < 0.1


def test_simulation_result_metadata() -> None:
    model = _build_mlp()
    x, y = _random_data()
    result = run_simulation(
        model,
        x,
        y,
        decomp=SVDDecomposition(rank=20),
        error_model=GaussianErrorModel(noise_std=0.05),
    )
    assert result.n_samples == 100
    assert result.noise_std == pytest.approx(0.05)
    assert result.rank == 20
