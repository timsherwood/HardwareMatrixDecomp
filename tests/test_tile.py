import numpy as np
import pytest

from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.tile import MAX_TILE_DIM, HardwareTile


def test_correct_matrix_multiply() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 40))
    x = rng.standard_normal((8, 50))
    tile = HardwareTile(W)
    np.testing.assert_allclose(tile.forward(x), x @ W)


def test_rejects_oversized_weights() -> None:
    with pytest.raises(ValueError, match="exceed MAX_TILE_DIM"):
        HardwareTile(np.ones((MAX_TILE_DIM + 1, 10)))


def test_rejects_non_2d_weights() -> None:
    with pytest.raises(ValueError):
        HardwareTile(np.ones(10))  # type: ignore[arg-type]


def test_max_size_is_accepted() -> None:
    tile = HardwareTile(np.zeros((MAX_TILE_DIM, MAX_TILE_DIM)))
    assert tile.weights.shape == (MAX_TILE_DIM, MAX_TILE_DIM)


def test_noisy_tile_output_differs_from_clean() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 50))
    x = np.ones((4, 50))
    clean = HardwareTile(W)
    noisy = HardwareTile(W, error_model=GaussianErrorModel(noise_std=0.2))
    assert not np.allclose(clean.forward(x), noisy.forward(x, np.random.default_rng(1)))


def test_noiseless_tile_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((30, 30))
    tile = HardwareTile(W)
    x = np.ones((4, 30))
    np.testing.assert_array_equal(tile.forward(x), tile.forward(x))
