import numpy as np

from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.tile_grid import TileGrid


def test_small_matrix_uses_single_tile() -> None:
    W = np.random.default_rng(0).standard_normal((50, 40))
    grid = TileGrid.from_matrix(W)
    assert len(grid.tiles) == 1
    assert len(grid.tiles[0]) == 1


def test_large_row_dim_creates_multiple_row_blocks() -> None:
    W = np.random.default_rng(0).standard_normal((250, 40))
    grid = TileGrid.from_matrix(W)
    assert len(grid.tiles) == 1
    assert len(grid.tiles[0]) == 3  # ceil(250/100)


def test_large_col_dim_creates_multiple_col_blocks() -> None:
    W = np.random.default_rng(0).standard_normal((40, 250))
    grid = TileGrid.from_matrix(W)
    assert len(grid.tiles) == 3  # ceil(250/100)
    assert len(grid.tiles[0]) == 1


def test_forward_matches_numpy_small() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((60, 40))
    x = rng.standard_normal((8, 60))
    np.testing.assert_allclose(TileGrid.from_matrix(W).forward(x), x @ W, rtol=1e-5)


def test_forward_matches_numpy_large() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((250, 180))
    x = rng.standard_normal((16, 250))
    np.testing.assert_allclose(TileGrid.from_matrix(W).forward(x), x @ W, rtol=1e-5)


def test_output_shape() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((150, 80))
    x = rng.standard_normal((32, 150))
    assert TileGrid.from_matrix(W).forward(x).shape == (32, 80)


def test_noisy_grid_differs_from_clean() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((100, 100))
    x = rng.standard_normal((8, 100))
    clean = TileGrid.from_matrix(W)
    noisy = TileGrid.from_matrix(W, error_model=GaussianErrorModel(noise_std=0.2))
    assert not np.allclose(clean.forward(x), noisy.forward(x, rng=np.random.default_rng(1)))
