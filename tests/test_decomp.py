import numpy as np

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.tile import MAX_TILE_DIM


def _low_rank_matrix(m: int, n: int, rank: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))


def test_full_rank_small_matrix_reconstructed_exactly() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 40))
    A, B = SVDDecomposition(rank=40).decompose(W)
    np.testing.assert_allclose(W, A @ B, atol=1e-10)


def test_output_shapes_are_consistent() -> None:
    rng = np.random.default_rng(0)
    for m, n in [(50, 50), (200, 100), (100, 300), (300, 400)]:
        W = rng.standard_normal((m, n))
        A, B = SVDDecomposition(rank=30).decompose(W)
        assert A.shape[0] == m
        assert B.shape[1] == n
        assert A.shape[1] == B.shape[0]


def test_rank_never_exceeds_max_tile_dim() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((500, 500))
    A, B = SVDDecomposition(rank=200).decompose(W)  # request > MAX_TILE_DIM
    assert A.shape[1] <= MAX_TILE_DIM
    assert B.shape[0] <= MAX_TILE_DIM


def test_explicit_rank_respected() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((200, 300))
    A, B = SVDDecomposition(rank=50).decompose(W)
    assert A.shape == (200, 50)
    assert B.shape == (50, 300)


def test_energy_threshold_captures_low_rank_matrix() -> None:
    W = _low_rank_matrix(200, 150, rank=10)
    A, B = SVDDecomposition(energy_threshold=0.999).decompose(W)
    err = np.linalg.norm(W - A @ B, "fro") / np.linalg.norm(W, "fro")
    assert err < 0.01


def test_reconstruction_error_method() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((100, 80))
    err = SVDDecomposition(rank=20).reconstruction_error(W)
    assert 0.0 <= err <= 1.0
