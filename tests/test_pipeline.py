import numpy as np

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.pipeline import LayerPipeline


def test_pipeline_exact_reconstruction_small() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 40))
    x = rng.standard_normal((8, 50))
    pipeline = LayerPipeline.from_weight_matrix(W, decomp=SVDDecomposition(rank=40))
    np.testing.assert_allclose(pipeline.forward(x), x @ W, atol=1e-8)


def test_pipeline_low_rank_approximation() -> None:
    rng = np.random.default_rng(0)
    true_rank = 20
    W = rng.standard_normal((200, true_rank)) @ rng.standard_normal((true_rank, 150))
    x = rng.standard_normal((16, 200))
    pipeline = LayerPipeline.from_weight_matrix(W, decomp=SVDDecomposition(rank=true_rank))
    np.testing.assert_allclose(pipeline.forward(x), x @ W, rtol=1e-4, atol=1e-6)


def test_pipeline_output_shape() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((300, 200))
    x = rng.standard_normal((32, 300))
    pipeline = LayerPipeline.from_weight_matrix(W, decomp=SVDDecomposition(rank=50))
    assert pipeline.forward(x).shape == (32, 200)


def test_noisy_pipeline_output_differs_from_clean() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((100, 80))
    x = rng.standard_normal((8, 100))
    clean = LayerPipeline.from_weight_matrix(W)
    noisy = LayerPipeline.from_weight_matrix(W, error_model=GaussianErrorModel(noise_std=0.5))
    assert not np.allclose(
        clean.forward(x), noisy.forward(x, rng=np.random.default_rng(1)), atol=1e-3
    )
