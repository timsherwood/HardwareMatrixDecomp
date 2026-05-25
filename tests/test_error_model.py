import numpy as np

from hardware_matrix_decomp.error_model import GaussianErrorModel


def test_zero_noise_std_returns_unchanged_weights() -> None:
    model = GaussianErrorModel(noise_std=0.0)
    W = np.ones((10, 10))
    result = model.apply(W, np.random.default_rng(0))
    np.testing.assert_array_equal(result, W)


def test_nonzero_noise_changes_weights() -> None:
    model = GaussianErrorModel(noise_std=0.1)
    W = np.ones((10, 10))
    result = model.apply(W, np.random.default_rng(0))
    assert not np.allclose(result, W)


def test_noise_has_correct_standard_deviation() -> None:
    # σ should equal noise_std × mean(|W|) = 0.01 × 10.0 = 0.1
    model = GaussianErrorModel(noise_std=0.01)
    W = np.ones((100, 100)) * 10.0
    rng = np.random.default_rng(42)
    noise_samples = [(model.apply(W, rng) - W).flatten() for _ in range(500)]
    noise = np.concatenate(noise_samples)
    assert abs(noise.std() - 0.1) < 0.01


def test_noise_is_zero_mean() -> None:
    model = GaussianErrorModel(noise_std=0.05)
    W = np.random.default_rng(1).standard_normal((50, 50))
    rng = np.random.default_rng(2)
    noise_samples = [(model.apply(W, rng) - W).flatten() for _ in range(300)]
    noise = np.concatenate(noise_samples)
    assert abs(noise.mean()) < 0.01


def test_successive_calls_produce_different_noise() -> None:
    model = GaussianErrorModel(noise_std=0.1)
    W = np.ones((10, 10))
    rng = np.random.default_rng(0)
    r1 = model.apply(W, rng)
    r2 = model.apply(W, rng)
    assert not np.allclose(r1, r2)
