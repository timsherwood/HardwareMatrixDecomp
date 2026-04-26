import numpy as np
import pytest
import torch
import torch.nn as nn

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.network import DecomposedMLP


def _build_mlp() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def test_output_shape() -> None:
    model = _build_mlp()
    net = DecomposedMLP.from_pytorch_sequential(model, decomp=SVDDecomposition(rank=32))
    x = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    assert net.forward(x).shape == (16, 10)


def test_matches_pytorch_without_noise() -> None:
    model = _build_mlp()
    model.eval()
    net = DecomposedMLP.from_pytorch_sequential(model, decomp=SVDDecomposition(rank=32))

    x_np = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.tensor(x_np)).numpy()

    np.testing.assert_allclose(net.forward(x_np), torch_out, rtol=1e-3, atol=1e-4)


def test_noisy_network_differs_from_clean() -> None:
    model = _build_mlp()
    model.eval()
    x = np.random.default_rng(1).standard_normal((16, 64)).astype(np.float32)

    clean = DecomposedMLP.from_pytorch_sequential(model, decomp=SVDDecomposition(rank=32))
    noisy = DecomposedMLP.from_pytorch_sequential(
        model, decomp=SVDDecomposition(rank=32), error_model=GaussianErrorModel(noise_std=0.5)
    )
    assert not np.allclose(
        clean.forward(x), noisy.forward(x, rng=np.random.default_rng(0)), atol=1e-3
    )


def test_unsupported_module_raises() -> None:
    model = nn.Sequential(nn.Linear(10, 5), nn.Tanh())
    with pytest.raises(ValueError, match="Unsupported module"):
        DecomposedMLP.from_pytorch_sequential(model)
