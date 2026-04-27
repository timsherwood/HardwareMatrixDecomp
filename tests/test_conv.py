"""Tests for conv layer support: unfold_input, DecomposedConv2d, DecomposedNetwork."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.network import (
    DecomposedConv2d,
    DecomposedNetwork,
    unfold_input,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv(
    in_channels: int = 3,
    out_channels: int = 8,
    kernel_size: int = 3,
    padding: int = 1,
    seed: int = 0,
) -> nn.Conv2d:
    torch.manual_seed(seed)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
    return conv


def _decomp(rank: int = 32) -> SVDDecomposition:
    return SVDDecomposition(rank=rank)


def _build_small_cnn() -> nn.Sequential:
    """Tiny CNN: Conv→ReLU→MaxPool→Flatten→Linear."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10),
    )


# ---------------------------------------------------------------------------
# unfold_input
# ---------------------------------------------------------------------------


def test_unfold_output_shape() -> None:
    x = np.ones((2, 3, 8, 8), dtype=np.float32)
    # kH=3, kW=3, stride=1, padding=1 → H_out=W_out=8
    col = unfold_input(x, kH=3, kW=3, stride=1, padding=1)
    assert col.shape == (2 * 8 * 8, 3 * 3 * 3)


def test_unfold_output_shape_no_padding() -> None:
    x = np.ones((4, 1, 6, 6), dtype=np.float32)
    # kH=3, kW=3, stride=1, padding=0 → H_out=W_out=4
    col = unfold_input(x, kH=3, kW=3, stride=1, padding=0)
    assert col.shape == (4 * 4 * 4, 1 * 3 * 3)


def test_unfold_output_shape_stride2() -> None:
    x = np.ones((2, 3, 8, 8), dtype=np.float32)
    # kH=3, kW=3, stride=2, padding=1 → H_out=W_out=4
    col = unfold_input(x, kH=3, kW=3, stride=2, padding=1)
    assert col.shape == (2 * 4 * 4, 3 * 3 * 3)


def test_unfold_matches_torch_unfold() -> None:
    """unfold_input result must match torch functional.unfold numerically."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    col_np = unfold_input(x_np, kH=3, kW=3, stride=1, padding=1)

    # functional.unfold returns (N, C*kH*kW, H_out*W_out); rearrange to (N*H_out*W_out, C*kH*kW)
    unfolded = functional.unfold(x_t, kernel_size=3, padding=1, stride=1)  # (2, 27, 64)
    col_torch = unfolded.permute(0, 2, 1).reshape(-1, 27).numpy()  # (128, 27)

    np.testing.assert_allclose(col_np, col_torch, rtol=1e-5, atol=1e-6)


def test_unfold_matches_torch_no_padding() -> None:
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((3, 2, 6, 6)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    col_np = unfold_input(x_np, kH=3, kW=3, stride=1, padding=0)
    unfolded = functional.unfold(x_t, kernel_size=3, padding=0, stride=1)  # (3, 18, 16)
    col_torch = unfolded.permute(0, 2, 1).reshape(-1, 18).numpy()

    np.testing.assert_allclose(col_np, col_torch, rtol=1e-5, atol=1e-6)


def test_unfold_1x1_kernel() -> None:
    """1×1 conv unfold is equivalent to reshape."""
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 4, 5, 5)).astype(np.float32)
    col = unfold_input(x_np, kH=1, kW=1, stride=1, padding=0)
    # Should be (2*5*5, 4) = (50, 4)
    assert col.shape == (50, 4)


# ---------------------------------------------------------------------------
# DecomposedConv2d
# ---------------------------------------------------------------------------


def test_decomposed_conv_output_shape() -> None:
    conv = _make_conv(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x = np.random.default_rng(0).standard_normal((2, 3, 8, 8)).astype(np.float32)
    dc = DecomposedConv2d.from_pytorch_conv2d(conv, decomp=_decomp(32))
    out = dc.forward(x)
    assert out.shape == (2, 8, 8, 8)


def test_decomposed_conv_output_shape_no_padding() -> None:
    conv = _make_conv(in_channels=3, out_channels=16, kernel_size=3, padding=0)
    x = np.random.default_rng(0).standard_normal((4, 3, 10, 10)).astype(np.float32)
    dc = DecomposedConv2d.from_pytorch_conv2d(conv, decomp=_decomp(32))
    out = dc.forward(x)
    assert out.shape == (4, 16, 8, 8)


def test_decomposed_conv_matches_pytorch_full_rank() -> None:
    """At full rank, DecomposedConv2d must match nn.Conv2d output numerically."""
    conv = _make_conv(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    conv.eval()
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    with torch.no_grad():
        ref = conv(x_t).numpy()  # (2, 8, 8, 8)

    dc = DecomposedConv2d.from_pytorch_conv2d(conv, decomp=SVDDecomposition(rank=None))
    out = dc.forward(x_np)  # (2, 8, 8, 8)

    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_decomposed_conv_with_relu() -> None:
    conv = _make_conv(in_channels=2, out_channels=4, kernel_size=3, padding=1)
    x = np.random.default_rng(5).standard_normal((1, 2, 6, 6)).astype(np.float32)
    dc = DecomposedConv2d.from_pytorch_conv2d(conv, decomp=_decomp(8), activation="relu")
    out = dc.forward(x)
    assert np.all(out >= 0.0), "ReLU output must be non-negative"
    assert out.shape == (1, 4, 6, 6)


def test_noisy_conv_differs_from_clean() -> None:
    conv = _make_conv(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x = np.random.default_rng(0).standard_normal((4, 3, 8, 8)).astype(np.float32)

    dc_clean = DecomposedConv2d.from_pytorch_conv2d(conv, decomp=_decomp(16), error_model=None)
    dc_noisy = DecomposedConv2d.from_pytorch_conv2d(
        conv, decomp=_decomp(16), error_model=GaussianErrorModel(noise_std=0.2)
    )

    rng = np.random.default_rng(99)
    out_clean = dc_clean.forward(x)
    out_noisy = dc_noisy.forward(x, rng=rng)

    assert not np.allclose(out_clean, out_noisy), "Noisy output should differ from clean"


# ---------------------------------------------------------------------------
# DecomposedNetwork (general: CNN + MLP)
# ---------------------------------------------------------------------------


def test_decomposed_network_linear_only() -> None:
    """DecomposedNetwork on an all-linear model matches legacy DecomposedMLP behavior."""
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    x = np.random.default_rng(0).standard_normal((10, 16)).astype(np.float32)

    net = DecomposedNetwork.from_pytorch_sequential(mlp, decomp=SVDDecomposition(rank=8))
    out = net.forward(x)
    assert out.shape == (10, 4)


def test_decomposed_network_full_cnn_output_shape() -> None:
    model = _build_small_cnn()  # input: (N, 3, 8, 8) → output: (N, 10)
    x = np.random.default_rng(0).standard_normal((4, 3, 8, 8)).astype(np.float32)

    net = DecomposedNetwork.from_pytorch_sequential(model, decomp=_decomp(8))
    out = net.forward(x)
    assert out.shape == (4, 10)


def test_decomposed_network_cnn_matches_pytorch() -> None:
    """At full rank, the full CNN pipeline must match PyTorch numerically."""
    model = _build_small_cnn()
    model.eval()

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((3, 3, 8, 8)).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    with torch.no_grad():
        ref = model(x_t).numpy()  # (3, 10)

    net = DecomposedNetwork.from_pytorch_sequential(model, decomp=SVDDecomposition(rank=None))
    out = net.forward(x_np)  # (3, 10)

    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)


def test_decomposed_network_maxpool_shape() -> None:
    """MaxPool2d passthrough halves the spatial dims correctly."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 5),
    )
    x = np.random.default_rng(0).standard_normal((2, 1, 8, 8)).astype(np.float32)
    net = DecomposedNetwork.from_pytorch_sequential(model, decomp=_decomp(4))
    out = net.forward(x)
    assert out.shape == (2, 5)


def test_decomposed_network_noisy_differs_from_clean() -> None:
    model = _build_small_cnn()
    x = np.random.default_rng(0).standard_normal((4, 3, 8, 8)).astype(np.float32)

    clean = DecomposedNetwork.from_pytorch_sequential(model, decomp=_decomp(8), error_model=None)
    noisy = DecomposedNetwork.from_pytorch_sequential(
        model, decomp=_decomp(8), error_model=GaussianErrorModel(noise_std=0.3)
    )

    rng = np.random.default_rng(0)
    out_clean = clean.forward(x)
    out_noisy = noisy.forward(x, rng=rng)

    assert not np.allclose(out_clean, out_noisy)


def test_decomposed_network_unsupported_module_raises() -> None:
    """Unsupported module types should raise ValueError."""
    model = nn.Sequential(nn.Linear(8, 4), nn.Dropout(0.5))
    with pytest.raises(ValueError, match="Unsupported"):
        DecomposedNetwork.from_pytorch_sequential(model, decomp=_decomp(4))


def test_decomposed_mlp_alias() -> None:
    """DecomposedMLP should remain importable and be the same as DecomposedNetwork."""
    from hardware_matrix_decomp.network import DecomposedMLP

    assert DecomposedMLP is DecomposedNetwork
