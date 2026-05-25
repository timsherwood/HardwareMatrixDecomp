"""Decompose a PyTorch network into hardware-tile pipelines.

Supported layer types:
  - nn.Linear      → DecomposedLinear (two TileGrids via SVD)
  - nn.Conv2d      → DecomposedConv2d (im2col + two TileGrids via SVD)
  - nn.ReLU        → fused into the preceding layer's activation slot
  - nn.MaxPool2d   → PassthroughModule (runs in PyTorch)
  - nn.Flatten     → PassthroughModule (runs in PyTorch)

DecomposedNetwork is the general container; DecomposedMLP is kept as an alias
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .pipeline import LayerPipeline

# ---------------------------------------------------------------------------
# im2col via numpy
# ---------------------------------------------------------------------------


def unfold_input(
    x: np.ndarray,
    kH: int,
    kW: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Convert (N, C, H, W) feature map into im2col matrix.

    Returns shape (N * H_out * W_out, C * kH * kW), matching the column
    layout expected by the weight matrix W of shape (C*kH*kW, C_out).
    """
    N, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    # Build index arrays for gather-based im2col (avoids stride_tricks aliasing issues)
    # i_base: (kH, kW) starting row offsets, then broadcast over output positions
    i0 = np.repeat(np.arange(kH), kW)  # (kH*kW,)
    i1 = stride * np.arange(H_out)  # (H_out,)
    j0 = np.tile(np.arange(kW), kH)  # (kH*kW,)
    j1 = stride * np.arange(W_out)  # (W_out,)

    i = i0[:, None] + i1[None, :]  # (kH*kW, H_out)  — row indices into padded image
    j = j0[:, None] + j1[None, :]  # (kH*kW, W_out)

    # x_pad has shape (N, C, H_pad, W_pad)
    # out[n, c, kk, h, w] = x_pad[n, c, i[kk,h], j[kk,w]]
    cols = x[:, :, i[:, :, None], j[:, None, :]]
    # cols: (N, C, kH*kW, H_out, W_out)

    # Rearrange to (N, H_out, W_out, C*kH*kW) then flatten to (N*H_out*W_out, C*kH*kW)
    cols = cols.transpose(0, 3, 4, 1, 2)  # (N, H_out, W_out, C, kH*kW)
    cols = cols.reshape(N * H_out * W_out, C * kH * kW)
    return np.ascontiguousarray(cols, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Layer containers
# ---------------------------------------------------------------------------


@dataclass
class DecomposedLinear:
    """Hardware-tile replacement for a single nn.Linear + optional activation."""

    pipeline: LayerPipeline
    bias: np.ndarray | None
    activation: str  # "none" | "relu"

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        y = self.pipeline.forward(x, rng)
        if self.bias is not None:
            y = y + self.bias
        if self.activation == "relu":
            y = np.maximum(y, 0.0)
        return y


@dataclass
class DecomposedConv2d:
    """Hardware-tile replacement for a single nn.Conv2d + optional activation.

    Forward pass:
      1. im2col: (N, C, H, W) → (N*H_out*W_out, C*kH*kW)
      2. tile pipeline matmul: → (N*H_out*W_out, C_out)
      3. bias add + activation
      4. reshape: → (N, C_out, H_out, W_out)
    """

    pipeline: LayerPipeline
    bias: np.ndarray | None
    activation: str  # "none" | "relu"
    kernel_size: int
    stride: int
    padding: int
    out_channels: int

    @classmethod
    def from_pytorch_conv2d(
        cls,
        conv: nn.Conv2d,
        decomp: SVDDecomposition | None = None,
        error_model: GaussianErrorModel | None = None,
        activation: str = "none",
        layer_id: str = "",
    ) -> DecomposedConv2d:
        if decomp is None:
            decomp = SVDDecomposition()

        # Conv2d weight: (C_out, C_in, kH, kW) → reshape to (C_in*kH*kW, C_out)
        w = conv.weight.detach().cpu().numpy()
        C_out, C_in, kH, kW = w.shape
        W = w.reshape(C_out, C_in * kH * kW).T  # (C_in*kH*kW, C_out)

        bias = conv.bias.detach().cpu().numpy() if conv.bias is not None else None

        ks = conv.kernel_size
        kernel_size = ks[0] if isinstance(ks, tuple) else int(ks)
        st = conv.stride
        stride = st[0] if isinstance(st, tuple) else int(st)
        pd = conv.padding
        padding = pd[0] if isinstance(pd, tuple) else int(pd)

        pipeline = LayerPipeline.from_weight_matrix(
            W, decomp=decomp, error_model=error_model, layer_id=layer_id or "conv"
        )

        return cls(
            pipeline=pipeline,
            bias=bias,
            activation=activation,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            out_channels=C_out,
        )

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        H_out = (H + 2 * self.padding - kH) // self.stride + 1
        W_out = (W + 2 * self.padding - kW) // self.stride + 1

        col = unfold_input(x, kH=kH, kW=kW, stride=self.stride, padding=self.padding)
        # col: (N*H_out*W_out, C*kH*kW)

        y = self.pipeline.forward(col, rng)  # (N*H_out*W_out, C_out)

        if self.bias is not None:
            y = y + self.bias  # broadcast over batch*spatial

        if self.activation == "relu":
            y = np.maximum(y, 0.0)

        # (N*H_out*W_out, C_out) → (N, H_out, W_out, C_out) → (N, C_out, H_out, W_out)
        y = y.reshape(N, H_out, W_out, self.out_channels)
        return y.transpose(0, 3, 1, 2)


@dataclass
class PassthroughModule:
    """Wraps an nn.Module that doesn't touch tile hardware (pool, flatten, etc.)."""

    module: nn.Module

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(x))
        with torch.no_grad():
            out = self.module(t)
        result: np.ndarray = out.numpy()
        return result


# ---------------------------------------------------------------------------
# Network container
# ---------------------------------------------------------------------------

_Layer = DecomposedLinear | DecomposedConv2d | PassthroughModule


@dataclass
class DecomposedNetwork:
    """General network container supporting Linear, Conv2d, and passthrough modules."""

    layers: list[_Layer]

    @classmethod
    def from_pytorch_sequential(
        cls,
        model: nn.Sequential,
        decomp: SVDDecomposition | None = None,
        error_model: GaussianErrorModel | None = None,
    ) -> DecomposedNetwork:
        if decomp is None:
            decomp = SVDDecomposition()

        layers: list[_Layer] = []

        for i, module in enumerate(model):
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy().T
                bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
                pipeline = LayerPipeline.from_weight_matrix(
                    W, decomp=decomp, error_model=error_model, layer_id=f"linear_{i}"
                )
                layers.append(DecomposedLinear(pipeline, bias=bias, activation="none"))

            elif isinstance(module, nn.Conv2d):
                dc = DecomposedConv2d.from_pytorch_conv2d(
                    module,
                    decomp=decomp,
                    error_model=error_model,
                    activation="none",
                    layer_id=f"conv_{i}",
                )
                layers.append(dc)

            elif isinstance(module, nn.ReLU):
                if layers and isinstance(layers[-1], (DecomposedLinear, DecomposedConv2d)):
                    layers[-1].activation = "relu"
                else:
                    # ReLU with no preceding tile layer — run as passthrough
                    layers.append(PassthroughModule(module))

            elif isinstance(module, (nn.MaxPool2d, nn.Flatten, nn.AdaptiveAvgPool2d)):
                layers.append(PassthroughModule(module))

            else:
                raise ValueError(
                    f"Unsupported module type {type(module).__name__} at position {i}. "
                    "Supported: nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Flatten, "
                    "nn.AdaptiveAvgPool2d."
                )

        return cls(layers=layers)

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, rng)
        return x


# Backward-compatible alias — all existing code using DecomposedMLP continues to work.
DecomposedMLP = DecomposedNetwork

__all__ = [
    "unfold_input",
    "DecomposedLinear",
    "DecomposedConv2d",
    "PassthroughModule",
    "DecomposedNetwork",
    "DecomposedMLP",
]
