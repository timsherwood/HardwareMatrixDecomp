"""Decompose a PyTorch MLP into a hardware-tile pipeline.

Each nn.Linear layer is replaced by a LayerPipeline (two TileGrids connected
through an SVD factorisation).  ReLU activations are fused into the preceding
layer's forward call.  Conv layers are not yet supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .pipeline import LayerPipeline


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
class DecomposedMLP:
    """Full MLP whose linear layers have been replaced by tile pipelines."""

    layers: list[DecomposedLinear]

    @classmethod
    def from_pytorch_sequential(
        cls,
        model: nn.Sequential,
        decomp: SVDDecomposition | None = None,
        error_model: GaussianErrorModel | None = None,
    ) -> DecomposedMLP:
        """Walk an nn.Sequential and replace Linear layers with DecomposedLinear."""
        if decomp is None:
            decomp = SVDDecomposition()

        layers: list[DecomposedLinear] = []
        for i, module in enumerate(model):
            if isinstance(module, nn.Linear):
                # PyTorch weight shape: (out, in) — transpose to (in, out) for x @ W convention.
                W = module.weight.detach().cpu().numpy().T
                bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
                pipeline = LayerPipeline.from_weight_matrix(
                    W, decomp=decomp, error_model=error_model, layer_id=f"layer_{i}"
                )
                layers.append(DecomposedLinear(pipeline, bias=bias, activation="none"))
            elif isinstance(module, nn.ReLU):
                if layers:
                    layers[-1].activation = "relu"
            else:
                raise ValueError(
                    f"Unsupported module type {type(module).__name__} at position {i}. "
                    "Only nn.Linear and nn.ReLU are currently supported."
                )

        return cls(layers=layers)

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, rng)
        return x
