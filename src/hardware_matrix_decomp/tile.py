from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .error_model import GaussianErrorModel

MAX_TILE_DIM = 100


@dataclass
class HardwareTile:
    """Single weight-stationary matrix multiply unit.

    Weights are at most MAX_TILE_DIM × MAX_TILE_DIM.  Computes y = x @ W,
    optionally injecting Gaussian noise into W before each multiply.
    """

    weights: np.ndarray
    tile_id: str = field(default="")
    error_model: GaussianErrorModel | None = field(default=None)

    def __post_init__(self) -> None:
        if self.weights.ndim != 2:
            raise ValueError(f"Tile weights must be 2-D, got shape {self.weights.shape}")
        h, w = self.weights.shape
        if h > MAX_TILE_DIM or w > MAX_TILE_DIM:
            raise ValueError(
                f"Tile weights {self.weights.shape} exceed MAX_TILE_DIM={MAX_TILE_DIM}"
            )

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Compute x @ W, with noise applied to W when an error model is set."""
        w = self.error_model.apply(self.weights, rng) if self.error_model else self.weights
        result: np.ndarray = x @ w
        return result
