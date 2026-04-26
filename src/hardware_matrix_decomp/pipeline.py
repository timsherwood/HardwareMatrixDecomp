from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .tile_grid import TileGrid


@dataclass
class LayerPipeline:
    """Decomposed linear layer expressed as an ordered sequence of TileGrids.

    For an SVD decomposition W ≈ A @ B the pipeline is [grid_A, grid_B]:
      x → grid_A (m×r) → intermediate (batch×r) → grid_B (r×n) → output (batch×n)

    Activations flow through the grids sequentially.  Each grid operates as an
    independent bank of HardwareTiles and can be dispatched in parallel via the
    protobuf transport layer (see transport.py) without changing this interface.
    """

    grids: list[TileGrid]

    @classmethod
    def from_weight_matrix(
        cls,
        W: np.ndarray,
        decomp: SVDDecomposition | None = None,
        error_model: GaussianErrorModel | None = None,
        layer_id: str = "",
    ) -> LayerPipeline:
        if decomp is None:
            decomp = SVDDecomposition()
        A, B = decomp.decompose(W)
        grid_A = TileGrid.from_matrix(A, grid_id=f"{layer_id}_A", error_model=error_model)
        grid_B = TileGrid.from_matrix(B, grid_id=f"{layer_id}_B", error_model=error_model)
        return cls(grids=[grid_A, grid_B])

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        for grid in self.grids:
            x = grid.forward(x, rng)
        return x
