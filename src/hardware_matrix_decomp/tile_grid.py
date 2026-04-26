from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .error_model import GaussianErrorModel
from .tile import MAX_TILE_DIM, HardwareTile


@dataclass
class TileGrid:
    """Block-tiled matrix multiply using a 2-D grid of HardwareTiles.

    Handles arbitrary (m×n) matrices by partitioning into ≤100×100 blocks.
    The forward pass accumulates partial sums across row-blocks and
    concatenates results across column-blocks, equivalent to x @ W.

    tiles[col_block][row_block] covers W[ri*100:(ri+1)*100, ci*100:(ci+1)*100].
    """

    tiles: list[list[HardwareTile]]  # [col_block][row_block]
    in_dim: int
    out_dim: int

    @classmethod
    def from_matrix(
        cls,
        W: np.ndarray,
        grid_id: str = "",
        error_model: GaussianErrorModel | None = None,
    ) -> TileGrid:
        m, n = W.shape
        n_row_blocks = max(1, (m + MAX_TILE_DIM - 1) // MAX_TILE_DIM)
        n_col_blocks = max(1, (n + MAX_TILE_DIM - 1) // MAX_TILE_DIM)

        tiles: list[list[HardwareTile]] = []
        for ci in range(n_col_blocks):
            col_start = ci * MAX_TILE_DIM
            col_end = min(col_start + MAX_TILE_DIM, n)
            col_tiles: list[HardwareTile] = []
            for ri in range(n_row_blocks):
                row_start = ri * MAX_TILE_DIM
                row_end = min(row_start + MAX_TILE_DIM, m)
                block = W[row_start:row_end, col_start:col_end]
                tile_id = f"{grid_id}_r{ri}_c{ci}"
                col_tiles.append(HardwareTile(block, tile_id=tile_id, error_model=error_model))
            tiles.append(col_tiles)

        return cls(tiles=tiles, in_dim=m, out_dim=n)

    def forward(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Compute x @ W using tiled multiply-accumulate."""
        col_results: list[np.ndarray] = []
        for col_tiles in self.tiles:
            partial: np.ndarray | None = None
            for ri, tile in enumerate(col_tiles):
                row_start = ri * MAX_TILE_DIM
                row_end = min(row_start + MAX_TILE_DIM, self.in_dim)
                x_block = x[:, row_start:row_end]
                out = tile.forward(x_block, rng)
                partial = out if partial is None else partial + out
            assert partial is not None
            col_results.append(partial)
        return np.concatenate(col_results, axis=1)
