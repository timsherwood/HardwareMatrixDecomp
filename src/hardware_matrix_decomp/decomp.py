from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tile import MAX_TILE_DIM


@dataclass
class SVDDecomposition:
    """Low-rank SVD factorisation of a weight matrix.

    Decomposes W (m×n) into A (m×r) and B (r×n) with r ≤ MAX_TILE_DIM,
    so each factor's inner dimension fits within a hardware tile.
    """

    rank: int | None = None
    energy_threshold: float = 0.99

    def decompose(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, B) such that W ≈ A @ B with inner dim r ≤ MAX_TILE_DIM."""
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        max_rank = min(len(s), MAX_TILE_DIM)

        if self.rank is not None:
            r = min(self.rank, max_rank)
        else:
            total = float(np.sum(s**2))
            if total == 0.0:
                r = 1
            else:
                cumulative = np.cumsum(s**2) / total
                r = int(np.searchsorted(cumulative, self.energy_threshold, side="left")) + 1
                r = min(r, max_rank)

        A = U[:, :r] * s[:r]  # (m, r)
        B = Vt[:r, :]  # (r, n)
        return A, B

    def reconstruction_error(self, W: np.ndarray) -> float:
        """Relative Frobenius reconstruction error ‖W − AB‖_F / ‖W‖_F."""
        norm_W = float(np.linalg.norm(W, "fro"))
        if norm_W == 0.0:
            return 0.0
        A, B = self.decompose(W)
        return float(np.linalg.norm(W - A @ B, "fro") / norm_W)
