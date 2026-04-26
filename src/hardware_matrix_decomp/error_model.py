from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianErrorModel:
    """Gaussian (shot) noise injected into tile weights at inference time.

    σ = noise_std × mean(|W|), re-sampled on every forward call.
    """

    noise_std: float = 0.01

    def apply(self, weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.noise_std == 0.0:
            return weights
        if rng is None:
            rng = np.random.default_rng()
        sigma = self.noise_std * float(np.mean(np.abs(weights)))
        if sigma == 0.0:
            sigma = self.noise_std
        noise = rng.normal(0.0, sigma, weights.shape)
        return weights + noise
