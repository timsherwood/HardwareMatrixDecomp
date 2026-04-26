"""End-to-end simulation: decompose a network, measure accuracy under noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .network import DecomposedMLP


@dataclass
class SimulationResult:
    baseline_accuracy: float
    noisy_accuracy: float
    n_samples: int
    noise_std: float
    rank: int | None


def run_simulation(
    model: nn.Sequential,
    x: np.ndarray,
    y_true: np.ndarray,
    decomp: SVDDecomposition | None = None,
    error_model: GaussianErrorModel | None = None,
    seed: int = 42,
) -> SimulationResult:
    """Decompose *model* and compare accuracy with and without noise.

    Parameters
    ----------
    model:
        PyTorch nn.Sequential containing only nn.Linear and nn.ReLU layers.
    x:
        Input activations, shape (N, in_dim), float32.
    y_true:
        Integer class labels, shape (N,).
    decomp:
        SVD decomposition config.  Defaults to SVDDecomposition().
    error_model:
        Noise model applied per tile.  None → noiseless baseline only.
    seed:
        RNG seed for reproducible noise sampling.
    """
    if decomp is None:
        decomp = SVDDecomposition()

    rng = np.random.default_rng(seed)

    baseline_net = DecomposedMLP.from_pytorch_sequential(model, decomp=decomp, error_model=None)
    baseline_logits = baseline_net.forward(x)
    baseline_preds = np.argmax(baseline_logits, axis=1)
    baseline_acc = float(np.mean(baseline_preds == y_true))

    if error_model is None:
        noisy_acc = baseline_acc
    else:
        noisy_net = DecomposedMLP.from_pytorch_sequential(
            model, decomp=decomp, error_model=error_model
        )
        noisy_logits = noisy_net.forward(x, rng=rng)
        noisy_preds = np.argmax(noisy_logits, axis=1)
        noisy_acc = float(np.mean(noisy_preds == y_true))

    return SimulationResult(
        baseline_accuracy=baseline_acc,
        noisy_accuracy=noisy_acc,
        n_samples=len(y_true),
        noise_std=error_model.noise_std if error_model else 0.0,
        rank=decomp.rank,
    )
