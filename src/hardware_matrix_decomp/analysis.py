"""Analysis utilities: per-layer error tracking and parameter sweep coordination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn

from .decomp import SVDDecomposition
from .error_model import GaussianErrorModel
from .network import DecomposedMLP
from .simulator import run_simulation


def forward_with_taps(
    net: DecomposedMLP,
    x: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Forward pass that captures post-activation output at each layer boundary.

    Returns (final_output, taps) where taps[i] is the activation after layer i.
    """
    taps: list[np.ndarray] = []
    current = x
    for layer in net.layers:
        current = layer.forward(current, rng)
        taps.append(current.copy())
    return current, taps


def per_layer_relative_error(
    clean_net: DecomposedMLP,
    noisy_net: DecomposedMLP,
    x: np.ndarray,
    n_samples: int = 20,
    seed: int = 0,
) -> list[float]:
    """Mean relative L2 error per layer, averaged over n_samples noise draws.

    Compares post-activation taps between clean_net (no noise) and noisy_net,
    computing ‖noisy − clean‖ / ‖clean‖ at each layer boundary.
    """
    _, clean_taps = forward_with_taps(clean_net, x)
    n_layers = len(clean_taps)
    errors_per_layer: list[list[float]] = [[] for _ in range(n_layers)]

    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        _, noisy_taps = forward_with_taps(noisy_net, x, rng)
        for i in range(n_layers):
            denom = float(np.linalg.norm(clean_taps[i]))
            if denom == 0.0:
                errors_per_layer[i].append(0.0)
            else:
                rel_err = float(np.linalg.norm(noisy_taps[i] - clean_taps[i]) / denom)
                errors_per_layer[i].append(rel_err)

    return [float(np.mean(errs)) for errs in errors_per_layer]


@dataclass
class SweepConfig:
    ranks: list[int]
    noise_stds: list[float]
    seeds: list[int]


@dataclass
class SweepResult:
    rank: int
    noise_std: float
    seed: int
    baseline_acc: float
    noisy_acc: float


def run_sweep(
    model: nn.Sequential,
    x: np.ndarray,
    y: np.ndarray,
    config: SweepConfig,
) -> list[SweepResult]:
    """Cross-product sweep over (rank, noise_std, seed), returning one SweepResult per combo.

    noise_std=0.0 produces a noiseless run (error_model=None), so baseline==noisy.
    """
    results: list[SweepResult] = []
    for rank in config.ranks:
        decomp = SVDDecomposition(rank=rank)
        for noise_std in config.noise_stds:
            error_model = GaussianErrorModel(noise_std=noise_std) if noise_std > 0.0 else None
            for seed in config.seeds:
                sim = run_simulation(
                    model=model,
                    x=x,
                    y_true=y,
                    decomp=decomp,
                    error_model=error_model,
                    seed=seed,
                )
                results.append(
                    SweepResult(
                        rank=rank,
                        noise_std=noise_std,
                        seed=seed,
                        baseline_acc=sim.baseline_accuracy,
                        noisy_acc=sim.noisy_accuracy,
                    )
                )
    return results
