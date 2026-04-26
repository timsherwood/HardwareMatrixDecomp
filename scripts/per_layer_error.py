"""Measure how per-tile Gaussian noise compounds across the layer chain.

For a fixed decomposition rank, computes mean relative L2 error at each
layer boundary under several noise levels, showing error growth with depth.

Usage:
    uv run python scripts/per_layer_error.py
    uv run python scripts/per_layer_error.py --rank 32 --n-samples 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hardware_matrix_decomp.analysis import per_layer_relative_error
from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.network import DecomposedMLP

STYLE = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

NOISE_LEVELS = [0.001, 0.01, 0.05, 0.1, 0.2]
LAYER_LABELS = ["layer 1\n(784→256)", "layer 2\n(256→128)", "layer 3\n(128→10)"]


def load_model(path: Path) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_samples(n: int, data_dir: Path = Path("data")) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=n, shuffle=False)
    x, _ = next(iter(loader))
    return x.reshape(x.size(0), -1).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer error growth analysis")
    parser.add_argument("--model", type=Path, default=Path("data/mlp_mnist.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--n-batch", type=int, default=64, help="Input samples for error estimation"
    )  # noqa: E501
    parser.add_argument("--n-samples", type=int, default=20, help="Noise draws per measurement")
    parser.add_argument("--output", type=Path, default=Path("figures/per_layer_error.png"))
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)
    x = load_samples(args.n_batch, args.data_dir)

    decomp = SVDDecomposition(rank=args.rank)
    clean_net = DecomposedMLP.from_pytorch_sequential(model, decomp=decomp, error_model=None)

    print(f"Computing per-layer errors (rank={args.rank}, {args.n_samples} noise samples each) ...")
    all_errors: dict[float, list[float]] = {}
    for noise_std in NOISE_LEVELS:
        noisy_net = DecomposedMLP.from_pytorch_sequential(
            model, decomp=decomp, error_model=GaussianErrorModel(noise_std=noise_std)
        )
        errors = per_layer_relative_error(clean_net, noisy_net, x, n_samples=args.n_samples, seed=0)
        all_errors[noise_std] = errors
        print(f"  σ={noise_std:5.3f}  layer errors: {[f'{e:.4f}' for e in errors]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(1, len(LAYER_LABELS) + 1)
        for noise_std, errors in all_errors.items():
            ax.plot(x_pos, errors, marker="o", label=f"σ = {noise_std:g}")

        ax.set_yscale("log")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(LAYER_LABELS)
        ax.set_xlabel("Layer (depth)")
        ax.set_ylabel("Mean relative L2 error  ‖noisy − clean‖ / ‖clean‖")
        ax.set_title(
            f"Error growth across tile chain  (rank={args.rank}, MNIST MLP)\n"
            f"averaged over {args.n_samples} noise draws on {args.n_batch} samples"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output)
        plt.close(fig)
    print(f"\n  → {args.output}")


if __name__ == "__main__":
    main()
