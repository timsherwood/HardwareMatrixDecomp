"""Run rank × noise_std sweep on the trained CNN, output results to CSV.

The CNN is decomposed with DecomposedNetwork which handles Conv2d layers via
im2col, then runs each conv weight matrix through the same SVD tile pipeline
as the linear layers.

Usage:
    uv run python scripts/run_cnn_sweep.py
    uv run python scripts/run_cnn_sweep.py --model data/cnn_mnist.pt --n-test 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hardware_matrix_decomp.decomp import SVDDecomposition
from hardware_matrix_decomp.error_model import GaussianErrorModel
from hardware_matrix_decomp.network import DecomposedNetwork


def load_model(path: Path) -> nn.Sequential:
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_test_data(n: int, data_dir: Path = Path("data")) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    test_ds = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=n, shuffle=False)
    x, y = next(iter(loader))
    # Keep as 4-D (N, C, H, W) for CNN — do NOT flatten
    return x.numpy().astype(np.float32), y.numpy()


def sweep_cnn(
    model: nn.Sequential,
    x: np.ndarray,
    y: np.ndarray,
    ranks: list[int],
    noise_stds: list[float],
    seeds: list[int],
) -> list[dict[str, float]]:
    rows = []
    total = len(ranks) * len(noise_stds) * len(seeds)
    done = 0
    for rank in ranks:
        decomp = SVDDecomposition(rank=rank)
        baseline_net = DecomposedNetwork.from_pytorch_sequential(model, decomp=decomp)
        baseline_logits = baseline_net.forward(x)
        baseline_acc = float(np.mean(np.argmax(baseline_logits, axis=1) == y))

        for noise_std in noise_stds:
            error_model = GaussianErrorModel(noise_std=noise_std) if noise_std > 0.0 else None
            for seed in seeds:
                rng = np.random.default_rng(seed)
                if error_model is None:
                    noisy_acc = baseline_acc
                else:
                    noisy_net = DecomposedNetwork.from_pytorch_sequential(
                        model, decomp=decomp, error_model=error_model
                    )
                    noisy_logits = noisy_net.forward(x, rng=rng)
                    noisy_acc = float(np.mean(np.argmax(noisy_logits, axis=1) == y))

                rows.append(
                    {
                        "rank": rank,
                        "noise_std": noise_std,
                        "seed": seed,
                        "baseline_acc": baseline_acc,
                        "noisy_acc": noisy_acc,
                    }
                )
                done += 1
                if done % 10 == 0 or done == total:
                    print(
                        f"  [{done:3d}/{total}] rank={rank:3d} σ={noise_std:.3f} "
                        f"seed={seed}  noisy_acc={noisy_acc:.4f}"
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank × noise sweep on trained CNN")
    parser.add_argument("--model", type=Path, default=Path("data/cnn_fmnist.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/cnn_results.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-test", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)

    print(f"Loading {args.n_test} MNIST test samples ...")
    x, y = load_test_data(args.n_test, args.data_dir)

    ranks = [4, 8, 16, 32]
    noise_stds = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
    seeds = list(range(3))

    total = len(ranks) * len(noise_stds) * len(seeds)
    print(
        f"Running {total} sweep combinations "
        f"({len(ranks)} ranks × {len(noise_stds)} noise levels × {len(seeds)} seeds) ..."
    )

    rows = sweep_cnn(model, x, y, ranks=ranks, noise_stds=noise_stds, seeds=seeds)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows → {args.output}")

    summary = df.groupby(["rank", "noise_std"])["noisy_acc"].mean().unstack("noise_std").round(3)
    print("\nMean noisy accuracy (rank × noise_std):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
