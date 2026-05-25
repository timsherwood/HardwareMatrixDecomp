"""Run rank × noise_std sweep on the trained MLP, output results to CSV.

Usage:
    uv run python scripts/run_sweep.py
    uv run python scripts/run_sweep.py --model data/mlp_mnist.pt --n-test 1000
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

from hardware_matrix_decomp.analysis import SweepConfig, SweepResult, run_sweep


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


def load_test_data(n: int, data_dir: Path = Path("data")) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=n, shuffle=False)
    x, y = next(iter(loader))
    x_flat = x.reshape(x.size(0), -1).numpy().astype(np.float32)
    return x_flat, y.numpy()


def results_to_dataframe(results: list[SweepResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": r.rank,
                "noise_std": r.noise_std,
                "seed": r.seed,
                "baseline_acc": r.baseline_acc,
                "noisy_acc": r.noisy_acc,
            }
            for r in results
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank × noise sweep on trained MLP")
    parser.add_argument("--model", type=Path, default=Path("data/mlp_mnist.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/results.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-test", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)

    print(f"Loading {args.n_test} MNIST test samples ...")
    x, y = load_test_data(args.n_test, args.data_dir)

    config = SweepConfig(
        ranks=[8, 16, 32, 64, 100],
        noise_stds=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        seeds=list(range(5)),
    )

    total = len(config.ranks) * len(config.noise_stds) * len(config.seeds)
    print(
        f"Running {total} sweep combinations ({len(config.ranks)} ranks × "
        f"{len(config.noise_stds)} noise levels × {len(config.seeds)} seeds) ..."
    )

    results = run_sweep(model, x, y, config)

    df = results_to_dataframe(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows → {args.output}")

    # Quick summary
    summary = df.groupby(["rank", "noise_std"])["noisy_acc"].mean().unstack("noise_std").round(3)
    print("\nMean noisy accuracy (rank × noise_std):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
