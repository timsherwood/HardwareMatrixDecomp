"""Generate accuracy figures from the sweep results CSV.

Produces three PNGs in figures/:
  - accuracy_vs_rank.png      : accuracy vs rank, one curve per noise level
  - accuracy_vs_noise.png     : accuracy vs noise_std, one curve per rank
  - heatmap_rank_x_noise.png  : 2-D mean accuracy heatmap

Usage:
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --input data/results.csv --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def plot_accuracy_vs_rank(df: pd.DataFrame, out: Path) -> None:
    grouped = df.groupby(["rank", "noise_std"])["noisy_acc"].agg(["mean", "std"]).reset_index()
    noise_levels = sorted(df["noise_std"].unique())

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for noise_std in noise_levels:
            sub = grouped[grouped["noise_std"] == noise_std].sort_values("rank")
            label = f"σ = {noise_std:g}" if noise_std > 0 else "σ = 0 (noiseless)"
            ax.errorbar(
                sub["rank"],
                sub["mean"],
                yerr=sub["std"],
                label=label,
                marker="o",
                capsize=3,
            )
        ax.set_xlabel("Decomposition rank")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy vs decomposition rank\n(MNIST MLP, per noise level)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  → {out}")


def plot_accuracy_vs_noise(df: pd.DataFrame, out: Path) -> None:
    grouped = df.groupby(["rank", "noise_std"])["noisy_acc"].agg(["mean", "std"]).reset_index()
    ranks = sorted(df["rank"].unique())
    noise_levels = sorted(df["noise_std"].unique())
    # Replace 0 with a small value for log-scale plotting; label separately
    plot_noise = [n if n > 0 else noise_levels[1] * 0.1 for n in noise_levels]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for rank in ranks:
            sub = grouped[grouped["rank"] == rank].sort_values("noise_std")
            ax.errorbar(
                plot_noise,
                sub["mean"].values,
                yerr=sub["std"].values,
                label=f"rank = {rank}",
                marker="o",
                capsize=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Noise std (relative to mean |W|)")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy vs tile noise level\n(MNIST MLP, per decomposition rank)")
        # Mark the σ=0 position with a dashed vertical line
        ax.axvline(plot_noise[0], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(plot_noise[0] * 1.1, ax.get_ylim()[0] + 0.01, "σ=0", fontsize=7, color="grey")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  → {out}")


def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    pivot = df.groupby(["rank", "noise_std"])["noisy_acc"].mean().unstack("noise_std")
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.5,
            vmax=1.0,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Noise std (σ)")
        ax.set_ylabel("Decomposition rank")
        ax.set_title("Mean test accuracy: rank × noise\n(MNIST MLP, averaged over 5 seeds)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Accuracy")
        # Annotate each cell
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val < 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("--input", type=Path, default=Path("data/results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures ...")
    plot_accuracy_vs_rank(df, args.output_dir / "accuracy_vs_rank.png")
    plot_accuracy_vs_noise(df, args.output_dir / "accuracy_vs_noise.png")
    plot_heatmap(df, args.output_dir / "heatmap_rank_x_noise.png")
    print("Done.")


if __name__ == "__main__":
    main()
