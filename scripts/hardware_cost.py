"""Estimate hardware cost (tile count, MACs) for each decomposition rank.

For each layer in the CNN we compute:
  - Weight matrix shape after im2col reshape (rows=C*kH*kW, cols=C_out for conv)
  - Number of hardware tiles needed at each rank (ceil(dim/100) tiling)
  - Multiply-accumulate operations per sample at each rank
  - How these compare to the sweep accuracy results

Usage:
    uv run python scripts/hardware_cost.py
    uv run python scripts/hardware_cost.py --results data/cnn_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAX_TILE_DIM = 100


def tiles_needed(rows: int, cols: int) -> int:
    """Number of 100×100 tiles to cover a (rows × cols) matrix."""
    import math
    return math.ceil(rows / MAX_TILE_DIM) * math.ceil(cols / MAX_TILE_DIM)


def svd_tiles(rows: int, cols: int, rank: int) -> int:
    """Tiles for SVD pair: A=(rows,r) and B=(r,cols)."""
    return tiles_needed(rows, rank) + tiles_needed(rank, cols)


def svd_macs(rows: int, cols: int, rank: int, n_vectors: int) -> int:
    """MACs for A @ B applied to n_vectors input rows.

    Stage 1: n_vectors × rows × rank    (but rows is the *inner* dim of x@A)
    Stage 2: n_vectors × rank × cols

    Actually: x has shape (n_vectors, rows), A=(rows, rank), B=(rank, cols)
      Stage 1: n_vectors * rows * rank
      Stage 2: n_vectors * rank * cols
    """
    return n_vectors * rows * rank + n_vectors * rank * cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware cost analysis")
    parser.add_argument("--results", type=Path, default=Path("data/cnn_results.csv"))
    args = parser.parse_args()

    # CNN layer weight matrix shapes (after im2col for conv)
    # Conv: W_col shape = (C_in * kH * kW,  C_out)
    # Linear: W shape = (in_features, out_features)
    layers = [
        {"name": "Conv1  (9→16)",    "rows": 1 * 3 * 3,  "cols": 16,  "n_vec": 14 * 14},
        {"name": "Conv2  (144→32)",  "rows": 16 * 3 * 3, "cols": 32,  "n_vec": 7 * 7},
        {"name": "Linear1 (1568→128)", "rows": 32 * 7 * 7, "cols": 128, "n_vec": 1},
        {"name": "Linear2 (128→10)", "rows": 128,         "cols": 10,  "n_vec": 1},
    ]

    ranks = [4, 8, 16, 32]

    # --- Per-layer cost table ---
    print("=" * 72)
    print("TILE COUNT PER LAYER  (one tile = 100×100 weight-stationary unit)")
    print("=" * 72)
    header = f"{'Layer':<22}" + "".join(f"  r={r:<4}" for r in ranks) + "  full-rank"
    print(header)
    print("-" * 72)
    for L in layers:
        full = tiles_needed(L["rows"], L["cols"])
        counts = [svd_tiles(L["rows"], L["cols"], r) for r in ranks]
        row = f"{L['name']:<22}" + "".join(f"  {c:<6}" for c in counts) + f"  {full}"
        print(row)

    # --- MACs per inference ---
    print()
    print("=" * 72)
    print("MACs PER SAMPLE  (batch=1, single image inference)")
    print("=" * 72)
    print(header)
    print("-" * 72)
    total_by_rank: dict[int, int] = {r: 0 for r in ranks}
    total_full = 0
    for L in layers:
        full_macs = L["n_vec"] * L["rows"] * L["cols"]
        macs = [svd_macs(L["rows"], L["cols"], r, L["n_vec"]) for r in ranks]
        for r, m in zip(ranks, macs, strict=True):
            total_by_rank[r] += m
        total_full += full_macs
        row = f"{L['name']:<22}" + "".join(f"  {m:<6,}" for m in macs) + f"  {full_macs:,}"
        print(row)
    print("-" * 72)
    total_row = (
        f"{'TOTAL':<22}" + "".join(f"  {total_by_rank[r]:<6,}" for r in ranks) + f"  {total_full:,}"
    )
    print(total_row)
    savings = {r: 1 - total_by_rank[r] / total_full for r in ranks}
    savings_row = f"{'vs full rank':<22}" + "".join(f"  -{savings[r]:.0%}  " for r in ranks)
    print(savings_row)

    # --- Accuracy / cost Pareto ---
    if args.results.exists():
        df = pd.read_csv(args.results)
        mean_acc = df.groupby(["rank", "noise_std"])["noisy_acc"].mean()

        print()
        print("=" * 72)
        print("ACCURACY vs TOTAL MACs  (noiseless baseline, rank × noise tradeoff)")
        print("=" * 72)
        print(
            f"{'Rank':<6}  {'MACs/sample':>12}  {'vs full':>8}"
            f"  {'σ=0 acc':>9}  {'σ=0.1 acc':>10}  {'σ=0.2 acc':>10}"
        )
        print("-" * 72)
        for r in ranks:
            macs = total_by_rank[r]
            saving = savings[r]
            try:
                acc0   = mean_acc.loc[(r, 0.0)]
                acc01  = mean_acc.loc[(r, 0.1)]
                acc02  = mean_acc.loc[(r, 0.2)]
                print(
                    f"  {r:<4}  {macs:>12,}  {f'-{saving:.0%}':>8}"
                    f"  {acc0:>9.1%}  {acc01:>10.1%}  {acc02:>10.1%}"
                )
            except KeyError:
                print(f"  {r:<4}  {macs:>12,}  {f'-{saving:.0%}':>8}  (no sweep data)")
        print(f"  {'full':4}  {total_full:>12,}  {'baseline':>8}")
    else:
        print(f"\n(No results CSV found at {args.results} — run run_cnn_sweep.py first)")


if __name__ == "__main__":
    main()
