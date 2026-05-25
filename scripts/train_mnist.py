"""MNIST 8×8 training on the memristor log-delay network.

Architecture: MemristorNet(n_inputs=64, hidden_sizes=[32], n_outputs=10)
  - 65×32 + 33×10 = 2410 signed differential branch pairs
  - 4820 individual delay cells

Input encoding: encode_time()
  - Bright pixels (near 1) fire early (T ≈ 0 ns)
  - Dark pixels (near 0) fire late (T ≈ T_inactive = 150 ns)

Training: mini-batch SGD on log-conductance u, cross-entropy loss.
Equivalent to the spec's local update rule (Section 4).

Usage:
    uv run python scripts/train_mnist.py
    uv run python scripts/train_mnist.py --epochs 30 --batch-size 64 --eta 0.02
    uv run python scripts/train_mnist.py --max-train 6000 --max-test 1000  # fast mode
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from memristor.mnist import load_mnist_8x8
from memristor.network import MemristorNet
from memristor.training import MemristorTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 8×8 on memristor delay network")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=32, help="Hidden neurons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None, help="Subsample training set")
    parser.add_argument("--max-test", type=int, default=None, help="Subsample test set")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("MEMRISTOR LOG-DELAY NETWORK — MNIST 8×8 TRAINING")
    print("=" * 62)
    print(f"Architecture: 64→{args.hidden}→10  (time-encoded inputs)")
    print(f"Params: eta={args.eta}  batch_size={args.batch_size}  epochs={args.epochs}")
    if args.max_train:
        print(f"Subsampled: {args.max_train} train / {args.max_test or 'full'} test")
    print()

    print("Loading MNIST 8×8 ...", end=" ", flush=True)
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist_8x8(
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
    )
    print(f"done ({time.time()-t0:.1f}s)  train={len(X_train)}  test={len(X_test)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = MemristorNet(
        n_inputs=64,
        hidden_sizes=[args.hidden],
        n_outputs=10,
        tau=10.0,
        tau_d=5.0,
        T_inactive=150.0,
    )
    trainer = MemristorTrainer(
        net,
        eta=args.eta,
        binary_input=False,
        multiclass=True,
        batch_size=args.batch_size,
    )

    # Quick sanity: random accuracy before training
    n_eval = min(500, len(X_test))
    acc0 = trainer.accuracy(X_test[:n_eval], y_test[:n_eval])
    print(f"Accuracy before training (random):  {acc0:.2%}  (expected ≈10%)")
    print()

    print(f"Training {args.epochs} epochs ...")
    t_start = time.time()
    losses = trainer.fit_batched(
        X_train,
        y_train,
        n_epochs=args.epochs,
        verbose=True,
        print_every=max(1, args.epochs // 10),
        val_xs=X_test[:n_eval],
        val_ys=y_test[:n_eval],
    )
    elapsed = time.time() - t_start
    print(f"\nTraining time: {elapsed:.1f}s  ({elapsed/args.epochs:.1f}s/epoch)")

    # Final evaluation
    print()
    print("Evaluating on test set ...", end=" ", flush=True)
    t_eval = time.time()
    test_acc = trainer.accuracy(X_test, y_test)
    print(f"done ({time.time()-t_eval:.1f}s)")

    n_train_eval = min(1000, len(X_train))
    train_acc = trainer.accuracy(X_train[:n_train_eval], y_train[:n_train_eval])

    print()
    print("=" * 62)
    print(f"Final train accuracy:  {train_acc:.2%}")
    print(f"Final test accuracy:   {test_acc:.2%}")
    print(f"Loss trajectory:  {losses[0]:.3f} → {losses[-1]:.3f}")
    print()

    # Per-class breakdown
    print("Per-class test accuracy:")
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            c_acc = trainer.accuracy(X_test[mask], y_test[mask])
            bar = "█" * int(c_acc * 20)
            print(f"  digit {c}:  {c_acc:5.1%}  {bar}")

    print()
    threshold = 0.40  # 8×8 MNIST, 20 epochs, full dataset; fast mode lower
    status = "PASS" if test_acc >= threshold else f"FAIL (< {threshold:.0%})"
    print(f"Test accuracy {test_acc:.1%}  threshold={threshold:.0%}  {status}")

    sys.exit(0 if test_acc >= threshold else 1)


if __name__ == "__main__":
    main()
