"""MNIST 8×8 training with hardware-in-the-loop SPSA.

Uses HILTrainer with method='spsa': each mini-batch perturbs all weights
simultaneously with random ±1 signs and measures the loss difference.
No backpropagation is needed — the gradient estimate comes from 3 forward
passes per mini-batch (L0, L+, L-).

Architecture: MemristorNet(n_inputs=64, hidden_sizes=[32], n_outputs=10)
Input encoding: encode_time() — bright pixels fire early

Usage:
    uv run python scripts/train_mnist_hil.py
    uv run python scripts/train_mnist_hil.py --epochs 30 --batch-size 128
    uv run python scripts/train_mnist_hil.py --max-train 6000  # fast mode
    uv run python scripts/train_mnist_hil.py --method dfa
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from memristor.hil_training import HILTrainer
from memristor.mnist import load_mnist_8x8
from memristor.network import MemristorNet


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 8×8 HIL training (SPSA/DFA)")
    parser.add_argument("--method", choices=["spsa", "dfa"], default="spsa")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1, help="SPSA perturbation size")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    print("=" * 62)
    print(f"MEMRISTOR LOG-DELAY NETWORK — MNIST 8×8 ({args.method.upper()})")
    print("=" * 62)
    print(f"Architecture: 64→{args.hidden}→10")
    print(f"eta={args.eta}  batch_size={args.batch_size}  epochs={args.epochs}", end="")
    if args.method == "spsa":
        print(f"  epsilon={args.epsilon}")
    else:
        print()
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
    trainer = HILTrainer(
        net,
        eta=args.eta,
        binary_input=False,
        multiclass=True,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    n_eval = min(1000, len(X_test))
    acc0 = trainer.accuracy(X_test[:n_eval], y_test[:n_eval])
    print(f"Accuracy before training (random):  {acc0:.2%}  (expected ≈10%)")
    print()
    print(f"Training {args.epochs} epochs ...")

    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        loss = trainer.epoch(X_train, y_train, method=args.method, spsa_epsilon=args.epsilon)
        if ep % max(1, args.epochs // 10) == 0 or ep == 1:
            val_acc = trainer.accuracy(X_test[:n_eval], y_test[:n_eval])
            elapsed = time.time() - t_start
            print(f"  epoch {ep:3d}  loss={loss:.3f}  val_acc={val_acc:.2%}  [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"\nTraining time: {elapsed:.1f}s  ({elapsed/args.epochs:.1f}s/epoch)")

    print()
    print("Evaluating on full test set ...", end=" ", flush=True)
    test_acc = trainer.accuracy(X_test, y_test)
    train_acc = trainer.accuracy(X_train[:1000], y_train[:1000])
    print("done")

    print()
    print("=" * 62)
    print(f"Final train accuracy:  {train_acc:.2%}")
    print(f"Final test  accuracy:  {test_acc:.2%}")
    print()

    print("Per-class test accuracy:")
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            c_acc = trainer.accuracy(X_test[mask], y_test[mask])
            bar = "█" * int(c_acc * 20)
            print(f"  digit {c}:  {c_acc:5.1%}  {bar}")

    print()
    threshold = 0.35
    status = "PASS" if test_acc >= threshold else f"FAIL (< {threshold:.0%})"
    print(f"Test accuracy {test_acc:.1%}  threshold={threshold:.0%}  {status}")
    sys.exit(0 if test_acc >= threshold else 1)


if __name__ == "__main__":
    main()
