"""Simulation: Hardware-in-the-loop training vs standard backprop.

Compares three training methods on XOR:
  1. Backprop (MemristorTrainer) — software baseline
  2. DFA (HILTrainer, method='dfa') — locally-computable gradients
  3. SPSA (HILTrainer, method='spsa') — 2-forward-pass gradient estimation

Reports convergence rate and epochs-to-convergence for each method.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from memristor.hil_training import HILTrainer
from memristor.network import MemristorNet
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

N_SEEDS = 10
MAX_EPOCHS = 4000
CHECK_EVERY = 50  # check accuracy every N epochs


def run_backprop(seed: int) -> int | None:
    """Return epoch of convergence or None if failed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
    trainer = MemristorTrainer(net, eta=0.06)
    for ep in range(CHECK_EVERY, MAX_EPOCHS + 1, CHECK_EVERY):
        trainer.fit(XOR_X, XOR_Y, n_epochs=CHECK_EVERY)
        if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            return ep
    return None


def run_dfa(seed: int) -> int | None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
    trainer = HILTrainer(net, eta=0.08, seed=seed)
    for ep in range(CHECK_EVERY, MAX_EPOCHS + 1, CHECK_EVERY):
        trainer.fit(XOR_X, XOR_Y, n_epochs=CHECK_EVERY, method="dfa")
        if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            return ep
    return None


def run_spsa(seed: int) -> int | None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
    trainer = HILTrainer(net, eta=0.3, seed=seed)
    for ep in range(CHECK_EVERY, MAX_EPOCHS + 1, CHECK_EVERY):
        trainer.fit(XOR_X, XOR_Y, n_epochs=CHECK_EVERY, method="spsa", spsa_epsilon=0.15)
        if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            return ep
    return None


def run_dfa_complementary(seed: int) -> int | None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)
    trainer = HILTrainer(net, eta=0.08, seed=seed)
    for ep in range(CHECK_EVERY, MAX_EPOCHS + 1, CHECK_EVERY):
        trainer.fit(XOR_X, XOR_Y, n_epochs=CHECK_EVERY, method="dfa")
        if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
            return ep
    return None


def summarize(name: str, results: list[int | None]) -> None:
    converged = [r for r in results if r is not None]
    rate = len(converged) / len(results)
    med = int(np.median(converged)) if converged else None
    print(f"  {name:<28s}  converged {len(converged):2d}/{len(results)}  ({rate:.0%})", end="")
    if med is not None:
        print(f"  median epochs = {med}")
    else:
        print("  (no convergence)")


def main() -> None:
    print()
    print("Hardware-in-the-Loop Training Simulation")
    print("=" * 60)
    print(f"Task: XOR (4 samples, 2 hidden neurons, {MAX_EPOCHS} max epochs)")
    print(f"Seeds: {N_SEEDS}")
    print()

    methods = [
        ("Backprop (baseline)", run_backprop),
        ("DFA (standard net)", run_dfa),
        ("DFA (complementary)", run_dfa_complementary),
        ("SPSA (standard net)", run_spsa),
    ]

    for name, fn in methods:
        t0 = time.time()
        results = [fn(seed) for seed in range(N_SEEDS)]
        elapsed = time.time() - t0
        summarize(name, results)
        print(f"    [{elapsed:.1f}s]")

    print()
    print("Key:")
    print("  Backprop  — exact gradient via autograd (not hardware-realizable)")
    print("  DFA       — fixed random feedback matrices, local weight updates")
    print("  SPSA      — 2 forward passes per step, no backprop")
    print()
    print("DFA and SPSA require only local information at each weight, making")
    print("them compatible with hardware-in-the-loop training. The forward")
    print("pass can run on real memristive hardware; only e_out (the output")
    print("error) needs to be broadcast back to drive weight updates.")


if __name__ == "__main__":
    main()
