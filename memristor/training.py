"""MemristorTrainer: local log-conductance update rule for temporal networks.

The spec's rule (Section 4):
    u_target = u + eta * lambda * d
    d_target = d * exp(-eta * lambda * d)

where lambda = dL/dd.

This is mathematically identical to SGD on the u parameters because:
    grad_u = dL/du = lambda * (dd/du) = lambda * (-d)
    → u_new = u - eta * grad_u = u + eta * lambda * d  ✓

So the trainer runs standard backprop to get grad_u, then applies a
plain SGD step — no custom optimizer needed.  The "local" character is
that each branch only uses its own grad_u (= -lambda_i * d_i) and eta.

Hardware translation
--------------------
After each training step, d_target for each branch is:
    d_target[i] = kappa * exp(-u_new[i])

The program-and-verify loop (DelayCell.program_and_verify) uses this
target to issue SET/RESET pulses until the measured delay converges.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from memristor.network import MemristorNet


class MemristorTrainer:
    """Trains a MemristorNet using the spec's local update rule.

    Parameters
    ----------
    net:
        The network to train.
    eta:
        Learning rate for the u (log-conductance) parameters.
    binary_input:
        If True, encode inputs via encode_binary; else encode_time.
    """

    def __init__(
        self,
        net: MemristorNet,
        eta: float = 0.05,
        binary_input: bool = True,
    ) -> None:
        self.net = net
        self.eta = eta
        self.binary_input = binary_input

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, y: float | np.ndarray) -> float:
        """One (x, y) training step.  Returns scalar BCE loss."""
        T_in = self.net.encode_binary(x) if self.binary_input else self.net.encode_time(x)
        p_out = self.net.forward(T_in)

        # Binary cross-entropy (works for n_outputs = 1 or > 1)
        y_t = torch.tensor(np.atleast_1d(np.asarray(y, dtype=np.float32)))
        p_t = p_out.clamp(1e-7, 1.0 - 1e-7)
        loss = functional.binary_cross_entropy(p_t, y_t)

        self.net.zero_grad()
        loss.backward()

        # SGD update on u — equivalent to spec's local rule
        with torch.no_grad():
            for layer in self.net.layers:
                for param in (layer.u_pos, layer.u_neg):
                    if param.grad is not None:
                        param.data -= self.eta * param.grad

        return float(loss.item())

    # ------------------------------------------------------------------
    # Epoch / evaluation helpers
    # ------------------------------------------------------------------

    def epoch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        shuffle: bool = True,
    ) -> float:
        """One pass over the dataset.  Returns mean loss."""
        idx = np.random.permutation(len(xs)) if shuffle else np.arange(len(xs))
        total = 0.0
        for i in idx:
            total += self.step(xs[i], ys[i])
        return total / len(xs)

    def fit(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_epochs: int = 1000,
        verbose: bool = False,
        print_every: int = 100,
    ) -> list[float]:
        """Train for n_epochs, returning per-epoch mean loss."""
        losses = []
        for ep in range(1, n_epochs + 1):
            loss = self.epoch(xs, ys)
            losses.append(loss)
            if verbose and ep % print_every == 0:
                acc = self.accuracy(xs, ys)
                print(f"  epoch {ep:5d}  loss={loss:.4f}  acc={acc:.2%}")
        return losses

    def accuracy(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """Classification accuracy (threshold at 0.5 for each output)."""
        correct = 0
        for x, y in zip(xs, ys, strict=True):
            p = self.net.predict(x, binary_input=self.binary_input)
            pred = (p > 0.5).astype(int)
            target = (np.atleast_1d(np.asarray(y)) > 0.5).astype(int)
            if np.array_equal(pred, target):
                correct += 1
        return correct / len(xs)

    def compute_branch_targets(self) -> list[dict[str, object]]:
        """Return d_target for each branch after the last update step.

        This is the hardware program-and-verify input: the trainer
        computes u_new, which translates to d_target = kappa * exp(-u_new).
        The physical device is then programmed to this target delay.
        """
        targets = []
        for layer_idx, layer in enumerate(self.net.layers):
            kappa = layer.kappa
            d_pos = (kappa * torch.exp(-layer.u_pos)).clamp(layer.d_min, layer.d_max)
            d_neg = (kappa * torch.exp(-layer.u_neg)).clamp(layer.d_min, layer.d_max)
            targets.append(
                {
                    "layer": layer_idx,
                    "d_pos_target": d_pos.detach().numpy().copy(),
                    "d_neg_target": d_neg.detach().numpy().copy(),
                }
            )
        return targets
