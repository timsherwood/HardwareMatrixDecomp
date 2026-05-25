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
    multiclass:
        If True, use cross-entropy loss with integer class labels (for
        n_outputs > 1 multi-class tasks like MNIST).  If False (default),
        use binary cross-entropy with float targets.
    batch_size:
        Mini-batch size for batch_step().  Only used in fit_batched().
        Has no effect on the single-sample step() method.
    """

    def __init__(
        self,
        net: MemristorNet,
        eta: float = 0.05,
        binary_input: bool = True,
        multiclass: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.net = net
        self.eta = eta
        self.binary_input = binary_input
        self.multiclass = multiclass
        self.batch_size = batch_size

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        return self.net.encode_binary(x) if self.binary_input else self.net.encode_time(x)

    def _sgd_update(self) -> None:
        with torch.no_grad():
            for layer in self.net.layers:
                for _name, param in layer.named_parameters():
                    if param.grad is not None:
                        param.data -= self.eta * param.grad

    # ------------------------------------------------------------------
    # Core step (single sample, used for XOR / binary tasks)
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, y: float | np.ndarray) -> float:
        """One (x, y) training step.  Returns scalar loss."""
        T_in = self._encode(x)
        self.net.zero_grad()

        if self.multiclass:
            logits = self.net.forward_logits(T_in).unsqueeze(0)  # (1, n_out)
            y_t = torch.tensor([int(y)], dtype=torch.long)
            loss = functional.cross_entropy(logits, y_t)
        else:
            p_out = self.net.forward(T_in)
            y_t = torch.tensor(np.atleast_1d(np.asarray(y, dtype=np.float32)))
            p_t = p_out.clamp(1e-7, 1.0 - 1e-7)
            loss = functional.binary_cross_entropy(p_t, y_t)

        loss.backward()
        self._sgd_update()
        return float(loss.item())

    def batch_step(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """Mini-batch training step.  Returns mean loss over the batch.

        Uses vectorised forward_logits_batch() — faster than looping step().
        For multiclass: cross-entropy over integer class labels.
        For binary: BCE over float targets.
        """
        T_batch = torch.stack([self._encode(x) for x in xs])
        self.net.zero_grad()

        if self.multiclass:
            logits = self.net.forward_logits_batch(T_batch)  # (B, n_out)
            y_t = torch.tensor(ys.astype(int), dtype=torch.long)
            loss = functional.cross_entropy(logits, y_t)
        else:
            probs = self.net.forward_batch(T_batch)  # (B, n_out)
            y_t = torch.tensor(np.atleast_2d(ys.astype(np.float32)))
            loss = functional.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), y_t)

        loss.backward()
        self._sgd_update()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Epoch / fit helpers
    # ------------------------------------------------------------------

    def epoch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        shuffle: bool = True,
    ) -> float:
        """One pass over the dataset using single-sample steps."""
        idx = np.random.permutation(len(xs)) if shuffle else np.arange(len(xs))
        total = 0.0
        for i in idx:
            total += self.step(xs[i], ys[i])
        return total / len(xs)

    def epoch_batched(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        shuffle: bool = True,
    ) -> float:
        """One pass using mini-batches (faster for larger datasets)."""
        idx = np.random.permutation(len(xs)) if shuffle else np.arange(len(xs))
        total = 0.0
        n_batches = 0
        for start in range(0, len(xs), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            total += self.batch_step(xs[batch_idx], ys[batch_idx])
            n_batches += 1
        return total / n_batches if n_batches else 0.0

    def fit(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_epochs: int = 1000,
        verbose: bool = False,
        print_every: int = 100,
    ) -> list[float]:
        """Train for n_epochs using single-sample steps."""
        losses = []
        for ep in range(1, n_epochs + 1):
            loss = self.epoch(xs, ys)
            losses.append(loss)
            if verbose and ep % print_every == 0:
                acc = self.accuracy(xs, ys)
                print(f"  epoch {ep:5d}  loss={loss:.4f}  acc={acc:.2%}")
        return losses

    def fit_batched(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_epochs: int = 50,
        verbose: bool = False,
        print_every: int = 5,
        val_xs: np.ndarray | None = None,
        val_ys: np.ndarray | None = None,
    ) -> list[float]:
        """Train for n_epochs using mini-batch steps."""
        losses = []
        for ep in range(1, n_epochs + 1):
            loss = self.epoch_batched(xs, ys)
            losses.append(loss)
            if verbose and ep % print_every == 0:
                acc = self.accuracy(xs, ys)
                val_str = ""
                if val_xs is not None and val_ys is not None:
                    val_acc = self.accuracy(val_xs, val_ys)
                    val_str = f"  val_acc={val_acc:.2%}"
                print(f"  epoch {ep:4d}  loss={loss:.4f}  acc={acc:.2%}{val_str}")
        return losses

    def accuracy(self, xs: np.ndarray, ys: np.ndarray, batch_size: int = 256) -> float:
        """Classification accuracy.

        Uses batched forward pass for speed when batch_size > 1.
        """
        if batch_size > 1:
            return self._accuracy_batched(xs, ys, batch_size)
        correct = 0
        for x, y in zip(xs, ys, strict=True):
            T_in = self._encode(x)
            with torch.no_grad():
                if self.multiclass:
                    logits = self.net.forward_logits(T_in)
                    pred = int(logits.argmax().item())
                    correct += int(pred == int(y))
                else:
                    p = self.net.forward(T_in).detach().numpy()
                    pred = (p > 0.5).astype(int)
                    target = (np.atleast_1d(np.asarray(y)) > 0.5).astype(int)
                    correct += int(np.array_equal(pred, target))
        return correct / len(xs)

    def _accuracy_batched(self, xs: np.ndarray, ys: np.ndarray, batch_size: int = 256) -> float:
        correct = 0
        with torch.no_grad():
            for start in range(0, len(xs), batch_size):
                xb = xs[start : start + batch_size]
                yb = ys[start : start + batch_size]
                T_batch = torch.stack([self._encode(x) for x in xb])
                if self.multiclass:
                    logits = self.net.forward_logits_batch(T_batch)
                    preds = logits.argmax(dim=1).numpy()
                    correct += int(np.sum(preds == yb.astype(int)))
                else:
                    probs = self.net.forward_batch(T_batch).detach().numpy()  # (B, n_out)
                    preds = (probs > 0.5).astype(int)  # (B, n_out)
                    # Build targets with matching shape: (B, n_out)
                    y_arr = np.asarray(yb)
                    if y_arr.ndim == 1:
                        y_arr = y_arr[:, np.newaxis]  # (B, 1)
                    targets = (y_arr.astype(np.float32) > 0.5).astype(int)
                    correct += int(
                        np.sum([np.array_equal(p, t) for p, t in zip(preds, targets, strict=True)])
                    )
        return correct / len(xs)

    def compute_branch_targets(self) -> list[dict[str, object]]:
        """Return d_target for each branch after the last update step.

        This is the hardware program-and-verify input: the trainer
        computes u_new, which translates to d_target = kappa * exp(-u_new).
        The physical device is then programmed to this target delay.

        Works with both DelayLayer (u_pos / u_neg) and ComplementaryDelayLayer (u).
        """
        targets = []
        for layer_idx, layer in enumerate(self.net.layers):
            d_pos, d_neg = layer.delays()
            targets.append(
                {
                    "layer": layer_idx,
                    "d_pos_target": d_pos.detach().numpy().copy(),
                    "d_neg_target": d_neg.detach().numpy().copy(),
                }
            )
        return targets
