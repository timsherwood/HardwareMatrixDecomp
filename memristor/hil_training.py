"""Hardware-in-the-loop (HIL) trainer: backprop-free weight updates.

Two gradient-free update rules that are physically realizable:

Direct Feedback Alignment (DFA)
--------------------------------
Each weight updates using only information measurable at that synapse:

    s_ij  = softmax(-A_ij / tau)      # soft-min weight, from TDC readings
    e_j   = B_j @ e_out               # error signal: fixed random wiring
    p_j   = sigmoid(margin_j / tau_d) # routing prob, from local margin

Output layer (exact, no approximation):
    grad_u_pos_ij = e_j * s_pos_ij * d_pos_ij
    grad_u_neg_ij = -e_j * s_neg_ij * d_neg_ij

Hidden layers (DFA approximation of backprop):
    grad_u_pos_ij = -e_j * p_j * s_pos_ij * d_pos_ij
    grad_u_neg_ij = -e_j * (1-p_j) * s_neg_ij * d_neg_ij

The feedback matrices B_l are fixed random (set once at fabrication).
They map output error (shape n_final_out) to hidden error (shape n_out_l).
Despite the approximation, DFA converges because the network adapts to
use whatever fixed error signal B provides.

Physical realization:
  - e_out: broadcast voltage from output comparison circuit
  - B_j:   fixed resistor divider (set at fab, never changed)
  - s_ij:  exp decay of (A_ij - A_min_j) from TDC, computable locally
  - p_j:   output of the timing comparator (pos vs neg race winner)

SPSA (Simultaneous Perturbation Stochastic Approximation)
----------------------------------------------------------
Estimates the full gradient using exactly 2 forward passes:

    delta_i ~ Bernoulli(±1)   # random perturbation sign per weight
    g_i ≈ (L(theta+eps*delta) - L(theta-eps*delta)) / (2*eps*delta_i)

Requires two hardware programming rounds per update step. Slower per-step
than DFA but provably convergent and completely architecture-agnostic.

Training modes:
  epoch(..., method='dfa')  — batched DFA: one vectorised forward pass over
                              the full dataset, ~10× faster than per-sample
  epoch(..., method='spsa') — per-sample SPSA (2 passes per sample)
  step(x, y)                — single-sample DFA (for online HIL use)
  step_spsa(x, y)           — single-sample SPSA (for online HIL use)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from memristor.network import ComplementaryDelayLayer, MemristorNet


class HILTrainer:
    """Backprop-free trainer using Direct Feedback Alignment or SPSA.

    Parameters
    ----------
    net:
        The MemristorNet to train.
    eta:
        Learning rate.
    feedback_std:
        Standard deviation for the fixed random feedback matrices used in DFA.
        Each B_l has shape (n_out_l, n_final_out) with entries ~ N(0, feedback_std).
    binary_input:
        Encode inputs via encode_binary (True) or encode_time (False).
    multiclass:
        If True, use cross-entropy loss with integer class labels (n_outputs > 1).
        Output error for DFA becomes softmax(logits) - one_hot(y).
    batch_size:
        Mini-batch size for epoch() calls (SPSA and DFA).  Full-batch if None.
    seed:
        RNG seed for reproducible feedback matrices.
    """

    def __init__(
        self,
        net: MemristorNet,
        eta: float = 0.05,
        feedback_std: float = 1.0,
        binary_input: bool = True,
        multiclass: bool = False,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.eta = eta
        self.binary_input = binary_input

        self.multiclass = multiclass
        self.batch_size = batch_size

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        n_final_out = net.layers[-1].n_out
        # One feedback matrix per hidden layer (none for output layer).
        # B_l maps output error (n_final_out,) → hidden error (n_out_l,).
        self.B: list[torch.Tensor] = []
        for layer in net.layers[:-1]:
            B = torch.randn(layer.n_out, n_final_out, generator=rng) * feedback_std
            self.B.append(B)

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        return self.net.encode_binary(x) if self.binary_input else self.net.encode_time(x)

    def _batch_loss(self, T_batch: torch.Tensor, y_t: torch.Tensor) -> float:
        """Compute batch loss (binary cross-entropy or cross-entropy) without grad."""
        with torch.no_grad():
            if self.multiclass:
                logits = self.net.forward_logits_batch(T_batch)  # (B, n_out)
                return float(functional.cross_entropy(logits, y_t.long().squeeze(1)))
            else:
                probs = self.net.forward_batch(T_batch)  # (B, 1)
                return float(functional.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), y_t))

    def _output_error(self, T_batch: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Compute ∂L/∂margin for each output neuron and sample.

        Returns (B, n_out) tensor of per-sample, per-neuron error signals.
        For binary: e = (sigmoid(margin/tau_d) - y) / tau_d
        For multiclass: e = (softmax(margins/tau_d) - one_hot(y)) / tau_d
        """
        with torch.no_grad():
            if self.multiclass:
                logits = self.net.forward_logits_batch(T_batch)  # (B, n_out)
                p = torch.softmax(logits, dim=1)  # (B, n_out)
                y_idx = y_t.long().squeeze(1)  # (B,)
                one_hot = torch.zeros_like(p)
                one_hot.scatter_(1, y_idx.unsqueeze(1), 1.0)
                return (p - one_hot) / self.net.tau_d
            else:
                margins = self.net._forward_margins_batch(T_batch)  # (B, 1)
                p = torch.sigmoid(margins / self.net.tau_d)
                return (p - y_t) / self.net.tau_d

    @staticmethod
    def _soft_weights(A: torch.Tensor, tau: float) -> torch.Tensor:
        """Soft-min weights from arrival times.  A: (n_in, n_out) → (n_in, n_out)."""
        log_s = -A / tau
        log_s = log_s - log_s.max(dim=0, keepdim=True).values
        s = torch.exp(log_s)
        return s / s.sum(dim=0, keepdim=True)

    @staticmethod
    def _soft_weights_batch(A: torch.Tensor, tau: float) -> torch.Tensor:
        """Batched soft-min weights.  A: (B, n_in, n_out) → (B, n_in, n_out)."""
        log_s = -A / tau
        log_s = log_s - log_s.max(dim=1, keepdim=True).values
        s = torch.exp(log_s)
        return s / s.sum(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Layer-level update helpers (single-sample and batched)
    # ------------------------------------------------------------------

    def _apply_update(
        self,
        layer: torch.nn.Module,
        e_j: torch.Tensor,
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor,
        p_j: torch.Tensor | None,
        is_last: bool,
    ) -> None:
        """Single-sample local gradient step for one layer.

        For the output layer (is_last=True) the formula is exact backprop.
        For hidden layers e_j is the DFA approximation of ∂L/∂T_h.
        All (n_in, n_out) except e_j (n_out,) and p_j (n_out,).
        """
        if is_last:
            # Exact: grad_u_pos = e_j * s_pos * d_pos
            if isinstance(layer, ComplementaryDelayLayer):
                grad_u = e_j.unsqueeze(0) * d_pos * (s_pos + s_neg)
                layer.u.data -= self.eta * grad_u
            else:
                layer.u_pos.data -= self.eta * e_j.unsqueeze(0) * s_pos * d_pos
                layer.u_neg.data -= self.eta * (-e_j.unsqueeze(0) * s_neg * d_neg)
        else:
            assert p_j is not None
            # DFA: e_j ≈ ∂L/∂T_h; ∂T_h/∂T_plus ≈ p_j
            if isinstance(layer, ComplementaryDelayLayer):
                grad_u = e_j.unsqueeze(0) * d_pos * (
                    (1.0 - p_j).unsqueeze(0) * s_neg - p_j.unsqueeze(0) * s_pos
                )
                layer.u.data -= self.eta * grad_u
            else:
                layer.u_pos.data -= self.eta * (
                    -e_j.unsqueeze(0) * p_j.unsqueeze(0) * s_pos * d_pos
                )
                layer.u_neg.data -= self.eta * (
                    -e_j.unsqueeze(0) * (1.0 - p_j).unsqueeze(0) * s_neg * d_neg
                )

    def _apply_update_batch(
        self,
        layer: torch.nn.Module,
        e_j_batch: torch.Tensor,
        s_pos_batch: torch.Tensor,
        s_neg_batch: torch.Tensor,
        d_pos: torch.Tensor,
        d_neg: torch.Tensor,
        p_j_batch: torch.Tensor | None,
        is_last: bool,
    ) -> None:
        """Batched local gradient step: gradient averaged over the batch.

        e_j_batch: (B, n_out), s_*_batch: (B, n_in, n_out),
        d_pos/d_neg: (n_in, n_out), p_j_batch: (B, n_out) or None.
        """
        e_bc = e_j_batch.unsqueeze(1)  # (B, 1, n_out)
        d_pos_bc = d_pos.unsqueeze(0)  # (1, n_in, n_out)
        d_neg_bc = d_neg.unsqueeze(0)

        if is_last:
            if isinstance(layer, ComplementaryDelayLayer):
                grad_u = (e_bc * d_pos_bc * (s_pos_batch + s_neg_batch)).mean(0)
                layer.u.data -= self.eta * grad_u
            else:
                layer.u_pos.data -= self.eta * (e_bc * s_pos_batch * d_pos_bc).mean(0)
                layer.u_neg.data -= self.eta * (-(e_bc * s_neg_batch * d_neg_bc).mean(0))
        else:
            assert p_j_batch is not None
            p_bc = p_j_batch.unsqueeze(1)  # (B, 1, n_out)
            if isinstance(layer, ComplementaryDelayLayer):
                grad_u = (
                    e_bc * d_pos_bc * ((1.0 - p_bc) * s_neg_batch - p_bc * s_pos_batch)
                ).mean(0)
                layer.u.data -= self.eta * grad_u
            else:
                layer.u_pos.data -= self.eta * (
                    -e_bc * p_bc * s_pos_batch * d_pos_bc
                ).mean(0)
                layer.u_neg.data -= self.eta * (
                    -e_bc * (1.0 - p_bc) * s_neg_batch * d_neg_bc
                ).mean(0)

    # ------------------------------------------------------------------
    # Single-sample steps (for online HIL use: one hardware sample at a time)
    # ------------------------------------------------------------------

    def step(self, x: np.ndarray, y: float) -> float:
        """Single-sample DFA step.  Returns scalar BCE loss.

        Suitable for online hardware-in-the-loop training where one input
        arrives at a time.  For simulation with a full dataset, epoch() is
        faster because it batches all samples in one forward pass.
        """
        T_in = self._encode(x)
        y_t = torch.tensor(np.atleast_1d(np.asarray(y, dtype=np.float32)))

        acts: list[dict[str, torch.Tensor]] = []
        T_current = T_in
        n_layers = len(self.net.layers)

        with torch.no_grad():
            for idx, layer in enumerate(self.net.layers):
                is_last = idx == n_layers - 1
                d_pos, d_neg = layer.delays()
                A_pos = T_current.unsqueeze(1) + d_pos  # (n_in, n_out)
                A_neg = T_current.unsqueeze(1) + d_neg

                T_plus = -self.net.tau * torch.logsumexp(-A_pos / self.net.tau, dim=0)
                T_minus = -self.net.tau * torch.logsumexp(-A_neg / self.net.tau, dim=0)
                margin = T_minus - T_plus

                act: dict[str, torch.Tensor] = {
                    "A_pos": A_pos,
                    "A_neg": A_neg,
                    "d_pos": d_pos,
                    "d_neg": d_neg,
                    "margin": margin,
                }
                if not is_last:
                    p = torch.sigmoid(margin / self.net.tau_d)
                    T_h = p * T_plus + (1.0 - p) * T_minus
                    act["p"] = p
                    T_current = torch.cat([T_h, torch.zeros(1)])
                acts.append(act)

        margin_out = acts[-1]["margin"]
        p_out = torch.sigmoid(margin_out / self.net.tau_d)
        loss = float(functional.binary_cross_entropy(p_out.clamp(1e-7, 1 - 1e-7), y_t))
        e_out = (p_out - y_t) / self.net.tau_d  # (n_final_out,)

        with torch.no_grad():
            for idx, (layer, act) in enumerate(zip(self.net.layers, acts, strict=True)):
                is_last = idx == n_layers - 1
                s_pos = self._soft_weights(act["A_pos"], self.net.tau)
                s_neg = self._soft_weights(act["A_neg"], self.net.tau)

                e_j = e_out if is_last else self.B[idx] @ e_out
                p_j = None if is_last else act["p"]

                self._apply_update(
                    layer, e_j, s_pos, s_neg, act["d_pos"], act["d_neg"], p_j, is_last
                )

        return loss

    def step_spsa(
        self,
        x: np.ndarray,
        y: float,
        epsilon: float = 0.1,
    ) -> float:
        """Single-sample SPSA step using 2 forward passes.  Returns loss at theta.

        Perturbs every u parameter simultaneously with random ±1 signs delta,
        measures L(theta+eps*delta) and L(theta-eps*delta), then applies:

            u_i -= eta * (L+ - L-) / (2*eps) * delta_i
        """
        T_in = self._encode(x)
        y_t = torch.tensor(np.atleast_1d(np.asarray(y, dtype=np.float32)))

        def _loss() -> float:
            with torch.no_grad():
                p = self.net.forward(T_in)
                return float(functional.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), y_t))

        deltas: list[dict[str, torch.Tensor]] = []
        for layer in self.net.layers:
            d: dict[str, torch.Tensor] = {}
            for name, param in layer.named_parameters():
                d[name] = torch.randint(0, 2, param.shape).float() * 2.0 - 1.0
            deltas.append(d)

        L0 = _loss()

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data += epsilon * d[name]
        L_plus = _loss()

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data -= 2.0 * epsilon * d[name]
        L_minus = _loss()

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data += epsilon * d[name]

        grad_scalar = (L_plus - L_minus) / (2.0 * epsilon)
        with torch.no_grad():
            for layer, d in zip(self.net.layers, deltas, strict=True):
                for name, param in layer.named_parameters():
                    param.data -= self.eta * grad_scalar * d[name]

        return L0

    # ------------------------------------------------------------------
    # Batched DFA / SPSA — single mini-batch
    # ------------------------------------------------------------------

    def _step_dfa_batch(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> float:
        """DFA update on one mini-batch. Returns scalar loss."""
        n = len(xs)
        T_batch = torch.stack([self._encode(x) for x in xs])  # (B, n_in+1)
        y_t = torch.tensor(ys[:, None].astype(np.float32))  # (B, 1)

        acts: list[dict[str, torch.Tensor]] = []
        T_current = T_batch
        n_layers = len(self.net.layers)

        with torch.no_grad():
            for idx_l, layer in enumerate(self.net.layers):
                is_last = idx_l == n_layers - 1
                d_pos, d_neg = layer.delays()  # (n_in, n_out)
                A_pos = T_current.unsqueeze(2) + d_pos.unsqueeze(0)  # (B, n_in, n_out)
                A_neg = T_current.unsqueeze(2) + d_neg.unsqueeze(0)

                T_plus = -self.net.tau * torch.logsumexp(-A_pos / self.net.tau, dim=1)
                T_minus = -self.net.tau * torch.logsumexp(-A_neg / self.net.tau, dim=1)
                margin = T_minus - T_plus  # (B, n_out)

                act: dict[str, torch.Tensor] = {
                    "A_pos": A_pos, "A_neg": A_neg,
                    "d_pos": d_pos, "d_neg": d_neg, "margin": margin,
                }
                if not is_last:
                    p = torch.sigmoid(margin / self.net.tau_d)
                    T_h = p * T_plus + (1.0 - p) * T_minus
                    act["p"] = p
                    T_current = torch.cat([T_h, torch.zeros(n, 1)], dim=1)
                acts.append(act)

        loss = self._batch_loss(T_batch, y_t)
        e_out_batch = self._output_error(T_batch, y_t)  # (B, n_final_out)

        with torch.no_grad():
            for idx_l, (layer, act) in enumerate(zip(self.net.layers, acts, strict=True)):
                is_last = idx_l == n_layers - 1
                s_pos = self._soft_weights_batch(act["A_pos"], self.net.tau)
                s_neg = self._soft_weights_batch(act["A_neg"], self.net.tau)

                e_j_batch = e_out_batch if is_last else e_out_batch @ self.B[idx_l].T
                p_j_batch = None if is_last else act["p"]

                self._apply_update_batch(
                    layer, e_j_batch, s_pos, s_neg,
                    act["d_pos"], act["d_neg"], p_j_batch, is_last,
                )
        return loss

    def _step_spsa_batch(
        self, xs: np.ndarray, ys: np.ndarray, epsilon: float
    ) -> float:
        """SPSA update on one mini-batch using 3 forward passes. Returns L0."""
        T_batch = torch.stack([self._encode(x) for x in xs])
        y_t = torch.tensor(ys[:, None].astype(np.float32))

        deltas: list[dict[str, torch.Tensor]] = []
        for layer in self.net.layers:
            d: dict[str, torch.Tensor] = {}
            for name, param in layer.named_parameters():
                d[name] = torch.randint(0, 2, param.shape).float() * 2.0 - 1.0
            deltas.append(d)

        L0 = self._batch_loss(T_batch, y_t)

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data += epsilon * d[name]
        L_plus = self._batch_loss(T_batch, y_t)

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data -= 2.0 * epsilon * d[name]
        L_minus = self._batch_loss(T_batch, y_t)

        for layer, d in zip(self.net.layers, deltas, strict=True):
            for name, param in layer.named_parameters():
                param.data += epsilon * d[name]

        grad_scalar = (L_plus - L_minus) / (2.0 * epsilon)
        with torch.no_grad():
            for layer, d in zip(self.net.layers, deltas, strict=True):
                for name, param in layer.named_parameters():
                    param.data -= self.eta * grad_scalar * d[name]
        return L0

    # ------------------------------------------------------------------
    # Epoch / fit helpers
    # ------------------------------------------------------------------

    def epoch(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        method: str = "spsa",
        shuffle: bool = True,
        spsa_epsilon: float = 0.1,
    ) -> float:
        """One pass over the dataset, optionally in mini-batches."""
        n = len(xs)
        bs = self.batch_size if self.batch_size is not None else n
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        total, count = 0.0, 0
        for start in range(0, n, bs):
            bi = idx[start : start + bs]
            if method == "dfa":
                total += self._step_dfa_batch(xs[bi], ys[bi])
            else:
                total += self._step_spsa_batch(xs[bi], ys[bi], spsa_epsilon)
            count += 1
        return total / count

    def fit(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        n_epochs: int = 1000,
        method: str = "dfa",
        verbose: bool = False,
        print_every: int = 100,
        spsa_epsilon: float = 0.1,
    ) -> list[float]:
        """Train for n_epochs. method: 'dfa' or 'spsa'."""
        losses = []
        for ep in range(1, n_epochs + 1):
            loss = self.epoch(xs, ys, method=method, spsa_epsilon=spsa_epsilon)
            losses.append(loss)
            if verbose and ep % print_every == 0:
                acc = self.accuracy(xs, ys)
                print(f"  epoch {ep:5d}  loss={loss:.4f}  acc={acc:.2%}")
        return losses

    def accuracy(self, xs: np.ndarray, ys: np.ndarray, batch_size: int = 256) -> float:
        """Classification accuracy over the dataset (batched for speed)."""
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
                    probs = self.net.forward_batch(T_batch).numpy()
                    preds = (probs > 0.5).astype(int)
                    targets = (yb.astype(np.float32)[:, None] > 0.5).astype(int)
                    correct += int(
                        np.sum([np.array_equal(p, t) for p, t in zip(preds, targets, strict=True)])
                    )
        return correct / len(xs)
