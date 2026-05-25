"""Memristive delay network: DelayLayer and MemristorNet.

Architecture overview
---------------------
Each scalar weight is a *signed differential branch pair*: one positive
delay cell (d_pos) and one negative delay cell (d_neg).  The effective
weight is d_minus - d_plus in timing space.

For each neuron, the positive side races the negative side:
    T_plus  = soft-min over positive-branch arrival times   (nLSE)
    T_minus = soft-min over negative-branch arrival times   (nLSE)
    margin  = T_minus - T_plus
    p       = sigmoid(margin / tau_d)

Hidden neurons pass a weighted timing combination to the next layer
so that both positive and negative branches receive gradient:
    T_h = p * T_plus + (1-p) * T_minus

This interpolates between the fast (positive wins) and slow (negative
wins) outcomes, keeping the computation differentiable.

Bias
----
Each layer receives an extra bias input that always fires at T = 0.
It is appended internally; callers supply raw inputs only.

XOR size (spec Section 13)
--------------------------
    MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
    → Layer 1: DelayLayer(3, 2)  (3 = 2 inputs + bias)
    → Layer 2: DelayLayer(3, 1)  (3 = 2 hidden + bias)
    Total signed branches: 12 + 6 = 18  ✓
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DelayLayer(nn.Module):
    """Differential signed delay layer mapping n_in inputs to n_out neurons.

    Stores log-conductance tensors u_pos and u_neg (shape n_in × n_out).
    Delays are d = kappa * exp(-u), clamped to [d_min, d_max].
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kappa: float = 15.81,
        d_min: float = 5.0,
        d_max: float = 50.0,
        init_std: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kappa = kappa
        self.d_min = d_min
        self.d_max = d_max
        # u=0 → d=kappa (geometric midpoint).  Small noise breaks symmetry.
        self.u_pos = nn.Parameter(torch.randn(n_in, n_out) * init_std)
        self.u_neg = nn.Parameter(torch.randn(n_in, n_out) * init_std)

    def delays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (d_pos, d_neg) tensors of shape (n_in, n_out)."""
        d_pos = torch.clamp(self.kappa * torch.exp(-self.u_pos), self.d_min, self.d_max)
        d_neg = torch.clamp(self.kappa * torch.exp(-self.u_neg), self.d_min, self.d_max)
        return d_pos, d_neg

    @property
    def n_signed_branches(self) -> int:
        """Number of signed differential branch pairs in this layer."""
        return self.n_in * self.n_out


class MemristorNet(nn.Module):
    """Multi-layer memristive delay network.

    Parameters
    ----------
    n_inputs:
        Number of raw inputs (bias handled internally).
    hidden_sizes:
        List of hidden neuron counts per hidden layer.
    n_outputs:
        Number of output neurons (= number of classes for softmax,
        or 1 for binary classification).
    tau:
        nLSE temperature (ns).  Controls soft-min sharpness; small tau
        approaches hard min (fastest arrival wins).
    tau_d:
        Decision sigmoid temperature (ns).  Controls boundary sharpness.
    T_inactive:
        Arrival time for inactive (0) binary inputs (ns).  Should be >> d_max
        so inactive inputs don't influence the nLSE race.
    kappa, d_min, d_max:
        Passed to each DelayLayer.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: list[int],
        n_outputs: int,
        tau: float = 10.0,
        tau_d: float = 5.0,
        T_inactive: float = 150.0,
        kappa: float = 15.81,
        d_min: float = 5.0,
        d_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.tau = tau
        self.tau_d = tau_d
        self.T_inactive = T_inactive

        # Input size for each layer: add 1 for the bias node
        in_sizes = [n_inputs + 1] + [h + 1 for h in hidden_sizes]
        out_sizes = hidden_sizes + [n_outputs]

        self.layers = nn.ModuleList(
            [
                DelayLayer(in_s, out_s, kappa, d_min, d_max)
                for in_s, out_s in zip(in_sizes, out_sizes, strict=True)
            ]
        )

    # ------------------------------------------------------------------
    # Input encoding
    # ------------------------------------------------------------------

    def encode_binary(self, x: np.ndarray) -> torch.Tensor:
        """Encode binary input vector as arrival times.

        Active (x > 0.5) → T = 0 ns.  Inactive (x ≤ 0.5) → T = T_inactive.
        Bias node is appended and always fires at T = 0.
        """
        T = torch.full((self.n_inputs + 1,), float(self.T_inactive))
        T[self.n_inputs] = 0.0  # bias always active
        for i in range(self.n_inputs):
            if float(x[i]) > 0.5:
                T[i] = 0.0
        return T

    def encode_time(self, x: np.ndarray, alpha: float = 50.0, eps: float = 0.02) -> torch.Tensor:
        """Encode real-valued input as arrival time via -ln(x) scaling.

        Bright (x→1) → T≈0.  Dark (x→0) → T→T_inactive.
        Used for MNIST pixel encoding (spec Section 28).
        """
        x_clipped = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0)
        raw = -alpha * np.log(x_clipped)
        T = torch.tensor(np.clip(raw, 0.0, self.T_inactive), dtype=torch.float32)
        bias = torch.zeros(1)
        return torch.cat([T, bias])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, T_in: torch.Tensor) -> torch.Tensor:
        """Temporal forward pass (single sample).

        Parameters
        ----------
        T_in:
            Arrival-time vector of shape (n_inputs + 1,) including bias.

        Returns
        -------
        p_out of shape (n_outputs,) — output probabilities via sigmoid.
        """
        return torch.sigmoid(self._forward_margins(T_in) / self.tau_d)

    def forward_logits(self, T_in: torch.Tensor) -> torch.Tensor:
        """Return raw margins (logits) for a single sample.

        Shape: (n_outputs,).  Use with F.cross_entropy for multi-class training.
        """
        return self._forward_margins(T_in) / self.tau_d

    def forward_batch(self, T_in: torch.Tensor) -> torch.Tensor:
        """Batched temporal forward pass.

        Parameters
        ----------
        T_in:
            Arrival-time matrix of shape (B, n_inputs + 1).

        Returns
        -------
        p_out of shape (B, n_outputs) — output probabilities via sigmoid.
        """
        return torch.sigmoid(self._forward_margins_batch(T_in) / self.tau_d)

    def forward_logits_batch(self, T_in: torch.Tensor) -> torch.Tensor:
        """Batched logit forward pass.  Shape: (B, n_outputs).

        Use with F.cross_entropy for multi-class mini-batch training.
        """
        return self._forward_margins_batch(T_in) / self.tau_d

    def _forward_margins(self, T_in: torch.Tensor) -> torch.Tensor:
        """Core single-sample pass; returns raw margins (n_outputs,)."""
        T_current = T_in
        n_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            is_last = idx == n_layers - 1
            d_pos, d_neg = layer.delays()

            # Arrival times for all branches: (n_in, n_out)
            A_pos = T_current.unsqueeze(1) + d_pos
            A_neg = T_current.unsqueeze(1) + d_neg

            # nLSE soft-min over incoming branches for each output neuron
            T_plus = -self.tau * torch.logsumexp(-A_pos / self.tau, dim=0)
            T_minus = -self.tau * torch.logsumexp(-A_neg / self.tau, dim=0)
            margin = T_minus - T_plus

            if is_last:
                return margin

            # Hidden layer: weighted timing output so gradient reaches both sides
            p = torch.sigmoid(margin / self.tau_d)
            T_h = p * T_plus + (1.0 - p) * T_minus
            # Append bias node (always fires at t = 0)
            T_current = torch.cat([T_h, torch.zeros(1)])

        raise RuntimeError("No layers defined")  # pragma: no cover

    def _forward_margins_batch(self, T_in: torch.Tensor) -> torch.Tensor:
        """Core batched pass; returns raw margins (B, n_outputs).

        T_in: (B, n_inputs + 1)
        """
        B = T_in.shape[0]
        T_current = T_in  # (B, n_in)
        n_layers = len(self.layers)

        for idx, layer in enumerate(self.layers):
            is_last = idx == n_layers - 1
            d_pos, d_neg = layer.delays()  # (n_in, n_out)

            # (B, n_in, 1) + (1, n_in, n_out) → (B, n_in, n_out)
            A_pos = T_current.unsqueeze(2) + d_pos.unsqueeze(0)
            A_neg = T_current.unsqueeze(2) + d_neg.unsqueeze(0)

            # nLSE over incoming branches dim=1 → (B, n_out)
            T_plus = -self.tau * torch.logsumexp(-A_pos / self.tau, dim=1)
            T_minus = -self.tau * torch.logsumexp(-A_neg / self.tau, dim=1)
            margin = T_minus - T_plus  # (B, n_out)

            if is_last:
                return margin

            p = torch.sigmoid(margin / self.tau_d)
            T_h = p * T_plus + (1.0 - p) * T_minus  # (B, n_out)
            # Append bias node: (B, n_out+1)
            T_current = torch.cat([T_h, torch.zeros(B, 1)], dim=1)

        raise RuntimeError("No layers defined")  # pragma: no cover

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray, binary_input: bool = True) -> np.ndarray:
        """Return output probabilities for a single input sample."""
        T_in = self.encode_binary(x) if binary_input else self.encode_time(x)
        with torch.no_grad():
            return self.forward(T_in).numpy()

    @property
    def n_signed_branches(self) -> int:
        """Total signed differential branch pairs across all layers."""
        return sum(layer.n_signed_branches for layer in self.layers)

    @property
    def n_delay_cells(self) -> int:
        """Total individual delay cells (pos + neg for each branch)."""
        return 2 * self.n_signed_branches

    def delay_summary(self) -> list[dict[str, object]]:
        """Return per-layer delay statistics for inspection."""
        summary = []
        for i, layer in enumerate(self.layers):
            d_pos, d_neg = layer.delays()
            summary.append(
                {
                    "layer": i,
                    "shape": (layer.n_in, layer.n_out),
                    "d_pos_mean": float(d_pos.mean()),
                    "d_neg_mean": float(d_neg.mean()),
                    "d_pos_range": (float(d_pos.min()), float(d_pos.max())),
                    "d_neg_range": (float(d_neg.min()), float(d_neg.max())),
                }
            )
        return summary
