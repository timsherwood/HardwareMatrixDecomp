"""Delay-space gradient analysis for memristive delay networks.

After a backward pass the u-space gradients (grad_u) are available on each
layer's parameters.  Since d = kappa * exp(-u), the chain rule gives:

    dL/dd  =  dL/du  *  du/dd  =  dL/du  *  (-1/d)

So:  grad_d = -grad_u / d

And the "delay-weighted gradient" (sensitivity per unit delay) is:

    delta_d  =  d * grad_d  =  -grad_u

This quantity peaks near d ≈ tau (the nLSE temperature) and decays
exponentially for d >> tau or T_in >> tau, directly revealing which
branches are in the active training window.
"""

from __future__ import annotations

import numpy as np
import torch

from memristor.network import MemristorNet


def extract_delay_gradients(net: MemristorNet) -> list[dict[str, np.ndarray]]:
    """Extract per-layer delay-space gradients after a backward pass.

    Handles both ``DelayLayer`` (has ``u_pos`` / ``u_neg``) and
    ``ComplementaryDelayLayer`` (has a single ``u``).

    Returns a list of dicts (one per layer) with keys:

    * ``d_pos``       — positive-branch delays (n_in × n_out)
    * ``d_neg``       — negative-branch delays (n_in × n_out)
    * ``grad_d_pos``  — ∂L/∂d_pos = -grad_u_pos / d_pos
    * ``grad_d_neg``  — ∂L/∂d_neg = -grad_u_neg / d_neg
    * ``delta_d_pos`` — d_pos · (∂L/∂d_pos) = -grad_u_pos
    * ``delta_d_neg`` — d_neg · (∂L/∂d_neg) = -grad_u_neg
    """
    result: list[dict[str, np.ndarray]] = []

    for layer in net.layers:
        d_pos, d_neg = layer.delays()
        d_pos_np = d_pos.detach().numpy().copy()
        d_neg_np = d_neg.detach().numpy().copy()

        if hasattr(layer, "u_pos"):
            # Standard DelayLayer: independent u_pos / u_neg
            g_u_pos = layer.u_pos.grad
            g_u_neg = layer.u_neg.grad
            gu_pos_np = g_u_pos.numpy().copy() if g_u_pos is not None else np.zeros_like(d_pos_np)
            gu_neg_np = g_u_neg.numpy().copy() if g_u_neg is not None else np.zeros_like(d_neg_np)
        else:
            # ComplementaryDelayLayer: single u drives both d_pos and d_neg
            # d_pos = clamp(kappa * exp(-u), ...) → grad_u flows into d_pos
            # d_neg = (d_min + d_max) - d_pos    → grad_u for d_neg is -grad_u_pos
            # The net grad on u is grad_u_pos - grad_u_pos_of_neg = grad_u_pos + grad_u_neg_contrib
            # But since we're extracting raw per-branch gradients we reconstruct from .grad on u:
            g_u = layer.u.grad  # type: ignore[attr-defined]
            gu_pos_np = g_u.numpy().copy() if g_u is not None else np.zeros_like(d_pos_np)
            # For complementary: d_neg = C - d_pos, so ∂d_neg/∂u = -∂d_pos/∂u
            # → grad_u for negative branch is effectively the negative of grad_u (same magnitude)
            gu_neg_np = gu_pos_np  # same grad_u accumulates for both via chain rule sign flip

        # grad_d = -grad_u / d  (∂L/∂d)
        eps = 1e-12
        grad_d_pos = -gu_pos_np / (d_pos_np + eps)
        grad_d_neg = -gu_neg_np / (d_neg_np + eps)

        # delta_d = -grad_u  (= d · ∂L/∂d)
        delta_d_pos = -gu_pos_np
        delta_d_neg = -gu_neg_np

        result.append(
            {
                "d_pos": d_pos_np,
                "d_neg": d_neg_np,
                "grad_d_pos": grad_d_pos,
                "grad_d_neg": grad_d_neg,
                "delta_d_pos": delta_d_pos,
                "delta_d_neg": delta_d_neg,
            }
        )

    return result


def gradient_active_fraction(
    grads: list[dict[str, np.ndarray]],
    threshold: float = 0.01,
) -> float:
    """Fraction of delay cells where |delta_d| > threshold * max(|delta_d|).

    ``delta_d = -grad_u`` is the delay-weighted gradient.  Cells with
    d >> tau or inactive T_in contribute near-zero delta_d; this metric
    shows how sparse the gradient support is across all layers.

    Parameters
    ----------
    grads:
        Output of ``extract_delay_gradients``.
    threshold:
        Fraction of global peak |delta_d| used as the activity threshold.

    Returns
    -------
    Active fraction in [0, 1].
    """
    all_delta: list[np.ndarray] = []
    for g in grads:
        all_delta.append(g["delta_d_pos"].ravel())
        all_delta.append(g["delta_d_neg"].ravel())

    if not all_delta:
        return 0.0

    all_vals = np.concatenate(all_delta)
    abs_vals = np.abs(all_vals)
    global_max = float(abs_vals.max())

    if global_max == 0.0:
        return 0.0

    active = abs_vals > threshold * global_max
    return float(active.sum()) / float(active.size)


def gradient_summary(
    net: MemristorNet,
    X_sample: np.ndarray,
    Y_sample: np.ndarray,
) -> dict[str, object]:
    """Run a forward+backward on a batch and return gradient summary statistics.

    Accumulates gradients over all samples in the batch (mean loss) so that
    the extracted gradients reflect the average training signal.

    Parameters
    ----------
    net:
        The ``MemristorNet`` whose gradients to analyse.
    X_sample:
        Input array of shape (N, n_inputs).
    Y_sample:
        Target array of shape (N,) with float labels in [0, 1].

    Returns
    -------
    Dict with keys:
    * ``n_samples``      — batch size used
    * ``loss``           — scalar loss value
    * ``active_fraction``— fraction of active cells (see ``gradient_active_fraction``)
    * ``layers``         — list of per-layer summary dicts with mean/std/active_fraction
    """
    import torch.nn.functional as functional

    net.zero_grad()

    total_loss = torch.tensor(0.0)
    for xi, yi in zip(X_sample, Y_sample, strict=True):
        T_in = net.encode_binary(xi)
        p_out = net.forward(T_in)
        y_t = torch.tensor(np.atleast_1d(np.asarray(yi, dtype=np.float32)))
        p_t = p_out.clamp(1e-7, 1.0 - 1e-7)
        total_loss = total_loss + functional.binary_cross_entropy(p_t, y_t)

    mean_loss = total_loss / len(X_sample)
    mean_loss.backward()

    grads = extract_delay_gradients(net)
    active_frac = gradient_active_fraction(grads)

    layer_summaries: list[dict[str, object]] = []
    for g in grads:
        delta_all = np.concatenate([g["delta_d_pos"].ravel(), g["delta_d_neg"].ravel()])
        abs_delta = np.abs(delta_all)
        abs_grad_d = np.abs(np.concatenate([g["grad_d_pos"].ravel(), g["grad_d_neg"].ravel()]))
        layer_max = float(abs_delta.max()) if abs_delta.size > 0 else 0.0
        threshold = 0.01
        layer_active = (
            float((abs_delta > threshold * layer_max).sum()) / abs_delta.size
            if abs_delta.size > 0
            else 0.0
        )
        layer_summaries.append(
            {
                "mean_abs_delta_d": float(abs_delta.mean()),
                "std_abs_delta_d": float(abs_delta.std()),
                "mean_abs_grad_d": float(abs_grad_d.mean()),
                "std_abs_grad_d": float(abs_grad_d.std()),
                "active_fraction": layer_active,
            }
        )

    return {
        "n_samples": len(X_sample),
        "loss": float(mean_loss.item()),
        "active_fraction": active_frac,
        "layers": layer_summaries,
    }
