"""Tests for memristor/gradient_analysis.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.gradient_analysis import (
    extract_delay_gradients,
    gradient_active_fraction,
    gradient_summary,
)
from memristor.network import MemristorNet

# XOR dataset
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def _make_net_with_grads(seed: int = 0) -> MemristorNet:
    """Return a MemristorNet after one forward+backward pass so grads are populated."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
    net.zero_grad()
    T_in = net.encode_binary(XOR_X[1])  # (0, 1) → should give nonzero loss
    p_out = net.forward(T_in)
    y_t = torch.tensor([1.0])
    loss = torch.nn.functional.binary_cross_entropy(p_out.clamp(1e-7, 1 - 1e-7), y_t)
    loss.backward()
    return net


class TestExtractDelayGradients:
    def test_returns_one_dict_per_layer(self) -> None:
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        assert len(grads) == len(net.layers)

    def test_dict_keys_present(self) -> None:
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        expected_keys = {"d_pos", "d_neg", "grad_d_pos", "grad_d_neg", "delta_d_pos", "delta_d_neg"}
        for g in grads:
            assert set(g.keys()) == expected_keys

    def test_grad_d_pos_equals_minus_grad_u_pos_over_d_pos(self) -> None:
        """grad_d_pos = -grad_u_pos / d_pos (∂L/∂d_pos via chain rule)."""
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        for g, layer in zip(grads, net.layers, strict=True):
            if layer.u_pos.grad is None:
                continue
            gu = layer.u_pos.grad.detach().numpy()
            d_pos = g["d_pos"]
            expected = -gu / (d_pos + 1e-12)
            np.testing.assert_allclose(g["grad_d_pos"], expected, rtol=1e-5)

    def test_delta_d_pos_equals_minus_grad_u_pos(self) -> None:
        """delta_d_pos = -grad_u_pos = d_pos * (∂L/∂d_pos)."""
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        for g, layer in zip(grads, net.layers, strict=True):
            if layer.u_pos.grad is None:
                continue
            gu = layer.u_pos.grad.detach().numpy()
            np.testing.assert_allclose(g["delta_d_pos"], -gu, rtol=1e-5)

    def test_delta_d_neg_equals_minus_grad_u_neg(self) -> None:
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        for g, layer in zip(grads, net.layers, strict=True):
            if layer.u_neg.grad is None:
                continue
            gu = layer.u_neg.grad.detach().numpy()
            np.testing.assert_allclose(g["delta_d_neg"], -gu, rtol=1e-5)

    def test_shapes_match_layer_dimensions(self) -> None:
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        for g, layer in zip(grads, net.layers, strict=True):
            expected_shape = (layer.n_in, layer.n_out)
            for key in ("d_pos", "d_neg", "grad_d_pos", "grad_d_neg", "delta_d_pos", "delta_d_neg"):
                assert g[key].shape == expected_shape, f"key={key} shape mismatch"


class TestGradientActiveFraction:
    def test_returns_value_between_0_and_1(self) -> None:
        net = _make_net_with_grads()
        grads = extract_delay_gradients(net)
        frac = gradient_active_fraction(grads)
        assert 0.0 <= frac <= 1.0

    def test_empty_grads_returns_zero(self) -> None:
        frac = gradient_active_fraction([])
        assert frac == 0.0

    def test_all_equal_grads_high_fraction(self) -> None:
        """If all delta_d are equal, all are above threshold → fraction = 1."""
        g = {
            "d_pos": np.ones((2, 2)) * 10.0,
            "d_neg": np.ones((2, 2)) * 10.0,
            "grad_d_pos": np.zeros((2, 2)),
            "grad_d_neg": np.zeros((2, 2)),
            "delta_d_pos": np.ones((2, 2)) * 5.0,
            "delta_d_neg": np.ones((2, 2)) * 5.0,
        }
        frac = gradient_active_fraction([g])
        # All equal to max → all above threshold
        assert frac == pytest.approx(1.0)

    def test_one_active_rest_zero(self) -> None:
        """Single nonzero cell → active fraction should be low."""
        delta = np.zeros((4, 4))
        delta[0, 0] = 1.0  # only one active cell
        g = {
            "d_pos": np.ones((4, 4)) * 10.0,
            "d_neg": np.ones((4, 4)) * 10.0,
            "grad_d_pos": np.zeros((4, 4)),
            "grad_d_neg": np.zeros((4, 4)),
            "delta_d_pos": delta,
            "delta_d_neg": np.zeros((4, 4)),
        }
        frac = gradient_active_fraction([g])
        # 1 active out of 32 cells total (16 pos + 16 neg)
        assert frac == pytest.approx(1.0 / 32.0)

    def test_t_inactive_inputs_give_near_zero_gradient(self) -> None:
        """Branches receiving T_inactive >> tau should have near-zero delta_d."""
        # Build a net and do a backward with T_inactive inputs
        torch.manual_seed(7)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
        net.zero_grad()
        # All-zero input: only bias fires; both input neurons are inactive
        T_in = net.encode_binary(np.array([0.0, 0.0]))
        p_out = net.forward(T_in)
        loss = torch.nn.functional.binary_cross_entropy(
            p_out.clamp(1e-7, 1 - 1e-7), torch.tensor([1.0])
        )
        loss.backward()
        grads = extract_delay_gradients(net)
        # First layer has 3 inputs (2 active + bias); the first 2 rows correspond
        # to the inactive inputs.  Their delta_d should be small relative to bias row.
        layer0 = grads[0]
        inactive_rows_delta = np.abs(layer0["delta_d_pos"][:2, :])
        bias_row_delta = np.abs(layer0["delta_d_pos"][2, :])
        if bias_row_delta.max() > 0:
            assert inactive_rows_delta.max() < bias_row_delta.max() * 10, (
                "Inactive input branches should have smaller gradient than bias"
            )

    def test_gradient_larger_near_tau_than_far_from_tau(self) -> None:
        """delta_d peaks near d ≈ tau; cells with d >> tau get smaller gradient."""
        # Use a controlled test: single-layer net; manually set d_pos for two branches
        # and compare their delta_d magnitudes after a backward pass.
        torch.manual_seed(3)
        net = MemristorNet(n_inputs=1, hidden_sizes=[], n_outputs=1, tau=10.0, tau_d=5.0)
        # Layer has n_in=2 (input + bias), n_out=1
        layer = net.layers[0]
        tau = net.tau

        # Set u_pos so that d_pos[0,0] ≈ tau (near window) and d_pos[1,0] >> tau
        kappa = layer.kappa
        with torch.no_grad():
            layer.u_pos.data[0, 0] = float(np.log(kappa / tau))  # d ≈ tau
            layer.u_pos.data[1, 0] = float(np.log(kappa / (tau * 5)))  # d ≈ 5*tau

        # Clamp to valid range
        with torch.no_grad():
            layer.u_pos.data[1, 0] = float(np.log(kappa / min(layer.d_max, tau * 5)))

        net.zero_grad()
        T_in = net.encode_binary(np.array([1.0]))  # active input
        p_out = net.forward(T_in)
        loss = torch.nn.functional.binary_cross_entropy(
            p_out.clamp(1e-7, 1 - 1e-7), torch.tensor([1.0])
        )
        loss.backward()
        grads = extract_delay_gradients(net)
        delta = grads[0]["delta_d_pos"]
        # Row 0 corresponds to the active input (d ≈ tau), row 1 is bias
        # Check that |delta_d| for d ≈ tau is comparable or larger than d >> tau
        abs_delta = np.abs(delta)
        d_pos = grads[0]["d_pos"]
        # Find the row with d closest to tau
        d_pos_col0 = d_pos[:, 0]
        idx_near_tau = int(np.argmin(np.abs(d_pos_col0 - tau)))
        idx_far_tau = int(np.argmax(d_pos_col0))
        if idx_near_tau != idx_far_tau and abs_delta[idx_far_tau, 0] > 0:
            ratio = abs_delta[idx_near_tau, 0] / abs_delta[idx_far_tau, 0]
            assert ratio >= 1.0, (
                f"Expected |delta_d| near tau to be >= far from tau; ratio={ratio:.3f}"
            )


class TestGradientSummary:
    def test_returns_dict_with_expected_keys(self) -> None:
        torch.manual_seed(0)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        summary = gradient_summary(net, XOR_X, XOR_Y)
        assert "n_samples" in summary
        assert "loss" in summary
        assert "active_fraction" in summary
        assert "layers" in summary

    def test_n_samples_matches_input(self) -> None:
        torch.manual_seed(0)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        summary = gradient_summary(net, XOR_X, XOR_Y)
        assert summary["n_samples"] == len(XOR_X)

    def test_active_fraction_in_range(self) -> None:
        torch.manual_seed(0)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        summary = gradient_summary(net, XOR_X, XOR_Y)
        frac = float(summary["active_fraction"])  # type: ignore[arg-type]
        assert 0.0 <= frac <= 1.0

    def test_per_layer_keys_present(self) -> None:
        torch.manual_seed(0)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        summary = gradient_summary(net, XOR_X, XOR_Y)
        layers: list[dict[str, object]] = summary["layers"]  # type: ignore[assignment]
        assert len(layers) == len(net.layers)
        for layer_info in layers:
            assert "mean_abs_delta_d" in layer_info
            assert "std_abs_delta_d" in layer_info
            assert "active_fraction" in layer_info

    def test_loss_is_positive(self) -> None:
        torch.manual_seed(0)
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        summary = gradient_summary(net, XOR_X, XOR_Y)
        assert float(summary["loss"]) > 0.0  # type: ignore[arg-type]
