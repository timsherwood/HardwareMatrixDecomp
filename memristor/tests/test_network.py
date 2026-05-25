"""Tests for DelayLayer and MemristorNet."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.network import DelayLayer, MemristorNet


class TestDelayLayer:
    def test_delays_shape(self) -> None:
        layer = DelayLayer(3, 2)
        d_pos, d_neg = layer.delays()
        assert d_pos.shape == (3, 2)
        assert d_neg.shape == (3, 2)

    def test_delays_within_bounds(self) -> None:
        layer = DelayLayer(4, 3, d_min=5.0, d_max=50.0)
        d_pos, d_neg = layer.delays()
        assert float(d_pos.detach().min()) >= 5.0
        assert float(d_pos.detach().max()) <= 50.0
        assert float(d_neg.detach().min()) >= 5.0
        assert float(d_neg.detach().max()) <= 50.0

    def test_n_signed_branches(self) -> None:
        layer = DelayLayer(3, 2)
        assert layer.n_signed_branches == 6

    def test_u_pos_neg_are_independent(self) -> None:
        torch.manual_seed(0)
        layer = DelayLayer(3, 2)
        assert not torch.allclose(layer.u_pos, layer.u_neg)

    def test_delays_differentiable(self) -> None:
        layer = DelayLayer(2, 2)
        d_pos, d_neg = layer.delays()
        loss = d_pos.sum() + d_neg.sum()
        loss.backward()
        assert layer.u_pos.grad is not None
        assert layer.u_neg.grad is not None


class TestMemristorNet:
    def make_xor_net(self, seed: int = 0) -> MemristorNet:
        torch.manual_seed(seed)
        return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)

    def test_xor_architecture_sizes(self) -> None:
        """XOR: 18 individual delay cells per sign side (spec Section 13).

        Spec counts each (input, neuron, sign) as one branch:
          hidden: 3*2*2 = 12,  output: 3*1*2 = 6  → 18 total.
        n_delay_cells = 2 * n_signed_branch_pairs = 18.
        """
        net = self.make_xor_net()
        # 9 signed pairs × 2 (pos + neg per pair) = 18 individual delay cells
        assert net.n_signed_branches == 9
        assert net.n_delay_cells == 18

    def test_layer_shapes(self) -> None:
        net = self.make_xor_net()
        assert net.layers[0].n_in == 3  # 2 inputs + bias
        assert net.layers[0].n_out == 2
        assert net.layers[1].n_in == 3  # 2 hidden + bias
        assert net.layers[1].n_out == 1

    def test_forward_output_shape_single_output(self) -> None:
        net = self.make_xor_net()
        T_in = net.encode_binary(np.array([1, 0]))
        p = net.forward(T_in)
        assert p.shape == (1,)

    def test_forward_output_in_unit_interval(self) -> None:
        net = self.make_xor_net()
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            T_in = net.encode_binary(np.array(x))
            p = net.forward(T_in)
            assert 0.0 < float(p) < 1.0

    def test_encode_binary_active_fires_at_zero(self) -> None:
        net = MemristorNet(n_inputs=2, hidden_sizes=[], n_outputs=1)
        T = net.encode_binary(np.array([1.0, 0.0]))
        assert float(T[0]) == pytest.approx(0.0)
        assert float(T[1]) == pytest.approx(150.0)  # inactive
        assert float(T[2]) == pytest.approx(0.0)  # bias always 0

    def test_encode_time_monotone(self) -> None:
        """Brighter pixels (x→1) should fire earlier (smaller T)."""
        net = MemristorNet(n_inputs=3, hidden_sizes=[], n_outputs=1)
        T_bright = net.encode_time(np.array([0.9, 0.5, 0.1]))
        assert float(T_bright[0]) < float(T_bright[1]) < float(T_bright[2])

    def test_multi_output_architecture(self) -> None:
        torch.manual_seed(1)
        net = MemristorNet(n_inputs=64, hidden_sizes=[16], n_outputs=10)
        x = np.random.default_rng(0).random(64)
        T = net.encode_binary(x)
        p = net.forward(T)
        assert p.shape == (10,)
        assert float(p.min()) > 0.0 and float(p.max()) < 1.0

    def test_predict_returns_numpy(self) -> None:
        net = self.make_xor_net()
        p = net.predict(np.array([1.0, 0.0]))
        assert isinstance(p, np.ndarray)

    def test_delay_summary(self) -> None:
        net = self.make_xor_net()
        summary = net.delay_summary()
        assert len(summary) == 2
        assert summary[0]["shape"] == (3, 2)
        assert summary[1]["shape"] == (3, 1)

    def test_backward_reaches_all_params(self) -> None:
        """Gradient should be nonzero for all u_pos and u_neg tensors."""
        torch.manual_seed(5)
        net = self.make_xor_net()
        T_in = net.encode_binary(np.array([1.0, 0.0]))
        p = net.forward(T_in)
        loss = p.sum()
        loss.backward()
        for layer in net.layers:
            assert layer.u_pos.grad is not None
            assert layer.u_neg.grad is not None
            # Both positive AND negative branches should get gradient
            assert layer.u_pos.grad.abs().sum() > 0, "u_pos grad is zero"
            assert layer.u_neg.grad.abs().sum() > 0, "u_neg grad is zero"
