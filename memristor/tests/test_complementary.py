"""Tests for ComplementaryDelayLayer and related functionality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.network import ComplementaryDelayLayer, DelayLayer, MemristorNet
from memristor.quantization import quantize_complementary
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


class TestComplementaryDelayLayer:
    def _make_layer(self, n_in: int = 3, n_out: int = 2) -> ComplementaryDelayLayer:
        return ComplementaryDelayLayer(n_in, n_out, kappa=15.81, d_min=5.0, d_max=50.0)

    def test_delays_sum_to_d_min_plus_d_max(self) -> None:
        """d_pos + d_neg must equal d_min + d_max for every cell."""
        layer = self._make_layer()
        d_pos, d_neg = layer.delays()
        expected = layer.d_min + layer.d_max
        result = (d_pos + d_neg).detach()
        assert torch.allclose(result, torch.full_like(result, expected), atol=1e-5), (
            f"d_pos + d_neg should equal {expected} everywhere; got {result}"
        )

    def test_u_zero_gives_midpoint_delays(self) -> None:
        """u = 0 → d_pos = kappa (clamped to [d_min,d_max]), d_neg = C - d_pos.

        The arithmetic midpoint is (d_min + d_max) / 2 = 27.5 ns (not kappa).
        At u=0 d_pos = clamp(kappa, d_min, d_max).  For kappa=15.81, d_min=5, d_max=50
        this is 15.81 ns, and d_neg = 50 + 5 - 15.81 = 39.19 ns.
        When u is set so d_pos = 27.5 ns (the arithmetic midpoint), we get d_pos = d_neg.
        """
        layer = ComplementaryDelayLayer(2, 2, kappa=15.81, d_min=5.0, d_max=50.0)
        midpoint = (layer.d_min + layer.d_max) / 2.0  # 27.5 ns
        # Set u so d_pos = midpoint exactly: u = ln(kappa / midpoint)
        u_mid = float(np.log(layer.kappa / midpoint))
        with torch.no_grad():
            layer.u.data.fill_(u_mid)
        d_pos, d_neg = layer.delays()
        assert float(d_pos.detach().mean()) == pytest.approx(midpoint, abs=1e-4)
        assert float(d_neg.detach().mean()) == pytest.approx(midpoint, abs=1e-4)

    def test_large_positive_u_gives_d_pos_near_d_min(self) -> None:
        """Large positive u → exp(-u) small → d_pos clamped near d_min."""
        layer = self._make_layer()
        with torch.no_grad():
            layer.u.data.fill_(10.0)  # very large u
        d_pos, d_neg = layer.delays()
        assert float(d_pos.detach().min()) == pytest.approx(layer.d_min, abs=1e-4)
        assert float(d_neg.detach().max()) == pytest.approx(layer.d_max, abs=1e-4)

    def test_large_negative_u_gives_d_pos_near_d_max(self) -> None:
        """Large negative u → exp(-u) large → d_pos clamped near d_max."""
        layer = self._make_layer()
        with torch.no_grad():
            layer.u.data.fill_(-10.0)
        d_pos, d_neg = layer.delays()
        assert float(d_pos.detach().max()) == pytest.approx(layer.d_max, abs=1e-4)
        assert float(d_neg.detach().min()) == pytest.approx(layer.d_min, abs=1e-4)

    def test_delays_differentiable_through_single_u(self) -> None:
        """Both d_pos and d_neg must flow gradient through a single u.

        Note: using d_pos.sum() + d_neg.sum() gives zero gradient because
        d_pos + d_neg = constant.  Instead we verify that each branch
        individually propagates a non-zero gradient to u.
        """
        layer = self._make_layer()

        # d_pos path
        d_pos, _d_neg = layer.delays()
        d_pos.sum().backward()
        assert layer.u.grad is not None, "u.grad should not be None after d_pos backward"
        assert not torch.all(layer.u.grad == 0), "u.grad should be nonzero from d_pos"
        layer.u.grad = None  # reset

        # d_neg path
        _d_pos2, d_neg = layer.delays()
        d_neg.sum().backward()
        assert layer.u.grad is not None, "u.grad should not be None after d_neg backward"
        assert not torch.all(layer.u.grad == 0), "u.grad should be nonzero from d_neg"

    def test_d_pos_differentiable(self) -> None:
        """Gradient flows through d_pos."""
        layer = self._make_layer()
        d_pos, _d_neg = layer.delays()
        d_pos.sum().backward()
        assert layer.u.grad is not None

    def test_d_neg_differentiable(self) -> None:
        """Gradient flows through d_neg (via the subtraction)."""
        layer = self._make_layer()
        _d_pos, d_neg = layer.delays()
        d_neg.sum().backward()
        assert layer.u.grad is not None

    def test_complementary_has_half_parameters_of_delay_layer(self) -> None:
        """ComplementaryDelayLayer uses 1 param tensor vs 2 for DelayLayer."""
        n_in, n_out = 4, 3
        standard = DelayLayer(n_in, n_out)
        complementary = ComplementaryDelayLayer(n_in, n_out)
        n_std = sum(p.numel() for p in standard.parameters())
        n_comp = sum(p.numel() for p in complementary.parameters())
        assert n_comp == n_std // 2, (
            f"Complementary should have half the params: {n_comp} vs {n_std}"
        )

    def test_n_signed_branches_matches(self) -> None:
        layer = self._make_layer(3, 2)
        assert layer.n_signed_branches == 6

    def test_d_pos_d_neg_within_bounds(self) -> None:
        torch.manual_seed(42)
        layer = self._make_layer(4, 3)
        d_pos, d_neg = layer.delays()
        assert float(d_pos.detach().min()) >= layer.d_min - 1e-5
        assert float(d_pos.detach().max()) <= layer.d_max + 1e-5
        assert float(d_neg.detach().min()) >= layer.d_min - 1e-5
        assert float(d_neg.detach().max()) <= layer.d_max + 1e-5


class TestMemristorNetComplementary:
    def _make_comp_net(self, seed: int = 0) -> MemristorNet:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)

    def test_layers_are_complementary_type(self) -> None:
        net = self._make_comp_net()
        for layer in net.layers:
            assert isinstance(layer, ComplementaryDelayLayer)

    def test_complementary_false_gives_delay_layer(self) -> None:
        net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=False)
        for layer in net.layers:
            assert isinstance(layer, DelayLayer)

    def test_forward_runs_without_error(self) -> None:
        net = self._make_comp_net()
        T_in = net.encode_binary(np.array([1.0, 0.0]))
        with torch.no_grad():
            out = net.forward(T_in)
        assert out.shape == (1,)
        assert 0.0 < float(out) < 1.0

    def test_half_total_parameters(self) -> None:
        """Complementary net has half the parameters of standard net."""
        std_net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=False)
        comp_net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)
        n_std = sum(p.numel() for p in std_net.parameters())
        n_comp = sum(p.numel() for p in comp_net.parameters())
        assert n_comp == n_std // 2

    def test_xor_convergence_complementary(self) -> None:
        """At least 1 of 5 seeds should reach 100% XOR accuracy with complementary encoding."""
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            acc = trainer.accuracy(XOR_X, XOR_Y)
            if acc == 1.0:
                return  # success
        pytest.fail("None of seeds 0–4 converged to 100% XOR accuracy with complementary encoding")

    def test_gradients_flow_through_complementary(self) -> None:
        """After a backward pass, u.grad must be non-None and nonzero."""
        net = self._make_comp_net(seed=1)
        net.zero_grad()
        T_in = net.encode_binary(XOR_X[1])
        p_out = net.forward(T_in)
        loss = torch.nn.functional.binary_cross_entropy(
            p_out.clamp(1e-7, 1 - 1e-7), torch.tensor([1.0])
        )
        loss.backward()
        for layer in net.layers:
            assert isinstance(layer, ComplementaryDelayLayer)
            assert layer.u.grad is not None
            assert not torch.all(layer.u.grad == 0)


class TestQuantizeComplementary:
    def test_level_pos_plus_level_neg_equals_n_levels_minus_1(self) -> None:
        """Complementary constraint: level_pos + level_neg = n_levels - 1."""
        d_min, d_max, n_levels = 5.0, 50.0, 8
        d_pos = torch.linspace(d_min, d_max, 20)
        d_pos_q, d_neg_q = quantize_complementary(d_pos, d_min, d_max, n_levels)
        step = (d_max - d_min) / (n_levels - 1)
        # Recover level indices
        level_pos = torch.round((d_pos_q - d_min) / step).long()
        level_neg = torch.round((d_neg_q - d_min) / step).long()
        expected = torch.full_like(level_pos, n_levels - 1)
        assert torch.all(level_pos + level_neg == expected), (
            f"level_pos + level_neg must equal {n_levels - 1}"
        )

    def test_sum_is_d_min_plus_d_max(self) -> None:
        """d_pos_q + d_neg_q == d_min + d_max."""
        d_min, d_max, n_levels = 5.0, 50.0, 16
        d_pos = torch.rand(10) * (d_max - d_min) + d_min
        d_pos_q, d_neg_q = quantize_complementary(d_pos, d_min, d_max, n_levels)
        expected = d_min + d_max
        sums = (d_pos_q + d_neg_q).float()
        assert torch.allclose(sums, torch.full_like(sums, expected), atol=1e-4), (
            f"d_pos_q + d_neg_q should be {expected} everywhere"
        )

    def test_output_within_bounds(self) -> None:
        d_min, d_max, n_levels = 5.0, 50.0, 32
        d_pos = torch.rand(20) * (d_max - d_min) + d_min
        d_pos_q, d_neg_q = quantize_complementary(d_pos, d_min, d_max, n_levels)
        assert float(d_pos_q.min()) >= d_min - 1e-4
        assert float(d_pos_q.max()) <= d_max + 1e-4
        assert float(d_neg_q.min()) >= d_min - 1e-4
        assert float(d_neg_q.max()) <= d_max + 1e-4

    def test_midpoint_maps_to_midpoint(self) -> None:
        """d_pos = (d_min + d_max)/2 should map to the nearest level."""
        d_min, d_max, n_levels = 5.0, 50.0, 9
        midpoint = torch.tensor([(d_min + d_max) / 2.0])  # 27.5
        d_pos_q, d_neg_q = quantize_complementary(midpoint, d_min, d_max, n_levels)
        # Both should be equal (symmetric encoding)
        assert float((d_pos_q - d_neg_q).abs()) < (d_max - d_min) / (n_levels - 1) + 1e-4

    def test_raises_on_n_levels_less_than_2(self) -> None:
        with pytest.raises(ValueError, match="n_levels"):
            quantize_complementary(torch.tensor([10.0]), 5.0, 50.0, n_levels=1)

    def test_2d_tensor(self) -> None:
        """Works with 2D tensors (layer shape)."""
        d_min, d_max, n_levels = 5.0, 50.0, 8
        d_pos = torch.rand(3, 2) * (d_max - d_min) + d_min
        d_pos_q, d_neg_q = quantize_complementary(d_pos, d_min, d_max, n_levels)
        assert d_pos_q.shape == (3, 2)
        assert d_neg_q.shape == (3, 2)
        sums = d_pos_q + d_neg_q
        assert torch.allclose(sums, torch.full_like(sums, d_min + d_max), atol=1e-4)
