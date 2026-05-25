"""Tests for Simulation 2: quantized delay model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.network import MemristorNet
from memristor.quantization import make_quantized_net, quantize_delays


class TestQuantizeDelays:
    def test_no_quantization_is_identity(self) -> None:
        d = torch.tensor([7.5, 15.0, 33.2])
        d_q = quantize_delays(d, d_min=5.0, d_max=50.0)
        assert torch.allclose(d_q, d)

    def test_n_levels_unique_values(self) -> None:
        d = torch.linspace(5.0, 50.0, 1000)
        d_q = quantize_delays(d, 5.0, 50.0, n_levels=16)
        assert len(torch.unique(d_q)) == 16

    def test_n_levels_step_size(self) -> None:
        """With n_levels=10: step=5 ns, bins at 5,10,...,50."""
        d = torch.tensor([7.3, 15.0, 22.8])
        d_q = quantize_delays(d, d_min=5.0, d_max=50.0, n_levels=10)
        assert float(d_q[0]) == pytest.approx(5.0)   # 7.3 → nearest 5ns bin = 5
        assert float(d_q[1]) == pytest.approx(15.0)
        assert float(d_q[2]) == pytest.approx(25.0)  # 22.8 → 25

    def test_tdc_snaps_to_resolution(self) -> None:
        d = torch.tensor([5.7, 11.3, 22.9])
        d_q = quantize_delays(d, 5.0, 50.0, tdc_res=1.0)
        assert float(d_q[0]) == pytest.approx(6.0)
        assert float(d_q[1]) == pytest.approx(11.0)
        assert float(d_q[2]) == pytest.approx(23.0)

    def test_tdc_half_ns(self) -> None:
        d = torch.tensor([5.3, 10.8, 22.4])
        d_q = quantize_delays(d, 5.0, 50.0, tdc_res=0.5)
        assert float(d_q[0]) == pytest.approx(5.5)
        assert float(d_q[1]) == pytest.approx(11.0)
        assert float(d_q[2]) == pytest.approx(22.5)

    def test_result_clamped_to_range(self) -> None:
        d = torch.tensor([4.9, 50.1])
        d_q = quantize_delays(d, 5.0, 50.0, n_levels=32)
        assert float(d_q[0]) >= 5.0
        assert float(d_q[1]) <= 50.0

    def test_both_quantizations_applied(self) -> None:
        """Conductance quantization is applied before TDC snapping."""
        d = torch.tensor([22.3])
        # n_levels=10 → step=5 → snaps to 20.0; tdc=1.0 → 20.0 already on grid
        d_q = quantize_delays(d, 5.0, 50.0, n_levels=10, tdc_res=1.0)
        assert float(d_q[0]) == pytest.approx(20.0)

    def test_n_levels_1_is_no_op(self) -> None:
        d = torch.tensor([12.5, 30.0])
        d_q = quantize_delays(d, 5.0, 50.0, n_levels=1)
        assert torch.allclose(d_q, d)


class TestMakeQuantizedNet:
    def _make_net(self, seed: int = 42) -> MemristorNet:
        torch.manual_seed(seed)
        return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)

    def test_is_deep_copy(self) -> None:
        net = self._make_net()
        q_net = make_quantized_net(net, n_levels=32)
        # Modifying original should not affect quantized copy
        with torch.no_grad():
            net.layers[0].u_pos.fill_(99.0)
        assert not torch.allclose(q_net.layers[0].u_pos, net.layers[0].u_pos)

    def test_quantized_delays_on_grid(self) -> None:
        net = self._make_net()
        q_net = make_quantized_net(net, n_levels=16, tdc_res=0.5)
        for layer in q_net.layers:
            d_pos, d_neg = layer.delays()
            # Every delay should be a multiple of 0.5 within floating-point tolerance
            for d in [d_pos, d_neg]:
                residuals = (d * 2).round() - d * 2  # check multiples of 0.5
                assert float(residuals.detach().abs().max()) < 0.01

    def test_high_levels_preserves_delays_closely(self) -> None:
        """128 levels → 0.35 ns step; quantized delays should be very close."""
        net = self._make_net()
        q_net = make_quantized_net(net, n_levels=128)
        for (orig_l, q_l) in zip(net.layers, q_net.layers, strict=True):
            d_pos_orig, d_neg_orig = orig_l.delays()
            d_pos_q, d_neg_q = q_l.delays()
            step = (orig_l.d_max - orig_l.d_min) / 127
            assert float((d_pos_orig - d_pos_q).detach().abs().max()) <= step + 1e-4
            assert float((d_neg_orig - d_neg_q).detach().abs().max()) <= step + 1e-4

    def test_quantized_net_forward_runs(self) -> None:
        net = self._make_net()
        q_net = make_quantized_net(net, n_levels=32, tdc_res=1.0)
        T_in = q_net.encode_binary(np.array([1.0, 0.0]))
        out = q_net.forward(T_in)
        assert out.shape == (1,)
        assert 0.0 < float(out) < 1.0


class TestQuantizationAccuracy:
    """Integration: high-resolution quantization should preserve XOR accuracy."""

    def test_128_levels_preserves_converged_accuracy(self) -> None:
        """A seed that converges with continuous delays should stay correct at 128 levels."""
        import numpy as np

        from memristor.training import MemristorTrainer

        XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                q_net = make_quantized_net(net, n_levels=128, tdc_res=0.5)
                q_trainer = MemristorTrainer(q_net, eta=0.0)
                assert q_trainer.accuracy(XOR_X, XOR_Y) == 1.0, (
                    f"128-level quantization broke accuracy (seed={seed})"
                )
                return
        pytest.fail("No seed among 0-4 converged; cannot test quantization")

    def test_coarse_quantization_degrades_gracefully(self) -> None:
        """16 levels + 2.0 ns TDC: mean XOR accuracy should still be > 0.5."""
        import numpy as np

        from memristor.training import MemristorTrainer

        XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

        accs = []
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                q_net = make_quantized_net(net, n_levels=16, tdc_res=2.0)
                preds = [int(float(q_net.predict(x)[0]) > 0.5) for x in XOR_X]
                accs.append(sum(p == int(y) for p, y in zip(preds, XOR_Y, strict=True)) / 4)

        if not accs:
            pytest.fail("No seeds converged")
        assert float(np.mean(accs)) > 0.5, "Even coarse quantization should beat chance"
