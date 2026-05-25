"""Tests for MNIST data pipeline and multi-class network architecture.

These tests do not train to convergence — they verify architecture shapes,
encoding correctness, batched forward/logits, and trainer multi-class mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.network import MemristorNet
from memristor.training import MemristorTrainer


def make_mnist_net(seed: int = 0) -> MemristorNet:
    torch.manual_seed(seed)
    return MemristorNet(n_inputs=64, hidden_sizes=[32], n_outputs=10)


class TestMnistArchitecture:
    def test_layer_shapes(self) -> None:
        net = make_mnist_net()
        assert net.layers[0].n_in == 65   # 64 pixels + bias
        assert net.layers[0].n_out == 32
        assert net.layers[1].n_in == 33   # 32 hidden + bias
        assert net.layers[1].n_out == 10

    def test_n_signed_branches(self) -> None:
        net = make_mnist_net()
        expected = 65 * 32 + 33 * 10  # 2080 + 330 = 2410
        assert net.n_signed_branches == expected

    def test_n_delay_cells(self) -> None:
        net = make_mnist_net()
        assert net.n_delay_cells == 2 * net.n_signed_branches  # 4820

    def test_forward_output_shape(self) -> None:
        net = make_mnist_net()
        x = np.random.default_rng(0).random(64).astype(np.float32)
        T_in = net.encode_time(x)
        p = net.forward(T_in)
        assert p.shape == (10,)

    def test_forward_probabilities_in_unit_interval(self) -> None:
        net = make_mnist_net()
        x = np.random.default_rng(1).random(64).astype(np.float32)
        T_in = net.encode_time(x)
        p = net.forward(T_in).detach()
        assert float(p.min()) > 0.0
        assert float(p.max()) < 1.0

    def test_forward_logits_shape(self) -> None:
        net = make_mnist_net()
        x = np.random.default_rng(2).random(64).astype(np.float32)
        T_in = net.encode_time(x)
        logits = net.forward_logits(T_in)
        assert logits.shape == (10,)

    def test_forward_batch_shape(self) -> None:
        net = make_mnist_net()
        rng = np.random.default_rng(3)
        X = rng.random((8, 64)).astype(np.float32)
        T_batch = torch.stack([net.encode_time(x) for x in X])
        p = net.forward_batch(T_batch)
        assert p.shape == (8, 10)

    def test_forward_logits_batch_shape(self) -> None:
        net = make_mnist_net()
        rng = np.random.default_rng(4)
        X = rng.random((8, 64)).astype(np.float32)
        T_batch = torch.stack([net.encode_time(x) for x in X])
        logits = net.forward_logits_batch(T_batch)
        assert logits.shape == (8, 10)

    def test_batch_matches_single_sample(self) -> None:
        """Batched forward should match looped single-sample forward."""
        torch.manual_seed(7)
        net = make_mnist_net()
        rng = np.random.default_rng(5)
        X = rng.random((4, 64)).astype(np.float32)
        T_list = [net.encode_time(x) for x in X]
        T_batch = torch.stack(T_list)

        p_single = torch.stack([net.forward(T) for T in T_list])
        p_batch = net.forward_batch(T_batch)
        assert torch.allclose(p_single, p_batch, atol=1e-5)

    def test_logits_batch_matches_single(self) -> None:
        torch.manual_seed(8)
        net = make_mnist_net()
        rng = np.random.default_rng(6)
        X = rng.random((4, 64)).astype(np.float32)
        T_list = [net.encode_time(x) for x in X]
        T_batch = torch.stack(T_list)

        logits_single = torch.stack([net.forward_logits(T) for T in T_list])
        logits_batch = net.forward_logits_batch(T_batch)
        assert torch.allclose(logits_single, logits_batch, atol=1e-5)


class TestEncodeTime:
    def test_bright_pixel_fires_early(self) -> None:
        net = MemristorNet(n_inputs=4, hidden_sizes=[], n_outputs=1)
        T = net.encode_time(np.array([1.0, 0.5, 0.1, 0.01]))
        assert float(T[0]) < float(T[1]) < float(T[2]) < float(T[3])

    def test_bias_always_zero(self) -> None:
        net = MemristorNet(n_inputs=4, hidden_sizes=[], n_outputs=1)
        T = net.encode_time(np.array([0.3, 0.7, 0.5, 0.2]))
        assert float(T[-1]) == pytest.approx(0.0)

    def test_output_clamped_to_t_inactive(self) -> None:
        net = MemristorNet(n_inputs=2, hidden_sizes=[], n_outputs=1, T_inactive=150.0)
        T = net.encode_time(np.array([0.001, 0.999]))
        assert float(T[0]) <= 150.0
        assert float(T[1]) >= 0.0


class TestMulticlassTrainer:
    def test_batch_step_returns_scalar(self) -> None:
        torch.manual_seed(0)
        net = make_mnist_net()
        trainer = MemristorTrainer(net, eta=0.01, binary_input=False, multiclass=True)
        rng = np.random.default_rng(0)
        X = rng.random((8, 64)).astype(np.float32)
        Y = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        loss = trainer.batch_step(X, Y)
        assert isinstance(loss, float) and loss > 0.0

    def test_params_change_after_batch_step(self) -> None:
        torch.manual_seed(1)
        net = make_mnist_net()
        trainer = MemristorTrainer(net, eta=0.01, binary_input=False, multiclass=True)
        u_before = net.layers[0].u_pos.data.clone()
        rng = np.random.default_rng(1)
        X = rng.random((8, 64)).astype(np.float32)
        Y = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        trainer.batch_step(X, Y)
        assert not torch.allclose(u_before, net.layers[0].u_pos.data)

    def test_accuracy_multiclass_between_0_and_1(self) -> None:
        torch.manual_seed(2)
        net = make_mnist_net()
        trainer = MemristorTrainer(net, eta=0.01, binary_input=False, multiclass=True)
        rng = np.random.default_rng(2)
        X = rng.random((20, 64)).astype(np.float32)
        Y = (np.arange(20) % 10).astype(np.int64)
        acc = trainer.accuracy(X, Y)
        assert 0.0 <= acc <= 1.0

    def test_cross_entropy_loss_decreases_on_easy_batch(self) -> None:
        """After many batch steps on a repeated sample, loss should drop."""
        torch.manual_seed(3)
        net = make_mnist_net()
        trainer = MemristorTrainer(net, eta=0.05, binary_input=False, multiclass=True,
                                   batch_size=1)
        rng = np.random.default_rng(3)
        X = rng.random((1, 64)).astype(np.float32)
        Y = np.array([5], dtype=np.int64)
        losses = [trainer.batch_step(X, Y) for _ in range(50)]
        assert losses[-1] < losses[0], "Loss should trend downward on a single sample"

    def test_gradient_reaches_all_layers_batch(self) -> None:
        torch.manual_seed(4)
        net = make_mnist_net()
        rng = np.random.default_rng(4)
        X = rng.random((4, 64)).astype(np.float32)
        Y = np.array([0, 3, 7, 9], dtype=np.int64)
        # Zero eta so params don't change; check grad exists after batch_step
        T_batch = torch.stack([net.encode_time(x) for x in X])
        net.zero_grad()
        logits = net.forward_logits_batch(T_batch)
        y_t = torch.tensor(Y, dtype=torch.long)
        import torch.nn.functional as functional
        loss = functional.cross_entropy(logits, y_t)
        loss.backward()
        for layer in net.layers:
            assert layer.u_pos.grad is not None
            assert layer.u_pos.grad.abs().sum() > 0
            assert layer.u_neg.grad is not None
            assert layer.u_neg.grad.abs().sum() > 0
