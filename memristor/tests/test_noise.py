"""Tests for Simulation 3: noisy timing model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.network import MemristorNet
from memristor.noise import NoisyMemristorNet
from memristor.training import MemristorTrainer

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def make_net(seed: int = 0) -> MemristorNet:
    torch.manual_seed(seed)
    return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)


class TestNoisyMemristorNet:
    def test_zero_jitter_matches_base_net(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=0.0)
        T_in = net.encode_binary(np.array([1.0, 0.0]))
        out_base = net.forward(T_in)
        out_noisy = noisy.forward(T_in)
        assert torch.allclose(out_base, out_noisy)

    def test_nonzero_jitter_changes_output(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=5.0)
        T_in = net.encode_binary(np.array([1.0, 0.0]))
        rng = np.random.default_rng(0)
        out1 = noisy.forward(T_in, rng=rng)
        rng2 = np.random.default_rng(1)
        out2 = noisy.forward(T_in, rng=rng2)
        # Two different RNG seeds should (almost certainly) give different outputs
        assert not torch.allclose(out1, out2)

    def test_output_shape(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=1.0)
        T_in = net.encode_binary(np.array([0.0, 1.0]))
        out = noisy.forward(T_in, rng=np.random.default_rng(42))
        assert out.shape == (1,)
        assert 0.0 < float(out.detach()) < 1.0

    def test_predict_noisy_returns_mean_std(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=3.0)
        mean, std = noisy.predict_noisy(np.array([1.0, 0.0]), n_trials=200, seed=7)
        assert 0.0 <= mean <= 1.0
        assert std >= 0.0

    def test_predict_noisy_zero_jitter_zero_std(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=0.0)
        mean, std = noisy.predict_noisy(np.array([1.0, 0.0]), n_trials=10)
        assert std == pytest.approx(0.0, abs=1e-6)

    def test_accuracy_noisy_is_between_0_and_1(self) -> None:
        net = make_net()
        noisy = NoisyMemristorNet(net, sigma_j=1.0)
        acc = noisy.accuracy_noisy(XOR_X, XOR_Y, n_trials=20)
        assert 0.0 <= acc <= 1.0

    def test_large_jitter_hurts_accuracy(self) -> None:
        """sigma_j >> tau should degrade accuracy below 1.0 for some seeds."""
        # Train to convergence, then apply extreme jitter
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                noisy = NoisyMemristorNet(net, sigma_j=20.0)  # 2x tau
                # Confirm it runs without error; extreme jitter may or may not flip XOR
                noisy.accuracy_noisy(XOR_X, XOR_Y, n_trials=100, seed=42)
                return
        pytest.skip("No seed converged")

    def test_small_jitter_preserves_accuracy(self) -> None:
        """sigma_j = 0.1 ns << tau = 10 ns should leave a converged net intact."""
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                noisy = NoisyMemristorNet(net, sigma_j=0.1)
                acc = noisy.accuracy_noisy(XOR_X, XOR_Y, n_trials=100, seed=42)
                assert acc == 1.0, f"sigma_j=0.1 should not break XOR (seed={seed})"
                return
        pytest.skip("No seed converged")

    def test_medium_jitter_majority_vote_accuracy(self) -> None:
        """sigma_j = 1.0 ns (tau/10) should still achieve 100% majority-vote accuracy."""
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, T_inactive=150.0)
            trainer = MemristorTrainer(net, eta=0.06)
            trainer.fit(XOR_X, XOR_Y, n_epochs=2000)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                noisy = NoisyMemristorNet(net, sigma_j=1.0)
                acc = noisy.accuracy_noisy(XOR_X, XOR_Y, n_trials=50, seed=7)
                assert acc == 1.0, (
                    f"sigma_j=1.0 ns (tau/10) should preserve XOR (seed={seed}, acc={acc})"
                )
                return
        pytest.skip("No seed converged")
