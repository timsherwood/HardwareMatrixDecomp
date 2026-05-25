"""Tests for HILTrainer (Direct Feedback Alignment and SPSA)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from memristor.hil_training import HILTrainer
from memristor.network import MemristorNet

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)


def _make_net(seed: int = 0, complementary: bool = False) -> MemristorNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=complementary)


# ---------------------------------------------------------------------------
# DFA output-layer formula matches exact backprop gradient direction
# ---------------------------------------------------------------------------


class TestDFAOutputLayerExact:
    """For the output layer, DFA is exact — verify it matches autograd."""

    def test_output_layer_grad_matches_backprop_direction(self) -> None:
        """DFA update for the output layer should have the same sign as backprop."""
        torch.manual_seed(7)
        net = _make_net(seed=7)
        x = XOR_X[1]  # [0, 1] → y=1
        y = XOR_Y[1]

        # --- Exact backprop gradient ---
        T_in = net.encode_binary(x)
        net.zero_grad()
        p = net.forward(T_in)
        loss = torch.nn.functional.binary_cross_entropy(
            p.clamp(1e-7, 1 - 1e-7), torch.tensor([float(y)])
        )
        loss.backward()
        bp_grad_u_pos = net.layers[-1].u_pos.grad.clone()
        bp_grad_u_neg = net.layers[-1].u_neg.grad.clone()

        # --- DFA step on a fresh copy (output layer only — eta=0 so hidden unchanged) ---
        # We re-run step() with eta=1 and compare the direction of output-layer update.
        trainer2 = HILTrainer(_make_net(seed=7), eta=1.0, seed=0)
        u_pos_before = trainer2.net.layers[-1].u_pos.data.clone()
        u_neg_before = trainer2.net.layers[-1].u_neg.data.clone()
        trainer2.step(x, y)
        dfa_delta_u_pos = trainer2.net.layers[-1].u_pos.data - u_pos_before
        dfa_delta_u_neg = trainer2.net.layers[-1].u_neg.data - u_neg_before

        # DFA step applies: u -= eta * grad_u, so delta = -grad_u
        # Check sign agreement: delta and -bp_grad should have positive dot product
        dot_pos = float((dfa_delta_u_pos * (-bp_grad_u_pos)).sum())
        dot_neg = float((dfa_delta_u_neg * (-bp_grad_u_neg)).sum())
        assert dot_pos > 0, f"u_pos update direction mismatch vs backprop: dot={dot_pos:.4f}"
        assert dot_neg > 0, f"u_neg update direction mismatch vs backprop: dot={dot_neg:.4f}"

    def test_loss_decreases_after_dfa_step(self) -> None:
        """A single DFA step should on average reduce loss on the same sample."""
        decreases = 0
        for seed in range(10):
            net = _make_net(seed)
            trainer = HILTrainer(net, eta=0.1, seed=seed)
            x, y = XOR_X[1], XOR_Y[1]
            T_in = net.encode_binary(x)
            with torch.no_grad():
                p0 = net.forward(T_in)
                L0 = float(
                    torch.nn.functional.binary_cross_entropy(
                        p0.clamp(1e-7, 1 - 1e-7), torch.tensor([float(y)])
                    )
                )
            trainer.step(x, y)
            with torch.no_grad():
                p1 = net.forward(T_in)
                L1 = float(
                    torch.nn.functional.binary_cross_entropy(
                        p1.clamp(1e-7, 1 - 1e-7), torch.tensor([float(y)])
                    )
                )
            if L1 < L0:
                decreases += 1
        assert decreases >= 6, f"Loss decreased only {decreases}/10 times — DFA not working"


# ---------------------------------------------------------------------------
# SPSA
# ---------------------------------------------------------------------------


class TestSPSA:
    def test_restore_leaves_params_unchanged(self) -> None:
        """After SPSA perturbation+restore, parameters must be identical to before."""
        net = _make_net(seed=3)
        trainer = HILTrainer(net, eta=0.0)  # eta=0 so no net update
        params_before = [p.data.clone() for p in net.parameters()]
        trainer.step_spsa(XOR_X[2], XOR_Y[2], epsilon=0.05)
        params_after = [p.data.clone() for p in net.parameters()]
        for b, a in zip(params_before, params_after, strict=True):
            # With eta=0 the SPSA still perturbs ±ε and restores, then applies 0*g
            # The restore must bring us back exactly
            assert torch.allclose(b, a, atol=0.0), "params not restored correctly"

    def test_spsa_step_returns_scalar_loss(self) -> None:
        net = _make_net(seed=4)
        trainer = HILTrainer(net, eta=0.05)
        loss = trainer.step_spsa(XOR_X[0], XOR_Y[0])
        assert isinstance(loss, float)
        assert 0.0 < loss < 10.0

    def test_loss_decreases_on_average_spsa(self) -> None:
        """Over multiple SPSA steps on a single sample, loss should trend down."""
        torch.manual_seed(99)
        net = _make_net(seed=99)
        trainer = HILTrainer(net, eta=0.3, seed=99)
        x, y = XOR_X[1], XOR_Y[1]
        losses = [trainer.step_spsa(x, y, epsilon=0.2) for _ in range(200)]
        # Average over last 50 steps should be lower than first 50
        early = float(np.mean(losses[:50]))
        late = float(np.mean(losses[150:]))
        assert late < early, f"SPSA loss not decreasing: early={early:.4f} late={late:.4f}"


# ---------------------------------------------------------------------------
# XOR convergence
# ---------------------------------------------------------------------------


class TestXORConvergence:
    def test_dfa_xor_convergence(self) -> None:
        """At least 2 of 8 seeds should reach 100% XOR accuracy with DFA."""
        successes = 0
        for seed in range(8):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
            trainer = HILTrainer(net, eta=0.08, seed=seed)
            trainer.fit(XOR_X, XOR_Y, n_epochs=3000, method="dfa")
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                successes += 1
        assert successes >= 2, f"DFA converged only {successes}/8 seeds on XOR"

    def test_spsa_xor_convergence(self) -> None:
        """At least 1 of 5 seeds should reach 100% XOR accuracy with batch SPSA."""
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
            trainer = HILTrainer(net, eta=0.5, seed=seed)
            trainer.fit(XOR_X, XOR_Y, n_epochs=3000, method="spsa", spsa_epsilon=0.15)
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                return
        pytest.fail("No seed converged to 100% XOR with batch SPSA in 3000 epochs")

    def test_dfa_complementary_xor_convergence(self) -> None:
        """DFA works with ComplementaryDelayLayer (single-u weights)."""
        for seed in range(8):
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1, complementary=True)
            trainer = HILTrainer(net, eta=0.08, seed=seed)
            trainer.fit(XOR_X, XOR_Y, n_epochs=3000, method="dfa")
            if trainer.accuracy(XOR_X, XOR_Y) == 1.0:
                return
        pytest.fail("DFA+complementary: no seed converged on XOR in 3000 epochs")


# ---------------------------------------------------------------------------
# HILTrainer structure
# ---------------------------------------------------------------------------


class TestHILTrainerStructure:
    def test_feedback_matrices_shape(self) -> None:
        net = _make_net()
        trainer = HILTrainer(net, seed=0)
        # One B per hidden layer (not output layer)
        assert len(trainer.B) == len(net.layers) - 1
        # B_l: (n_out_l, n_final_out)
        for B, layer in zip(trainer.B, net.layers[:-1], strict=True):
            assert B.shape == (layer.n_out, net.layers[-1].n_out)

    def test_single_layer_net_has_no_feedback_matrices(self) -> None:
        """A net with no hidden layers needs no B matrices."""
        net = MemristorNet(n_inputs=2, hidden_sizes=[], n_outputs=1)
        trainer = HILTrainer(net, seed=0)
        assert len(trainer.B) == 0

    def test_dfa_step_returns_float_loss(self) -> None:
        net = _make_net()
        trainer = HILTrainer(net, eta=0.05, seed=0)
        loss = trainer.step(XOR_X[0], XOR_Y[0])
        assert isinstance(loss, float)
        assert 0.0 < loss < 5.0

    def test_accuracy_before_training_is_not_perfect(self) -> None:
        torch.manual_seed(42)
        net = _make_net(seed=42)
        trainer = HILTrainer(net)
        # A freshly-initialized network shouldn't be perfect on XOR
        acc = trainer.accuracy(XOR_X, XOR_Y)
        assert acc < 1.0

    def test_fit_returns_loss_list(self) -> None:
        net = _make_net()
        trainer = HILTrainer(net, eta=0.05, seed=0)
        losses = trainer.fit(XOR_X, XOR_Y, n_epochs=10, method="dfa")
        assert len(losses) == 10
        assert all(isinstance(v, float) for v in losses)

    def test_no_autograd_used_in_dfa_step(self) -> None:
        """DFA step must not leave any grad tensors on parameters."""
        net = _make_net()
        trainer = HILTrainer(net, eta=0.05, seed=0)
        trainer.step(XOR_X[1], XOR_Y[1])
        for param in net.parameters():
            assert param.grad is None, "DFA step left grad on a parameter"

    def test_dfa_and_spsa_modify_params(self) -> None:
        """Both methods must actually change at least one parameter."""
        for method in ("dfa", "spsa"):
            net = _make_net(seed=5)
            trainer = HILTrainer(net, eta=0.1, seed=5)
            params_before = [p.data.clone() for p in net.parameters()]
            if method == "spsa":
                trainer.step_spsa(XOR_X[1], XOR_Y[1])
            else:
                trainer.step(XOR_X[1], XOR_Y[1])
            params_after = [p.data.clone() for p in net.parameters()]
            changed = any(
                not torch.equal(b, a)
                for b, a in zip(params_before, params_after, strict=True)
            )
            assert changed, f"{method}: no parameter changed after step"
