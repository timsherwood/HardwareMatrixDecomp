"""Tests for Tier2Network: end-to-end PCB hardware simulation.

Validates:
  1. Forward pass shape and output range
  2. Margin signs match the software (MemristorNet) model
  3. SPSA convergence on XOR with realistic device noise
  4. P&V programming precision
  5. BJT sense-amp verification of a trained solution
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from memristor.network import MemristorNet
from memristor.tier2.network import Tier2Network

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_Y = np.array([0, 1, 1, 0], dtype=int)


class TestTier2NetworkBasic:
    def test_forward_output_shape_xor(self):
        """Forward pass returns shape (n_outputs,) = (1,) for XOR network."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        T_in = net.encode_binary(np.array([0.0, 1.0]))
        p = net.forward(T_in)
        assert p.shape == (1,)

    def test_forward_output_in_unit_interval(self):
        """Outputs are probabilities in (0, 1)."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        for x in XOR_X:
            T_in = net.encode_binary(x)
            p = net.forward(T_in)
            for pi in p:
                assert 0.0 < float(pi) < 1.0

    def test_encode_binary_active_fires_at_zero(self):
        """Active input (x > 0.5) → T = 0 µs."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        T = net.encode_binary(np.array([1.0, 0.0]))
        assert T[0] == pytest.approx(0.0)

    def test_encode_binary_inactive_fires_at_t_inactive(self):
        """Inactive input (x ≤ 0.5) → T = T_inactive_us."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        T = net.encode_binary(np.array([1.0, 0.0]))
        assert T[1] == pytest.approx(net.T_inactive_us)

    def test_bias_node_always_zero(self):
        """Bias node (last element of T_in) is always 0 µs."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        for x in XOR_X:
            T = net.encode_binary(x)
            assert T[-1] == pytest.approx(0.0)

    def test_t_inactive_scaled_from_ns(self):
        """T_inactive_us = 1500 µs (= 150 ns × 10,000 scale factor)."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        assert net.T_inactive_us == pytest.approx(1500.0, rel=1e-6)

    def test_kappa_us_is_geometric_mean(self):
        """κ_us = sqrt(d_min_us × d_max_us) ≈ 158.1 µs."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        expected = math.sqrt(net.d_min_us * net.d_max_us)
        assert net.kappa_us == pytest.approx(expected, rel=1e-4)


class TestTier2NetworkSoftwareAlignment:
    def test_margin_signs_match_software_model(self):
        """Hardware (analytical nLSE at τ_sense) and software model agree on p > 0.5."""
        # Build software network with fixed seed
        torch.manual_seed(3)
        soft = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        # Build hardware network with same weights (scaled to µs)
        hw = Tier2Network.from_software_model(soft)
        for x in XOR_X:
            T_hw = hw.encode_binary(x)
            p_hw = float(hw.forward(T_hw)[0])
            with torch.no_grad():
                p_sw = float(soft.predict(x)[0])
            assert (p_hw > 0.5) == (p_sw > 0.5), (
                f"x={x}: hw p={p_hw:.3f}, sw p={p_sw:.3f} — classification disagrees"
            )

    def test_from_software_model_preserves_delay_ratios(self):
        """Hardware delays (µs) = software delays (ns) × (κ_us / κ_ns) ≈ 10.

        Both models use the same u values; the ratio of kappa constants converts
        nanosecond delays to microsecond delays via the larger PCB capacitor.
        """
        torch.manual_seed(0)
        soft = MemristorNet(n_inputs=2, hidden_sizes=[2], n_outputs=1)
        hw = Tier2Network.from_software_model(soft)
        import math as _math
        scale = _math.sqrt(50.0 * 500.0) / soft.layers[0].kappa  # κ_us_µs / κ_si_ns ≈ 10
        d_pos_sw, d_neg_sw = soft.layers[0].delays()
        d_pos_hw, d_neg_hw = hw.layers[0].delays_us()
        assert d_pos_hw[0, 0] == pytest.approx(float(d_pos_sw[0, 0]) * scale, rel=1e-4)
        assert d_neg_hw[0, 0] == pytest.approx(float(d_neg_sw[0, 0]) * scale, rel=1e-4)


class TestTier2NetworkSPSA:
    def test_spsa_converges_on_xor(self):
        """SPSA finds XOR solution within 3000 epochs."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        result = net.train_spsa(XOR_X, XOR_Y, n_epochs=3000, eta=0.05)
        assert result["accuracy"] == pytest.approx(1.0), (
            f"XOR not solved: accuracy={result['accuracy']:.0%}, "
            f"converged at epoch {result.get('converged_epoch', 'never')}"
        )

    def test_xor_all_four_patterns_correct(self):
        """After training, all 4 XOR patterns are classified correctly."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        net.train_spsa(XOR_X, XOR_Y, n_epochs=3000, eta=0.05)
        preds = net.predict_all(XOR_X)
        assert list(preds) == list(XOR_Y), (
            f"Expected {list(XOR_Y)}, got {list(preds)}"
        )

    def test_spsa_result_keys(self):
        """train_spsa returns expected keys."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        result = net.train_spsa(XOR_X, XOR_Y, n_epochs=10)
        assert {"accuracy", "final_loss", "converged_epoch"}.issubset(result.keys())


class TestTier2NetworkDeviceNoise:
    def test_device_noise_copy_is_independent(self):
        """with_device_noise returns a new object; original is unchanged."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        net.train_spsa(XOR_X, XOR_Y, n_epochs=3000, eta=0.05)
        # Store original delays
        d_orig, _ = net.layers[0].delays_us()
        d_orig_val = float(d_orig[0, 0])
        noisy = net.with_device_noise(noise_frac=0.20, rng=np.random.default_rng(1))
        d_noisy, _ = noisy.layers[0].delays_us()
        # Original should be unchanged
        d_after, _ = net.layers[0].delays_us()
        assert float(d_after[0, 0]) == pytest.approx(d_orig_val)
        # Noisy copy should differ (with high probability at 20% noise)
        assert float(d_noisy[0, 0]) != pytest.approx(d_orig_val, rel=1e-6)

    def test_trained_network_robust_to_small_device_noise(self):
        """Trained XOR net classifies correctly under 5% device noise (most seeds)."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        net.train_spsa(XOR_X, XOR_Y, n_epochs=3000, eta=0.05)
        n_trials = 30
        correct_trials = 0
        for seed in range(n_trials):
            noisy = net.with_device_noise(noise_frac=0.05, rng=np.random.default_rng(seed))
            preds = noisy.predict_all(XOR_X)
            if list(preds) == list(XOR_Y):
                correct_trials += 1
        # Expect ≥ 80% of noisy trials still correct at 5% noise
        assert correct_trials / n_trials >= 0.80, (
            f"Only {correct_trials}/{n_trials} noisy trials correct"
        )


class TestTier2NetworkBJTVerification:
    def test_bjt_hil_training_converges_on_xor(self):
        """Hardware-in-the-loop SPSA (BJT loss) finds XOR solution.

        This is Route 2 from HARDWARE_SPEC.md §4.2: analytical SPSA may find
        weights where τ_sense(d_min) differs enough from τ_training to cause
        the BJT circuit to misclassify XOR [1,1].  Training with BJT loss and
        an analytical warm-start adapts weights to the actual τ_sense so the
        physical circuit classifies all four patterns correctly.
        """
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        result = net.train_spsa_bjt(XOR_X, XOR_Y)  # warm-start + early-stop
        assert result["accuracy"] == pytest.approx(1.0), (
            f"BJT-HIL training did not converge: accuracy={result['accuracy']:.0%}"
        )

    def test_bjt_hil_all_patterns_correct(self):
        """After BJT-HIL training, all 4 XOR patterns pass BJT classification."""
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        net.train_spsa_bjt(XOR_X, XOR_Y)  # warm-start + early-stop
        preds_bjt = net.predict_all_bjt(XOR_X)
        assert list(preds_bjt) == list(XOR_Y), (
            f"Expected {list(XOR_Y)}, got {list(preds_bjt)}"
        )

    def test_bjt_easy_patterns_match_analytical(self):
        """Analytically-trained net: [0,0] [0,1] [1,0] agree with BJT (clear margins).

        XOR [1,1] is the τ_sense-sensitive case — excluded here.
        Use hardware-in-the-loop training (train_spsa_bjt) for full agreement.
        """
        net = Tier2Network(n_inputs=2, hidden_sizes=[2], n_outputs=1, seed=0)
        net.train_spsa(XOR_X, XOR_Y, n_epochs=3000, eta=0.05)
        easy_patterns = XOR_X[:3]   # [0,0], [0,1], [1,0] — not XOR [1,1]
        easy_labels = XOR_Y[:3]
        for x, _y in zip(easy_patterns, easy_labels, strict=True):
            T_in = net.encode_binary(x)
            p_analytical = float(net.forward(T_in)[0])
            p_bjt = float(net.forward_bjt(T_in)[0])
            assert (p_bjt > 0.5) == (p_analytical > 0.5), (
                f"x={x}: BJT p={p_bjt:.3f}, analytical p={p_analytical:.3f}"
            )
