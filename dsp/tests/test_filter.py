"""Tests for FIRFilter and IIRFilter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp

from dsp.filter import FIRFilter, IIRFilter

# ---------------------------------------------------------------------------
# FIRFilter
# ---------------------------------------------------------------------------


class TestFIRFilter:
    def test_identity_filter(self) -> None:
        """h=[1] passes signal through unchanged."""
        fir = FIRFilter(np.array([1.0]))
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        out = fir.process(sig)
        # With a single-tap filter and initial state 0, output is the delayed input
        assert out.shape == sig.shape

    def test_moving_average_matches_numpy(self) -> None:
        """3-tap uniform average: h = [1/3, 1/3, 1/3]."""
        h = np.ones(3) / 3.0
        fir = FIRFilter(h)
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(100)
        out = fir.process(sig)
        # Compare against scipy lfilter (same causal FIR, same initial zeros)
        ref = sp.lfilter(h, [1.0], sig)
        assert out == pytest.approx(ref, abs=1e-10)

    def test_arbitrary_coefficients_match_scipy(self) -> None:
        h = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
        fir = FIRFilter(h)
        rng = np.random.default_rng(13)
        sig = rng.standard_normal(200)
        out = fir.process(sig)
        ref = sp.lfilter(h, [1.0], sig)
        assert out == pytest.approx(ref, abs=1e-10)

    def test_order_property(self) -> None:
        h = np.ones(7)
        fir = FIRFilter(h)
        assert fir.order == 6

    def test_reset_clears_state(self) -> None:
        h = np.array([0.5, 0.3, 0.2])
        fir = FIRFilter(h)
        sig = np.ones(10)
        fir.process(sig)
        fir.reset()
        out_after_reset = fir.process(sig)
        fir2 = FIRFilter(h)
        out_fresh = fir2.process(sig)
        assert out_after_reset == pytest.approx(out_fresh)

    def test_jitter_output_differs_from_noiseless(self) -> None:
        h = np.array([0.2, 0.5, 0.3])
        fir_clean = FIRFilter(h)
        fir_noisy = FIRFilter(h, jitter_std=0.5)
        rng = np.random.default_rng(0)
        sig = np.random.default_rng(1).standard_normal(100)
        clean = fir_clean.process(sig)
        noisy = fir_noisy.process(sig, rng=rng)
        assert not np.allclose(clean, noisy)

    def test_process_vs_tick(self) -> None:
        """batch process() and sequential tick() must agree."""
        h = np.array([0.25, 0.5, 0.25])
        fir_batch = FIRFilter(h)
        fir_tick = FIRFilter(h)
        rng = np.random.default_rng(99)
        sig = rng.standard_normal(50)
        batch_out = fir_batch.process(sig)
        tick_out = np.array([fir_tick.tick(float(s)) for s in sig])
        assert batch_out == pytest.approx(tick_out, abs=1e-12)

    def test_dirac_response_is_coefficients(self) -> None:
        """Response to a unit impulse equals h, then zeros."""
        h = np.array([1.0, -0.5, 0.25])
        fir = FIRFilter(h)
        impulse = np.zeros(10)
        impulse[0] = 1.0
        out = fir.process(impulse)
        # Causal FIR: y[0]=h[0]*x[0]=h[0], y[1]=h[0]*x[1]+h[1]*x[0]=h[1], etc.
        assert out[:3] == pytest.approx(h, abs=1e-12)
        assert out[3:] == pytest.approx(np.zeros(7), abs=1e-12)


# ---------------------------------------------------------------------------
# IIRFilter
# ---------------------------------------------------------------------------


class TestIIRFilter:
    def test_no_feedback_matches_fir(self) -> None:
        """IIR with empty a-coefficients should equal FIR."""
        b = np.array([0.3, 0.4, 0.3])
        iir = IIRFilter(b)
        fir = FIRFilter(b)
        rng = np.random.default_rng(5)
        sig = rng.standard_normal(80)
        assert iir.process(sig) == pytest.approx(fir.process(sig), abs=1e-12)

    def test_first_order_lowpass_matches_scipy(self) -> None:
        """Simple single-pole lowpass: b=[0.1], a=[−0.9] → y[n]=0.1x[n]+0.9y[n-1]."""
        b = np.array([0.1])
        a = np.array([-0.9])  # sign convention: y -= a[k]*y[n-k]
        iir = IIRFilter(b, a)
        rng = np.random.default_rng(3)
        sig = rng.standard_normal(100)
        out = iir.process(sig)
        # scipy lfilter: a_scipy = [1, 0.9] (a[1]=0.9 means +0.9*y[n-1])
        # Our sign: y[n] = b@x_taps - a@y_taps
        # With a=[-0.9]: y[n] = 0.1*x[n] - (-0.9)*y[n-1] = 0.1*x[n] + 0.9*y[n-1]
        ref = sp.lfilter([0.1], [1.0, -0.9], sig)
        assert out == pytest.approx(ref, abs=1e-10)

    def test_reset_restores_initial_state(self) -> None:
        b = np.array([0.5, 0.3])
        a = np.array([-0.5])
        iir = IIRFilter(b, a)
        sig = np.random.default_rng(7).standard_normal(30)
        out1 = iir.process(sig)
        iir.reset()
        out2 = iir.process(sig)
        assert out1 == pytest.approx(out2)

    def test_iir_has_infinite_impulse_response(self) -> None:
        """Stable pole at 0.9: impulse response should decay slowly, not hit 0."""
        b = np.array([1.0])
        a = np.array([-0.9])
        iir = IIRFilter(b, a)
        impulse = np.zeros(50)
        impulse[0] = 1.0
        out = iir.process(impulse)
        # After the impulse (index > 0), output should still be nonzero (decaying)
        assert abs(out[20]) > 1e-3
        assert abs(out[40]) > 1e-5

    def test_tick_vs_process_agree(self) -> None:
        b = np.array([0.2, 0.4, 0.4])
        a = np.array([-0.3])
        iir_batch = IIRFilter(b, a)
        iir_tick = IIRFilter(b, a)
        rng = np.random.default_rng(8)
        sig = rng.standard_normal(60)
        batch = iir_batch.process(sig)
        sequential = np.array([iir_tick.tick(float(s)) for s in sig])
        assert batch == pytest.approx(sequential, abs=1e-12)
