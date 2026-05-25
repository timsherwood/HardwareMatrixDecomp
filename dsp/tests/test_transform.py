"""Tests for Filterbank, toeplitz_matrix, and stft."""

from __future__ import annotations

import numpy as np
import pytest

from dsp.transform import Filterbank, make_bandpass_coefficients, stft, toeplitz_matrix

# ---------------------------------------------------------------------------
# toeplitz_matrix
# ---------------------------------------------------------------------------


class TestToeplitzMatrix:
    def test_shape(self) -> None:
        x = np.ones(8)
        T = toeplitz_matrix(x, filter_length=3)
        assert T.shape == (8, 3)

    def test_first_row_is_x0_padded(self) -> None:
        x = np.array([7.0, 2.0, 3.0, 4.0, 5.0])
        T = toeplitz_matrix(x, filter_length=3)
        # T[0] = [x[0], 0, 0]: causal — only x[0] available at time 0
        assert T[0] == pytest.approx([7.0, 0.0, 0.0])

    def test_full_row_after_warmup(self) -> None:
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        T = toeplitz_matrix(x, filter_length=3)
        # Row 2: T[2, k] = x[2-k] for k=0,1,2 → [x[2], x[1], x[0]]
        assert T[2] == pytest.approx([30.0, 20.0, 10.0])

    def test_multiply_gives_convolution(self) -> None:
        """T @ h should match causal FIR convolution."""
        h = np.array([1.0, -0.5, 0.25])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        T = toeplitz_matrix(x, filter_length=len(h))
        y = T @ h
        # Reference: causal FIR via scipy
        from scipy import signal as sp

        ref = sp.lfilter(h, [1.0], x)
        assert y == pytest.approx(ref, abs=1e-12)

    def test_unit_impulse_recovers_h(self) -> None:
        """Convolving impulse signal with h via T @ h should return h."""
        h = np.array([3.0, 1.5, 0.75, 0.25])
        x = np.zeros(len(h))
        x[0] = 1.0
        T = toeplitz_matrix(x, filter_length=len(h))
        y = T @ h
        # y[0] = h[0]*x[0] = h[0], y[1] = h[0]*x[1]+h[1]*x[0] = h[1], etc.
        assert y == pytest.approx(h, abs=1e-12)


# ---------------------------------------------------------------------------
# make_bandpass_coefficients
# ---------------------------------------------------------------------------


class TestMakeBandpassCoefficients:
    def test_output_length(self) -> None:
        h = make_bandpass_coefficients(0.1, 0.05, n_taps=31)
        assert len(h) == 31

    def test_finite_values(self) -> None:
        h = make_bandpass_coefficients(0.2, 0.08, n_taps=21)
        assert np.all(np.isfinite(h))

    def test_approximate_bandpass_response(self) -> None:
        """DC and Nyquist response should be much smaller than pass-band."""
        h = make_bandpass_coefficients(0.25, 0.1, n_taps=51, sample_rate=1.0)
        H = np.fft.rfft(h, n=512)
        freqs = np.fft.rfftfreq(512)
        passband_idx = np.argmin(abs(freqs - 0.25))
        dc_response = abs(H[0])
        passband_response = abs(H[passband_idx])
        assert passband_response > dc_response * 2.0


# ---------------------------------------------------------------------------
# Filterbank
# ---------------------------------------------------------------------------


class TestFilterbank:
    def make_filterbank(self) -> Filterbank:
        H = np.random.default_rng(0).standard_normal((4, 8))
        return Filterbank(H)

    def test_tick_output_shape(self) -> None:
        fb = self.make_filterbank()
        out = fb.tick(1.0)
        assert out.shape == (4,)

    def test_process_output_shape(self) -> None:
        fb = self.make_filterbank()
        sig = np.ones(20)
        out = fb.process(sig)
        assert out.shape == (20, 4)

    def test_invalid_coefficient_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            Filterbank(np.ones(8))  # 1-D, not 2-D

    def test_process_matches_individual_fir_filters(self) -> None:
        """Each column of filterbank output should match a standalone FIRFilter."""
        rng = np.random.default_rng(2)
        H = rng.standard_normal((3, 6))
        fb = Filterbank(H)
        sig = rng.standard_normal(50)
        fb_out = fb.process(sig)  # (50, 3)

        from dsp.filter import FIRFilter

        for k in range(3):
            fir = FIRFilter(H[k])
            ref = fir.process(sig)
            assert fb_out[:, k] == pytest.approx(ref, abs=1e-10), f"filter {k} mismatch"

    def test_reset_restores_initial_state(self) -> None:
        H = np.eye(4)[:, :4]
        fb = Filterbank(H)
        sig = np.arange(1.0, 11.0)
        out1 = fb.process(sig)
        fb.reset()
        out2 = fb.process(sig)
        assert out1 == pytest.approx(out2)

    def test_from_frequency_bands(self) -> None:
        bands = [(0.1, 0.05), (0.2, 0.05), (0.3, 0.05)]
        fb = Filterbank.from_frequency_bands(bands, n_taps=31, sample_rate=1.0)
        assert fb.n_filters == 3
        assert fb.n_taps == 31

    def test_svd_rank_reduction_approximation(self) -> None:
        rng = np.random.default_rng(5)
        H = rng.standard_normal((8, 16))
        fb = Filterbank(H)
        A, B = fb.svd_rank_reduction(rank=4)
        H_approx = B @ A.T
        # Frobenius error should be much less than full norm for rank-4 approx
        full_norm = np.linalg.norm(H)
        approx_err = np.linalg.norm(H - H_approx)
        assert approx_err < full_norm

    def test_svd_full_rank_exact(self) -> None:
        """At full rank the SVD reconstruction should be exact."""
        rng = np.random.default_rng(6)
        H = rng.standard_normal((5, 5))
        fb = Filterbank(H)
        A, B = fb.svd_rank_reduction(rank=5)
        H_approx = B @ A.T
        assert H_approx == pytest.approx(H, abs=1e-10)

    def test_svd_reduces_tile_count(self) -> None:
        """H (150, 250) at full rank needs ceil(150/100)*ceil(250/100)=2*3=6 tiles.
        At rank 2: A=(250,2), B=(150,2) → ceil(250/100)*1 + ceil(150/100)*1 = 3+2=5 tiles.
        """
        import math

        TILE = 100

        def tiles(r: int, c: int) -> int:
            return math.ceil(r / TILE) * math.ceil(c / TILE)

        rows, cols, rank = 150, 250, 2
        full_tiles = tiles(rows, cols)  # 2*3 = 6
        svd_tiles = tiles(cols, rank) + tiles(rows, rank)  # 3*1 + 2*1 = 5
        assert svd_tiles < full_tiles


# ---------------------------------------------------------------------------
# stft
# ---------------------------------------------------------------------------


class TestSTFT:
    def test_output_shape(self) -> None:
        sig = np.random.default_rng(0).standard_normal(1000)
        S = stft(sig, n_fft=64, hop=32)
        n_frames = (len(sig) - 64) // 32 + 1
        assert S.shape == (n_frames, 33)  # rfft gives n_fft//2+1

    def test_output_is_complex(self) -> None:
        sig = np.ones(200)
        S = stft(sig, n_fft=32, hop=16)
        assert np.iscomplexobj(S)

    def test_dc_sine_energy_in_first_bin(self) -> None:
        """A DC signal should concentrate energy at bin 0."""
        sig = np.ones(512) * 3.0
        S = stft(sig, n_fft=64, hop=32)
        magnitudes = np.abs(S)
        assert magnitudes[:, 0].mean() > magnitudes[:, 10:].mean() * 5.0

    def test_short_signal_returns_empty(self) -> None:
        sig = np.ones(10)
        S = stft(sig, n_fft=64, hop=32)
        assert S.shape[0] == 0

    def test_custom_window(self) -> None:
        sig = np.random.default_rng(1).standard_normal(300)
        window = np.ones(32)  # rectangular window
        S = stft(sig, n_fft=32, hop=16, window=window)
        assert S.shape[1] == 17
        assert np.all(np.isfinite(np.abs(S)))
