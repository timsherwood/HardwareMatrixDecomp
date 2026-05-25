"""Spectral analysis and the Toeplitz / SVD hardware mapping.

Filterbank
----------
A filterbank is a set of K bandpass FIR filters applied in parallel to
the same input stream.  In delay-space terms: one shared TappedDelayLine
feeds K separate coefficient dot-products.  The coefficient matrix has
shape (K, N) and the tap vector has length N, so the full filterbank
output is a single matrix-vector product:

    y = H @ taps       H: (K, N),  taps: (N,)

When N ≤ 100 and K ≤ 100 this fits in one hardware tile.  For larger
filters the SVDDecomposition from hardware_matrix_decomp decomposes H
into two smaller matrices, reducing tile count and MAC cost exactly as
for the CNN weight matrices.

toeplitz_matrix
---------------
FIR convolution of a length-L signal with a length-N kernel h is
equivalent to the matrix product:

    y = T @ h

where T is an L×N Toeplitz signal matrix with T[i,k] = signal[i-k].
This matches the tap matrix returned by TappedDelayLine.process() and
enables direct application of SVDDecomposition to filter weights h.

Relationship to UCSB delay-space research
------------------------------------------
In the UCSB temporal-logic model:
- Each row of T corresponds to one clock cycle's tap vector.
- The hardware tiles execute the H (or T) matrix-vector product.
- Timing jitter in the delay elements propagates as noise on the taps,
  which is exactly the GaussianErrorModel applied to a tile's inputs.
- SVD-decomposing T splits the convolution into two cheaper matmuls,
  reducing both tile count and energy — the core hardware-efficiency
  argument from the matrix-decomp framework.
"""

from __future__ import annotations

import numpy as np

from dsp.delay_line import TappedDelayLine


def toeplitz_matrix(signal: np.ndarray, filter_length: int) -> np.ndarray:
    """Build the (L × N) Toeplitz signal matrix for FIR convolution.

    T[i, k] = signal[i - k]  for i - k >= 0, else 0.

    Multiplying T @ h gives the causal FIR output y = conv(signal, h),
    matching the tap-matrix produced by TappedDelayLine.process().
    Used to apply SVDDecomposition to a filter's weight matrix h.

    Parameters
    ----------
    signal:
        1-D input signal of length L.
    filter_length:
        Number of filter taps N.

    Returns
    -------
    T of shape (L, N).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    L = len(signal)
    N = filter_length
    T = np.zeros((L, N), dtype=np.float64)
    for i in range(L):
        for k in range(N):
            if i - k >= 0:
                T[i, k] = signal[i - k]
    return T


def make_bandpass_coefficients(
    center_freq: float,
    bandwidth: float,
    n_taps: int,
    sample_rate: float = 1.0,
) -> np.ndarray:
    """Sinc-windowed bandpass FIR coefficients.

    Parameters
    ----------
    center_freq:
        Centre frequency in Hz (or normalised 0–0.5 if sample_rate=1).
    bandwidth:
        Filter bandwidth in Hz.
    n_taps:
        Number of taps (odd recommended for linear phase).
    sample_rate:
        Sample rate in Hz.
    """
    fc = center_freq / sample_rate
    bw = bandwidth / sample_rate
    fc_lo = fc - bw / 2.0
    fc_hi = fc + bw / 2.0
    M = n_taps - 1
    n = np.arange(n_taps) - M / 2.0
    # Avoid divide-by-zero at n=0
    with np.errstate(invalid="ignore", divide="ignore"):
        h_lo = np.where(n == 0, 2.0 * fc_lo, np.sin(2 * np.pi * fc_lo * n) / (np.pi * n))
        h_hi = np.where(n == 0, 2.0 * fc_hi, np.sin(2 * np.pi * fc_hi * n) / (np.pi * n))
    h = h_hi - h_lo
    # Hamming window to reduce sidelobes
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_taps) / M)
    return h * window


class Filterbank:
    """Parallel bank of bandpass FIR filters sharing one delay line.

    The bank exposes the delay-space view: after shifting each input
    sample into the shared TappedDelayLine, a single matrix-vector
    product ``H @ taps`` produces all K filter outputs at once.

    Parameters
    ----------
    coefficient_matrix:
        Shape (K, N): row k is the coefficient vector for filter k.
    jitter_std:
        Timing jitter applied to the shared delay line.
    """

    def __init__(self, coefficient_matrix: np.ndarray, jitter_std: float = 0.0) -> None:
        H = np.asarray(coefficient_matrix, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError("coefficient_matrix must be 2-D (K, N)")
        self.H = H
        self.n_filters, self.n_taps = H.shape
        self._delay_line = TappedDelayLine(self.n_taps, jitter_std=jitter_std)

    @classmethod
    def from_frequency_bands(
        cls,
        bands: list[tuple[float, float]],
        n_taps: int,
        sample_rate: float = 1.0,
        jitter_std: float = 0.0,
    ) -> Filterbank:
        """Construct a filterbank from a list of (center_freq, bandwidth) pairs."""
        rows = [make_bandpass_coefficients(fc, bw, n_taps, sample_rate) for fc, bw in bands]
        return cls(np.stack(rows, axis=0), jitter_std=jitter_std)

    def tick(self, x: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Process one sample; return shape-(K,) array of filter outputs."""
        self._delay_line.tick(float(x), rng=rng)
        # Single matrix-vector product — one tile group operation
        return self.H @ self._delay_line.taps

    def process(self, signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Process a 1-D signal; return shape-(len(signal), K) output matrix."""
        signal = np.asarray(signal, dtype=np.float64).ravel()
        tap_matrix = self._delay_line.process(signal, rng=rng)
        # (L, N) @ (N, K)  — equivalent to running all K filters in parallel
        return tap_matrix @ self.H.T

    def reset(self) -> None:
        self._delay_line.reset()

    def svd_rank_reduction(self, rank: int) -> tuple[np.ndarray, np.ndarray]:
        """Factor H ≈ U_r @ Vt_r via truncated SVD.

        Returns (A, B) such that B @ A ≈ H, where:
            A  shape (N, rank)
            B  shape (K, rank)

        The filterbank output then becomes  (B @ (A.T @ taps)) — two
        smaller dot products, mapping to two tile rows instead of one
        large tile group.  This is the same decomposition used for CNN
        weight matrices in hardware_matrix_decomp.
        """
        U, s, Vt = np.linalg.svd(self.H, full_matrices=False)
        U_r = U[:, :rank] * s[:rank]  # (K, rank)
        Vt_r = Vt[:rank, :]  # (rank, N)
        # Convention: A=(N,rank), B=(K,rank) so B @ (A.T @ taps) = H @ taps
        A = Vt_r.T  # (N, rank)
        B = U_r  # (K, rank)
        return A, B


def stft(
    signal: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray | None = None,
    jitter_std: float = 0.0,
) -> np.ndarray:
    """Short-Time Fourier Transform via a DFT filterbank.

    Each DFT basis function is a complex exponential FIR filter of length
    n_fft; the filterbank output at each hop position is one STFT frame.

    In delay-space terms: the n_fft-tap delay line captures a window of
    signal history; the DFT matrix multiplied against the tap vector
    produces all frequency bins simultaneously.

    Parameters
    ----------
    signal:
        1-D real-valued input.
    n_fft:
        FFT size (= number of taps in the delay line).
    hop:
        Step size between frames in samples.
    window:
        Analysis window of length n_fft (Hann by default).
    jitter_std:
        Timing jitter on the delay line.

    Returns
    -------
    Complex array of shape (n_frames, n_fft // 2 + 1).
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if window is None:
        window = np.hanning(n_fft)
    window = np.asarray(window, dtype=np.float64)

    delay_line = TappedDelayLine(n_fft, jitter_std=jitter_std)
    tap_matrix = delay_line.process(signal)  # (L, n_fft)

    frames: list[np.ndarray] = []
    for i in range(0, len(signal) - n_fft + 1, hop):
        taps = tap_matrix[i + n_fft - 1]  # tap vector after sample i+n_fft-1
        windowed = taps * window
        frames.append(np.fft.rfft(windowed))

    if not frames:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.complex128)
    return np.stack(frames, axis=0)
