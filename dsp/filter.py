"""FIR and IIR filters built on the TappedDelayLine primitive.

FIR filter
----------
A length-N FIR filter is a dot product of N tap values against N
fixed coefficients h[0..N-1]:

    y[n] = sum_{k=0}^{N-1} h[k] * x[n-k]

This is exactly one row of a matrix-vector product when the input
is presented as a Toeplitz column matrix (see toeplitz_matrix in
transform.py).  The hardware tile operates on exactly this representation.

IIR filter
----------
An IIR filter adds a recursive feedback path:

    y[n] = sum_{k=0}^{N-1} b[k]*x[n-k]  -  sum_{k=1}^{M} a[k]*y[n-k]

The feedback is modelled by a second TappedDelayLine on the output
stream.  This is the delay-space incarnation of the standard Direct
Form I structure.

Hardware mapping
----------------
Both filters operate through dot-products of tap vectors against
coefficient vectors.  For a filter bank (see transform.py) the
coefficient matrix is (num_filters × N), which maps to a single
tile group via SVDDecomposition when N ≤ 100.
"""

from __future__ import annotations

import numpy as np

from dsp.delay_line import TappedDelayLine


class FIRFilter:
    """Finite Impulse Response filter via a tapped delay line.

    Parameters
    ----------
    coefficients:
        FIR tap weights h[0..N-1].  h[0] multiplies the current sample,
        h[1] multiplies the previous sample, etc.
    jitter_std:
        Timing jitter standard deviation passed to the underlying delay line.
    """

    def __init__(self, coefficients: np.ndarray, jitter_std: float = 0.0) -> None:
        self.h = np.asarray(coefficients, dtype=np.float64)
        self._delay_line = TappedDelayLine(len(self.h), jitter_std=jitter_std)

    @property
    def order(self) -> int:
        return len(self.h) - 1

    def tick(self, x: float, rng: np.random.Generator | None = None) -> float:
        """Process one sample, return one output sample."""
        taps = self._delay_line.tick(x, rng=rng)
        return float(np.dot(self.h, taps))

    def process(self, signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Filter an entire 1-D signal.

        Returns an array the same length as signal.  The first (order)
        samples are in the transient/warm-up region (initial state = 0).
        """
        signal = np.asarray(signal, dtype=np.float64).ravel()
        tap_matrix = self._delay_line.process(signal, rng=rng)
        # tap_matrix shape: (len(signal), N)
        # Each row dotted with h gives one output sample — one tile row op
        return tap_matrix @ self.h

    def reset(self) -> None:
        self._delay_line.reset()


class IIRFilter:
    """Infinite Impulse Response filter (Direct Form I) via two delay lines.

    Parameters
    ----------
    b:
        Feed-forward (numerator) coefficients, length N.
    a:
        Feed-back (denominator) coefficients.  a[0] is assumed 1 and
        omitted; provide a[1..M].  Pass ``[]`` for a pure FIR.
    jitter_std:
        Timing jitter applied to *both* delay lines.
    """

    def __init__(
        self,
        b: np.ndarray,
        a: np.ndarray | None = None,
        jitter_std: float = 0.0,
    ) -> None:
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a if a is not None else [], dtype=np.float64)
        self._x_line = TappedDelayLine(len(self.b), jitter_std=jitter_std)
        self._y_line: TappedDelayLine | None = (
            TappedDelayLine(len(self.a), jitter_std=jitter_std) if len(self.a) > 0 else None
        )

    def tick(self, x: float, rng: np.random.Generator | None = None) -> float:
        """Process one sample, return one output sample."""
        x_taps = self._x_line.tick(float(x), rng=rng)
        y_ff = float(np.dot(self.b, x_taps))
        if self._y_line is not None:
            y_taps = self._y_line.taps
            y_fb = float(np.dot(self.a, y_taps))
            y_out = y_ff - y_fb
            self._y_line.tick(y_out, rng=rng)
        else:
            y_out = y_ff
        return y_out

    def process(self, signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Filter an entire 1-D signal sample-by-sample (IIR requires sequential)."""
        signal = np.asarray(signal, dtype=np.float64).ravel()
        return np.array([self.tick(s, rng=rng) for s in signal])

    def reset(self) -> None:
        self._x_line.reset()
        if self._y_line is not None:
            self._y_line.reset()
