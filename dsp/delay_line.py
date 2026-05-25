"""Core delay primitives: DelayElement (z^{-1}) and TappedDelayLine.

Delay is the fundamental computational primitive in this DSP model.
A single DelayElement holds one sample for one clock cycle. Chaining N
of them gives a shift register; exposing all intermediate states gives
a TappedDelayLine, the building block for FIR filters.

Timing jitter — random variation in when a delay resolves — is modelled
by perturbing the output with additive Gaussian noise, mirroring the
GaussianErrorModel used in the parent matrix-decomp package.
"""

from __future__ import annotations

import numpy as np


class DelayElement:
    """Single z^{-1} register: stores one scalar or array sample.

    On each call to ``tick(x)`` the *current* stored value is returned
    and replaced by ``x``.  The initial state is zero.

    Parameters
    ----------
    shape:
        Shape of the signal stored in this register.  Use ``()`` for a
        scalar delay, or ``(C,)`` for a C-channel delay.
    jitter_std:
        Standard deviation of additive Gaussian jitter applied to the
        output each tick.  Represents timing uncertainty in the physical
        delay element.  ``0.0`` means no noise.
    """

    def __init__(self, shape: tuple[int, ...] = (), jitter_std: float = 0.0) -> None:
        self._state: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.jitter_std = jitter_std

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def tick(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Advance clock: return old state, store x."""
        out = self._state.copy()
        if self.jitter_std > 0.0:
            _rng = rng if rng is not None else np.random.default_rng()
            out = out + _rng.normal(0.0, self.jitter_std, size=out.shape)
        self._state = np.asarray(x, dtype=np.float64)
        return out

    def reset(self) -> None:
        self._state = np.zeros(self._state.shape, dtype=np.float64)


class TappedDelayLine:
    """N-element shift register exposing all tap outputs simultaneously.

    This is the canonical delay-space representation of a signal's recent
    history.  Given a stream of scalar samples x[n], after N calls the tap
    vector is ``[x[n], x[n-1], …, x[n-N+1]]``.

    Multiplying taps element-wise by coefficient vector h and summing gives
    a single FIR output sample (see FIRFilter).  The whole operation is
    equivalent to a dot product ``h · taps``, which maps naturally to a
    single hardware tile row.

    Parameters
    ----------
    length:
        Number of delay elements (filter order + 1 for FIR).
    jitter_std:
        Per-element jitter applied when reading each tap.
    """

    def __init__(self, length: int, jitter_std: float = 0.0) -> None:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        self.length = length
        self._elements: list[DelayElement] = [
            DelayElement(shape=(), jitter_std=jitter_std) for _ in range(length)
        ]

    @property
    def taps(self) -> np.ndarray:
        """Current tap values as a length-N array (tap 0 = most recent)."""
        return np.array([e.state for e in self._elements])

    def tick(self, x: float | np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Shift new sample x in; return updated tap vector.

        The sample propagates through the chain: element 0 receives x,
        element 1 receives element 0's old value, etc.
        """
        x_val = float(x)
        for element in self._elements:
            x_val = float(element.tick(np.asarray(x_val), rng=rng))
        return self.taps

    def process(self, signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Shift an entire 1-D signal through, returning (len(signal), N) tap matrix.

        Row i of the result is the tap vector after sample i has been clocked in,
        giving the full history matrix needed for batch FIR computation.
        """
        signal = np.asarray(signal, dtype=np.float64).ravel()
        tap_matrix = np.zeros((len(signal), self.length), dtype=np.float64)
        for i, s in enumerate(signal):
            self.tick(s, rng=rng)
            tap_matrix[i] = self.taps
        return tap_matrix

    def reset(self) -> None:
        for e in self._elements:
            e.reset()
