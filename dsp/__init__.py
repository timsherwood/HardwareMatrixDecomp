"""Delay-space DSP module.

Implements digital signal processing primitives where signal delay (z^{-1})
is the fundamental computational element. A tapped delay line naturally
represents FIR convolution; recursive feedback gives IIR; a bank of
bandpass delay lines gives spectral analysis.

The Toeplitz matrices arising from FIR convolution can be factored via the
SVDDecomposition in the parent hardware_matrix_decomp package, mapping delay-
space computation directly onto the tile hardware.

Core classes:
    DelayElement     — single z^{-1} register
    TappedDelayLine  — shift-register exposing all N taps
    FIRFilter        — weighted tap sum over a TappedDelayLine
    IIRFilter        — FIR + recursive feedback taps
    Filterbank       — parallel array of bandpass FIR filters
    toeplitz_matrix  — build the convolution matrix for a set of coefficients
"""

from dsp.delay_line import DelayElement, TappedDelayLine
from dsp.filter import FIRFilter, IIRFilter
from dsp.transform import Filterbank, toeplitz_matrix

__all__ = [
    "DelayElement",
    "TappedDelayLine",
    "FIRFilter",
    "IIRFilter",
    "Filterbank",
    "toeplitz_matrix",
]
