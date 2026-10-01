"""Deterministic short-time harmonic coefficients for passive RF records."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import hermitian_real_coordinates, projective_gram


def _finite_signal(signal: Any) -> NDArray[np.complex128]:
    values = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be finite and nonempty")
    if float(np.linalg.norm(values)) <= np.finfo(np.float64).tiny:
        raise ValueError("signal must have nonzero energy")
    return values


def gabor_coefficient_matrix(
    signal: Any,
    *,
    window_length: int = 64,
    hop_length: int = 32,
    frequency_bins: int = 16,
    reference: Any | None = None,
    remove_dominant_tone: bool = True,
) -> NDArray[np.complex128]:
    """Return centered short-time Fourier coefficients as ``bins x frames``.

    If ``reference`` is supplied, the signal is first dechirped.  Dominant-tone
    removal uses only the peak index, so nonzero global complex scaling leaves
    the selected index and the final projective Gram unchanged.
    """

    values = _finite_signal(signal)
    if reference is not None:
        template = _finite_signal(reference)
        if template.shape != values.shape:
            raise ValueError("reference must have the same length as signal")
        values = values * template.conj()
    if window_length <= 1 or window_length > values.size:
        raise ValueError("window_length must lie between 2 and signal length")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if frequency_bins <= 0 or frequency_bins > window_length:
        raise ValueError("frequency_bins must lie in [1, window_length]")
    if remove_dominant_tone:
        tone_fft_length = int(4 * values.size)
        peak = int(np.argmax(np.abs(np.fft.fft(values, n=tone_fft_length)) ** 2))
        cycles_per_sample = peak / float(tone_fft_length)
        sample = np.arange(values.size, dtype=np.float64)
        values = values * np.exp(-2j * np.pi * cycles_per_sample * sample)
    starts = np.arange(0, values.size - window_length + 1, hop_length, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("no complete analysis window is available")
    window = np.hanning(window_length).astype(np.float64)
    frames = np.stack([values[start : start + window_length] * window for start in starts])
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    center = window_length // 2
    begin = center - frequency_bins // 2
    stop = begin + frequency_bins
    if begin < 0 or stop > window_length:
        raise RuntimeError("centered frequency selection is out of range")
    return np.asarray(spectrum[:, begin:stop].T, dtype=np.complex128)


def harmonic_projective_features(
    signals: Any,
    **analysis_kwargs: Any,
) -> NDArray[np.float64]:
    values = np.asarray(signals)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must have shape (rows, samples)")
    rows = [
        hermitian_real_coordinates(
            projective_gram(gabor_coefficient_matrix(signal, **analysis_kwargs))
        )
        for signal in values
    ]
    return np.stack(rows).astype(np.float64, copy=False)


def harmonic_complex_unit_features(
    signals: Any,
    **analysis_kwargs: Any,
) -> NDArray[np.float64]:
    """Flatten complex coefficients and apply only global L2 normalization."""

    values = np.asarray(signals)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must have shape (rows, samples)")
    output = []
    for signal in values:
        coefficients = gabor_coefficient_matrix(signal, **analysis_kwargs).reshape(-1)
        norm = float(np.linalg.norm(coefficients))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise ValueError("harmonic complex feature has zero or unsafe norm")
        unit = coefficients / norm
        output.append(np.concatenate((unit.real, unit.imag)))
    return np.stack(output).astype(np.float64, copy=False)


def harmonic_magnitude_features(
    signals: Any,
    **analysis_kwargs: Any,
) -> NDArray[np.float64]:
    values = np.asarray(signals)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must have shape (rows, samples)")
    output = []
    for signal in values:
        coefficients = np.abs(gabor_coefficient_matrix(signal, **analysis_kwargs)).reshape(-1)
        norm = float(np.linalg.norm(coefficients))
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise ValueError("harmonic magnitude feature has zero or unsafe norm")
        output.append(coefficients / norm)
    return np.stack(output).astype(np.float64, copy=False)


def segmented_harmonic_coefficient_matrix(
    segments: Any,
    *,
    frequency_bins: int = 16,
    reference: Any | None = None,
    remove_dominant_tone: bool = True,
) -> NDArray[np.complex128]:
    """Analyze aligned segments as columns of one harmonic coefficient matrix."""

    values = np.asarray(segments, dtype=np.complex128)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("segments must have shape (segments, samples)")
    if not np.all(np.isfinite(values)):
        raise ValueError("segments must be finite")
    segment_count, sample_count = values.shape
    if frequency_bins <= 0 or frequency_bins > sample_count:
        raise ValueError("frequency_bins must lie in [1, samples]")
    if reference is not None:
        template = _finite_signal(reference)
        if template.shape != (sample_count,):
            raise ValueError("reference must match one segment")
        values = values * template.conj()[None, :]
    sample = np.arange(sample_count, dtype=np.float64)
    window = np.hanning(sample_count).astype(np.float64)
    spectra = np.empty((segment_count, frequency_bins), dtype=np.complex128)
    center = sample_count // 2
    begin = center - frequency_bins // 2
    stop = begin + frequency_bins
    for index, segment in enumerate(values):
        local = segment
        if remove_dominant_tone:
            tone_fft_length = 4 * sample_count
            peak = int(np.argmax(np.abs(np.fft.fft(local, n=tone_fft_length)) ** 2))
            local = local * np.exp(
                -2j * np.pi * (peak / float(tone_fft_length)) * sample
            )
        spectrum = np.fft.fftshift(np.fft.fft(local * window))
        spectra[index] = spectrum[begin:stop]
    return np.asarray(spectra.T, dtype=np.complex128)
