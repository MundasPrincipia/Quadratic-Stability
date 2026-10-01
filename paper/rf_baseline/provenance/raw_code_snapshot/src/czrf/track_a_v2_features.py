"""Deterministic communication-signal features for Track A V2 audits.

The functions in this module are deliberately label-free.  They operate on
aligned LoRa preamble chirps and summarize residual amplitude, instantaneous
frequency, and dechirped spectrum after removing the ideal modulation and a
per-chirp complex gain / dominant tone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct, fft, fftshift


DEFAULT_SAMPLE_RATE_HZ = 1_000_000
DEFAULT_BANDWIDTH_HZ = 125_000
DEFAULT_SYMBOL_SAMPLES = 1024
DEFAULT_SEGMENTS_PER_FRAME = 8
RESIDUAL_COMPONENT_DIMENSION = 64
RESIDUAL_SEGMENT_DIMENSION = 3 * RESIDUAL_COMPONENT_DIMENSION
RESIDUAL_FRAME_DIMENSION = 2 * RESIDUAL_SEGMENT_DIMENSION


def ideal_upchirp(
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    bandwidth_hz: int = DEFAULT_BANDWIDTH_HZ,
    symbol_samples: int = DEFAULT_SYMBOL_SAMPLES,
) -> NDArray[np.complex128]:
    """Generate the label-free ideal baseband LoRa upchirp reference."""

    if sample_rate_hz <= 0 or bandwidth_hz <= 0 or symbol_samples <= 0:
        raise ValueError("sample rate, bandwidth, and symbol length must be positive")
    sample = np.arange(symbol_samples, dtype=np.float64)
    time = sample / float(sample_rate_hz)
    duration = symbol_samples / float(sample_rate_hz)
    phase = 2.0 * np.pi * (
        -0.5 * float(bandwidth_hz) * time
        + 0.5 * float(bandwidth_hz) / duration * time**2
    )
    return np.exp(1j * phase)


def _validated_frames(
    frame_iq: NDArray[np.complexfloating],
    *,
    segments_per_frame: int,
    symbol_samples: int,
) -> NDArray[np.complex128]:
    values = np.asarray(frame_iq)
    expected = (segments_per_frame, symbol_samples)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1:] != expected:
        raise ValueError(
            "frame_iq must have shape "
            f"(frames,{segments_per_frame},{symbol_samples})"
        )
    if not np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("frame_iq must be finite complex data")
    energy = np.mean(np.abs(values) ** 2, axis=-1)
    if np.any(energy <= np.finfo(np.float64).tiny):
        raise ValueError("frame_iq contains a zero-energy chirp")
    return np.asarray(values, dtype=np.complex128)


def _segment_features(
    values: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    *,
    tone_fft_length: int,
) -> NDArray[np.float64]:
    """Return 192 residual features for a batch of aligned chirps."""

    symbol_samples = values.shape[1]
    rms = np.sqrt(np.mean(np.abs(values) ** 2, axis=1, keepdims=True))
    normalized = values / rms
    dechirped = normalized * np.conj(reference)[None, :]

    tone_spectrum = np.abs(fft(dechirped, n=tone_fft_length, axis=1)) ** 2
    peak = np.argmax(tone_spectrum, axis=1)
    signed_peak = np.where(peak <= tone_fft_length // 2, peak, peak - tone_fft_length)
    cycles_per_sample = signed_peak.astype(np.float64) / float(tone_fft_length)
    sample = np.arange(symbol_samples, dtype=np.float64)
    residual = dechirped * np.exp(
        -2j * np.pi * cycles_per_sample[:, None] * sample[None, :]
    )

    phase_reference = np.sum(residual, axis=1)
    safe_phase = np.where(
        np.abs(phase_reference) > np.finfo(np.float64).tiny,
        np.angle(phase_reference),
        0.0,
    )
    residual *= np.exp(-1j * safe_phase)[:, None]

    floor = np.finfo(np.float64).eps
    log_amplitude = np.log(np.maximum(np.abs(residual), floor))
    log_amplitude -= np.mean(log_amplitude, axis=1, keepdims=True)
    amplitude_coefficients = dct(log_amplitude, type=2, norm="ortho", axis=1)[
        :, 1 : RESIDUAL_COMPONENT_DIMENSION + 1
    ]

    instantaneous_frequency = np.angle(residual[:, 1:] * np.conj(residual[:, :-1]))
    instantaneous_frequency -= np.median(
        instantaneous_frequency, axis=1, keepdims=True
    )
    frequency_coefficients = dct(
        instantaneous_frequency, type=2, norm="ortho", axis=1
    )[:, 1 : RESIDUAL_COMPONENT_DIMENSION + 1]

    residual_spectrum = fftshift(fft(residual, n=symbol_samples, axis=1), axes=1)
    log_power = np.log(np.maximum(np.abs(residual_spectrum) ** 2, floor))
    center = symbol_samples // 2
    half = RESIDUAL_COMPONENT_DIMENSION // 2
    centered_power = log_power[:, center - half : center + half]
    centered_power -= np.mean(centered_power, axis=1, keepdims=True)

    output = np.concatenate(
        (amplitude_coefficients, frequency_coefficients, centered_power), axis=1
    )
    if output.shape != (values.shape[0], RESIDUAL_SEGMENT_DIMENSION):
        raise RuntimeError("unexpected residual segment feature shape")
    return output


def dechirped_residual_frame_features(
    frame_iq: NDArray[np.complexfloating],
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    bandwidth_hz: int = DEFAULT_BANDWIDTH_HZ,
    symbol_samples: int = DEFAULT_SYMBOL_SAMPLES,
    segments_per_frame: int = DEFAULT_SEGMENTS_PER_FRAME,
    tone_fft_length: int = 4096,
    frame_batch_size: int = 32,
) -> NDArray[np.float32]:
    """Extract a 384D mean/std residual summary per multi-chirp frame.

    All operations are deterministic and do not depend on device labels.  A
    frame is summarized by the mean and population standard deviation of its
    eight 192D per-chirp residual feature vectors.
    """

    if tone_fft_length < symbol_samples or tone_fft_length % symbol_samples:
        raise ValueError("tone_fft_length must be a multiple of symbol_samples")
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    values = _validated_frames(
        frame_iq,
        segments_per_frame=segments_per_frame,
        symbol_samples=symbol_samples,
    )
    reference = ideal_upchirp(
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=bandwidth_hz,
        symbol_samples=symbol_samples,
    )
    output = np.empty((values.shape[0], RESIDUAL_FRAME_DIMENSION), dtype=np.float32)
    for begin in range(0, values.shape[0], frame_batch_size):
        stop = min(begin + frame_batch_size, values.shape[0])
        local = values[begin:stop]
        segment = _segment_features(
            local.reshape(-1, symbol_samples),
            reference,
            tone_fft_length=tone_fft_length,
        ).reshape(stop - begin, segments_per_frame, RESIDUAL_SEGMENT_DIMENSION)
        frame_mean = np.mean(segment, axis=1)
        frame_std = np.std(segment, axis=1, ddof=0)
        output[begin:stop] = np.concatenate((frame_mean, frame_std), axis=1).astype(
            np.float32
        )
    if output.shape != (values.shape[0], RESIDUAL_FRAME_DIMENSION):
        raise RuntimeError("unexpected residual frame feature shape")
    if not np.all(np.isfinite(output)):
        raise RuntimeError("residual feature extraction produced non-finite values")
    return output

