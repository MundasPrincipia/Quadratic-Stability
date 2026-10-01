"""Source-native complex-linear front ends for passive LoRa records."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import hermitian_real_coordinates, projective_gram


def complex_unit_rows(values: Any) -> NDArray[np.complex128]:
    """Return finite nonzero rows normalized in complex Euclidean norm."""

    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must have shape (rows, complex_samples)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("values contain a zero or unsafe row")
    return np.asarray(matrix / norms, dtype=np.complex128)


def segmented_fourier_frontend(
    signals: Any,
    *,
    segment_count: int,
    coefficients_per_segment: int,
    normalize: bool = True,
) -> NDArray[np.complex128]:
    """Select uniformly spaced complex Fourier coefficients per segment.

    The operation is deterministic and complex-linear before optional row
    normalization.  Thus ``B(c x) = c B(x)`` for every complex scalar ``c``.
    """

    values = np.asarray(signals, dtype=np.complex128)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("signals must have shape (rows, complex_samples)")
    if not np.all(np.isfinite(values)):
        raise ValueError("signals must be finite")
    if segment_count <= 0 or values.shape[1] % segment_count:
        raise ValueError("segment_count must divide the signal length")
    segment_length = values.shape[1] // segment_count
    if coefficients_per_segment <= 0 or coefficients_per_segment > segment_length:
        raise ValueError("coefficients_per_segment is outside the segment length")
    if segment_length % coefficients_per_segment:
        raise ValueError(
            "segment length must be divisible by coefficients_per_segment"
        )
    segments = values.reshape(values.shape[0], segment_count, segment_length)
    spectrum = np.fft.fft(segments, axis=-1, norm="ortho")
    stride = segment_length // coefficients_per_segment
    indices = np.arange(coefficients_per_segment, dtype=np.int64) * stride
    output = np.asarray(spectrum[..., indices].reshape(values.shape[0], -1))
    return complex_unit_rows(output) if normalize else output


def complex_real_coordinates(values: Any) -> NDArray[np.float64]:
    """Encode complex rows as concatenated real and imaginary coordinates."""

    unit = complex_unit_rows(values)
    return np.concatenate((unit.real, unit.imag), axis=1).astype(
        np.float64, copy=False
    )


def left_gram_vector_features(
    values: Any, *, matrix_rows: int
) -> NDArray[np.float64]:
    """Apply the deliberately overquotienting left-Gram control to vectors."""

    unit = complex_unit_rows(values)
    if matrix_rows <= 0 or unit.shape[1] % matrix_rows:
        raise ValueError("matrix_rows must divide the complex feature dimension")
    matrices = unit.reshape(unit.shape[0], matrix_rows, -1)
    output = [
        hermitian_real_coordinates(projective_gram(matrix)) for matrix in matrices
    ]
    return np.stack(output).astype(np.float64, copy=False)
