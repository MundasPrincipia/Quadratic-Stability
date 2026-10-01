"""Finite quadratic measurements of exact complex-projective pure states."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProjectiveMeasurementBank:
    vectors: NDArray[np.complex128]
    seed: int
    input_dimension: int
    output_dimension: int
    sha256: str

    def prefix(self, output_dimension: int) -> "ProjectiveMeasurementBank":
        if output_dimension <= 0 or output_dimension > self.output_dimension:
            raise ValueError("prefix dimension is outside the frozen bank")
        values = np.asarray(self.vectors[:output_dimension], dtype=np.complex128).copy()
        values.setflags(write=False)
        return ProjectiveMeasurementBank(
            vectors=values,
            seed=self.seed,
            input_dimension=self.input_dimension,
            output_dimension=int(output_dimension),
            sha256=_measurement_digest(values, self.seed),
        )


def _measurement_digest(values: NDArray[np.complex128], seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"complex-spherical-v1:{int(seed)}:{values.shape}\n".encode("utf-8"))
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def projective_measurement_bank(
    input_dimension: int,
    output_dimension: int,
    *,
    seed: int,
) -> ProjectiveMeasurementBank:
    """Draw a deterministic label-independent complex spherical bank."""

    if input_dimension <= 0 or output_dimension <= 0:
        raise ValueError("measurement dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    values = rng.normal(size=(output_dimension, input_dimension)) + 1j * rng.normal(
        size=(output_dimension, input_dimension)
    )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).tiny):
        raise RuntimeError("generated an unsafe projective measurement vector")
    vectors = np.asarray(values / norms, dtype=np.complex128)
    vectors.setflags(write=False)
    return ProjectiveMeasurementBank(
        vectors=vectors,
        seed=int(seed),
        input_dimension=int(input_dimension),
        output_dimension=int(output_dimension),
        sha256=_measurement_digest(vectors, int(seed)),
    )


def projective_intensity_features(
    signals: Any,
    bank: ProjectiveMeasurementBank,
    *,
    chunk_size: int = 512,
) -> NDArray[np.float64]:
    """Measure ``P(z)=zz*`` through ``|a_l*z|^2`` coordinates.

    Rows are normalized internally, so the result is exactly invariant to a
    nonzero row-wise complex scale.  The deterministic factor
    ``sqrt(n(n+1)/m)`` gives an isotropic secant scale for complex spherical
    rank-one measurements; fold-local standardization remains downstream.
    """

    values = np.asarray(signals, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must have shape (rows, complex_dimension)")
    if values.shape[1] != bank.input_dimension:
        raise ValueError("signals do not match the frozen measurement bank")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("signals must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("signals contain a zero or unsafe row")
    unit = values / norms
    output = np.empty(
        (values.shape[0], bank.output_dimension), dtype=np.float64
    )
    scale = np.sqrt(
        bank.input_dimension
        * (bank.input_dimension + 1.0)
        / bank.output_dimension
    )
    for begin in range(0, values.shape[0], chunk_size):
        stop = min(begin + chunk_size, values.shape[0])
        response = unit[begin:stop] @ bank.vectors.conj().T
        output[begin:stop] = scale * np.abs(response) ** 2
    if not np.all(np.isfinite(output)):
        raise RuntimeError("projective intensity sketch produced non-finite values")
    return output
