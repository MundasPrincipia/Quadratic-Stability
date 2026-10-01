"""Fold-local affine transports in a common finite-dimensional coordinate space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_rows(values: Any) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("transport data must have at least two finite rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transport data must be finite")
    return matrix


@dataclass(frozen=True)
class AffineTransport:
    linear: NDArray[np.float64]
    source_mean: NDArray[np.float64]
    target_mean: NDArray[np.float64]
    method: str
    regularization: float

    def transform(self, values: Any) -> NDArray[np.float64]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.linear.shape[0]:
            raise ValueError("values do not match the transport dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must be finite")
        return np.asarray(
            (matrix - self.source_mean) @ self.linear + self.target_mean,
            dtype=np.float64,
        )


def _paired(source: Any, target: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left = _finite_rows(source)
    right = _finite_rows(target)
    if left.shape != right.shape:
        raise ValueError("paired transport centroids must have equal shape")
    return left, right


def identity_transport(dimension: int) -> AffineTransport:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    zero = np.zeros(dimension, dtype=np.float64)
    return AffineTransport(
        linear=np.eye(dimension),
        source_mean=zero,
        target_mean=zero,
        method="no_alignment",
        regularization=0.0,
    )


def mean_shift_transport(source: Any, target: Any) -> AffineTransport:
    left, right = _paired(source, target)
    return AffineTransport(
        linear=np.eye(left.shape[1]),
        source_mean=np.mean(left, axis=0),
        target_mean=np.mean(right, axis=0),
        method="mean_shift",
        regularization=0.0,
    )


def orthogonal_procrustes_transport(source: Any, target: Any) -> AffineTransport:
    left, right = _paired(source, target)
    source_mean = np.mean(left, axis=0)
    target_mean = np.mean(right, axis=0)
    cross = (left - source_mean).T @ (right - target_mean)
    u_matrix, _, vh_matrix = np.linalg.svd(cross, full_matrices=False)
    linear = u_matrix @ vh_matrix
    return AffineTransport(
        linear=np.asarray(linear, dtype=np.float64),
        source_mean=source_mean,
        target_mean=target_mean,
        method="orthogonal_procrustes",
        regularization=0.0,
    )


def _symmetric_power(matrix: NDArray[np.float64], exponent: float) -> NDArray[np.float64]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if np.any(values <= 0.0):
        raise ValueError("regularized covariance is not positive definite")
    return np.asarray(
        (vectors * np.power(values, exponent)[None, :]) @ vectors.T,
        dtype=np.float64,
    )


def coral_transport(
    source: Any, target: Any, *, relative_regularization: float = 1.0e-3
) -> AffineTransport:
    left, right = _paired(source, target)
    if relative_regularization <= 0.0 or not np.isfinite(relative_regularization):
        raise ValueError("relative_regularization must be finite and positive")
    source_mean = np.mean(left, axis=0)
    target_mean = np.mean(right, axis=0)
    x = left - source_mean
    y = right - target_mean
    source_covariance = x.T @ x / max(1, x.shape[0] - 1)
    target_covariance = y.T @ y / max(1, y.shape[0] - 1)
    scale = max(
        float(np.trace(source_covariance + target_covariance))
        / (2.0 * left.shape[1]),
        np.finfo(np.float64).tiny,
    )
    regularization = relative_regularization * scale
    identity = np.eye(left.shape[1])
    source_covariance += regularization * identity
    target_covariance += regularization * identity
    linear = _symmetric_power(source_covariance, -0.5) @ _symmetric_power(
        target_covariance, 0.5
    )
    return AffineTransport(
        linear=linear,
        source_mean=source_mean,
        target_mean=target_mean,
        method="coral",
        regularization=float(regularization),
    )


def restricted_ridge_transport(
    source: Any, target: Any, *, relative_regularization: float = 0.1
) -> AffineTransport:
    left, right = _paired(source, target)
    if relative_regularization <= 0.0 or not np.isfinite(relative_regularization):
        raise ValueError("relative_regularization must be finite and positive")
    source_mean = np.mean(left, axis=0)
    target_mean = np.mean(right, axis=0)
    x = left - source_mean
    y = right - target_mean
    gram = x.T @ x
    scale = max(
        float(np.trace(gram)) / left.shape[1], np.finfo(np.float64).tiny
    )
    regularization = relative_regularization * scale
    linear = np.linalg.solve(
        gram + regularization * np.eye(left.shape[1]), x.T @ y
    )
    return AffineTransport(
        linear=np.asarray(linear, dtype=np.float64),
        source_mean=source_mean,
        target_mean=target_mean,
        method="restricted_ridge",
        regularization=float(regularization),
    )


def normalized_transport_residual(
    prediction: Any, target: Any
) -> float:
    predicted = _finite_rows(prediction)
    expected = _finite_rows(target)
    if predicted.shape != expected.shape:
        raise ValueError("residual arrays must have equal shape")
    denominator = max(
        float(np.linalg.norm(expected - np.mean(expected, axis=0), ord="fro")),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(predicted - expected, ord="fro") / denominator)
