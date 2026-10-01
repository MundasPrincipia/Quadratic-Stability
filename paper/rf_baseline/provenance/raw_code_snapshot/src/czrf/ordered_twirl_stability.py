"""Global quotient metrics and stability bounds for ordered Haar twirls."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .passive_quotient.ordered_filtration import discrete_polynomial_basis


ComplexArray = NDArray[np.complex128]


def _hermitian(value: Any) -> ComplexArray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("value must be a square matrix")
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError("value must be finite")
    return np.asarray((matrix + matrix.conj().T) / 2.0, dtype=np.complex128)


def psd_square_root(value: Any, *, tolerance: float = 1e-12) -> ComplexArray:
    """Return the Hermitian PSD square root with a fail-closed PSD check."""

    matrix = _hermitian(value)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.min(eigenvalues, initial=0.0)) < -float(tolerance) * scale:
        raise ValueError("value is not positive semidefinite within tolerance")
    clipped = np.maximum(eigenvalues, 0.0)
    return np.asarray(
        (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T,
        dtype=np.complex128,
    )


def bures_squared_psd(
    first: Any,
    second: Any,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Generalized Bures/Procrustes distance squared for unnormalized PSDs."""

    left = _hermitian(first)
    right = _hermitian(second)
    if left.shape != right.shape:
        raise ValueError("PSD matrices must have the same shape")
    left_root = psd_square_root(left, tolerance=tolerance)
    right_root = psd_square_root(right, tolerance=tolerance)
    # Evaluate the equivalent Procrustes residual directly.  The tempting
    # trace formula tr(A)+tr(B)-2||sqrt(A)sqrt(B)||_* loses all relative
    # accuracy near zero and would create an artificial sqrt(eps) metric
    # dead zone.  Direct alignment retains genuine arbitrarily small
    # distances (up to the accuracy of the PSD square roots themselves).
    left_singular, _, right_h = np.linalg.svd(
        left_root.conj().T @ right_root,
        full_matrices=True,
    )
    unitary = right_h.conj().T @ left_singular.conj().T
    residual = left_root - right_root @ unitary
    value = float(np.linalg.norm(residual, ord="fro") ** 2)
    if not np.isfinite(value):
        raise RuntimeError("Bures square became nonfinite")
    return value


def ordered_twirl_blocks(
    state: Any,
    matrix_rows: int,
    time_length: int,
    level: int,
    *,
    basis: Any | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the resolved rank-one block and unscaled residual left Gram."""

    rows = int(matrix_rows)
    length = int(time_length)
    selected = int(level)
    if rows < 1 or length < 1 or selected < -1 or selected >= length:
        raise ValueError("invalid tensor dimensions or level")
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size != rows * length or not np.all(np.isfinite(vector)):
        raise ValueError("state has the wrong size or is nonfinite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("state must be nonzero")
    coefficients = (vector / norm).reshape(rows, length)
    ordered = (
        discrete_polynomial_basis(length).astype(np.complex128)
        if basis is None
        else np.asarray(basis, dtype=np.complex128)
    )
    if ordered.shape != (length, length):
        raise ValueError("basis has the wrong shape")
    gram_error = float(
        np.linalg.norm(ordered.conj().T @ ordered - np.eye(length), ord=2)
    )
    if not np.isfinite(gram_error) or gram_error > 1e-10:
        raise ValueError("basis must be unitary")
    adapted = coefficients @ ordered.conj()
    resolved_dimension = selected + 1
    resolved = adapted[:, :resolved_dimension].reshape(-1)
    residual = adapted[:, resolved_dimension:]
    resolved_block = np.outer(resolved, resolved.conj())
    residual_gram = residual @ residual.conj().T
    return (
        np.asarray(resolved_block, dtype=np.complex128),
        np.asarray(residual_gram, dtype=np.complex128),
    )


def ordered_twirl_output_frobenius_from_blocks(
    first_blocks: tuple[Any, Any],
    second_blocks: tuple[Any, Any],
    *,
    residual_dimension: int,
) -> float:
    """Compute the Frobenius output distance from invariant blocks."""

    first_resolved = _hermitian(first_blocks[0])
    second_resolved = _hermitian(second_blocks[0])
    first_gram = _hermitian(first_blocks[1])
    second_gram = _hermitian(second_blocks[1])
    if first_resolved.shape != second_resolved.shape:
        raise ValueError("resolved blocks have different shapes")
    if first_gram.shape != second_gram.shape:
        raise ValueError("residual Gram matrices have different shapes")
    residual = int(residual_dimension)
    resolved_square = float(
        np.linalg.norm(first_resolved - second_resolved, ord="fro") ** 2
    )
    gram_square = float(np.linalg.norm(first_gram - second_gram, ord="fro") ** 2)
    if residual == 0:
        if gram_square > 1e-20:
            raise ValueError("zero residual dimension requires zero residual Gram")
        return float(np.sqrt(resolved_square))
    if residual < 0:
        raise ValueError("residual_dimension must be nonnegative")
    return float(np.sqrt(resolved_square + gram_square / residual))


def ordered_twirl_quotient_vector_distance_from_blocks(
    first_blocks: tuple[Any, Any],
    second_blocks: tuple[Any, Any],
    *,
    tolerance: float = 1e-12,
) -> float:
    """Exact projective residual-unitary quotient distance between factors."""

    resolved = bures_squared_psd(
        first_blocks[0], second_blocks[0], tolerance=tolerance
    )
    residual = bures_squared_psd(
        first_blocks[1], second_blocks[1], tolerance=tolerance
    )
    return float(np.sqrt(max(0.0, resolved + residual)))


def ordered_twirl_holder_constant(
    matrix_rows: int,
    time_length: int,
    level: int,
) -> float:
    """Return kappa in d_quotient <= kappa * sqrt(output Frobenius distance)."""

    rows = int(matrix_rows)
    length = int(time_length)
    selected = int(level)
    if rows < 1 or length < 1 or selected < -1 or selected >= length:
        raise ValueError("invalid tensor dimensions or level")
    resolved = selected + 1
    residual = length - resolved
    residual_rank = min(rows, residual)
    coefficient_square = 0.0
    if resolved:
        coefficient_square += 2.0
    if residual:
        coefficient_square += 2.0 * residual_rank * residual
    return float(coefficient_square ** 0.25)


def ordered_twirl_holder_bound(
    output_frobenius_distance: float,
    matrix_rows: int,
    time_length: int,
    level: int,
) -> float:
    """Evaluate the global half-Hölder quotient-distance certificate."""

    distance = float(output_frobenius_distance)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("output distance must be finite and nonnegative")
    return float(
        ordered_twirl_holder_constant(matrix_rows, time_length, level)
        * np.sqrt(distance)
    )
