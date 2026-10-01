"""Projective Gram and Grassmann primitives with explicit validity audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_complex_matrix(values: Any) -> NDArray[np.complex128]:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("analysis coefficients must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("analysis coefficients must be finite")
    return matrix


def projective_gram(
    coefficients: Any, *, energy_floor: float = np.finfo(np.float64).tiny
) -> NDArray[np.complex128]:
    """Return the normalized left Gram ``CC*/tr(CC*)``.

    The map is invariant to nonzero global complex scaling, but it is also
    invariant to every right-unitary transformation ``C -> C U``.  It is
    therefore a partial-trace/left-covariance summary, not an injective
    embedding of the global-scale projective quotient when ``C`` has more
    than one column.
    """

    matrix = _finite_complex_matrix(coefficients)
    if not np.isfinite(energy_floor) or energy_floor <= 0.0:
        raise ValueError("energy_floor must be finite and positive")
    gram = matrix @ matrix.conj().T
    energy = float(np.trace(gram).real)
    if not np.isfinite(energy) or energy <= energy_floor:
        raise ValueError("analysis coefficients have zero or unsafe energy")
    normalized = gram / energy
    # Remove round-off anti-Hermitian residue without changing the contract.
    return np.asarray(0.5 * (normalized + normalized.conj().T), dtype=np.complex128)


def hermitian_real_coordinates(matrix: Any) -> NDArray[np.float64]:
    """Encode an ``m x m`` Hermitian matrix in exactly ``m**2`` real values."""

    values = _finite_complex_matrix(matrix)
    if values.shape[0] != values.shape[1]:
        raise ValueError("Hermitian coordinates require a square matrix")
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.linalg.norm(values, ord="fro"))
    )
    if float(np.linalg.norm(values - values.conj().T, ord="fro")) > tolerance:
        raise ValueError("matrix is not Hermitian within numerical tolerance")
    diagonal = np.diag(values).real
    upper = np.triu_indices(values.shape[0], k=1)
    return np.concatenate((diagonal, values[upper].real, values[upper].imag)).astype(
        np.float64,
        copy=False,
    )


@dataclass(frozen=True)
class SubspaceAudit:
    projector: NDArray[np.complex128]
    singular_values: NDArray[np.float64]
    rank: int
    gap: float
    relative_gap: float
    qualified: bool
    reason: str


def dominant_left_subspace(
    coefficients: Any,
    rank: int,
    *,
    minimum_relative_gap: float = 0.0,
    minimum_energy: float = np.finfo(np.float64).tiny,
) -> SubspaceAudit:
    """Return a left-singular projector and an explicit spectral-gap verdict."""

    matrix = _finite_complex_matrix(coefficients)
    maximum_rank = min(matrix.shape)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must lie in [1,{maximum_rank}]")
    if minimum_relative_gap < 0.0 or not np.isfinite(minimum_relative_gap):
        raise ValueError("minimum_relative_gap must be finite and nonnegative")
    energy = float(np.linalg.norm(matrix, ord="fro") ** 2)
    if not np.isfinite(energy) or energy <= minimum_energy:
        raise ValueError("analysis coefficients have zero or unsafe energy")
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    next_value = float(singular_values[rank]) if rank < maximum_rank else 0.0
    gap = float(singular_values[rank - 1] - next_value)
    leading = float(singular_values[0])
    relative_gap = gap / max(leading, np.finfo(np.float64).tiny)
    qualified = bool(relative_gap >= minimum_relative_gap and gap > 0.0)
    projector = left[:, :rank] @ left[:, :rank].conj().T
    return SubspaceAudit(
        projector=np.asarray(projector, dtype=np.complex128),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        rank=int(rank),
        gap=gap,
        relative_gap=float(relative_gap),
        qualified=qualified,
        reason="QUALIFIED" if qualified else "DEFER_SPECTRAL_GAP",
    )


def projector_distance(first: Any, second: Any) -> float:
    """Normalized Frobenius distance between equal-rank Hermitian projectors."""

    left = _finite_complex_matrix(first)
    right = _finite_complex_matrix(second)
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("projectors must be square matrices with equal shape")
    left_rank = float(np.trace(left).real)
    right_rank = float(np.trace(right).real)
    if not np.isclose(left_rank, right_rank, atol=1e-9, rtol=1e-9):
        raise ValueError("projectors must have equal rank")
    rank = max(left_rank, np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right, ord="fro") / np.sqrt(2.0 * rank))
