"""Exact state--task interface for the ordered residual-unitary quotient."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .passive_quotient.ordered_filtration import discrete_polynomial_basis


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def _basis(time_length: int, basis: Any | None) -> ComplexArray:
    length = int(time_length)
    ordered = (
        discrete_polynomial_basis(length).astype(np.complex128)
        if basis is None
        else np.asarray(basis, dtype=np.complex128)
    )
    if ordered.shape != (length, length):
        raise ValueError("basis has the wrong shape")
    error = float(np.linalg.norm(ordered.conj().T @ ordered - np.eye(length)))
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError("basis must be unitary")
    return ordered


def normalized_adapted_matrices(
    values: Any,
    matrix_rows: int,
    time_length: int,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Return normalized K by T row-major matrices in the ordered basis."""

    rows = int(matrix_rows)
    length = int(time_length)
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 1:
        array = array[None, :]
    if (
        rows < 1
        or length < 1
        or array.ndim != 2
        or array.shape[1] != rows * length
    ):
        raise ValueError("values have the wrong row-major dimensions")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("values must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("values must be nonzero")
    matrices = (array / norms[:, None]).reshape(array.shape[0], rows, length)
    return np.asarray(matrices @ _basis(length, basis).conj(), dtype=np.complex128)


def ordered_hilbert_action(
    time_length: int,
    resolved_dimension: int,
    residual_unitary: Any,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Return the physical temporal Hilbert action for a complex flag.

    If the columns of ``basis`` are the flag-adapted directions and
    ``D = I_q direct-sum U``, the returned action is ``basis @ D @ basis*``.
    This definition avoids identifying a complex flag with its conjugate under
    the transpose appearing in row-major vectorization.
    """

    length = int(time_length)
    resolved = int(resolved_dimension)
    residual = length - resolved
    if length < 1 or resolved < 0 or resolved > length:
        raise ValueError("ordered dimensions are invalid")
    unitary = np.asarray(residual_unitary, dtype=np.complex128)
    if unitary.shape != (residual, residual):
        raise ValueError("residual_unitary has the wrong shape")
    if residual:
        error = float(
            np.linalg.norm(unitary.conj().T @ unitary - np.eye(residual))
        )
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError("residual_unitary must be unitary")
    adapted = np.eye(length, dtype=np.complex128)
    if residual:
        adapted[resolved:, resolved:] = unitary
    ordered = _basis(length, basis)
    return np.asarray(ordered @ adapted @ ordered.conj().T, dtype=np.complex128)


def ordered_matrix_right_action(
    matrices: Any,
    resolved_dimension: int,
    residual_unitary: Any,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Apply the matrix action corresponding to ``ordered_hilbert_action``.

    For a flag basis ``B`` and ``D = I_q direct-sum U``, row-major matrices
    transform as ``C -> C conj(B) D^T B^T``.  Consequently
    ``vec_r(C') = (I_K tensor B D B*) vec_r(C)``.
    """

    value = np.asarray(matrices, dtype=np.complex128)
    if value.ndim < 2:
        raise ValueError("matrices must have at least two dimensions")
    length = int(value.shape[-1])
    resolved = int(resolved_dimension)
    residual = length - resolved
    unitary = np.asarray(residual_unitary, dtype=np.complex128)
    if resolved < 0 or resolved > length or unitary.shape != (residual, residual):
        raise ValueError("ordered action dimensions are invalid")
    action = ordered_hilbert_action(
        length, resolved, unitary, basis=basis
    )
    return np.asarray(value @ action.T, dtype=np.complex128)


def ordered_conditional_expectation_adapted(
    operator: Any,
    matrix_rows: int,
    time_length: int,
    resolved_dimension: int,
) -> ComplexArray:
    """Apply the ordered conditional expectation in its adapted basis."""

    rows = int(matrix_rows)
    length = int(time_length)
    resolved = int(resolved_dimension)
    dimension = rows * length
    value = np.asarray(operator, dtype=np.complex128)
    if (
        rows < 1
        or length < 1
        or resolved < 0
        or resolved > length
        or value.shape != (dimension, dimension)
    ):
        raise ValueError("operator or ordered dimensions are invalid")
    if not np.all(np.isfinite(value.real)) or not np.all(np.isfinite(value.imag)):
        raise ValueError("operator must be finite")
    if resolved == length:
        return value.copy()
    residual = length - resolved
    tensor = value.reshape(rows, length, rows, length)
    output = np.zeros_like(tensor)
    if resolved:
        output[:, :resolved, :, :resolved] = tensor[
            :, :resolved, :, :resolved
        ]
    partial = np.einsum(
        "krlr->kl",
        tensor[:, resolved:, :, resolved:],
        optimize=True,
    )
    identity = np.eye(residual, dtype=np.complex128) / residual
    output[:, resolved:, :, resolved:] = np.einsum(
        "kl,rs->krls", partial, identity, optimize=True
    )
    return np.asarray(output.reshape(dimension, dimension), dtype=np.complex128)


def pairwise_task_operators_adapted(
    measurement_vectors: Any,
    pair_weights: Any,
    feature_bound: float,
    matrix_rows: int,
    time_length: int,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Construct H_cd = B sum_i v_cd,i |a_i><a_i| in the ordered basis."""

    bank = normalized_adapted_matrices(
        measurement_vectors,
        matrix_rows,
        time_length,
        basis=basis,
    ).reshape(np.asarray(measurement_vectors).shape[0], -1)
    weights = np.asarray(pair_weights, dtype=np.float64)
    bound = float(feature_bound)
    if weights.ndim != 2 or weights.shape[1] != bank.shape[0]:
        raise ValueError("pair_weights and measurement bank disagree")
    if not np.all(np.isfinite(weights)) or not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("pair weights and feature bound must be finite")
    operators = bound * np.einsum(
        "pm,mi,mj->pij", weights, bank, bank.conj(), optimize=True
    )
    return np.asarray(
        (operators + operators.conj().transpose(0, 2, 1)) / 2.0,
        dtype=np.complex128,
    )


def ordered_projected_measurement_features(
    values: Any,
    measurement_vectors: Any,
    feature_bound: float,
    matrix_rows: int,
    time_length: int,
    resolved_dimension: int,
    *,
    basis: Any | None = None,
) -> RealArray:
    """Evaluate B tr(P_i E_q(rho)) for every state and bank projector."""

    rows = int(matrix_rows)
    length = int(time_length)
    resolved = int(resolved_dimension)
    if resolved < 0 or resolved > length:
        raise ValueError("resolved_dimension must lie in {0,...,T}")
    states = normalized_adapted_matrices(
        values, rows, length, basis=basis
    )
    bank = normalized_adapted_matrices(
        measurement_vectors, rows, length, basis=basis
    )
    output = np.zeros((states.shape[0], bank.shape[0]), dtype=np.float64)
    if resolved:
        inner = np.einsum(
            "mkt,nkt->nm",
            bank[:, :, :resolved].conj(),
            states[:, :, :resolved],
            optimize=True,
        )
        output += np.abs(inner) ** 2
    residual = length - resolved
    if residual:
        state_tail = states[:, :, resolved:]
        bank_tail = bank[:, :, resolved:]
        state_gram = np.einsum(
            "nkt,nlt->nkl", state_tail, state_tail.conj(), optimize=True
        )
        bank_gram = np.einsum(
            "mkt,mlt->mkl", bank_tail, bank_tail.conj(), optimize=True
        )
        residual_value = (
            state_gram.reshape(states.shape[0], -1)
            @ bank_gram.reshape(bank.shape[0], -1).conj().T
        ).real / residual
        output += residual_value
    bound = float(feature_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("feature_bound must be finite and positive")
    return np.asarray(bound * output, dtype=np.float64)


def ordered_task_obstruction(
    pair_operators_adapted: Any,
    matrix_rows: int,
    time_length: int,
    resolved_dimension: int,
    *,
    compute_operator_norm: bool = True,
) -> dict[str, Any]:
    """Project pair operators to the fixed algebra and quantify the residual."""

    operators = np.asarray(pair_operators_adapted, dtype=np.complex128)
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("pair_operators_adapted must be a stack of square matrices")
    hermitian_error = np.linalg.norm(
        operators - operators.conj().transpose(0, 2, 1), axis=(1, 2)
    )
    hermitian_scale = np.maximum(1.0, np.linalg.norm(operators, axis=(1, 2)))
    if np.any(hermitian_error > 256.0 * np.finfo(float).eps * hermitian_scale):
        raise ValueError("pair_operators_adapted must be Hermitian")
    projected = np.empty_like(operators)
    for index, operator in enumerate(operators):
        projected[index] = ordered_conditional_expectation_adapted(
            operator,
            matrix_rows,
            time_length,
            resolved_dimension,
        )
    residual = np.asarray(operators - projected, dtype=np.complex128)
    residual = np.asarray(
        (residual + residual.conj().transpose(0, 2, 1)) / 2.0,
        dtype=np.complex128,
    )
    frobenius = np.linalg.norm(residual, axis=(1, 2))
    total = np.linalg.norm(operators, axis=(1, 2))
    relative = np.divide(
        frobenius,
        total,
        out=np.zeros_like(frobenius),
        where=total > 0.0,
    )
    if compute_operator_norm:
        eigenvalues = np.linalg.eigvalsh(residual)
        operator_norm = np.max(np.abs(eigenvalues), axis=1)
    else:
        eigenvalues = np.zeros((operators.shape[0], 0), dtype=np.float64)
        operator_norm = np.full(operators.shape[0], np.nan, dtype=np.float64)
    scale = np.maximum(1.0, total)
    fixed = frobenius <= 256.0 * np.finfo(np.float64).eps * scale
    return {
        "projected_operators": projected,
        "residual_operators": residual,
        "frobenius_radius": np.asarray(frobenius, dtype=np.float64),
        "operator_radius": np.asarray(operator_norm, dtype=np.float64),
        "relative_frobenius_obstruction": np.asarray(relative, dtype=np.float64),
        "fixed_pairwise_operator": np.asarray(fixed, dtype=bool),
        "residual_eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
    }
