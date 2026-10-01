r"""Mechanism controls for the Track C V1.1 projective-quotient audit.

The routines in this module keep three objects separate:

* the exact projective state of a physically ordered ``K x T`` matrix;
* its right-unitary conditional expectation (the normalized left Gram);
* finite quadratic measurements of either object.

All vectorization is NumPy row-major.  For a matrix ``C`` this means that
``C.reshape(-1)`` identifies ``C^K \otimes C^T`` and right multiplication by
``U`` acts as ``I_K \otimes U.T``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .frontends import complex_unit_rows
from .sketch import ProjectiveMeasurementBank


def right_action_antirepresentation(
    unitary: Any, *, matrix_rows: int
) -> NDArray[np.complex128]:
    """Return ``I_K kron U.T`` for the row-major right matrix action.

    This map is an anti-representation: ``rho(U @ V) = rho(V) @ rho(U)``.
    The terminology matters even though transposition preserves Haar measure
    and therefore leaves the conditional-expectation integral unchanged.
    """

    values = np.asarray(unitary, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("unitary must be a square matrix")
    if matrix_rows <= 0:
        raise ValueError("matrix_rows must be positive")
    identity = np.eye(int(matrix_rows), dtype=np.complex128)
    return np.asarray(np.kron(identity, values.T), dtype=np.complex128)


def partial_trace_second_factor(
    operators: Any,
    *,
    first_dimension: int,
    second_dimension: int,
) -> NDArray[np.complex128]:
    """Trace operators over the second factor of ``C^K tensor C^T``."""

    values = np.asarray(operators, dtype=np.complex128)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    total = int(first_dimension) * int(second_dimension)
    if first_dimension <= 0 or second_dimension <= 0:
        raise ValueError("tensor dimensions must be positive")
    if values.ndim != 3 or values.shape[1:] != (total, total):
        raise ValueError("operators do not match the tensor dimensions")
    blocks = values.reshape(
        values.shape[0],
        first_dimension,
        second_dimension,
        first_dimension,
        second_dimension,
    )
    reduced = np.einsum("nitjt->nij", blocks)
    return reduced[0] if single else np.asarray(reduced, dtype=np.complex128)


def right_unitary_conditional_expectation(
    operators: Any,
    *,
    first_dimension: int,
    second_dimension: int,
) -> NDArray[np.complex128]:
    """Apply ``Tr_T(X) tensor I_T / T`` on the full operator algebra."""

    reduced = partial_trace_second_factor(
        operators,
        first_dimension=first_dimension,
        second_dimension=second_dimension,
    )
    single = reduced.ndim == 2
    if single:
        reduced = reduced[None, ...]
    identity = np.eye(second_dimension, dtype=np.complex128) / float(second_dimension)
    output = np.stack([np.kron(item, identity) for item in reduced], axis=0)
    return output[0] if single else np.asarray(output, dtype=np.complex128)


def right_unitary_fiber_witness(
    first: Any,
    second: Any,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[complex, NDArray[np.complex128]]:
    """Construct ``a, U`` with ``D = a C U`` for a common normalized left Gram.

    The construction also covers rank-deficient matrices by extending the
    partial isometry between their row spaces to a unitary on the full window
    factor.
    """

    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("fiber matrices must have the same two-dimensional shape")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("fiber matrices must be nonzero")
    left_unit = left / left_norm
    right_unit = right / right_norm
    left_gram = left_unit @ left_unit.conj().T
    right_gram = right_unit @ right_unit.conj().T
    if not np.allclose(left_gram, right_gram, atol=tolerance, rtol=tolerance):
        raise ValueError("matrices do not share a normalized left Gram")

    left_pinv = np.linalg.pinv(left_unit, rcond=tolerance)
    partial = left_pinv @ right_unit
    _, left_singular, left_vh = np.linalg.svd(left_unit, full_matrices=True)
    _, right_singular, right_vh = np.linalg.svd(right_unit, full_matrices=True)
    left_threshold = max(tolerance, tolerance * float(left_singular[0]))
    right_threshold = max(tolerance, tolerance * float(right_singular[0]))
    left_rank = int(np.sum(left_singular > left_threshold))
    right_rank = int(np.sum(right_singular > right_threshold))
    if left_rank != right_rank:
        raise RuntimeError("common left Gram produced inconsistent numerical ranks")
    left_null = left_vh.conj().T[:, left_rank:]
    right_null = right_vh.conj().T[:, right_rank:]
    unitary = partial + left_null @ right_null.conj().T
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    if not np.allclose(unitary.conj().T @ unitary, identity, atol=5 * tolerance, rtol=5 * tolerance):
        raise RuntimeError("failed to extend the fiber partial isometry to a unitary")
    scale = complex(right_norm / left_norm)
    if not np.allclose(scale * (left @ unitary), right, atol=5 * tolerance, rtol=5 * tolerance):
        raise RuntimeError("constructed fiber witness does not reconstruct the target")
    return scale, np.asarray(unitary, dtype=np.complex128)


def projective_fiber_dimension(
    matrix_rows: int, matrix_columns: int, rank: int
) -> int:
    """Return the generic real left-Gram fiber dimension on a rank stratum."""

    if matrix_rows <= 0 or matrix_columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    if rank <= 0 or rank > min(matrix_rows, matrix_columns):
        raise ValueError("rank is outside the matrix stratum")
    return int(2 * rank * matrix_columns - rank * rank - 1)


def hermitian_twirl_kernel_dimension(matrix_rows: int, matrix_columns: int) -> int:
    """Return the real dimension of the Hermitian kernel of the twirl map."""

    if matrix_rows <= 0 or matrix_columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    return int(matrix_rows**2 * (matrix_columns**2 - 1))


def physically_order_segment_major_rows(
    values: Any,
    *,
    segment_count: int,
    coefficients_per_segment: int,
) -> NDArray[np.complex128]:
    """Return row-major vectors of ``frequency x ordered-segment`` matrices.

    The input is assumed to be flattened as ``segment x frequency``.  The
    operation is a fixed permutation and therefore preserves Euclidean norms
    and exact projective geometry.
    """

    unit = complex_unit_rows(values)
    expected = int(segment_count) * int(coefficients_per_segment)
    if segment_count <= 0 or coefficients_per_segment <= 0:
        raise ValueError("segment and coefficient counts must be positive")
    if unit.shape[1] != expected:
        raise ValueError("input dimension does not match the segment contract")
    matrices = unit.reshape(unit.shape[0], segment_count, coefficients_per_segment)
    return np.asarray(matrices.transpose(0, 2, 1).reshape(unit.shape[0], expected))


def physically_order_segment_major_bank(
    bank: ProjectiveMeasurementBank,
    *,
    segment_count: int,
    coefficients_per_segment: int,
) -> ProjectiveMeasurementBank:
    """Apply the same fixed segment/frequency permutation to a frozen bank."""

    expected = int(segment_count) * int(coefficients_per_segment)
    if bank.input_dimension != expected:
        raise ValueError("measurement bank does not match the segment contract")
    vectors = np.asarray(
        bank.vectors.reshape(
            bank.output_dimension, segment_count, coefficients_per_segment
        )
        .transpose(0, 2, 1)
        .reshape(bank.output_dimension, expected),
        dtype=np.complex128,
    )
    vectors.setflags(write=False)
    digest = hashlib.sha256()
    digest.update(
        (
            f"physical-permutation-v1:{bank.sha256}:{segment_count}:"
            f"{coefficients_per_segment}:{vectors.shape}\n"
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(vectors).view(np.uint8))
    return ProjectiveMeasurementBank(
        vectors=vectors,
        seed=bank.seed,
        input_dimension=bank.input_dimension,
        output_dimension=bank.output_dimension,
        sha256=digest.hexdigest(),
    )


def coefficient_matrices(values: Any, *, matrix_rows: int) -> NDArray[np.complex128]:
    """Normalize complex rows and reshape them as row-major ``K x T`` matrices."""

    unit = complex_unit_rows(values)
    if matrix_rows <= 0 or unit.shape[1] % matrix_rows:
        raise ValueError("matrix_rows must divide the complex dimension")
    return np.asarray(
        unit.reshape(unit.shape[0], matrix_rows, unit.shape[1] // matrix_rows),
        dtype=np.complex128,
    )


def normalized_left_grams(
    values: Any, *, matrix_rows: int
) -> NDArray[np.complex128]:
    """Return ``C C*`` for unit-Frobenius row-major coefficient matrices."""

    matrices = coefficient_matrices(values, matrix_rows=matrix_rows)
    grams = matrices @ matrices.conj().transpose(0, 2, 1)
    return np.asarray(0.5 * (grams + grams.conj().transpose(0, 2, 1)))


def hermitian_hs_coordinates(matrices: Any) -> NDArray[np.float64]:
    """Coordinates whose Euclidean product equals the Hermitian HS product.

    Unlike the historical classifier coordinates, off-diagonal real and
    imaginary parts are multiplied by ``sqrt(2)``.  Consequently, for
    Hermitian ``A`` and ``B``, ``coord(A) @ coord(B) = tr(A B)``.
    """

    values = np.asarray(matrices, dtype=np.complex128)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrices must have shape (..., K, K)")
    residue = np.linalg.norm(
        values - values.conj().transpose(0, 2, 1), axis=(1, 2)
    )
    scale = np.maximum(1.0, np.linalg.norm(values, axis=(1, 2)))
    if np.any(residue > 256.0 * np.finfo(np.float64).eps * scale):
        raise ValueError("input is not Hermitian within floating-point tolerance")
    diagonal = np.diagonal(values, axis1=1, axis2=2).real
    upper = np.triu_indices(values.shape[1], k=1)
    root_two = np.sqrt(2.0)
    output = np.concatenate(
        (
            diagonal,
            root_two * values[:, upper[0], upper[1]].real,
            root_two * values[:, upper[0], upper[1]].imag,
        ),
        axis=1,
    ).astype(np.float64, copy=False)
    return output[0] if single else output


def analytic_right_unitary_twirl_features(
    values: Any,
    bank: ProjectiveMeasurementBank,
    *,
    matrix_rows: int,
    chunk_size: int = 256,
) -> NDArray[np.float64]:
    """Measure the analytic Haar average of each projective pure state.

    If ``a_l = vec(A_l)`` and ``L(C)=C C* / ||C||_F^2``, the returned raw
    coordinate is

    ``sqrt(n(n+1)/m) * tr(A_l A_l* L(C)) / T``.

    This is exactly ``E_U |a_l* vec(CU)|^2`` for normalized Haar measure on
    ``U(T)``; no Monte Carlo unitary sampling is used in production.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    unit = complex_unit_rows(values)
    if unit.shape[1] != bank.input_dimension:
        raise ValueError("signals do not match the frozen measurement bank")
    if matrix_rows <= 0 or unit.shape[1] % matrix_rows:
        raise ValueError("matrix_rows must divide the complex dimension")
    matrix_columns = unit.shape[1] // matrix_rows
    measurement_matrices = bank.vectors.reshape(
        bank.output_dimension, matrix_rows, matrix_columns
    )
    measurement_left = (
        measurement_matrices @ measurement_matrices.conj().transpose(0, 2, 1)
    )
    measurement_coordinates = hermitian_hs_coordinates(measurement_left)
    scale = np.sqrt(
        bank.input_dimension
        * (bank.input_dimension + 1.0)
        / bank.output_dimension
    )
    output = np.empty((unit.shape[0], bank.output_dimension), dtype=np.float64)
    for begin in range(0, unit.shape[0], chunk_size):
        stop = min(begin + chunk_size, unit.shape[0])
        left = normalized_left_grams(unit[begin:stop], matrix_rows=matrix_rows)
        coordinates = hermitian_hs_coordinates(left)
        output[begin:stop] = (
            scale
            * (coordinates @ measurement_coordinates.T)
            / float(matrix_columns)
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("analytic twirl produced non-finite features")
    return output


def projective_kernel_blocks(
    first: Any,
    second: Any | None = None,
    *,
    matrix_rows: int,
) -> dict[str, NDArray[np.float64]]:
    """Return exact, conditional-expectation, and deleted-component kernels."""

    left_values = complex_unit_rows(first)
    right_values = left_values if second is None else complex_unit_rows(second)
    if left_values.shape[1] != right_values.shape[1]:
        raise ValueError("kernel inputs must share the complex dimension")
    if matrix_rows <= 0 or left_values.shape[1] % matrix_rows:
        raise ValueError("matrix_rows must divide the complex dimension")
    matrix_columns = left_values.shape[1] // matrix_rows
    exact = np.abs(left_values @ right_values.conj().T) ** 2
    left_gram = hermitian_hs_coordinates(
        normalized_left_grams(left_values, matrix_rows=matrix_rows)
    )
    right_gram = hermitian_hs_coordinates(
        normalized_left_grams(right_values, matrix_rows=matrix_rows)
    )
    parallel = (left_gram @ right_gram.T) / float(matrix_columns)
    lost = exact - parallel
    return {
        "exact": np.asarray(exact.real, dtype=np.float64),
        "parallel": np.asarray(parallel, dtype=np.float64),
        "lost": np.asarray(lost, dtype=np.float64),
    }


def lost_component_squared_distance(kernel: Any) -> NDArray[np.float64]:
    """Convert a square deleted-component Gram matrix to squared distances."""

    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("kernel must be square")
    diagonal = np.diag(values)
    distance = diagonal[:, None] + diagonal[None, :] - 2.0 * values
    return np.maximum(distance, 0.0)


@dataclass(frozen=True)
class RealQuadraticBank:
    vectors: NDArray[np.float64]
    seed: int
    input_dimension: int
    output_dimension: int
    sha256: str


def real_quadratic_bank(
    input_dimension: int, output_dimension: int, *, seed: int
) -> RealQuadraticBank:
    """Draw a deterministic real-spherical bank for the Gram capacity control."""

    if input_dimension <= 0 or output_dimension <= 0:
        raise ValueError("bank dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    vectors = rng.normal(size=(output_dimension, input_dimension))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.asarray(vectors, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(
        f"real-spherical-quadratic-v1:{seed}:{vectors.shape}\n".encode("utf-8")
    )
    digest.update(np.ascontiguousarray(vectors).view(np.uint8))
    vectors.setflags(write=False)
    return RealQuadraticBank(
        vectors=vectors,
        seed=int(seed),
        input_dimension=int(input_dimension),
        output_dimension=int(output_dimension),
        sha256=digest.hexdigest(),
    )


def quadratic_gram_control_features(
    values: Any,
    *,
    matrix_rows: int,
    output_dimension: int,
    seed: int,
) -> tuple[NDArray[np.float64], RealQuadraticBank]:
    """Frozen quadratic random features of the retained left-Gram object."""

    coordinates = hermitian_hs_coordinates(
        normalized_left_grams(values, matrix_rows=matrix_rows)
    )
    bank = real_quadratic_bank(
        coordinates.shape[1], output_dimension, seed=int(seed)
    )
    scale = np.sqrt(
        coordinates.shape[1]
        * (coordinates.shape[1] + 2.0)
        / output_dimension
    )
    response = coordinates @ bank.vectors.T
    features = scale * response**2
    return np.asarray(features, dtype=np.float64), bank


def isometric_embedding(
    input_dimension: int, output_dimension: int, *, seed: int
) -> NDArray[np.float64]:
    """Return an output-by-input matrix with orthonormal columns."""

    if input_dimension <= 0 or output_dimension < input_dimension:
        raise ValueError("an isometric lift requires output_dimension >= input_dimension")
    if output_dimension == input_dimension:
        return np.eye(input_dimension, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(output_dimension, input_dimension))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return np.asarray(q, dtype=np.float64)


def covariance_diagnostics(values: Any, *, ridge_alpha: float = 1.0) -> dict[str, Any]:
    """Return finite covariance-spectrum diagnostics for a feature block."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("values must have at least two rows and one column")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigenvalues = singular**2 / max(1, matrix.shape[0] - 1)
    total = float(np.sum(eigenvalues))
    squared = float(np.sum(eigenvalues**2))
    effective_rank = 0.0 if squared <= 0.0 else total**2 / squared
    largest = float(eigenvalues[0]) if eigenvalues.size else 0.0
    smallest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    condition = (largest + ridge_alpha) / (smallest + ridge_alpha)
    return {
        "rows": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "numerical_rank": int(np.sum(singular > np.finfo(np.float64).eps * singular[0]))
        if singular.size and singular[0] > 0.0
        else 0,
        "effective_rank": float(effective_rank),
        "largest_covariance_eigenvalue": largest,
        "smallest_retained_covariance_eigenvalue": smallest,
        "condition_sigma_plus_alpha_i": float(condition),
        "singular_values_head": [float(value) for value in singular[:16]],
    }
