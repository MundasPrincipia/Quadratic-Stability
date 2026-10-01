"""Ordered conditional expectations and local/global mechanism audits.

The time coordinate is the second tensor factor in the row-major
identification C^K tensor C^T.  No time permutation is quotiented out by
default.  The group at level m fixes a pre-registered resolved subspace
V_m pointwise and Haar-averages only its orthogonal residual.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .mechanism import coefficient_matrices, hermitian_hs_coordinates


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def discrete_polynomial_basis(time_length: int) -> RealArray:
    """Return a deterministic orthonormal ordered basis on 0,...,T-1.

    The columns are obtained by QR orthogonalization of increasing powers
    of the centered, scaled time coordinate.  Column signs are fixed by
    requiring the first numerically nonzero entry to be positive.
    """

    length = int(time_length)
    if length < 1:
        raise ValueError("time_length must be positive")
    if length == 1:
        return np.ones((1, 1), dtype=np.float64)
    coordinate = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    vandermonde = np.vander(coordinate, N=length, increasing=True)
    basis, _ = np.linalg.qr(vandermonde)
    for column in range(length):
        nonzero = np.flatnonzero(np.abs(basis[:, column]) > 1e-14)
        if nonzero.size and basis[nonzero[0], column] < 0.0:
            basis[:, column] *= -1.0
    return np.asarray(basis, dtype=np.float64)


def resolved_projector(
    time_length: int,
    level: int,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Project onto V_level; level=-1 denotes the zero subspace."""

    length = int(time_length)
    selected = int(level)
    if length < 1 or selected < -1 or selected >= length:
        raise ValueError("level must lie in {-1,...,time_length-1}")
    ordered = (
        discrete_polynomial_basis(length)
        if basis is None
        else np.asarray(basis, dtype=np.complex128)
    )
    if ordered.shape != (length, length):
        raise ValueError("basis must have shape (time_length, time_length)")
    gram_error = float(
        np.linalg.norm(ordered.conj().T @ ordered - np.eye(length), ord=2)
    )
    if not np.isfinite(gram_error) or gram_error > 1e-10:
        raise ValueError("basis must be unitary")
    if selected == -1:
        return np.zeros((length, length), dtype=np.complex128)
    resolved = ordered[:, : selected + 1]
    return np.asarray(resolved @ resolved.conj().T, dtype=np.complex128)


def row_major_projector(coefficients: Any) -> ComplexArray:
    """Return the normalized rank-one operator for a K by T matrix."""

    matrix = np.asarray(coefficients, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("coefficients must be a nonempty matrix")
    vector = matrix.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("coefficients must be finite and nonzero")
    vector = vector / norm
    return np.outer(vector, vector.conj())


def ordered_conditional_expectation(
    operator: Any,
    matrix_rows: int,
    time_length: int,
    level: int,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Closed-form Haar twirl fixing V_level and isotropizing W_level."""

    rows = int(matrix_rows)
    length = int(time_length)
    if rows < 1 or length < 1:
        raise ValueError("matrix_rows and time_length must be positive")
    value = np.asarray(operator, dtype=np.complex128)
    expected = rows * length
    if value.shape != (expected, expected):
        raise ValueError("operator has the wrong tensor-product shape")
    if not np.all(np.isfinite(value.real)) or not np.all(np.isfinite(value.imag)):
        raise ValueError("operator must be finite")
    time_resolved = resolved_projector(length, level, basis=basis)
    time_residual = np.eye(length, dtype=np.complex128) - time_resolved
    residual_dimension = length - (int(level) + 1)
    if residual_dimension == 0:
        return value.copy()

    resolved = np.kron(np.eye(rows, dtype=np.complex128), time_resolved)
    residual = np.kron(np.eye(rows, dtype=np.complex128), time_residual)
    residual_block = residual @ value @ residual
    tensor = residual_block.reshape(rows, length, rows, length)
    partial_trace = np.einsum("atbt->ab", tensor, optimize=True)
    isotropic = np.einsum(
        "ab,st->asbt",
        partial_trace,
        time_residual / residual_dimension,
        optimize=True,
    ).reshape(expected, expected)
    return np.asarray(resolved @ value @ resolved + isotropic, dtype=np.complex128)


def ordered_band(
    operator: Any,
    matrix_rows: int,
    time_length: int,
    level: int,
    *,
    basis: Any | None = None,
) -> ComplexArray:
    """Return Delta_level = E_level - E_(level-1)."""

    selected = int(level)
    if selected < 0 or selected >= int(time_length):
        raise ValueError("band level must lie in {0,...,time_length-1}")
    return ordered_conditional_expectation(
        operator, matrix_rows, time_length, selected, basis=basis
    ) - ordered_conditional_expectation(
        operator, matrix_rows, time_length, selected - 1, basis=basis
    )


def permutation_twirl_enumerated(
    operator: Any,
    matrix_rows: int,
    time_length: int,
) -> ComplexArray:
    """Enumerate the full symmetric-group twirl for small T."""

    rows = int(matrix_rows)
    length = int(time_length)
    value = np.asarray(operator, dtype=np.complex128)
    expected = rows * length
    if value.shape != (expected, expected):
        raise ValueError("operator has the wrong tensor-product shape")
    total = np.zeros_like(value)
    count = 0
    identity = np.eye(length, dtype=np.complex128)
    for order in permutations(range(length)):
        permutation = identity[:, order]
        representation = np.kron(np.eye(rows), permutation)
        total += representation @ value @ representation.conj().T
        count += 1
    return total / count


def normalized_energy_norm(values: Any) -> float:
    """Root-mean-square norm used by the local-amplification counterfamily."""

    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size < 1:
        raise ValueError("values must be nonempty")
    return float(np.linalg.norm(vector) / np.sqrt(vector.size))


def local_evaluation_amplification(time_length: int) -> float:
    """Operator norm of one coordinate under the normalized energy norm."""

    length = int(time_length)
    if length < 1:
        raise ValueError("time_length must be positive")
    return float(np.sqrt(length))


def random_common_plus_direction_basis(time_length: int, *, seed: int) -> RealArray:
    """Return the pre-registered matched control span{u0, v}.

    The random vector is real, projected onto the zero-sum subspace, and its
    sign is fixed deterministically.  This makes one seed define the same
    temporal direction in every source with the same T.
    """

    length = int(time_length)
    if length < 2:
        raise ValueError("time_length must be at least two")
    common = np.ones(length, dtype=np.float64) / np.sqrt(length)
    rng = np.random.default_rng(int(seed))
    direction = rng.normal(size=length)
    direction -= common * float(common @ direction)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-14:
        raise RuntimeError("random direction collapsed after zero-sum projection")
    direction /= norm
    nonzero = np.flatnonzero(np.abs(direction) > 1e-14)
    if nonzero.size and direction[nonzero[0]] < 0.0:
        direction *= -1.0
    return np.column_stack((common, direction))


def ordered_subspace_feature_bank(
    values: Any,
    *,
    matrix_rows: int,
    resolved_basis: Any,
) -> RealArray:
    """Explicit isometric features for the ordered-subspace Haar twirl.

    If V has dimension r and W has dimension d, the feature is the direct
    sum of Hermitian coordinates of the full resolved rank-one block and
    Hermitian coordinates of the residual left Gram divided by sqrt(d).
    Its Euclidean inner product is exactly the Hilbert--Schmidt kernel of the
    conditional expectation in :func:`ordered_conditional_expectation`.
    """

    matrices = coefficient_matrices(values, matrix_rows=int(matrix_rows))
    length = matrices.shape[2]
    basis = np.asarray(resolved_basis, dtype=np.complex128)
    if basis.ndim != 2 or basis.shape[0] != length or not (1 <= basis.shape[1] <= length):
        raise ValueError("resolved_basis must have shape (T, r), 1 <= r <= T")
    gram_error = float(np.linalg.norm(basis.conj().T @ basis - np.eye(basis.shape[1])))
    if not np.isfinite(gram_error) or gram_error > 1e-10:
        raise ValueError("resolved_basis columns must be orthonormal")
    resolved_coefficients = np.einsum(
        "nkt,tr->nkr", matrices, basis.conj(), optimize=True
    )
    resolved_vectors = resolved_coefficients.reshape(matrices.shape[0], -1)
    resolved_operators = np.einsum(
        "ni,nj->nij", resolved_vectors, resolved_vectors.conj(), optimize=True
    )
    parts = [
        np.asarray(hermitian_hs_coordinates(resolved_operators), dtype=np.float64)
    ]
    residual_dimension = length - basis.shape[1]
    if residual_dimension:
        reconstructed = np.einsum(
            "nkr,tr->nkt", resolved_coefficients, basis, optimize=True
        )
        residual = matrices - reconstructed
        residual_gram = np.einsum(
            "nkt,nlt->nkl", residual, residual.conj(), optimize=True
        )
        parts.append(
            np.asarray(hermitian_hs_coordinates(residual_gram), dtype=np.float64)
            / np.sqrt(residual_dimension)
        )
    return np.concatenate(parts, axis=1).astype(np.float64, copy=False)


def noncyclic_lag_feature_bank(
    values: Any,
    *,
    matrix_rows: int,
    lag: int = 1,
) -> RealArray:
    """Return Hermitian real/imaginary coordinates of a noncyclic lag."""

    matrices = coefficient_matrices(values, matrix_rows=int(matrix_rows))
    selected = int(lag)
    length = matrices.shape[2]
    if selected < 1 or selected >= length:
        raise ValueError("lag must lie in {1,...,T-1}")
    gamma = np.einsum(
        "nkt,nlt->nkl",
        matrices[:, :, selected:],
        matrices[:, :, : length - selected].conj(),
        optimize=True,
    )
    root_two = np.sqrt(2.0)
    hermitian_real = (gamma + gamma.conj().transpose(0, 2, 1)) / root_two
    hermitian_imag = (gamma - gamma.conj().transpose(0, 2, 1)) / (1j * root_two)
    return np.concatenate(
        (
            hermitian_hs_coordinates(hermitian_real),
            hermitian_hs_coordinates(hermitian_imag),
        ),
        axis=1,
    ).astype(np.float64, copy=False)


def static_dynamic_drift_audit(
    *,
    step_budget: float,
    interval_radius: float,
    endpoint: float,
) -> dict[str, float | int | bool]:
    """Construct the exact one-dimensional static/dynamic gluing witness."""

    budget = float(step_budget)
    radius = float(interval_radius)
    terminal = float(endpoint)
    if budget <= 0.0 or radius < 0.0 or terminal <= 2.0 * radius:
        raise ValueError("require step_budget>0 and endpoint>2*interval_radius")
    steps = int(np.ceil(terminal / budget))
    centers = np.linspace(0.0, terminal, steps + 1, dtype=np.float64)
    lower = float(np.max(centers - radius))
    upper = float(np.min(centers + radius))
    maximum_step = float(np.max(np.diff(centers))) if steps else 0.0
    return {
        "number_of_steps": steps,
        "maximum_step": maximum_step,
        "step_budget": budget,
        "static_intersection_lower": lower,
        "static_intersection_upper": upper,
        "static_intersection_empty": bool(lower > upper),
        "dynamic_truth_path_feasible": bool(maximum_step <= budget * (1.0 + 1e-14)),
        "minimum_jacobian_singular_value": 1.0,
    }
