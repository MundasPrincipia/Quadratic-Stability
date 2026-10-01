"""Sound task certificates for sequential complex-projective measurements.

The certificate concerns a frozen linear decision head on features

    z_i(x) = B |a_i^* x|^2,  ||x||_2 = ||a_i||_2 = 1.

For an unobserved coordinate set R, every pairwise residual score is the
quadratic form of a Hermitian operator.  Its absolute value is bounded by the
minimum of an entrywise L1 bound and the operator's Frobenius norm.  Hence a
strictly positive lower pairwise margin certifies the full-head decision for
every physically admissible completion, without assuming independent feature
coordinates.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _real_matrix(value: Any, name: str) -> RealArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix")
    return array


def _stable_psd_solve(
    gram: RealArray,
    right: RealArray,
    *,
    rcond: float,
    residual_tolerance: float,
) -> dict[str, Any]:
    """Solve a symmetric PSD system by an explicit spectral pseudoinverse.

    NumPy 2.4 on Windows can return a permutation-dependent rank from
    ``lstsq`` for a well-conditioned symmetric multi-right-hand-side problem.
    The eigensolver path keeps the matrix semantics explicit and fails closed
    if the right-hand side is not in the retained range.
    """

    matrix = np.asarray((gram + gram.T) / 2.0, dtype=np.float64)
    target = np.asarray(right, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    if target.ndim != 2 or target.shape[0] != matrix.shape[0]:
        raise ValueError("right-hand side has the wrong shape")
    condition = float(rcond)
    gate = float(residual_tolerance)
    if not np.isfinite(condition) or condition < 0.0:
        raise ValueError("rcond must be finite and nonnegative")
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    negativity_tolerance = max(
        condition, 64.0 * np.finfo(np.float64).eps * max(1, matrix.shape[0])
    )
    if float(np.min(eigenvalues, initial=0.0)) < -negativity_tolerance * scale:
        raise RuntimeError("observed projector Gram is materially indefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    threshold = max(
        condition, 64.0 * np.finfo(np.float64).eps * max(1, matrix.shape[0])
    ) * scale
    retained = clipped > threshold
    rank = int(np.sum(retained))
    if rank:
        basis = eigenvectors[:, retained]
        coefficients = basis @ (
            (basis.T @ target) / clipped[retained, None]
        )
        smallest = float(np.min(clipped[retained]))
        largest = float(np.max(clipped[retained]))
        effective_condition = largest / smallest
    else:
        coefficients = np.zeros_like(target)
        effective_condition = float("inf")
    residual = matrix @ coefficients - target
    denominator = max(float(np.linalg.norm(target, ord="fro")), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual, ord="fro") / denominator)
    if relative_residual > gate:
        raise RuntimeError(
            "PSD solve failed the a-posteriori residual gate: "
            f"{relative_residual:.3e} > {gate:.3e}"
        )
    return {
        "coefficients": np.asarray(coefficients, dtype=np.float64),
        "rank": rank,
        "singular_values": np.asarray(clipped[::-1], dtype=np.float64),
        "relative_residual": relative_residual,
        "effective_condition_number": float(effective_condition),
        "eigenvalue_threshold": float(threshold),
    }


def _roundoff_gamma(term_count: int, safety_factor: float) -> float:
    """Return a scale-aware binary64 dot-product audit envelope."""

    count = max(1, int(term_count))
    safety = float(safety_factor)
    if not np.isfinite(safety) or safety < 1.0:
        raise ValueError("roundoff_safety_factor must be finite and at least one")
    raw = safety * np.finfo(np.float64).eps * count
    if raw >= 0.5:
        raise RuntimeError("roundoff envelope is outside the supported regime")
    return float(raw / (1.0 - raw))


def projective_feature_upper_bound(input_dimension: int, output_dimension: int) -> float:
    """Return the exact scaling bound used by ``projective_intensity_features``."""

    dimension = int(input_dimension)
    measurements = int(output_dimension)
    if dimension < 1 or measurements < 1:
        raise ValueError("projective dimensions must be positive")
    return float(np.sqrt(dimension * (dimension + 1.0) / measurements))


def raw_linear_head(
    standardized_weights: Any,
    standardized_intercept: Any,
    mean: Any,
    scale: Any,
) -> tuple[RealArray, RealArray]:
    """Move a linear head through a fixed z-score standardizer exactly."""

    weights = _real_matrix(standardized_weights, "standardized_weights")
    intercept = np.asarray(standardized_intercept, dtype=np.float64).reshape(-1)
    center = np.asarray(mean, dtype=np.float64).reshape(-1)
    width = np.asarray(scale, dtype=np.float64).reshape(-1)
    if intercept.shape != (weights.shape[0],):
        raise ValueError("intercept has the wrong shape")
    if center.shape != (weights.shape[1],) or width.shape != center.shape:
        raise ValueError("standardizer vectors have the wrong shape")
    if not np.all(np.isfinite(intercept)) or not np.all(np.isfinite(center)):
        raise ValueError("head and standardizer must be finite")
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        raise ValueError("standardizer scale must be finite and positive")
    raw_weights = weights / width[None, :]
    raw_intercept = intercept - raw_weights @ center
    return (
        np.asarray(raw_weights, dtype=np.float64),
        np.asarray(raw_intercept, dtype=np.float64),
    )


def projector_gram(vectors: Any, *, tolerance: float = 1e-10) -> RealArray:
    """Return ``G_ij = tr(P_i P_j) = |a_i^* a_j|^2`` for unit rows."""

    bank = np.asarray(vectors, dtype=np.complex128)
    if bank.ndim != 2 or bank.shape[0] < 1 or bank.shape[1] < 1:
        raise ValueError("vectors must be a nonempty matrix")
    if not np.all(np.isfinite(bank.real)) or not np.all(np.isfinite(bank.imag)):
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(bank, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > float(tolerance):
        raise ValueError("projective measurement rows must be unit norm")
    gram = np.abs(bank @ bank.conj().T) ** 2
    return np.asarray((gram + gram.T) / 2.0, dtype=np.float64)


def centered_projector_gram(
    vectors: Any, *, tolerance: float = 1e-10
) -> RealArray:
    """Return the Gram matrix of ``Q_i=a_i a_i^* - I/n``."""

    bank = np.asarray(vectors, dtype=np.complex128)
    gram = projector_gram(bank, tolerance=tolerance)
    centered = gram - 1.0 / bank.shape[1]
    return np.asarray((centered + centered.T) / 2.0, dtype=np.float64)


def unordered_class_pairs(class_count: int) -> list[tuple[int, int]]:
    classes = int(class_count)
    if classes < 2:
        raise ValueError("at least two classes are required")
    return list(combinations(range(classes), 2))


def pair_weight_matrix(
    weights: Any,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[RealArray, list[tuple[int, int]]]:
    """Return one feature-weight difference row per unordered class pair."""

    matrix = _real_matrix(weights, "weights")
    selected = unordered_class_pairs(matrix.shape[0]) if pairs is None else list(pairs)
    if not selected:
        raise ValueError("at least one class pair is required")
    rows = []
    for first, second in selected:
        if first < 0 or second <= first or second >= matrix.shape[0]:
            raise ValueError("pairs must be ordered valid class indices")
        rows.append(matrix[first] - matrix[second])
    return np.asarray(rows, dtype=np.float64), selected


def residual_task_radii(
    pair_weights: Any,
    gram: Any,
    observed: Any,
    feature_bound: float,
) -> dict[str, RealArray]:
    """Return sound L1, Frobenius, and combined residual score radii."""

    differences = _real_matrix(pair_weights, "pair_weights")
    geometry = _real_matrix(gram, "gram")
    if geometry.shape != (differences.shape[1], differences.shape[1]):
        raise ValueError("projector Gram has the wrong shape")
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    if mask.shape != (differences.shape[1],):
        raise ValueError("observed mask has the wrong shape")
    bound = float(feature_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("feature_bound must be finite and positive")
    residual = differences.copy()
    residual[:, mask] = 0.0
    quadratic = np.einsum(
        "pi,ij,pj->p", residual, geometry, residual, optimize=True
    )
    scale = np.maximum(1.0, np.sum(np.abs(residual), axis=1) ** 2)
    if np.any(quadratic < -1e-10 * scale):
        raise RuntimeError("projector Gram produced a materially negative square")
    frobenius = bound * np.sqrt(np.maximum(0.0, quadratic))
    coordinate_l1 = bound * np.sum(np.abs(residual), axis=1)
    return {
        "coordinate_l1": np.asarray(coordinate_l1, dtype=np.float64),
        "operator_frobenius": np.asarray(frobenius, dtype=np.float64),
        "combined": np.asarray(
            np.minimum(coordinate_l1, frobenius), dtype=np.float64
        ),
    }


def task_weight_order(weights: Any) -> NDArray[np.int64]:
    """Order coordinates by aggregate absolute pairwise task weight."""

    differences, _ = pair_weight_matrix(weights)
    importance = np.sum(np.abs(differences), axis=0)
    return np.asarray(
        sorted(range(importance.size), key=lambda index: (-importance[index], index)),
        dtype=np.int64,
    )


def task_operator_greedy_order(weights: Any, gram: Any) -> NDArray[np.int64]:
    """Greedily reduce aggregate pairwise residual-operator Frobenius energy.

    The schedule is a design heuristic, not an optimality theorem.  All final
    decisions remain sound because they are separately checked by
    :func:`residual_task_radii`.
    """

    differences, _ = pair_weight_matrix(weights)
    geometry = _real_matrix(gram, "gram")
    if geometry.shape != (differences.shape[1], differences.shape[1]):
        raise ValueError("projector Gram has the wrong shape")
    residual = differences.copy()
    gram_response = residual @ geometry
    diagonal = np.diag(geometry)
    available = np.ones(residual.shape[1], dtype=bool)
    order: list[int] = []
    for _ in range(residual.shape[1]):
        reduction = np.sum(
            2.0 * residual * gram_response - residual**2 * diagonal[None, :],
            axis=0,
        )
        reduction[~available] = -np.inf
        best_value = float(np.max(reduction))
        candidates = np.flatnonzero(
            available & np.isclose(reduction, best_value, rtol=0.0, atol=1e-14)
        )
        selected = int(candidates[0])
        column = residual[:, selected].copy()
        gram_response -= column[:, None] * geometry[selected][None, :]
        residual[:, selected] = 0.0
        available[selected] = False
        order.append(selected)
    return np.asarray(order, dtype=np.int64)


def residual_radii_along_order(
    pair_weights: Any,
    gram: Any,
    order: Any,
    budgets: Any,
    feature_bound: float,
) -> dict[int, dict[str, RealArray]]:
    """Evaluate residual radii at many prefix budgets by rank-one updates."""

    differences = _real_matrix(pair_weights, "pair_weights")
    geometry = _real_matrix(gram, "gram")
    if geometry.shape != (differences.shape[1], differences.shape[1]):
        raise ValueError("projector Gram has the wrong shape")
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    if permutation.shape != (differences.shape[1],) or not np.array_equal(
        np.sort(permutation), np.arange(differences.shape[1])
    ):
        raise ValueError("order must be a full coordinate permutation")
    selected_budgets = sorted(set(int(value) for value in budgets))
    if (
        not selected_budgets
        or selected_budgets[0] < 0
        or selected_budgets[-1] > differences.shape[1]
    ):
        raise ValueError("budgets fall outside the measurement range")
    bound = float(feature_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("feature_bound must be finite and positive")
    residual = differences.copy()
    response = residual @ geometry
    current = 0
    output: dict[int, dict[str, RealArray]] = {}
    for budget in selected_budgets:
        for selected in permutation[current:budget]:
            index = int(selected)
            column = residual[:, index].copy()
            response -= column[:, None] * geometry[index][None, :]
            residual[:, index] = 0.0
        current = budget
        quadratic = np.sum(residual * response, axis=1)
        scale = np.maximum(1.0, np.sum(np.abs(residual), axis=1) ** 2)
        if np.any(quadratic < -1e-10 * scale):
            raise RuntimeError("updated projector Gram square became negative")
        frobenius = bound * np.sqrt(np.maximum(0.0, quadratic))
        coordinate_l1 = bound * np.sum(np.abs(residual), axis=1)
        output[budget] = {
            "coordinate_l1": np.asarray(coordinate_l1, dtype=np.float64),
            "operator_frobenius": np.asarray(frobenius, dtype=np.float64),
            "combined": np.asarray(
                np.minimum(coordinate_l1, frobenius), dtype=np.float64
            ),
        }
    return output


def residual_pair_spectral_bounds(
    vectors: Any,
    pair_weights: Any,
    observed: Any,
    feature_bound: float,
) -> dict[str, RealArray]:
    """Return exact unit-sphere support bounds for each residual task operator."""

    bank = np.asarray(vectors, dtype=np.complex128)
    differences = _real_matrix(pair_weights, "pair_weights")
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    if bank.ndim != 2 or bank.shape[0] != differences.shape[1]:
        raise ValueError("vectors and pair weights have incompatible dimensions")
    if mask.shape != (bank.shape[0],):
        raise ValueError("observed mask has the wrong shape")
    if not np.all(np.isfinite(bank.real)) or not np.all(np.isfinite(bank.imag)):
        raise ValueError("vectors must be finite")
    bound = float(feature_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("feature_bound must be finite and positive")
    residual = differences.copy()
    residual[:, mask] = 0.0
    lower = np.empty(residual.shape[0], dtype=np.float64)
    upper = np.empty_like(lower)
    for index, coefficients in enumerate(residual):
        operator = bound * (
            bank.T @ (coefficients[:, None] * bank.conj())
        )
        operator = (operator + operator.conj().T) / 2.0
        eigenvalues = np.linalg.eigvalsh(operator)
        lower[index] = float(eigenvalues[0])
        upper[index] = float(eigenvalues[-1])
    return {"lower": lower, "upper": upper}


def certified_linear_labels_directional(
    features: Any,
    weights: Any,
    intercept: Any,
    observed: Any,
    pair_lower: Any,
    pair_upper: Any,
    pairs: list[tuple[int, int]],
    *,
    strict_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Certify labels using directional spectral support for pair residuals."""

    values = _real_matrix(features, "features")
    head = _real_matrix(weights, "weights")
    bias = np.asarray(intercept, dtype=np.float64).reshape(-1)
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    lower = np.asarray(pair_lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(pair_upper, dtype=np.float64).reshape(-1)
    if values.shape[1] != head.shape[1] or mask.shape != (head.shape[1],):
        raise ValueError("feature/head/observed dimensions disagree")
    if bias.shape != (head.shape[0],) or lower.shape != (len(pairs),):
        raise ValueError("intercept or lower support has the wrong shape")
    if upper.shape != lower.shape or not np.all(np.isfinite(lower)) or not np.all(
        np.isfinite(upper)
    ):
        raise ValueError("spectral supports have the wrong shape or are nonfinite")
    if np.any(lower > upper):
        raise ValueError("spectral lower support exceeds upper support")
    partial = np.broadcast_to(bias, (values.shape[0], bias.size)).copy()
    if np.any(mask):
        partial += values[:, mask] @ head[:, mask].T
    robust = np.ones((values.shape[0], head.shape[0]), dtype=bool)
    minimum_lower = np.full_like(partial, np.inf)
    tolerance = float(strict_tolerance)
    for index, (first, second) in enumerate(pairs):
        margin = partial[:, first] - partial[:, second]
        first_lower = margin + lower[index]
        second_lower = -margin - upper[index]
        robust[:, first] &= first_lower > tolerance
        robust[:, second] &= second_lower > tolerance
        minimum_lower[:, first] = np.minimum(minimum_lower[:, first], first_lower)
        minimum_lower[:, second] = np.minimum(minimum_lower[:, second], second_lower)
    count = np.sum(robust, axis=1).astype(np.int64)
    if np.any(count > 1):
        raise RuntimeError("multiple labels passed directional certification")
    prediction = np.full(values.shape[0], -1, dtype=np.int64)
    accepted = count == 1
    prediction[accepted] = np.argmax(robust[accepted], axis=1)
    return {
        "prediction_index": prediction,
        "accepted": accepted,
        "robust_count": count,
        "partial_scores": partial,
        "minimum_pairwise_lower": minimum_lower,
    }


def projective_span_task_center_radius(
    features: Any,
    pair_weights: Any,
    gram: Any,
    observed: Any,
    feature_bound: float,
    *,
    rcond: float = 1e-12,
    outward_slack: float = 0.0,
    outward_relative_slack: float = 0.0,
    solver_residual_tolerance: float = 1e-9,
    roundoff_safety_factor: float = 128.0,
) -> dict[str, Any]:
    """Project each full task operator onto the observed projector span.

    Any finite coefficient vector returned by the least-squares solve gives a
    sound decomposition.  Solver optimality affects only the residual radius.
    The radius is recomputed from the explicit residual coefficients rather
    than by subtracting nearly equal squared norms.
    """

    values = _real_matrix(features, "features")
    differences = _real_matrix(pair_weights, "pair_weights")
    geometry = _real_matrix(gram, "gram")
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    if values.shape[1] != differences.shape[1]:
        raise ValueError("features and pair_weights disagree")
    if geometry.shape != (differences.shape[1], differences.shape[1]):
        raise ValueError("projector Gram has the wrong shape")
    if mask.shape != (differences.shape[1],):
        raise ValueError("observed mask has the wrong shape")
    bound = float(feature_bound)
    condition = float(rcond)
    slack = float(outward_slack)
    relative_slack = float(outward_relative_slack)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("feature_bound must be finite and positive")
    if not np.isfinite(condition) or condition < 0.0:
        raise ValueError("rcond must be finite and nonnegative")
    if not np.isfinite(slack) or slack < 0.0:
        raise ValueError("outward_slack must be finite and nonnegative")
    if not np.isfinite(relative_slack) or relative_slack < 0.0:
        raise ValueError("outward_relative_slack must be finite and nonnegative")
    selected = np.flatnonzero(mask)
    task_coefficients = bound * differences
    residual_coefficients = task_coefficients.copy()
    if selected.size:
        local_gram = geometry[np.ix_(selected, selected)]
        right = geometry[selected, :] @ task_coefficients.T
        solve = _stable_psd_solve(
            local_gram,
            right,
            rcond=condition,
            residual_tolerance=solver_residual_tolerance,
        )
        coefficients = solve["coefficients"]
        rank = solve["rank"]
        singular = solve["singular_values"]
        residual_coefficients[:, selected] -= coefficients.T
        observations = values[:, selected] / bound
        centers = observations @ coefficients
        gamma = _roundoff_gamma(selected.size, roundoff_safety_factor)
        center_roundoff = gamma * (np.abs(observations) @ np.abs(coefficients))
    else:
        coefficients = np.zeros((0, differences.shape[0]), dtype=np.float64)
        centers = np.zeros((values.shape[0], differences.shape[0]), dtype=np.float64)
        rank = 0
        singular = np.zeros(0, dtype=np.float64)
        solve = {
            "relative_residual": 0.0,
            "effective_condition_number": 1.0,
            "eigenvalue_threshold": 0.0,
        }
        center_roundoff = np.zeros_like(centers)
    quadratic = np.einsum(
        "pi,ij,pj->p",
        residual_coefficients,
        geometry,
        residual_coefficients,
        optimize=True,
    )
    scale = np.maximum(
        1.0, np.sum(np.abs(residual_coefficients), axis=1) ** 2
    )
    if np.any(quadratic < -1e-10 * scale):
        raise RuntimeError("projected task residual square became negative")
    structural_radius = np.sqrt(np.maximum(0.0, quadratic))
    scale_radius = relative_slack * np.sum(np.abs(task_coefficients), axis=1)
    radius = structural_radius + scale_radius + slack
    return {
        "pair_task_center_without_intercept": np.asarray(
            centers, dtype=np.float64
        ),
        "pair_radius": np.asarray(radius, dtype=np.float64),
        "pair_center_roundoff": np.asarray(center_roundoff, dtype=np.float64),
        "pair_structural_radius": np.asarray(
            structural_radius, dtype=np.float64
        ),
        "pair_scale_outward_radius": np.asarray(
            scale_radius, dtype=np.float64
        ),
        "span_coefficients": np.asarray(coefficients, dtype=np.float64),
        "residual_coefficients": np.asarray(
            residual_coefficients, dtype=np.float64
        ),
        "observed_rank": int(rank),
        "observed_singular_values": np.asarray(singular, dtype=np.float64),
        "solver_relative_residual": float(solve["relative_residual"]),
        "solver_effective_condition_number": float(
            solve["effective_condition_number"]
        ),
        "solver_eigenvalue_threshold": float(solve["eigenvalue_threshold"]),
    }


def certified_labels_from_pair_intervals(
    pair_centers: Any,
    pair_radii: Any,
    pairs: list[tuple[int, int]],
    class_count: int,
    *,
    strict_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Certify labels from symmetric intervals for unordered pair margins."""

    centers = _real_matrix(pair_centers, "pair_centers")
    radii = np.asarray(pair_radii, dtype=np.float64).reshape(-1)
    classes = int(class_count)
    if centers.shape[1] != len(pairs) or radii.shape != (len(pairs),):
        raise ValueError("pair center/radius dimensions disagree")
    if classes < 2 or np.any(radii < 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("invalid class count or pair radii")
    robust = np.ones((centers.shape[0], classes), dtype=bool)
    minimum_lower = np.full((centers.shape[0], classes), np.inf)
    tolerance = float(strict_tolerance)
    for index, (first, second) in enumerate(pairs):
        if first < 0 or second <= first or second >= classes:
            raise ValueError("invalid class pair")
        first_lower = centers[:, index] - radii[index]
        second_lower = -centers[:, index] - radii[index]
        robust[:, first] &= first_lower > tolerance
        robust[:, second] &= second_lower > tolerance
        minimum_lower[:, first] = np.minimum(minimum_lower[:, first], first_lower)
        minimum_lower[:, second] = np.minimum(minimum_lower[:, second], second_lower)
    count = np.sum(robust, axis=1).astype(np.int64)
    if np.any(count > 1):
        raise RuntimeError("multiple labels passed pair-interval certification")
    prediction = np.full(centers.shape[0], -1, dtype=np.int64)
    accepted = count == 1
    prediction[accepted] = np.argmax(robust[accepted], axis=1)
    return {
        "prediction_index": prediction,
        "accepted": accepted,
        "robust_count": count,
        "minimum_pairwise_lower": minimum_lower,
    }


def centered_projective_span_task_bounds(
    features: Any,
    pair_weights: Any,
    centered_gram: Any,
    observed: Any,
    feature_bound: float,
    state_dimension: int,
    *,
    rcond: float = 1e-12,
    outward_slack: float = 0.0,
    outward_relative_slack: float = 0.0,
    solver_residual_tolerance: float = 1e-9,
    roundoff_safety_factor: float = 128.0,
) -> dict[str, Any]:
    """Use the exact trace-one constraint before observed-span recovery."""

    values = _real_matrix(features, "features")
    differences = _real_matrix(pair_weights, "pair_weights")
    geometry = _real_matrix(centered_gram, "centered_gram")
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    dimension = int(state_dimension)
    bound = float(feature_bound)
    condition = float(rcond)
    slack = float(outward_slack)
    relative_slack = float(outward_relative_slack)
    if values.shape[1] != differences.shape[1]:
        raise ValueError("features and pair_weights disagree")
    if geometry.shape != (differences.shape[1], differences.shape[1]):
        raise ValueError("centered projector Gram has the wrong shape")
    if mask.shape != (differences.shape[1],):
        raise ValueError("observed mask has the wrong shape")
    if dimension < 2 or not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("state_dimension and feature_bound are invalid")
    if not np.isfinite(condition) or condition < 0.0:
        raise ValueError("rcond must be finite and nonnegative")
    if not np.isfinite(slack) or slack < 0.0:
        raise ValueError("outward_slack must be finite and nonnegative")
    if not np.isfinite(relative_slack) or relative_slack < 0.0:
        raise ValueError("outward_relative_slack must be finite and nonnegative")
    selected = np.flatnonzero(mask)
    task_coefficients = bound * differences
    residual_coefficients = task_coefficients.copy()
    scalar_center = (bound / dimension) * np.sum(differences, axis=1)
    if selected.size:
        local_gram = geometry[np.ix_(selected, selected)]
        right = geometry[selected, :] @ task_coefficients.T
        solve = _stable_psd_solve(
            local_gram,
            right,
            rcond=condition,
            residual_tolerance=solver_residual_tolerance,
        )
        coefficients = solve["coefficients"]
        rank = solve["rank"]
        singular = solve["singular_values"]
        residual_coefficients[:, selected] -= coefficients.T
        centered_observations = values[:, selected] / bound - 1.0 / dimension
        centers = centered_observations @ coefficients + scalar_center[None, :]
        gamma = _roundoff_gamma(selected.size + 1, roundoff_safety_factor)
        center_roundoff = gamma * (
            np.abs(centered_observations) @ np.abs(coefficients)
            + np.abs(scalar_center)[None, :]
        )
    else:
        coefficients = np.zeros((0, differences.shape[0]), dtype=np.float64)
        centers = np.broadcast_to(
            scalar_center, (values.shape[0], scalar_center.size)
        ).copy()
        rank = 0
        singular = np.zeros(0, dtype=np.float64)
        solve = {
            "relative_residual": 0.0,
            "effective_condition_number": 1.0,
            "eigenvalue_threshold": 0.0,
        }
        center_roundoff = _roundoff_gamma(
            1, roundoff_safety_factor
        ) * np.broadcast_to(
            np.abs(scalar_center), (values.shape[0], scalar_center.size)
        )
    quadratic = np.einsum(
        "pi,ij,pj->p",
        residual_coefficients,
        geometry,
        residual_coefficients,
        optimize=True,
    )
    scale = np.maximum(
        1.0, np.sum(np.abs(residual_coefficients), axis=1) ** 2
    )
    if np.any(quadratic < -1e-10 * scale):
        raise RuntimeError("centered task residual square became negative")
    state_centered_norm = np.sqrt(1.0 - 1.0 / dimension)
    structural_radius = state_centered_norm * np.sqrt(
        np.maximum(0.0, quadratic)
    )
    scale_radius = relative_slack * np.sum(np.abs(task_coefficients), axis=1)
    radius = structural_radius + scale_radius + slack
    total_radius = radius[None, :] + center_roundoff
    return {
        "pair_task_center_without_intercept": np.asarray(
            centers, dtype=np.float64
        ),
        "pair_lower_without_intercept": np.asarray(
            centers - total_radius, dtype=np.float64
        ),
        "pair_upper_without_intercept": np.asarray(
            centers + total_radius, dtype=np.float64
        ),
        "pair_radius": np.asarray(radius, dtype=np.float64),
        "pair_total_radius": np.asarray(total_radius, dtype=np.float64),
        "pair_center_roundoff": np.asarray(center_roundoff, dtype=np.float64),
        "pair_structural_radius": np.asarray(
            structural_radius, dtype=np.float64
        ),
        "pair_scale_outward_radius": np.asarray(
            scale_radius, dtype=np.float64
        ),
        "span_coefficients": np.asarray(coefficients, dtype=np.float64),
        "residual_coefficients": np.asarray(
            residual_coefficients, dtype=np.float64
        ),
        "observed_rank": int(rank),
        "observed_singular_values": np.asarray(singular, dtype=np.float64),
        "state_centered_norm": float(state_centered_norm),
        "solver_relative_residual": float(solve["relative_residual"]),
        "solver_effective_condition_number": float(
            solve["effective_condition_number"]
        ),
        "solver_eigenvalue_threshold": float(solve["eigenvalue_threshold"]),
    }


def centered_projective_cholesky_path_components(
    features: Any,
    pair_weights: Any,
    centered_gram: Any,
    order: Any,
    feature_bound: float,
    state_dimension: int,
    *,
    solver_residual_tolerance: float = 1e-10,
    roundoff_safety_factor: float = 128.0,
) -> dict[str, Any]:
    """Build a stable all-prefix orthogonal ledger for an SPD projector bank.

    If ``G_order = L L^T`` and the ordered centered projectors satisfy
    ``q_i = sum_k L_ik e_k``, triangular solves recover the coordinates of
    every observed state and pairwise task operator in the same orthonormal
    basis.  Prefix centers and residual norms then update by scalar products,
    avoiding one least-squares solve per budget.

    This routine deliberately fails closed when the centered Gram is singular
    or any Cholesky/solve residual exceeds the declared gate.  A rank-deficient
    bank requires the separately audited spectral-pseudoinverse interface.
    """

    values = _real_matrix(features, "features")
    differences = _real_matrix(pair_weights, "pair_weights")
    geometry = _real_matrix(centered_gram, "centered_gram")
    permutation = np.asarray(order, dtype=np.int64).reshape(-1)
    dimension = int(state_dimension)
    bound = float(feature_bound)
    gate = float(solver_residual_tolerance)
    if values.shape[1] != differences.shape[1]:
        raise ValueError("features and pair_weights disagree")
    measurements = differences.shape[1]
    if geometry.shape != (measurements, measurements):
        raise ValueError("centered projector Gram has the wrong shape")
    if permutation.shape != (measurements,) or not np.array_equal(
        np.sort(permutation), np.arange(measurements)
    ):
        raise ValueError("order must be a complete measurement permutation")
    if dimension < 2 or not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("state_dimension and feature_bound are invalid")
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("solver_residual_tolerance must be positive")

    permuted = np.asarray(
        geometry[np.ix_(permutation, permutation)], dtype=np.float64
    )
    permuted = (permuted + permuted.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(permuted)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    positivity_gate = 64.0 * np.finfo(np.float64).eps * measurements * scale
    if float(eigenvalues[0]) <= positivity_gate:
        raise RuntimeError(
            "all-prefix Cholesky ledger requires a qualified positive-definite Gram"
        )
    factor = np.linalg.cholesky(permuted)
    factor_residual = float(
        np.linalg.norm(permuted - factor @ factor.T, ord="fro")
        / max(np.linalg.norm(permuted, ord="fro"), np.finfo(float).tiny)
    )

    centered_observations = values[:, permutation] / bound - 1.0 / dimension
    state_coordinates = solve_triangular(
        factor, centered_observations.T, lower=True, check_finite=False
    ).T
    task_coefficients = bound * differences
    task_inner = geometry[permutation, :] @ task_coefficients.T
    task_coordinates = solve_triangular(
        factor, task_inner, lower=True, check_finite=False
    )
    state_residual = float(
        np.linalg.norm(
            factor @ state_coordinates.T - centered_observations.T,
            ord="fro",
        )
        / max(np.linalg.norm(centered_observations, ord="fro"), np.finfo(float).tiny)
    )
    task_residual = float(
        np.linalg.norm(factor @ task_coordinates - task_inner, ord="fro")
        / max(np.linalg.norm(task_inner, ord="fro"), np.finfo(float).tiny)
    )
    if max(factor_residual, state_residual, task_residual) > gate:
        raise RuntimeError(
            "all-prefix Cholesky ledger failed its a-posteriori residual gate"
        )

    full_square = np.einsum(
        "pi,ij,pj->p",
        task_coefficients,
        geometry,
        task_coefficients,
        optimize=True,
    )
    coordinate_square = np.sum(task_coordinates**2, axis=0)
    reconstruction_scale = np.maximum(1.0, np.abs(full_square))
    reconstruction_error = float(
        np.max(np.abs(full_square - coordinate_square) / reconstruction_scale)
    )
    if reconstruction_error > gate:
        raise RuntimeError("orthogonal task ledger failed full-span reconstruction")
    residual_square_by_budget = np.zeros(
        (measurements + 1, task_coordinates.shape[1]), dtype=np.float64
    )
    residual_square_by_budget[:-1] = np.flip(
        np.cumsum(np.flip(task_coordinates**2, axis=0), axis=0), axis=0
    )
    gamma = _roundoff_gamma(measurements + 1, roundoff_safety_factor)
    return {
        "order": np.asarray(permutation, dtype=np.int64),
        "state_coordinates": np.asarray(state_coordinates, dtype=np.float64),
        "task_coordinates": np.asarray(task_coordinates, dtype=np.float64),
        "task_full_frobenius_square": np.asarray(
            np.maximum(0.0, full_square), dtype=np.float64
        ),
        "task_residual_frobenius_square_by_budget": np.asarray(
            residual_square_by_budget, dtype=np.float64
        ),
        "scalar_center": np.asarray(
            (bound / dimension) * np.sum(differences, axis=1),
            dtype=np.float64,
        ),
        "task_l1_scale": np.asarray(
            np.sum(np.abs(task_coefficients), axis=1), dtype=np.float64
        ),
        "state_centered_norm": float(np.sqrt(1.0 - 1.0 / dimension)),
        "roundoff_gamma": gamma,
        "diagnostics": {
            "measurement_count": measurements,
            "minimum_gram_eigenvalue": float(eigenvalues[0]),
            "maximum_gram_eigenvalue": float(eigenvalues[-1]),
            "gram_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "cholesky_factor_relative_residual": factor_residual,
            "state_solve_relative_residual": state_residual,
            "task_solve_relative_residual": task_residual,
            "full_span_reconstruction_relative_error": reconstruction_error,
        },
    }


def certified_labels_from_pair_bounds(
    pair_lower: Any,
    pair_upper: Any,
    pairs: list[tuple[int, int]],
    class_count: int,
    *,
    strict_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Certify labels from possibly asymmetric unordered-pair bounds."""

    lower = _real_matrix(pair_lower, "pair_lower")
    upper = _real_matrix(pair_upper, "pair_upper")
    classes = int(class_count)
    if lower.shape != upper.shape or lower.shape[1] != len(pairs):
        raise ValueError("pair bound dimensions disagree")
    if classes < 2 or np.any(lower > upper):
        raise ValueError("invalid class count or inverted pair bounds")
    robust = np.ones((lower.shape[0], classes), dtype=bool)
    minimum_lower = np.full((lower.shape[0], classes), np.inf)
    tolerance = float(strict_tolerance)
    for index, (first, second) in enumerate(pairs):
        first_lower = lower[:, index]
        second_lower = -upper[:, index]
        robust[:, first] &= first_lower > tolerance
        robust[:, second] &= second_lower > tolerance
        minimum_lower[:, first] = np.minimum(minimum_lower[:, first], first_lower)
        minimum_lower[:, second] = np.minimum(minimum_lower[:, second], second_lower)
    count = np.sum(robust, axis=1).astype(np.int64)
    if np.any(count > 1):
        raise RuntimeError("multiple labels passed pair-bound certification")
    prediction = np.full(lower.shape[0], -1, dtype=np.int64)
    accepted = count == 1
    prediction[accepted] = np.argmax(robust[accepted], axis=1)
    return {
        "prediction_index": prediction,
        "accepted": accepted,
        "robust_count": count,
        "minimum_pairwise_lower": minimum_lower,
    }


def certified_linear_labels(
    features: Any,
    weights: Any,
    intercept: Any,
    observed: Any,
    pair_radii: Any,
    pairs: list[tuple[int, int]],
    *,
    strict_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Certify a unique full-head label from a subset of observed features."""

    values = _real_matrix(features, "features")
    head = _real_matrix(weights, "weights")
    bias = np.asarray(intercept, dtype=np.float64).reshape(-1)
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    radii = np.asarray(pair_radii, dtype=np.float64).reshape(-1)
    if values.shape[1] != head.shape[1] or mask.shape != (head.shape[1],):
        raise ValueError("feature/head/observed dimensions disagree")
    if bias.shape != (head.shape[0],) or radii.shape != (len(pairs),):
        raise ValueError("intercept or pair_radii has the wrong shape")
    if np.any(radii < 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("pair radii must be finite and nonnegative")
    partial = np.broadcast_to(bias, (values.shape[0], bias.size)).copy()
    if np.any(mask):
        partial += values[:, mask] @ head[:, mask].T
    robust = np.ones((values.shape[0], head.shape[0]), dtype=bool)
    minimum_lower = np.full_like(partial, np.inf)
    tolerance = float(strict_tolerance)
    for index, (first, second) in enumerate(pairs):
        margin = partial[:, first] - partial[:, second]
        first_lower = margin - radii[index]
        second_lower = -margin - radii[index]
        robust[:, first] &= first_lower > tolerance
        robust[:, second] &= second_lower > tolerance
        minimum_lower[:, first] = np.minimum(minimum_lower[:, first], first_lower)
        minimum_lower[:, second] = np.minimum(minimum_lower[:, second], second_lower)
    count = np.sum(robust, axis=1).astype(np.int64)
    if np.any(count > 1):
        raise RuntimeError("mutually inconsistent multiple robust labels")
    prediction = np.full(values.shape[0], -1, dtype=np.int64)
    accepted = count == 1
    prediction[accepted] = np.argmax(robust[accepted], axis=1)
    return {
        "prediction_index": prediction,
        "accepted": accepted,
        "robust_count": count,
        "partial_scores": partial,
        "minimum_pairwise_lower": minimum_lower,
    }
