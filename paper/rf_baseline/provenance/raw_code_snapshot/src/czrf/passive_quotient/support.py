"""Known-only class-conditional support sets for frozen representations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SupportState:
    classes: NDArray[np.int64]
    centroids: NDArray[np.float64]
    diagonal_variances: NDArray[np.float64]
    shrinkage: float
    dimension: int


def fit_support_state(
    values: Any,
    labels: Any,
    *,
    classes: Any | None = None,
    shrinkage: float = 0.1,
) -> SupportState:
    array = np.asarray(values, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != truth.size:
        raise ValueError("values and labels must be aligned nonempty arrays")
    if not np.all(np.isfinite(array)):
        raise ValueError("support training values must be finite")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    class_array = (
        np.unique(truth)
        if classes is None
        else np.asarray(classes, dtype=np.int64).reshape(-1)
    )
    if class_array.size == 0 or np.unique(class_array).size != class_array.size:
        raise ValueError("classes must be nonempty and unique")
    centroids = np.empty((class_array.size, array.shape[1]), dtype=np.float64)
    raw_variances = np.empty_like(centroids)
    for position, label in enumerate(class_array):
        local = array[truth == label]
        if local.shape[0] < 2:
            raise ValueError("each support class needs at least two training rows")
        centroids[position] = np.mean(local, axis=0)
        raw_variances[position] = np.var(local, axis=0, ddof=0)
    positive = raw_variances[raw_variances > np.finfo(np.float64).tiny]
    target = float(np.median(positive)) if positive.size else 1.0
    variances = (1.0 - shrinkage) * raw_variances + shrinkage * target
    variances = np.maximum(variances, np.finfo(np.float64).tiny)
    return SupportState(
        classes=class_array,
        centroids=centroids,
        diagonal_variances=variances,
        shrinkage=float(shrinkage),
        dimension=int(array.shape[1]),
    )


def class_nonconformity(
    method: str,
    state: SupportState,
    values: Any,
    *,
    ridge_scores: Any | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != state.dimension:
        raise ValueError("values have incompatible support dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError("support target values must be finite")
    if method == "ridge_class_score":
        if ridge_scores is None:
            raise ValueError("ridge_class_score requires class-aligned scores")
        scores = np.asarray(ridge_scores, dtype=np.float64)
        if scores.shape != (array.shape[0], state.classes.size):
            raise ValueError("ridge scores have incompatible shape")
        if not np.all(np.isfinite(scores)):
            raise ValueError("ridge scores must be finite")
        return np.logaddexp(0.0, -scores)

    delta = array[:, None, :] - state.centroids[None, :, :]
    if method == "centroid_euclidean":
        return np.linalg.norm(delta, axis=2) / np.sqrt(state.dimension)
    if method == "centroid_angular":
        numerator = array @ state.centroids.T
        denominator = np.linalg.norm(array, axis=1, keepdims=True) * np.linalg.norm(
            state.centroids, axis=1
        )[None, :]
        cosine = numerator / np.maximum(denominator, np.finfo(np.float64).tiny)
        return 1.0 - np.clip(cosine, -1.0, 1.0)
    if method == "diagonal_shrinkage_mahalanobis":
        squared = np.mean(
            np.square(delta) / state.diagonal_variances[None, :, :], axis=2
        )
        return np.sqrt(np.maximum(squared, 0.0))
    raise ValueError(f"unknown support method: {method}")


def order_statistic_rank(sample_count: int, alpha: float) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return min(sample_count, int(math.ceil((sample_count + 1) * (1.0 - alpha))))


def calibrate_class_thresholds(
    class_scores: Any,
    labels: Any,
    classes: Any,
    *,
    alpha: float,
    floor: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    scores = np.asarray(class_scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    class_array = np.asarray(classes, dtype=np.int64).reshape(-1)
    if scores.shape != (truth.size, class_array.size):
        raise ValueError("calibration scores and labels are not class aligned")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores must be finite")
    thresholds = np.empty(class_array.size, dtype=np.float64)
    ranks = np.empty(class_array.size, dtype=np.int64)
    for position, label in enumerate(class_array):
        local = scores[truth == label, position]
        rank = order_statistic_rank(int(local.size), alpha)
        thresholds[position] = max(float(np.partition(local, rank - 1)[rank - 1]), floor)
        ranks[position] = rank
    return thresholds, ranks


def support_outputs(
    class_scores: Any,
    thresholds: Any,
    identity_scores: Any,
    classes: Any,
) -> dict[str, NDArray[Any]]:
    scores = np.asarray(class_scores, dtype=np.float64)
    threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    identity = np.asarray(identity_scores, dtype=np.float64)
    class_array = np.asarray(classes, dtype=np.int64).reshape(-1)
    expected = (scores.shape[0], class_array.size)
    if scores.shape != expected or identity.shape != expected:
        raise ValueError("class and identity scores must align with classes")
    if threshold_array.shape != (class_array.size,) or np.any(threshold_array <= 0.0):
        raise ValueError("thresholds must be positive and class aligned")
    normalized = scores / threshold_array[None, :]
    membership = normalized <= 1.0
    set_size = np.sum(membership, axis=1).astype(np.int64)
    novelty = np.min(normalized, axis=1)
    restricted = np.full(scores.shape[0], -1, dtype=np.int64)
    nonempty = set_size > 0
    masked = np.where(membership[nonempty], identity[nonempty], -np.inf)
    restricted[nonempty] = class_array[np.argmax(masked, axis=1)]
    return {
        "normalized_nonconformity": normalized,
        "membership": membership,
        "set_size": set_size,
        "novelty": novelty,
        "restricted_prediction": restricted,
    }
