"""Cluster-level simultaneous inference for frozen representation contrasts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def device_accuracy_effect_matrix(
    truth: Any,
    devices: Any,
    candidates: Mapping[str, Any],
    reference: Any,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return one paired accuracy effect per physical device and candidate."""

    y = np.asarray(truth, dtype=np.int64)
    group = np.asarray(devices, dtype=np.int64)
    baseline = np.asarray(reference, dtype=np.int64)
    if y.ndim != 1 or group.shape != y.shape or baseline.shape != y.shape:
        raise ValueError("truth, devices, and reference must share one-dimensional shape")
    names = tuple(candidates)
    if not names:
        raise ValueError("at least one candidate is required")
    predictions = []
    for name in names:
        values = np.asarray(candidates[name], dtype=np.int64)
        if values.shape != y.shape:
            raise ValueError(f"candidate {name} does not match truth")
        predictions.append(values)
    unique = np.unique(group)
    if unique.size < 2:
        raise ValueError("device bootstrap requires at least two physical devices")
    effects = np.empty((unique.size, len(names)), dtype=np.float64)
    reference_correct = baseline == y
    for row, device in enumerate(unique):
        local = group == device
        reference_accuracy = np.mean(reference_correct[local])
        for column, values in enumerate(predictions):
            effects[row, column] = float(
                np.mean(values[local] == y[local]) - reference_accuracy
            )
    return np.asarray(unique, dtype=np.int64), effects


def _standard_error(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    count = values.shape[axis]
    if count < 2:
        raise ValueError("at least two clusters are required")
    return np.std(values, axis=axis, ddof=1) / np.sqrt(float(count))


def simultaneous_cluster_lower_bounds(
    families: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Studentized shared-resample max-t lower bounds within/across families.

    Each family is a ``clusters x hypotheses`` effect matrix. A single cluster
    index matrix is used for every hypothesis in that family, preserving the
    dependence induced by evaluating frozen banks on the same physical devices.
    Families are independently resampled and aligned by replicate for the
    global max-t distribution.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if not families:
        raise ValueError("at least one family is required")
    rng = np.random.default_rng(int(seed))
    prepared: dict[str, dict[str, Any]] = {}
    all_studentized = []
    for name, raw in families.items():
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
            raise ValueError(f"family {name} must be clusters x hypotheses")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"family {name} contains non-finite effects")
        clusters, hypotheses = values.shape
        point = np.mean(values, axis=0)
        standard_error = _standard_error(values, axis=0)
        sampled = rng.integers(0, clusters, size=(replicates, clusters))
        bootstrap_values = values[sampled, :]
        bootstrap_mean = np.mean(bootstrap_values, axis=1)
        bootstrap_se = _standard_error(bootstrap_values, axis=1)
        floor = np.maximum(
            standard_error * 1.0e-8,
            np.finfo(np.float64).eps,
        )
        stabilized = np.maximum(bootstrap_se, floor[None, :])
        studentized = (bootstrap_mean - point[None, :]) / stabilized
        family_max = np.max(studentized, axis=1)
        critical = float(np.quantile(family_max, confidence))
        lower = point - critical * standard_error
        minimum_distribution = np.min(bootstrap_mean, axis=1)
        digest = hashlib.sha256(
            np.ascontiguousarray(sampled).view(np.uint8)
        ).hexdigest()
        prepared[name] = {
            "clusters": int(clusters),
            "hypotheses": int(hypotheses),
            "point": point,
            "standard_error": standard_error,
            "bootstrap_mean": bootstrap_mean,
            "studentized": studentized,
            "family_critical_value": critical,
            "family_simultaneous_lower": lower,
            "minimum_point": float(np.min(point)),
            "minimum_lower_one_sided": float(
                np.quantile(minimum_distribution, 1.0 - confidence)
            ),
            "minimum_lower_two_sided": float(
                np.quantile(minimum_distribution, (1.0 - confidence) / 2.0)
            ),
            "minimum_upper_two_sided": float(
                np.quantile(
                    minimum_distribution, 1.0 - (1.0 - confidence) / 2.0
                )
            ),
            "minimum_probability_positive": float(
                np.mean(minimum_distribution > 0.0)
            ),
            "shared_cluster_index_sha256": digest,
            "zero_observed_se_hypotheses": int(np.sum(standard_error == 0.0)),
            "stabilized_bootstrap_cells": int(
                np.sum(bootstrap_se < floor[None, :])
            ),
        }
        all_studentized.append(studentized)
    global_max = np.max(np.concatenate(all_studentized, axis=1), axis=1)
    global_critical = float(np.quantile(global_max, confidence))
    output_families = {}
    for name, item in prepared.items():
        global_lower = item["point"] - global_critical * item["standard_error"]
        output_families[name] = {
            key: value
            for key, value in item.items()
            if key not in {"bootstrap_mean", "studentized"}
        }
        output_families[name]["global_simultaneous_lower"] = global_lower
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "global_critical_value": global_critical,
        "families": output_families,
    }
