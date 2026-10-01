from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .core import (
    ComplexArray,
    FloatArray,
    add_awgn,
    apply_channel,
    generate_channel_taps,
    memory_polynomial,
    normalize_rms,
    normalized_probe_features,
    perturb_state,
)


@dataclass(frozen=True)
class ExperimentDataset:
    features: dict[str, FloatArray]
    labels: NDArray[np.int64]
    groups: NDArray[np.int64]
    measured_snr_db: FloatArray


def raw_probe_features(y: ComplexArray, band_edges: Iterable[float]) -> FloatArray:
    y = np.asarray(y, dtype=np.complex128)
    edges = np.asarray(list(band_edges), dtype=np.float64)
    spectrum = np.fft.fftshift(np.fft.fft(y, norm="ortho"))
    frequencies = np.fft.fftshift(np.fft.fftfreq(y.size))
    values: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        include_right = np.isclose(right, edges[-1])
        mask = (frequencies >= left) & ((frequencies <= right) if include_right else (frequencies < right))
        values.append(float(np.log(np.linalg.norm(spectrum[mask]) + np.finfo(float).eps)))
    values.append(float(np.log(np.sqrt(np.mean(np.abs(y) ** 2)) + np.finfo(float).eps)))
    papr = float(np.max(np.abs(y) ** 2) / (np.mean(np.abs(y) ** 2) + np.finfo(float).eps))
    values.append(float(np.log1p(papr)))
    return np.asarray(values, dtype=np.float64)


def spectrum_probe_features(y: ComplexArray, bins: int = 7) -> FloatArray:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(y, norm="ortho")))
    reference = float(np.linalg.norm(spectrum)) + np.finfo(float).eps
    blocks = np.array_split(spectrum, bins)
    return np.asarray([np.linalg.norm(block) / reference for block in blocks], dtype=np.float64)


def _mp_design(probe: ComplexArray, orders: tuple[int, ...], memory_length: int) -> ComplexArray:
    columns: list[ComplexArray] = []
    for order in orders:
        for delay in range(memory_length):
            delayed = np.zeros_like(probe, dtype=np.complex128)
            if delay == 0:
                delayed[:] = probe
            else:
                delayed[delay:] = probe[:-delay]
            columns.append(delayed * np.abs(delayed) ** (order - 1))
    return np.column_stack(columns).astype(np.complex128)


def mp_ridge_features(
    observations: list[ComplexArray],
    probes: list[ComplexArray],
    ridge_alpha: float,
    orders: tuple[int, ...] = (1, 3, 5),
    memory_length: int = 4,
) -> FloatArray:
    design = np.vstack([_mp_design(probe, orders, memory_length) for probe in probes])
    target = np.concatenate(observations)
    gram = design.conj().T @ design
    beta = np.linalg.solve(
        gram + ridge_alpha * np.eye(gram.shape[0], dtype=np.complex128),
        design.conj().T @ target,
    )
    beta = beta / (beta[0] + np.finfo(float).eps)
    reduced = beta[1:]
    return np.concatenate([reduced.real, reduced.imag]).astype(np.float64)


def extract_feature_sets(
    observations: list[ComplexArray],
    probes: list[ComplexArray],
    band_edges: Iterable[float],
    ridge_alpha: float,
    requested_methods: Iterable[str] | None = None,
) -> dict[str, FloatArray]:
    requested = set(requested_methods or ["Power", "Spectrum", "MP-Ridge", "ProbeNorm-Ratio", "Raw-Probe"])
    result: dict[str, FloatArray] = {}
    if "Power" in requested:
        result["Power"] = np.asarray(
            [np.log(np.mean(np.abs(y) ** 2) + np.finfo(float).eps) for y in observations],
            dtype=np.float64,
        )
    if "Spectrum" in requested:
        result["Spectrum"] = np.concatenate([spectrum_probe_features(y) for y in observations])
    if "MP-Ridge" in requested:
        result["MP-Ridge"] = mp_ridge_features(observations, probes, ridge_alpha)
    if "ProbeNorm-Ratio" in requested:
        result["ProbeNorm-Ratio"] = np.concatenate(
            [normalized_probe_features(y, band_edges) for y in observations]
        )
    if "Raw-Probe" in requested:
        result["Raw-Probe"] = np.concatenate([raw_probe_features(y, band_edges) for y in observations])
    unknown = requested.difference(result)
    if unknown:
        raise ValueError(f"Unknown feature methods: {sorted(unknown)}")
    return result


def build_experiment_dataset(
    *,
    devices: list[dict[int, ComplexArray]],
    groups_per_device: int,
    replicas_per_group: int,
    probes: list[ComplexArray],
    state_sigma: float,
    channel_kind: str,
    snr_db: float,
    band_edges: Iterable[float],
    seed: int,
    ridge_alpha: float = 1e-3,
    equal_receive_power: bool = False,
    requested_methods: Iterable[str] | None = None,
) -> ExperimentDataset:
    rng = np.random.default_rng(seed)
    method_names = list(requested_methods or ["Power", "Spectrum", "MP-Ridge", "ProbeNorm-Ratio", "Raw-Probe"])
    feature_rows: dict[str, list[FloatArray]] = {name: [] for name in method_names}
    labels: list[int] = []
    groups: list[int] = []
    snrs: list[float] = []
    group_id = 0
    for device_id, coefficients in enumerate(devices):
        for _ in range(groups_per_device):
            state_coefficients = perturb_state(coefficients, state_sigma, rng)
            channel_taps = generate_channel_taps(channel_kind, rng)
            for _replica in range(replicas_per_group):
                observations: list[ComplexArray] = []
                measured: list[float] = []
                for probe in probes:
                    clean = apply_channel(memory_polynomial(probe, state_coefficients), channel_taps)
                    observed, _, actual_snr = add_awgn(clean, snr_db, rng)
                    if equal_receive_power:
                        observed = normalize_rms(observed, 1.0)
                    observations.append(observed)
                    measured.append(actual_snr)
                extracted = extract_feature_sets(
                    observations, probes, band_edges, ridge_alpha, requested_methods=method_names
                )
                for method, vector in extracted.items():
                    feature_rows[method].append(vector)
                labels.append(device_id)
                groups.append(group_id)
                snrs.append(float(np.mean(measured)))
            group_id += 1
    return ExperimentDataset(
        features={name: np.vstack(rows).astype(np.float64) for name, rows in feature_rows.items()},
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        measured_snr_db=np.asarray(snrs, dtype=np.float64),
    )


def stratified_group_split(
    labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    train_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    train_groups: set[int] = set()
    for label in np.unique(labels):
        label_groups = np.unique(groups[labels == label]).copy()
        rng.shuffle(label_groups)
        count = min(max(1, int(np.floor(train_fraction * label_groups.size))), label_groups.size - 1)
        train_groups.update(int(group) for group in label_groups[:count])
    train_mask = np.asarray([int(group) in train_groups for group in groups], dtype=bool)
    train_idx = np.flatnonzero(train_mask).astype(np.int64)
    test_idx = np.flatnonzero(~train_mask).astype(np.int64)
    if set(groups[train_idx]).intersection(set(groups[test_idx])):
        raise AssertionError("Group leakage detected.")
    if set(np.unique(labels[train_idx])) != set(np.unique(labels[test_idx])):
        raise AssertionError("Every class must occur in both train and test partitions.")
    return train_idx, test_idx


def _pooled_eer(test_labels: NDArray[np.int64], classes: NDArray[np.int64], scores: FloatArray) -> float:
    truth = (test_labels[:, None] == classes[None, :]).astype(np.int64).ravel()
    fpr, tpr, _ = roc_curve(truth, scores.ravel())
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5 * (fpr[index] + fnr[index]))


def evaluate_cross_domain(
    train_features: FloatArray,
    test_features: FloatArray,
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
) -> dict[str, object]:
    scaler = StandardScaler().fit(train_features[train_idx])
    train_x = scaler.transform(train_features[train_idx])
    test_x = scaler.transform(test_features[test_idx])
    train_y = labels[train_idx]
    test_y = labels[test_idx]
    classes = np.unique(train_y)
    centroids = np.vstack([train_x[train_y == label].mean(axis=0) for label in classes])
    distances = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
    predictions = classes[np.argmin(distances, axis=1)]
    return {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "eer": _pooled_eer(test_y, classes, -distances),
        "feature_dimension": int(train_features.shape[1]),
        "test_count": int(test_idx.size),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=classes).tolist(),
    }


def nominal_channel_residual(kind: str) -> float:
    scales = {
        "flat": (0.0, 0.0),
        "mild": (0.075, 0.030),
        "medium": (0.18, 0.080),
        "strong": (0.36, 0.180),
    }
    first, second = scales[kind]
    return float(np.hypot(first, second))
