from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def normalize_rms(x: ComplexArray, target_rms: float = 1.0) -> ComplexArray:
    x = np.asarray(x, dtype=np.complex128)
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite signal.")
    return x * (target_rms / rms)


def lora_like_chirp(
    n_samples: int,
    drive_rms: float,
    direction: int = 1,
    bandwidth_fraction: float = 0.80,
    phase0: float = 0.0,
) -> ComplexArray:
    """Generate a deterministic CSS probe, not a bit-exact LoRa modem waveform."""
    t = np.arange(n_samples, dtype=np.float64)
    signed_bw = float(np.sign(direction) or 1) * bandwidth_fraction
    f0 = -0.5 * signed_bw
    slope = signed_bw / n_samples
    phase = 2.0 * np.pi * (f0 * t + 0.5 * slope * t**2) + phase0
    return normalize_rms(np.exp(1j * phase), drive_rms)


def dual_tone_probe(
    n_samples: int,
    drive_rms: float,
    f1: float,
    f2: float,
    phase2: float,
) -> ComplexArray:
    t = np.arange(n_samples, dtype=np.float64)
    x = np.exp(2j * np.pi * f1 * t) + 0.82 * np.exp(1j * (2 * np.pi * f2 * t + phase2))
    return normalize_rms(x, drive_rms)


def random_papr_probe(n_samples: int, drive_rms: float, rng: np.random.Generator) -> ComplexArray:
    qpsk = np.exp(1j * (np.pi / 4.0 + (np.pi / 2.0) * rng.integers(0, 4, n_samples)))
    shaped = signal.lfilter(np.array([0.18, 0.64, 0.18]), np.array([1.0]), qpsk)
    return normalize_rms(np.asarray(shaped, dtype=np.complex128), drive_rms)


def generate_probe_bank(
    n_samples: int,
    count: int,
    drive_rms: float,
    seed: int,
) -> list[ComplexArray]:
    rng = np.random.default_rng(seed)
    probes: list[ComplexArray] = []
    for j in range(count):
        family = j % 3
        if family == 0:
            probes.append(
                lora_like_chirp(
                    n_samples,
                    drive_rms,
                    direction=1 if (j // 3) % 2 == 0 else -1,
                    phase0=0.17 * j,
                )
            )
        elif family == 1:
            offset = 0.006 * (j // 3)
            probes.append(dual_tone_probe(n_samples, drive_rms, 0.075 + offset, 0.145 - offset, 0.31 * j))
        else:
            probes.append(random_papr_probe(n_samples, drive_rms, rng))
    return probes


def _base_coefficients(memory_length: int) -> dict[int, ComplexArray]:
    if memory_length < 1:
        raise ValueError("memory_length must be positive")
    templates = {
        1: np.array([1.0 + 0.0j, 0.10 + 0.018j, 0.038 - 0.012j, 0.014 + 0.004j]),
        3: np.array([-0.175 + 0.030j, -0.048 + 0.012j, -0.014 + 0.004j, -0.004 + 0.001j]),
        5: np.array([0.034 - 0.009j, 0.008 - 0.002j, 0.0025 + 0.0005j, 0.0008 + 0.0002j]),
    }
    result: dict[int, ComplexArray] = {}
    for order, values in templates.items():
        if memory_length <= len(values):
            result[order] = values[:memory_length].astype(np.complex128)
        else:
            tail_count = memory_length - len(values)
            tail = values[-1] * (0.45 ** np.arange(1, tail_count + 1))
            result[order] = np.concatenate([values, tail]).astype(np.complex128)
    return result


def generate_device_bank(
    device_count: int,
    memory_length: int,
    static_sigma: float,
    seed: int,
) -> list[dict[int, ComplexArray]]:
    rng = np.random.default_rng(seed)
    base = _base_coefficients(memory_length)
    devices: list[dict[int, ComplexArray]] = []
    for _ in range(device_count):
        coeffs: dict[int, ComplexArray] = {}
        for order, values in base.items():
            order_sigma = static_sigma * (0.30 if order == 1 else 1.0)
            perturb = (rng.normal(size=values.size) + 1j * rng.normal(size=values.size)) / np.sqrt(2.0)
            coeffs[order] = values * (1.0 + order_sigma * perturb)
        devices.append(coeffs)
    return devices


def perturb_state(
    coeffs: dict[int, ComplexArray],
    state_sigma: float,
    rng: np.random.Generator,
) -> dict[int, ComplexArray]:
    state: dict[int, ComplexArray] = {}
    for order, values in coeffs.items():
        perturb = (rng.normal(size=values.size) + 1j * rng.normal(size=values.size)) / np.sqrt(2.0)
        state[order] = values * (1.0 + state_sigma * perturb)
    return state


def memory_polynomial(x: ComplexArray, coeffs: dict[int, ComplexArray]) -> ComplexArray:
    x = np.asarray(x, dtype=np.complex128)
    y = np.zeros_like(x)
    for order, taps in coeffs.items():
        if order < 1 or order % 2 == 0:
            raise ValueError("Only positive odd baseband orders are supported.")
        for delay, coefficient in enumerate(np.asarray(taps, dtype=np.complex128)):
            delayed = np.zeros_like(x)
            if delay == 0:
                delayed[:] = x
            elif delay < x.size:
                delayed[delay:] = x[:-delay]
            y += coefficient * delayed * np.abs(delayed) ** (order - 1)
    return y


def generate_channel_taps(kind: str, rng: np.random.Generator) -> ComplexArray:
    amplitude = rng.uniform(0.88, 1.12)
    phase = rng.uniform(-np.pi, np.pi)
    a = amplitude * np.exp(1j * phase)
    scales = {
        "flat": (0.0, 0.0),
        "mild": (0.075, 0.030),
        "medium": (0.18, 0.080),
        "strong": (0.36, 0.180),
    }
    if kind not in scales:
        raise ValueError(f"Unknown channel kind: {kind}")
    s1, s2 = scales[kind]
    e1 = amplitude * s1 * np.exp(1j * rng.uniform(-np.pi, np.pi))
    e2 = amplitude * s2 * np.exp(1j * rng.uniform(-np.pi, np.pi))
    return np.array([a, e1, e2], dtype=np.complex128)


def apply_channel(x: ComplexArray, taps: ComplexArray) -> ComplexArray:
    return np.convolve(np.asarray(x, dtype=np.complex128), np.asarray(taps, dtype=np.complex128), mode="full")[: x.size]


def add_awgn(
    clean: ComplexArray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[ComplexArray, ComplexArray, float]:
    clean = np.asarray(clean, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(clean) ** 2))
    if signal_power <= 0.0:
        raise ValueError("Signal power must be positive.")
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=clean.size) + 1j * rng.normal(size=clean.size)
    )
    measured = 10.0 * np.log10(signal_power / float(np.mean(np.abs(noise) ** 2)))
    return clean + noise, np.asarray(noise, dtype=np.complex128), float(measured)


def normalized_probe_features(y: ComplexArray, band_edges: Iterable[float]) -> FloatArray:
    y = np.asarray(y, dtype=np.complex128)
    edges = np.asarray(list(band_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("band_edges must be a strictly increasing one-dimensional sequence.")
    spectrum = np.fft.fftshift(np.fft.fft(y, norm="ortho"))
    frequencies = np.fft.fftshift(np.fft.fftfreq(y.size))
    reference = float(np.linalg.norm(spectrum)) + np.finfo(float).eps
    ratios: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        include_right = np.isclose(right, edges[-1])
        mask = (frequencies >= left) & ((frequencies <= right) if include_right else (frequencies < right))
        ratios.append(float(np.linalg.norm(spectrum[mask]) / reference))
    half = y.size // 2
    ratios.append(float(np.linalg.norm(y[:half]) / (np.linalg.norm(y) + np.finfo(float).eps)))
    papr = float(np.max(np.abs(y) ** 2) / (np.mean(np.abs(y) ** 2) + np.finfo(float).eps))
    ratios.append(float(np.log1p(papr)))
    features = np.asarray(ratios, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Non-finite feature encountered.")
    return features


def _sample_signature(
    probes: list[ComplexArray],
    state_coeffs: dict[int, ComplexArray],
    channel_taps: ComplexArray,
    snr_db: float,
    band_edges: Iterable[float],
    rng: np.random.Generator,
) -> tuple[FloatArray, list[float]]:
    feature_blocks: list[FloatArray] = []
    measured_snr: list[float] = []
    for probe in probes:
        pa_output = memory_polynomial(probe, state_coeffs)
        channel_output = apply_channel(pa_output, channel_taps)
        observed, _, measured = add_awgn(channel_output, snr_db, rng)
        feature_blocks.append(normalized_probe_features(observed, band_edges))
        measured_snr.append(measured)
    return np.concatenate(feature_blocks), measured_snr


@dataclass(frozen=True)
class DatasetBundle:
    features: FloatArray
    labels: NDArray[np.int64]
    groups: NDArray[np.int64]
    measured_snr_db: FloatArray


def build_dataset(
    *,
    device_count: int,
    groups_per_device: int,
    replicas_per_group: int,
    probes: list[ComplexArray],
    memory_length: int,
    static_sigma: float,
    state_sigma: float,
    channel_kind: str,
    snr_db: float,
    band_edges: Iterable[float],
    seed: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    devices = generate_device_bank(device_count, memory_length, static_sigma, seed + 101)
    features: list[FloatArray] = []
    labels: list[int] = []
    groups: list[int] = []
    snrs: list[float] = []
    group_id = 0
    for device_id, coeffs in enumerate(devices):
        for _ in range(groups_per_device):
            state_coeffs = perturb_state(coeffs, state_sigma, rng)
            channel_taps = generate_channel_taps(channel_kind, rng)
            for _replica in range(replicas_per_group):
                signature, measured = _sample_signature(
                    probes,
                    state_coeffs,
                    channel_taps,
                    snr_db,
                    band_edges,
                    rng,
                )
                features.append(signature)
                labels.append(device_id)
                groups.append(group_id)
                snrs.append(float(np.mean(measured)))
            group_id += 1
    return DatasetBundle(
        features=np.vstack(features).astype(np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        measured_snr_db=np.asarray(snrs, dtype=np.float64),
    )


def grouped_split(
    labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    train_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(labels.size), labels, groups))
    if set(groups[train_idx]).intersection(set(groups[test_idx])):
        raise AssertionError("Group leakage detected.")
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def nearest_centroid_metrics(
    features: FloatArray,
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
) -> dict[str, object]:
    scaler = StandardScaler().fit(features[train_idx])
    train_x = scaler.transform(features[train_idx])
    test_x = scaler.transform(features[test_idx])
    train_y = labels[train_idx]
    test_y = labels[test_idx]
    classes = np.unique(train_y)
    centroids = np.vstack([train_x[train_y == cls].mean(axis=0) for cls in classes])
    distances = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
    predictions = classes[np.argmin(distances, axis=1)]
    return {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=classes).tolist(),
        "test_count": int(test_idx.size),
    }


def centroid_separation_metrics(
    features: FloatArray,
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
) -> dict[str, float]:
    """Report standardized within-class and nearest-other-centroid distances."""
    scaler = StandardScaler().fit(features[train_idx])
    train_x = scaler.transform(features[train_idx])
    test_x = scaler.transform(features[test_idx])
    train_y = labels[train_idx]
    test_y = labels[test_idx]
    classes = np.unique(train_y)
    centroids = np.vstack([train_x[train_y == cls].mean(axis=0) for cls in classes])
    class_to_index = {int(cls): idx for idx, cls in enumerate(classes)}
    distances = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
    own_index = np.asarray([class_to_index[int(label)] for label in test_y], dtype=np.int64)
    own_distance = distances[np.arange(test_y.size), own_index]
    other_distances = distances.copy()
    other_distances[np.arange(test_y.size), own_index] = np.inf
    nearest_other = np.min(other_distances, axis=1)
    centroid_pairs = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    centroid_pairs[np.eye(classes.size, dtype=bool)] = np.inf
    return {
        "mean_within_distance": float(np.mean(own_distance)),
        "mean_nearest_other_distance": float(np.mean(nearest_other)),
        "mean_distance_margin": float(np.mean(nearest_other - own_distance)),
        "min_inter_centroid_distance": float(np.min(centroid_pairs)),
    }


def channel_matrix(taps: ComplexArray, n_samples: int) -> ComplexArray:
    matrix = np.zeros((n_samples, n_samples), dtype=np.complex128)
    for row in range(n_samples):
        for delay, tap in enumerate(taps):
            column = row - delay
            if column >= 0:
                matrix[row, column] = tap
    return matrix


def channel_bound_diagnostic(x: ComplexArray, taps: ComplexArray) -> dict[str, float | bool]:
    x = np.asarray(x, dtype=np.complex128)
    matrix = channel_matrix(np.asarray(taps, dtype=np.complex128), x.size)
    a = complex(taps[0])
    residual = matrix - a * np.eye(x.size, dtype=np.complex128)
    residual_norm = float(np.linalg.norm(residual, ord=2))
    lhs = abs(float(np.linalg.norm(matrix @ x)) - abs(a) * float(np.linalg.norm(x)))
    rhs = residual_norm * float(np.linalg.norm(x))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "residual_norm": residual_norm,
        "holds": bool(lhs <= rhs + 1e-10),
    }
