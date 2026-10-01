"""Run the pre-frozen Track C M2R1 exact-projective single-seed kill test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from czrf.passive_quotient import (  # noqa: E402
    fit_group_standardizer,
    projective_intensity_features,
    projective_measurement_bank,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "track_c_v1"
    / "M2R1_SINGLE_SEED_V1_20260827_150341.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def decode_bytes(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(values).reshape(-1)
    ]


def normalized_rows(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(source) else np.float64
    matrix = np.asarray(source, dtype=dtype)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).tiny):
        raise RuntimeError("normalization encountered zero or non-finite row energy")
    return matrix / norms


def selection_digest(indices: np.ndarray, hashes: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index, value in zip(indices, decode_bytes(hashes), strict=True):
        digest.update(f"{int(index)}:{value}\n".encode("utf-8"))
    return digest.hexdigest()


def complex_from_real_coordinates(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] % 2:
        raise ValueError("complex real coordinates must have even width")
    half = matrix.shape[1] // 2
    return normalized_rows(matrix[:, :half] + 1j * matrix[:, half:])


def nested_sketches(
    complex_unit: np.ndarray,
    *,
    maximum_dimension: int,
    prefixes: list[int],
    seed: int,
    expected_hash: str,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bank = projective_measurement_bank(
        complex_unit.shape[1], maximum_dimension, seed=seed
    )
    if bank.sha256 != expected_hash:
        raise RuntimeError(f"{label} frozen measurement-bank hash mismatch")
    maximum = projective_intensity_features(complex_unit, bank).astype(np.float32)
    output: dict[str, np.ndarray] = {}
    for dimension in prefixes:
        if dimension == maximum_dimension:
            output[f"m{dimension}"] = maximum
        else:
            output[f"m{dimension}"] = (
                np.sqrt(maximum_dimension / float(dimension))
                * maximum[:, :dimension]
            ).astype(np.float32)
    return output, {
        "seed": int(seed),
        "input_dimension": int(complex_unit.shape[1]),
        "maximum_output_dimension": int(maximum_dimension),
        "prefixes": [int(value) for value in prefixes],
        "sha256": bank.sha256,
    }


def load_m2_harmonic_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for setting in ("S3_eq0", "S3_eq1", "S4"):
            local = {
                "selected_indices": np.asarray(
                    payload[f"{setting}_selected_indices"], dtype=np.int64
                ),
                "labels": np.asarray(payload[f"{setting}_labels"], dtype=np.int64),
                "day": np.asarray(payload[f"{setting}_day"], dtype=np.int64),
                "A0_complex_unit": np.asarray(
                    payload[f"{setting}_harmonic_complex"], dtype=np.float64
                ),
                "A0_left_gram": np.asarray(
                    payload[f"{setting}_harmonic_projective"], dtype=np.float64
                ),
            }
            if setting.startswith("S3"):
                local["receiver"] = np.asarray(
                    payload[f"{setting}_receiver"], dtype=np.int64
                )
            else:
                local["file_id"] = np.asarray(
                    payload[f"{setting}_file_id"], dtype=np.int64
                )
            output[setting] = local
    return output


def augment_s3(
    data: dict[str, np.ndarray],
    setting: str,
    cache_path: Path,
    expected_selection_hash: str,
) -> dict[str, Any]:
    eq = int(setting[-1])
    selected = data["selected_indices"]
    with h5py.File(cache_path, "r") as handle:
        group = handle[f"eq{eq}/development"]
        hashes = np.asarray(group["selection_sha256"][selected])
        labels = np.asarray(group["tx_position"][selected], dtype=np.int64)
        receiver = np.asarray(group["rx_position"][selected], dtype=np.int64)
        day = np.asarray(group["day_position"][selected], dtype=np.int64)
        x_rms = np.asarray(group["x_rms"][selected], dtype=np.float64)
    observed_digest = selection_digest(selected, hashes)
    if observed_digest != expected_selection_hash:
        raise RuntimeError(f"{setting} selection hash changed")
    if not (
        np.array_equal(labels, data["labels"])
        and np.array_equal(receiver, data["receiver"])
        and np.array_equal(day, data["day"])
    ):
        raise RuntimeError(f"{setting} source metadata differs from M2 harmonic cache")
    iq_components = x_rms.reshape(-1, 256, 2)
    full_iq = normalized_rows(iq_components[:, :, 0] + 1j * iq_components[:, :, 1])
    raw_real = normalized_rows(x_rms)
    fft_magnitude = normalized_rows(np.abs(np.fft.fft(full_iq, axis=1)))
    groups = np.asarray(
        [
            f"{setting}:tx{int(a)}:rx{int(b)}:day{int(c)}"
            for a, b, c in zip(labels, receiver, day, strict=True)
        ],
        dtype=object,
    )
    return {
        **data,
        "groups": groups,
        "selection_sha256": observed_digest,
        "A1_complex": full_iq,
        "blocks": {
            "raw_real_unit_512": raw_real,
            "fft_magnitude_256": fft_magnitude,
            "A0_complex_unit": data["A0_complex_unit"],
            "A0_left_gram": data["A0_left_gram"],
            "A1_complex_unit": np.concatenate(
                (full_iq.real, full_iq.imag), axis=1
            ),
        },
    }


def augment_s4(
    data: dict[str, np.ndarray],
    feature_cache_path: Path,
    residual_cache_path: Path,
    expected_selection_hash: str,
) -> dict[str, Any]:
    selected = data["selected_indices"]
    with h5py.File(feature_cache_path, "r") as handle:
        roles = np.asarray(handle["role"][selected], dtype=np.int64)
        if np.any(roles > 1):
            raise RuntimeError("forbidden S4 role entered M2R1")
        labels = np.asarray(handle["device"][selected], dtype=np.int64)
        day = np.asarray(handle["day"][selected], dtype=np.int64)
        file_id = np.asarray(handle["file_id"][selected], dtype=np.int64)
        hashes = np.asarray(handle["packet_sha256"][selected])
        fft_features = np.asarray(handle["fft"][selected], dtype=np.float64)
    observed_digest = selection_digest(selected, hashes)
    if observed_digest != expected_selection_hash:
        raise RuntimeError("S4 selection hash changed")
    if not (
        np.array_equal(labels, data["labels"])
        and np.array_equal(day, data["day"])
        and np.array_equal(file_id, data["file_id"])
    ):
        raise RuntimeError("S4 source metadata differs from M2 harmonic cache")
    with np.load(residual_cache_path, allow_pickle=False) as payload:
        residual_indices = np.asarray(payload["cache_indices"], dtype=np.int64)
        order = np.argsort(residual_indices)
        sorted_indices = residual_indices[order]
        positions = np.searchsorted(sorted_indices, selected)
        if np.any(positions >= sorted_indices.size) or not np.array_equal(
            sorted_indices[positions], selected
        ):
            raise RuntimeError("S4 selected row absent from residual cache")
        residual = np.asarray(payload["residual"][order[positions]], dtype=np.float64)
    groups = np.asarray([f"file:{int(value)}" for value in file_id], dtype=object)
    return {
        **data,
        "groups": groups,
        "selection_sha256": observed_digest,
        "blocks": {
            "standardized_fft_2048": fft_features,
            "dechirped_residual_384": residual,
            "A0_complex_unit": data["A0_complex_unit"],
            "A0_left_gram": data["A0_left_gram"],
        },
    }


def build_folds(data: dict[str, Any], fold_protocol: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    day = np.asarray(data["day"], dtype=np.int64)
    if "receiver" in data:
        receiver = np.asarray(data["receiver"], dtype=np.int64)
        for item in fold_protocol:
            held = np.isin(receiver, np.asarray(item["receiver_group"], dtype=np.int64))
            validation = (day == int(item["target_day"])) & held
            training = (day != int(item["target_day"])) & ~held
            folds.append(
                {
                    "fold_id": item["fold_id"],
                    "training": training,
                    "validation": validation,
                }
            )
    else:
        for item in fold_protocol:
            validation = day == int(item["target_day"])
            folds.append(
                {
                    "fold_id": item["fold_id"],
                    "training": ~validation,
                    "validation": validation,
                }
            )
    coverage = np.zeros(day.size, dtype=np.int64)
    for fold in folds:
        training = fold["training"]
        validation = fold["validation"]
        if not np.any(training) or not np.any(validation) or np.any(training & validation):
            raise RuntimeError("M2R1 fold is empty or overlapping")
        coverage += validation.astype(np.int64)
    if not np.all(coverage == 1):
        raise RuntimeError("M2R1 folds do not partition selected rows")
    return folds


def transform_blocks(
    blocks: dict[str, np.ndarray],
    components: list[str],
    training: np.ndarray,
    validation: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_rows = np.flatnonzero(training)
    validation_rows = np.flatnonzero(validation)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    ledger: list[dict[str, Any]] = []
    for name in components:
        block = np.asarray(blocks[name], dtype=np.float64)
        state = fit_group_standardizer(
            block[train_rows],
            groups[train_rows],
            forbidden_groups=groups[validation_rows],
        )
        train_value = state.transform(block[train_rows])
        validation_value = state.transform(block[validation_rows])
        if len(components) > 1:
            train_value /= np.sqrt(block.shape[1])
            validation_value /= np.sqrt(block.shape[1])
        train_parts.append(train_value)
        validation_parts.append(validation_value)
        ledger.append(
            {
                "block": name,
                "dimension": int(block.shape[1]),
                "fit_group_sha256": state.fit_group_sha256,
                "fit_groups": len(state.fit_groups),
            }
        )
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(validation_parts, axis=1),
        ledger,
    )


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    present = np.unique(truth)
    recalls = np.asarray(
        [np.mean(prediction[truth == label] == label) for label in present]
    )
    return {
        "rows": int(truth.size),
        "present_classes": int(present.size),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1_present_classes": float(
            f1_score(truth, prediction, labels=present, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "worst_present_device_recall": float(np.min(recalls)),
    }


def aggregate(rows: list[dict[str, Any]], truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    names = (
        "accuracy",
        "macro_f1_present_classes",
        "balanced_accuracy",
        "worst_present_device_recall",
    )
    output = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        output[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "fold_count": len(rows),
        "feature_dimension": int(rows[0]["feature_dimension"]),
        "classifier_parameter_count": int(rows[0]["classifier_parameter_count"]),
        "metrics": output,
        "pooled_oof": metrics(truth, prediction),
        "all_groups_disjoint": bool(all(row["group_overlap_count"] == 0 for row in rows)),
        "all_features_finite": bool(all(row["features_finite"] for row in rows)),
        "all_classifiers_complete": bool(all(row["classifier_complete"] for row in rows)),
    }


def evaluate(
    setting: str,
    data: dict[str, Any],
    folds: list[dict[str, Any]],
    candidates: dict[str, list[str]],
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    groups = np.asarray(data["groups"], dtype=object)
    summaries: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for candidate, components in candidates.items():
        print(f"evaluate {setting} / {candidate}", flush=True)
        oof = np.full(labels.size, -1, dtype=np.int64)
        fold_rows = []
        for fold in folds:
            started = time.perf_counter()
            training = np.asarray(fold["training"], dtype=bool)
            validation = np.asarray(fold["validation"], dtype=bool)
            train_groups = set(str(value) for value in groups[training])
            validation_groups = set(str(value) for value in groups[validation])
            overlap = train_groups.intersection(validation_groups)
            train_x, validation_x, scaler_ledger = transform_blocks(
                data["blocks"], components, training, validation, groups
            )
            finite = bool(
                np.all(np.isfinite(train_x)) and np.all(np.isfinite(validation_x))
            )
            classifier = RidgeClassifier(alpha=alpha)
            classifier.fit(train_x, labels[training])
            prediction = np.asarray(classifier.predict(validation_x), dtype=np.int64)
            oof[validation] = prediction
            train_classes = np.unique(labels[training])
            validation_classes = np.unique(labels[validation])
            complete = bool(
                set(validation_classes.tolist()).issubset(train_classes.tolist())
                and np.array_equal(classifier.classes_, train_classes)
                and prediction.shape == labels[validation].shape
            )
            row = {
                "setting": setting,
                "representation": candidate,
                "components": "+".join(components),
                "fold_id": fold["fold_id"],
                "train_rows": int(np.sum(training)),
                "validation_rows": int(np.sum(validation)),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap_count": len(overlap),
                "features_finite": finite,
                "classifier_complete": complete,
                "feature_dimension": int(train_x.shape[1]),
                "classifier_parameter_count": int(
                    classifier.coef_.size + classifier.intercept_.size
                ),
                "elapsed_seconds": float(time.perf_counter() - started),
                "scaler_ledger": scaler_ledger,
                **metrics(labels[validation], prediction),
            }
            raw_rows.append(row)
            fold_rows.append(row)
        if np.any(oof < 0):
            raise RuntimeError(f"{setting}/{candidate} has incomplete OOF predictions")
        summaries[candidate] = aggregate(fold_rows, labels, oof)
        predictions[candidate] = oof
    return summaries, raw_rows, predictions


def empirical_secant_audit(
    complex_unit: np.ndarray,
    features: np.ndarray,
    *,
    seed: int,
    pairs: int = 5000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = complex_unit.shape[0]
    first = rng.integers(0, count, size=pairs)
    second = rng.integers(0, count, size=pairs)
    same = first == second
    while np.any(same):
        second[same] = rng.integers(0, count, size=int(np.sum(same)))
        same = first == second
    inner = np.sum(complex_unit[first].conj() * complex_unit[second], axis=1)
    projector_frobenius = np.sqrt(
        2.0 * np.maximum(0.0, 1.0 - np.minimum(1.0, np.abs(inner) ** 2))
    )
    feature_distance = np.linalg.norm(features[first] - features[second], axis=1)
    ratio = feature_distance / np.maximum(
        projector_frobenius, np.finfo(np.float64).tiny
    )
    return {
        "pairs": int(pairs),
        "seed": int(seed),
        "ratio_q01": float(np.quantile(ratio, 0.01)),
        "ratio_q10": float(np.quantile(ratio, 0.10)),
        "ratio_median": float(np.median(ratio)),
        "ratio_q90": float(np.quantile(ratio, 0.90)),
        "near_collision_fraction_below_1e_3": float(np.mean(ratio < 1.0e-3)),
    }


def candidate_map(setting: str, strongest: str) -> dict[str, list[str]]:
    if setting.startswith("S3"):
        return {
            "raw_real_unit_512": ["raw_real_unit_512"],
            "fft_magnitude_256": ["fft_magnitude_256"],
            "A0_complex_unit": ["A0_complex_unit"],
            "A0_left_gram": ["A0_left_gram"],
            "A0_sketch_m256": ["A0_sketch_m256"],
            "A0_sketch_m512": ["A0_sketch_m512"],
            "A1_complex_unit": ["A1_complex_unit"],
            "A1_sketch_m256": ["A1_sketch_m256"],
            "A1_sketch_m512": ["A1_sketch_m512"],
            "A1_sketch_m1024": ["A1_sketch_m1024"],
            "strongest_fixed_plus_primary_exact_sketch_equal_block": [
                strongest,
                "A1_sketch_m1024",
            ],
        }
    return {
        "standardized_fft_2048": ["standardized_fft_2048"],
        "dechirped_residual_384": ["dechirped_residual_384"],
        "equal_block_hybrid_2432": [
            "standardized_fft_2048",
            "dechirped_residual_384",
        ],
        "A0_complex_unit": ["A0_complex_unit"],
        "A0_left_gram": ["A0_left_gram"],
        "A0_sketch_m256": ["A0_sketch_m256"],
        "A0_sketch_m512": ["A0_sketch_m512"],
        "strongest_fixed_plus_primary_exact_sketch_equal_block": [
            "standardized_fft_2048",
            "dechirped_residual_384",
            "A0_sketch_m512",
        ],
    }


def macro(summary: dict[str, Any], name: str, statistic: str = "mean") -> float:
    return float(summary[name]["metrics"]["macro_f1_present_classes"][statistic])


def adjudicate(
    summaries: dict[str, dict[str, Any]],
    strongest: dict[str, str],
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    s4 = summaries["S4"]
    s4_quotient_delta = macro(s4, "A0_sketch_m512") - macro(s4, "A0_left_gram")
    s4_fixed_delta = macro(s4, "A0_sketch_m512") - macro(s4, strongest["S4"])
    s4_complement_min_delta = (
        macro(
            s4,
            "strongest_fixed_plus_primary_exact_sketch_equal_block",
            "minimum",
        )
        - macro(s4, strongest["S4"], "minimum")
    )
    s4_quotient_pass = bool(
        s4_quotient_delta
        >= float(thresholds["S4_sketch512_minus_left_gram_mean_macro_f1_at_least"])
    )
    s4_fixed_pass = bool(
        s4_fixed_delta
        >= float(
            thresholds[
                "S4_sketch512_minus_strongest_fixed_mean_macro_f1_at_least"
            ]
        )
    )
    s4_complement_pass = bool(
        s4_complement_min_delta
        >= float(
            thresholds[
                "S4_or_complement_minimum_fold_macro_f1_gain_over_fixed_at_least"
            ]
        )
    )
    s4_primary = bool(s4_quotient_pass and (s4_fixed_pass or s4_complement_pass))

    s3_gains = {}
    s3_noninferior = []
    for setting in ("S3_eq0", "S3_eq1"):
        local = summaries[setting]
        gain = macro(local, "A1_sketch_m1024") - macro(local, "A0_sketch_m512")
        fixed_delta = macro(local, "A1_sketch_m1024") - macro(
            local, strongest[setting]
        )
        s3_gains[setting] = {
            "A1_minus_A0_mean_macro_f1": gain,
            "A1_minus_fixed_mean_macro_f1": fixed_delta,
            "front_end_gain_pass": bool(
                gain
                >= float(
                    thresholds[
                        "S3_A1_sketch1024_minus_A0_sketch512_mean_macro_f1_at_least_each"
                    ]
                )
            ),
            "noninferiority_pass": bool(
                fixed_delta
                >= float(thresholds["S3_A1_sketch1024_noninferior_to_fixed_margin"])
            ),
        }
        if s3_gains[setting]["noninferiority_pass"]:
            s3_noninferior.append(setting)
    both_front_end = bool(all(item["front_end_gain_pass"] for item in s3_gains.values()))
    s3_noninferiority_gate = bool(
        len(s3_noninferior) >= int(thresholds["S3_noninferior_in_at_least_settings"])
    )

    parameter = {
        "S3_eq0": macro(summaries["S3_eq0"], "A1_sketch_m256")
        - macro(summaries["S3_eq0"], "A1_sketch_m1024"),
        "S3_eq1": macro(summaries["S3_eq1"], "A1_sketch_m256")
        - macro(summaries["S3_eq1"], "A1_sketch_m1024"),
        "S4": macro(summaries["S4"], "A0_sketch_m256")
        - macro(summaries["S4"], "A0_sketch_m512"),
    }
    parameter_pass = {
        setting: bool(
            value
            >= float(
                thresholds["parameter_matched_m256_minus_primary_mean_macro_f1_at_least"]
            )
        )
        for setting, value in parameter.items()
    }
    all_rows = [candidate for local in summaries.values() for candidate in local.values()]
    implementation_pass = bool(
        all(item["all_groups_disjoint"] for item in all_rows)
        and all(item["all_features_finite"] for item in all_rows)
        and all(item["all_classifiers_complete"] for item in all_rows)
    )
    cross_source = bool(s4_primary and both_front_end and s3_noninferiority_gate)
    if not implementation_pass or not s4_primary:
        decision = "STOP_TRACK_C_PERFORMANCE_ROUTE"
    elif cross_source:
        decision = "CONTINUE_TO_THREE_SEED_CONFIRMATION"
    else:
        decision = "SCOPE_S4_AND_CONFIRM_THREE_SEEDS"
    gates = {
        "implementation_and_leakage_pass": implementation_pass,
        "S4": {
            "sketch512_minus_left_gram_mean_macro_f1": s4_quotient_delta,
            "sketch512_minus_fixed_mean_macro_f1": s4_fixed_delta,
            "complement_minus_fixed_minimum_fold_macro_f1": s4_complement_min_delta,
            "quotient_gain_pass": s4_quotient_pass,
            "fixed_noninferiority_pass": s4_fixed_pass,
            "complement_worst_fold_pass": s4_complement_pass,
            "primary_gate_pass": s4_primary,
        },
        "S3": {
            "settings": s3_gains,
            "both_front_end_gain_pass": both_front_end,
            "noninferiority_settings": s3_noninferior,
            "noninferiority_gate_pass": s3_noninferiority_gate,
        },
        "parameter_efficiency": {
            "m256_minus_primary": parameter,
            "pass_by_setting": parameter_pass,
            "claim_allowed_settings": sorted(
                setting for setting, passed in parameter_pass.items() if passed
            ),
            "does_not_control_core_decision": True,
        },
        "cross_source_single_seed_gate_pass": cross_source,
    }
    return gates, decision


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Track C M2R1 Single-Seed Kill Test",
        "",
        f"- Decision: `{result['decision']}`",
        "- Scope: retrospective development-only; seed 20260827; not confirmation.",
        "",
        "## Summary",
        "",
        "| Setting | Representation | Mean Macro-F1 | Minimum-fold Macro-F1 | Dimension | Parameters |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for setting, local in result["summaries"].items():
        for name, item in local.items():
            metric = item["metrics"]["macro_f1_present_classes"]
            lines.append(
                f"| {setting} | {name} | {metric['mean']:.4f} | "
                f"{metric['minimum']:.4f} | {item['feature_dimension']} | "
                f"{item['classifier_parameter_count']} |"
            )
    gates = result["gates"]
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
            f"1. S4 quotient rescue delta: {gates['S4']['sketch512_minus_left_gram_mean_macro_f1']:+.4f}; pass={gates['S4']['quotient_gain_pass']}.",
            f"2. S4 fixed delta: {gates['S4']['sketch512_minus_fixed_mean_macro_f1']:+.4f}; noninferior={gates['S4']['fixed_noninferiority_pass']}.",
            f"3. S4 complement minimum-fold delta: {gates['S4']['complement_minus_fixed_minimum_fold_macro_f1']:+.4f}; pass={gates['S4']['complement_worst_fold_pass']}.",
            f"4. S3 both front-end gains pass={gates['S3']['both_front_end_gain_pass']}; noninferiority settings={', '.join(gates['S3']['noninferiority_settings']) or 'none'}.",
            f"5. Leakage/finite/completion pass={gates['implementation_and_leakage_pass']}.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if result["decision"] == "CONTINUE_TO_THREE_SEED_CONFIRMATION":
        lines.append(
            "The single-seed cross-source gate passed. This authorizes, but does not replace, the pre-frozen two additional seed replications."
        )
    elif result["decision"] == "SCOPE_S4_AND_CONFIRM_THREE_SEEDS":
        lines.append(
            "The corrected quotient is viable only for the S4 mechanism at this stage. Run the two additional seeds for S4; do not restore the four-source claim."
        )
    else:
        lines.append(
            "The corrected quotient did not rescue the S4 anchor under the frozen gate. Stop the Track C performance route and retain the result as a negative mechanism audit."
        )
    lines.extend(
        [
            "",
            "The valid M2 failure remains permanent. This single-seed run is development evidence and cannot itself support a stochastic-sketch paper claim.",
            "",
            f"Raw fold table: `{result['artifacts']['raw_table']['path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_M2R1_FEATURE_EXTRACTION_OR_SCORING":
        raise RuntimeError("M2R1 protocol is not frozen before scoring")
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if sha256(path) != parent["sha256"]:
            raise RuntimeError(f"parent artifact changed: {relative(path)}")
    m2 = json.loads((ROOT / config["parents"]["valid_m2_result"]["path"]).read_text(encoding="utf-8"))
    c206 = json.loads((ROOT / config["parents"]["c206_failure_audit"]["path"]).read_text(encoding="utf-8"))
    m2r0 = json.loads((ROOT / config["parents"]["m2r0_sanity"]["path"]).read_text(encoding="utf-8"))
    if m2["decision"] != config["parents"]["valid_m2_result"]["required_decision"]:
        raise RuntimeError("valid M2 parent decision mismatch")
    if c206["decision"] != config["parents"]["c206_failure_audit"]["required_decision"]:
        raise RuntimeError("C206 parent decision mismatch")
    if m2r0["status"] != config["parents"]["m2r0_sanity"]["required_status"]:
        raise RuntimeError("M2R0 parent status mismatch")
    for key in ("harmonic_feature_cache", "S3_source_cache", "S4_feature_cache", "S4_residual_cache"):
        path = ROOT / config["data"][key]["path"]
        if sha256(path) != config["data"][key]["sha256"]:
            raise RuntimeError(f"frozen M2R1 input changed: {relative(path)}")

    harmonic_path = ROOT / config["data"]["harmonic_feature_cache"]["path"]
    raw = load_m2_harmonic_cache(harmonic_path)
    settings: dict[str, dict[str, Any]] = {}
    for setting in ("S3_eq0", "S3_eq1"):
        settings[setting] = augment_s3(
            raw[setting],
            setting,
            ROOT / config["data"]["S3_source_cache"]["path"],
            config["data"]["S3_source_cache"]["selection_sha256"][setting],
        )
    settings["S4"] = augment_s4(
        raw["S4"],
        ROOT / config["data"]["S4_feature_cache"]["path"],
        ROOT / config["data"]["S4_residual_cache"]["path"],
        config["data"]["S4_feature_cache"]["selection_sha256"],
    )

    bank_ledger = {}
    for setting, data in settings.items():
        if setting.startswith("S3"):
            a0_config = config["measurement_banks"]["A0_S3"]
            a1_config = config["measurement_banks"]["A1_S3"]
            a0_complex = complex_from_real_coordinates(data["A0_complex_unit"])
            a0_features, a0_ledger = nested_sketches(
                a0_complex,
                maximum_dimension=int(a0_config["maximum_output_dimension"]),
                prefixes=[int(value) for value in a0_config["candidate_prefixes"]],
                seed=int(a0_config["seed"]),
                expected_hash=a0_config["sha256"],
                label=f"{setting}/A0",
            )
            a1_features, a1_ledger = nested_sketches(
                data["A1_complex"],
                maximum_dimension=int(a1_config["maximum_output_dimension"]),
                prefixes=[int(value) for value in a1_config["candidate_prefixes"]],
                seed=int(a1_config["seed"]),
                expected_hash=a1_config["sha256"],
                label=f"{setting}/A1",
            )
            for suffix, values in a0_features.items():
                data["blocks"][f"A0_sketch_{suffix}"] = values
            for suffix, values in a1_features.items():
                data["blocks"][f"A1_sketch_{suffix}"] = values
            bank_ledger[setting] = {"A0": a0_ledger, "A1": a1_ledger}
        else:
            a0_config = config["measurement_banks"]["A0_S4"]
            a0_complex = complex_from_real_coordinates(data["A0_complex_unit"])
            a0_features, a0_ledger = nested_sketches(
                a0_complex,
                maximum_dimension=int(a0_config["maximum_output_dimension"]),
                prefixes=[int(value) for value in a0_config["candidate_prefixes"]],
                seed=int(a0_config["seed"]),
                expected_hash=a0_config["sha256"],
                label="S4/A0",
            )
            for suffix, values in a0_features.items():
                data["blocks"][f"A0_sketch_{suffix}"] = values
            bank_ledger[setting] = {"A0": a0_ledger}
        if not all(np.all(np.isfinite(value)) for value in data["blocks"].values()):
            raise RuntimeError(f"{setting} contains a non-finite candidate block")

    folds = {
        setting: build_folds(data, m2["fold_protocol"][setting])
        for setting, data in settings.items()
    }
    strongest = config["representations"]["fixed"]
    summaries: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for setting, data in settings.items():
        local_summary, local_rows, local_predictions = evaluate(
            setting,
            data,
            folds[setting],
            candidate_map(setting, strongest[setting]),
            float(config["evaluator"]["alpha"]),
        )
        summaries[setting] = local_summary
        raw_rows.extend(local_rows)
        predictions[f"{setting}__truth"] = np.asarray(data["labels"], dtype=np.int64)
        predictions[f"{setting}__selected_indices"] = np.asarray(
            data["selected_indices"], dtype=np.int64
        )
        for name, values in local_predictions.items():
            predictions[f"{setting}__{name}"] = values

    secants: dict[str, Any] = {}
    for setting, data in settings.items():
        a0_complex = complex_from_real_coordinates(data["A0_complex_unit"])
        for dimension in (256, 512):
            secants[f"{setting}__A0_m{dimension}"] = empirical_secant_audit(
                a0_complex,
                np.asarray(data["blocks"][f"A0_sketch_m{dimension}"], dtype=np.float64),
                seed=20260827 + dimension,
            )
        if setting.startswith("S3"):
            for dimension in (256, 512, 1024):
                secants[f"{setting}__A1_m{dimension}"] = empirical_secant_audit(
                    data["A1_complex"],
                    np.asarray(data["blocks"][f"A1_sketch_m{dimension}"], dtype=np.float64),
                    seed=20260828 + dimension,
                )

    gates, decision = adjudicate(
        summaries, strongest, config["pre_frozen_gates"]
    )
    release_id = str(args.release_id)
    result_dir = ROOT / "results" / "track_c_v1"
    run_dir = ROOT / "runs" / "track_c_v1"
    report_dir = ROOT / "refine-logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"M2R1_SINGLE_SEED_{release_id}.json"
    table_path = result_dir / f"M2R1_RAW_TABLE_{release_id}.csv"
    feature_path = run_dir / f"M2R1_PROJECTIVE_FEATURES_{release_id}.npz"
    prediction_path = run_dir / f"M2R1_OOF_PREDICTIONS_{release_id}.npz"
    report_path = report_dir / f"TRACK_C_M2R1_SINGLE_SEED_{release_id}.md"
    for path in (result_path, table_path, feature_path, prediction_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    csv_rows = []
    for row in raw_rows:
        local = {key: value for key, value in row.items() if key != "scaler_ledger"}
        local["scaler_fit_group_sha256"] = ";".join(
            item["fit_group_sha256"] for item in row["scaler_ledger"]
        )
        csv_rows.append(local)
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(prediction_path, **predictions)
    feature_payload = {}
    for setting, data in settings.items():
        feature_payload[f"{setting}__selected_indices"] = data["selected_indices"]
        for name, values in data["blocks"].items():
            if "sketch" in name:
                feature_payload[f"{setting}__{name}"] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(feature_path, **feature_payload)

    result = {
        "protocol_id": config["protocol_id"],
        "release_id": release_id,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": config["scope"],
        "decision": decision,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "h5py": h5py.__version__,
        },
        "inputs": {
            "config": {"path": relative(config_path), "sha256": sha256(config_path)},
            "runner": {
                "path": relative(Path(__file__)),
                "sha256": sha256(Path(__file__)),
            },
            "parents": config["parents"],
        },
        "data_audit": {
            setting: {
                "selected_rows": int(data["labels"].size),
                "classes": int(np.unique(data["labels"]).size),
                "groups": int(np.unique(data["groups"]).size),
                "selection_sha256": data["selection_sha256"],
            }
            for setting, data in settings.items()
        },
        "measurement_bank_ledger": bank_ledger,
        "summaries": summaries,
        "empirical_projective_secants": secants,
        "gates": gates,
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "raw_table": {"path": relative(table_path), "sha256": sha256(table_path)},
            "projective_features": {
                "path": relative(feature_path),
                "sha256": sha256(feature_path),
            },
            "predictions": {
                "path": relative(prediction_path),
                "sha256": sha256(prediction_path),
            },
            "report": {"path": relative(report_path)},
        },
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(report_markdown(result), encoding="utf-8")
    latest = {
        result_path: result_dir / "M2R1_SINGLE_SEED.json",
        table_path: result_dir / "M2R1_RAW_TABLE.csv",
        feature_path: run_dir / "M2R1_PROJECTIVE_FEATURES.npz",
        prediction_path: run_dir / "M2R1_OOF_PREDICTIONS.npz",
        report_path: report_dir / "TRACK_C_M2R1_SINGLE_SEED.md",
    }
    for source, destination in latest.items():
        shutil.copy2(source, destination)
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"result: {relative(result_path)}", flush=True)
    print(f"report: {relative(report_path)}", flush=True)


if __name__ == "__main__":
    main()
