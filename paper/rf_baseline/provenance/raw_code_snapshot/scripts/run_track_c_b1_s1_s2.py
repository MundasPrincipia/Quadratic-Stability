"""Score the pre-frozen S1/S2 portion of Track C B1."""

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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_track_c_m2r1_single_seed import (  # noqa: E402
    empirical_secant_audit,
    nested_sketches,
)
from czrf.passive_quotient import (  # noqa: E402
    complex_real_coordinates,
    fit_group_standardizer,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "track_c_v1" / "B1_FOUR_SOURCE_V1_20260827_154500.json"
)
DEFAULT_AMENDMENT = (
    ROOT
    / "configs"
    / "track_c_v1"
    / "B1_V1_SEED_AND_SOURCE_AGGREGATION_AMENDMENT_20260827_160500.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in np.asarray(values).reshape(-1)
        ],
        dtype=object,
    )


def load_source(cache_path: Path, source: str) -> dict[str, Any]:
    with h5py.File(cache_path, "r") as handle:
        group = handle[source]
        mapping = json.loads(str(group.attrs["environment_to_code"]))
        complex_front = np.asarray(group["complex_front"], dtype=np.complex128)
        return {
            "labels": np.asarray(group["label"], dtype=np.int64),
            "environment": np.asarray(group["environment_code"], dtype=np.int64),
            "environment_to_code": {str(key): int(value) for key, value in mapping.items()},
            "groups": decode_array(np.asarray(group["group_id"])),
            "complex_front": complex_front,
            "blocks": {
                "strongest_fixed": np.asarray(group["fixed"], dtype=np.float64),
                "segmented_fourier_magnitude_256": np.asarray(
                    group["segmented_fourier_magnitude"], dtype=np.float64
                ),
                "complex_unit_512": complex_real_coordinates(complex_front),
                "left_gram_256": np.asarray(group["left_gram"], dtype=np.float64),
            },
        }


def build_folds(source: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = np.asarray(data["environment"], dtype=np.int64)
    mapping = data["environment_to_code"]
    folds: list[dict[str, Any]] = []
    if source == "S1":
        for name, code in sorted(mapping.items(), key=lambda item: item[1]):
            validation = environment == code
            folds.append(
                {
                    "fold_id": f"leave_environment_{name}",
                    "training": ~validation,
                    "validation": validation,
                }
            )
    elif source == "S2":
        training = environment == mapping["development"]
        for name in ("calibration", "known_test"):
            folds.append(
                {
                    "fold_id": f"development_to_{name}",
                    "training": training,
                    "validation": environment == mapping[name],
                }
            )
    else:
        raise ValueError(source)
    validation_coverage = np.zeros(environment.size, dtype=np.int64)
    for fold in folds:
        if not np.any(fold["training"]) or not np.any(fold["validation"]):
            raise RuntimeError(f"empty {source} fold {fold['fold_id']}")
        if np.any(fold["training"] & fold["validation"]):
            raise RuntimeError(f"overlapping {source} fold {fold['fold_id']}")
        validation_coverage += fold["validation"].astype(np.int64)
    expected = np.ones(environment.size, dtype=np.int64)
    if source == "S2":
        expected[environment == mapping["development"]] = 0
    if not np.array_equal(validation_coverage, expected):
        raise RuntimeError(f"{source} validation coverage differs from frozen role contract")
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


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    present = np.unique(truth)
    recalls = np.asarray(
        [np.mean(prediction[truth == label] == label) for label in present],
        dtype=np.float64,
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
        "cohen_kappa": float(cohen_kappa_score(truth, prediction, labels=present)),
        "matthews_correlation": float(matthews_corrcoef(truth, prediction)),
    }


def aggregate(
    rows: list[dict[str, Any]], truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    names = (
        "accuracy",
        "macro_f1_present_classes",
        "balanced_accuracy",
        "worst_present_device_recall",
        "cohen_kappa",
        "matthews_correlation",
    )
    output: dict[str, Any] = {}
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
        "pooled_validation": classification_metrics(truth, prediction),
        "all_groups_disjoint": bool(all(row["group_overlap_count"] == 0 for row in rows)),
        "all_features_finite": bool(all(row["features_finite"] for row in rows)),
        "all_classifiers_complete": bool(all(row["classifier_complete"] for row in rows)),
        "fit_and_predict_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
    }


def evaluate_candidates(
    source: str,
    data: dict[str, Any],
    folds: list[dict[str, Any]],
    candidates: dict[str, list[str]],
    *,
    seed: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    groups = np.asarray(data["groups"], dtype=object)
    validation_union = np.any(
        np.stack([np.asarray(fold["validation"], dtype=bool) for fold in folds]), axis=0
    )
    summaries: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for candidate, components in candidates.items():
        print(f"evaluate {source} seed={seed} / {candidate}", flush=True)
        oof = np.full(labels.size, -1, dtype=np.int64)
        fold_rows: list[dict[str, Any]] = []
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
            classifier = RidgeClassifier(alpha=1.0, class_weight=None)
            classifier.fit(train_x, labels[training])
            prediction = np.asarray(classifier.predict(validation_x), dtype=np.int64)
            if np.any(oof[validation] >= 0):
                raise RuntimeError(f"overlapping validation predictions in {source}")
            oof[validation] = prediction
            train_classes = np.unique(labels[training])
            validation_classes = np.unique(labels[validation])
            complete = bool(
                set(validation_classes.tolist()).issubset(train_classes.tolist())
                and np.array_equal(classifier.classes_, train_classes)
                and prediction.shape == labels[validation].shape
            )
            row = {
                "source": source,
                "sketch_seed": "" if seed is None else int(seed),
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
                **classification_metrics(labels[validation], prediction),
            }
            raw_rows.append(row)
            fold_rows.append(row)
        if np.any(oof[validation_union] < 0) or np.any(oof[~validation_union] >= 0):
            raise RuntimeError(f"{source}/{candidate} validation prediction contract failed")
        summaries[candidate] = aggregate(
            fold_rows, labels[validation_union], oof[validation_union]
        )
        predictions[candidate] = oof
    return summaries, raw_rows, predictions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["scaler_ledger"] = json.dumps(
                rendered["scaler_ledger"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    amendment_path = args.amendment.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_S1_OR_S2_PROJECTIVE_FEATURE_EXTRACTION_OR_B1_SCORING":
        raise RuntimeError("B1 protocol is not pre-score frozen")
    if amendment["status"] != "FROZEN_AFTER_FEATURE_CACHE_BUT_BEFORE_ANY_B1_CLASSIFIER_FIT_OR_SCORE":
        raise RuntimeError("B1 aggregation amendment is not pre-score frozen")
    if sha256(config_path) != amendment["parent_protocol"]["sha256"]:
        raise RuntimeError("B1 aggregation amendment parent hash mismatch")
    cache_path = ROOT / amendment["feature_cache"]["path"]
    cache_report_path = ROOT / amendment["feature_cache"]["report"]
    if sha256(cache_path) != amendment["feature_cache"]["sha256"]:
        raise RuntimeError("B1 S1/S2 feature cache hash mismatch")
    if sha256(cache_report_path) != amendment["feature_cache"]["report_sha256"]:
        raise RuntimeError("B1 S1/S2 feature cache report hash mismatch")
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if cache_report["status"] != "PASS_FROZEN_S1_S2_FRONTEND_CACHE":
        raise RuntimeError("B1 S1/S2 feature cache has not passed")

    controls = {
        "strongest_fixed": ["strongest_fixed"],
        "segmented_fourier_magnitude_256": ["segmented_fourier_magnitude_256"],
        "complex_unit_512": ["complex_unit_512"],
        "left_gram_256": ["left_gram_256"],
    }
    seed_candidates = {
        "sketch_m256": ["sketch_m256"],
        "sketch_m512": ["sketch_m512"],
        "sketch_m1024": ["sketch_m1024"],
        "fixed_plus_sketch_m256": ["strongest_fixed", "sketch_m256"],
        "fixed_plus_sketch_m1024": ["strongest_fixed", "sketch_m1024"],
    }
    sources = {source: load_source(cache_path, source) for source in ("S1", "S2")}
    folds = {source: build_folds(source, data) for source, data in sources.items()}
    control_summaries: dict[str, Any] = {}
    seed_summaries: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    predictions_payload: dict[str, np.ndarray] = {}
    feature_payload: dict[str, np.ndarray] = {}
    secant_audits: dict[str, Any] = {}
    for source, data in sources.items():
        summary, rows, predictions = evaluate_candidates(
            source, data, folds[source], controls, seed=None
        )
        control_summaries[source] = summary
        raw_rows.extend(rows)
        for name, values in predictions.items():
            predictions_payload[f"control__{source}__{name}"] = values

    for seed in config["measurement_banks"]["seeds"]:
        seed_text = str(seed)
        seed_summaries[seed_text] = {}
        secant_audits[seed_text] = {}
        for source, data in sources.items():
            features, ledger = nested_sketches(
                data["complex_front"],
                maximum_dimension=1024,
                prefixes=[256, 512, 1024],
                seed=int(seed),
                expected_hash=config["measurement_banks"]["n256_m1024_sha256"][seed_text],
                label=f"B1/{source}/seed{seed}",
            )
            for dimension in (256, 512, 1024):
                key = f"sketch_m{dimension}"
                data["blocks"][key] = features[f"m{dimension}"]
                feature_payload[f"seed{seed}__{source}__m{dimension}"] = features[
                    f"m{dimension}"
                ]
            summary, rows, predictions = evaluate_candidates(
                source, data, folds[source], seed_candidates, seed=int(seed)
            )
            seed_summaries[seed_text][source] = summary
            raw_rows.extend(rows)
            for name, values in predictions.items():
                predictions_payload[f"seed{seed}__{source}__{name}"] = values
            secant_audits[seed_text][source] = {
                "bank": ledger,
                "m1024": empirical_secant_audit(
                    data["complex_front"], features["m1024"], seed=int(seed) + 97
                ),
            }

    result_root = ROOT / "results" / "track_c_v1"
    run_root = ROOT / "runs" / "track_c_v1"
    result_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    raw_path = result_root / f"B1_S1_S2_RAW_TABLE_{args.release_id}.csv"
    latest_raw = result_root / "B1_S1_S2_RAW_TABLE.csv"
    write_csv(raw_path, raw_rows)
    shutil.copyfile(raw_path, latest_raw)
    feature_path = run_root / f"B1_S1_S2_PROJECTIVE_FEATURES_{args.release_id}.npz"
    prediction_path = run_root / f"B1_S1_S2_PREDICTIONS_{args.release_id}.npz"
    np.savez_compressed(feature_path, **feature_payload)
    np.savez_compressed(prediction_path, **predictions_payload)
    latest_feature = run_root / "B1_S1_S2_PROJECTIVE_FEATURES.npz"
    latest_prediction = run_root / "B1_S1_S2_PREDICTIONS.npz"
    shutil.copyfile(feature_path, latest_feature)
    shutil.copyfile(prediction_path, latest_prediction)

    all_valid = bool(
        all(
            summary["all_groups_disjoint"]
            and summary["all_features_finite"]
            and summary["all_classifiers_complete"]
            for source in control_summaries.values()
            for summary in source.values()
        )
        and all(
            summary["all_groups_disjoint"]
            and summary["all_features_finite"]
            and summary["all_classifiers_complete"]
            for seed in seed_summaries.values()
            for source in seed.values()
            for summary in source.values()
        )
    )
    payload = {
        "protocol_id": config["protocol_id"],
        "stage": "C304_C306_S1_S2_B1_SCORING",
        "generated_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": (
            "S1_S2_SCORING_COMPLETE_FOR_B1_SYNTHESIS"
            if all_valid
            else "S1_S2_TECHNICAL_FAIL"
        ),
        "inputs": {
            "config": {"path": relative(config_path), "sha256": sha256(config_path)},
            "amendment": {"path": relative(amendment_path), "sha256": sha256(amendment_path)},
            "cache": {"path": relative(cache_path), "sha256": sha256(cache_path)},
        },
        "environment_maps": {
            source: data["environment_to_code"] for source, data in sources.items()
        },
        "control_summaries": control_summaries,
        "seed_summaries": seed_summaries,
        "secant_audits": secant_audits,
        "technical_gates": {
            "all_groups_disjoint_features_finite_classifiers_complete": all_valid,
            "all_three_frozen_bank_hashes_match": True,
            "source_count": 2,
        },
        "claim_boundary": [
            "These are retrospective within-source representation scores.",
            "S1 and S2 are not untouched confirmations.",
            "No device-population confidence interval is implied by packet rows.",
            "The four-source decision is deferred to a separate synthesis with inherited S3/S4 evidence.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "artifacts": {
            "raw_table": {"path": relative(raw_path), "sha256": sha256(raw_path)},
            "features": {"path": relative(feature_path), "sha256": sha256(feature_path)},
            "predictions": {"path": relative(prediction_path), "sha256": sha256(prediction_path)},
        },
    }
    result_path = result_root / f"B1_S1_S2_RESULTS_{args.release_id}.json"
    latest_result = result_root / "B1_S1_S2_RESULTS.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    result_path.write_text(rendered, encoding="utf-8")
    latest_result.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "result": relative(result_path),
                "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
