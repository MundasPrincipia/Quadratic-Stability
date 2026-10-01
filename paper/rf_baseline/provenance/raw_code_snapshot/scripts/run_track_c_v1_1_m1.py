"""Run the frozen Track C V1.1 M1 representation intervention."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_track_c_b1_s1_s2 import (  # noqa: E402
    build_folds as build_s12_folds,
    classification_metrics,
    load_source as load_s12_source,
)
from scripts.run_track_c_m2r1_single_seed import (  # noqa: E402
    build_folds as build_m2_folds,
    complex_from_real_coordinates,
    load_m2_harmonic_cache,
)
from czrf.passive_quotient import (  # noqa: E402
    analytic_right_unitary_twirl_features,
    covariance_diagnostics,
    fit_group_standardizer,
    isometric_embedding,
    left_gram_vector_features,
    physically_order_segment_major_bank,
    physically_order_segment_major_rows,
    projective_intensity_features,
    projective_measurement_bank,
    quadratic_gram_control_features,
)


DEFAULT_CONTRACT = ROOT / "configs/track_c_v1_1/M1_EXECUTION_CONTRACT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--release-id", required=True)
    return parser.parse_args()


def validate_contract(contract_path: Path, contract: dict[str, Any]) -> None:
    if (
        contract["status"]
        != "FROZEN_BEFORE_ANY_PHYSICAL_GRAM_TWIRL_OR_QUADRATIC_CONTROL_SCORE"
    ):
        raise RuntimeError("M1 execution contract is not frozen before scoring")
    for name, spec in contract["parents"].items():
        path = ROOT / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"M1 parent changed: {name}")
        if "required_status" in spec:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["status"] != spec["required_status"]:
                raise RuntimeError(f"M1 parent status mismatch: {name}")
    for name, spec in contract["data"].items():
        if not isinstance(spec, dict) or "path" not in spec:
            continue
        path = ROOT / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"M1 data changed: {name}")
    if not contract_path.is_file():
        raise RuntimeError("M1 contract path is missing")


def prepare_settings(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    s12_path = ROOT / contract["data"]["S1_S2_complex_cache"]["path"]
    for source in ("S1", "S2"):
        data = load_s12_source(s12_path, source)
        physical = physically_order_segment_major_rows(
            data["complex_front"],
            segment_count=8,
            coefficients_per_segment=32,
        )
        physical_gram = left_gram_vector_features(physical, matrix_rows=32)
        settings[source] = {
            "source": source,
            "setting": source,
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "device": np.asarray(data["labels"], dtype=np.int64),
            "environment": np.asarray(data["environment"], dtype=np.int64),
            "groups": np.asarray(data["groups"], dtype=object),
            "historical_complex": np.asarray(data["complex_front"], dtype=np.complex128),
            "physical_complex": physical,
            "historical_left_gram": np.asarray(
                data["blocks"]["left_gram_256"], dtype=np.float64
            ),
            "physical_left_gram": physical_gram,
            "matrix_rows": 32,
            "matrix_columns": 8,
            "sketch_dimension": int(contract["sources"][source]["sketch_dimension"]),
            "maximum_bank_dimension": 1024,
            "folds": build_s12_folds(source, data),
        }

    harmonic_path = ROOT / contract["data"]["S3_S4_harmonic_cache"]["path"]
    raw = load_m2_harmonic_cache(harmonic_path)
    m2r1_path = ROOT / contract["parents"]["M2R1_protocol"]["path"]
    m2r1 = json.loads(m2r1_path.read_text(encoding="utf-8"))
    valid_m2_path = ROOT / m2r1["parents"]["valid_m2_result"]["path"]
    if sha256(valid_m2_path) != m2r1["parents"]["valid_m2_result"]["sha256"]:
        raise RuntimeError("valid M2 result changed")
    valid_m2 = json.loads(valid_m2_path.read_text(encoding="utf-8"))
    for setting in ("S3_eq0", "S3_eq1", "S4"):
        data = raw[setting]
        physical = complex_from_real_coordinates(data["A0_complex_unit"])
        physical_gram = left_gram_vector_features(physical, matrix_rows=16)
        labels = np.asarray(data["labels"], dtype=np.int64)
        day = np.asarray(data["day"], dtype=np.int64)
        if setting.startswith("S3"):
            receiver = np.asarray(data["receiver"], dtype=np.int64)
            groups = np.asarray(
                [
                    f"{setting}:tx{int(tx)}:rx{int(rx)}:day{int(local_day)}"
                    for tx, rx, local_day in zip(
                        labels, receiver, day, strict=True
                    )
                ],
                dtype=object,
            )
            environment = int(setting[-1]) * 10000 + day * 100 + receiver
            fold_data = {**data, "groups": groups}
            source = "S3"
            columns = 7
        else:
            file_id = np.asarray(data["file_id"], dtype=np.int64)
            groups = np.asarray([f"file:{int(value)}" for value in file_id], dtype=object)
            environment = day
            fold_data = {**data, "groups": groups}
            source = "S4"
            columns = 8
        settings[setting] = {
            "source": source,
            "setting": setting,
            "labels": labels,
            "device": labels,
            "environment": np.asarray(environment, dtype=np.int64),
            "groups": groups,
            "historical_complex": physical,
            "physical_complex": physical,
            "historical_left_gram": np.asarray(
                data["A0_left_gram"], dtype=np.float64
            ),
            "physical_left_gram": physical_gram,
            "matrix_rows": 16,
            "matrix_columns": columns,
            "sketch_dimension": int(contract["sources"][source]["sketch_dimension"]),
            "maximum_bank_dimension": 512,
            "folds": build_m2_folds(
                fold_data, valid_m2["fold_protocol"][setting]
            ),
            "selected_indices": np.asarray(data["selected_indices"], dtype=np.int64),
        }
    return settings


def validation_union(folds: list[dict[str, Any]], rows: int) -> np.ndarray:
    output = np.zeros(rows, dtype=bool)
    for fold in folds:
        validation = np.asarray(fold["validation"], dtype=bool)
        if np.any(output & validation):
            raise RuntimeError("validation folds overlap")
        output |= validation
    return output


def summarize_fold_rows(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "macro_f1_present_classes",
        "balanced_accuracy",
        "worst_present_device_recall",
        "cohen_kappa",
        "matthews_correlation",
    )
    metrics = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        metrics[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "fold_count": len(rows),
        "feature_dimension": int(rows[0]["feature_dimension"]),
        "classifier_parameter_count": int(rows[0]["classifier_parameter_count"]),
        "metrics": metrics,
        "pooled_validation": classification_metrics(truth[mask], predictions[mask]),
        "all_groups_disjoint": bool(all(row["group_overlap_count"] == 0 for row in rows)),
        "all_features_finite": bool(all(row["features_finite"] for row in rows)),
        "all_classifiers_complete": bool(all(row["classifier_complete"] for row in rows)),
        "fit_and_predict_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
        "canonical_first_fold_covariance": diagnostics,
    }


def evaluate_block(
    data: dict[str, Any],
    representation: str,
    block: np.ndarray,
    *,
    seed: int | None,
    post_transform: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    setting = str(data["setting"])
    print(f"evaluate {setting} seed={seed} / {representation}", flush=True)
    values = np.asarray(block, dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    groups = np.asarray(data["groups"], dtype=object)
    folds = data["folds"]
    mask = validation_union(folds, labels.size)
    classes = np.unique(labels)
    class_column = {int(value): index for index, value in enumerate(classes)}
    oof = np.full(labels.size, -1, dtype=np.int64)
    oof_scores = np.full((labels.size, classes.size), np.nan, dtype=np.float64)
    rows = []
    diagnostics: dict[str, Any] = {}
    for fold_index, fold in enumerate(folds):
        started = time.perf_counter()
        training = np.asarray(fold["training"], dtype=bool)
        validation = np.asarray(fold["validation"], dtype=bool)
        train_groups = set(str(value) for value in groups[training])
        validation_groups = set(str(value) for value in groups[validation])
        overlap = train_groups.intersection(validation_groups)
        state = fit_group_standardizer(
            values[training],
            groups[training],
            forbidden_groups=groups[validation],
        )
        train_x = state.transform(values[training])
        validation_x = state.transform(values[validation])
        if post_transform is not None:
            train_x = train_x @ post_transform.T
            validation_x = validation_x @ post_transform.T
        if fold_index == 0:
            diagnostic_started = time.perf_counter()
            diagnostics = {
                "fold_id": fold["fold_id"],
                "before_fold_scaling": covariance_diagnostics(values[training]),
                "after_fold_scaling_and_optional_lift": covariance_diagnostics(train_x),
            }
            diagnostics["elapsed_seconds"] = float(
                time.perf_counter() - diagnostic_started
            )
        finite = bool(
            np.all(np.isfinite(train_x)) and np.all(np.isfinite(validation_x))
        )
        classifier = RidgeClassifier(alpha=1.0, class_weight=None)
        classifier.fit(train_x, labels[training])
        prediction = np.asarray(classifier.predict(validation_x), dtype=np.int64)
        decision = np.asarray(classifier.decision_function(validation_x), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.column_stack((-decision, decision))
        if np.any(oof[validation] >= 0):
            raise RuntimeError(f"overlapping OOF predictions: {setting}/{representation}")
        oof[validation] = prediction
        validation_rows = np.flatnonzero(validation)
        for local_column, label in enumerate(classifier.classes_):
            oof_scores[validation_rows, class_column[int(label)]] = decision[:, local_column]
        train_classes = np.unique(labels[training])
        validation_classes = np.unique(labels[validation])
        complete = bool(
            set(validation_classes.tolist()).issubset(train_classes.tolist())
            and np.array_equal(classifier.classes_, train_classes)
            and prediction.shape == labels[validation].shape
        )
        rows.append(
            {
                "source": data["source"],
                "setting": setting,
                "seed": "" if seed is None else int(seed),
                "representation": representation,
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
                "scaler_fit_group_sha256": state.fit_group_sha256,
                **classification_metrics(labels[validation], prediction),
            }
        )
    if np.any(oof[mask] < 0) or np.any(oof[~mask] >= 0):
        raise RuntimeError(f"OOF coverage failed: {setting}/{representation}")
    if np.any(~np.isfinite(oof_scores[mask])):
        raise RuntimeError(f"OOF decision scores incomplete: {setting}/{representation}")
    summary = summarize_fold_rows(rows, labels, oof, mask, diagnostics)
    return summary, rows, oof, oof_scores


def frozen_bank(
    contract: dict[str, Any], data: dict[str, Any], seed: int
) -> tuple[Any, dict[str, Any]]:
    source = data["source"]
    maximum = int(data["maximum_bank_dimension"])
    physical = np.asarray(data["physical_complex"])
    bank = projective_measurement_bank(physical.shape[1], maximum, seed=int(seed))
    expected = contract["measurement_banks"]["hashes"][source][str(seed)]
    if bank.sha256 != expected:
        raise RuntimeError(f"frozen bank hash mismatch: {source}/{seed}")
    original_hash = bank.sha256
    if source in {"S1", "S2"}:
        bank = physically_order_segment_major_bank(
            bank, segment_count=8, coefficients_per_segment=32
        )
    dimension = int(data["sketch_dimension"])
    prefix = bank if dimension == maximum else bank.prefix(dimension)
    return prefix, {
        "seed": int(seed),
        "input_dimension": int(physical.shape[1]),
        "maximum_output_dimension": maximum,
        "used_output_dimension": dimension,
        "frozen_maximum_bank_sha256": original_hash,
        "physical_bank_sha256": bank.sha256,
        "used_prefix_sha256": prefix.sha256,
    }


def report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C V1.1 M1 Representation Intervention",
        "",
        f"- Status: `{payload['status']}`",
        "- Scores are post-V1 retrospective mechanism evidence on frozen public-data rows/folds.",
        "- C807 device-level simultaneous inference is separate and has not been replaced by point metrics below.",
        "",
        "| Setting | Seed | Representation | Mean Macro-F1 | Pooled DBA | Dimension | Parameters | Seconds |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for setting, item in payload["settings"].items():
        for name, summary in item["controls"].items():
            lines.append(
                f"| {setting} | — | {name} | "
                f"{summary['metrics']['macro_f1_present_classes']['mean']:.4f} | "
                f"{summary['pooled_validation']['balanced_accuracy']:.4f} | "
                f"{summary['feature_dimension']} | {summary['classifier_parameter_count']} | "
                f"{summary['fit_and_predict_seconds']:.2f} |"
            )
        for seed, seed_item in item["seeds"].items():
            for name, summary in seed_item["representations"].items():
                lines.append(
                    f"| {setting} | {seed} | {name} | "
                    f"{summary['metrics']['macro_f1_present_classes']['mean']:.4f} | "
                    f"{summary['pooled_validation']['balanced_accuracy']:.4f} | "
                    f"{summary['feature_dimension']} | {summary['classifier_parameter_count']} | "
                    f"{summary['fit_and_predict_seconds']:.2f} |"
                )
    lines.extend(
        [
            "",
            "## Technical decision",
            "",
            f"- Parent/bank/fold/finite implementation Gate: `{payload['technical_gates']['pass']}`.",
            f"- Maximum S1/S2 physical Sketch replay error: {payload['technical_gates']['maximum_S1_S2_sketch_replay_error']:.3e}.",
            f"- Maximum S3/S4 physical Gram replay error: {payload['technical_gates']['maximum_S3_S4_gram_replay_error']:.3e}.",
            f"- Maximum isometric Ridge decision-score error: {payload['technical_gates']['maximum_isometric_decision_error']:.3e}.",
            "",
            "Historical S1/S2 16×16 Gram scores are diagnostic only. The physical mechanism Gate will compare Sketch with physical Gram and analytic twirl in C807.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract_path, contract)
    settings = prepare_settings(contract)
    seeds = [int(value) for value in contract["measurement_banks"]["seeds"]]
    summaries: dict[str, Any] = {}
    raw_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    bank_ledger: dict[str, Any] = {}
    replay_errors: dict[str, Any] = {}
    gram_replay_errors: dict[str, float] = {}
    lift_errors: dict[str, float] = {}

    for setting, data in settings.items():
        labels = np.asarray(data["labels"], dtype=np.int64)
        mask = validation_union(data["folds"], labels.size)
        predictions[f"{setting}__truth"] = labels
        predictions[f"{setting}__device"] = np.asarray(data["device"], dtype=np.int64)
        predictions[f"{setting}__environment"] = np.asarray(
            data["environment"], dtype=np.int64
        )
        predictions[f"{setting}__validation_mask"] = mask.astype(np.uint8)
        if "selected_indices" in data:
            predictions[f"{setting}__selected_indices"] = np.asarray(
                data["selected_indices"], dtype=np.int64
            )

        local_controls = {}
        historical_summary, rows, historical_oof, _ = evaluate_block(
            data,
            "historical_left_gram",
            data["historical_left_gram"],
            seed=None,
        )
        local_controls["historical_left_gram"] = historical_summary
        raw_rows.extend(rows)
        predictions[f"control__{setting}__historical_left_gram"] = historical_oof

        physical_summary, rows, physical_oof, physical_scores = evaluate_block(
            data,
            "physical_left_gram",
            data["physical_left_gram"],
            seed=None,
        )
        local_controls["physical_left_gram"] = physical_summary
        raw_rows.extend(rows)
        predictions[f"control__{setting}__physical_left_gram"] = physical_oof

        gram_dimension = int(np.asarray(data["physical_left_gram"]).shape[1])
        lift_dimension = max(gram_dimension, int(data["sketch_dimension"]))
        lift = isometric_embedding(
            gram_dimension,
            lift_dimension,
            seed=int(contract["representations"]["isometric_lift"]["seed"]),
        )
        lift_summary, rows, lift_oof, lift_scores = evaluate_block(
            data,
            "physical_left_gram_isometric_lift",
            data["physical_left_gram"],
            seed=None,
            post_transform=lift,
        )
        local_controls["physical_left_gram_isometric_lift"] = lift_summary
        raw_rows.extend(rows)
        predictions[
            f"control__{setting}__physical_left_gram_isometric_lift"
        ] = lift_oof
        lift_errors[setting] = float(
            np.max(np.abs(physical_scores[mask] - lift_scores[mask]))
        )

        if data["source"] in {"S3", "S4"}:
            gram_replay_errors[setting] = float(
                np.max(
                    np.abs(
                        np.asarray(data["historical_left_gram"], dtype=np.float64)
                        - np.asarray(data["physical_left_gram"], dtype=np.float64)
                    )
                )
            )

        summaries[setting] = {"controls": local_controls, "seeds": {}}
        bank_ledger[setting] = {}
        replay_errors[setting] = {}
        for seed in seeds:
            bank, ledger = frozen_bank(contract, data, seed)
            bank_ledger[setting][str(seed)] = ledger
            physical = np.asarray(data["physical_complex"], dtype=np.complex128)
            sketch = projective_intensity_features(physical, bank)
            twirl = analytic_right_unitary_twirl_features(
                physical,
                bank,
                matrix_rows=int(data["matrix_rows"]),
            )
            quadratic, quadratic_bank = quadratic_gram_control_features(
                physical,
                matrix_rows=int(data["matrix_rows"]),
                output_dimension=int(data["sketch_dimension"]),
                seed=int(seed) + 910000,
            )
            ledger["quadratic_bank"] = {
                "seed": quadratic_bank.seed,
                "input_dimension": quadratic_bank.input_dimension,
                "output_dimension": quadratic_bank.output_dimension,
                "sha256": quadratic_bank.sha256,
            }
            if data["source"] in {"S1", "S2"}:
                old_bank = projective_measurement_bank(
                    physical.shape[1], int(data["maximum_bank_dimension"]), seed=seed
                )
                if int(data["sketch_dimension"]) != int(data["maximum_bank_dimension"]):
                    old_bank = old_bank.prefix(int(data["sketch_dimension"]))
                historical_sketch = projective_intensity_features(
                    data["historical_complex"], old_bank
                )
                replay_errors[setting][str(seed)] = float(
                    np.max(np.abs(sketch - historical_sketch))
                )
            else:
                replay_errors[setting][str(seed)] = 0.0

            seed_summaries = {}
            for representation, features in (
                ("analytic_twirl_sketch", twirl),
                ("original_projective_sketch", sketch),
                ("quadratic_physical_gram_control", quadratic),
            ):
                summary, rows, oof, _ = evaluate_block(
                    data,
                    representation,
                    features,
                    seed=seed,
                )
                seed_summaries[representation] = summary
                raw_rows.extend(rows)
                predictions[f"seed{seed}__{setting}__{representation}"] = oof
            summaries[setting]["seeds"][str(seed)] = {
                "representations": seed_summaries
            }
            del sketch, twirl, quadratic

    all_summaries = []
    for item in summaries.values():
        all_summaries.extend(item["controls"].values())
        for seed_item in item["seeds"].values():
            all_summaries.extend(seed_item["representations"].values())
    implementation_pass = bool(
        all(summary["all_groups_disjoint"] for summary in all_summaries)
        and all(summary["all_features_finite"] for summary in all_summaries)
        and all(summary["all_classifiers_complete"] for summary in all_summaries)
    )
    maximum_s12_replay = max(
        replay_errors[setting][str(seed)]
        for setting in ("S1", "S2")
        for seed in seeds
    )
    maximum_s34_gram = max(gram_replay_errors.values())
    maximum_lift = max(lift_errors.values())
    technical_pass = bool(
        implementation_pass
        and maximum_s12_replay
        <= float(contract["implementation_gates"]["S1_S2_physical_sketch_replay_max_abs"])
        and maximum_s34_gram
        <= float(
            contract["implementation_gates"][
                "S3_S4_recomputed_physical_gram_vs_historical_max_abs"
            ]
        )
        and maximum_lift
        <= float(
            contract["implementation_gates"]["isometric_decision_function_max_abs"]
        )
    )

    result_dir = ROOT / "results/track_c_v1_1"
    run_dir = ROOT / "runs/track_c_v1_1"
    report_dir = ROOT / "refine-logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"C805_C806_M1_RESULTS_{args.release_id}.json"
    table_path = result_dir / f"C805_C806_M1_RAW_TABLE_{args.release_id}.csv"
    prediction_path = run_dir / f"C805_C806_M1_OOF_{args.release_id}.npz"
    report_path = report_dir / f"TRACK_C_V1_1_M1_{args.release_id}.md"
    for path in (result_path, table_path, prediction_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)
    np.savez_compressed(prediction_path, **predictions)
    payload = {
        "contract_id": contract["contract_id"],
        "stage": "C805_C806_M1_REPRESENTATION_INTERVENTION",
        "generated_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_M1_TECHNICAL_READY_FOR_C807" if technical_pass else "FAIL_STOP_V1_1_IMPLEMENTATION",
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
            "contract": {"path": relative(contract_path), "sha256": sha256(contract_path)},
            "runner": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        },
        "settings": summaries,
        "bank_ledger": bank_ledger,
        "technical_gates": {
            "all_groups_disjoint_features_finite_classifiers_complete": implementation_pass,
            "S1_S2_sketch_replay_errors": replay_errors,
            "maximum_S1_S2_sketch_replay_error": maximum_s12_replay,
            "S3_S4_gram_replay_errors": gram_replay_errors,
            "maximum_S3_S4_gram_replay_error": maximum_s34_gram,
            "isometric_decision_errors": lift_errors,
            "maximum_isometric_decision_error": maximum_lift,
            "pass": technical_pass,
        },
        "claim_boundary": contract["boundaries"],
        "artifacts": {
            "raw_table": {"path": relative(table_path), "sha256": sha256(table_path)},
            "oof_predictions": {
                "path": relative(prediction_path),
                "sha256": sha256(prediction_path),
            },
            "report": {"path": relative(report_path)},
        },
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(report_markdown(payload), encoding="utf-8")
    latest = {
        result_path: result_dir / "C805_C806_M1_RESULTS.json",
        table_path: result_dir / "C805_C806_M1_RAW_TABLE.csv",
        prediction_path: run_dir / "C805_C806_M1_OOF.npz",
        report_path: report_dir / "TRACK_C_V1_1_M1.md",
    }
    for source, destination in latest.items():
        shutil.copyfile(source, destination)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "result": relative(result_path),
            }
        ),
        flush=True,
    )
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
