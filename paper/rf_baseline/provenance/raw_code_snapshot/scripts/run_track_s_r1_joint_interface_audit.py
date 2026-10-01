"""Run the Track S R1 fixed-algebra and stable all-budget certificate audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
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
)
from scripts.run_track_c_v1_1_m1 import frozen_bank  # noqa: E402
from czrf.ordered_task_interface import (  # noqa: E402
    ordered_projected_measurement_features,
    ordered_task_obstruction,
    pairwise_task_operators_adapted,
)
from czrf.passive_quotient import (  # noqa: E402
    fit_group_standardizer,
    left_gram_vector_features,
    physically_order_segment_major_rows,
    projective_intensity_features,
)
from czrf.projective_task_certificate import (  # noqa: E402
    centered_projective_cholesky_path_components,
    centered_projector_gram,
    certified_labels_from_pair_bounds,
    certified_labels_from_pair_intervals,
    pair_weight_matrix,
    projector_gram,
    projective_feature_upper_bound,
    raw_linear_head,
    task_operator_greedy_order,
    task_weight_order,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_contract(path: Path, contract: dict[str, Any]) -> None:
    if contract.get("status") != "FROZEN_BEFORE_TRACK_S_R1_REPAIR_SCORES":
        raise RuntimeError("Track S R1 contract status mismatch")
    for section in ("parents", "data"):
        for name, spec in contract[section].items():
            target = ROOT / spec["path"]
            if sha256(target) != spec["sha256"]:
                raise RuntimeError(f"Track S R1 {section} changed: {name}")
            if "required_status" in spec:
                payload = json.loads(target.read_text(encoding="utf-8"))
                if payload.get("status") != spec["required_status"]:
                    raise RuntimeError(f"Track S R1 parent status mismatch: {name}")
    for spec in contract["implementation"]:
        target = ROOT / spec["path"]
        if sha256(target) != spec["sha256"]:
            raise RuntimeError(f"Track S R1 implementation changed: {spec['path']}")
    if not path.is_file():
        raise RuntimeError("Track S R1 contract is missing")


def load_s1(m1: dict[str, Any]) -> dict[str, Any]:
    cache = ROOT / m1["data"]["S1_S2_complex_cache"]["path"]
    data = load_s12_source(cache, "S1")
    physical = physically_order_segment_major_rows(
        data["complex_front"], segment_count=8, coefficients_per_segment=32
    )
    return {
        "source": "S1",
        "setting": "S1",
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "environment": np.asarray(data["environment"], dtype=np.int64),
        "groups": np.asarray(data["groups"], dtype=object),
        "physical_complex": np.asarray(physical, dtype=np.complex128),
        "physical_left_gram": left_gram_vector_features(physical, matrix_rows=32),
        "matrix_rows": 32,
        "matrix_columns": 8,
        "sketch_dimension": int(m1["sources"]["S1"]["sketch_dimension"]),
        "maximum_bank_dimension": 1024,
        "folds": build_s12_folds("S1", data),
        "loader_ledger": {
            "container": relative(cache),
            "container_group_read": "S1",
            "S2_group_read": False,
        },
    }


def load_s4(m1: dict[str, Any]) -> dict[str, Any]:
    cache = ROOT / m1["data"]["S3_S4_harmonic_cache"]["path"]
    keys = (
        "S4_selected_indices",
        "S4_labels",
        "S4_day",
        "S4_harmonic_complex",
        "S4_harmonic_projective",
        "S4_file_id",
    )
    with np.load(cache, allow_pickle=False) as payload:
        selected = np.asarray(payload[keys[0]], dtype=np.int64)
        labels = np.asarray(payload[keys[1]], dtype=np.int64)
        day = np.asarray(payload[keys[2]], dtype=np.int64)
        harmonic = np.asarray(payload[keys[3]], dtype=np.float64)
        historical_gram = np.asarray(payload[keys[4]], dtype=np.float64)
        file_id = np.asarray(payload[keys[5]], dtype=np.int64)
    physical = complex_from_real_coordinates(harmonic)
    groups = np.asarray([f"file:{int(value)}" for value in file_id], dtype=object)
    m2r1_path = ROOT / m1["parents"]["M2R1_protocol"]["path"]
    m2r1 = json.loads(m2r1_path.read_text(encoding="utf-8"))
    valid_path = ROOT / m2r1["parents"]["valid_m2_result"]["path"]
    if sha256(valid_path) != m2r1["parents"]["valid_m2_result"]["sha256"]:
        raise RuntimeError("valid M2 result changed")
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    fold_data = {
        "selected_indices": selected,
        "labels": labels,
        "day": day,
        "file_id": file_id,
        "groups": groups,
    }
    return {
        "source": "S4",
        "setting": "S4",
        "labels": labels,
        "environment": day,
        "groups": groups,
        "physical_complex": np.asarray(physical, dtype=np.complex128),
        "historical_left_gram": historical_gram,
        "physical_left_gram": left_gram_vector_features(physical, matrix_rows=16),
        "matrix_rows": 16,
        "matrix_columns": 8,
        "sketch_dimension": int(m1["sources"]["S4"]["sketch_dimension"]),
        "maximum_bank_dimension": 512,
        "folds": build_m2_folds(fold_data, valid["fold_protocol"]["S4"]),
        "selected_indices": selected,
        "loader_ledger": {
            "container": relative(cache),
            "array_keys_read": list(keys),
            "S3_array_keys_read": False,
        },
    }


def deterministic_random_order(
    measurements: int, salt: str, source: str, fold_id: str
) -> np.ndarray:
    digest = hashlib.sha256(f"{salt}|{source}|{fold_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed).permutation(measurements).astype(np.int64)


def pair_bias(intercept: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(
        [intercept[first] - intercept[second] for first, second in pairs],
        dtype=np.float64,
    )


def q8_gs2_ledger() -> dict[str, Any]:
    path = ROOT / "results/track_s_flagship/GS2_RF_ADAPTER_ROW_RAW_20260830_221214.csv"
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["q"]) == 8:
                rows.append(row)
    output: dict[str, Any] = {}
    for source in ("S1", "S4"):
        local = [row for row in rows if row["source"] == source]
        fold_groups: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        accept = 0
        wrong = 0
        for row in local:
            fold_groups[row["fold_id"]][row["group"]].append(
                row["exact_true_covered"] == "True"
            )
            if row["exact_state"] == "ACCEPT":
                accept += 1
                wrong += int(int(row["exact_prediction"]) != int(row["device"]))
        fold_coverage = {}
        all_group_values = []
        for fold_id, groups in fold_groups.items():
            values = [all(flags) for flags in groups.values()]
            fold_coverage[fold_id] = {
                "covered": int(sum(values)),
                "total": len(values),
                "rate": float(np.mean(values)),
            }
            all_group_values.extend(values)
        output[source] = {
            "rows": len(local),
            "row_coverage": float(
                np.mean([row["exact_true_covered"] == "True" for row in local])
            ),
            "group_coverage": {
                "covered": int(sum(all_group_values)),
                "total": len(all_group_values),
                "rate": float(np.mean(all_group_values)),
            },
            "worst_fold_group_coverage": float(
                min(item["rate"] for item in fold_coverage.values())
            ),
            "fold_group_coverage": fold_coverage,
            "diagnostic_singleton_hits": accept,
            "wrong_singleton_hits": wrong,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    contract_path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract_path, contract)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/track_s_shared", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(f"Track S R1 tests failed:\n{tests.stdout}\n{tests.stderr}")

    m1_path = ROOT / contract["parents"]["track_c_m1_contract"]["path"]
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    sources = {"S1": load_s1(m1), "S4": load_s4(m1)}
    q_values = [int(value) for value in contract["ordered_interface"]["q_values"]]
    measurement_seed = int(contract["measurement_bank"]["seed"])
    safety = float(contract["numerics"]["roundoff_safety_factor"])
    solve_gate = float(contract["numerics"]["solve_relative_residual_gate"])
    random_salt = str(contract["all_budget_certificate"]["random_salt"])
    canonical = contract["canonical_folds"]
    started = time.perf_counter()

    head_rows: list[dict[str, Any]] = []
    obstruction_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    source_ledger: dict[str, Any] = {}
    total_truth_violations = 0
    total_replication_violations = 0
    total_solver_failures = 0

    for source, data in sources.items():
        print(f"Track S R1 source {source}: precompute", flush=True)
        labels = np.asarray(data["labels"], dtype=np.int64)
        groups = np.asarray(data["groups"], dtype=object)
        physical = np.asarray(data["physical_complex"], dtype=np.complex128)
        rows = int(data["matrix_rows"])
        length = int(data["matrix_columns"])
        bank, bank_ledger = frozen_bank(m1, data, measurement_seed)
        features = projective_intensity_features(physical, bank)
        bound = projective_feature_upper_bound(bank.input_dimension, bank.output_dimension)
        full_gram = projector_gram(bank.vectors)
        centered_gram = centered_projector_gram(bank.vectors)
        projected_features: dict[int, np.ndarray] = {}
        residual_measurement_gram: dict[int, np.ndarray] = {}
        projected_bank_replay_error: dict[int, float] = {}
        for q in q_values:
            print(f"Track S R1 {source}: ordered q={q}", flush=True)
            projected_features[q] = ordered_projected_measurement_features(
                physical, bank.vectors, bound, rows, length, q
            )
            projected_bank = ordered_projected_measurement_features(
                bank.vectors, bank.vectors, 1.0, rows, length, q
            )
            projected_bank = np.asarray(
                (projected_bank + projected_bank.T) / 2.0, dtype=np.float64
            )
            residual = np.asarray(
                (full_gram - projected_bank + (full_gram - projected_bank).T) / 2.0,
                dtype=np.float64,
            )
            eigen_min = float(np.min(np.linalg.eigvalsh(residual)))
            if eigen_min < -2e-9:
                raise RuntimeError(f"ordered residual Gram is indefinite: {source}/q={q}")
            residual_measurement_gram[q] = residual
            if q == length:
                projected_bank_replay_error[q] = float(
                    np.max(np.abs(projected_features[q] - features))
                )

        oof_full = np.full(labels.size, -1, dtype=np.int64)
        for fold in data["folds"]:
            fold_id = str(fold["fold_id"])
            print(f"Track S R1 {source} {fold_id}: fit head", flush=True)
            training = np.asarray(fold["training"], dtype=bool)
            validation = np.asarray(fold["validation"], dtype=bool)
            state = fit_group_standardizer(
                features[training], groups[training], forbidden_groups=groups[validation]
            )
            classifier = RidgeClassifier(alpha=1.0, class_weight=None)
            classifier.fit(state.transform(features[training]), labels[training])
            weights, intercept = raw_linear_head(
                classifier.coef_, classifier.intercept_, state.mean, state.scale
            )
            validation_features = features[validation]
            full_scores = validation_features @ weights.T + intercept
            full_index = np.argmax(full_scores, axis=1)
            full_label = classifier.classes_[full_index]
            oof_full[validation] = full_label
            metrics = classification_metrics(labels[validation], full_label)
            head_rows.append(
                {
                    "source": source,
                    "fold_id": fold_id,
                    "train_rows": int(np.sum(training)),
                    "validation_rows": int(np.sum(validation)),
                    "feature_dimension": int(features.shape[1]),
                    "parameter_count": int(weights.size + intercept.size),
                    **metrics,
                }
            )
            differences, pairs = pair_weight_matrix(weights)
            intercept_pairs = pair_bias(intercept, pairs)
            truth_pairs = validation_features @ differences.T + intercept_pairs

            exact_operators = None
            if fold_id == str(canonical[source]):
                exact_operators = pairwise_task_operators_adapted(
                    bank.vectors, differences, bound, rows, length
                )
            for q in q_values:
                center = projected_features[q][validation] @ differences.T + intercept_pairs
                gram_residual = residual_measurement_gram[q]
                square = bound**2 * np.einsum(
                    "pi,ij,pj->p",
                    differences,
                    gram_residual,
                    differences,
                    optimize=True,
                )
                scale = np.maximum(1.0, np.sum(np.abs(differences), axis=1) ** 2)
                if np.any(square < -2e-9 * scale):
                    raise RuntimeError("ordered task obstruction square became negative")
                frobenius_radius = np.sqrt(np.maximum(0.0, square))
                total = bound * np.sqrt(
                    np.maximum(
                        0.0,
                        np.einsum(
                            "pi,ij,pj->p",
                            differences,
                            full_gram,
                            differences,
                            optimize=True,
                        ),
                    )
                )
                relative_obstruction = np.divide(
                    frobenius_radius,
                    total,
                    out=np.zeros_like(frobenius_radius),
                    where=total > 0.0,
                )
                truth_violation = int(
                    np.sum(np.abs(truth_pairs - center) > frobenius_radius[None, :] + 2e-8)
                )
                total_truth_violations += truth_violation
                certificate = certified_labels_from_pair_intervals(
                    center,
                    frobenius_radius,
                    pairs,
                    weights.shape[0],
                )
                accepted = np.asarray(certificate["accepted"], dtype=bool)
                prediction_index = np.asarray(
                    certificate["prediction_index"], dtype=np.int64
                )
                prediction = np.full(full_label.size, -1, dtype=np.int64)
                prediction[accepted] = classifier.classes_[prediction_index[accepted]]
                replication = int(np.sum(accepted & (prediction != full_label)))
                total_replication_violations += replication
                exact_operator_max = ""
                exact_certified = ""
                exact_replication = ""
                exact_fixed_pairs = ""
                if exact_operators is not None:
                    exact = ordered_task_obstruction(
                        exact_operators, rows, length, q, compute_operator_norm=True
                    )
                    exact_operator_max = float(np.max(exact["operator_radius"]))
                    exact_fixed_pairs = int(np.sum(exact["fixed_pairwise_operator"]))
                    exact_certificate = certified_labels_from_pair_intervals(
                        center,
                        exact["operator_radius"],
                        pairs,
                        weights.shape[0],
                    )
                    exact_mask = np.asarray(exact_certificate["accepted"], dtype=bool)
                    exact_prediction_index = np.asarray(
                        exact_certificate["prediction_index"], dtype=np.int64
                    )
                    exact_prediction = np.full(full_label.size, -1, dtype=np.int64)
                    exact_prediction[exact_mask] = classifier.classes_[
                        exact_prediction_index[exact_mask]
                    ]
                    exact_certified = int(np.sum(exact_mask))
                    exact_replication = int(
                        np.sum(exact_mask & (exact_prediction != full_label))
                    )
                    total_replication_violations += int(exact_replication)
                obstruction_rows.append(
                    {
                        "source": source,
                        "fold_id": fold_id,
                        "q": q,
                        "validation_rows": int(np.sum(validation)),
                        "pair_count": len(pairs),
                        "fixed_pair_count_frobenius": int(
                            np.sum(frobenius_radius <= 2e-10 * np.maximum(1.0, total))
                        ),
                        "median_relative_frobenius_obstruction": float(
                            np.median(relative_obstruction)
                        ),
                        "maximum_relative_frobenius_obstruction": float(
                            np.max(relative_obstruction)
                        ),
                        "maximum_frobenius_radius": float(np.max(frobenius_radius)),
                        "frobenius_certified_count": int(np.sum(accepted)),
                        "frobenius_certification_rate": float(np.mean(accepted)),
                        "frobenius_replication_violations": replication,
                        "frobenius_identity_correct_count": int(
                            np.sum(accepted & (prediction == labels[validation]))
                        ),
                        "truth_interval_violations": truth_violation,
                        "exact_operator_max_radius_canonical": exact_operator_max,
                        "exact_operator_fixed_pairs_canonical": exact_fixed_pairs,
                        "exact_operator_certified_count_canonical": exact_certified,
                        "exact_operator_replication_violations_canonical": exact_replication,
                    }
                )

            measurements = bank.output_dimension
            strategies = {
                "fixed_prefix": np.arange(measurements, dtype=np.int64),
                "random": deterministic_random_order(
                    measurements, random_salt, source, fold_id
                ),
                "task_weight": task_weight_order(weights),
                "task_operator_greedy": task_operator_greedy_order(
                    weights, centered_gram
                ),
            }
            for strategy, order in strategies.items():
                print(
                    f"Track S R1 {source} {fold_id}: all budgets {strategy}",
                    flush=True,
                )
                try:
                    path = centered_projective_cholesky_path_components(
                        validation_features,
                        differences,
                        centered_gram,
                        order,
                        bound,
                        bank.input_dimension,
                        solver_residual_tolerance=solve_gate,
                        roundoff_safety_factor=safety,
                    )
                except RuntimeError:
                    total_solver_failures += 1
                    raise
                centers = np.broadcast_to(
                    path["scalar_center"],
                    (validation_features.shape[0], len(pairs)),
                ).copy()
                absolute_center_ledger = np.broadcast_to(
                    np.abs(path["scalar_center"]), centers.shape
                ).copy()
                cumulative_lower = np.full_like(centers, -np.inf)
                cumulative_upper = np.full_like(centers, np.inf)
                first_budget = np.full(validation_features.shape[0], -1, dtype=np.int64)
                first_prediction = np.full(validation_features.shape[0], -1, dtype=np.int64)
                for budget in range(measurements + 1):
                    if budget:
                        local = budget - 1
                        contribution = (
                            path["state_coordinates"][:, local, None]
                            * path["task_coordinates"][local, None, :]
                        )
                        centers += contribution
                        absolute_center_ledger += np.abs(contribution)
                    structural = path["state_centered_norm"] * np.sqrt(
                        path["task_residual_frobenius_square_by_budget"][budget]
                    )
                    raw_gamma = (
                        safety * np.finfo(np.float64).eps * max(1, budget + 1)
                    )
                    gamma = raw_gamma / (1.0 - raw_gamma)
                    numerical = gamma * (
                        absolute_center_ledger
                        + path["task_l1_scale"][None, :]
                    )
                    total_radius = structural[None, :] + numerical
                    lower = centers - total_radius + intercept_pairs[None, :]
                    upper = centers + total_radius + intercept_pairs[None, :]
                    violation = int(
                        np.sum(
                            (truth_pairs < lower - 2e-8)
                            | (truth_pairs > upper + 2e-8)
                        )
                    )
                    total_truth_violations += violation
                    cumulative_lower = np.maximum(cumulative_lower, lower)
                    cumulative_upper = np.minimum(cumulative_upper, upper)
                    certificate = certified_labels_from_pair_bounds(
                        cumulative_lower,
                        cumulative_upper,
                        pairs,
                        weights.shape[0],
                    )
                    accepted = np.asarray(certificate["accepted"], dtype=bool)
                    prediction_index = np.asarray(
                        certificate["prediction_index"], dtype=np.int64
                    )
                    prediction = np.full(full_label.size, -1, dtype=np.int64)
                    prediction[accepted] = classifier.classes_[prediction_index[accepted]]
                    replication = int(np.sum(accepted & (prediction != full_label)))
                    total_replication_violations += replication
                    new = accepted & (first_budget < 0)
                    first_budget[new] = budget
                    first_prediction[new] = prediction[new]
                    budget_rows.append(
                        {
                            "source": source,
                            "fold_id": fold_id,
                            "strategy": strategy,
                            "budget": budget,
                            "measurement_count": measurements,
                            "validation_rows": int(np.sum(validation)),
                            "certified_count": int(np.sum(accepted)),
                            "certification_rate": float(np.mean(accepted)),
                            "head_replication_violations": replication,
                            "identity_correct_certified_count": int(
                                np.sum(accepted & (prediction == labels[validation]))
                            ),
                            "truth_interval_violations": violation,
                            "maximum_structural_radius": float(np.max(structural)),
                            "maximum_total_radius": float(np.max(total_radius)),
                            "minimum_pairwise_lower_for_full_winner": float(
                                np.min(certificate["minimum_pairwise_lower"])
                            ),
                        }
                    )
                validation_indices = np.flatnonzero(validation)
                for local, global_index in enumerate(validation_indices):
                    first_rows.append(
                        {
                            "source": source,
                            "fold_id": fold_id,
                            "strategy": strategy,
                            "row_index": int(global_index),
                            "device": int(labels[global_index]),
                            "full_head_prediction": int(full_label[local]),
                            "full_head_correct": bool(
                                full_label[local] == labels[global_index]
                            ),
                            "first_certified_budget": int(first_budget[local]),
                            "first_certified_prediction": int(first_prediction[local]),
                            "certified_before_full": bool(
                                0 <= first_budget[local] < measurements
                            ),
                        }
                    )
                diagnostics = path["diagnostics"]
                if max(
                    diagnostics["cholesky_factor_relative_residual"],
                    diagnostics["state_solve_relative_residual"],
                    diagnostics["task_solve_relative_residual"],
                    diagnostics["full_span_reconstruction_relative_error"],
                ) > solve_gate:
                    total_solver_failures += 1

        source_ledger[source] = {
            "rows": int(labels.size),
            "devices": int(np.unique(labels).size),
            "folds": len(data["folds"]),
            "loader": data["loader_ledger"],
            "bank": bank_ledger,
            "projected_full_feature_max_abs_replay_error": projected_bank_replay_error[
                length
            ],
            "pooled_full_head_metrics": classification_metrics(labels, oof_full),
        }

    aggregate_budget: list[dict[str, Any]] = []
    for source in ("S1", "S4"):
        measurements = int(source_ledger[source]["bank"]["used_output_dimension"])
        for strategy in contract["all_budget_certificate"]["strategies"]:
            local_first = [
                row
                for row in first_rows
                if row["source"] == source and row["strategy"] == strategy
            ]
            prefull = [bool(row["certified_before_full"]) for row in local_first]
            first_values = np.asarray(
                [int(row["first_certified_budget"]) for row in local_first],
                dtype=np.int64,
            )
            aggregate_budget.append(
                {
                    "source": source,
                    "strategy": strategy,
                    "rows": len(local_first),
                    "prefull_certified_count": int(sum(prefull)),
                    "prefull_certification_rate": float(np.mean(prefull)),
                    "full_or_earlier_certified_count": int(np.sum(first_values >= 0)),
                    "minimum_first_certified_budget": int(np.min(first_values)),
                    "median_first_certified_budget": float(np.median(first_values)),
                    "measurement_count": measurements,
                }
            )

    result_root = ROOT / "results/track_s_flagship_r1"
    result_root.mkdir(parents=True, exist_ok=True)
    head_path = result_root / f"R1_FULL_HEAD_FOLD_RAW_{args.release_id}.csv"
    obstruction_path = result_root / f"R1_ORDERED_TASK_OBSTRUCTION_RAW_{args.release_id}.csv"
    budget_path = result_root / f"R1_ALL_INTEGER_BUDGET_RAW_{args.release_id}.csv"
    first_path = result_root / f"R1_FIRST_CERTIFICATION_ROW_RAW_{args.release_id}.csv"
    result_path = result_root / f"R1_JOINT_INTERFACE_AUDIT_{args.release_id}.json"
    write_csv(head_rows, head_path)
    write_csv(obstruction_rows, obstruction_path)
    write_csv(budget_rows, budget_path)
    write_csv(first_rows, first_path)
    gates = {
        "tests_pass": tests.returncode == 0,
        "truth_interval_violations_zero": total_truth_violations == 0,
        "head_replication_violations_zero": total_replication_violations == 0,
        "solver_failures_zero": total_solver_failures == 0,
        "full_projected_feature_replay_within_1e_10": all(
            item["projected_full_feature_max_abs_replay_error"] <= 1e-10
            for item in source_ledger.values()
        ),
        "all_rows_certified_by_full_budget": all(
            int(row["full_or_earlier_certified_count"]) == int(row["rows"])
            for row in aggregate_budget
        ),
    }
    status = (
        "PASS_TRACK_S_R1_JOINT_INTERFACE_AND_ALL_BUDGET_AUDIT"
        if all(gates.values())
        else "FAIL_TRACK_S_R1_JOINT_INTERFACE_AND_ALL_BUDGET_AUDIT"
    )
    payload = {
        "run_id": "TRACK_S_R1_JOINT_INTERFACE_AUDIT",
        "release_id": args.release_id,
        "status": status,
        "contract": {"path": relative(contract_path), "sha256": sha256(contract_path)},
        "source_ledger": source_ledger,
        "gs2_q8_reinterpreted_ledger": q8_gs2_ledger(),
        "all_budget_aggregate": aggregate_budget,
        "ordered_obstruction_rows": len(obstruction_rows),
        "all_integer_budget_rows": len(budget_rows),
        "first_certification_rows": len(first_rows),
        "implementation_audit": {
            "truth_interval_violations": total_truth_violations,
            "head_replication_violations": total_replication_violations,
            "solver_failures": total_solver_failures,
        },
        "gates": gates,
        "elapsed_seconds": float(time.perf_counter() - started),
        "raw_outputs": [
            relative(head_path),
            relative(obstruction_path),
            relative(budget_path),
            relative(first_path),
        ],
        "claim_boundary": contract["claim_boundary"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "h5py": importlib.metadata.version("h5py"),
            "platform": platform.platform(),
        },
        "data_request": {"needed_now": False},
    }
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status.startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
