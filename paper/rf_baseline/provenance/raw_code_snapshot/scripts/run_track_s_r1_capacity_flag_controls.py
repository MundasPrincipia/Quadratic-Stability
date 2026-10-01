"""Run matched registry-capacity and temporal-flag controls for Track S R1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dct
from sklearn.linear_model import RidgeClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_track_c_b1_s1_s2 import classification_metrics  # noqa: E402
from scripts.run_track_c_v1_1_m1 import frozen_bank  # noqa: E402
from scripts.run_track_s_r1_joint_interface_audit import (  # noqa: E402
    load_s1,
    load_s4,
    relative,
    sha256,
    validate_contract as validate_joint_contract,
    write_csv,
)
from czrf.ordered_twirl_adapter import ordered_twirl_pairwise_geometry  # noqa: E402


def validate(path: Path, contract: dict[str, Any]) -> None:
    if contract.get("status") != "FROZEN_BEFORE_TRACK_S_R1_CAPACITY_FLAG_SCORES":
        raise RuntimeError("capacity/flag control contract status mismatch")
    for section in ("parents", "data"):
        for name, spec in contract[section].items():
            target = ROOT / spec["path"]
            if sha256(target) != spec["sha256"]:
                raise RuntimeError(f"capacity/flag {section} changed: {name}")
            if "required_status" in spec:
                payload = json.loads(target.read_text(encoding="utf-8"))
                if payload.get("status") != spec["required_status"]:
                    raise RuntimeError(f"capacity/flag parent status mismatch: {name}")
    for spec in contract["implementation"]:
        target = ROOT / spec["path"]
        if sha256(target) != spec["sha256"]:
            raise RuntimeError(f"capacity/flag implementation changed: {spec['path']}")
    if not path.is_file():
        raise RuntimeError("capacity/flag contract is missing")


def unit_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("projective rows must be finite and nonzero")
    return np.asarray(array / norms[:, None], dtype=np.complex128)


def hash_order(indices: np.ndarray, salt: str) -> np.ndarray:
    return np.asarray(
        sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                f"{salt}|{int(index)}".encode("utf-8")
            ).hexdigest(),
        ),
        dtype=np.int64,
    )


def fixed_prototypes(
    training_indices: np.ndarray,
    labels: np.ndarray,
    per_device: int,
    salt: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    selected_labels: list[int] = []
    for device in sorted(np.unique(labels[training_indices]).tolist()):
        local = training_indices[labels[training_indices] == device]
        ordered = hash_order(local, f"{salt}|device={int(device)}")
        chosen = ordered[: min(int(per_device), ordered.size)]
        selected.extend(chosen.tolist())
        selected_labels.extend([int(device)] * chosen.size)
    return np.asarray(selected, dtype=np.int64), np.asarray(selected_labels, dtype=np.int64)


def projective_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(first @ second.conj().T), dtype=np.float64)


def majority_knn(
    similarity: np.ndarray,
    train_labels: np.ndarray,
    classes: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    k = min(int(neighbors), similarity.shape[1])
    nearest = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    votes = np.zeros((similarity.shape[0], classes.size), dtype=np.int64)
    score = np.zeros_like(votes, dtype=np.float64)
    for column, label in enumerate(classes):
        local = train_labels[nearest] == label
        votes[:, column] = np.sum(local, axis=1)
        score[:, column] = np.sum(similarity[np.arange(similarity.shape[0])[:, None], nearest] * local, axis=1)
    maximum = np.max(votes, axis=1, keepdims=True)
    tied = votes == maximum
    tie_score = np.where(tied, score, -np.inf)
    return classes[np.argmax(tie_score, axis=1)]


def temporal_bases(length: int, seed: int) -> dict[str, np.ndarray]:
    from czrf.passive_quotient.ordered_filtration import discrete_polynomial_basis

    polynomial = discrete_polynomial_basis(length).astype(np.complex128)
    coordinate = np.eye(length, dtype=np.complex128)
    cosine = dct(np.eye(length), type=2, norm="ortho", axis=0).astype(np.complex128)
    fourier = np.fft.fft(np.eye(length), axis=0) / np.sqrt(length)
    rng = np.random.default_rng(int(seed))
    gaussian = rng.normal(size=(length, length))
    random_basis, triangular = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    random_basis = random_basis @ np.diag(signs)
    permutation = np.eye(length)[rng.permutation(length)]
    permuted_polynomial = permutation @ polynomial
    output = {
        "polynomial": polynomial,
        "coordinate": coordinate,
        "dct": cosine,
        "dft": fourier,
        "random_orthogonal": random_basis.astype(np.complex128),
        "permuted_polynomial": permuted_polynomial.astype(np.complex128),
    }
    for name, basis in output.items():
        error = float(np.linalg.norm(basis.conj().T @ basis - np.eye(length)))
        if error > 1e-10:
            raise RuntimeError(f"temporal basis is not unitary: {name}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate(path, contract)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/track_s_shared", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(f"shared tests failed:\n{tests.stdout}\n{tests.stderr}")
    joint_contract_path = ROOT / contract["parents"]["joint_execution_contract"]["path"]
    joint_contract = json.loads(joint_contract_path.read_text(encoding="utf-8"))
    validate_joint_contract(joint_contract_path, joint_contract)
    m1_path = ROOT / joint_contract["parents"]["track_c_m1_contract"]["path"]
    m1 = json.loads(m1_path.read_text(encoding="utf-8"))
    sources = {"S1": load_s1(m1), "S4": load_s4(m1)}
    measurement_seed = int(joint_contract["measurement_bank"]["seed"])
    prototype_count = int(contract["capacity_controls"]["prototypes_per_device"])
    knn_k = int(contract["capacity_controls"]["knn_k"])
    salt = str(contract["capacity_controls"]["prototype_salt"])
    flag_seed = int(contract["flag_controls"]["basis_seed"])
    q_values = [int(value) for value in contract["flag_controls"]["q_values"]]
    canonical = contract["flag_controls"]["canonical_folds"]
    started = time.perf_counter()
    capacity_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []

    for source, data in sources.items():
        labels = np.asarray(data["labels"], dtype=np.int64)
        values = unit_rows(data["physical_complex"])
        classes = np.unique(labels)
        rows = int(data["matrix_rows"])
        length = int(data["matrix_columns"])
        bank, _ = frozen_bank(m1, data, measurement_seed)
        for fold in data["folds"]:
            fold_id = str(fold["fold_id"])
            print(f"Track S R1 capacity {source} {fold_id}", flush=True)
            training = np.asarray(fold["training"], dtype=bool)
            validation = np.asarray(fold["validation"], dtype=bool)
            train_indices = np.flatnonzero(training)
            validation_indices = np.flatnonzero(validation)
            train = values[training]
            test = values[validation]
            train_labels = labels[training]
            truth = labels[validation]
            similarity = projective_similarity(test, train)

            one_indices, one_labels = fixed_prototypes(
                train_indices, labels, 1, salt
            )
            multi_indices, multi_labels = fixed_prototypes(
                train_indices, labels, prototype_count, salt
            )
            local_position = {int(index): local for local, index in enumerate(train_indices)}
            one_columns = np.asarray([local_position[int(index)] for index in one_indices])
            multi_columns = np.asarray([local_position[int(index)] for index in multi_indices])
            predictions: dict[str, tuple[np.ndarray, int, int]] = {}
            predictions["one_hash_prototype"] = (
                one_labels[np.argmax(similarity[:, one_columns], axis=1)],
                int(one_indices.size),
                0,
            )
            predictions[f"multi_{prototype_count}_hash_prototype"] = (
                multi_labels[np.argmax(similarity[:, multi_columns], axis=1)],
                int(multi_indices.size),
                0,
            )
            predictions[f"knn_{knn_k}_all_training"] = (
                majority_knn(similarity, train_labels, classes, knn_k),
                int(train_indices.size),
                0,
            )
            centroid_score = np.column_stack(
                [
                    np.mean(similarity[:, train_labels == device] ** 2, axis=1)
                    for device in classes
                ]
            )
            predictions["mean_projector_centroid"] = (
                classes[np.argmax(centroid_score, axis=1)],
                int(classes.size),
                int(classes.size * values.shape[1] * values.shape[1]),
            )

            medoid_global: list[int] = []
            for device in classes:
                device_positions = np.flatnonzero(train_labels == device)
                device_values = train[device_positions]
                within = projective_similarity(device_values, device_values)
                medoid_global.append(int(device_positions[np.argmax(np.mean(within**2, axis=1))]))
            medoid_global_array = np.asarray(medoid_global, dtype=np.int64)
            predictions["exact_projective_medoid"] = (
                classes[np.argmax(similarity[:, medoid_global_array], axis=1)],
                int(classes.size),
                0,
            )

            prototype_train = projective_similarity(train, values[multi_indices])
            prototype_test = projective_similarity(test, values[multi_indices])
            train_kernel = np.exp(-2.0 * (1.0 - prototype_train))
            test_kernel = np.exp(-2.0 * (1.0 - prototype_test))
            kernel_head = RidgeClassifier(alpha=1.0, class_weight=None)
            kernel_head.fit(train_kernel, train_labels)
            predictions[f"dt_rbf_ridge_{multi_indices.size}"] = (
                np.asarray(kernel_head.predict(test_kernel), dtype=np.int64),
                int(multi_indices.size),
                int(kernel_head.coef_.size + kernel_head.intercept_.size),
            )
            for method, (prediction, references, parameters) in predictions.items():
                metrics = classification_metrics(truth, prediction)
                capacity_rows.append(
                    {
                        "source": source,
                        "fold_id": fold_id,
                        "method": method,
                        "train_rows": int(np.sum(training)),
                        "validation_rows": int(np.sum(validation)),
                        "reference_states": references,
                        "learned_parameters": parameters,
                        **metrics,
                    }
                )

            if fold_id != str(canonical[source]):
                continue
            prototype_values = values[multi_indices]
            bases = temporal_bases(length, flag_seed)
            for basis_name, basis in bases.items():
                for q in q_values:
                    geometry = ordered_twirl_pairwise_geometry(
                        values[validation_indices],
                        prototype_values,
                        matrix_rows=rows,
                        time_length=length,
                        level=q - 1,
                        basis=basis,
                        batch_size=64,
                    )
                    distance = geometry["quotient_distance"]
                    prediction = multi_labels[np.argmin(distance, axis=1)]
                    metrics = classification_metrics(truth, prediction)
                    flag_rows.append(
                        {
                            "source": source,
                            "fold_id": fold_id,
                            "basis": basis_name,
                            "q": q,
                            "prototype_count": int(multi_indices.size),
                            "validation_rows": int(np.sum(validation)),
                            **metrics,
                        }
                    )

    result_root = ROOT / "results/track_s_flagship_r1"
    capacity_path = result_root / f"R1_CAPACITY_CONTROL_RAW_{args.release_id}.csv"
    flag_path = result_root / f"R1_FLAG_SENSITIVITY_RAW_{args.release_id}.csv"
    result_path = result_root / f"R1_CAPACITY_FLAG_CONTROL_{args.release_id}.json"
    write_csv(capacity_rows, capacity_path)
    write_csv(flag_rows, flag_path)
    summaries: list[dict[str, Any]] = []
    for source in ("S1", "S4"):
        methods = sorted({row["method"] for row in capacity_rows if row["source"] == source})
        for method in methods:
            local = [row for row in capacity_rows if row["source"] == source and row["method"] == method]
            accuracy = np.asarray([float(row["accuracy"]) for row in local])
            macro = np.asarray([float(row["macro_f1_present_classes"]) for row in local])
            summaries.append(
                {
                    "source": source,
                    "method": method,
                    "folds": len(local),
                    "accuracy_mean": float(np.mean(accuracy)),
                    "accuracy_minimum": float(np.min(accuracy)),
                    "accuracy_maximum": float(np.max(accuracy)),
                    "macro_f1_mean": float(np.mean(macro)),
                    "reference_states": int(local[0]["reference_states"]),
                    "learned_parameters": int(local[0]["learned_parameters"]),
                }
            )
    full_basis_spread = []
    for source in ("S1", "S4"):
        local = [row for row in flag_rows if row["source"] == source and int(row["q"]) == 8]
        values_local = np.asarray([float(row["accuracy"]) for row in local])
        full_basis_spread.append(
            {"source": source, "q8_accuracy_spread": float(np.ptp(values_local))}
        )
    payload = {
        "run_id": "TRACK_S_R1_CAPACITY_FLAG_CONTROLS",
        "release_id": args.release_id,
        "status": "PASS_TRACK_S_R1_CAPACITY_AND_FLAG_CONTROLS",
        "contract": {"path": relative(path), "sha256": sha256(path)},
        "capacity_summary": summaries,
        "full_level_basis_invariance": full_basis_spread,
        "capacity_rows": len(capacity_rows),
        "flag_rows": len(flag_rows),
        "raw_outputs": [relative(capacity_path), relative(flag_path)],
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": contract["claim_boundary"],
        "data_request": {"needed_now": False},
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
