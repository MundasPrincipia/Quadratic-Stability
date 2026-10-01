"""Numerical consistency check of the frozen RF feature-to-task interface.

Uses only the packaged v16 derived inputs and fold heads. No training,
optimization, raw IQ access, or interval certification is performed.
Run with Python 3.12 and the existing recorded NumPy requirements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
PARENT_NAME = "rf_baseline"
TOLERANCE = 1e-8


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def maximum(value):
    result = float(np.max(np.abs(value), initial=0.0))
    if not np.isfinite(result):
        raise ValueError("Nonfinite residual in interface check")
    return result


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("Unsafe input normalization")
    return values / norms


def operators(bank, weights, kappa):
    result = kappa * np.einsum("jm,mi,mk->jik", weights, bank,
                               bank.conj(), optimize=True)
    return (result + result.conj().transpose(0, 2, 1)) / 2


def quadratic_scores(states, matrices, intercepts):
    return np.einsum("ri,jik,rk->rj", states.conj(), matrices,
                     states, optimize=True) + intercepts


def adapt_rows(rows, basis):
    t = basis.shape[0]
    return (rows.reshape(len(rows), -1, t) @ basis.conj()).reshape(rows.shape)


def transform_operators(matrices, basis):
    t = basis.shape[0]
    k = matrices.shape[1] // t
    # Q = I_K tensor B^dagger in row-major column-vector coordinates.
    return np.einsum("at,cjakb,bu->cjtku", basis.conj(),
                     matrices.reshape(len(matrices), k, t, k, t), basis,
                     optimize=True).reshape(matrices.shape)


def check(parent, output):
    data = parent / "data/raw_rebuild_run02"
    flag_path = data / "FLAGS.npz"
    with np.load(flag_path, allow_pickle=False) as archive:
        flags = {key: archive[key] for key in archive.files}
    if len(flags) != 6:
        raise ValueError("Expected the six frozen observation flags")
    inputs = {flag_path.relative_to(parent).as_posix(): sha256(flag_path)}
    rows, source_records, failures = [], [], []
    score_keys = ("feature_vs_saved_score", "quadratic_vs_feature_score",
                  "standardized_reexpression_score", "pairwise_score",
                  "direct_pair_operator_score", "basis_feature_vs_physical_score",
                  "basis_quadratic_vs_physical_score", "imaginary_score")
    with threadpool_limits(limits=1):
        for source, expected_folds, expected_rows in (("S1", 13, 5200), ("S4", 3, 480)):
            source_path = data / source / "SOURCE_INPUTS.npz"
            inputs[source_path.relative_to(parent).as_posix()] = sha256(source_path)
            with np.load(source_path, allow_pickle=False) as saved:
                states = normalize_rows(saved["C"])
                original_bank = saved["measurement_vectors"]
                bank = normalize_rows(original_bank)
                saved_features = saved["features"]
                groups = saved["groups"]
            n, m = states.shape[1], len(bank)
            kappa = np.sqrt(n * (n + 1.0) / m)
            features = kappa * np.abs(states @ original_bank.conj().T) ** 2
            feature_error = maximum(features - saved_features)
            heads = sorted((data / source).glob("*/COMMON_HEAD.npz"))
            if len(heads) != expected_folds or len(states) != expected_rows:
                raise ValueError("Frozen source coverage changed")
            adapted = {}
            for name, basis in flags.items():
                psib = adapt_rows(states, basis)
                ab = adapt_rows(bank, basis)
                adapted[name] = (psib, ab, kappa * np.abs(psib @ ab.conj().T) ** 2)
            seen = []
            for head_path in heads:
                inputs[head_path.relative_to(parent).as_posix()] = sha256(head_path)
                with np.load(head_path, allow_pickle=False) as head:
                    ids, train = head["validation_ids"], head["train_ids"]
                    weights, beta = head["raw_weights"], head["raw_intercept"]
                    means, scales = head["standardizer_mean"], head["standardizer_scale"]
                    saved_h, saved_scores = head["class_operators_physical"], head["full_scores"]
                    pairs = head["pairs"].astype(int)
                if set(train) & set(ids) or set(groups[train]) & set(groups[ids]):
                    raise ValueError("Training/validation overlap")
                if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
                    raise ValueError("Invalid saved scales")
                seen.extend(ids.tolist())
                matrices = operators(bank, weights, kappa)
                linear = features[ids] @ weights.T + beta
                quadratic = quadratic_scores(states[ids], matrices, beta)
                # Standardized coefficients are not independently saved. This
                # inverse re-expression checks the formula, not the original fit.
                tilde_weights = weights * scales[None, :]
                tilde_beta = beta + weights @ means
                standardized = ((features[ids] - means) / scales) @ tilde_weights.T + tilde_beta
                means_error = maximum(saved_features[train].mean(axis=0) - means)
                expected_scale = saved_features[train].std(axis=0, ddof=0)
                expected_scale = np.where(expected_scale > np.finfo(np.float64).tiny,
                                          expected_scale, 1.0)
                scales_error = maximum(expected_scale - scales)
                sample = np.unique(np.linspace(0, len(ids) - 1, min(16, len(ids)), dtype=int))
                pair_h = matrices[pairs[:, 0]] - matrices[pairs[:, 1]]
                pair_beta = beta[pairs[:, 0]] - beta[pairs[:, 1]]
                direct_pairs = quadratic_scores(states[ids[sample]], pair_h, pair_beta)
                linear_pairs = linear[:, pairs[:, 0]] - linear[:, pairs[:, 1]]
                task_pairs = quadratic[:, pairs[:, 0]] - quadratic[:, pairs[:, 1]]
                shared = {
                    "source": source, "fold": head_path.parent.name,
                    "validation_rows": len(ids), "classes": len(beta), "pairs": len(pairs),
                    "n": n, "M": m, "kappa": float(kappa),
                    "feature_vs_saved_features": feature_error,
                    "feature_vs_saved_score": maximum(linear - saved_scores),
                    "quadratic_vs_feature_score": maximum(quadratic - linear),
                    "standardized_reexpression_score": maximum(standardized - linear),
                    "pairwise_score": maximum(task_pairs - linear_pairs),
                    "direct_pair_operator_score": maximum(direct_pairs - linear_pairs[sample]),
                    "direct_pair_validation_rows": len(sample),
                    "operator_vs_saved": maximum(matrices - saved_h),
                    "standardizer_mean": means_error, "standardizer_scale": scales_error,
                    "sum_class_operators_frobenius": float(np.linalg.norm(matrices.sum(axis=0))),
                }
                for name, basis in flags.items():
                    psib, ab, features_b = adapted[name]
                    hb = operators(ab, weights, kappa)
                    transformed = transform_operators(matrices, basis)
                    scores_b = quadratic_scores(psib[ids], hb, beta)
                    metrics = dict(shared, flag=name,
                        basis_feature_vs_physical_score=maximum(features_b[ids] @ weights.T + beta - linear),
                        basis_quadratic_vs_physical_score=maximum(scores_b - linear),
                        operator_coordinate_transform=maximum(hb - transformed),
                        basis_defect_frobenius=float(np.linalg.norm(basis.conj().T @ basis - np.eye(8))),
                        imaginary_score=max(maximum(quadratic.imag), maximum(scores_b.imag)))
                    checked = score_keys + ("operator_vs_saved", "operator_coordinate_transform",
                                              "standardizer_mean", "standardizer_scale",
                                              "feature_vs_saved_features")
                    metrics["status"] = "PASS" if all(metrics[key] <= TOLERANCE for key in checked) else "FAIL"
                    if metrics["status"] != "PASS":
                        failures.append({"source": source, "fold": head_path.parent.name, "flag": name,
                                         "residuals": {key: metrics[key] for key in checked if metrics[key] > TOLERANCE}})
                    rows.append(metrics)
                print(f"Checked {source}/{head_path.parent.name}: {len(ids)} held-out rows, six flags", flush=True)
            if sorted(seen) != list(range(expected_rows)):
                raise ValueError("Held-out rows are not covered exactly once")
            source_records.append({"source": source, "folds": len(heads), "held_out_rows": len(seen),
                                   "class_score_cells": len(seen) * len(beta),
                                   "pair_difference_cells": len(seen) * len(pairs)})
    changed_inputs = [name for name, digest in inputs.items() if sha256(parent / name) != digest]
    if changed_inputs:
        raise ValueError("Frozen inputs changed during check: " + repr(changed_inputs))
    summary = {
        "version": "v23", "kind": "numerical consistency check",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL", "absolute_tolerance": TOLERANCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "blas_threads": 1},
        "sources": source_records, "folds": 16, "flags": list(flags), "fold_flag_cells": len(rows),
        "max_score_residual": max(row[key] for row in rows for key in score_keys),
        "max_operator_residual": max(max(row["operator_vs_saved"], row["operator_coordinate_transform"]) for row in rows),
        "max_basis_defect_frobenius": max(row["basis_defect_frobenius"] for row in rows),
        "input_sha256": inputs, "inputs_unchanged_after_check": True,
        "standardized_coefficients": "Reconstructed algebraically from saved raw parameters; no independent refit or original standardized-parameter comparison.",
        "pair_coverage": "All held-out class-pair differences; direct quadratic evaluation of all 45 pair matrices on 16 evenly spaced validation rows per fold.",
        "scope": "Implementation consistency only. Does not prove the algebra, certify RF endpoints by interval arithmetic, or validate a physical RF noise model.",
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "FOLD_FLAG_CHECKS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        "# RF interface numerical consistency check\n\n"
        + f"Status: {summary['status']}. All 16 folds and 96 fold/flag cells are included.\n\n"
        + f"Maximum score residual: {summary['max_score_residual']:.17g}; threshold: {TOLERANCE:g}.\n\n"
        + summary["scope"] + "\n\n" + summary["standardized_coefficients"] + "\n\n"
        + summary["pair_coverage"] + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "fold_flag_cells", "max_score_residual", "max_operator_residual")}), flush=True)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=HERE.parent / PARENT_NAME,
                        help="Packaged v16 dependency directory; defaults to its sibling location")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists() or output == parent or parent in output.parents or output in parent.parents:
        parser.error("Output must be fresh and must not overlap the frozen dependency")
    check(parent, output)


if __name__ == "__main__":
    main()
