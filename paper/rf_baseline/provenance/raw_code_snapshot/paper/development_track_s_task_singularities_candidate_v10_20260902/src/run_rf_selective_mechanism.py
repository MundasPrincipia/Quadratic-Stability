"""v10-only q=7 common-head selective and exact-mechanism experiment.

Frozen inputs are read only.  One Ridge model is materialized per outer fold;
all six flags use its covariant task.  Ranking curves are retrospective,
descriptive matched-coverage comparisons, not deployable calibrated rules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import mpmath
import numpy as np
import psutil
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_track_c_v1_1_m1 import frozen_bank
from scripts.run_track_s_r1_capacity_flag_controls import temporal_bases
from scripts.run_track_s_r1_joint_interface_audit import load_s1, load_s4, pair_bias
from czrf.ordered_task_interface import normalized_adapted_matrices, pairwise_task_operators_adapted
from czrf.passive_quotient import fit_group_standardizer, projective_intensity_features
from czrf.projective_task_certificate import raw_linear_head, pair_weight_matrix, projective_feature_upper_bound

FLAGS = ("coordinate", "dct", "dft", "polynomial", "random_orthogonal", "permuted_polynomial")
PROTOCOL = HERE / "protocols/RF_EXECUTION_ADDENDUM_20260902_183700.md"
SOURCE_FREEZE = ROOT / "research-extensions/track_s_risk_resolution_v1/SOURCE_FREEZE.sha256"
METHODS = ("exact_fiber", "center_margin", "maximum_response", "negative_entropy", "full_state_margin")
EPS = np.finfo(float).eps
TAU = 1e-12


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(2**20), b""):
            value.update(part)
    return value.hexdigest()


def array_sha(array):
    a = np.asarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, values):
    if not values:
        raise RuntimeError("empty result is not silently written")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=list(values[0]))
        out.writeheader()
        out.writerows(values)


def gamma(n):
    return 4096.0 * max(1, n) * EPS / max(1.0 - n * EPS, 0.5)


def pairwise_minimum(center, radius, pairs, class_count):
    values = np.full((len(center), class_count), np.inf)
    which = np.full((len(center), class_count), -1, dtype=int)
    signs = np.zeros((len(center), class_count), dtype=int)
    for p, (a, b) in enumerate(pairs):
        for candidate, sign in ((a, 1), (b, -1)):
            val = sign * center[:, p] - radius[:, p]
            better = val < values[:, candidate]
            values[better, candidate] = val[better]
            which[better, candidate] = p
            signs[better, candidate] = sign
    return values, which, signs


def confidence(scores):
    ordered = np.sort(scores, axis=1)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.maximum(probs, np.finfo(float).tiny)), axis=1)
    return ordered[:, -1] - ordered[:, -2], np.max(scores, axis=1), -entropy


def stable_top_k(score, ids, k):
    return np.lexsort((ids, -np.asarray(score)))[:int(k)]


def safe_rate(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def verify_source_freeze():
    records = {}
    for line in SOURCE_FREEZE.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        got = sha(ROOT / relative)
        if got != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")
        records[relative] = got
    return records


def load_high_precision_checker():
    path = ROOT / "research-extensions/track_s_risk_resolution_v1/run_extension.py"
    spec = importlib.util.spec_from_file_location("v10_readonly_historical_recheck", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.high_precision_candidate_check


def numeric_interval(adapted, adapted_bank, weights, intercept, pairs, bound):
    """Independent measurement-sum implementation, including error envelopes."""
    n, k, length = adapted.shape
    x = adapted[:, :, :7].reshape(n, -1)
    y = adapted[:, :, 7:].reshape(n, -1)
    ax = adapted_bank[:, :, :7].reshape(len(adapted_bank), -1)
    ay = adapted_bank[:, :, 7:].reshape(len(adapted_bank), -1)
    u, v = x @ ax.conj().T, y @ ay.conj().T
    full_features = bound * np.abs(u + v)**2
    projected = bound * (np.abs(u)**2 + np.abs(v)**2)
    center_class = projected @ weights.T + intercept
    full_class = full_features @ weights.T + intercept
    class_cross = bound * (np.conj(u) * v) @ weights.T
    center = np.stack([center_class[:, a] - center_class[:, b] for a, b in pairs], axis=1)
    cross = np.stack([class_cross[:, a] - class_cross[:, b] for a, b in pairs], axis=1)
    pair_weights = np.stack([weights[a] - weights[b] for a, b in pairs])
    pair_intercepts = np.asarray([intercept[a] - intercept[b] for a, b in pairs])
    ue = gamma(x.shape[1]) * (np.abs(x) @ np.abs(ax).T)
    ve = gamma(y.shape[1]) * (np.abs(y) @ np.abs(ay).T)
    fe = bound * (2 * np.abs(u + v) * (ue + ve) + (ue + ve)**2)
    fe += 4096 * EPS * np.maximum(1.0, full_features)
    ce = np.abs(v) * ue + np.abs(u) * ve + ue * ve
    ce += 4096 * EPS * np.maximum(1.0, np.abs(np.conj(u) * v))
    envelope = fe @ np.abs(pair_weights).T + 4 * bound * ce @ np.abs(pair_weights).T
    # Cover the independent binary64 final sums (the inherited high-precision
    # recheck uses exact-decimal accumulation of these final terms instead).
    envelope += gamma(len(adapted_bank)) * (
        full_features @ np.abs(pair_weights).T
        + 4 * bound * np.abs(np.conj(u) * v) @ np.abs(pair_weights).T
        + np.abs(pair_intercepts)[None, :]
    )
    radius = 2 * np.abs(cross)
    return dict(x=x, y=y, u=u, v=v, ue=ue, ve=ve, center_class=center_class,
                full_class=full_class, center=center, cross=cross, radius=radius,
                envelope=np.nextafter(envelope, np.inf), pair_weights=pair_weights,
                pair_intercepts=pair_intercepts)


def witness_for_candidates(calc, candidate, pairs, bound):
    """Construct a concrete phase state at each candidate's bottleneck pair."""
    lower, which, signs = pairwise_minimum(calc["center"], calc["radius"], pairs, 10)
    rows = np.arange(len(candidate))
    p = which[rows, candidate]
    sign = signs[rows, candidate]
    z = sign * calc["cross"][rows, p]
    theta = np.pi - np.angle(z)
    phase = np.exp(1j * theta)
    inner = calc["u"] + phase[:, None] * calc["v"]
    features = bound * np.abs(inner)**2
    pw = sign[:, None] * calc["pair_weights"][p]
    pi = sign * calc["pair_intercepts"][p]
    direct = np.sum(features * pw, axis=1) + pi
    inner_error = calc["ue"] + calc["ve"] + 4096 * EPS * np.maximum(1.0, np.abs(calc["v"]))
    feature_error = bound * (2 * np.abs(inner) * inner_error + inner_error**2)
    feature_error += 4096 * EPS * np.maximum(1.0, features)
    error = np.sum(feature_error * np.abs(pw), axis=1)
    error += gamma(features.shape[1]) * (np.sum(features * np.abs(pw), axis=1) + np.abs(pi))
    expected = lower[rows, candidate]
    if np.max(np.abs(direct - expected)) > 1e-7:
        raise RuntimeError("phase witness does not attain its endpoint")
    upper = np.nextafter(direct + error, np.inf)
    return dict(pair=p, sign=sign, phase=theta, direct=direct, error=error,
                upper=upper, expected=expected, residual=np.abs(direct - expected))


def mechanism_rows(source, fold, flag, ids, labels, classes, calc, candidate, witness, pair_cross_norm):
    output = []
    a, b = np.sum(np.abs(calc["x"])**2, axis=1), np.sum(np.abs(calc["y"])**2, axis=1)
    for row, original in enumerate(ids):
        p, sign = int(witness["pair"][row]), int(witness["sign"][row])
        z = sign * calc["cross"][row, p]
        kappa = float(pair_cross_norm[p])
        denom = float(np.sqrt(a[row] * b[row]) * kappa)
        alignment = float(abs(z) / denom) if denom > 0 else 0.0
        r = float(2 * abs(z))
        phase = float(np.angle(z))
        signed_center = float(sign * calc["center"][row, p])
        pair = calc["pairs"][p]
        other = pair[1] if sign == 1 else pair[0]
        full_margin = float(calc["full_class"][row, candidate[row]] - calc["full_class"][row, other])
        reconstructed = float(2 * np.sqrt(a[row] * b[row]) * kappa * alignment)
        output.append(dict(source=source, fold=fold, flag=flag, row_id=int(original),
            device_label=int(labels[row]), candidate_rule="full_state_reference",
            candidate_class=int(classes[candidate[row]]), competitor_class=int(classes[other]),
            pair_index=p, pair_sign=sign, resolved_mass=float(a[row]), residual_mass=float(b[row]),
            cross_block_frobenius=kappa, alignment=alignment, phase=phase, center=signed_center,
            radius=r, lower=signed_center-r, full_state_margin=full_margin,
            radius_identity_residual=abs(r-reconstructed),
            score_identity_residual=abs(full_margin-signed_center-r*np.cos(phase)),
            witness_phase=float(witness["phase"][row]), witness_score=float(witness["direct"][row]),
            witness_upper=float(witness["upper"][row]), witness_residual=float(witness["residual"][row]),
            zero_factor=bool(denom == 0), status="VERIFIED"))
    return output


def summarize_mechanism(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["source"], row["fold"], row["flag"]), []).append(row)
    result = []
    names = ("resolved_mass", "residual_mass", "cross_block_frobenius", "alignment", "phase", "radius", "center", "lower")
    for (source, fold, flag), local in grouped.items():
        out = dict(source=source, fold=fold, flag=flag, rows=len(local))
        for name in names:
            vals = np.asarray([row[name] for row in local])
            out[f"median_{name}"] = float(np.median(vals))
            out[f"q25_{name}"] = float(np.quantile(vals, .25))
            out[f"q75_{name}"] = float(np.quantile(vals, .75))
        out["maximum_radius_identity_residual"] = max(r["radius_identity_residual"] for r in local)
        out["maximum_score_identity_residual"] = max(r["score_identity_residual"] for r in local)
        result.append(out)
    return result


def run(output):
    started = time.perf_counter()
    if output.exists():
        raise RuntimeError("new result directory required; overwriting is forbidden")
    if not PROTOCOL.is_file():
        raise RuntimeError("RF protocol addendum missing")
    inputs_before = verify_source_freeze()
    output.mkdir(parents=True)
    config = dict(status="FROZEN_BEFORE_V10_RF_E4_E5", q=7, T=8, flags=list(FLAGS),
        measurement_seed=20260827, basis_seed=2026083101, ridge_alpha=1.0, class_weight=None,
        tau=TAU, coverage_grid=[.05,.10,.20,.40,.60,.80,1.0],
        matched_certificate_k=True, methods=list(METHODS), thread_limit=1,
        state_scope="frozen harmonic/frontend C, not raw IQ", hp_dps=100,
        ranking="descending score, stable local source row id; no label feedback",
        protocol_sha256=sha(PROTOCOL), runner_sha256=sha(__file__), source_hashes=inputs_before)
    write_json(output / "CONFIG.json", config)
    write_json(output / "ENVIRONMENT.json", dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, h5py=h5py.__version__,
        mpmath=mpmath.__version__, platform=platform.platform()))
    hp = load_high_precision_checker()
    m1 = json.loads((ROOT / "configs/track_c_v1_1/M1_EXECUTION_CONTRACT.json").read_text())
    bases = temporal_bases(8, config["basis_seed"])
    np.savez_compressed(output / "FLAGS.npz", **{name: bases[name] for name in FLAGS})
    selective, mechanism, lineage, hp_rows, raw_rows = [], [], [], [], []
    fit_count, total_accept, replay_max, total_replay_violations = 0, 0, 0.0, 0
    fold_paths = []
    with threadpool_limits(limits=1):
        for source, loader in (("S1", load_s1), ("S4", load_s4)):
            data = loader(m1)
            c = np.asarray(data["physical_complex"], dtype=np.complex128)
            labels = np.asarray(data["labels"], dtype=np.int64)
            groups = np.asarray(data["groups"], dtype=str)
            rows, length = int(data["matrix_rows"]), 8
            bank, bank_info = frozen_bank(m1, data, config["measurement_seed"])
            feature_bound = projective_feature_upper_bound(bank.input_dimension, bank.output_dimension)
            features = projective_intensity_features(c, bank)
            source_root = output / source
            source_root.mkdir()
            np.savez_compressed(source_root / "SOURCE_INPUTS.npz", C=c, labels=labels, groups=groups,
                environment=data["environment"], row_ids=np.arange(len(c)), measurement_vectors=bank.vectors)
            write_json(source_root / "BANK_LINEAGE.json", bank_info)
            for fold in data["folds"]:
                fold_id = str(fold["fold_id"])
                fold_root = source_root / fold_id
                fold_root.mkdir()
                training, validation = np.asarray(fold["training"], bool), np.asarray(fold["validation"], bool)
                ids = np.flatnonzero(validation)
                state = fit_group_standardizer(features[training], groups[training], forbidden_groups=groups[validation])
                classifier = RidgeClassifier(alpha=1.0, class_weight=None)
                classifier.fit(state.transform(features[training]), labels[training])
                fit_count += 1
                weights, intercept = raw_linear_head(classifier.coef_, classifier.intercept_, state.mean, state.scale)
                classes = classifier.classes_
                if len(classes) != 10:
                    raise RuntimeError("class registry changed")
                differences, pairs = pair_weight_matrix(weights)
                raw_logits = features[validation] @ weights.T + intercept
                full_pred = np.argmax(raw_logits, axis=1)
                class_h = pairwise_task_operators_adapted(bank.vectors, weights, feature_bound, rows, length, basis=np.eye(length))
                np.savez_compressed(fold_root / "COMMON_HEAD.npz", train_ids=np.flatnonzero(training),
                    validation_ids=ids, classes=classes, raw_weights=weights, raw_intercept=intercept,
                    standardizer_mean=state.mean, standardizer_scale=state.scale,
                    class_operators_physical=class_h, pairs=np.asarray(pairs))
                task_sha = array_sha(class_h)
                lineage.append(dict(source=source, fold=fold_id, train_rows=int(training.sum()),
                    validation_rows=len(ids), fit_calls=1, train_ids_sha256=array_sha(np.flatnonzero(training)),
                    validation_ids_sha256=array_sha(ids), standardizer_groups_sha256=state.fit_group_sha256,
                    raw_weights_sha256=array_sha(weights), raw_intercept_sha256=array_sha(intercept),
                    class_operators_sha256=task_sha, head_artifact_sha256=sha(fold_root / "COMMON_HEAD.npz")))
                for flag in FLAGS:
                    adapted = normalized_adapted_matrices(c[validation], rows, length, basis=bases[flag])
                    adapted_bank = normalized_adapted_matrices(bank.vectors, rows, length, basis=bases[flag])
                    operators = pairwise_task_operators_adapted(bank.vectors, weights, feature_bound, rows, length, basis=bases[flag])
                    calc = numeric_interval(adapted, adapted_bank, weights, intercept, pairs, feature_bound)
                    calc["pairs"] = pairs
                    flat = adapted.reshape(len(ids), -1)
                    direct = np.stack([np.einsum("ni,ij,nj->n", flat.conj(), op, flat, optimize=True).real for op in operators], axis=1) + intercept
                    replay = max(float(np.max(np.abs(direct - raw_logits))), float(np.max(np.abs(calc["full_class"] - raw_logits))))
                    replay_max = max(replay_max, replay)
                    if replay > 1e-8:
                        raise RuntimeError(f"q8 replay failed: {source}/{fold_id}/{flag} {replay}")
                    if np.any(np.argmax(calc["full_class"], axis=1) != full_pred):
                        raise RuntimeError("full-head label replay failed")
                    raw_min, pair_index, signs = pairwise_minimum(calc["center"], calc["radius"], pairs, 10)
                    safe_min, _, _ = pairwise_minimum(calc["center"], calc["radius"] + calc["envelope"], pairs, 10)
                    cp = np.argmax(calc["center_class"], axis=1)
                    cert_index = np.argmax(safe_min, axis=1)
                    accept = safe_min[np.arange(len(ids)), cert_index] > TAU
                    hp_lower = np.full(len(ids), np.nan)
                    for j in np.flatnonzero(accept):
                        passed, value, method = hp(adapted[j], adapted_bank, differences, pair_bias(intercept, pairs),
                            pairs, int(cert_index[j]), feature_bound, TAU, 100)
                        hp_lower[j] = value
                        hp_rows.append(dict(source=source, fold=fold_id, flag=flag, row_id=int(ids[j]),
                            candidate_class=int(classes[cert_index[j]]), passed=passed, safe_lower=value, method=method))
                        if not passed:
                            raise RuntimeError("high precision acceptance recheck failed")
                    if np.any(accept & (cert_index != full_pred)):
                        raise RuntimeError("certified prediction differs from full head")
                    total_accept += int(accept.sum())
                    full_witness = witness_for_candidates(calc, full_pred, pairs, feature_bound)
                    center_witness = witness_for_candidates(calc, cp, pairs, feature_bound)
                    resolved_indices = np.arange(rows*length).reshape(rows,length)[:, :7].reshape(-1)
                    residual_indices = np.arange(rows*length).reshape(rows,length)[:, 7:].reshape(-1)
                    hcross = operators[:, resolved_indices][:, :, residual_indices]
                    pair_norm = np.asarray([np.linalg.norm(hcross[a]-hcross[b]) for a,b in pairs])
                    mech = mechanism_rows(source, fold_id, flag, ids, labels[validation], classes, calc,
                        full_pred, full_witness, pair_norm)
                    mechanism.extend(mech)
                    cm, maxr, entropy = confidence(calc["center_class"])
                    fm, _, _ = confidence(raw_logits)
                    scores = dict(center_margin=cm, maximum_response=maxr, negative_entropy=entropy, full_state_margin=fm)
                    for mode in ("certificate_matched", "fixed_coverage_grid"):
                        sizes = [int(accept.sum())] if mode == "certificate_matched" else sorted({int(round(frac * len(ids))) for frac in config["coverage_grid"]})
                        for k in sizes:
                            for method in METHODS:
                                if method == "exact_fiber":
                                    if mode != "certificate_matched":
                                        continue
                                    chosen, candidate = np.flatnonzero(accept), cert_index
                                else:
                                    chosen = stable_top_k(scores[method], ids, k)
                                    candidate = full_pred if method == "full_state_margin" else cp
                                wit = full_witness if method in ("exact_fiber", "full_state_margin") else center_witness
                                sound = safe_min[np.arange(len(ids)), candidate] > 0
                                unsound = wit["upper"] < 0
                                if np.any(sound & unsound):
                                    raise RuntimeError("sound and unsound evidence overlap")
                                unknown = ~(sound | unsound)
                                count = len(chosen)
                                un = int(unsound[chosen].sum())
                                unk = int(unknown[chosen].sum())
                                agree = int(np.sum(candidate[chosen] == full_pred[chosen]))
                                correct = int(np.sum(classes[candidate[chosen]] == labels[validation][chosen]))
                                selective.append(dict(source=source, fold=fold_id, flag=flag, mode=mode, method=method,
                                    total_rows=len(ids), selected_rows=count, coverage=safe_rate(count,len(ids)),
                                    verified_unsound_rows=un, unresolved_rows=unk,
                                    fiber_unsoundness_lower=safe_rate(un,count), fiber_unsoundness_upper=safe_rate(un+unk,count),
                                    reference_agreement_rows=agree, reference_head_agreement=safe_rate(agree,count),
                                    device_correct_rows=correct, device_label_precision=safe_rate(correct,count),
                                    criterion="numerical phase witness upper<0; positive conservative lower certifies sound", status="VERIFIED"))
                    for j, identity in enumerate(ids):
                        raw_rows.append(dict(source=source, fold=fold_id, flag=flag, row_id=int(identity),
                            device_label=int(labels[validation][j]), full_prediction=int(classes[full_pred[j]]),
                            center_prediction=int(classes[cp[j]]), exact_accepted=bool(accept[j]),
                            exact_prediction=int(classes[cert_index[j]]) if accept[j] else -1,
                            center_margin=float(cm[j]), maximum_response=float(maxr[j]), negative_entropy=float(entropy[j]),
                            full_state_margin=float(fm[j]), full_candidate_lower=float(raw_min[j,full_pred[j]]),
                            center_candidate_lower=float(raw_min[j,cp[j]]),
                            full_unsound_witness_upper=float(full_witness["upper"][j]),
                            center_unsound_witness_upper=float(center_witness["upper"][j]),
                            high_precision_lower=None if not accept[j] else float(hp_lower[j])))
                    artifact = fold_root / f"{flag}_ROW_ARRAYS.npz"
                    np.savez_compressed(artifact, row_ids=ids, labels=labels[validation], full_scores=raw_logits,
                        center_class=calc["center_class"], pair_center=calc["center"], pair_radius=calc["radius"],
                        pair_cross=calc["cross"], endpoint_envelope=calc["envelope"], exact_accepted=accept,
                        high_precision_lower=hp_lower, full_bottleneck_pair=full_witness["pair"],
                        full_bottleneck_sign=full_witness["sign"], full_witness_phase=full_witness["phase"],
                        full_witness_score=full_witness["direct"], full_witness_upper=full_witness["upper"],
                        center_bottleneck_pair=center_witness["pair"], center_bottleneck_sign=center_witness["sign"],
                        center_witness_phase=center_witness["phase"], center_witness_score=center_witness["direct"],
                        center_witness_upper=center_witness["upper"], pair_cross_frobenius=pair_norm)
                    fold_paths.append(str(artifact.relative_to(output)))
                    print(f"RF {source}/{fold_id}/{flag}: n={len(ids)} certified={accept.sum()} replay={replay:.2e}", flush=True)
                write_json(fold_root / "LINEAGE.json", lineage[-1])
    if fit_count != 16 or len(raw_rows) != 34080:
        raise RuntimeError("full 16-fold scope incomplete")
    if verify_source_freeze() != inputs_before:
        raise RuntimeError("frozen inputs changed during run")
    write_csv(output / "SELECTIVE_RAW.csv", selective)
    write_csv(output / "FLAG_MECHANISM_RAW.csv", mechanism)
    write_csv(output / "FLAG_MECHANISM_FOLD_SUMMARY.csv", summarize_mechanism(mechanism))
    write_csv(output / "ROW_DECISIONS.csv", raw_rows)
    write_csv(output / "HIGH_PRECISION_ACCEPTANCE.csv", hp_rows)
    write_json(output / "HEAD_LINEAGE.json", lineage)
    # Aggregate matched-k rows by source/flag/method without IID standard errors.
    agg = []
    for source in ("S1","S4"):
        for flag in FLAGS:
            for method in METHODS:
                local=[r for r in selective if r["source"]==source and r["flag"]==flag and r["method"]==method and r["mode"]=="certificate_matched"]
                total=sum(r["total_rows"] for r in local); selected=sum(r["selected_rows"] for r in local)
                unsound=sum(r["verified_unsound_rows"] for r in local); unknown=sum(r["unresolved_rows"] for r in local)
                agg.append(dict(source=source,flag=flag,method=method,folds=len(local),total_rows=total,selected_rows=selected,
                    coverage=safe_rate(selected,total),verified_unsound_rows=unsound,unresolved_rows=unknown,
                    fiber_unsoundness_lower=safe_rate(unsound,selected),fiber_unsoundness_upper=safe_rate(unsound+unknown,selected),
                    reference_head_agreement=safe_rate(sum(r["reference_agreement_rows"] for r in local),selected),
                    device_label_precision=safe_rate(sum(r["device_correct_rows"] for r in local),selected)))
    write_csv(output / "SELECTIVE_MATCHED_SOURCE_SUMMARY.csv", agg)
    summary=dict(status="VERIFIED_V10_RF_E4_E5_DESCRIPTIVE",folds=fit_count,flag_row_cells=len(raw_rows),
        selective_rows=len(selective),mechanism_rows=len(mechanism),accepted=total_accept,
        accepted_reference_violations=total_replay_violations,hp_rechecks=len(hp_rows),
        maximum_full_head_replay_error=replay_max,
        maximum_radius_identity_residual=max(r["radius_identity_residual"] for r in mechanism),
        maximum_score_identity_residual=max(r["score_identity_residual"] for r in mechanism),
        maximum_witness_endpoint_residual=max(r["witness_residual"] for r in mechanism),
        maximum_alignment=max(r["alignment"] for r in mechanism),
        elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=psutil.Process().memory_info().peak_wset,
        scope="q7 exact fibers; fixed heads; fold-descriptive; no new RF activation or population claim",
        unsoundness="verified phase witnesses with outward arithmetic envelope; unresolved counts explicit",
        inputs_unchanged=True, historical_accepts=6049,
        historical_count_note="new conservative measurement-sum envelope need not equal prior binary64 radius implementation")
    write_json(output / "SUMMARY.json", summary)
    manifest=[]
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest.append(dict(path=str(path.relative_to(output)).replace("\\","/"),sha256=sha(path),bytes=path.stat().st_size))
    write_json(output / "RESULT_MANIFEST.json", dict(status="VERIFIED",entries=manifest))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    destination=args.output.resolve()
    if not destination.is_relative_to((HERE/"results/rf").resolve()):
        raise SystemExit("output must be a new named v10 results/rf child")
    run(destination)
