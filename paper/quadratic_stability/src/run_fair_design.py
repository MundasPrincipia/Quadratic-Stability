"""Fixed-reference common-shift-invariant designs; no training or old-result writes."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from numerical_bounds import envelope, gauge_bound, score_enclosure, radius_lower, obs_distance

HERE = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def candidate_witness(h, beta, x, u, k, q, c):
    """Inherited v16 residual-ray search, unchanged grid and verification."""
    y = (c @ u[:, q:].T).reshape(-1)
    mass = np.vdot(y, y).real
    if mass <= 1e-30:
        return None
    hx, hy = h @ x, h @ y
    a0, b0 = float(np.vdot(x, hx).real), float(np.vdot(y, hy).real)
    cross = float(np.vdot(x, hy).real)
    overlap, xmass = float(np.vdot(x, y).real), float(np.vdot(x, x).real)
    err = envelope(len(h), np.linalg.norm(h, "fro") + abs(beta))

    def ray_upper(s):
        gamma, r = np.sqrt(max(0., 1 - s * mass)), np.sqrt(s)
        den = gamma * gamma * xmass + s * mass - 2 * gamma * r * overlap
        return (gamma * gamma * a0 + s * b0 - 2 * gamma * r * cross) / den + beta + 2 * err

    grid = np.r_[np.logspace(-8, np.log10(.25 * (1 - 1e-5)), 60), .25 * (1 - 1e-5)]
    previous = 0.
    for s in grid:
        if ray_upper(s) <= 0:
            left, right = previous, float(s)
            for _ in range(40):
                mid = (left + right) / 2
                if ray_upper(mid) <= 0:
                    right = mid
                else:
                    left = mid
            right = min(float(s), right * (1 + 1e-5) + 1e-12)
            psi = np.sqrt(max(0., 1 - right * mass)) * x - np.sqrt(right) * y
            psi /= np.linalg.norm(psi)
            value, _, upper = score_enclosure(h, beta, psi)
            distance, distance_upper = obs_distance(psi, x, u, k, q)
            if upper <= 0 and distance_upper <= .25:
                return psi, distance, distance_upper, value, upper
        previous = float(s)
    return None


def run(args):
    parent = args.parent.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite a result directory: " + str(out))
    inputs = {}

    def input_path(path):
        path = Path(path).resolve()
        if str(path) not in inputs:
            inputs[str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
        return path

    anchors = read_csv(input_path(parent / "protocol/ANCHORS.csv"))
    old_design = read_csv(input_path(parent / "results/e3_run01/DESIGN_RESULTS.csv"))
    old_pairs = read_csv(input_path(parent / "results/e3_run01/PAIR_RESULTS.csv"))
    base_rows = {int(r["anchor"]): r for r in old_design if r["method"] == "baseline"}
    base_pairs = {(int(r["anchor"]), int(r["competitor"])): r for r in old_pairs if r["method"] == "baseline"}
    if len(anchors) != 160 or set(base_rows) != set(range(160)):
        raise AssertionError("Frozen anchor identities differ")
    out.mkdir(parents=True)
    sources, heads, pair_rows, design_rows = {}, {}, [], []
    start = datetime.now(timezone.utc)
    for ai, anchor in enumerate(anchors):
        source, fold = anchor["source"], anchor["fold"]
        if source not in sources:
            with np.load(input_path(parent / "data/raw_rebuild_run02" / source / "SOURCE_INPUTS.npz"), allow_pickle=False) as f:
                sources[source] = {key: f[key] for key in ("C", "labels")}
        key = (source, fold)
        if key not in heads:
            with np.load(input_path(parent / "data/raw_rebuild_run02" / source / fold / "COMMON_HEAD.npz"), allow_pickle=False) as f:
                heads[key] = {field: f[field] for field in ("class_operators_physical", "raw_intercept", "classes")}
        h, beta, classes = (heads[key][field] for field in ("class_operators_physical", "raw_intercept", "classes"))
        with np.load(input_path(parent / "results/e3_run01" / f"ANCHOR_{ai:03d}_WITNESSES.npz"), allow_pickle=False) as f:
            x = f["base"]
        raw = sources[source]["C"][int(anchor["array_index"])].astype(np.complex128)
        raw /= np.linalg.norm(raw)
        krow = len(raw) // 8
        expected = raw.reshape(krow, 8).copy()
        expected[:, 5:] = 0
        expected = expected.reshape(-1) / np.linalg.norm(expected)
        if np.linalg.norm(expected - x / np.linalg.norm(x)) > envelope(len(x)):
            raise AssertionError("Reference changed")
        old = base_rows[ai]
        if any(str(old[f]) != str(anchor[f]) for f in ("source", "fold", "device", "row_id")):
            raise AssertionError("Anchor metadata changed")
        a = int(np.flatnonzero(classes == int(old["candidate"]))[0])
        b = int(np.flatnonzero(classes == int(old["runner_up"]))[0])
        scores = np.array([score_enclosure(hj, bj, x)[0] for hj, bj in zip(h, beta)])
        if not np.array_equal(np.argsort(-scores, kind="stable")[:2], [a, b]):
            raise AssertionError("Reference candidate or runner-up changed")
        gradients = np.stack([(hj @ x).reshape(krow, 8)[:, 5:] for hj in h])
        centered = gradients - gradients.mean(axis=0)
        competitors = [j for j in range(len(h)) if j != a]
        margins = np.array([float(base_pairs[ai, int(classes[j])]["margin"]) for j in competitors])
        fallback_design = bool(np.any(margins <= 0))
        weighted = centered if fallback_design else np.stack([
            (gradients[a] - gradients[j]) * (margins.min() / margins[index])
            for index, j in enumerate(competitors)])
        rotations = {}
        for name, stack in (("centered_scores", centered), ("margin_differences", weighted)):
            _, _, vh = np.linalg.svd(stack.conj().reshape(-1, 3), full_matrices=True)
            rotations[name] = vh.conj().T
        payload = {"base": x}
        for name, v in rotations.items():
            defect = float(np.linalg.norm(v.conj().T @ v - np.eye(3), "fro"))
            if defect >= .01:
                raise AssertionError("Invalid design basis")
            for added in (1, 2):
                q, d = 5 + added, 3 - added
                u = np.eye(8, dtype=complex)
                u[5:, 5:] = v
                payload[f"{name}_k{added}_basis"] = u
                local = []
                for competitor in competitors:
                    hd, bd = h[a] - h[competitor], float(beta[a] - beta[competitor])
                    saved = base_pairs[ai, int(classes[competitor])]
                    m, ml, _ = score_enclosure(hd, bd, x)
                    if abs(m - float(saved["margin"])) > envelope(len(hd), abs(m)) or abs(ml - float(saved["margin_lower"])) > envelope(len(hd), abs(ml)):
                        raise AssertionError("Saved reference margin mismatch")
                    # These task quantities do not depend on the observation design.
                    m, ml, mh = (float(saved[f]) for f in ("margin", "margin_lower", "M_upper"))
                    g = (gradients[a] - gradients[competitor]) @ v[:, added:].conj()
                    extra = envelope(len(hd), np.linalg.norm(hd, "fro")) + 2 * defect * np.linalg.norm(hd, "fro")
                    gauge = gauge_bound(g, d, extra)
                    if not 0 <= gauge["primal"] <= gauge["upper"] or not np.isfinite(gauge["upper"]):
                        raise AssertionError("Unordered coefficient bounds")
                    dl = radius_lower(ml, gauge["upper"], mh)
                    found = candidate_witness(hd, bd, x, u, krow, q, gauge["direction"]) if ml > 0 else None
                    tag = f"{name}_k{added}_competitor{competitor}"
                    payload[tag + "_direction"] = gauge["direction"]
                    if found is not None:
                        payload[tag + "_witness"] = found[0]
                        if dl > found[2]:
                            raise AssertionError("Radius bounds crossed")
                    r = dict(anchor=ai, source=source, fold=fold, device=int(anchor["device"]), row_id=int(anchor["row_id"]),
                        method=name, k=added, q=q, d=d, candidate=int(classes[a]), competitor=int(classes[competitor]), target_pair=competitor == b,
                        margin=m, margin_lower=ml, M_upper=mh, alpha_lower=gauge["primal"], alpha_upper=gauge["upper"],
                        support_upper=gauge["support_upper"], fallback_upper=gauge["fallback_upper"], used_fallback=gauge["used_fallback"],
                        solver_success=gauge["success"], solver_message=gauge["message"], radius_lower=dl,
                        radius_upper=None if found is None else found[2], censored=found is None,
                        witness_score=None if found is None else found[3], witness_score_upper=None if found is None else found[4],
                        witness_distance=None if found is None else found[1], no_strict_base=ml <= 0, design_fallback=name == "margin_differences" and fallback_design)
                    pair_rows.append(r)
                    local.append(r)
                target = next(r for r in local if r["target_pair"])
                available = [r["radius_upper"] for r in local if r["radius_upper"] is not None]
                tau, tau_upper = obs_distance(raw, x, u, krow, q)
                transfer_pair = max(0., target["radius_lower"] - tau_upper)
                multi_lower = min(r["radius_lower"] for r in local)
                transfer_multi = max(0., multi_lower - tau_upper)
                distance_error = envelope(len(raw)) + 8 * defect / (1 + np.sqrt(1 - defect))
                design_rows.append(dict(anchor=ai, source=source, fold=fold, device=int(anchor["device"]), row_id=int(anchor["row_id"]),
                    method=name, k=added, q=q, d=d, candidate=int(classes[a]), runner_up=int(classes[b]), label=int(sources[source]["labels"][int(anchor["array_index"])]),
                    pair_alpha_lower=target["alpha_lower"], pair_alpha_upper=target["alpha_upper"], pair_radius_lower=target["radius_lower"],
                    pair_radius_upper=target["radius_upper"], pair_censored=target["censored"], multiclass_radius_lower=multi_lower,
                    multiclass_radius_upper=min(available) if available else None, multiclass_censored=not bool(available),
                    solver_failures=sum(not r["solver_success"] for r in local), fallback_count=sum(r["used_fallback"] for r in local),
                    design_fallback=name == "margin_differences" and fallback_design, basis_defect=defect,
                    tau_direct=tau, tau_lower=max(0., tau - distance_error), tau_upper=tau_upper,
                    pair_transfer_lower=transfer_pair, multiclass_transfer_lower=transfer_multi))
        np.savez_compressed(out / f"ANCHOR_{ai:03d}.npz", **payload)
        print(f"Fair design {ai + 1}/160 {source} {fold}", flush=True)
    expected_keys = {(ai, name, k) for ai in range(160) for name in ("centered_scores", "margin_differences") for k in (1, 2)}
    if len(design_rows) != 640 or len(pair_rows) != 5760 or {(r["anchor"], r["method"], r["k"]) for r in design_rows} != expected_keys:
        raise AssertionError("Incomplete run")
    write_csv(out / "PAIR_RESULTS.csv", pair_rows)
    write_csv(out / "DESIGN_RESULTS.csv", design_rows)
    for path in (HERE / "PROTOCOL.md", Path(__file__), HERE / "src/numerical_bounds.py"):
        input_path(path)
    write_json(out / "MANIFEST.json", dict(started=start.isoformat(), completed=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, python=sys.version, numpy=np.__version__, design_cells=len(design_rows), competitor_cells=len(pair_rows),
        numerical_scope="Inherited normwise binary64 assumptions; not formally verified arithmetic"))
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=HERE.parent / "rf_baseline")
    parser.add_argument("--output", type=Path, default=HERE / "results/fair_design_run01")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
