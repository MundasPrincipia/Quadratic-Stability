"""Derived fixed-reference transfer analysis; no training or optimization.

Inputs are the completed v16 E3 records. New output directories must not
exist. The normwise arithmetic model is inherited from numerical_bounds.py;
the calculations are not a formally verified implementation of BLAS.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = HERE.parent / "rf_baseline"
EPS = np.finfo(float).eps
METHODS = ("difference", "raw_scores", "random_17", "random_29", "random_43")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def envelope(n, scale=1.0):
    # Same normwise budget as v16; no optimizer or eigensolver is used.
    t = 8192 * int(n)**2 * EPS
    if t >= 0.01:
        raise ValueError("dimension outside inherited arithmetic budget")
    return float(np.nextafter(t / (1 - t) * (1 + float(scale)), np.inf))


def score_bounds(h, beta, state):
    z = state / np.linalg.norm(state)
    value = float(np.vdot(z, h @ z).real + beta)
    err = envelope(len(h), np.linalg.norm(h, "fro") + abs(beta))
    return value, float(np.nextafter(value - err, -np.inf))


def distances(psi, base, basis, k):
    n = len(psi)
    rows = n // 8
    q, d = 5 + k, 3 - k
    psi = psi / np.linalg.norm(psi)
    base = base / np.linalg.norm(base)
    zmat, xmat = psi.reshape(rows, 8) @ basis.conj(), base.reshape(rows, 8) @ basis.conj()
    z, x = zmat[:, :q].reshape(-1), xmat[:, :q].reshape(-1)
    block = np.outer(z, z.conj()) - np.outer(x, x.conj())
    y, ybase = zmat[:, q:], xmat[:, q:]
    residual = y @ y.conj().T - ybase @ ybase.conj().T
    direct = float(np.sqrt(np.linalg.norm(block, "fro")**2 + np.linalg.norm(residual, "fro")**2 / d))
    s0 = float(np.vdot(psi.reshape(rows, 8)[:, 5:], psi.reshape(rows, 8)[:, 5:]).real)
    newly = zmat[:, 5:q]
    t = float(np.vdot(newly, newly).real)
    s = float(np.vdot(y, y).real)
    formula = float(np.sqrt(s*s + np.linalg.norm(y @ y.conj().T, "fro")**2 / d + 2*t))
    defect = float(np.linalg.norm(basis.conj().T @ basis - np.eye(8), "fro"))
    if defect >= 0.01:
        raise AssertionError("basis outside inherited polar-defect allowance")
    err = envelope(n) + 8*defect/(1 + math.sqrt(1-defect))
    # The exact identity describes the ideal unitary frame. The direct
    # calculation uses the saved reference itself, never a recentered one.
    tolerance = max(1e-10, 4*err)
    mass_error = abs(s+t-s0)
    formula_error = abs(formula-direct)
    if mass_error > tolerance or formula_error > tolerance:
        raise AssertionError(f"mass/formula mismatch: {mass_error}, {formula_error}, {tolerance}")
    upper = float(np.nextafter(max(direct, formula) + err, np.inf))
    lower = max(0., float(np.nextafter(direct - err, -np.inf)))
    return dict(initial_residual_mass=s0, newly_resolved_mass=t, remaining_residual_mass=s,
                tau_direct=direct, tau_formula=formula, tau_lower=lower, tau_upper=upper,
                distance_envelope=err, basis_defect=defect,
                mass_identity_error=mass_error, formula_distance_error=formula_error)


def transfer(radius, tau, error):
    raw = max(0., float(np.nextafter(radius - tau, -np.inf)))
    near = abs(radius-tau) <= max(1e-10, 4*error)
    # Prespecified conservative treatment, retaining rather than deleting it.
    return (0. if near else raw), near


def stats(rows, grouping, source, fold, device, method, k):
    out = dict(grouping=grouping, source=source, fold=fold, device=device, method=method, k=k,
               cells=len(rows), anchors=len({int(r["anchor"]) for r in rows}))
    for prefix in ("pair", "multiclass"):
        out[prefix+"_positive_transfer"] = sum(r[prefix+"_transfer_lower"] > 0 for r in rows)
        out[prefix+"_near_threshold"] = sum(r[prefix+"_near_threshold"] for r in rows)
    columns = ("initial_residual_mass", "newly_resolved_mass", "remaining_residual_mass",
               "tau_upper", "pair_radius_lower", "multiclass_radius_lower",
               "pair_transfer_lower", "multiclass_transfer_lower",
               "pair_slack", "multiclass_slack",
               "paired_multiclass_radius_gain_vs_raw", "paired_multiclass_transfer_gain_vs_raw",
               "paired_multiclass_transfer_relative_vs_raw")
    for key in columns:
        values = np.asarray([r[key] for r in rows if r[key] is not None], dtype=float)
        out[key+"_n"] = len(values)
        for suffix, quantile in (("min",0.),("q25",.25),("median",.5),("q75",.75),("max",1.)):
            out[key+"_"+suffix] = float(np.quantile(values, quantile)) if len(values) else None
    return out


def run(parent, output):
    if output.exists():
        raise FileExistsError("Choose a fresh --output directory; existing results are never overwritten.")
    if not (HERE / "PROTOCOL.md").is_file():
        raise FileNotFoundError("Local pre-analysis protocol is missing.")
    data = parent / "data/raw_rebuild_run02"
    e3 = parent / "results/e3_run01"
    used = set()

    def input_path(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        used.add(path)
        return path

    anchors = read_csv(input_path(parent/"protocol/ANCHORS.csv"))
    design = read_csv(input_path(e3/"DESIGN_RESULTS.csv"))
    pairs = read_csv(input_path(e3/"PAIR_RESULTS.csv"))
    if len(anchors) != 160 or len(design) != 1760 or len(pairs) != 15840:
        raise AssertionError("Fixed E3 row counts differ")
    by_anchor, pair_lookup = defaultdict(list), defaultdict(list)
    for r in design:
        by_anchor[int(r["anchor"])].append(r)
    for r in pairs:
        pair_lookup[(int(r["anchor"]),r["method"],int(r["k"]))].append(r)
    if set(by_anchor) != set(range(160)):
        raise AssertionError("Anchor identities differ")
    source_arrays, head_cache, results = {}, {}, []
    expected_keys = {("baseline",0)} | {(method,k) for method in METHODS for k in (1,2)}
    nested_checked = 0
    max_base_defect = 0.
    for ai, anchor in enumerate(anchors):
        source, fold = anchor["source"], anchor["fold"]
        if source not in source_arrays:
            with np.load(input_path(data/source/"SOURCE_INPUTS.npz"), allow_pickle=False) as f:
                source_arrays[source] = {k:f[k] for k in ("C","labels")}
        key = (source,fold)
        if key not in head_cache:
            with np.load(input_path(data/source/fold/"COMMON_HEAD.npz"), allow_pickle=False) as f:
                head_cache[key] = {k:f[k] for k in ("class_operators_physical","raw_intercept","classes")}
        h, beta, classes = (head_cache[key][k] for k in ("class_operators_physical","raw_intercept","classes"))
        raw = np.asarray(source_arrays[source]["C"][int(anchor["array_index"])], dtype=np.complex128)
        raw = raw/np.linalg.norm(raw)
        with np.load(input_path(e3/f"ANCHOR_{ai:03d}_WITNESSES.npz"), allow_pickle=False) as f:
            base = f["base"]
        ideal_base = raw.reshape(-1,8).copy()
        ideal_base[:,5:] = 0
        ideal_base = ideal_base.reshape(-1)
        ideal_base /= np.linalg.norm(ideal_base)
        base_norm = base/np.linalg.norm(base)
        base_defect = float(np.linalg.norm(base_norm-ideal_base))
        max_base_defect = max(max_base_defect, base_defect)
        if base_defect > envelope(len(raw)):
            raise AssertionError("Saved reference does not match the fixed q=5 reference")
        rows = by_anchor[ai]
        if len(rows) != 11 or {(r["method"],int(r["k"])) for r in rows} != expected_keys:
            raise AssertionError("Fixed design set differs")
        candidates = {int(r["candidate"]) for r in rows}
        if len(candidates) != 1:
            raise AssertionError("Candidate changed under a design")
        candidate = candidates.pop()
        a = int(np.flatnonzero(classes == candidate)[0])
        scores = np.asarray([score_bounds(hj,bj,raw)[0] for hj,bj in zip(h,beta)])
        full_winner = int(classes[int(np.argmax(scores))])
        strict_margins = [score_bounds(h[a]-h[j],float(beta[a]-beta[j]),raw)[1] for j in range(len(h)) if j != a]
        min_margin_lower = min(strict_margins)
        with np.load(input_path(e3/f"ANCHOR_{ai:03d}_BASES.npz"), allow_pickle=False) as saved:
            bases = {name:saved[name] for name in saved.files}
        local = {}
        for old in rows:
            method,k = old["method"],int(old["k"])
            for field in ("source","fold","device","row_id"):
                if str(old[field]) != str(anchor[field]):
                    raise AssertionError(f"Row identity mismatch: {field}")
            competitors = pair_lookup[(ai,method,k)]
            if len(competitors) != len(classes)-1 or {int(r["competitor"]) for r in competitors} != set(map(int,classes))-{candidate}:
                raise AssertionError("Incomplete competitor set")
            if any(int(r["candidate"]) != candidate for r in competitors):
                raise AssertionError("Pair records use another candidate")
            target = [r for r in competitors if r["target_pair"] == "True"]
            if len(target) != 1 or int(target[0]["competitor"]) != int(old["runner_up"]):
                raise AssertionError("Target pair changed")
            pair_radius, multi_radius = float(old["pair_radius_lower"]),float(old["multiclass_radius_lower"])
            if pair_radius != float(target[0]["radius_lower"]) or multi_radius != min(float(r["radius_lower"]) for r in competitors):
                raise AssertionError("Saved lower radii do not match competitor rows")
            u = bases[f"{method}_k{k}"]
            d = distances(raw,base,u,k)
            pair_transfer,pair_near = transfer(pair_radius,d["tau_upper"],d["distance_envelope"])
            multi_transfer,multi_near = transfer(multi_radius,d["tau_upper"],d["distance_envelope"])
            if multi_transfer > 0 and not min_margin_lower > 0:
                raise AssertionError("Positive transfer fails the full-state strict-winner check")
            r = dict(anchor=ai,source=source,fold=fold,device=int(anchor["device"]),
                     row_id=int(anchor["row_id"]),array_index=int(anchor["array_index"]),
                     method=method,k=k,q=5+k,d=3-k,candidate=candidate,runner_up=int(old["runner_up"]),
                     label=int(old["label"]),full_state_winner=full_winner,
                     full_state_candidate_margin_lower=min_margin_lower,
                     same_full_state_winner=full_winner == candidate,
                     saved_reference_defect=base_defect,**d,
                     pair_radius_lower=pair_radius,multiclass_radius_lower=multi_radius,
                     pair_radius_upper=old["pair_radius_upper"],multiclass_radius_upper=old["multiclass_radius_upper"],
                     pair_censored=old["pair_censored"],multiclass_censored=old["multiclass_censored"],
                     solver_failures=int(old["solver_failures"]),
                     pair_transfer_lower=pair_transfer,multiclass_transfer_lower=multi_transfer,
                     pair_near_threshold=pair_near,multiclass_near_threshold=multi_near,
                     pair_slack=pair_radius-d["tau_upper"],multiclass_slack=multi_radius-d["tau_upper"],
                     paired_multiclass_radius_gain_vs_raw=None,
                     paired_multiclass_transfer_gain_vs_raw=None,
                     paired_multiclass_transfer_relative_vs_raw=None)
            local[(method,k)] = r
        baseline = local[("baseline",0)]
        for method in METHODS:
            for k in (1,2):
                r = local[(method,k)]
                if r["tau_upper"] < baseline["tau_lower"]:
                    raise AssertionError("Distance decreased below the baseline beyond the allowance")
                comparator = local[("raw_scores",k)]
                r["paired_multiclass_radius_gain_vs_raw"] = r["multiclass_radius_lower"]-comparator["multiclass_radius_lower"]
                r["paired_multiclass_transfer_gain_vs_raw"] = r["multiclass_transfer_lower"]-comparator["multiclass_transfer_lower"]
                if comparator["multiclass_transfer_lower"] > 0:
                    r["paired_multiclass_transfer_relative_vs_raw"] = r["multiclass_transfer_lower"]/comparator["multiclass_transfer_lower"]-1
            u1,u2 = bases[f"{method}_k1"][:,:6],bases[f"{method}_k2"][:,:7]
            nesting_defect = float(np.linalg.norm((np.eye(8)-u2@u2.conj().T)@u1,"fro"))
            if nesting_defect <= envelope(8):
                nested_checked += 1
                if local[(method,2)]["tau_upper"] < local[(method,1)]["tau_lower"]:
                    raise AssertionError("Nested refinement decreased distance beyond allowance")
            else:
                raise AssertionError("Saved within-method k=1,2 designs are unexpectedly not nested")
        results.extend(local.values())
        if (ai+1) % 20 == 0:
            print(f"Transfer: {ai+1}/160 fixed anchors", flush=True)
    if len(results) != 1760:
        raise AssertionError("A fixed cell was lost")
    summaries = []
    group_specs = (
        ("source", ("source","method","k")),
        ("fold", ("source","fold","method","k")),
        ("device", ("source","device","method","k")),
    )
    for level,fields in group_specs:
        grouped = defaultdict(list)
        for r in results:
            grouped[tuple(r[k] for k in fields)].append(r)
        for key, group in sorted(grouped.items()):
            meta = dict(zip(fields,key))
            summaries.append(stats(group,level,meta["source"],meta.get("fold",""),
                                   meta.get("device",""),meta["method"],meta["k"]))
    overall = dict(
        scope="Fixed v16 E3 reference states and designs; derived transfer only",
        anchors=160,design_cells=1760,pair_records_checked=15840,
        pair_positive_transfer=sum(r["pair_transfer_lower"]>0 for r in results),
        multiclass_positive_transfer=sum(r["multiclass_transfer_lower"]>0 for r in results),
        near_threshold_pair=sum(r["pair_near_threshold"] for r in results),
        near_threshold_multiclass=sum(r["multiclass_near_threshold"] for r in results),
        nested_design_comparisons=nested_checked,
        min_tau_upper=min(r["tau_upper"] for r in results),
        max_tau_upper=max(r["tau_upper"] for r in results),
        max_pair_radius_lower=max(r["pair_radius_lower"] for r in results),
        max_multiclass_radius_lower=max(r["multiclass_radius_lower"] for r in results),
        max_pair_slack=max(r["pair_slack"] for r in results),
        max_multiclass_slack=max(r["multiclass_slack"] for r in results),
        max_formula_distance_error=max(r["formula_distance_error"] for r in results),
        max_mass_identity_error=max(r["mass_identity_error"] for r in results),
        max_saved_reference_defect=max_base_defect,
        full_state_candidate_agreement_anchors=sum(r["same_full_state_winner"] for r in results if r["k"]==0),
        conditional_interpretation="Positive transfer certifies strictly smaller radii. Zero transfer does not mean zero exact decision radius.",
        numerical_scope="Inherited normwise binary64 envelope and polar-defect correction; not formally verified BLAS",
    )
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"TRANSFER_RESULTS.csv",results)
    write_csv(output/"TRANSFER_SUMMARIES.csv",summaries)
    with (output/"SUMMARY.json").open("x",encoding="utf-8") as stream:
        json.dump(overall,stream,indent=2,allow_nan=False)
        stream.write("\n")
    # This inventory names only actually opened inputs. It is not a new
    # cryptographic validation or a mutation of any inherited manifest.
    manifest = dict(
        analysis_version="v17-transfer-1",
        protocol=str((HERE/"PROTOCOL.md").relative_to(HERE)),
        entry_point="src/analyze_rf_transfer.py",
        parent_directory=parent.name,
        inputs=[{"path":str(p.relative_to(parent)).replace("\\","/"),"bytes":p.stat().st_size} for p in sorted(used)],
        outputs=["TRANSFER_RESULTS.csv","TRANSFER_SUMMARIES.csv","SUMMARY.json"],
        checks="Inline identity/row/competitor/refinement checks only; no training, optimizer, or broad test suite",
    )
    with (output/"MANIFEST.json").open("x",encoding="utf-8") as stream:
        json.dump(manifest,stream,indent=2)
        stream.write("\n")
    print(json.dumps(overall,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=DEFAULT_PARENT)
    parser.add_argument("--output",type=Path,default=HERE/"results/transfer_run01")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.parent.resolve(),args.output.resolve())
