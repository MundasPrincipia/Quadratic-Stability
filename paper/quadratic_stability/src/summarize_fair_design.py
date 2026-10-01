"""Summaries and replacement figures from saved records; no optimizer calls."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parent / "rf_baseline"
V17 = HERE.parent / "reference_transfer"
NAMES = {"raw_scores": "Raw scores", "centered_scores": "Centered scores",
         "margin_differences": "Margin-weighted", "random_mean": "Random", "difference": "Top-two difference", "baseline": "Baseline"}
COMPARATORS = ["raw_scores", "centered_scores", "margin_differences", "random_mean"]


def quantile(x, q=.5):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x[np.isfinite(x)], q)) if np.isfinite(x).any() else None


def save_figure(fig, name, artifacts):
    fig.savefig(artifacts / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(artifacts / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def radius_formula(m, alpha, mh):
    m, alpha, mh = (np.asarray(x, dtype=float) for x in (m, alpha, mh))
    safe_m = np.maximum(m, 0.)
    denominator = alpha + np.sqrt(alpha**2 + 20 * mh * safe_m)
    t = np.full_like(m, np.inf)
    np.divide(2 * safe_m, denominator, out=t, where=denominator > 0)
    value = np.nextafter(np.minimum(.25, t**2) * (1-1e-10), 0.)
    return np.where(m > 0, value, 0.)


def validate_records(design, pairs, new, results):
    """Validate the actual input/output associations before any aggregation."""
    dk, pk = ["anchor", "method", "k"], ["anchor", "method", "k", "competitor"]
    if design.duplicated(dk).any() or pairs.duplicated(pk).any():
        raise AssertionError("Duplicate design or competitor keys")
    anchors = pd.read_csv(PARENT / "protocol/ANCHORS.csv")
    baseline = design[design.method == "baseline"].set_index("anchor")
    if len(anchors) != 160 or set(baseline.index) != set(range(160)):
        raise AssertionError("Incorrect anchor coverage")
    expected_new = {(i, method, k) for i in range(160)
                    for method in ("centered_scores", "margin_differences") for k in (1, 2)}
    if set(new[dk].itertuples(index=False, name=None)) != expected_new:
        raise AssertionError("Incorrect added-design coverage")
    expected_old = {(i, "baseline", 0) for i in range(160)} | {
        (i, method, k) for i in range(160) for method in
        ("difference", "raw_scores", "random_17", "random_29", "random_43") for k in (1, 2)}
    if set(design[dk].itertuples(index=False, name=None)) != expected_old | expected_new:
        raise AssertionError("Incorrect inherited comparison coverage")
    expected_pairs = set()
    for (source, fold), group in baseline.groupby(["source", "fold"]):
        with np.load(PARENT / "data/raw_rebuild_run02" / source / fold / "COMMON_HEAD.npz", allow_pickle=False) as f:
            classes = f["classes"]
        if len(classes) != 10 or len(np.unique(classes)) != 10:
            raise AssertionError("Nonunique class labels")
        for ai, row in group.iterrows():
            anchor = anchors.iloc[ai]
            if any(row[field] != anchor[field] for field in ("source", "fold", "device", "row_id")):
                raise AssertionError("Anchor identity mismatch")
            if tuple(row[["k", "q", "d"]]) != (0, 5, 3):
                raise AssertionError("Incorrect baseline observation")
            if row.candidate == row.runner_up or any(np.count_nonzero(classes == label) != 1 for label in (row.candidate, row.runner_up)):
                raise AssertionError("Candidate/runner-up label mismatch")
            for item in design[design.anchor == ai].itertuples():
                if any(getattr(item, field) != row[field] for field in ("source", "fold", "device", "row_id", "candidate", "runner_up")):
                    raise AssertionError("Design identity mismatch")
                if item.q != 5 + item.k or item.d != 3 - item.k:
                    raise AssertionError("Incorrect design budget")
                expected_pairs.update((ai, item.method, item.k, int(c)) for c in classes if c != row.candidate)
            with np.load(results / f"ANCHOR_{ai:03d}.npz", allow_pickle=False) as payload:
                for method in ("centered_scores", "margin_differences"):
                    for k in (1, 2):
                        u = payload[f"{method}_k{k}_basis"]
                        defect = np.linalg.norm(u.conj().T @ u - np.eye(8), "fro")
                        if not np.isfinite(u).all() or not np.isfinite(defect) or defect >= .01:
                            raise AssertionError("Nonfinite or invalid saved basis")
    if set(pairs[pk].itertuples(index=False, name=None)) != expected_pairs:
        raise AssertionError("Competitor coverage differs from the trained class labels")
    joined = pairs.merge(design[dk + ["source", "fold", "device", "row_id", "candidate", "runner_up", "q", "d"]],
                         on=dk, suffixes=("", "_design"), validate="many_to_one")
    for field in ("source", "fold", "device", "row_id", "candidate", "q", "d"):
        if not joined[field].eq(joined[field + "_design"]).all():
            raise AssertionError("Competitor metadata mismatch: " + field)
    if not joined.target_pair.eq(joined.competitor.eq(joined.runner_up)).all():
        raise AssertionError("Target-pair mismatch")
    fixed = pairs[pairs.method == "baseline"][["anchor", "competitor", "margin", "margin_lower", "M_upper"]]
    joined = pairs.merge(fixed, on=["anchor", "competitor"], suffixes=("", "_baseline"), validate="many_to_one")
    for field in ("margin", "margin_lower", "M_upper"):
        if not joined[field].eq(joined[field + "_baseline"]).all():
            raise AssertionError("Reused task quantity changed: " + field)
    finite_fields = ["margin", "margin_lower", "M_upper", "alpha_lower", "alpha_upper", "radius_lower"]
    if not np.isfinite(pairs[finite_fields].to_numpy()).all():
        raise AssertionError("Nonfinite task or bound")
    if not ((pairs.margin_lower <= pairs.margin) & (pairs.M_upper >= 0) &
            (pairs.alpha_lower >= 0) & (pairs.alpha_upper >= pairs.alpha_lower) &
            (pairs.radius_lower >= 0) & (pairs.radius_lower <= .25)).all():
        raise AssertionError("Unordered task or radius bounds")
    witnessed = pairs[~pairs.censored]
    witness_fields = ["radius_upper", "witness_score", "witness_score_upper", "witness_distance"]
    if not pairs.loc[pairs.censored, witness_fields].isna().all().all():
        raise AssertionError("Censored competitor has an assigned upper bound")
    if not np.isfinite(witnessed[["radius_upper", "witness_score", "witness_score_upper", "witness_distance"]].to_numpy()).all():
        raise AssertionError("Nonfinite witness")
    if not ((witnessed.witness_score_upper <= 0) & (witnessed.witness_distance >= 0) &
            (witnessed.witness_distance <= witnessed.radius_upper) &
            (witnessed.radius_lower <= witnessed.radius_upper) & (witnessed.radius_upper <= .25)).all():
        raise AssertionError("Invalid witness or crossed radius bounds")
    added = pairs[pairs.method.isin(["centered_scores", "margin_differences"])]
    if not np.isfinite(added[["support_upper", "fallback_upper"]].to_numpy()).all() or not (
            (added.alpha_upper <= added.support_upper) & (added.alpha_upper <= added.fallback_upper)).all():
        raise AssertionError("Unordered diagnostic coefficient bounds")
    if not np.isfinite(new[["tau_lower", "tau_direct", "tau_upper", "pair_transfer_lower", "multiclass_transfer_lower"]].to_numpy()).all():
        raise AssertionError("Nonfinite transfer quantity")
    if not ((0 <= new.tau_lower) & (new.tau_lower <= new.tau_direct) & (new.tau_direct <= new.tau_upper)).all():
        raise AssertionError("Unordered distance bounds")
    recomputed = radius_formula(pairs.margin_lower, pairs.alpha_upper, pairs.M_upper)
    if not np.allclose(recomputed, pairs.radius_lower, rtol=1e-12, atol=0):
        raise AssertionError("Saved lower radius differs from the declared formula")
    indexed = design.set_index(dk)
    for key, frame in pairs.groupby(dk):
        row = indexed.loc[key]
        target = frame[frame.target_pair]
        if len(target) != 1:
            raise AssertionError("Missing or multiple target pairs")
        target = target.iloc[0]
        values = {"pair_alpha_lower": target.alpha_lower, "pair_alpha_upper": target.alpha_upper,
                  "pair_radius_lower": target.radius_lower, "pair_radius_upper": target.radius_upper,
                  "multiclass_radius_lower": frame.radius_lower.min(),
                  "multiclass_radius_upper": frame.loc[~frame.censored, "radius_upper"].min()}
        if any(not np.isclose(row[field], value, rtol=1e-12, atol=0, equal_nan=True) for field, value in values.items()):
            raise AssertionError("Design summary disagrees with competitor records")
        if (row.pair_censored != target.censored or row.multiclass_censored != frame.censored.all()
                or row.solver_failures != (~frame.solver_success).sum()):
            raise AssertionError("Design status summary differs from competitors")


def run(results, output=None):
    # Portable replay redirects only generated summaries and figures. Scientific
    # inputs and the source implementation's reference paths remain unchanged.
    if output is not None:
        output = output.resolve()
        if output.exists() or output == results.resolve() or results.resolve() in output.parents:
            raise FileExistsError("Use a fresh output directory outside the saved result directory")
    destination = results if output is None else output
    artifacts = HERE if output is None else output
    old = pd.read_csv(PARENT / "results/e3_run01/DESIGN_RESULTS.csv")
    # The inherited design table stores q,d in its competitor rows only.
    # Recover the declared budgets here and validate them against those rows.
    old["q"], old["d"] = 5 + old.k, 3 - old.k
    new = pd.read_csv(results / "DESIGN_RESULTS.csv")
    all_design = pd.concat([old, new], ignore_index=True)
    all_pairs = pd.concat([pd.read_csv(PARENT / "results/e3_run01/PAIR_RESULTS.csv"),
                           pd.read_csv(results / "PAIR_RESULTS.csv")], ignore_index=True)
    legacy_path = PARENT / "legacy_evidence/matrix/design/run_20260905_044629_927159/design.csv"
    legacy = pd.read_csv(legacy_path)
    transfer = pd.read_csv(V17 / "results/transfer_run01/TRANSFER_RESULTS.csv")
    if len(new) != 640 or len(all_design) != 2400 or len(all_pairs) != 21600:
        raise AssertionError("Wrong comparison coverage")
    if all_design.duplicated(["anchor", "method", "k"]).any():
        raise AssertionError("Duplicate design keys")
    validate_records(all_design, all_pairs, new, results)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "figures").mkdir()
        (output / "tables").mkdir()
    # Propagate the recorded coefficient brackets through the same monotone
    # radius formula. This brackets that formula, NOT the true decision radius.
    optimistic = radius_formula(all_pairs.margin_lower, all_pairs.alpha_lower, all_pairs.M_upper)
    if not np.isfinite(optimistic).all() or np.any(optimistic < all_pairs.radius_lower.to_numpy() * (1-1e-12)):
        raise AssertionError("Unordered formula-value brackets")
    all_pairs["formula_radius_upper"] = optimistic
    formula = all_pairs.groupby(["anchor", "method", "k"], as_index=False).formula_radius_upper.min()
    all_design = all_design.merge(formula, on=["anchor", "method", "k"], validate="one_to_one")
    random = all_design[all_design.method.str.startswith("random_")].groupby(["anchor", "k"], as_index=False).agg(
        source=("source", "first"), fold=("fold", "first"), device=("device", "first"),
        multiclass_radius_lower=("multiclass_radius_lower", "mean"), formula_radius_upper=("formula_radius_upper", "mean"))
    random["method"] = "random_mean"
    # A mean over random radii is a descriptive comparator, not one design's radius.
    random["multiclass_radius_upper"] = np.nan
    random["multiclass_censored"] = True
    compare_data = pd.concat([all_design, random], ignore_index=True)
    dif = all_design[all_design.method == "difference"]
    paired = []
    for method in COMPARATORS:
        comparator = compare_data[compare_data.method == method]
        merged = dif.merge(comparator, on=["anchor", "k"], suffixes=("_d", "_c"), validate="one_to_one")
        for r in merged.itertuples():
            dl, cl = r.multiclass_radius_lower_d, r.multiclass_radius_lower_c
            cu, du = r.multiclass_radius_upper_c, r.multiclass_radius_upper_d
            paired.append(dict(anchor=r.anchor, source=r.source_d, fold=r.fold_d, device=r.device_d, k=r.k, comparator=method,
                difference_lower=dl, comparator_lower=cl, absolute_gain=dl-cl,
                relative_gain=(dl/cl-1) if cl>0 else np.nan,
                formula_gain_lower=dl/r.formula_radius_upper_c-1 if r.formula_radius_upper_c>0 else np.nan,
                formula_gain_upper=r.formula_radius_upper_d/cl-1 if cl>0 else np.nan,
                formula_order_stable=dl>r.formula_radius_upper_c,
                lower_improves=dl>cl,
                separated_difference=bool(method != "random_mean" and not r.multiclass_censored_c and np.isfinite(cu) and dl>cu),
                separated_comparator=bool(method != "random_mean" and not r.multiclass_censored_d and np.isfinite(du) and cl>du)))
    paired = pd.DataFrame(paired)
    paired.to_csv(destination / "PAIRED_COMPARISONS.csv", index=False)
    summaries = []
    for level, groups in [("source", ["source", "k", "comparator"]),
                          ("fold", ["source", "fold", "k", "comparator"]),
                          ("device", ["source", "device", "k", "comparator"])]:
        for keys, frame in paired.groupby(groups, sort=True):
            info = dict(zip(groups, keys))
            summaries.append(dict(level=level, source=info["source"], group=str(info.get("fold", info.get("device", "all"))),
                k=info["k"], comparator=info["comparator"], n=len(frame),
                absolute_gain_median=quantile(frame.absolute_gain), relative_gain_median=quantile(frame.relative_gain),
                relative_gain_q25=quantile(frame.relative_gain, .25), relative_gain_q75=quantile(frame.relative_gain, .75),
                formula_gain_median_lower=quantile(frame.formula_gain_lower),
                formula_gain_median_upper=quantile(frame.formula_gain_upper),
                formula_order_stable=int(frame.formula_order_stable.sum()),
                lower_improves=int(frame.lower_improves.sum()), separated_difference=int(frame.separated_difference.sum()),
                separated_comparator=int(frame.separated_comparator.sum())))
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(destination / "COMPARISON_SUMMARIES.csv", index=False)
    absolute = []
    for (source, method, k), frame in all_design.groupby(["source", "method", "k"]):
        ratio = frame.multiclass_radius_upper/frame.multiclass_radius_lower
        absolute.append(dict(source=source, method=method, k=int(k), n=len(frame),
            lower_median=quantile(frame.multiclass_radius_lower), lower_q25=quantile(frame.multiclass_radius_lower, .25),
            lower_q75=quantile(frame.multiclass_radius_lower, .75), upper_median=quantile(frame.multiclass_radius_upper),
            upper_lower_ratio_median=quantile(ratio), censored=int(frame.multiclass_censored.sum())))
    pd.DataFrame(absolute).to_csv(destination / "ABSOLUTE_RADII.csv", index=False)
    diagnostics = []
    all_pairs["relative_gap"] = (all_pairs.alpha_upper-all_pairs.alpha_lower)/all_pairs.alpha_upper.replace(0, np.nan)
    for (method, success), frame in all_pairs.groupby(["method", "solver_success"]):
        fallback_known = frame.used_fallback.notna()
        diagnostics.append(dict(method=method, solver_success=bool(success), count=len(frame),
            relative_gap_median=quantile(frame.relative_gap), relative_gap_max=float(frame.relative_gap.max()),
            fallback_known=int(fallback_known.sum()), fallback_count=int(frame.loc[fallback_known,"used_fallback"].eq(True).sum())))
    pd.DataFrame(diagnostics).to_csv(destination / "NUMERICAL_SUMMARIES.csv", index=False)
    head_summary = summary_frame[summary_frame.level == "source"].to_dict("records")
    result = dict(anchors=160, old_design_cells=len(old), added_design_cells=len(new), all_design_cells=len(all_design),
        added_competitor_cells=int((all_pairs.method.isin(["centered_scores", "margin_differences"])).sum()),
        comparison=head_summary, absolute=absolute, numerical=diagnostics,
        added_positive_transfer=int((new.multiclass_transfer_lower>0).sum()),
        added_pair_positive_transfer=int((new.pair_transfer_lower>0).sum()),
        added_min_tau_lower=float(new.tau_lower.min()), added_max_tau_upper=float(new.tau_upper.max()),
        all_positive_transfer=int((new.multiclass_transfer_lower>0).sum()+(transfer.multiclass_transfer_lower>0).sum()),
        added_censored_multiclass=int(new.multiclass_censored.sum()), added_design_fallback=int(new.design_fallback.sum()),
        added_non_success=int((all_pairs.method.isin(["centered_scores", "margin_differences"]) & ~all_pairs.solver_success).sum()),
        scipy_version=scipy.__version__, numpy_version=np.__version__,
        parent_inputs=[str(PARENT / "results/e3_run01/DESIGN_RESULTS.csv"),str(PARENT / "results/e3_run01/PAIR_RESULTS.csv"),str(legacy_path),str(V17 / "results/transfer_run01/TRANSFER_RESULTS.csv")])

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.labelsize":8,
        "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7, "pdf.fonttype":42,
        "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(7.15,3.5),sharex=True)
    colors = ["#777777", "#0072B2", "#D55E00", "#009E73"]
    for si, source in enumerate(["S1","S4"]):
        for ki, k in enumerate([1,2]):
            ax = axes[si,ki]
            ax.axhline(0,color="black",lw=.65,ls="--")
            for i, method in enumerate(COMPARATORS):
                vals = 100*paired[(paired.source==source)&(paired.k==k)&(paired.comparator==method)].relative_gain.dropna().to_numpy()
                lo, mid, hi = np.quantile(vals,[.25,.5,.75])
                ax.errorbar(i,mid,yerr=[[mid-lo],[hi-mid]],fmt="o",color=colors[i],ms=4,capsize=3,lw=1.4)
            ax.set_title(f"{source}, k = {k} (n = {130 if source=='S1' else 30})",loc="left",fontsize=9)
            ax.set_xticks(range(4),["Raw", "Centered", "Weighted", "Random"])
            ax.grid(axis="y",alpha=.15)
            if ki==0: ax.set_ylabel(r"Paired change in $\delta_L$ (%)")
    fig.tight_layout(pad=.8)
    save_figure(fig,"wangs2",artifacts)

    # Replace the inherited synthetic plot: all finite bounds enter every summary.
    finite = np.isfinite(legacy.alpha_primal)&np.isfinite(legacy.alpha_upper)&(legacy.alpha_primal>=0)&(legacy.alpha_upper>=legacy.alpha_primal)
    if not finite.all(): raise AssertionError("Unexpected nonfinite or unordered legacy record")
    single = legacy[legacy.kind=="single"].copy()
    bases = single[(single.k==0)&(single.design=="svd")].set_index("family").alpha_upper.to_dict()
    single["ratio"] = single.apply(lambda r:r.alpha_upper/bases[r.family],axis=1)
    fig, axes = plt.subplots(1,4,figsize=(7.15,2.25))
    ordering = []
    for ax,d in zip(axes[:3],[2,3,4]):
        for method,color,marker in [("svd","#0072B2","o"),("random","#E69F00","s")]:
            frame=single[(single.d==d)&(single.design.eq("svd") if method=="svd" else single.design.str.startswith("random"))]
            xs,meds,lows,highs=[],[],[],[]
            for k,f in frame.groupby("k"):
                xs.append(k); meds.append(f.ratio.median()); lows.append(f.ratio.min()); highs.append(f.ratio.max())
            ax.plot(xs,meds,color=color,marker=marker,ms=3,lw=1,label="SVD" if method=="svd" else "Random")
            ax.fill_between(xs,lows,highs,color=color,alpha=.13)
            bad=frame[~frame.success]
            ax.scatter(bad.k,bad.ratio,marker="x",s=18,color="black",linewidths=.8,zorder=5)
        ax.set_title(f"d = {d}",loc="left",fontsize=9); ax.set_xlabel("Added directions k");ax.set_xticks(range(d+1));ax.grid(alpha=.12)
        for k in range(d+1):
            f=single[(single.d==d)&(single.k==k)]
            for use_success in [False,True]:
                g=f[f.success] if use_success else f
                sv=g[g.design=="svd"].ratio; rd=g[g.design.str.startswith("random")].ratio
                ordering.append(dict(d=d,k=k,success_only=use_success,svd_median=quantile(sv),random_median=quantile(rd)))
    axes[0].set_ylabel("Coefficient upper / baseline")
    axes[0].legend(frameon=False)
    multi=legacy[legacy.kind=="multi"].groupby(["family","k","target_rank"],as_index=False).agg(alpha_upper=("alpha_upper","max"),success=("success","all"))
    mb=multi[multi.k==0].set_index("family").alpha_upper.to_dict()
    multi["ratio"]=multi.apply(lambda r:r.alpha_upper/mb[r.family],axis=1)
    multi["offset"]=multi.k-multi.target_rank
    group=multi.groupby("offset").ratio
    axes[3].plot(group.median().index,group.median(),"o-",color="#009E73",ms=3,lw=1)
    axes[3].fill_between(group.min().index,group.min(),group.max(),color="#009E73",alpha=.13)
    bad=multi[~multi.success]
    axes[3].scatter(bad.offset,bad.ratio,marker="x",s=18,color="black",linewidths=.8,zorder=5)
    axes[3].axvline(0,color="gray",ls="--",lw=.6)
    axes[3].set_title("Three-task families",loc="left",fontsize=9);axes[3].set_xlabel(r"k $-$ stacked rank");axes[3].grid(alpha=.12)
    fig.tight_layout(pad=.6)
    save_figure(fig,"wangs5",artifacts)
    result["legacy_all_bound_count"]=int(finite.sum())
    result["legacy_success_comparison"]=ordering
    with (destination/"SUMMARY.json").open("w",encoding="utf-8") as stream: json.dump(result,stream,indent=2,allow_nan=False)

    # Compact source-level exact-number companion to the replacement figure.
    lines=[r"\begin{table}[!htb]",r"\centering\small",r"\caption{Top-two difference design versus invariant comparators. Gains are medians of paired multiclass lower-radius changes; $n=130$ for S1 and $n=30$ for S4. $n_+$ counts positive lower-bound differences. $n_{\rm sep}$ counts disjoint radius intervals favoring the top-two design, under the numerical error assumptions.}",
           r"\label{tab:v18-fair}",r"\begin{tabular}{@{}lclrrr@{}}",r"\toprule",r"Source & $k$ & Comparator & Gain (\%) & $n_+$ & $n_{\rm sep}$\\",r"\midrule"]
    for source in ["S1","S4"]:
        for k in [1,2]:
            for method in ["centered_scores","margin_differences"]:
                row=next(r for r in head_summary if r["source"]==source and r["k"]==k and r["comparator"]==method)
                name="Centered" if method=="centered_scores" else "Weighted"
                lines.append(f"{source} & {k} & {name} & {100*row['relative_gain_median']:.2f} & {row['lower_improves']} & {row['separated_difference']}"+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (artifacts/"tables/fair.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ["comparison","added_positive_transfer","added_min_tau_lower","added_non_success","added_censored_multiclass","added_design_fallback","numerical"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=HERE/"results/fair_design_run01")
    parser.add_argument("--output",type=Path,help="Fresh summary/figure destination; does not modify packaged reference results")
    args=parser.parse_args()
    run(args.results.resolve(), args.output)
