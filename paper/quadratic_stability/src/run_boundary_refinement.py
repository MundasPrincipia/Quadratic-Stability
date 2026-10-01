"""Exact three-design example with 100-digit outward interval verification.

No RF inputs, optimization, training or Monte Carlo. Requires mpmath.
The symbolic full-set proof is in appendices/general_and_refinement.tex.
Intervals enclose exact algebraic witnesses, not rounded unit-vector files.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import mpmath as mp

HERE = Path(__file__).resolve().parents[1]
iv = mp.iv


def frac(t):
    sign, mantissa, exponent, _ = t
    v = Fraction((-1 if sign else 1) * mantissa)
    return v * 2**exponent if exponent >= 0 else v / 2**(-exponent)


def ends(x):
    return tuple(frac(t) for t in x._mpi_)


def decimal_out(v, rounding):
    with localcontext() as ctx:
        ctx.prec = 110
        ctx.rounding = rounding
        return str(Decimal(v.numerator) / Decimal(v.denominator))


def record(x):
    lo, hi = ends(x)
    return {
        "lower": decimal_out(lo, ROUND_FLOOR),
        "upper": decimal_out(hi, ROUND_CEILING),
        "binary_endpoints": [list(t) for t in x._mpi_],
        "plot_midpoint": float((lo + hi) / 2),
    }


def contains_zero(x):
    lo, hi = ends(x)
    assert lo <= 0 <= hi, record(x)


def positive(x):
    assert ends(x)[0] > 0, record(x)


def ceiling(v):
    return -((-v.numerator) // v.denominator)


def unique_ceiling(x):
    lo, hi = ends(x)
    n0, n1 = ceiling(lo), ceiling(hi)
    assert n0 == n1 and n0 >= 1, (n0, n1)
    return n0


def norm_cdf(z):
    """Outward enclosure from the alternating erf series, z near 1.645.

    For x=z/sqrt(2) in (1,1.2), successive term ratios are <1.
    221 terms (indices 0..220) end positive; the next term bounds
    the remainder. All arithmetic, including pi, uses the interval context.
    """
    x = z / iv.sqrt(2)
    assert Fraction(1) < ends(x)[0] <= ends(x)[1] < Fraction(6, 5)
    term = x
    total = iv.mpf(0)
    for n in range(221):
        total += term if n % 2 == 0 else -term
        term *= x*x * (2*n + 1) / ((n+1)*(2*n+3))
    # Next term is negative, and the alternating remainder is in [-term,0].
    erf = 2 / iv.sqrt(iv.pi) * (total + iv.mpf([-term.b, 0]))
    return (1 + erf) / 2


def quantile():
    """Use high precision only to propose a bracket; certify both CDF signs."""
    proposed = mp.sqrt(2) * mp.erfinv(mp.mpf("0.9"))
    scale = 10**95
    n = int(mp.floor(proposed * scale))
    lo = iv.mpf(n) / scale
    hi = iv.mpf(n+1) / scale
    target = Fraction(19, 20)
    assert ends(norm_cdf(lo))[1] < target
    assert ends(norm_cdf(hi))[0] > target
    return iv.mpf([lo.a, hi.b])


def loss(design, delta):
    a = iv.sqrt(iv.mpf(2)/3)
    if design == "A":
        return 2 * iv.sqrt(a*delta*(1-a*delta))
    if design == "B":
        return iv.sqrt(2)*delta*iv.sqrt(1-delta*delta/2)
    s = delta/iv.sqrt(2)
    return 2*iv.sqrt(s*(1-s))


def witness(design, delta, sign):
    if design == "A":
        mass = iv.sqrt(iv.mpf(2)/3)*delta
    elif design == "B":
        mass = delta*delta/2
    else:
        mass = delta/iv.sqrt(2)
    return [iv.sqrt(1-mass), sign*iv.sqrt(mass), iv.mpf(0)]


def observe(rho, resolved):
    """Build the full 3-by-3 averaged matrix directly in physical coordinates."""
    unresolved = [i for i in range(3) if i not in resolved]
    r = len(unresolved)
    out = [[iv.mpf(0) for j in range(3)] for i in range(3)]
    for i in resolved:
        for j in resolved:
            out[i][j] = rho[i][j]
    mass = sum((rho[i][i] for i in unresolved), iv.mpf(0))
    for i in unresolved:
        out[i][i] = mass/r
    return out


def direct_checks(cfg, design, delta, sign):
    psi = witness(design["id"], delta, sign)
    rho = [[psi[i]*psi[j] for j in range(3)] for i in range(3)]
    obs = observe(rho, design["resolved"])
    ref = cfg["reference"]
    ref_rho = [[iv.mpf(ref[i]*ref[j]) for j in range(3)] for i in range(3)]
    ref_obs = observe(ref_rho, design["resolved"])
    distance_sq = sum(((obs[i][j]-ref_obs[i][j])**2
                       for i in range(3) for j in range(3)), iv.mpf(0))
    score = sum((cfg["task"][i][j]*rho[j][i]
                 for i in range(3) for j in range(3)), iv.mpf(0))
    norm_error = sum((v*v for v in psi), iv.mpf(0))-1
    contains_zero(norm_error)
    contains_zero(distance_sq-delta*delta)
    contains_zero(score-sign*loss(design["id"], delta))
    return {
        "sign": sign, "coordinates": [record(v) for v in psi],
        "norm_error": record(norm_error),
        "distance_squared": record(distance_sq),
        "budget_identity_residual": record(distance_sq-delta*delta),
        "score": record(score),
        "endpoint_residual": record(score-sign*loss(design["id"], delta)),
    }, distance_sq, score


def critical(design, margin):
    hm = 1+iv.sqrt(1-margin*margin)
    if design == "A":
        return margin*margin/(2*iv.sqrt(iv.mpf(2)/3)*hm)
    if design == "B":
        return margin/iv.sqrt(hm)
    return margin*margin/(iv.sqrt(2)*hm)


def grid(spec):
    vals = [(f'10^({spec["start_exponent"]}+{j}/{spec["step_denominator"]})',
             iv.mpf(spec["base"])**(iv.mpf(spec["start_exponent"])+iv.mpf(j)/spec["step_denominator"]))
            for j in range(spec["count"])]
    vals.extend((s,iv.mpf(s)) for s in spec.get("extra", []))
    return sorted(vals, key=lambda p: ends(p[1])[0])


def run(output, config):
    if output.exists():
        raise FileExistsError("Output must be a fresh directory: "+str(output))
    cfg = json.loads(config.read_text(encoding="utf-8"))
    iv.dps = cfg["precision_decimal_digits"]
    mp.mp.dps = iv.dps+30
    assert cfg["task"] == [[0,1,0],[1,0,0],[0,0,0]]
    assert cfg["reference"] == [1,0,0]
    assert [(d["id"],d["resolved"],d["D"]) for d in cfg["designs"]] == [
        ("A",[0],2),("B",[0,1],5),("C",[0,2],5)]
    assert (cfg["sigma"],cfg["eta"],cfg["gamma"]) == ("1","0.05","0.9")
    eps = iv.mpf(cfg["strict_witness_slack"])
    q = quantile()
    ell = iv.ln(iv.mpf(1)/iv.mpf(cfg["eta"]))
    endpoints, margins = [], []
    for design in cfg["designs"]:
        for tag, delta in grid(cfg["tolerance_grid"]):
            entry = {"design":design["id"],"D":design["D"],"grid":tag,
                     "delta":record(delta),"loss":record(loss(design["id"],delta)),
                     "attaining_witnesses":[]}
            for sign in (-1,1):
                w,_,_ = direct_checks(cfg,design,delta,sign)
                entry["attaining_witnesses"].append(w)
            if ends(delta)[0] > 0:
                inward = delta*(1-eps)
                iw, dsq, _ = direct_checks(cfg,design,inward,-1)
                positive(delta*delta-dsq)
                entry["strictly_inward_witness"] = iw
            endpoints.append(entry)
        b2 = design["D"] + 2*iv.sqrt(design["D"]*ell) + 2*ell
        for tag,m in grid(cfg["margin_grid"]):
            r = critical(design["id"],m)
            positive(iv.mpf("0.25")-r)
            tie = m-loss(design["id"],r)
            contains_zero(tie)
            sufficient_radius = iv.mpf(cfg["gamma"])*r
            positive(m-loss(design["id"],sufficient_radius))
            failure_radius = r*(1+eps)
            fw,_,fs = direct_checks(cfg,design,failure_radius,-1)
            positive(-(m+fs))
            lower = 4*q*q/(r*r)
            upper = 4*b2/(sufficient_radius*sufficient_radius)
            nl,nu = unique_ceiling(lower),unique_ceiling(upper)
            assert nl <= nu
            margins.append({
                "design":design["id"],"D":design["D"],"grid":tag,
                "margin":record(m),"critical_radius":record(r),
                "tie_residual":record(tie),
                "strictly_positive_lower_score":record(m-loss(design["id"],sufficient_radius)),
                "strict_failure_radius":record(failure_radius),
                "strict_failure_score":record(m+fs),"failure_witness":fw,
                "normal_quantile":record(q),"b_squared":record(b2),
                "N_necessary_real":record(lower),"N_sufficient_real":record(upper),
                "N_lower":nl,"N_upper":nu,
                "scalar_lower":design["D"]*nl,"scalar_upper":design["D"]*nu,
            })
    assert len(endpoints)==39 and len(margins)==27
    # Same-task refinement must reduce loss and increase the first-failure radius.
    for tag,delta in grid(cfg["tolerance_grid"]):
        if ends(delta)[0] > 0:
            positive(loss("A",delta)-loss("C",delta))
            positive(loss("C",delta)-loss("B",delta))
    for tag,m in grid(cfg["margin_grid"]):
        positive(critical("B",m)-critical("C",m))
        positive(critical("C",m)-critical("A",m))
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (("CONFIG.json",cfg),("ENDPOINTS.json",endpoints),("MARGINS.json",margins)):
        (output/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    summary = {
        "status":"PASS","precision_decimal_digits":iv.dps,
        "endpoint_cells":len(endpoints),"positive_tolerance_cells":36,"zero_tolerance_cells":3,
        "margin_cells":len(margins),"all_cells_retained":True,
        "normal_quantile_bracket_certified":True,
        "all_integer_bound_ceilings_unambiguous":True,
        "strict_inward_and_failure_witnesses_checked":True,
        "RF_retraining":False,"Monte_Carlo":False,
        "scope":"Numerical verification of the exact analytic example; full-set coverage is proved in the manuscript.",
        "interval_serialization":"Exact binary endpoint tuples; decimal strings rounded outward at 110 digits; plot_midpoint is visualization only.",
        "environment":{"python":platform.python_version(),"mpmath":mp.__version__},
        "inputs_sha256":{config.name:hashlib.sha256(config.read_bytes()).hexdigest(),
                        Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("status","endpoint_cells","margin_cells","precision_decimal_digits")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=HERE/"protocol/boundary_refinement.json")
    args=parser.parse_args()
    run(args.output.resolve(),args.config.resolve())
