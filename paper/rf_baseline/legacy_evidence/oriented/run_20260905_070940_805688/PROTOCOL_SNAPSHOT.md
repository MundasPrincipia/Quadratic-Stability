# v12 oriented endpoint regression protocol

Created UTC: 2026-09-05 06:59:35. Status: APPROVED by the root agent on 2026-09-05 after review of this protocol and NUMERIC_REDUCTION_NOTES.md. Approval was received before execution; the exact freeze time is recorded in FROZEN_INPUTS.json. This is a local prospective protocol, not an external preregistration.

## Scope and claims

R16 checks the newly explicit oriented endpoint and fixed-head margin consequences on the v11 common model. R10 documents the exact inherited gauge routine and the objects certified by interval arithmetic. R11 reaggregates the unmodified inherited 432 design records, including all 41 non-success statuses. No inherited experiment is rerun, no RF head is refitted, and no arbitrary-dimensional endpoint optimizer is claimed.

The two claims are: (C1) separate full-set lower and upper endpoints imply the indicated directed threshold only when the downward coefficient is positive; (C2) a zero coefficient does not in general imply a flat lower endpoint. The quadratic positive control is globally downward-flat, while the linear-leading control descends quadratically.

## Fixed model and prescribed grid

Take K=1, q=2, d=1, x=e1, z=diag(1,0,0), delta in [0,1/4]. A normalized complex state psi=(u,w,v) has s=|v|^2 and p=|w|^2. Its full observation budget is exactly 2(s^2+p)<=delta^2; every such nonnegative pair with s+p<=1 has a normalized complex representative. On this delta window normalization is automatic. See proofs/NUMERIC_REDUCTION_NOTES.md for full-set proofs.

The seven fixed matrices, in this order, are E13+E31, E12+E21, E23+E32, E22, I3, -E22, and diag(0,-1,1). Save them explicitly as complex matrices. No matrix or base state depends on the margin.

Endpoint grid: the nine exact symbolic tolerances 10^(-2-j/2), j=0,...,8, for all seven tasks (63 rows). Margin grid: m=10^(-r), r=2,...,6, for all seven tasks (35 rows). The score is F_m(psi)=psi^*H psi + (m-h), h=e1^*H e1; only this intercept changes with m. Save the intercept in each margin row.

## Exact targets

Let A(delta)=2 sqrt[(delta/sqrt(2))(1-delta/sqrt(2))], B(delta)=sqrt(2) delta sqrt(1-delta^2/2), and C(delta)=(2/3)^(3/4) delta^(3/2).

| Task | Lower endpoint | Upper endpoint | Downward loss | First-failure tolerance for positive m |
|---|---|---|---|---|
| E13+E31 | -A | A | A | m^2/[sqrt(2)(1+sqrt(1-m^2))] |
| E12+E21 | -B | B | B | m/sqrt(1+sqrt(1-m^2)) |
| E23+E32 | -C | C | C | sqrt(3/2) m^(2/3) |
| E22 | 0 | delta^2/2 | 0 | none; globally flat downward |
| I3 | 1 | 1 | 0 | none; constant |
| -E22 | -delta^2/2 | 0 | delta^2/2 | sqrt(2m) |
| diag(0,-1,1) | -delta^2/2 | delta/sqrt(2) | delta^2/2 | sqrt(2m) |

All 25 non-flat roots for the prescribed margins lie in (0,1/4). The ten flat rows are saved with an explicit no-failure status and no invented finite root or positive coefficient. The first-failure endpoint itself has margin zero and therefore fails the strict-positive rule.

## Evidence layers and witnesses

Binary64 rows independently form rho=psi psi^*, apply the block observation, and evaluate trace(H rho). Save both lower and upper complex state witnesses, raw density matrices and observations. Use an exact unit phase (3+4i)/5 for nonzero extremizer coordinates; for the mixed task a separate phase on u also creates non-real observed off-diagonal entries. Additionally save one fixed-phase feasible complex probe at every task/tolerance pair, with w=i sqrt(p), v=(3+4i)sqrt(s)/5, s=delta/sqrt(6), p=delta^2/3. This checks the phase-sensitive raw calculation inside the same prescribed grid; it is not a new optimization sweep.

Interval certificates use mpmath interval arithmetic at 100 decimal places; ordinary mpmath arithmetic uses 130 places only to choose candidates and display values. Exact binary interval endpoint tuples are authoritative; displayed decimal bounds are rounded outwards. Mathematical normalization follows from the symbolic square-root construction. Raw interval normalization must enclose 1, and raw interval observation squared must have upper bound at most the lower bound of delta^2 for an inward witness built at (1-10^-30)delta.

For each of 63 endpoint rows, certify separately L in [L_analytic.lower, score_lower_witness.upper] and U in [score_upper_witness.lower, U_analytic.upper]. The outward analytic sides are global because of the full-set scalar proof, not because the witness is an optimizer. Save the endpoint intervals, raw-score intervals, observation intervals, normalization enclosures, ordering and feasibility booleans.

For each of 25 non-flat margins, certify the unique first root using strict outward interval signs of D(delta)-m on a dyadic bracket in [0,1/4]; stop at absolute width <=10^-50. Exact bisection points are passed to interval arithmetic without decimal truncation. Monotonicity is proved separately. Evaluate the closed-form root by outward arithmetic and require overlap with the sign-certified bracket. Store lower/upper raw complex threshold witnesses at the bracket ends as binary64 diagnostics; their binary64 margins are not the root certificate. For each flat case certify D(1/4)=0<m and record the separate analytical global-flat reason. No root slope is fitted for flat cases.

## Metrics, success and failures

Decisive: all 63 endpoint interval certificates and all 35 root/control records pass; all 25 roots have strict certified endpoint signs; 63+35 prescribed rows are retained. Secondary: maximum binary64 score/budget/normalization discrepancies, maximum oriented interval widths, exact-formula/threshold relative errors, and descriptive all-grid log slopes for five positive-loss tasks. Do not use slopes as proofs or tune the grids after seeing them.

Any exception produces a retained failed row and the exception text. A failed certificate blocks a passed numerical conclusion; it does not trigger an unrecorded rerun or grid deletion. Software corrections are permitted only with a new timestamped output directory and a recorded reason.

## Provenance and run order

1. Root review accepts this protocol and the scalar reduction before new experiments.
2. Implement src/v12_oriented_regression.py and focused tests/test_v12_oriented.py; freeze hashes immediately before execution.
3. Create results/oriented/run_<UTC timestamp>/, exclusively. Save PROTOCOL_SNAPSHOT.md, run configuration, input matrices/witnesses, CSV records, interval certificates, SUMMARY.json and MANIFEST.json.
4. Manifest SHA-256 hashes cover the fixed protocol, reduction notes, executed code, tests, inherited numerics_common.py, inherited design.csv and every new output except the manifest itself. Record runtime versions and code/protocol hashes before running numerical work.
5. Separately aggregate all inherited design.csv rows by (kind,d,seed,k), and also provide one family summary per (kind,d,seed) in TeX. Define success exactly by the saved Boolean; report valid finite bounds independently, never silently replace a non-success by a success. Reconcile totals 432=391+41 and preserve access to every original row.
6. Hand completed evidence and the numerical supplement snippet to root; root incorporates main prose and figures. Planned snippet labels: app:v12-numerical-details, eq:v12-saved-dual-bound, eq:v12-numeric-seven-endpoints, tab:v12-design-status.

Estimated budget: no GPU, no training, under two CPU minutes for the prescribed interval checks and tests; elapsed implementation/review time is separate. There are no nice-to-have sweeps, stochastic seeds, new baselines or frontier-model components.
