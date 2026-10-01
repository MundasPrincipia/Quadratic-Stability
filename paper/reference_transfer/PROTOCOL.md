# v17: one-sided classification and fixed-reference RF transfer

Declared locally on 2026-09-19 before the new derived analysis. This is a prospective internal analysis protocol, not a public preregistration.

## Scope and parent

Parent: `../rf_baseline` (v16). Preserve its manuscripts, PDFs, data, code, protocols, results, manifests and archives unchanged. The new manuscript inherits the current v16 source, not the earlier source snapshot inside its experimental ZIP.

The only new scientific work is (i) an extension of Proposition 6 to singular positive-semidefinite residual blocks at the same resolved-only pure-state boundary, and (ii) a derived transfer analysis of the existing E3 designs. Theorem 3, Theorem 14, their constants, the trained classifiers, source scope and experimental designs remain unchanged. No new raw-IQ reconstruction, training, sample selection, random design or optimizer run is authorized by this protocol.

## Theory target

For g=b=0 and A=K^r positive semidefinite, use the norm restricted to ker(A). Nonzero coupling J on that kernel gives a three-halves downward loss with its sharp coefficient and an explicit fixed-matrix remainder. Otherwise use the Moore--Penrose Schur complement to distinguish quadratic loss from global flatness. Handle A=0 and a zero-dimensional tangent space explicitly. Combine this result with the existing endpoint laws to state an exhaustive one-sided classification at the existing boundary, not at arbitrary positive-rank centers. Keep all proofs in the manuscript. No priority claim is added.

## Fixed RF inputs

Use every one of the 160 preselected anchors in `protocol/ANCHORS.csv`, the 1,760 cells in `results/e3_run01/DESIGN_RESULTS.csv`, and the corresponding saved bases, reference states and pair records. Reconstructed source states come from `data/raw_rebuild_run02/{S1,S4}/SOURCE_INPUTS.npz`; classifier heads come from each saved fold's `COMMON_HEAD.npz`. These are read-only v16 inputs.

Keep the same physical reference state, candidate, competitor set and design in every cell. There is one k=0 baseline and five designs at each k=1,2: difference-head, raw-score and random seeds 17,29,43. Do not replace zero, unsuccessful or censored records. Do not select a new boundary reference after adding directions.

## Estimands and numerical policy

For each cell record the initial residual mass s0, newly resolved mass t, remaining mass s, direct observation distance, its mass-formula value and a conservative upper bound tau_U. The exact identity is tau^2=s^2+||YY^dagger||_F^2/(3-k)+2t. Use the inherited normwise floating-point envelope and temporal-basis defect correction for the direct distance. If necessary enlarge that bound to cover disagreement with the formula; report any material discrepancy rather than silently changing the data. This is conservative numerical analysis under the inherited arithmetic assumptions, not a formal verification of BLAS.

The two primary derived endpoints are max(0, delta_L_pair-tau_U) and max(0, delta_L_multiclass-tau_U), using saved v16 lower radii. A positive value certifies radii strictly below it. A zero does not assert that the exact decision radius is zero. If |delta_L-tau_U| is at most max(1e-10,4 times the distance envelope), retain the row and resolve it with higher precision or conservatively withhold a positive transfer claim.

Required consistency checks are internal to this analysis: fixed row count and identities; s+t=s0 within the numerical allowance; mass formula versus direct distance; nondecreasing distance from k=0 under refinement; k=1 to k=2 monotonicity only after verifying nested subspaces; and agreement of any positive multiclass transfer with the original state's strict winner under all fixed competitors. A violated consistency condition stops the analysis; it does not trigger replacement of the sample or design.

## Summaries and interpretation

Retain all 1,760 rows. Summarize by source/method/k and separately by fold and device. Report positive-transfer counts, absolute lower radii, distances, transfer radii, mass scales and paired absolute design differences. No inference treats repeated anchor--design cells as independent samples. Relative comparisons with a zero denominator are undefined, not zero percent. No re-optimization or alternative design is introduced after observing the result. If all transfer radii are zero, report that outcome explicitly.

The abstract observation-cost convention uses the full Hermitian fixed-point space including its trace coordinate: D(k)=[K(q+k)]^2+K^2 for k<d, and D(d)=(KT)^2. Repetition and scalar-coordinate costs are sufficient budgets under the existing Gaussian model, not measured RF hardware costs.

## Deliverables

One derived analysis entry point, its row/summary results, one compact manifest and one README; updated single-column, double-column and supporting PDFs. No separate review report, new experiment suite or broad integrity-testing campaign. Replays write to a fresh result directory and must not overwrite a completed run.
