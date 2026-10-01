# v18: common-shift-invariant observation-design comparisons

Declared locally on 2026-09-22 before running the added designs. This is an internal prospective protocol, not public preregistration. Manuscript parent: v17; numerical inputs: unchanged v16 E3. All parent manuscripts, data, results, manifests and archives remain read-only.

## Question and fixed inputs

Does the top-two difference design improve finite multiclass radius bounds compared with common-score-shift-invariant designs, at the same boundary reference and observation budget? The comparison may be negative. It does not test device-label accuracy, hardware cost, or establish an optimal multiclass design.

Use all 160 anchors from v16 `protocol/ANCHORS.csv` (130 S1 and 30 S4), the saved q=5 reference states, trained fold-specific heads, candidate class, runner-up, all nine competitors, and old raw-score/difference/random results. No IQ reconstruction, retraining, new dataset, relabeling, anchor replacement, or change of reference is permitted. Random comparators retain seeds 17,29,43 and are averaged within anchor before aggregation.

## Two deterministic added designs

Write G_j for the K-by-3 residual gradient matrix at the saved reference. Both rules reveal the leading k right singular vectors of the entrywise-conjugated, vertically stacked matrix, for k=1,2; k=0 reuses the existing baseline.

1. `centered_scores`: stack G_j - mean_l G_l over all classes.
2. `margin_differences`: for fixed candidate a, stack (G_a-G_b)/m_ab over every b != a, with m_ab the saved nominal positive reference score difference. A common positive rescaling by min_b m_ab is allowed for numerical conditioning and does not change singular subspaces. No tunable weight or fitted clipping constant is introduced. If any nominal margin is nonpositive, keep the anchor with a deterministic centered-design fallback and flag it; never divide by a nonpositive margin or remove the record.

The weighted rule is a singular-vector surrogate for controlling the worst alpha_ab/m_ab, not a proven optimizer of that maximum or of finite radius. Centering and candidate-to-all differences have the same joint cancellation subspace; their different finite-budget weighting, not different rank thresholds, motivates the comparison.

## Evaluation and failures

Add exactly 640 design cells and 5,760 competitor records. Reuse the inherited coefficient optimizer, support-residual/Frobenius upper bound, task-norm bound, radius formula, witness ray, search grid, search cap 1/4 and numerical allowances. Reuse saved task-norm bounds and reference margins after checking their row identities and recomputed margins. Save all bases, coefficient directions, failure witnesses, solver statuses, fallback choices and coefficient intervals. Non-success status never removes a row. The inherited normwise binary64 allowances remain assumptions, not a formal operation-by-operation rounding proof.

Primary comparisons: paired multiclass delta_L differences/ratios between top-two difference design and each new comparator at fixed k; absolute lower and upper radii; counts of delta_L(A)>delta_U(B) only when B has an actual failure witness. Include all anchors, zeros, censored bounds and non-improvements. Ratios with zero denominator are unavailable. Report source, fold and device summaries; repeated design cells are not independent samples. Recompute original-to-reference distances and transfer bounds for the added designs, without changing the reference or candidate. Do not infer a zero true radius from a zero lower bound.

Secondary diagnostics: relative coefficient interval widths and support/Frobenius fallback counts by method and optimizer status. Replot the existing 432-row synthetic design study using all finite ordered bounds; distinguish non-success points without excluding them from the aggregate. This reaggregation does not rerun that experiment.

## Execution and delivery

One new execution entry point, one summary/figure entry point, one result directory with raw records and a compact input manifest, one README, and the three manuscript PDFs. Existing numeric helper code is copied into this version and only augmented with diagnostics. Input hashes are recorded once for the files actually read; no broad integrity or regression-test campaign. Required implementation checks are fixed-key coverage, same reference/class/competitors, finite ordered intervals, witness score upper bound <= 0, and finite unitary-basis defects. Stop on violations instead of replacing configurations.

The main manuscript retains every proof of its main results. Compress prose and replace existing experimental displays rather than appending a second results narrative. IEEE TIT limits verified on 2026-09-22: at most 50 single-column pages at submission and 25 double-column pages in the final version, including references. The separate supplement remains explicitly referenced. No claim of full independent mathematical or machine-arithmetic certification is made.
