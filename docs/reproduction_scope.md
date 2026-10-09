# Reproduction details

## Self-contained analytic example

The included configuration specifies 39 endpoint cells and 27 margin cells.
`run_boundary_refinement.py` evaluates the analytic formulas and their witnesses
using 100-digit interval arithmetic with outward rounding. The inputs are the
configured designs, tolerance grid, and margin grid.

The frozen `SUMMARY.json` retains the source digest and environment of its
original run. A later documentation-only change to the computation script changed
its file digest. Each replay records its current script digest. Compare
`CONFIG.json`, `ENDPOINTS.json`, `MARGINS.json`, and scientific summary fields
separately from source-digest and environment metadata.

## RF numerical consistency check

`rf-interface` runs `check_rf_interface.py` using this repository's saved inputs
by default. An external companion can still be selected with `--companion`.
Paths below are relative to `paper/`. Its inputs are:

- `rf_baseline/data/raw_rebuild_run02/FLAGS.npz`;
- S1 and S4 `SOURCE_INPUTS.npz`;
- all 13 S1 and 3 S4 saved `COMMON_HEAD.npz` files.

The 16 folds and six flags yield 96 fold/flag cells. The check includes the score
and coordinate identities detailed in its generated `SUMMARY.json`. Standardized
coefficients are reconstructed algebraically from saved raw parameters. The
threshold is `1e-8`, applied to the score and operator identities evaluated by
the check. Results are written to `FOLD_FLAG_CHECKS.csv` and `SUMMARY.json`.

RF calculations evaluate analytical bounds conditional on the binary64 normwise
error budgets described in the supplement. The interface check tests numerical
consistency of the classifier representations. The analytic example uses the
separate interval-arithmetic procedure described above.

## RF summary replay

`rf-summaries` uses the included saved inputs, including:

- baseline anchors, fold class labels, designs, and competitor records;
- later fair-design records and 160 saved anchor/witness payloads;
- historical design comparisons and fixed-reference transfer records.

The existing implementation checks identities, class/competitor coverage, saved
bases, numerical ordering, and bound formulas before aggregation. It covers
2,400 design cells and 21,600 competitor records.
The generated CSVs include `ABSOLUTE_RADII.csv`, `COMPARISON_SUMMARIES.csv`,
`NUMERICAL_SUMMARIES.csv`, and `PAIRED_COMPARISONS.csv`. The first three are
included as aggregate references; per-anchor paired records are included in
`paper/quadratic_stability/results/fair_design_run01/`.
Output path fields can differ across installations.

The reference CSVs also include historical applicability and transfer summaries
by source, fold, and device. The `rf-summaries` command rebuilds the four comparison
CSVs listed above; the other historical tables are provided as saved results.

All 2,400 fixed-reference transfer cells have a recorded lower bound of zero.
These are the computed lower bounds on the transfer radii; the exact radii remain
unresolved where the available bounds do not determine them.

## Reconstructible E1 intermediates

The repository provides inputs for regenerating the 96 per-sample NPZ files below
`rf_baseline/results/e1_run01/S1/` and `S4/`. It includes the source inputs,
classifiers, flag records, natural-row table, stratified applicability table,
and all finite-bracket and aggregate summaries. `rf-applicability` reruns the
frozen E1 numerical routine from those inputs into a new directory and checks
every array payload against `checks/rf_applicability/REFERENCE_ARRAYS.json`.
The accompanying CSVs and scientific summary fields are compared separately.
The default entry point uses the unchanged RF implementation in
`paper/rf_baseline/src/`. External companions retain their existing layout;
a data-only companion can use the same repository implementation.

## Historical processing and optimization code

The historical source covers IQ processing, classifier fitting, observation
design, and transfer calculations. Running those scripts requires the original
recordings, selection metadata, and workspace layout described in the
[acquisition guide](data_acquisition_and_processing.md). The root-level commands cover the analytic example, saved-classifier checks,
E1 reconstruction, and saved-result aggregation.
