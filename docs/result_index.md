# Result-to-file index: Quadratic Stability

This index uses paths relative to the repository root. The
scientific records retain their original run and version names. Figure and
table numbers with an S prefix refer to the Supporting Results; unprefixed
numbers refer to the main paper. Section numbers are stable navigation targets;
use the supplement's contents page for its current page numbers.

## Directory aliases

| Alias used below | Directory relative to the repository root |
|---|---|
| RF | `paper/rf_baseline` |
| TRANSFER | `paper/reference_transfer` |
| PAPER | `paper/quadratic_stability` |

Replace an alias with its directory when locating a file. For example,
`PAPER/results/boundary_refinement_run01/ENDPOINTS.json` means
`paper/quadratic_stability/results/boundary_refinement_run01/ENDPOINTS.json`.

## Results and their records

| Result or location | Frozen records relative to the aliased directory | Replay or checking route |
|---|---|---|
| Main-paper exact same-task refinement example, Figure 2 | `PAPER/results/boundary_refinement_run01/CONFIG.json`, `ENDPOINTS.json`, `MARGINS.json`, `SUMMARY.json`; plotted in `PAPER/figures/wang2.pdf` | `boundary-refinement`: 39 endpoint cells and 27 margin cells at 100 decimal digits |
| RF interface and exact-fiber counts, Section V and Table S2 | `RF/data/raw_rebuild_run02/SUMMARY.json`, `BASELINE_REPLAY.csv`; source inputs in `RF/data/raw_rebuild_run02/*/SOURCE_INPUTS.npz`; fold heads in `RF/data/raw_rebuild_run02/*/*/COMMON_HEAD.npz` and six flag records in `RF/data/raw_rebuild_run02/*/*/*_BASELINE.npz` per fold | `baseline` uses the saved derived inputs; raw-IQ reconstruction has separate acquisition requirements |
| Boundary applicability and finite-tolerance brackets, Section V | `RF/results/e1_run01/` retains the natural-row and stratified summaries; regenerate its 96 source/fold/flag arrays with `rf-applicability`; `RF/results/analysis_run03/E1_SOURCE_FLAG.csv`, `E1_RESIDUAL_DISTRIBUTIONS.csv`, `E1_FINITE_BRACKETS.csv` | `rf-applicability` or `baseline`; saved summaries cover all 408,960 cells, including the 101,918 satisfying the joint applicability test |
| Controlled task perturbations, Figure S1 and the scalar reduction in Section VII | `RF/results/e2_run02/CANONICAL_INTERVALS.csv`, `ANCHOR_CHECKS.csv`, `SUMMARY.json`; `RF/results/analysis_run03/E2_WINDOWS.csv` | `baseline`; run02 is authoritative for the 972 canonical interval cells and 160 RF embeddings. Run01 is retained as provenance |
| Historical observation-design radii, Table S1 | `RF/results/e3_run01/DESIGN_RESULTS.csv`, `PAIR_RESULTS.csv`, `ANCHOR_*_BASES.npz`, `ANCHOR_*_WITNESSES.npz`; paired summaries in `RF/results/analysis_run03/E3_*.csv` | `baseline` reruns 1,760 design cells and 15,840 competitor rows; `rf-summaries` reads frozen records |
| Fixed-reference transfer and sufficient-budget ratios, Tables S3--S4 | `TRANSFER/results/transfer_run01/TRANSFER_RESULTS.csv`, `TRANSFER_SUMMARIES.csv`, `SUMMARY.json` | `transfer`; all 1,760 historical transfer cells are retained. Transfer lower bounds are distinct from the boundary-reference budget comparisons |
| Common-shift-invariant design comparisons, Figure S2 and Table S5 | `PAPER/results/fair_design_run01/DESIGN_RESULTS.csv`, `PAIR_RESULTS.csv`, `PAIRED_COMPARISONS.csv`, `COMPARISON_SUMMARIES.csv`, `ABSOLUTE_RADII.csv`, `NUMERICAL_SUMMARIES.csv` | `fair-design` reruns the 640 added cells and 5,760 competitor rows; `rf-summaries` recomputes comparisons from these records and the inherited E3 records |
| Full-budget matrix checks and four rates, Section VII | `RF/legacy_evidence/matrix/budget_rates/run_20260905_045339_758316/E1_BUDGET.csv`, `E1_ENDPOINT_BRACKETS.csv`, `E2_RATES.csv`, `E2_INTERVAL_CERTIFICATES.json` | Preserved matrix records; these are separate from the RF E1/E2/E3 runs |
| Positive residual rank and weak activation, Figures S3--S4 | `RF/legacy_evidence/matrix/positive_rank/run_20260905_124913_710642/endpoint_brackets.csv`, `interval_certificates.json`; `RF/legacy_evidence/matrix/weak_margin/run_20260905_130201/weak_profile.csv`, `margin_transition.csv`, `INTERVAL_CERTIFICATES.json` | Complete saved matrix endpoint and threshold records, including all 27 positive-rank brackets |
| Matrix observation design, Figure S5 and Table S6 | `RF/legacy_evidence/matrix/design/run_20260905_044629_927159/design.csv`, `SUMMARY.json` | All 432 evaluations remain in the record, including 324 solver calls and 41 non-success statuses |

In each table cell, a filename without a repeated directory uses the preceding
directory in that cell. Patterns such as `ANCHOR_*_BASES.npz` denote all matching
files, not a selected subset. The JSON manifests describe configurations and
lineage. Their historical path and source-digest fields retain the original run values. Internal release checks are maintained separately from submission requirements.

## RF interface check

`python -B reproduce.py rf-interface --output <fresh-directory>` checks the saved
RF inputs across 13 S1 folds, three S4 folds and six flags. The reference records
are in `paper/checks/rf_interface/`. Its README distinguishes all-row class/pair
differences from the direct pair-matrix subsample. The mathematical interface
is described in the main paper and Supporting Results Section V-A.

The [data acquisition and processing guide](data_acquisition_and_processing.md)
gives provider links, subsets and preprocessing, with the additional selection
metadata required by the original raw-IQ reconstruction route.

## Replay environments

Use `reproduce.py` at the portable root and a fresh output directory outside the
reference directories. The README at the portable root gives the commands and scope.
The accompanying [data acquisition and processing guide](data_acquisition_and_processing.md)
lists the exact S1/S4 source subsets and preprocessing steps, and identifies
the additional historical inputs required for the raw-IQ route. Bundled
derived-input replay does not require downloading those recordings again.

| Mode | Recorded environment or requirements |
|---|---|
| `boundary-refinement` | Python 3.12; root `requirements-example.txt`; no SciPy dependency |
| `rf-summaries`, `fair-design` | Root `requirements-rf.txt`; recorded SciPy 1.18.0 |
| `transfer` | NumPy-based transfer calculation; the historical requirements above are sufficient |
| `rf-applicability` | Reconstructs 96 E1 arrays using the frozen numerical routine and validates original array payload digests; same recorded RF environment as `baseline` |
| `baseline` | `RF/requirements-replay.txt`; recorded Python 3.12.6, NumPy 2.4.4 and SciPy 1.17.1, with one BLAS thread |

Check the imported numerical-library versions when comparing replay outputs.
Library differences can affect optimizer statuses and small numerical values.
Scientific values and coverage are compared separately from paths and environment metadata.

The RF bounds are conditional on the declared normwise binary64 error budgets.
The canonical scalar interval calculation has its own reduction and outward
rounding guarantee. Neither replay nor the canonical intervals turn the RF
pipeline into an operation-by-operation interval proof. Failed, censored and
non-improving cases remain part of the frozen records.

## Command locations in the GitHub repository

The root `reproduce.py` supports `boundary-refinement`, `rf-interface`,
`rf-summaries`, and `rf-applicability`. It uses the included files by default.
The historical `baseline`, `fair-design`, and `transfer` labels above refer to
`paper/rf_baseline/src/replay_baseline.py --derived`,
`paper/quadratic_stability/src/run_fair_design.py`, and
`paper/reference_transfer/src/analyze_rf_transfer.py`, respectively.
Use each script's `--help` for its arguments. The source-data acquisition
route has the separate prerequisites listed in the acquisition guide.
