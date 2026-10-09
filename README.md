# Quadratic-Stability

Code accompanying **Sharp Local Stability of Quadratic Functionals under
Unitary Averaging**, by **Hongyuan Wang**, College of Informatics, Harbin Institute
of Technology, Shenzhen.

Repository: [https://github.com/MundasPrincipia/Quadratic-Stability](https://github.com/MundasPrincipia/Quadratic-Stability)  
Contact: **2025310659@stu.hit.edu.cn**  
ORCID: [0009-0007-5829-4266](https://orcid.org/0009-0007-5829-4266)

Version **1.0.0** is available in [Release v1.0.0](https://github.com/MundasPrincipia/Quadratic-Stability/releases/tag/v1.0.0).
Published: **10-Oct-2026 00:41:47** (**Asia/Shanghai**, UTC+08:00),
equivalently **2026-10-09T16:41:47Z**.

This repository provides the analytic observation-refinement example, RF
processing and analysis code, processed S1/S4 inputs, saved classifiers, and
frozen results accompanying the paper. The input and result files are taken
from the supplementary materials prepared for submission. The analytic example
and saved-input RF checks run from the files included here.

The Supporting Results PDF is submitted separately with the manuscript as
`quadratic_stability_supplement.pdf`. Code, data-access instructions and
reproduction documentation are maintained in this repository.

## Examples and analysis

| Entry point | Inputs | Results |
|---|---|---|
| `boundary-refinement` | Included configuration | 39 endpoint cells, 27 margin cells, 100-digit interval checks, and the four-panel figure |
| `rf-interface` | Included S1/S4 inputs and fold heads | Numerical consistency check for 13 S1 folds, 3 S4 folds, and 6 observation flags |
| `rf-applicability` | Included saved inputs and original array digests | Rebuild and verify the 96 E1 per-sample arrays |
| `rf-summaries` | Included designs, pair records, heads, and witnesses | Design/transfer summaries, tables, and figures |

## Run the analytic example

Use Python 3.12; the recorded environment is Python 3.12.6. Install the example
dependencies in your preferred environment, then run from this repository root:

```text
python -m pip install -r requirements-example.txt
python -B reproduce.py boundary-refinement --output ../quadratic_example_replay
```

The command creates the sibling directory `../quadratic_example_replay` with
the following outputs. Choose a new sibling directory for each run.

| Output | Contents |
|---|---|
| CONFIG.json | Parameters and evaluation grids |
| ENDPOINTS.json | Exact-loss evaluations, interval checks, and endpoint witnesses |
| MARGINS.json | Critical radii and necessary/sufficient sampling bounds |
| SUMMARY.json | Check status and numerical environment |
| figures/ | Four-panel figure in PDF and PNG formats |

The [reference records](paper/quadratic_stability/results/boundary_refinement_run01/)
provide the corresponding results used in the paper.

## Replay saved RF inputs

The processed inputs and saved models are in
[`paper/rf_baseline/data/raw_rebuild_run02/`](paper/rf_baseline/data/raw_rebuild_run02/).
Design and witness records are in the corresponding `results/` directories;
reference checks are in [`paper/checks/`](paper/checks/). The
[result index](docs/result_index.md) maps the paper's results to these files.

Run the following commands from the repository root:

```text
python -m pip install -r requirements-rf.txt
python -B reproduce.py rf-interface --output ../quadratic_rf_interface
python -B reproduce.py rf-summaries --output ../quadratic_rf_summaries
```

An existing extracted companion can still be selected with
`--companion <path-to-reproducibility>`.

`rf-interface` compares feature-based, quadratic, pairwise, and coordinate
representations of the saved classifier scores at tolerance `1e-8`. It writes
FOLD_FLAG_CHECKS.csv and SUMMARY.json.

`rf-summaries` validates the saved records and rebuilds the comparison tables and
figures for 2,400 design cells. The [reproduction details](docs/reproduction_scope.md)
describe the required inputs, numerical conventions, and result files.

## Rebuild the omitted applicability arrays

The repository includes the S1/S4 derived inputs, frozen classifiers,
summary tables and design records. The 96 E1 per-sample arrays can be reconstructed:

```text
python -B reproduce.py rf-applicability --output ../quadratic_rf_applicability
```

The output includes `results/e1_run01/` and `REPLAY_CHECK.json`. The command runs
the original E1 calculation and checks array payloads against the recorded digests,
along with the frozen E1 CSVs and scientific summary fields. It uses the saved
classifier without refitting or design optimization.

## Processing original recordings

The [data acquisition and processing guide](docs/data_acquisition_and_processing.md)
gives the S1/S4 download links, selected subsets, normalization, feature map, and
training-fold conventions. Rebuilding the historical study from IQ recordings
also requires the selection metadata and workspace files listed in Section 6
of that guide.

## Repository layout

| Location | Contents |
|---|---|
| `reproduce.py` | Supported public entry points |
| `paper/quadratic_stability/src/` | Analytic, task-interface, design, and summary implementations |
| `paper/quadratic_stability/protocol/` | Frozen analytic-example configuration |
| `paper/quadratic_stability/results/` | Analytic reference records and selected RF aggregate CSVs |
| `paper/rf_baseline/src/` | Historical RF processing and analysis source |
| `paper/rf_baseline/data/` | Processed S1/S4 inputs, saved fold classifiers and baseline records |
| `paper/rf_baseline/results/` | Frozen applicability, perturbation and design records |
| `paper/rf_baseline/provenance/raw_code_snapshot/` | Preserved helper-code snapshot |
| `paper/reference_transfer/` | Transfer-analysis source and saved results |
| `paper/checks/` | Saved interface checks and applicability-array digests |
| `docs/` | Data acquisition, input requirements, and numerical details |

## Citation and terms

See [CITATION.cff](CITATION.cff) for the author and manuscript information.

Software license: pending.
Third-party data access and use follow the original providers' terms, described
in the [data-source notes](docs/data_sources.md).
