# Quadratic-Stability

Code accompanying **Sharp Local Stability of Quadratic Functionals under
Unitary Averaging**, by **Hongyuan Wang**, College of Informatics, Harbin Institute
of Technology, Shenzhen.

Repository: [https://github.com/MundasPrincipia/Quadratic-Stability](https://github.com/MundasPrincipia/Quadratic-Stability)  
Contact: **2025310659@stu.hit.edu.cn**  
ORCID: [0009-0007-5829-4266](https://orcid.org/0009-0007-5829-4266)

This repository provides the analytic observation-refinement example, RF
processing and analysis code, and reference results accompanying the paper.
The analytic example runs directly from the included configuration. RF checks
and summary calculations use saved inputs from a separate reproducibility
companion.

## Examples and analysis

| Entry point | Inputs | Results |
|---|---|---|
| `boundary-refinement` | Included configuration | 39 endpoint cells, 27 margin cells, 100-digit interval checks, and the four-panel figure |
| `rf-interface` | External complete companion with saved S1/S4 inputs and fold heads | Numerical consistency check for 13 S1 folds, 3 S4 folds, and 6 observation flags |
| `rf-summaries` | External complete companion with saved designs, pair records, heads, and witnesses | Design/transfer summaries, tables, and figures |

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

The RF commands require derived S1/S4 inputs, saved classifiers and measurement
vectors, and design/witness records from the separate
`reproducibility/` directory of the supplementary archive
`quadratic_stability_supplementary_materials.zip`. This repository includes RF
source code and aggregate reference tables. Contact
[2025310659@stu.hit.edu.cn](mailto:2025310659@stu.hit.edu.cn) for companion access details.

Extract the supplementary archive into a separate directory. Replace the
`--companion` path below with its `reproducibility/` directory, containing
`rf_baseline/`, `reference_transfer/` and `analysis/`. The containing supplement
directory and the earlier companion layout are also accepted.

```text
python -m pip install -r requirements-rf.txt
python -B reproduce.py rf-interface --companion ../quadratic_stability_supplementary_materials/reproducibility --output ../quadratic_rf_interface
python -B reproduce.py rf-summaries --companion ../quadratic_stability_supplementary_materials/reproducibility --output ../quadratic_rf_summaries
```

`rf-interface` compares feature-based, quadratic, pairwise, and coordinate
representations of the saved classifier scores at tolerance `1e-8`. It writes
FOLD_FLAG_CHECKS.csv and SUMMARY.json.

`rf-summaries` validates the saved records and rebuilds the comparison tables and
figures for 2,400 design cells. The [reproduction details](docs/reproduction_scope.md)
describe the required inputs, numerical conventions, and result files.

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
| `paper/rf_baseline/provenance/raw_code_snapshot/` | Preserved helper-code snapshot |
| `paper/reference_transfer/` | Transfer-analysis source and aggregate results |
| `docs/` | Data acquisition, input requirements, and numerical details |

## Citation and terms

See [CITATION.cff](CITATION.cff) for the author and manuscript information.

Software license: pending.
Third-party data access and use follow the original providers' terms.
