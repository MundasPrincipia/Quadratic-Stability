# Quadratic-Stability

Code accompanying **Sharp Local Stability of Quadratic Functionals under
Unitary Averaging**, by **Hongyuan Wang**, College of Informatics, Harbin Institute
of Technology, Shenzhen.

Repository: [https://github.com/MundasPrincipia/Quadratic-Stability](https://github.com/MundasPrincipia/Quadratic-Stability)  
Contact: **2025310659@stu.hit.edu.cn**  
ORCID: [0009-0007-5829-4266](https://orcid.org/0009-0007-5829-4266)

This repository contains the deterministic analytic example, RF processing and
analysis source code, and selected frozen aggregate results. The analytic example
is self-contained. RF replay requires the separately supplied complete
`quadratic_stability_reproducibility` companion; the commands below never download
data or train a model.

## What can be reproduced here?

| Entry point | Inputs | Output and scope |
|---|---|---|
| `boundary-refinement` | Included configuration; no RF data | 39 endpoint cells, 27 margin cells, 100-digit interval checks, and the four-panel figure |
| `rf-interface` | External complete companion with saved S1/S4 inputs and fold heads | Numerical consistency check for 13 S1 folds, 3 S4 folds, and 6 observation flags |
| `rf-summaries` | External complete companion with saved designs, pair records, heads, and witnesses | Validated design/transfer summaries, tables, and figures; no design optimization |

Original IQ, derived per-sample arrays, trained fold parameters, measurement banks,
and per-anchor witness files are **not included in this code repository**. The
small RF CSV files under `paper/` are aggregate reference results; they alone
cannot regenerate the full RF analysis. The complete companion is a separate
artifact, and no public download location for it is asserted here. Contact the
author for access questions.

## Run the analytic example

Use Python 3.12; the recorded environment is Python 3.12.6. Install the example
dependencies in your preferred environment, then run from this repository root:

```text
python -m pip install -r requirements-example.txt
python -B reproduce.py boundary-refinement --output ../quadratic_example_replay
```

The output directory must not already exist and must be outside this repository.
It will contain `CONFIG.json`, `ENDPOINTS.json`, `MARGINS.json`, `SUMMARY.json`,
and a `figures/` directory with PDF and PNG plots. `SUMMARY.json` reports the
check status and numerical environment. Frozen reference records are in
[`paper/quadratic_stability/results/boundary_refinement_run01/`](paper/quadratic_stability/results/boundary_refinement_run01/).

These interval checks verify the low-dimensional analytic example. They do not
replace its mathematical proof or certify the RF pipeline by interval arithmetic.

## Replay saved RF inputs

First extract the **complete companion** into a separate directory. In the
examples below, replace `../quadratic_stability_reproducibility` with its actual
extracted root, containing `paper/rf_baseline`, `paper/reference_transfer`, and
`paper/quadratic_stability`.

```text
python -m pip install -r requirements-rf.txt
python -B reproduce.py rf-interface --companion ../quadratic_stability_reproducibility --output ../quadratic_rf_interface
python -B reproduce.py rf-summaries --companion ../quadratic_stability_reproducibility --output ../quadratic_rf_summaries
```

Both modes read frozen records and write only to the fresh output directory.
The wrapper requires the external input path explicitly and limits BLAS threads
to one. The summary adapter changes input-directory bindings only; the copied
scientific implementation is unchanged. Different numerical-library versions
may change replay values; use the recorded requirements for comparisons.

`rf-interface` checks feature, saved, quadratic, pairwise, and coordinate scores
at tolerance `1e-8`. It is a **numerical consistency check**, not an independent
refit or an RF interval certificate. RF numerical bounds are evaluations of
analytic bounds conditional on the declared binary64 normwise error budgets.
Local applicability, endpoint tightness, and successful decision certification
remain distinct. In particular, the reported fixed-reference transfer lower
bounds are zero for all 2,400 design cells; zero lower bounds do not imply zero
true radii.

## Processing original recordings

The [data acquisition and processing guide](docs/data_acquisition_and_processing.md)
gives the S1/S4 provider links, selected subsets, normalization, feature map,
training-fold conventions, and historical prerequisites. Original-recording
downloads alone do not supply every selection artifact needed by the unchanged
historical raw-replay procedure. That procedure is not one of this repository's
self-contained entry points.

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
| `docs/` | Input requirements, acquisition steps, and reproduction limits |

Historical identifiers inside scientific source and frozen records are retained
for traceability. The public entry points and repository name use the paper's
current subject, rather than an internal revision label.

## Citation and terms

[`CITATION.cff`](CITATION.cff) records the author and associated manuscript title.
It does not assert a DOI, publication acceptance, or a repository release date.
No software license has been selected for this repository.
Third-party data access and use follow the original providers' terms.

See [reproduction scope](docs/reproduction_scope.md) for what is included,
what remains external, and how the checks should be interpreted.
