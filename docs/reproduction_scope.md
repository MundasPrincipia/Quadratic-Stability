# Reproduction scope

## Self-contained analytic example

The example configuration and all 39 endpoint and 27 margin reference cells are
included without modification. The computation uses 100-digit interval checks,
with outward rounding as implemented in `run_boundary_refinement.py`. It does
not use RF data, training, random resampling, or Monte Carlo.

The frozen `SUMMARY.json` retains the source digest and environment of its
original run. A later documentation-only change to the computation script changed
its file digest. A new replay must report the actual current digest; compare
`CONFIG.json`, `ENDPOINTS.json`, `MARGINS.json`, and scientific summary fields
separately from source-digest and environment metadata. Frozen reference records
are not overwritten to hide such metadata differences.

## RF numerical consistency check

`rf-interface` runs this repository's unchanged `check_rf_interface.py` against
the explicitly supplied complete companion. It needs:

- `paper/rf_baseline/data/raw_rebuild_run02/FLAGS.npz`;
- S1 and S4 `SOURCE_INPUTS.npz`;
- all 13 S1 and 3 S4 saved `COMMON_HEAD.npz` files.

The 16 folds and six flags yield 96 fold/flag cells. The check includes the score
and coordinate identities detailed in its generated `SUMMARY.json`. Standardized
coefficients are reconstructed algebraically from saved raw parameters; this is
not an independent comparison to originally stored standardized coefficients or
a new fit. The threshold is `1e-8`. These checks support implementation
consistency, not unconditional machine certification of RF endpoints.

## RF summary replay

`rf-summaries` uses saved inputs from the complete companion, including:

- baseline anchors, fold class labels, designs, and competitor records;
- later fair-design records and 160 saved anchor/witness payloads;
- historical design comparisons and fixed-reference transfer records.

The existing implementation checks identities, class/competitor coverage, saved
bases, numerical ordering, and bound formulas before aggregation. It covers
2,400 design cells and 21,600 competitor records. It does not rerun an optimizer.
The generated CSVs include `ABSOLUTE_RADII.csv`, `COMPARISON_SUMMARIES.csv`,
`NUMERICAL_SUMMARIES.csv`, and `PAIRED_COMPARISONS.csv`. The first three are
included here as small aggregate references; per-anchor paired records remain in
the complete companion. Output path fields can differ across installations.

The source-level and fold/device-level CSV aggregates included in this repository
are evidence snapshots, not substitutes for the inputs required for validation.
Historical applicability and transfer aggregate CSVs are included for inspection;
the supported summary mode does not regenerate every historical CSV.

## Historical processing and optimization code

The preserved source shows how original IQ was processed and how other RF
calculations were performed. These historical scripts have additional file-layout
and input dependencies; their presence is not a claim that this code-only
repository supports raw-data-to-paper replay. The
[acquisition guide](data_acquisition_and_processing.md) identifies these limits.
Training, raw-IQ reconstruction, and new design optimization are outside the three
supported replay modes.

## Relation to the submission

This repository is the smaller code companion. The manuscript, supplement, complete
companion, and their frozen records are maintained separately. No manuscript
conclusion, numerical result, or scientific implementation was changed when
preparing this repository. Publishing this code companion does not modify the
manuscript or the separate complete companion.
