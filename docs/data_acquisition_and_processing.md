# RF data acquisition and processing

Paper: *Sharp Local Stability of Quadratic Functionals under Unitary Averaging*  
Contact: Hongyuan Wang, 2025310659@stu.hit.edu.cn  
Prepared: 2026-10-01

## 1. Scope of this code repository

This repository contains processing source code and selected aggregate results.
It does **not** contain original IQ, derived per-sample arrays, trained fold
parameters, measurement banks, row-selection files, or per-anchor witnesses.
The analytic example runs without any RF inputs. The supported RF commands and
their external input requirements are in [the main README](../README.md).

Sections 2--6 below preserve the acquisition and processing description of the
separate `quadratic_stability_reproducibility.zip` companion. Unless identified
as historical workspace paths, the paths in those sections refer to that
**complete extracted companion**, not to files promised by this smaller
repository. References to bundled lineage, heads, or inputs mean bundled in
the complete companion. No public download location for that companion is
asserted here; access questions can be directed to the author.

Provider downloads alone do not restore every historical selection artifact.
Section 6 lists the additional prerequisites, including items absent even from
the complete companion. Raw reconstruction is therefore a conditional historical
procedure, not a one-command capability of this repository.

## 2. Obtain the original recordings, if raw reconstruction is needed

### S1: LoRa_RFFI, Shen et al.

- [Provider README and dataset description](https://github.com/gxhen/LoRa_RFFI/blob/main/Openset_RFFI_TIFS/readme.md).
- [IEEE DataPort dataset entry](https://ieee-dataport.org/open-access/lorarffidataset),
  linked by that README; dataset DOI: [10.21227/qqt4-kz19](https://doi.org/10.21227/qqt4-kz19).
- Cite G. Shen, J. Zhang, A. Marshall, and J. R. Cavallaro,
  “Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification
  for LoRa,” *IEEE Transactions on Information Forensics and Security*, 2022.

Follow the provider's access instructions and retain the accompanying source and
license information. This guide supplies the provider's published DataPort link;
the DataPort page could not be independently retrieved during this documentation
update. A fresh download or a particular account's access was not tested.

The frozen study uses these **13 test files**, not the provider's augmented
training data or its neural-network training pipeline:

| Location under the provider's test-data directory | Files |
|---|---|
| Test root | `dataset_residential.h5` |
| `channel_problem/` | `A.h5`, `B.h5`, `C.h5`, `D.h5`, `E.h5`, `F.h5` |
| `channel_problem/` | `B_walk.h5`, `F_walk.h5`, `moving_office.h5`, `moving_meeting_room.h5` |
| `channel_problem/` | `B_antenna.h5`, `F_antenna.h5` |

The selected pool contains devices **31--40**, with **40 rows per device per
environment**: 400 rows per file and 5,200 rows in total. Use the recorded rows;
selecting the first 40 packets or drawing a new sample would change the inputs.

The exact file identities, sizes, and existing source digests are recorded in
`paper/rf_baseline/data/raw_rebuild_run02/S1/RAW_SOURCE_HASHES.json`.
The corresponding `RAW_ROW_LINEAGE.csv` gives each selected file and its
zero-based HDF5 `raw_row`, device, environment, and output `row_id`.
For the unchanged historical script, the local test-data directory is:

```text
<historical-workspace>/ieee_dataport/LoRa_RFFI-main/LoRa_RFFI/dataset/Test/
```

The provider may use `Dataset/Test` in its distribution. The path above is the
historical local layout expected by this project's selection manifest; preserve
the required spelling and case when recreating that layout.

### S4: Oregon State LoRa collection, Elmaghbub and Hamdaoui

- [Official release note, October 2023, Version 2](https://research.engr.oregonstate.edu/hamdaoui/sites/research.engr.oregonstate.edu.hamdaoui/files/release_note_lora_datasets_final_oct2023_v2.pdf).
- [Provider dataset directory](https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/).
- [Different Days Indoor subset](https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup/).
- Cite A. Elmaghbub and B. Hamdaoui, “LoRa Device Fingerprinting in the Wild:
  Disclosing RF Data-Driven Fingerprint Sensitivity to Deployment Variability,”
  *IEEE Access*, vol. 9, pp. 142893--142909, 2021.

The frozen study uses **40 IQ files** from `Diff_Days_Indoor_Setup`, on
**Day1, Day3, and Day5**. In the following table, `1` means `IQ_1.dat` and `2`
means `IQ_2.dat`, inside the indicated `Day*/Device*/` directory. This table
lists the exact selected files; it does not mean every device/day has both files.

| Device directory | Day1 | Day3 | Day5 |
|---|---|---|---|
| `Device2` | 1 | 1, 2 | 2 |
| `Device3` | 1 | 1, 2 | 1 |
| `Device7` | 1, 2 | 1 | 1 |
| `Device8` | 1 | 1, 2 | 1 |
| `Device9` | 1, 2 | 2 | 1 |
| `Device10` | 1 | 1 | 1, 2 |
| `Device11` | 2 | 2 | 1, 2 |
| `Device13` | 1, 2 | 1 | 2 |
| `Device14` | 1, 2 | 1 | 1 |
| `Device15` | 1 | 1, 2 | 2 |

Each selected file contributes **12 frames**, giving **480 frames**: 168 on
Day1, 168 on Day3, and 144 on Day5. The selected device labels are
2, 3, 7, 8, 9, 10, 11, 13, 14, and 15, with 48 frames per device.
These project subset counts are distinct from the provider's full collection.

`paper/rf_baseline/data/raw_rebuild_run02/S4/RAW_SOURCE_HASHES.json` contains
the exact 40 file paths and source identities. `RAW_ROW_LINEAGE.csv` in that
directory gives each frame's **zero-based `start_sample`**, device, day, and
output `row_id`. Its S4 `raw_row` refers to the historical packet registry,
not a row of the binary `.dat` file. Use `start_sample` to extract IQ.
The historical local layout is:

```text
<historical-workspace>/ieee_dataport/Datasets/Oregon_LoRa/
    Diff_Days_Indoor_Setup/Day*/Device*/IQ_*.dat
```

Download the IQ recordings, rather than the provider's FFT files, for this
pipeline. Retain the provider's metadata and use conditions. The directory and
release-note links were accessible during this update; no new recordings were
downloaded. Existing source digests identify the historical inputs and are
reproduction aids, not extra IEEE submission attachments.

## 3. S1 processing: IQ to the saved complex state

The executed procedure is preserved in `paper/rf_baseline/src/rebuild_raw_iq.py`
(`s1`). Its helper implementations are in
`paper/rf_baseline/provenance/raw_code_snapshot/src/czrf/`.

1. Read the selected HDF5 rows in the original manifest order. For each row,
   the `data` array stores 8,192 real I values followed by 8,192 real Q values.
   Form `iq = data[:8192] + 1j * data[8192:]`. Check the stored `label` against
   the selected device identity; this pipeline retains labels 31--40.
2. Split the 8,192 complex samples into eight consecutive segments of 1,024
   samples. Apply the orthonormal FFT to each segment and retain the 32 bins
   `0, 32, ..., 992`. Flatten initially in segment-major order and normalize
   the entire 256-dimensional complex vector to unit norm.
3. Preserve the historical precision step: convert to `complex64`, then use
   `physically_order_segment_major_rows`, which promotes to `complex128`,
   normalizes again, and permutes from segment-by-frequency to
   frequency-by-segment order. The final vector is the row-wise flattening of
   a **32-by-8** matrix.
4. Keep the recorded group identifiers. Encode the 13 environment names by
   their sorted-name indices, and retain output rows `0,...,5199` in the
   original order.

The saved `S1/SOURCE_INPUTS.npz` has `C.shape == (5200, 256)` and
`features.shape == (5200, 512)`. The complex state ordering and intermediate
precision conversions are part of the frozen implementation.

## 4. S4 processing: recording to the saved complex state

The executed procedure is `s4` in the same rebuild script. The detector is
preserved in `provenance/raw_code_snapshot/src/czrf/oregon_lora_adapter.py`.

1. Read each `.dat` file as interleaved little-endian Float32 I/Q, represented
   by NumPy dtype `<c8`. The sample rate is 1,000,000 Hz and the bandwidth
   used by the frontend is 125,000 Hz.
2. Detect eight-segment frame units on the fixed 1,024-sample symbol grid.
   The saved detector uses adjacent-symbol correlation threshold 0.99,
   minimum correlation-run length 6, minimum up-chirp dechirp score 0.04,
   minimum median up/down score ratio 5, minimum frame RMS `1e-6`, and minimum
   start separation 128 symbols. Detection does not use device labels.
3. Preserve the historical selection: among eligible registry rows with
   `role` 0 or 1, sort by the recorded packet-identity hash within each file,
   take the first 12, then restore registry order. Do not substitute the
   first 12 detections in time. The already selected starts are recorded
   explicitly in `RAW_ROW_LINEAGE.csv`; `DETECTOR_REPLAY.csv` records the
   historical check that all selected starts were reproduced.
4. Extract exactly 8,192 complex samples at each selected start, arranged as
   eight segments of 1,024 samples. Dechirp against the ideal up-chirp.
   For each segment, estimate the dominant tone with a length-4,096 FFT and
   remove that tone, apply a Hann window, and retain the central 16 bins of
   the shifted FFT. `segmented_harmonic_coefficient_matrix` returns a
   **16-by-8** frequency-by-segment matrix.
5. Flatten that matrix row-wise and normalize. Preserve the historical
   conversion of concatenated real and imaginary coordinates through
   `float32` and back to `float64`; reconstruct the complex vector and
   normalize again. The output state has 128 complex coordinates.
6. Keep device and day labels, the original `file:<file_id>` group identifiers,
   and output rows `0,...,479`. File IDs refer to the historical registry,
   not a newly sorted list of only the downloaded files.

The saved `S4/SOURCE_INPUTS.npz` has `C.shape == (480, 128)` and
`features.shape == (480, 256)`.

## 5. Features, folds, and the quadratic-task interface

Use the frozen unit measurement vectors stored in each `SOURCE_INPUTS.npz`;
do not generate a new measurement bank. With `n_sig = K*T`, the implementation
normalizes each state and computes

```text
kappa = sqrt(n_sig * (n_sig + 1) / M)
features[i, l] = kappa * abs(a_l.conj() @ psi_i)**2
```

S1 uses `n_sig=256, M=512`; S4 uses `n_sig=128, M=256`. There are 13 S1
leave-environment-out folds and three S4 leave-day-out folds. The exact
`train_ids` and `validation_ids` are saved in each fold's `COMMON_HEAD.npz`.

In the historical raw reconstruction, the standardizer is fitted on training
rows only (population standard deviation, `ddof=0`; an effectively zero scale
is replaced by 1). Training and validation groups are checked for overlap.
The standardized features are fitted with `RidgeClassifier(alpha=1.0,
class_weight=None)`. The standardizer is absorbed into the saved raw-feature
weights and intercepts:

```text
v_j = w_tilde_j / scale
beta_j = b_tilde_j - v_j @ mean
H_j = kappa * sum_l v_j[l] * outer(a_l, a_l.conj())
score_j(psi) = beta_j + psi.conj() @ H_j @ psi
```

Pairwise tasks use differences of class operators and intercepts. Consult the
main manuscript and supplement Section V-A for the full coordinate convention.
The saved classifiers remain fixed when checking the six observation flags:
`coordinate`, `dct`, `dft`, `polynomial`, `random_orthogonal`, and
`permuted_polynomial`. The interface check in Section 1 performs a numerical
consistency check over the 16 folds and six flags; it does not refit the heads.

## 6. Additional requirements of the unchanged raw-replay entry point

Downloading the recordings is sufficient to obtain the raw signals, but is
**not by itself sufficient to run the unchanged historical raw-replay script**.
That script also reads author-generated selection and qualification artifacts.
Their exact paths and existing digests are recorded in
`paper/rf_baseline/provenance/RAW_REBUILD_DEPENDENCIES_V2.json`:

```text
configs/track_c_v1/B1_ROW_MANIFEST_V1_20260827_154500.csv
runs/s4_oregon/S4_OREGON_FEATURE_CACHE_20260827_032417.h5
runs/track_c_v0/M2_SMOKE_HARMONIC_FEATURES_20260827_144025.npz
results/s4_oregon_qualification/S4_OREGON_SOURCE_INVENTORY_20260827_030353.csv
results/s4_oregon_qualification/S4_OREGON_SOURCE_QUALIFICATION_20260827_030353.json
```

These five artifacts are **not included in this companion** and are not files
provided by the original dataset authors. The bundled lineage CSVs document
the selected rows/starts, but do not replace all of the script's metadata reads.
For access questions about the historical selection artifacts, contact the
corresponding author at the address above.

The raw script additionally expects the historical module layout and the
reference directory
`paper/revision_track_s_decision_semantics_v13_20260905/inherited_v11/inherited_rf`
inside the workspace selected by `--workspace`. That complete historical
reference directory is not supplied at that path in this companion. It contains
the fixed measurement-bank design and regression targets. The saved old states
and task operators are comparison targets, not substitutes for reading raw IQ
and fitting the training folds. The code snapshot and dependency map are
preserved under `paper/rf_baseline/provenance/`.

Only after those prerequisites are available in a complete historical workspace,
the existing entry point is:

```text
python -B paper/rf_baseline/src/replay_v16.py --workspace "<complete-historical-workspace>" --destination "../raw_iq_replay"
```

Replace the workspace placeholder with its actual path and use a destination
that does not exist. Omitting `--derived` selects raw reconstruction and refitting.
In the complete companion, root-level `reproduce.py inherited` explicitly selects the bundled
derived-input route and does not download or reconstruct IQ.

The historical baseline recorded Python 3.12.6, NumPy 2.4.4, SciPy 1.17.1,
and scikit-learn 1.9.0, with one BLAS thread; the complete recorded requirements
are in `paper/rf_baseline/requirements-replay.txt`. Different numerical-library
versions may change results. The existing
`paper/rf_baseline/provenance/RAW_TO_RESULTS_REPLAY_RECEIPT.json` documents the
earlier successful local raw replay; it is not a claim that raw reconstruction
was rerun, or that this archive alone supports it, during this documentation update.

## 7. Data source attribution

Use the provider links above for recordings and their current access/use terms.
These processing instructions do not assign a new license to third-party data.
The full local companion is preserved separately; this smaller repository does
not redistribute its RF per-sample arrays.
