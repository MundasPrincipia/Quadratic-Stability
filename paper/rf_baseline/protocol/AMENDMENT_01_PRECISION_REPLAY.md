# Implementation correction before new RF analyses

2026-09-07. The first rebuild run stopped at the fixed S4 feature replay gate (maximum error 1.8724533131857752e-8; gate 1e-10). All S1 feature/head comparisons had passed, and all 40 S4 raw file hashes, selected detector starts and packet identity digests had passed. No E1/E2/E3 RF result was produced from that run.

Cause: the historical M2 writer (`scripts/run_track_c_m2_smoke.py`, line 934) casts the concatenated normalized real/imaginary harmonic coordinates to float32 before saving. The first new implementation retained float64 at this boundary. The first-row diagnostic reproduced the original stored coordinates exactly after that cast (maximum error 0). The corrected v16 adapter restores float32 -> float64 -> complex renormalization, as required by the original preserve-dtype protocol. No tolerance, scientific task, selected row, model parameter or experiment grid changes.

The failed `data/raw_rebuild_run01` remains intact and is excluded from downstream experiments. Corrected outputs go to `data/raw_rebuild_run02`. A successful complete baseline replay remains mandatory. This is a documented implementation correction, not a new favorable-result selection.
