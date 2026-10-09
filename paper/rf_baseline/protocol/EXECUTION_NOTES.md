# Append-only execution notes

2026-09-07: The mathematical protocol, grids, sample indices and acceptance contract were fixed and contract-reviewed before any v16 experiment. During implementation, two canonical E2 function smoke checks were evaluated after contract acceptance but immediately before writing LOCK.json: (mixed, preserving, delta=1e-6, epsilon=0) and (quadratic, cross, delta=0.25, epsilon=0.1). This is a deviation from the intended digest-before-all-numerical-results ordering, not an externally registered experiment. No protocol, sample, grid, window or task was changed based on those checks; no RF experiment had run. Both cells remain in the complete fixed grid. The original local protocol text and its lock are preserved.

The S4 packet_sha256 field is a deterministic source/start identity digest, not a digest of packet bytes. Rebuild checks that original identity and additionally saves a new raw-byte digest; the paper will not confuse these roles.
