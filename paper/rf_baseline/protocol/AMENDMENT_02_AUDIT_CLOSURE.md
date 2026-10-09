# Amendment 02: regression coverage and interval serialization

Date: 2026-09-07. Post-outcome numerical-audit repair, not a new analysis selection.

The independent experiment audit identified that the original theory regression
file recorded two edge cases as prose rather than executed checks and omitted
the four fixed-family repetition-rate regressions. A separate expanded regression
runner and result file will exercise these obligations. The old file is retained;
its PASS applies only to its executed cases, not the entire protocol checklist.

E2 run01 has authoritative high-precision decimal intervals, but its convenient
binary64 lower/upper columns were rounded to nearest. Run02 preserves the same
solver, grids, anchors, windows and formulas, writes outward-rounded binary64 and
decimal endpoints, and also records their exact binary rational representations.
Run01 remains available; only run02 will be used for the final E2 presentation.

Per-run provenance manifests are issued during closure. They distinguish a
post-run source snapshot from a pre-run lock and do not claim retrospective
preregistration. Failed and superseded runs remain listed. No source, data,
sample, flag, epsilon or tolerance is selected or removed using the results.
