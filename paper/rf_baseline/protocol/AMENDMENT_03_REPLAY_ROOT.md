# Amendment 03: isolated replay root

Date: 2026-09-07. The common I/O module accepts CZOPERATOR_WORKSPACE so
an isolated replay can read the same raw IQ, fixed metadata and historical
regression targets without writing into any old directory. With the variable
absent, the original workspace resolution is unchanged. No mathematical
operation, input selection, dtype, model, grid or result threshold changes.

The replay entry creates a fresh destination and refuses to reuse one.
It runs the raw reconstruction, regression tests and E1/E2/E3 afresh, then
compares all scientific arrays/tables with the authoritative v16 runs.
Timestamps and path-dependent provenance digests are compared separately
from scientific outputs; equality of arrays is not called byte identity.
