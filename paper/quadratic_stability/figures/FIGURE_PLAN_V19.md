# v19 figure contract

Figure: wang2.pdf, four panels at full text width.
Question: can equal-rank refinements of one task at one reference have
different stability and sampling powers?

Source: results/boundary_refinement_run01/{CONFIG,ENDPOINTS,MARGINS,SUMMARY}.json.
Every grid point is included. Three zero-loss cells remain in the ledger
but are omitted from logarithmic axes.

- (a) Exact downward loss versus tolerance; smaller is better.
- (b) First-failure radius versus margin; larger is better.
- (c) Necessary lower and sufficient upper repetition bounds.
- (d) Each bound multiplied by D.

Shading is a deterministic bracket for an unknown minimum, not a confidence
interval. sigma=1, eta=0.05, radius fraction=0.9. A has D=2; B and C have D=5.
Color and markers both identify designs; dashes identify necessary bounds.
No simulated data or fitted slopes.
