# v16 local pre-analysis protocol

Date: 2026-09-07. This is a locally frozen analysis protocol, not an external preregistration.
The user approved raw-IQ reconstruction and separate natural/controlled evidence. Old results and all freezes remain read-only.

## Claims, success and failure
C1: evaluate the exact budget's finite task response and sensitivity to lower-order leakage (E1/E2).
C2: evaluate task-directed information revelation and connect certified radii to matched Gaussian repetition bounds (E3/theory regressions).
Success means valid, traceable bounds and a completed prespecified comparison, not a forced positive RF gain. No accuracy-oriented model search, extra dataset, raw-noise Gaussian claim, or hardware interpretation is introduced.

## Inputs and baseline
S1: exactly B1_ROW_MANIFEST_V1_20260827_154500.csv S1 rows, devices 31--40, 13 environments, 5200 rows. Read published complex IQ from the specified HDF5 rows; eight 1024-sample segments, orthonormal FFT, bins 0,32,...,992; globally normalize, preserve the historical complex64 cache cast, transpose 8x32 to 32x8, then the inherited unit-row boundary.
S4: exactly the 480 selected role-0/1 rows in the inherited harmonic registry, 40 files and days 1/3/5. Re-run the frozen label-blind grid detector on those files and confirm retained starts/packet hashes. Read 8x1024 samples, frozen dechirp/dominant-tone removal/Hann/FFT recipe, retain 16 centered bins, global normalization.
Retain bank seed 20260827, prefixes 512 (S1)/256 (S4), ridge alpha=1, class_weight=None, group-guarded training-only standardizer, 13 leave-environment and 3 leave-day folds. Refit once per fold from reconstructed features. Raw source archives are not redistributed.
Metadata/row identity must match exactly. Rebuilt normalized C must have max absolute difference <=1e-10 from the corresponding saved interface when using the same dtype boundaries; full-score maximum absolute replay error <=1e-8. Recomputed zero-tolerance acceptance uses the inherited envelope and strict >1e-12 criterion, with mismatches individually diagnosed before E1/E2/E3. These tolerances are numerical replay checks, not physical noise tolerances.

## Fixed grids and selection
Radii: 10^(-6+j/2), j=0,...,10, and 0.25 (12 values).
Epsilon: 0 and 10^j, j=-8,...,-1 (9 values).
ANCHORS.csv chooses one validation row per source/fold/device by the smallest SHA256 of v16|source|fold|device|row_id: 160 anchors, selected from metadata before outcomes. Invalid anchors are retained with reasons and never replaced.
All natural E1 rows remain; no label-based filtering. Random-design seeds: 17,29,43.

## E1
q=7,d=1; all six inherited flags. For every normalized row, s=||Y||_F^2, x=u/||u||, tau=sqrt(2)*s. Zero u is marked inapplicable. Save applicability tau<=delta/2 and tau+delta<=1/4, tau/delta, all margins and finite brackets where the boundary theorem applies. Outer centers use delta+tau; inner witnesses use delta-tau only if nonnegative, never clamp to zero and invent an inclusion. Raw state/exact-phase witnesses remain valid independent feasible comparisons. Report distributions and fractions by source, fold, flag, device; denominator includes every original row. Report no applicable rows when that is the outcome.

## E2
q=7,d=1; coordinate flag; all 160 anchors. Retain the unmodified learned pair head as a separate diagnostic. Candidate and competitor are the first and second anchor scores (stable class order); labels do not choose tasks. Construct orthonormal x,w,v with x the normalized resolved direction, w a resolved tangent and v a residual direction selected from the pair head (leading singular directions of J when nonzero, deterministic coordinate fallback otherwise).
Canonical downward tasks in basis (x,w,v): H_3/2=E23+E32; H_2=-E22. Perturbations: preserving E=H_p; cross E13+E31; tangential linear E12+E21; residual linear -E33; plus mixed E23+E32 for p=2. All are added with nonnegative epsilon. Radius/epsilon grids are those above.
Set s=||Y||_F^2, t=||w||^2 and gamma^2=1-s-t. Use the exact d=1 budget 2s^2+2t<=delta^2. An optimizer of each downward profile exists with t=delta^2/2-s^2, reducing to s in [0,delta/sqrt(2)]; prove the reduction for epsilon<=0.1 and delta<=0.25 before claiming a full-set bracket. Preserve cases use exact inherited profiles. Compute inward feasible witnesses and outward upper bounds, target relative bracket width <=1e-5 (or explicitly record unresolved bounds). Save canonical profiles once and raw embedded matrix validation for all anchors; those embeddings are not independent replications of the identical canonical law.
For H_p+epsilon*E, E denotes the unscaled perturbation direction and B_(epsilon E)=epsilon*B_E. The canonical 25% sufficient-window predicate is e_0(delta)+epsilon*B_E(delta)<=c*delta^p/4, using the proved original endpoint remainder e_0. Save false/true window status for every cell, with no fitted-window selection. Floating embedding defects add a separate B_defect(delta) on the left for embedded-matrix checks. Slopes, where displayed, are descriptive and not proof.

## E3
Coordinate prefix q=5,d=3, k=0,1,2. At each fixed anchor x choose the strict top anchor class a and runner-up b from rebuilt head scores (stable ordering); ties are retained as no-strict-base cases. Target-pair design: top-k right singular directions of conjugate reshaped gradient for H_a-H_b. Raw-score design: top-k right singular directions of vertically stacked conjugate gradients for the ten original heads. Random design: Haar complex QR subspaces for each fixed seed. At k=0 all methods share the same baseline.
Evaluate alpha for the target pair and all a-versus-other differences, using feasible dual-gauge candidates and valid support-residual upper bounds. No successful-only subset summary. A safe analytic alpha upper fallback may replace a failed coefficient solve, with status retained. M_H and base margins are fixed across designs.
The lower certified radius is the solution of m-alpha_upper*sqrt(delta)-5*M_H*delta>0 capped at 1/4, using safe outward envelopes. Multiclass takes the minimum over all a-versus-other heads. For every design, anchor, k and competitor, save each explicitly normalized full-state failure witness, its actual redesigned-observation distance and its nonpositive difference score. Target-pair upper bounds use competitor b; multiclass upper bounds are the minimum across all witnessed competitors. No witness by the cap is separately censored for pair and multiclass, not an infinite true radius. Compare all methods paired at the same anchor and k. Leading-coefficient optimality is not finite-radius optimality; a bound improvement is not automatically an exact-radius improvement.

## Theoretical regression and reporting
Test Schur negative/PSD/zero-tangent cases, reject singular Kr use, check perturbation branches and Gaussian two-point normal-CDF bound. Fixed H/intercept-only margin families establish the four repetition exponents; no large Monte Carlo or estimated physical RF sigma.
Summaries: retain source and fold structure, summarize paired device effects; any resampling is by device with recording nesting, never by flag-row. Separate head invariance, correct device labels, and conditional-on-acceptance metrics.
Run order: proof closure -> lock contract/protocol/anchor digest -> raw reconstruction and numerical anchors -> E1 -> E2 -> E3 -> figures/prose -> fresh audits and clean replay. CPU-only fixed frontend and ridge fitting, no neural model training or parameter search. Large intermediate IQ is streamed per source file. Source/algorithm changes after lock require an append-only amendment and a distinct run directory; never overwrite results.
