# Revision change log (Session 1, surgical revision, 2026-11-02)

Working file: `cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex`. Rollback point: commit d3f1be8 (identical to HEAD 4e03ec8 at the start). Final build after round 2 (below): 12 pages, no errors, no undefined references, last page balanced, abstract 250 words. `survival.py`: 69.1% (Chris lowered the round's target to 75% and the floor to 68% during the session). Revised body 9,512 tokens. Diff PDF: `cwaight_systems_paper/cwaight_systems_paper_diff.pdf` (15 pages).

Compiled numbering. Equations: (7) scope, (8) model-error bound, (9) error bound, (10) covariance, (11) attraction law, (12) orbital law, (13)-(15) robot model, (16) sampled closed loop, (17) gain bound, (18) ultimate bound, (19) orbit radial dynamics, (20) orbit closed form. Propositions 1 (minimality) and 2 (bounded tracking). Tables: I fields, II simulated attraction and classification, III simulated orbits, IV hardware attraction, V hardware orbits, VI matched baselines. Tables IV and V keep the numbers the reviewers cited. Figures (after round 2): 1 six fields, 2 architecture (a) and SAS formation (b), 3 simulated convergence, 4 nonlinear fields (a, b flows with trajectories; c, d size and curvature), 5 testbed, 6 reconstructed fields, 7 final error against control effort.

Part 1 and the survival table below record round 1. Round 2 edits follow at the end and supersede rows 44, 47 and the round-1 survival table.

Edit types follow the prompt: (a) in-place change, (b) clause or sentence added inside a paragraph, (c) new paragraph, equation, subsection, or table, (d) deletion, (e) paragraph replaced. SPACE marks an edit made only to meet the 12-page cap. ERR marks an outright error in the submitted text.

## Part 1. One row per edit

| # | Action | Type | Location | Edit, and the reason for each (d) or (e) |
|---|---|---|---|---|
| 1 | (prompt) | (c) | Preamble | xcolor and the `\syn`, `\rev`, `\open` markers. |
| 2 | AE.1-8, AE.1-3 | (c) | Preamble | `\newtheorem{proposition}`; both propositions use it and `IEEEproof` (IEEE form). |
| 3 | AE.1-2, AE.1-3, R1.7-1 | (b) | Abstract | One sentence: the estimation error is bounded, and the attraction error stays bounded under lag, sampling, friction. |
| 4 | AE.1-1, R1.6-5 | (a)(d) | Abstract | "and two nonlinear fields" added; "in both noisy and noise-free models" deleted (the reconstructed fields are fixed distortion, N4; also needed for the 250-word IEEE limit). |
| 5 | R1.6-2, R3.5-1 | (a) | Abstract | 100% / 157 / 0.021 m replaced by `\syn` 100% (saddle), 86% (vortex) within 0.05 m, 158 trials, mean error 0.037 m. |
| 6 | R1.4-1, R1.4-2, R3.5-1 | (a) | Abstract | "12 orbital trials ... six commanded radii ... less than 0.25 m error at small radii" replaced by `\open{36}` trials at `\open{six}` radii holding bounded orbits, radial bias within 0.25 r_d from `\syn` 0.40/0.50 m. |
| 7 | ERR | (a) | Abstract, VII | Source-comment placement fixed so the no-advection sentences render (a `%` had swallowed them mid-session). |
| 8 | R1.7-1 | (a) | I, para. 9 | "its eigenvalues fix the identity" to "indicate the type". |
| 9 | R3.1-1 | (b) | I, para. 9 | Two sentences: the construction is two affine fits and an inversion; the paper closes a loop on their zero, bounds the errors, and demonstrates it in hardware. |
| 10 | R2.1-2, AE.1-2 | (b) | I, contributions | First: bound and certification sentence appended. |
| 11 | R1.7-3, R3.1-2 | (a)(b) | I, contributions | Second: "non-collinear" and "they suffice for every nondegenerate affine field". |
| 12 | R2.1-2, AE.1-1, AE.1-3 | (b) | I, contributions | Third: "and two nonlinear fields" plus a sentence on bounded attraction error. |
| 13 | AE.1-5, R1.7-1 | (a) | I, contributions | Fourth: "169" to `\open{campaign}`. |
| 14 | R1.5-1, AE.1-4, AE.1-3 | (a) | I, roadmap | Section list updated (III bounds the closed-loop error; IV adds two nonlinear fields; VI compares with matched baselines). |
| 15 | SPACE | (c) | II-A, Table I | Fourth column with the canonical field of each type; replaces Appendix A. "λ =" dropped from cells to fit. |
| 16 | SPACE | (a) | II-A, Fig. 1 | New file `six_vector_fields_3x2.png` (the 2x3 grid rearranged 3x2; original untouched) at 0.9 column width. |
| 17 | AE.1-2, R1.1-2, R1.3-1, R1.3-2 | (c) | II-B, after (7) | Barycentric form and correlation; bound (8)-(9) in M, R, ε̄, κ0, σ_min; existence test; covariance (10). All submitted II-B text kept. |
| 18 | AE.1-8, R2.3-1, R3.1-2, R1.7-3 | (c) | II-C | Proposition 1 with proof. No submitted sentence deleted (none is false). |
| 19 | AE.1-2 | (a) | II-D, first sentence | p_c definition removed (now in II-B). |
| 20 | R2.4-1 | (b)(a) | II-D-1 | Closed-loop error dynamics and exponential solution; "τ = 1/k" to "1/k" (clash with the actuator τ). |
| 21 | AE.1-10, R2.1-1 | (a)(b) | III intro | 2D [5],[8],[10] and 3D [7],[9] citations; sentence that the structure follows [5],[8] and the cluster space controller [50]. |
| 22 | SPACE | (d) | III intro | Enumerated three-layer list deleted. Reason: III-A to III-C describe each layer (Chris authorized heavier trimming of III). |
| 23 | AE.1-10, R2.1-1 | (b) | III intro | "The architecture itself is not a contribution", and which blocks this paper fills. |
| 24 | SPACE | (d) | III intro, para. 2 | Three middle sentences of the per-cycle walkthrough deleted (repeat III-B and III-C). The cycle rate and the centralized-computation sentence kept. |
| 25 | SPACE, AE.1-10 | (a) | Fig. 2 | Architecture and SAS figures merged with (a)/(b) labels; caption adds "as in [5],[8]" and that the contents of the feature estimation and navigation controller blocks are new. |
| 26 | R1.2-1, ERR | (a)(b)(c) | III-A | Lag plus Coulomb model (13)-(15) with ν, [j]; v_max = 0.4 m/s replaces the wrong 0.3 m/s; v_c symbol on the 0.025 m/s value; parked-robot and friction-ring statements. |
| 27 | R1.2-1 | (a) | III-A, last para. | Equation range to (15); v_des to ν_des,i; pointer to VI-C (ripple of edit 66). |
| 28 | SPACE | (a) | III-B | "as visualized in Fig. 2(b)"; "The full equations are given in Appendix B" to "Their derivation follows [50]" (ripple of edit 70). |
| 29 | SPACE | (d)(a) | III-C | First two sentences condensed to "The adaptive navigation layer uses three functional blocks" (they restated the layer's role). |
| 30 | AE.1-3, R1.2-1..4, R3.3-2..4 | (c) | III-D (new) | Sampled closed loop (16), Jury bound (17), γ, friction as disturbance, Proposition 2 with (18), near-singular handling with `\open`, orbit bound (19), r = 0 guard. |
| 31 | R1.2-1 | (b) | IV intro | Robot model of III-A and the 0.5 m/s simulation command limit. |
| 32 | R1.6-5 | (d)(b) | IV intro | "incorporate realistic sensor noise" replaced by the fixed-distortion sentence (item 8a). Reason: the MLP fields carry no temporal noise. |
| 33 | AE.1-7, R1.6-1, R1.6-3, R3.5-3, R1.4-2, R1.7-4 | (c) | IV intro | Metric definitions shared by simulation and hardware, E_u for effort. |
| 34 | R1.6-1, R3.5-3 | (d)(a) | IV-A | "(mean error) and precision (standard deviation). A trial was counted as convergent ... neighborhood" to "the metrics defined above". Reason: bias is not mean error (R1.6, R3.5); threshold now numeric. |
| 35 | AE.1-7, R1.6-2, R1.6-3 | (e) | IV-A results | Rewritten. Reason: every number superseded by the Coulomb-model reruns; Chris authorized a heavier rewrite of IV. |
| 36 | AE.1-7, R1.6-2..4, R3.5-2, R1.7-4 | (c) | Table II | Revised Table II (six identical analytical rows merged, transposed) plus two classification rows. |
| 37 | AE.1-6, R1.4-5 | (b) | IV-B | 100 repeats per field and radius. |
| 38 | R1.4-5 | (c) | Table III | Revised Table III (RMSE columns omitted). |
| 39 | R1.4-1, R1.4-2, R3.2-1, SPACE | (a)(b)(d) | IV-B results | Numbers swapped; "slight" deleted; item 6 concession; ripple clause and three per-field sentences replaced by one trend sentence (SPACE); practical range by the 0.25 r_d criterion. |
| 40 | AE.1-1, R1.1-4, R3.2-2..4, R1.6-6, R1.3-3 | (c) | IV-C (new), Fig. 3 | Gyre cell and curved saddle; size cubed; curvature χ; basins and start distance; orbits; distortion against noise; covariance validated. |
| 41 | R1.7-4, R1.3-4 | (c) | IV-D (new) | Classifier band ε_c, accuracy by type, coarse classes, orbit agreement. |
| 42 | SPACE | (d) | IV-A | `figure_5.png` and its pointer sentence removed. |
| 43 | SPACE | (d) | V-A | `color_sensor_hsv_predictions.png` and "(see Fig.)" removed; RMSE values stay in the text. |
| 44 | SPACE | (d) | V-A | Testbed photo and "Fig. shows the complete testbed" removed. Not on the prompt's list; the photo showed four robots. |
| 45 | ERR | (a) | V-A | "maximum robot velocity of 0.3 m/s" to "a field speed of 0.3 m/s" (v_max is 0.4 m/s; N1-N3). |
| 46 | R1.6-5 | (a) | V-A | "realistic noise model" to "fixed distortion model" (item 8b). |
| 47 | SPACE | (a) | Fig. 4 | Shrunk to 0.28 text width. |
| 48 | AE.1-5 | (a) | V-B | `\open{20}` s and `\open{0.33 or 0.30}` m. |
| 49 | AE.1-5 | (d) | V-B | "Due to file management errors ... 157 usable convergence trials" deleted. Reason: describes the superseded campaign. |
| 50 | R1.6-1, R1.7-4 | (b) | V-B | Randomized order; metrics of IV and final speed reported (VI-B note); classifier logged, not fed back. |
| 51 | AE.1-5, R1.4-4, R3.5-4 | (a)(b) | V-B | `\open{six}` radii, `\open{three}` repeats, `\open{1800}` cycles. |
| 52 | ERR, SPACE | (d) | V-C | Paragraph on Figs. 9-10 and both figures deleted. Reason: four-robot vortex data under a three-robot caption. |
| 53 | R1.6-2, R1.6-3, R3.5-2, R1.7-4 | (a)(b) | V-C | Counts and numbers swapped (`\syn`); thresholds; mean error against bias; localization; hardware classification `\open{campaign}`. |
| 54 | R1.6-2..4, R3.5-2 | (c)(d) | Table IV | Transposed metric table (`\syn`); caption keeps its frame; rounding sentence deleted (no coordinates column). |
| 55 | R1.4-4 | (a)(d) | V-D | `\open{1800}` cycles; statistics after the first 10 s; "Unlike the simulation statistics ... initial transient" deleted (now false). |
| 56 | R1.4-1, R1.4-2, R1.4-4, R3.5-1, R3.5-4, AE.1-5 | (a)(b) | V-D | Numbers swapped (`\syn`); between-trial against within-run spread; practical range; bounded-orbit statement; failures. |
| 57 | SPACE | (d) | V-D | `orbit040_analytical_vs_real.png` and its pointer sentence removed (old mixed-gain runs). |
| 58 | R3.5-4, R1.4-4 | (c) | Table V | Synthetic three-robot table, six radii (0.05 and 0.15 m rows omitted). |
| 59 | R1.6-5 | (a) | VI-A, para. 1 | "sensor noise" to "field distortion and measurement noise". |
| 60 | R1.4-2, R1.4-3 | (d)(c) | VI-A, para. 3 | Point-mass replication sentence (legacy-model numbers) replaced by the closed form (20), r_min, gain and lag dependence. |
| 61 | R1.6-5, R1.6-6, R1.3-2 | (a)(b)(d) | VI-A, para. 4 | First sentence reworded in place to distortion; N1 and N3 sentences added; σ_p sentence tied to (10); "calibration errors ... 0.03 m" and "This matches ... cannot remove" deleted (N2: they treat fixed calibration error as noise). |
| 62 | R3.4-4, R1.6-6, R3.5-4 | (c) | VI-A | Hardware orbit metrics paragraph (V_VIA, merged and trimmed). |
| 63 | R1.3-2, AE.1-2 | (b) | VI-A, geometry | κ0 sentence appended. |
| 64 | R1.3-2, R1.3-4, ERR | (a)(b) | VI-A, conditioning | κ(Ĵ) to σ_min forms; negative-bias claim checked against revised Table III and corrected; "non-hyperbolic" to "degenerate"; run-time detectability; center sentence. |
| 65 | R1.2-2, R3.3-1 | (a)(b) | VI-A, last para. | Stability sentence replaced by the input δ of (16); "As a result" sentence cites (18), (19) and Proposition 2. |
| 66 | R1.5-1..2, R3.4-1..4, AE.1-4 | (a)(d)(c) | VI-B | Title changed; vector-sum and vector-to-scalar paragraphs deleted (vector-to-scalar orbit never simulated; drift shows only missing radial feedback); `orbit_demonstration.png` removed; matched-baseline text and Table VI. First paragraph kept. |
| 67 | R2.5-1, AE.1-11 | (b) | VI-C, Second | Four real-flow challenges appended; submitted sentences kept. |
| 68 | R2.2-1 | (b) | VI-C, Third | Larger-team sentence appended. |
| 69 | AE.1-1, R3.2-1 | (a) | VI-C, Fourth | Last clause replaced (IV-C now probes curvature); semicolon split. |
| 70 | R1.2-3 | (d) | VI-C, Finally | Sampling-rate limitation deleted. Reason: Proposition 2 covers sampled updates; dropouts moved to Second. |
| 71 | R1.7-1, R1.7-4 | (a) | VI-D, para. 1 | "not exercised" amended in place; confidence item. |
| 72 | SPACE | (d) | VI-D | Advection paragraph deleted (repeats VI-C Second and III-A). |
| 73 | AE.1-1, R1.1-4 | (b) | VII | One sentence on curved fields and basins. |
| 74 | R3.5-1, R1.4-1 | (a) | VII | 157 / 100% / 0.021 m and "0.25 m error at small radii" replaced (`\syn`), bounded-orbit wording. |
| 75 | SPACE | (d) | Appendix A | Removed; fields in Table I. |
| 76 | SPACE | (d) | Appendix B | Removed (step 7); III-B points to [50]. |

## Part 2. One row per tracker action

| Action | Answered where | Note |
|---|---|---|
| AE.1-1 | IV-C, Fig. 3, Abstract, I, VII | Gyre cell and curved saddle. |
| AE.1-2 | II-B (8)-(10), existence test | |
| AE.1-3 | III-D (16)-(19), Proposition 2 | |
| AE.1-4 | VI-B, Table VI | Oracle and magnitude-only, same laws. |
| AE.1-5 | V-B, V-D, Table V | **Placeholder data**; campaign not run. |
| AE.1-6 | IV-B, Table III | 100 repeats. |
| AE.1-7 | IV intro, Tables II-V | |
| AE.1-8 | II-C, Proposition 1 | |
| AE.1-9 | Abstract, I, III, V-A, VI-C Fifth | Kept from the approved baseline. |
| AE.1-10 | III intro, Fig. 2 caption | |
| AE.1-11 | VI-C Second | |
| R1.1-1 | II-B (7), VI-C Fourth | Approved baseline. |
| R1.1-2 | II-B (8)-(9), existence test | |
| R1.1-4 | IV-C | Gyre has five critical points in view; no Duffing field. |
| R1.2-1 | III-A (13)-(15), III-D, Proposition 2 | |
| R1.2-2 | VI-A last paragraph | |
| R1.2-3 | III-D (17)-(18) | |
| R1.2-4 | III-D end | r = 0 argument and guard. |
| R1.3-1 | II-B barycentric sentence | |
| R1.3-2 | II-B (10), VI-A | |
| R1.3-3 | IV-C last paragraph | Median ratio 1.004 over 66 settings. |
| R1.3-4 | VI-A conditioning paragraph, IV-D | Degenerate against non-hyperbolic. |
| R1.4-1 | V-D, Abstract, VII | |
| R1.4-2 | IV intro (criterion), IV-B, V-D, VI-A (20) | |
| R1.4-3 | VI-A (20) and r_min values | |
| R1.4-4 | V-B, V-D, Table V | **Placeholder data.** |
| R1.4-5 | IV-B, Table III | |
| R1.5-1 | VI-B | |
| R1.5-2 | VI-B, Table VI | |
| R1.5-3 | Letter only | The vector-to-scalar orbit was never run; the paper no longer describes it. |
| R1.6-1 | IV intro | |
| R1.6-2 | Tables II and IV | |
| R1.6-3 | IV intro, Tables II and IV | |
| R1.6-4 | Tables II and IV (median, p95, mean ± sd, three thresholds) | The hardware CDF figure was not used (space). |
| R1.6-5 | IV intro, V-A, VI-A | |
| R1.6-6 | IV-C last paragraph, VI-A | |
| R1.7-1 | I, Abstract, VII, VI-D | |
| R1.7-2 | Abstract, I, III, VI-C Fifth | Approved baseline. |
| R1.7-3 | Proposition 1, contribution two | |
| R1.7-4 | IV-D and Table II (simulation); V-B, V-C `\open{campaign}` (hardware) | |
| R1.7-5 | Abstract, III-A, VI-C, VII | |
| R2.1-1 | III intro, Fig. 2 | [50] prints as [42]. |
| R2.1-2 | I contributions | |
| R2.2-1 | VI-C Third | |
| R2.3-1 | Proposition 1 | |
| R2.4-1 | II-D-1 | |
| R2.5-1 | VI-C Second | |
| R2.5-2 | Abstract, VII, VI-C | |
| R3.1-1 | I (novelty sentences) | |
| R3.1-2 | Proposition 1 | |
| R3.1-3 | Abstract, I, III, V-A | Approved baseline. |
| R3.2-1 | IV-B | |
| R3.2-2 | IV-C, Fig. 3 | |
| R3.2-3 | IV-C basins by start distance | |
| R3.2-4 | IV-C, Fig. 3(a) | |
| R3.3-1 | VI-A last paragraph | |
| R3.3-2 | III-D (δ ≤ ηe + δ̄) | |
| R3.3-3 | III-D (16)-(18) | |
| R3.3-4 | III-D near-singular paragraph | No σ_min gate; invalid-estimate behavior `\open`. |
| R3.4-1 | VI-B | |
| R3.4-2 | VI-B, Table VI | Magnitude-only estimate with the same orbital law, not contour following. |
| R3.4-3 | Table VI | 500 attraction and 50 orbit trials per setting. |
| R3.4-4 | VI-B, IV-C, VI-A | Simulation oracle; hardware oracle not run. |
| R3.5-1 | Abstract, V-D, VII | |
| R3.5-2 | Tables II and IV | |
| R3.5-3 | IV intro | |
| R3.5-4 | V-D, Table V | **Placeholder data.** |

All 67 actions have a location. Four rest on the campaign and use placeholder data or open settings: AE.1-5, R1.4-4, R3.5-4, and the hardware half of R1.7-4 (plus the hardware rows of R1.6-2/3/4 and R3.5-2). R1.5-3 is answered only in the letter.

## Donor blocks reused from `Queued_Edits/rewrite_v1/`

Each was checked against its source and trimmed. II-B bound block (from the error note, with (9) written as φ(‖e‖)/σ_J). Proposition 1 and proof (sufficiency shortened, points to the existing paragraph). Robot model equations (13)-(15). III-D (trimmed, IEEE theorem and proof environments, equation numbers no longer sentence subjects). Metric definitions. Nonlinear and classification subsections (trimmed; classification table folded into Table II). VI-B text and Table VI (oracle rows merged). Table I canonical-field column. Revised tables with markers (restructured). Fig. 2 caption marking. The v1 ledger as a number checklist. Not taken: the v1 roadmap deletion, abstract, contributions, conclusion, VI-A rewrite, condensed architecture prose, replaced II-C text, and hardware rows merged into Table II.

## Per-section survival (final)

| Submitted section | Words | Kept |
|---|---|---|
| Front matter (abstract) | 317 | 77.0% |
| I Introduction | 1018 | 97.7% |
| II Critical Point Detection and Control | 976 | 88.3% |
| III Multilayer Control Architecture | 761 | 73.6% |
| IV Control Performance in Simulation | 635 | 64.1% |
| V Hardware Validation | 1141 | 67.7% |
| VI Discussion | 1354 | 60.9% |
| VII Conclusion | 98 | 76.5% |
| Appendix A (vector fields) | 142 | 0% |
| Appendix B (cluster equations) | 415 | 0% |
| Disclosure | 102 | 100% |
| **Total** | 6960 | **69.6%** |

## Round 2 (Chris's request: restore the testbed photo, add flow, convergence and effort figures, cut prose the figures and tables carry)

New figure files in `cwaight_systems_paper/`, drawn only from saved data (no new simulation): `fig_nonlinear_flows.pdf`, `fig_sim_convergence.pdf`, `fig_sim_effort.pdf`. Script: `make_revision_figures.py` in this session's scratchpad (not in the repo; it reads `hyperparameter_study/results_v3/cells` E1hw cells and `revision_metrics/results/baselines_attraction.csv`, Coulomb model after the 2026-10-28 sign fix). Palette checked with the dataviz validator; every series also carries a marker shape for print.

| # | Action | Type | Location | Edit, and the reason for each (d) |
|---|---|---|---|---|
| R2-1 | AE.1-1, R1.1-4, R3.2-3 | (c) | Fig. 4(a)(b) | Gyre cell and curved-saddle streamlines with six saved plane-fit attraction trajectories; the gyre shows starts outside det J = 0 reaching corner saddles. Old (a)/(b) relabeled (c)/(d); text refs updated. |
| R2-2 | AE.1-7, R1.6-4 | (c) | Fig. 3, IV-A | Distance to the critical point against time, six saved trials each on the analytical vortex and both reconstructed fields; friction ring marked. |
| R2-3 | AE.1-4, R1.5-2, R3.4-3 | (c) | Fig. 7, VI-B | Final error against control effort per trial for plane fit, oracle, magnitude (analytical fields and reconstructed vortex). |
| R2-4 | (Chris) | (c) | V-A, Fig. 5 | Testbed photo and its pointer sentence restored (0.15 text width). Reverses round-1 row 44. |
| R2-5 | SPACE | (d) | IV-A results | Bias and success numbers (in Table II) cut; takeaways kept. |
| R2-6 | SPACE | (d) | IV-B results | Radius numbers (in Table III) cut from the submitted sentence frame. |
| R2-7 | SPACE | (d) | IV-C | Size and curvature numbers reduced to takeaways; basin text points to Fig. 4(a); noise range stated as a maximum change. |
| R2-8 | SPACE | (d) | IV-D | Per-type accuracies reduced to the boundary-type takeaway; orbit-agreement sentence cut. |
| R2-9 | SPACE | (d) | V-C, V-D | Mean-error, localization and between-trial numbers (in Tables IV, V) cut; one explanatory clause cut in V-D. |
| R2-10 | SPACE | (d) | VI-A | Localization numbers cut; κ0 sentence cut; submitted "modeling discrepancies" sentence deleted (the identified robot model of III-A supersedes it). |
| R2-11 | SPACE | (d) | VI-B | Submitted modular-architecture paragraph deleted (repeats the matched setup); estimate-error numbers left to Table VI; magnitude paragraph shortened. |
| R2-12 | SPACE | (d) | III-A, III-D | Parked-robot sentence (III-D restates it) and a duplicate command-limit sentence cut. |
| R2-13 | SPACE | (a) | Figs. 1, 4, 6, captions | Small width reductions; two caption clauses that repeat the text cut. |
| R2-14 | style | (a) | IV-C, IV-D | Three sentences over 40 words split. |

### Per-section survival after round 2

| Submitted section | Words | Kept |
|---|---|---|
| Front matter (abstract) | 317 | 77.0% |
| I Introduction | 1018 | 97.7% |
| II Critical Point Detection and Control | 976 | 88.3% |
| III Multilayer Control Architecture | 761 | 73.6% |
| IV Control Performance in Simulation | 635 | 62.0% |
| V Hardware Validation | 1141 | 71.3% |
| VI Discussion | 1354 | 56.5% |
| VII Conclusion | 98 | 76.5% |
| Appendix A | 142 | 0% |
| Appendix B | 415 | 0% |
| Disclosure | 102 | 100% |
| **Total** | 6960 | **69.1%** |

## Round 3 (Chris's review of round 2, 2026-11-02)

Requests: drop Fig. 4(c)(d) and Fig. 2(b); redraw the reconstruction figure as 3 columns by 2 rows; bring back the submitted hardware trajectory and distance-over-time figures side by side, reusing the old images as placeholders; make III-D a little shorter and easier to read; stay at 12 pages. Chris then approved a short simulation run to replace the effort scatter (Fig. 8 after renumbering) with speed against time (row R3-10).

New files in `cwaight_systems_paper/`: `fig_reconstruction_3x2.png` (the submitted figure's robot readings, same loaders as `VF_Robot/experiments/raw_point_field_comparison.py`, grid subsample of 17 by 17 cells with seed 0), `placeholder_hw_trajectories_a.png`, `placeholder_hw_trajectories_b.png` (the two panels of `allrunsadjusted.png`), and `placeholder_hw_distance.png` (`fig9.png` with its title cropped). Script: `make_revision_figures.py` in this session's scratchpad.

| # | Action | Type | Location | Edit, and the reason for each (d) |
|---|---|---|---|---|
| R3-1 | (Chris) | (d) | Fig. 4, IV-C | Panels (c) and (d) removed with their caption text. The size and curvature takeaways stay in the text without figure pointers. |
| R3-2 | (Chris) | (d) | Fig. 2, III-B | SAS panel (b) removed. The SAS definition now cites [42] (`\cite{50}`). Fig. 2 is the architecture alone at 0.66 column width. |
| R3-3 | (Chris) | (a) | Fig. 6, V-A | Reconstruction figure redrawn as 2 rows (vortex, saddle) by 3 columns (analytical, reconstructed, error) at 0.86 column width. "(bottom row)" became "(right column)". |
| R3-4 | (Chris), R1.6-4 | (c) | Fig. 7, V-C | Submitted Figs. 8 and 9 restored as one float: (a)(b) trajectories side by side, (c) distance over time. Caption starts with `\syn{[Placeholder]}`, and the source comment says that the vortex panel and vortex curve are four-robot data and that (c) is in map units. The submitted sentences describing both figures are restored, merged into one body sentence plus the caption. |
| R3-5 | (Chris) | (a) | III-D | Run-in headings (Sampled Loop, Friction and Estimate Error, Near-Singular Estimates, Orbits). Jury polynomial dropped (it follows from (16)), γ explained as the worst-case gain, friction sentence merged, a one-line reading of (18) added (affine, error-free case reduces to the friction ring), "tangential term grows" sentence cut. About 3% shorter (597 against 614 rendered words). The r = 0 passage the letter quotes for R1.2(e) is unchanged. |
| R3-6 | SPACE | (d) | II-B | Sentence restating how curvature and conditioning enter (the bound (9) and the following sentence carry it) and the "whichever computer" sentence cut. |
| R3-7 | SPACE | (d) | III-A | Field-does-not-advect paragraph shortened to its submitted sentences. The forward pointer to VI-C was dropped. |
| R3-8 | SPACE | (d) | VI-A | σ_p restatement of (10) cut. |
| R3-9 | SPACE | (a) | Figs. 1, 3, 4, 5, 8 | Width reductions (Fig. 1 0.82 to 0.68 column width; the others by 0.02 to 0.08). |
| R3-10 | AE.1-4, R1.5-2, R3.4-3 (Chris approved the run) | (c) | Fig. 8, VI-B | Final-error-against-effort scatter replaced by centroid speed against time: trials 0 to 2 of the Table VI attraction batches on the analytical and reconstructed vortex, for oracle, plane fit and magnitude. The batches were rerun with per-step recording (`run_speed_traces.py` in the scratchpad: 500 trials, seed 0, Coulomb model, default gains). Every trial's effort and final error match `baselines_attraction.csv` to 1e-16. Traces saved to `revision_metrics/results/baselines_speed_traces.npz`. Caption numbers in `\rev`. The VI-B "every metric" pointer now cites Table VI as well. `fig_sim_effort.pdf` is left in the folder, unused. |

Letter (`Response_to_Reviewers.tex`) updated to match (after R3-10, its four descriptions of Fig. 8 as a per-trial final-error plot were rewritten, and the not-adopted item now reads "Per-trial error distributions", answered by statistics with no distribution plot): effort figure renumbered Fig. 7 to Fig. 8 (eight places); Fig. 4(c)/(d) pointers removed from AE (b) and 3.2 (d, e, f); the four-robot paragraph now says `\syn{Fig. 7 redraws those two figures from the new campaign}` instead of "those two figures were removed"; 1.6 (d), 3.5 (d, e), and the not-adopted hardware item now mention Fig. 7 in `\syn`. All 20 quoted passages still match the paper.

Build: 12 pages, no errors, no overfull boxes, no undefined references. Letter: 11 pages, clean. Survival 69.2% (68% floor).

### Per-section survival after round 3

| Submitted section | Words | Kept |
|---|---|---|
| Front matter (abstract) | 317 | 77.0% |
| I Introduction | 1018 | 97.7% |
| II Critical Point Detection and Control | 976 | 87.3% |
| III Multilayer Control Architecture | 761 | 70.8% |
| IV Control Performance in Simulation | 635 | 62.0% |
| V Hardware Validation | 1141 | 74.2% |
| VI Discussion | 1354 | 56.5% |
| VII Conclusion | 98 | 76.5% |
| Appendix A | 142 | 0% |
| Appendix B | 415 | 0% |
| Disclosure | 102 | 100% |
| **Total** | 6960 | **69.2%** |
