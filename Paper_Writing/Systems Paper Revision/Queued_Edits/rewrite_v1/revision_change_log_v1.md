# Revision change log (Session 1, 2026-11-02)

Applied to `cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex` from `REVISION_PROMPT_1_paper.md`. Rollback point: commit d3f1be8. The paper builds to 12 pages with no errors or undefined references.

Numbers below are the compiled numbers from the `.aux`. Equations: (6) estimate, (7) scope, (8) model-error bound, (9) error bound, (10) covariance, (11) attraction law, (12) orbital law, (13)-(15) robot model, (16) sampled closed loop, (17) gain bound, (18) ultimate bound, (19) orbit radial dynamics, (20) orbit closed form. Tables: I fields and types, II attraction (simulation and hardware), III simulated orbits, IV classification, V hardware orbits, VI matched baselines. Figures: 1 six fields, 2 architecture and SAS, 3 nonlinear fields, 4 testbed, 5 reconstructed fields, 6 hardware per-trial error CDF.

## Section map

| Section | Change | Draft used |
|---|---|---|
| Abstract | Rewritten to the new evidence. Hardware numbers `\syn`. No 157, 100% (as a claim over all trials), 0.021 m, 12 orbital trials or "less than 0.25 m at small radii". | prompt Stage 2.8 |
| I | Contributions replaced (batch 1). Novelty sentences appended (batch 2). "fix the identity" became "indicate the type" (batch 3). Roadmap paragraph deleted (style guide 6.4). Distributed/centralized wording kept as it was. | prose batch 1-3 |
| II-A | Table I gains the canonical field of each type, which replaces Appendix A. | Stage 1.3 |
| II-B | Everything from "This relation is exact." to the end of that paragraph replaced by the error note: barycentric form, (8), (9), existence test, (10). The Laplacian sentence was left out. | error note, adopted |
| II-C | Replaced by Proposition 1 with proof, with **w** renamed **m**1. | minimality note, adopted |
| II-D | p_c is now defined in II-B. The closed loop is written as e-hat-dot = -k e-hat, with e-hat(t) = e-hat(0)e^{-kt} for a fixed estimate (batch 9). (J - kI)e appears only in Limitation Two. | prose batch 9 |
| III | Replaced with the one-pager's architecture text: reuse cited [5], [8], [10], [7], [9], [50], plus "The architecture itself is not a contribution". Figs. 2 and 3 merged side by side, and the caption marks the new blocks. | one-pager, adopted |
| III-A | Lag plus Coulomb robot model, (13)-(15), with nu for robot velocity and [j] for the index. Parameters are in the text and the parameter table was dropped to save space. The hardware-validation paragraph (4R vortex) is excluded. | robot model note, refined |
| III-B | Appendix B condensed into four sentences that cite [50]. No heading formula is given, which avoids the Python/Simulink convention difference. | Stage 1.2 |
| III-C | One-pager text, with the laws referenced by number. | one-pager |
| III-D (new) | Sampled closed loop (16), Jury bound (17), gamma, friction as a disturbance, Proposition 2 with the error-note assumptions, near-singular handling, orbit bound (19), and the r = 0 guard. v_min became v_c, and the per-axis wording became "the command limit". | stability note plus error note items 2a-2d, refined |
| IV | Rewritten. IV-A metric definitions, used by simulation and hardware alike. IV-B revised Tables II and III (Coulomb model, after the sign fix), the batch 6 concession, and the orbit closed form (20) with r_min and gain dependence. IV-C nonlinear fields: gyre cell, curved saddle, size cubed, start distance, basins, orbits, and distortion against noise. IV-D classification with band epsilon_c and Table IV. | revision_metrics tables, hyperparameter study v3, error note item 7 |
| V-A | Batch 8b ("fixed distortion model"). "Maximum robot velocity of 0.3 m/s" became "a field speed of 0.3 m/s", because v_max is now 0.4 m/s. The duplicate 0.3 m/s command-limit sentence was removed (it stays in III-A). The HSV-prediction figure was cut because its RMSE values are in the text. | prose batch 8b |
| V-B | Test plan as the campaign will read, with open settings `\open`. Points to IV-A for metrics. Hardware classifier logged without feeding the controller. | V_VIA draft, 3R only |
| V-C | Attraction results (old 3R logs, `\syn`), hardware rows in Table II, and the 3R-only per-trial CDF (Fig. 6). Hardware classification `\open{campaign}`. Old Figs. 9 and 10 (4R vortex) and the 157-trial text removed. | V_VIA draft |
| V-D | Orbit results with bias, between-trial and within-run spread, practical range and failures, plus Table V (3R columns only, `\syn`). The old 0.40 m orbit comparison figure and the synthetic orbit figure were cut. | V_VIA draft |
| VI-A | Rewritten. Four-source opening, dynamics paragraph (gain trade), N3 distortion paragraph with (10) for time-varying noise, the two V_VIA paragraphs, error note items 4, 5, 7 and 8 (sigma_min, degenerate, centers), and the stability note's last paragraph. | error note, N3, V_VIA, stability note |
| VI-B | Replaced by the matched-baselines section and Table VI. The vector-sum comparison and Fig. `orbit_demonstration.png` were deleted. The PENDING hardware-oracle sentence was left out. | vib_baselines note |
| VI-C | Limitation Two now covers the four real-flow challenges (batch 5). Limitation Three covers larger teams (batch 4). Limitation Four now points to IV-C. The sampling-rate limitation was deleted, since Proposition 2 covers it, and "Fifth" became "Finally". | prose batch 4, 5 |
| VI-D | Rewritten as one paragraph. The classification paragraph became a classifier-confidence item, the advection item points to VI-C, and the time-varying and separatrix items were kept. | Stage 1.5 |
| VII | Rewritten to the new evidence. Hardware numbers `\syn`. | Stage 2.8 |
| Appendices | Removed (A into Table I, B into III-B). | Stage 1.2-1.3 |

## Tracker actions (67)

| ID | What changed | Where |
|---|---|---|
| AE.1-1 | Nonlinear benchmarks on the gyre cell and curved saddle: size, curvature, start distance, basins, orbits | IV-C, Fig. 3 |
| AE.1-2 | Error bound in M, R, epsilon-bar, kappa_0, sigma_min(J), with invertibility through Weyl | II-B, (8)-(9) |
| AE.1-3 | Sampled loop with lag, changing estimate, friction and saturation; Proposition 2; near-singular handling | III-D, (16)-(18) |
| AE.1-4 | Matched baselines (oracle, magnitude-only) under the same laws, gains, robot model and formation | VI-B, Table VI |
| AE.1-5 | Repeated hardware orbits, written as the campaign will report them. **Placeholder data**, not yet run | V-B, V-D, Table V |
| AE.1-6 | Repeated simulated orbits, 100 per field and radius | IV-B, Table III |
| AE.1-7 | Metrics defined numerically, one definition for simulation and hardware | IV-A, Tables II, III, V |
| AE.1-8 | Three-robot minimum as a proposition | II-C, Proposition 1 |
| AE.1-9 | Centralized computation, spatially distributed sensing (already in the file, kept) | Abstract, I, III, Limitation Five, VII |
| AE.1-10 | Reused layers cited and new blocks marked | III, Fig. 2 caption |
| AE.1-11 | Static-lab limitations expanded to four real-flow challenges | VI-C Limitation Two |
| R1.1-1 | Hats and the scope equation (already in the file, kept) | II-B (7), VI-C Limitation Four |
| R1.1-2 | Error bound and existence test | II-B (8)-(9) and the paragraph after |
| R1.1-4 | Nonlinear fields with more than one critical point (gyre cell). No Duffing field | IV-C |
| R1.2-1 | Lag, saturation, friction and sampling as explicit assumptions | III-A (13)-(15), III-D, Proposition 2 |
| R1.2-2 | VI-A stability sentence replaced with a pointer to Proposition 2 | VI-A last paragraph |
| R1.2-3 | Ultimate bound and the gain condition on k Delta t | III-D (17)-(18) |
| R1.2-4 | r = 0 is not reached by the ideal dynamics; guard n = [1, 0]^T below 1e-6 m | III-D end |
| R1.3-1 | Correlated J-hat and h-hat errors act through one barycentric combination | II-B first paragraph |
| R1.3-2 | First-order covariance | II-B (10) |
| R1.3-3 | Covariance validated against simulation (median ratio 1.004 over 66 settings) | IV-C last paragraph |
| R1.3-4 | Singular against non-hyperbolic; centers keep sigma_min > 0 | VI-A conditioning and centers paragraphs |
| R1.4-1 | Small radii called bounded orbits, not tracking | V-D, Abstract, VII |
| R1.4-2 | Practical range defined as abs(bias) <= 0.25 r_d; r_min = sqrt(C) | IV-A, IV-B (20), V-D |
| R1.4-3 | Dependence on lag and gains (r_min for k_t, k_r, lag off) | IV-B (20) and the text after |
| R1.4-4 | Repeated hardware orbits with the same 10 s exclusion. **Placeholder data** | V-B, V-D, Table V |
| R1.4-5 | Repeated simulated orbits with the same 10 s exclusion | IV-B, Table III |
| R1.5-1 | Superiority wording on drift removed; the comparison is now matched baselines | VI-B |
| R1.5-2 | Matched attraction and orbit comparison with convergence time, localization error, radial error and effort | VI-B, Table VI |
| R1.5-3 | Modified vector-to-scalar law. **Not in the paper.** The comparison was removed, and the letter explains that the matched baselines replace it | letter |
| R1.6-1 | Metric definitions paragraph | IV-A |
| R1.6-2 | Mean +/- sd of per-trial error beside the bias | Table II |
| R1.6-3 | Thresholds 3/4/5 cm, convergence time, failure | IV-A, Table II |
| R1.6-4 | Per-trial distributions: hardware CDF (Fig. 6), simulation median and p95 (Table II) | V-C, Fig. 6, Table II |
| R1.6-5 | MLP fields are a fixed spatial distortion | IV intro (batch 8a), V-A (8b) |
| R1.6-6 | Distortion against temporal noise, friction off | IV-C last paragraph, VI-A distortion paragraph |
| R1.7-1 | "indicate the type"; scope of abstract, introduction, conclusion | I, Abstract, VII |
| R1.7-2 | Cooperative estimation from spatially distributed measurements; centralized; OptiTrack (kept) | Abstract, I, III, Limitation Five |
| R1.7-3 | "instantaneous" and every nondegenerate affine field in the proposition | II-C Proposition 1 |
| R1.7-4 | Classification evaluated in simulation (Table IV, band epsilon_c = 0.2); hardware classifier logged with values `\open{campaign}` | IV-D, V-B, V-C |
| R1.7-5 | No-advection statement in abstract and conclusion (kept) | Abstract, VII, III-A |
| R2.1-1 | Reused layers named [5], [8], [50] (PDF [42] is key 50), and reused and new blocks marked in the figure | III, Fig. 2 |
| R2.1-2 | Contributions rewritten with ordinals | I |
| R2.2-1 | Larger teams: averaging, dropout tolerance, load, footprint | VI-C Limitation Three |
| R2.3-1 | Proposition with assumptions | II-C |
| R2.4-1 | Closed-loop error dynamics e-hat-dot = -k e-hat and exponential solution | II-D |
| R2.5-1 | Advection, drift, dropouts, latency, each with its effect | VI-C Limitation Two |
| R2.5-2 | Lab results separated from real-flow performance | Abstract, VII, VI-C |
| R3.1-1 | Novelty beyond two affine fits: the loop, the bound and the hardware | I (batch 2), contributions |
| R3.1-2 | Proposition 1 | II-C |
| R3.1-3 | Centralized computation distinguished from distributed sensing (kept) | Abstract, I, III, V-A, VI-C |
| R3.2-1 | Affine fields do not test extrapolation | IV-B |
| R3.2-2 | Curvature sweep (chi) and size sweep | IV-C, Fig. 3 |
| R3.2-3 | Start distances: gyre basins by distance, curved saddle up to 0.7 m | IV-C |
| R3.2-4 | Formation side 0.01 to 1 m, error grows as side cubed | IV-C, Fig. 3 left |
| R3.3-1 | Stability sentence replaced | VI-A last paragraph |
| R3.3-2 | Estimate modeled as p* + delta with delta <= eta e + delta-bar | III-D |
| R3.3-3 | Bound with lag alpha and 10 Hz updates | III-D (16)-(18) |
| R3.3-4 | Near-singular handling: no estimate below abs(det) 1e-10, hold last valid `\open`, no sigma_min gate | III-D |
| R3.4-1 | Comparison reframed, drift point conceded (letter carries the concession) | VI-B |
| R3.4-2 | Fixed-radius baseline under the same sensing: magnitude-only estimate with the same orbital law. Not contour following at abs(v(r_d)) | VI-B, Table VI |
| R3.4-3 | Repeated simulation trials (500 attraction, 50 orbit per setting) | Table VI |
| R3.4-4 | Oracle run with the true p* | VI-B, Table VI; IV-B; IV-C |
| R3.5-1 | Accuracy claims reworded, achievable range cited | Abstract, V-D, VII |
| R3.5-2 | Per-trial distance errors | Table II, Fig. 6 |
| R3.5-3 | Numerical convergence threshold | IV-A |
| R3.5-4 | Repeated orbits, between-trial sd against within-run sd. **Placeholder data** | V-D, Table V |

Every action has a location. Five rest on the campaign and use placeholder data or open settings: AE.1-5, R1.4-4, R3.5-4, the hardware half of R1.7-4, and the hardware rows of R1.6-2/3/4 and R3.5-2. R1.5-3 is answered only in the letter.

## Cuts and decisions for you to check

1. **Roadmap paragraph deleted** (Waight_style 6.4). The VI-B note's roadmap edit is therefore moot.
2. **Appendix A moved into Table I** as a fourth column. This removed the `\appendices` block once B was gone.
3. **Figure cuts:**
   - `figure_5.png` (sample trajectories, legacy-model simulation);
   - `color_sensor_hsv_predictions.png` (its RMSE values are in the V-A text);
   - `orbit_demonstration.png` (VI-B);
   - `allrunsadjusted.png` and `fig9.png` (4R vortex);
   - `orbit040_analytical_vs_real.png` (old mixed-gain orbit);
   - the synthetic hardware orbit figure, because Table V carries the same numbers.
4. **Figure shrinks:** Fig. 1 (0.35 to 0.29 textwidth), Fig. 4 (0.28 to 0.22) and Fig. 5 (0.43 to 0.35).
5. **Table merges:**
   - Hardware attraction became two `\syn` rows of Table II, which saved a full-width table. Old Table IV no longer exists, so the reviewers' "Table IV" and "Table V" now map to Table II and Table V.
   - Table II collapses the six identical linear rows into one.
6. **New figure files** in `cwaight_systems_paper/`:
   - `fig_nonlinear_size_curvature.png` is a crop of panels (a) and (c) of the study's `fig3_size_curvature.png`. The legend was split into two rows and the panel titles were cut, so the caption says Left and Right.
   - `fig_hw_attraction_ecdf.pdf` and `fig_hw_orbit_radius.pdf` (the latter unused) are three-robot-only versions made with a scratch copy of the draft's `figures.py`.
   - The drafts and `trunk/` were not edited.
7. **Robot model parameter table dropped.** The parameters are in the III-A text.
8. **Curvature coefficient renamed chi.** The figure axis still says "eps", and the caption maps it. epsilon is the convergence threshold, epsilon-bar the measurement-error bound, and epsilon_c the classifier band.
9. **VI-A orbital-bias sentences changed** although the error note said to keep them. The revised Table III no longer shows a negative bias at 0.5 m, so the text now cites Bias against Ctrl., and kappa(J-hat) became 1/sigma_min.
10. **Error-note Laplacian sentence omitted,** for space.
11. **Metric definitions moved to IV-A,** which is first use. V-B points to IV-A, and this supersedes prose batch item 7 and the V_VIA V-B paragraph.
12. **Effort metric.** Hardware uses E_u from revision_metrics Table IV, not the V_VIA draft's path length L, so simulation and hardware share one effort metric.
13. **Simulation command limit stated** as 0.5 m/s, from revision_metrics/sim.py. Hardware states 0.3 m/s per axis. The campaign will align them.
14. **Prose batch item 10 not applied,** as instructed. VI-B describes the magnitude baseline itself.
15. **Numbers from drafted notes are marked `\rev`,** although they are outside the authorized source list:
    - gamma = 1.0003 (stability note);
    - 2e-5 m/s (prose batch N1);
    - the classification percentages (authorized).
16. **The contribution sentence "validates ... in a hardware campaign on two printed fields" is wrapped in `\syn`,** because the campaign has not run.

## Campaign-dependent sentences

- **Abstract:** last numeric sentence.
- **I:** fourth contribution.
- **V-B:** every `\open` item.
- **V-C:** every `\syn` number, the friction-ring comparison and the classification sentence.
- **V-D:** the whole results paragraph.
- **VI-A:** the hardware-metrics paragraph.
- **VII:** the hardware sentences.
- **III-D:** "holds the last valid estimate" (`\open`).
