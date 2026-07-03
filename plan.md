# Separatrix / Okubo-Weiss Paper: Test Plan and Completion Tracker

Working notes for finishing Paper_Draft_2A.tex (IEEE Transactions on Robotics target).
If a session is cut off, read this file top to bottom and resume at "Current status".

## Current status (2026-07-02, second session pass)

- Phase: BUILD. Test plan agreed and locked (see Agreed decisions below).
- Phase 0 infrastructure DONE: measurement-noise hook, heading_offset in reset,
  smoke tests pass, single-trial benchmark 53 ms (200 steps).
- LaTeX untouched so far. Noise-subsection rewrite proposed to user as
  before/after, awaiting approval before writing to disk.

## Agreed decisions (user, 2026-07-02)

1. Merge the two test plans: draft sec:test_plan metrics (time-to-band,
   tracking error, shape error, effort) PLUS straddle-retention success metric
   (Michini-comparable). Straddle = robots on both sides of the separatrix.
2. Theorem 3 / Problem 1: reframe as separatrix NETWORK traversal; terminal
   capture becomes the tightened-threshold special case. The eps-knob
   "park vs traverse" contrast gets one demonstration example and a
   Discussion paragraph (user considers this a key strength).
3. Position noise: KEEP the code's semantics (field sampled at perturbed
   location, fit uses nominal positions). Rewrite paper Eq. (pos_noise) to
   match. Reason: apples-to-apples with prior papers, not OptiTrack.
4. Measurement noise: additive Gaussian on (u,v) readings, Michini Eq. (6)
   style, now implemented as cluster.measurement_noise_std.
5. 2D noise sweep (sigma_uv x sigma_p), extend levels upward until success
   rate drops below ~10 percent. 1000 trials/cell for development, 10,000
   for the final run (benchmark says ~3.5 h single-core, fine parallelized).
   Random initial heading included via reset(heading_offset=...).
6. Clean-case trial length: fixed step count is acceptable; use L_Gamma/m
   theory bound only if trivial to implement and explain.
7. OW corner hop: document in Failure Modes FIRST; prototype the corner-only
   Hessian eigen tie-break (inside eps_grad ball) at the END of the work
   plan (explicit go-ahead given, but last so context loss cannot orphan it).
   Controller section will then need a Sutton-and-Barto selector update.
8. Results order: estimator-accuracy-vs-noise subsection stays first;
   controller comparison demoted to a paragraph; tables from CSVs, at most
   one heatmap. Concept trajectories deferred to Results with a forward
   reference from the test plan; start-location map may sit in methods.
9. Ocean HFR methods subsection: rewrite entirely (2km, alpha_mom=0,
   stiction 0.002, gain 1.8, V_MAX 0.04, TIME_WARP 6000, 0.7x formation).
10. Traceability rule: every experiment script header names the paper
    figure/table it produces; CSVs to experiments/outputs/ with metadata
    (git commit, seed, params) in header comments.

## Noise verification result (user asked; resolved 2026-07-02)

- "eps" in config/fields/*.yaml is the Shadden double-gyre time-dependence
  parameter (all 0.0 = steady). NOT noise.
- eps_raw / eps_dim / eps_grad are controller band thresholds. NOT noise.
- The remembered noise feature is noise_std on NNField/RBFField/EnsembleField
  in the ARCHIVED 3-robot codebase (trunk/Python_Simulations/Archive/...),
  additive Gaussian on each (u,v) reading. It was dropped in the 6-robot port;
  the draft's Noise Model text was written against the archive.
- Resolution: ported the same semantics to the 6-robot sampling path as
  cluster.measurement_noise_std in _sample_vector_at_robots
  (pentagon_primitives.py), next to the existing position_noise_std.
  The two attributes ARE the 2D sweep the files dictate.

## Phase 1 results (2026-07-02, user has NOT yet reviewed; nothing in .tex)

Scripts: experiments/estimator_accuracy_sweep.py (data, 4 CSVs in
experiments/outputs/estimator_accuracy/) and plot_estimator_accuracy.py
(review figure estimator_accuracy_vs_noise.png in same folder).
Approved noise subsection WAS written into Paper_Draft_2A.tex and compiles.

Key numbers (strain-region location, rho = 0.075, N = 200k draws/cell):
- Geometry: pentagon_small.yaml is an exact pentagon plus center; robot 0
  at centroid (3e-7 off), ring radius rho = 0.0750.
- Lemma 1 VERIFIED: all six coefficient noise stds match closed forms
  within 0.2 percent (u0 1.000, ux/uy 1.001, uxx 0.999, uxy 1.002, uyy 1.001
  empirical/lemma ratios).
- Position-noise coupling VERIFIED: per-robot reading-error std matches
  first-order prediction sigma_p * ||row J|| within ~1 percent.
- Reliability thresholds (median rel. error = signal): grad D at
  sigma_uv ~ 0.008-0.01; D at ~ 0.06; closed-loop cliff expected near
  sigma_uv ~ 0.01 since Logic C leans on grad D.
- FINDING (needs paper treatment): H_D estimate has a STRUCTURAL truncation
  floor ~ 0.9 relative error at ALL rho (formula eq:hess_det drops
  third-derivative-times-field-value terms; exact only for quadratic
  fields). BUT eigenvectors survive: 0.0-0.35 deg error on the separatrix
  (symmetry keeps H_hat diagonal in the trench frame), 3.8 deg generic.
  Eigenvalues biased (transverse curvature reads ~9.5 vs true 19.2; near
  the saddle well both eigenvalues nearly vanish, so the lam1*lam2>=0
  attract-fallback test may not fire there even noise-free). Sensitivity
  subsection's "errors linear in coefficient errors" claim is wrong for
  H_D; rho* U-shape does NOT apply to H_D (floor-dominated to rho=0.3).
- Radius trade-off VERIFIED for D and grad D: U-shape minima at
  rho ~ 0.11 (D) and ~ 0.23 (grad D) at sigma_uv = 0.01. Estimator-optimal
  radius is ~3x the nominal formation; closed-loop tuning (ocean) chose
  smaller. Estimator-optimal vs controller-optimal is a Discussion nugget.

## Build phases (in order)

- Phase 0 (DONE): noise hooks + heading_offset + smoke test + benchmark.
- Phase 1 (DONE, awaiting user review of numbers/figure above): estimator
  accuracy experiment. Results opener.
- Phase 2: clean-case runs, 6 starts, network traversal + one park-vs-traverse
  threshold demo. Extend main_separatrix_v6r (longer SIM_STEPS).
- Phase 3: OW figure fix (loop-closure stopping rule, one clean diamond
  circuit, corner-hop documentation for Failure Modes).
- Phase 4: Monte Carlo sweep driver (parallel, seeded, CSV per trial),
  auto-extend noise grid until <10 percent success. Both controllers.
- Phase 5: Ocean HFR figures (4-panel progression, FTLE snapshots,
  ~100-start branch-sensitivity map, FTLE-ridge-distance metric).
- Phase 6: paper integration (diffs shown for approval first): merged test
  plan, noise subsection, ocean subsection, Problem 1/Theorem 3 reframe,
  Results, Discussion (one-knob paragraph), Failure Modes.
- Phase 7 (LAST): corner tie-break prototype + head-to-head at corners;
  controller text update only if it wins.

## Key files

- Target draft: `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex`
- Prior draft (sections II-V merged 2026-07-02): `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_1A.tex`
- Math walkthrough: `Paper_Writing/Separatrix_and_OW_Paper/teaching_notes.tex`
- Style/appendix references (submitted T-Mech paper, same codebase):
  `Paper_Writing/Vector Field Paper/Paper_Draft_3C.tex` and `Paper_Draft_4A.tex`
- Reference papers: `Paper_Writing/Reference Papers/` (papers [3] and [34] per draft bibliography)
- Simulation scripts (under `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/`):
  - Clean separatrix runs: `main_separatrix_v6r.py`
  - OW boundary tracking: `main_logic_g_newton_pentagon.py` (rename to
    `OW_contour_following_w_6_robots` later is pre-approved; add name-change comment in file)
  - Ocean HFR baseline: `main_ocean_hfr_2km_ftle_overlay.py`
- Planned results output folder: `experiments/outputs/` (CSVs per run)

## Proposed test plan (user's, under discussion)

1. Clean noise-free runs: ~6 starts (straddling and not), converge to separatrix,
   continue through bifurcations. Script: main_separatrix_v6r.
2. OW tracking (logic G): needs a better, more intuitive figure than current
   90-degree-turn-then-diagonal image.
3. Statistical noise sweep: start straddling, run the separatrix, success =
   complete without losing straddle. 100 or 1000 trials per noise level.
4. Real-world 2km ocean HFR: follow separatrix in real data. CRITICAL: double gyre
   uses alpha=0.7 + stiction; ocean runs use alpha=0 (time dilation, ~10 min/sample).
   Must be explained explicitly in the paper.

Results section: A) clean runs, B) noise table from CSVs + link to sensitivity theory,
C) OW path + stats table, D) ocean HFR 4x1 grid + FTLE snapshots + start-sensitivity.

## Constraints and style rules (do not violate)

- No emojis, no em-dashes, no AI voice. Sutton and Barto pseudocode style.
- Show diffs before writing to paper; user reviews all numbers before .tex entry.
- Inline numeric citations, hand-written thebibliography. No BibTeX.
- alpha collision: alpha_eig (eigenvalue real part) vs alpha_mom (momentum). Disambiguate.
- OW tangent is perp(grad D), NOT Hessian eigen-tangent (removed 2026-07-02, do not reintroduce).
- Measurement-noise hook on AnalyticalField not yet implemented (blocker for noise sweeps).

## Future tasks (after test plan agreed)

- Fix "??" section refs; 3D replacement for Fig 3; delete Figs 5 and 6.
- Section VI restructure: move Fig 4 to appendix, rewrite Cluster Space Controller
  subsection (6-robot pentagon, forward/inverse kinematics in appendix, style per
  Paper_Draft_4A/3C), summarize simulation program from 3C, update Ocean HFR subsection
  (2km, time interpolation, alpha/stiction explanation).
- Verify written noise model matches Python implementation.
- Failure Modes section: remove or expand honestly. Re-evaluate 3 listed limitations.
- Clean Future Work, rewrite Conclusion. Update CLAUDE.md at the end.

## Findings from context reading (2026-07-02)

1. Paper_Draft_2A.tex ALREADY contains a Testing Plan subsection (sec:test_plan,
   ~line 1459): 6 sigma_uv levels x 4 sigma_p levels, 200 trials/cell, metrics =
   time-to-band, tracking error (steps 100-200), shape error, control effort.
   Must be merged with the user's proposed plan (different success metric).
2. Draft's grid [-0.8,0.8]^2 exceeds the domain y-range [-0.5,0.5]. Bug in draft.
3. Draft says 200 steps/trial; scripts use 100 (separatrix) and 300 (OW).
4. Measurement-noise hook does NOT exist anywhere in src/ (draft's claim that
   NNField/RBFField/EnsembleField accept noise_std is false; verified by grep).
   Position noise exists but semantics differ from paper Eq. (pos_noise):
   code perturbs the field-query location, fit uses true positions; paper says
   reported positions are perturbed. Must reconcile code vs text.
5. Draft's Ocean HFR subsection is STALE: describes 6km data, alpha_mom=0.7,
   tau~28min, stiction retained. Validated 2km script uses MOMENTUM_ALPHA=0.0,
   STICTION=0.002, CONTROL_GAIN=1.8, V_MAX=0.04, pentagon_small_2km.yaml (0.7x),
   TIME_WARP=6000, 168 steps = 28 h.
6. Theory-experiment tension: Theorem 3(iii) claims terminal capture at the
   saddle (attract fallback), but with eps_dim=0.025 the entire separatrix
   including the well is inside the FLOW band (|D|/||H||_F <= ~5e-4), so the
   team is ejected along the wall trench (saddle's unstable manifold) and
   continues, which is exactly the "continues past bifurcation" behavior the
   user wants to show. Paper must present these consistently (user to decide).
7. OW diagonal-runaway mechanism: D=0 is a periodic family of straight lines
   (diamonds touch at corners, X-crossings where grad D = 0); at wall corners
   the flow fallback hands the team onto the neighboring tile's line, which
   runs straight forever (analytic field tiles the plane; no domain fence).
8. Michini [34] Table I is the precedent for the straddle-failure noise metric
   (N_fail vs noise intensity I, sigma up to ~mean flow speed 0.32).
   Brinon-Arranz [3] validates estimator error bounds in simulation only.
9. Formation scale rho ~ 0.08-0.115 (pentagon_small). Hessian noise gain
   (8/sqrt(10)) sigma/rho^2 ~ 250 sigma vs signal ||H_D|| ~ 19: draft's
   sigma_uv range {0..0.05} spans the interesting degradation regime.

## Session log

- 2026-07-02: Repo committed and pushed (fb4d429). plan.md created. Read
  Paper_Draft_2A.tex in full, Paper_Draft_4A.tex (sim/hardware/appendix),
  3C structure, both reference papers ([3], [34]), all three experiment
  scripts, formation yamls, noise-hook status in src/. Delivered test-plan
  critique; awaiting user decisions (see Findings 1, 4, 6 and critique).
