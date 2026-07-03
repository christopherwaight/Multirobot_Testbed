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

## HANDOFF (written 2026-07-02 at ~90 percent context; resume here)

State when this was written: Phases 0 and 1 complete, reviewed, approved,
committed, pushed. ALL approved paper edits are already IN Paper_Draft_2A.tex
and it compiles clean (ignore the pre-existing pcr font warning; missfont.log
predates this work): Noise Model subsection (sec:noise_model), III-D pointer
sentence, III-E radius-scope fix + propagation rewrite + Remark rem:eigvec,
fig:est_accuracy caption, and the full "Estimator Accuracy versus Noise"
Results prose (approved verbatim by user). Nothing else in the .tex has
been touched. User process rules: every NEW paper passage must be shown as
before/after and approved BEFORE writing to disk; user reviews all numbers;
no emojis, no em-dashes, plain prose; commit+push at every milestone.

Compile check: cd "Paper_Writing/Separatrix_and_OW_Paper" &&
/Library/TeX/texbin/pdflatex -interaction=nonstopmode Paper_Draft_2A.tex

### Phase 2 (NEXT): clean-case separatrix runs + continuation mechanism
1. Add mode logging to separatrix_logic_c_step (pentagon_primitives.py):
   if the cluster has a `diagnostics` list, append per call a dict with
   {mode: FLOW|ATTRACT|SLIDE, det, det_ratio (=|D|/||H||_F), lam1, lam2,
   flow_dot_w1}. Attribute-gated, backward compatible (cluster.diagnostics
   exists, nothing writes to it yet; reset() clears it).
2. New experiments/separatrix_clean_runs.py (traceability header: makes
   fig:sep_trajectories = figures/separatrix_trajectories.png). Six starts:
   (-0.45,0.30), (0.05,0.40), (0.00,0.00) crest, (0.10,-0.20), (0.25,0.42),
   (-0.30,-0.35). SIM_STEPS=600 (53 ms per 200 steps, trivial), V_MAX=0.04,
   GAIN=3.0, eps defaults 1e-3/0.025, headless Agg. SINGLE-panel figure:
   gray streamlines, D=0 diamonds, dashed separatrix, saddles marked, 6
   centroid paths in the validated palette + one pentagon footprint at t=0
   for scale. Robots omitted for clarity. CSV per run: start, time-to-band
   (|x_c|<0.05 for 10 consecutive steps), final pos, min dist to each
   saddle, mode occupancy fractions, whether it continued past a saddle
   (came within 0.05 of a saddle then later |y_c| beyond it / moved along
   the wall by >0.1). Outputs: experiments/outputs/separatrix_clean/.
3. Mechanism question to answer from the mode logs (see corrected finding 6
   above): near the well, does the selector fail to enter ATTRACT because
   the biased eigenvalues are indefinite? Expected yes. Report to user.
4. Park-vs-traverse demo (user wants ONE clean example, Discussion): same
   start (0.05,0.40); run A = defaults (expected: traverse through crest,
   continue past bottom saddle along wall trench); run B = candidate knob.
   Try in order until one parks: (i) eps_dim=0.0 and eps_raw=0 disabling
   the band near the well is NOT the knob (band already fails there);
   (ii) an estimator-aware attract test: replace lam1*lam2>=0 with
   ||H_hat||_F below threshold OR |D| large + grad small => attract;
   (iii) simplest honest knob: detect |D| > 0.5*pi^4*A^2 with small ||g||
   (deep well) and switch to attract. Whichever works, it is a CONTROLLER
   CHANGE: show the user the mechanism and the proposed selector edit
   BEFORE touching pentagon_primitives beyond the diagnostics hook.

### Phase 3 STATUS (2026-07-02 second window): runs DONE, decision pending
- experiments/ow_clean_runs.py + outputs/ow_clean/ committed. Four runs:
  lock-on 13-16 steps, mean |D| 0.003-0.007 on-boundary (0.3-0.8 percent
  of well depth 0.974), corner dwell 4-8 steps. KEY RESULT: the D=0 set is
  a straight-LINE NETWORK crossing at the diamond corners; the
  perp-gradient tangent follows its line STRAIGHT THROUGH each crossing
  (Newton resumes on the continuing line, as the draft Remark says), so
  diamond circumnavigation NEVER happens; runs exit the domain along
  their line. Loop-closure rule never triggers. This is the honest
  behavior of the published law, and it explains the user's original
  "90-degree turn then diagonal" complaint: the old figure showed the
  approach snap plus a straight zero line.
- DECISION NEEDED FROM USER (check-in sent):
  (a) Honest reframe: Problem 2 wording "circulate along" -> "travel
      along" the boundary; figure presents line-network following (the
      current PNG concept, restyle only). No controller change. Theorem
      thm:ow already matches (a).
  (b) Corner-turn branch selection to achieve circulation: at an
      X-crossing the local branch directions are the asymptotes of the
      indefinite quadratic (directions t with t^T H t = 0, computable
      from H_hat); pick the asymptote that turns. This is the Phase 7
      prototype pulled forward, a controller change (new primitive like
      logic_g_fable, keep logic_g_newton as revert per user pattern).
  Recommendation given: (a) for the main results + (b) as the Phase 7
  prototype; if (b) validates, present both behaviors (straight-through
  vs circulate) as ANOTHER one-knob pair, mirroring park-vs-traverse.

### Phase 3 original spec (superseded by status above)
- experiments/ow_clean_runs.py from main_logic_g_newton_pentagon.py (user
  pre-approved renaming logic G files to OW_contour_following_w_6_robots
  style later; add a name-change comment inside if renamed). Loop-closure
  stopping rule: stop when centroid returns within 0.05 of its first
  D=0 crossing point after step 50; report circumnavigation time and mean
  |D| over the circuit. Starts (-0.5,0.25) and (0.5,0.25) give clean
  diamond circuits for fig:ow_trajectories. Log corner events (||g||<
  eps_grad fallback active) per step. For Failure Modes: document the
  corner hop mechanism (D=0 is the periodic line family x_f +/- y_f in
  1/2+Z; corners are X-crossings with grad D=0; flow fallback can hand the
  team to the neighboring tile's line, which runs straight forever since
  the analytic field tiles the plane and there is no domain fence).
- Corner-only Hessian eigen tie-break: user gave explicit go-ahead to
  prototype BUT LAST (Phase 7), after everything else, in case context
  dies. Do not reintroduce the eigen-tangent as the tracking direction.

### Phase 4: Monte Carlo sweep (both controllers)
- experiments/mc_sweep_separatrix.py + mc_sweep_ow.py + shared
  experiments/_mc_common.py. multiprocessing Pool; per-trial seed =
  stable hash of (controller, sigma_uv, sigma_p, trial_idx); rng seeds
  np.random.seed per worker trial (hooks use global np.random).
- Grid: sigma_uv in {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1} x sigma_p in
  {0, 0.005, 0.01, 0.02, 0.05}; AUTO-EXTEND the top of each axis x2 until
  the success rate in the corner cell drops below 10 percent (user wants
  the cliff fully bracketed). Starts: 5x5 grid over [-0.8,0.8]x[-0.4,0.4]
  (NOTE: the draft's [-0.8,0.8]^2 is a bug, y domain is [-0.5,0.5]) plus
  random jitter; random heading via reset(heading_offset=U[0,2pi)).
  Trials: 1000/cell development, 10000/cell final run (53 ms/trial,
  ~3.5 h single-core at 10k x 24 cells; parallelize, 8 workers ~30 min).
- Metrics per trial (CSV row): controller, sigma_uv, sigma_p, start_x/y,
  heading, seed, success_band (time-to-band <= budget, no formation
  collapse: shape RMS > 2x nominal pair length), success_straddle (robots
  on both sides of x=0 maintained from first straddle to end; Michini
  Table I comparison), t_band, mean+p95 tracking error steps 100-200
  (|x_c| for Logic C, |D| for OW), shape RMS, control effort.
- Compare measured cliff onset to the sigma_uv~0.01 open-loop gradient
  threshold (prediction already written into Results prose; the measured/
  predicted ratio quantifies closed-loop robustness, a headline number).
- Also verify: success independent of initial heading (Lemma isotropy).

### Phase 5: Ocean HFR (2km)
- Baseline main_ocean_hfr_2km_ftle_overlay.py works as-is (validated
  2026-07-02 config in-file: alpha=0, stiction 0.002, V_MAX 0.04,
  GAIN 1.8, TIME_WARP 6000, 168 steps, pentagon_small_2km.yaml 0.7x,
  start (34.4,-120.39)). New figures: (a) 1x4 grid of path-so-far over the
  INSTANTANEOUS current field + OW boundary at ~7h intervals; (b) FTLE at
  4-5 snapshot times (field evolution); (c) existing full-path FTLE
  overlay. (d) branch-sensitivity: ~100 starts jittered around the ridge,
  color-coded by final branch/landfall, plus distance-to-nearest-FTLE-
  ridge metric per path (FTLE via _ftle_common.compute_ftle_field).
- Methods rewrite (Section VI Ocean HFR subsection, lines ~1387-1429 in
  the current draft): STALE, still describes 6km data and alpha=0.7 with
  tau~28min as reasonable. Must be rewritten entirely per user decision 9:
  2km resolution, time-dilation argument => alpha_mom=0, stiction floor
  negligible, table contrasting Decabot vs ocean operating points. Show
  diff for approval first.

### Phase 6: paper integration (each block: diff -> approval -> write)
- Merge testing plans in sec:test_plan (fix grid bug; add straddle metric
  + Michini comparison sentence; trial counts 10k; auto-extension rule).
- Problem 1 + Theorem 3 reframe to network traversal (decision 2): capture
  becomes the special case; use Phase 2 mechanism results to write it
  honestly (the biased-eigenvalue continuation vs threshold-based capture).
- Park-vs-traverse Discussion paragraph (user calls this a key strength).
- Failure Modes: corner hop (Phase 3), formation degeneracy (det Phi
  monitoring), FLOW-band chatter role of eps_dim.
- Results B/C/D from Phase 3-5 CSVs. Parameter table TODO at line ~964.
- Fill Related Work TODO stubs? NO, user marked out-of-scope for now.
- Later cleanup list from user (turn 1): fix "??" section refs; replace
  Fig 3 with 3D render (script in vortex_field_analysis_plots area, save
  PNG to outputs, link); DELETE Figs 5 and 6 (trench cross-section keep?
  user said Figs 5/6 "don't make sense", verify which labels those are
  before deleting); move current Fig 4 (six-robot ring) to Appendix;
  rewrite Cluster Space Controller subsection (pentagon, not SAS-only;
  forward/inverse kinematics + Jacobians in Appendix, style of
  Paper_Draft_4A/3C appendices, no cluster-of-clusters concept); Sim
  program subsection summarized from 3C without raw script filenames in
  text (filenames in LaTeX comments only); verify noise model text matches
  code (DONE); Failure Modes expand (Phase 3); re-evaluate 3 limitations;
  clean Future Work; rewrite Conclusion; update CLAUDE.md at the very end.

### Phase 7 (LAST): corner tie-break prototype
- Inside eps_grad ball only: pick exit branch via Hessian eigenvector,
  head-to-head vs flow-drift fallback at all 8 diamond corners + crest.
  If it wins, propose Sutton-and-Barto selector update + control-law
  equations to user (controller section edit needs approval). If it
  loses, keep flow fallback and say so in Failure Modes.

### Build phase status

- Phase 0 (DONE): noise hooks + heading_offset + smoke test + benchmark.
- Phase 1 (DONE, reviewed, approved, integrated): estimator accuracy.
  All paper edits from it are in the .tex. eigvec_check.csv committed.
- Phase 2 (STARTED 2026-07-02, session ended at 98 percent context):
  - DONE: mode-logging hook in separatrix_logic_c_step (5 modes: FLOW,
    FLOW_DRIFT, SLIDE, ATTRACT, ATTRACT_FALLBACK; attribute-gated on
    cluster.diagnostics). DONE: experiments/separatrix_clean_runs.py ran;
    outputs in experiments/outputs/separatrix_clean/ (clean_runs.csv +
    separatrix_trajectories.png review copy).
  - MECHANISM QUESTION ANSWERED (user has NOT seen this yet, report it):
    5 of 6 starts reached a saddle (min dist 0.000-0.007) and ALL FIVE
    continued past it. Within 0.10 of a saddle the estimated eigenvalue
    pairs are indefinite and ~10x too small (e.g. -1.99/+1.96 vs true
    +18.3/+19.2), so the lam1*lam2>=0 capture test never fires as the
    theory expects: continuation is CAUSED by the H_D structural bias,
    confirming the corrected finding 6. Theorem-3-style capture does not
    occur with the realistic estimator even noise-free. The park knob for
    the park-vs-traverse demo must therefore be an estimator-aware
    selector change (plan Phase 2.4 options ii/iii), which is a
    controller change requiring user approval first.
  - FIGURE ITERATION NEEDED next session before it is paper-ready:
    trajectories are correct and legible (S1-S5 converge to x=0, ride the
    trench, then continue along the bottom-wall trench in both
    directions) but (a) runs should STOP at domain exit or ~50 steps
    after first saddle contact; currently 600 steps lets paths tile-run
    far outside [-1,1]x[-0.5,0.5] (finals like y=-1.8 are fictitious
    tiling structure) and leaves loop scribbles at the bottom saddle;
    (b) S6 (-0.30,-0.35) never banded; it rode the bottom-wall trench of
    the tiling left out of the domain (final -2.95,-0.71); replace with a
    start like (-0.20,-0.30) or keep and present as network behavior;
    (c) S1 label collides with the pentagon-footprint dots; (d) add
    direct labels for saddle markers and D=0 diamonds in the caption.
  - CSV facts for the eventual Results text: t_band = 35/0/0/6/18/-1
    steps for S1-S6; mode occupancy roughly FLOW 0.27-0.45, SLIDE
    0.28-0.34 on successful runs.
  - FIGURE ITERATION DONE (next session: candidate figures ready for user
    sign-off, review copies in experiments/outputs/separatrix_clean/):
    stopping rules added (domain exit |x|>1 or |y|>0.52; bottom-saddle
    contact + 150 steps; contact keys on the BOTTOM saddle only since all
    traverses end there), S6 swapped to (-0.20,-0.30), S1 label moved,
    domain rectangle drawn. All six starts now band (t_band
    35/0/0/6/18/13), reach the bottom saddle (0.000-0.019) and continue
    along the wall trench in both directions.
  - PARK-VS-TRAVERSE DEMO DONE (experiments/park_vs_traverse_demo.py,
    park_vs_traverse.png + .csv): start S4 (0.10,-0.20). Traverse = plain
    Logic C, exits domain at (+1.00,-0.51) after 228 steps. Park = wrapper
    with estimator-aware capture (fires when D_hat < D_capture = -0.5,
    commands saturated gradient descent on D using only D_hat and g_hat,
    the quantities the estimator delivers reliably); parks 0.0124 m from
    the bottom saddle, tail position std exactly 0 over last 100 steps.
    One threshold is the knob: D_capture = -inf reproduces traversal.
    Library primitive NOT modified; wrapper lives in the demo script.
  - AWAITING USER APPROVAL (selector change proposal, see session log /
    last check-in): add the D-based capture test to
    separatrix_logic_c_step as a fourth mode (PARK), replacing the
    eigenvalue-based capture that the H_D bias defeats. If approved,
    controller section needs: Sutton-and-Barto selector update, the park
    law equation, and the Discussion one-knob paragraph.
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
6. CORRECTED 2026-07-02 (earlier version had a factor-100 arithmetic slip):
   on the trench |D| peaks at pi^4 A^2 = 0.974 at the wells, and
   |D|/||H_D||_F at the well is ~0.036 > eps_dim = 0.025, so the FLOW band
   covers only roughly y in (-0.25, 0.25) around the crest and the band test
   FAILS near the wells. With EXACT estimates the attract fallback therefore
   does fire near the well and Theorem 3 capture is self-consistent at the
   default thresholds. The observed continuation past the saddle is instead
   most likely caused by the H_D structural bias near the wells (estimated
   eigenvalues there are near zero and indefinite, e.g. -0.72/+0.43 vs true
   +18.3/+19.2, so the lam1*lam2>=0 attract test fails and the selector takes
   the saddle branch) and possibly momentum. Phase 2 must log the selector
   branch per step to pin the mechanism empirically. The network-traversal
   reframe (decision 2) stands; the park-vs-traverse knob demo may end up
   being an estimator-aware fallback or threshold change, not eps_dim alone.
   NOTE: pentagon_primitives does not yet write cluster.diagnostics; Phase 2
   adds per-step mode logging there (attribute-gated, backward compatible).
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
