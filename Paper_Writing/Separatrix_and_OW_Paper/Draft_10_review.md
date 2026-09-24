# Referee report: Draft_10

**Paper:** Multirobot Tracking of Separatrices and Objective Eulerian Coherent Structures
**Against:** `Draft_10.tex` at commit `6b3fabc` (working tree identical to HEAD), 2026-09-23.
**Replaces:** `reviewer1_feedback.md`, `reviewer2_feedback.md` (both against Draft 8a),
`Reviewer1_Response.md` (superseded), `reviewer_status.md` (index of those two). They
were deleted on 2026-09-23 and remain in git history. What survives from them is in
Part B (reverified closed items) and Part C (standing decisions).

**Method.** Full read of the .tex, all nine figures and their `.meta.json` sidecars.
Closed forms and most quoted numbers were recomputed in scratch scripts; findings marked
**[verified]** were recomputed. The `paper-search` server was down, so the "first
demonstration" claim and the characterizations of cited works are unchecked.

**How to read this file.** Part A is the review. Part B reverifies every item the old
reviews marked closed. Part C lists what was declined on scope; do not re-raise those.
Part D is housekeeping. Items fixed in the .tex since the review was written have been
removed from this file, not marked. Numbers quoted here still need author sign-off
before entering any .tex.

---

# Part A. Review

## 1. Summary and narrative arc

**The core gap.** Existing robotic trackers begin straddling a structure whose identity
is given in advance [26-29], or compute FTLE on board over a finite horizon [30]. None
acquires a structure it does not already occupy from instantaneous measurements alone.

**Stated contributions.** (i) A second-order estimator recovering the full local
quadratic model from one synchronous sample, six robots minimal. (ii) Two trackers
reading the two terms of D = omega^2/4 - s1^2 that acquire and ride the structure
without pre-straddling. (iii) In the abstract only: the objectivity versus noise
tradeoff, and "first multirobot tracking of an OECS."

**Do they fill the gap?** The acquisition-without-pre-straddling gap is filled on the
benchmark. The contribution list misplaces the novelty.

- The estimator is thin as a headline. "Proved minimal" is a count of six unknowns per
  component, and the formation is that of [4]. It enables the work; it is not the
  novelty.
- The frame-dependent versus objective contrast is the paper's claim, and Section I
  never mentions it. "Objective," "OECS," "frame," and "rotating observer" do not appear
  in the Introduction. Title and abstract headline objectivity; the reader then does not
  meet it again until II-D.
- Nowhere does the paper say when a physical cluster is a rotating observer. The only
  hint is the Conclusion's "A cluster that can certify its orientation should carry the
  D tracker." That scenario (no trustworthy heading reference) is the motivation for
  s1 and belongs in the Introduction.

**Theory to results.** Mostly tight. sigma_eff, the s1 bias, the splitting identity,
D', and the trench-coincidence identity are all consumed downstream. Exceptions:

- The appendix derives a critical rate Omega of about 0.238 that is never tested.
- The convergence arguments use a_perp = kappa_perp / kappa_perp_hat without its size
  (see A2.1).

**Bonus contributions.**

1. The argument that an objective tracker must carry its tangent sign in memory, and
   that the memory, not per-cycle accuracy, sets its noise robustness. The channel
   ablation (clean flow seed moves success 44.6% to 45.8%, clean argmax to 100%)
   demonstrates the mechanism. It is the most interesting idea in the paper.
2. The reachable-set estimate (47.4% grid, 47.1% random). Prior AN primitives report
   local convergence only.
3. Conic degeneracy as the single failure mode of the estimator, and "a ring fails at
   any size."
4. The D tracker's benchmark success rests partly on an estimator artifact (A2.1). It is
   honestly reported and useful as a caution.

**Reordering.**

- One Introduction paragraph on objectivity and when a cluster is a rotating observer.
- Contribution paragraph rewritten as three items matching the abstract.
- Move the sign-memory paragraph out of IV-E into IV-F or its own subsection. (This does
  not ask for the abstract to be rebuilt around IV-E, which was declined; see Part C.)
- II-C (Sensitivity) analyzes D_hat, grad D_hat, H_D_hat, and s1_hat before II-D says
  why they matter. Swap the two subsections.

## 2. Technical rigor

Ordered by severity.

### A2.1 H_D_hat is not a consistent estimator of H_D at any radius [verified]

Eq. (hess_det) is the Hessian of det of the fitted affine Jacobian. The true Hessian of
det J also contains terms of the form J times third derivatives of the field (for
example u_x v_yxx), which a quadratic fit cannot see. Shrinking rho does not remove them.
On the benchmark separatrix the second-order part is
2 pi^6 A^2 diag(sin^2 pi y_f, -sin^2 pi y_f); the true Hessian is
2 pi^6 A^2 diag(1, cos 2 pi y_f). They agree only at the origin.

| Point | True eigenvalues | Fitted, rho=0.075 | Fitted, rho=0.005 | Rel. error at rho=0.005 |
|---|---|---|---|---|
| (-0.3, 0.1) | -15.6, -5.9 | -4.3, 4.8 | -4.8, 4.8 | 0.91 |
| (0, 0.3) | 5.9, 19.2 | -5.9, 6.5 | -6.6, 6.6 | 0.88 |
| (0, 0.45) | 18.3, 19.2 | -0.26, 0.43 | -0.46, 0.47 | 1.00 |
| (0, 0) | -19.2, 19.2 | -18.9, 18.9 | -19.2, 19.2 | 0.00 |

The sidecar agrees: `floor_H_D_vs_rho` is 0.918 at rho = 0.02.

- "Held there by truncation" (IV-C) will be read as finite-radius truncation. It is
  model-order truncation.
- II-D says s1's transverse curvature "lives in third derivatives, outside a quadratic
  fit's span," ruling out a Newton step for s1. The same is true of D. As written the
  paper implies D has a usable Hessian and s1 does not; neither does.
- "True" names two objects: the true H_D, and in Limitations "the true (hess_det)," the
  second-order part. Give the second its own symbol.
- On the benchmark a_perp = 1/sin^2(pi y_f), about 45 at (0, 0.45). The Lyapunov
  argument is continuous-time; the loop is discrete and lagged (Delta t = 0.1,
  alpha = 0.7, k = 3). At that gain my rough estimate is that the proportional region
  loses stability within about 0.1 of each saddle, giving limit cycles on the scale of
  k c_max Delta t, about 0.012. That would explain the D-tracker loops around p1* in
  Fig. 7 (S4, S6), which the text does not mention. Inference, not simulated.

### A2.2 The "strain-region point" is rotation-dominated [verified]

IV-C and Fig. 3 call (-0.3, 0.1) a strain-region point. D there is +0.545 (sidecar
`D_true`), inside the left gyre's Okubo-Weiss diamond, and H_D is negative definite.
There is no trench there.

- "The cliff and an estimator threshold nearly coincide" (0.0079 vs 0.0075) compares a
  closed-loop cliff on the separatrix from (0, 0.35) with an open-loop threshold at a
  gyre point where |grad D| is 3.42, against about 2.5 at (0, 0.35). The 6% agreement is
  likely coincidence.
- "39 degrees at the cliff" is the eigenframe of a gyre-point Hessian.

Fix: rerun the sweep at a separatrix point (one script parameter). The verification
script already labels the point `generic`, not strain-region.

### A2.3 The pentagon aliases cubic terms into the second-order coefficients [verified]

II-B: "the odd ring count suppresses third-order truncation bias." A pure cubic u = x^3
(all true coefficients zero) fit by the pentagon-plus-center returns (a4, a5, a6) that
rotate with ring phase at a 72 degree period, amplitude 0.0375 at rho = 0.075 and 0.0188
at rho/2 (linear in rho). On five points cos 3 theta = cos(5 phi - 2 theta), so the
third harmonic lands on the second. Any five-point ring has this. The gradient instead
gets a phase-independent O(rho^2) bias. IV-C's own "Rotating the ring through its 72
degree period moves that slope between 4.0 and 9.7, and halving rho halves the spread"
is exactly this alias. The paper's result contradicts the claim attributed to [4]; check
what [4] actually says.

### A2.4 Table I units contradict 0.87 m/s [verified]

With c_max in length per step (600 s), sqrt(2) k c_max is 8.72 m/s. Only length per
time unit (6000 s) gives the paper's 0.872 m/s. The double gyre's Omega r <= 0.10 vs
k c_max = 0.12 also only works in per-time units. g_perp "1/step" cannot be right: it
multiplies grad s1_hat, which is not a displacement.

### A2.5 The s1 latch arms almost everywhere [verified]

First contact is s1_hat < -s_trim, with s_trim = 0.05 against a well depth of 0.987.
That holds at the start over 89% of the benchmark domain, and the ocean run latched at
step one. The beta = 0 phase in which "the descent term homes onto the trench" runs in
about 11% of starts. Elsewhere acquisition is the ride plus projected descent, with the
tangent seeded off the structure. It works (six of six acquire), but the text describes
a different mechanism.

### A2.6 "Objectivity forces memory" is scoped too broadly

IV-E excludes the fitted curvature, the measured flow, and a world axis as sign
references. A fourth candidate exists: the along-trench component of grad s1_hat is
objective and signed. It fails only because the ride must pass through the along-trench
maximum, where it vanishes and reverses. The recursion is forced by objectivity plus the
traversal requirement. Saying so strengthens the claim by naming the exact condition.

### A2.7 The objectivity headline rests on one trial

- One start, one Omega. The appendix predicts a critical Omega of about 0.238 at this
  start. A sweep across it tests the appendix directly.
- Fig. 8(b) shows more than "leaves the structure." The dashed run drifts slightly west
  in the upper half, reaches p1*, then takes the opposite branch along the bottom wall
  trench, ending near (0.4, -0.6). The branch choice is consistent with the transport
  term Omega(-y, x) = (+0.1, 0) at p1* flipping the flow sign that orients w1 every
  cycle. That is the same failure the appendix analyzes for the s1 seed, applied every
  cycle, and it is reopened item E7 (Part B) showing up in the paper's own figure. The
  text does not say so.
- The abstract's s1 "recovers the identical material curve in any frame" overstates the
  appendix (bounded divergence, one exempt step, a margin condition, 0.025 gap). This is
  old item C6, still live.

### A2.8 The ocean validation has no null baseline

- 1.8 and 2.4 km mean distances on a 2 km grid where high-FTLE filaments are dense. What
  do a passive drifter and a straight line from the same start score?
- Forward-FTLE ridges are repelling. The abstract advertises "the curve along which
  floating material collects," which is attracting. Justify, or show backward FTLE.
- 84/100 jittered starts: jitter size not stated.
- 2|mu|/r has a 90th percentile of 3 to 5.9. Where mu > r, s1 = mu - r > 0: no
  compressive direction, and the "s1 trench" is set by divergence.

### A2.9 No sensor model

The paper never says how a robot measures local velocity. As a fraction of peak speed
the noise cliffs are about 2.5% (D) and 0.6% (s1). HF-radar and small-vessel current
estimates are commonly quoted at several cm/s against speeds of tens of cm/s. The ocean
trial samples the gridded product noise-free. So the paper shows noise-free success on
real flow and failure on synthetic flow at noise levels likely below real sensing.
Limitations should say so.

### Test plan versus results

IV-A announces five families; the paper runs at least ten (those five plus the
reachable-set grid, unsteady spot check, r_band sweep, rho sweep, channel ablation, 100
jittered ocean starts). Several lack a documented setup (jitter size, unsteady omega).

- The headline "four to five times" is in no figure. Fig. 4 is s1 only; D survives as a
  dash-dotted line in Fig. 3.
- Fig. 4(b), position noise, is never discussed (s1 50% at sigma_p about 0.0023).
- Straddle retention is plotted, never quantified. At sigma_uv = 0.002 retention is 23%
  against 44% success, so about half the successful runs lost the straddle at least
  once.
- Success falls toward 0%, not 50%. A single seed flip would plateau near 50%. Repeated
  re-signing with the near saddle 0.15 away and the far one 0.85 away explains the fall;
  "a sign inversion" undersells it.
- Tracking error versus noise is recorded, not reported (old R1 item 26).

## 3. Structural and narrative consistency

**Three stories.** The Abstract is the objectivity/noise tradeoff and a first OECS
tracker. The Introduction is an estimator plus trackers that need no pre-straddling. The
Conclusion is a selection rule decided by the observing platform. Old item C2 is closed
("field and mission" is gone), but the mismatch has moved to the Introduction, which now
states no selection rule and no objectivity motivation.

| Claim | Where | Check |
|---|---|---|
| 4 to 5x noise tolerance | Abstract, Conclusion | Matches (3.9 to 5.3) |
| Final gaps 1.219 / 0.025 | Abstract, Conclusion | Numbers match; "departs" understates Fig. 8(b) (A2.7) |
| "within 1.8 and 2.4 km" | Conclusion | These are means; "within" reads as a bound |
| "slightly different trajectories" / "separate only near landfall" | Abstract, IV-G | Fig. 9 s1 path runs about 5 km west early and ends near the western island |
| 0.87 m/s | IV-B | Only true with per-time-unit c_max (A2.4) |
| kappa 8.26 / 564, Ro 4.9, tau 0.28, correlation 0.44, appendix 0.096 / 0.081 / 1.19 / 0.238, [18.3, 19.2], thresholds 0.07 / 0.0075, 24 / 39 / 2.2 degrees, 0.94 | various | All recompute |

**Clarity and continuity.**

- "The Jacobian determinant traces separatrices" (title, abstract) holds only for
  vorticity-free trench networks.
- The sigma_eff check cannot test the approximation: on the double gyre
  |grad u| = |grad v| everywhere, exactly where Eq. (sigma_eff) is exact.
- "-120.39 deg W" is a double negative (twice). "Core wells" reads as gyre cores.

**Figures and references.**

- Fig. 2 (`s1_trench`) and Fig. 5 (`s1_channels`) are never referenced in the text.
- Fig. 7 axes say meters; the field is non-dimensional. Its square end markers are an
  unstated 150-step cutoff.
- Fig. 9 title contains the code variable `TIME_WARP=6000x`.
- Fig. 5(b) draws t pointing +y; downstream on x = 0 is -y.
- Fig. 4 does not name its primitive; its caption says "conditioned on" but the curves
  are joint rates.
- All 36 references are cited. `\cite{2}, \cite{3}` should be `\cite{2,3}`. [34] has
  "C. Kitts" where others have "C. A. Kitts." [36] lacks year and URL.

## 4. Prose, stylometry, readability

| Section | Grade | Why |
|---|---|---|
| Abstract | B- | Clear arc; clunky opener; jargon before any gloss |
| I Introduction | C+ | Human and readable; "spewing vortices," empty "contributes a novel approach," two-sentence contribution paragraph |
| II-A | A- | Best writing in the paper |
| II-B | B+ | Clean; "Alternately" should be "Alternatively" |
| II-C | C | About 130-word run-on across the sigma_eff derivation; noise algebra in prose |
| II-D | C+ | Identity well presented; surrounding sentences assume the argument is known |
| II-E | B- | Concrete except "signs its segments" |
| III-A | B | 75-word list sentence; latch used before defined |
| III-B | C | Units |
| III-C | C | "Height-ridge sense" unglossed; convergence compressed to assertion |
| III-D | C+ | Figure helps but is unreferenced |
| IV-A/B | B | Checkable setup |
| IV-C | C+ | Number-dense, fragments ("An individual formation orientation is.") |
| IV-D | C | "Watershed ... composited with the flow direction" |
| IV-E | C- | Grammar slip ("with no noise, which then separates them") |
| IV-G | B- | Clear setup |
| Conclusion | B- | Direct; overclaims |
| Appendix | B- | Followable |

The issue is concept density, not sentence length (mean 21.9 words, Flesch-Kincaid 10
to 15). From II-C on there is about one unglossed term of art per paragraph, and claims
and reasons share one clause joined by ", so" (55 times).

**Prose economy.** Overclaims or sentences that do not earn their place: "This paper
contributes a novel approach to the problem." "Six robots are proved minimal."

**Statistics.** 391 sentences, 7,359 tokens, 1,607 types, MATTR(100) 0.72 (high for
technical prose). Sentence-length coefficient of variation 0.54 overall but uneven: III-C
is 41 sentences at 20 +/- 7 words (0.36); the abstract and II-B are 0.38; II-C is 1.05
from one run-on. Top content words: tracker 55, field 47, trench 46, cluster 38, flow
36, gradient 36, against 28, point 27, carries 17. Top bigrams: "the [D/s1] tracker" 48,
"the cluster" 26, "the trench" 21, "the along-trench" 13, "rather than" 11. Top 4-grams:
"from one synchronous sample" 3, "four to five times" 3. Constructions: 49
negation/contrast phrases, 55 ", so", 40 "same." Fingerprints: "X against Y" for every
numeric comparison, and "carries" for any property a quantity has.

**AI detection.** Yes, in part. The register changes at about II-C, where the proofread
marker sits. Above it the prose has human texture ("spewing vortices," "Alternately,"
double spaces, "a robot's or a system of robots' motion"). Below it the prose is
compressed, and math objects are given agency ("carry," "read"). If told the whole
paper came from a strong model, I would believe it for II-C through the Appendix, not
for I through II-B. The house rules remove the surface tells, so what remains is
structural. The fix is to unpack each aphorism into a number or mechanism, or delete
it. Check IEEE's current wording on disclosing AI-generated text in the
acknowledgments.

## 5. Future work and the new gap

Stated: time-varying stability, online reshaping, Decabot then surface vessels.

**The new gap.** A tracker that is objective and noise-robust. The paper shows s1's
fragility comes from the sign memory that objectivity plus traversal forces. Next
question: does relaxing the traversal requirement (ride toward the deeper s1 well), or
temporal filtering of the carried sign, recover D-level robustness? The rho-sweep result
belongs with that follow-on.

**Missing limitations.** No sensor model (A2.9). H_D_hat biased at any radius (A2.1).
s1 not fully objective (the seed), omitted from abstract and conclusion. A 5.4 km radius
means links of about 11 km, so "ample bandwidth" is an assumption at that scale.
Temporal filtering across cycles is never named or excluded; filtering coefficients
across cycles is still instantaneous in the field sense.

**Optional additions (cheap).**

1. Rerun the estimator sweep at a separatrix point, e.g. (0, 0.35); relabel Fig. 3
   (fixes A2.2).
2. Sweep Omega through the predicted 0.238 (tests the appendix).
3. Both trackers' success curves on one axis.
4. Null baseline for the FTLE distance; backward FTLE.
5. Divergent form of the identity, D = mu^2 + omega^2/4 - r^2 and s1 = mu - r, so it
   holds on the ocean without the hedge.
6. A seventh robot in simulation for a fit residual and an online noise estimate.
7. Raise s_trim, or report how many acquisitions happen with beta = 0.

## 6. Final verdict

**Enduring ideas.** (1) The splitting identity as a design lens: two trackers read the
two terms of D = omega^2/4 - s1^2 from one fit, one frame-dependent and one objective.
(2) Objectivity plus traversal forces sign memory, and the memory sets noise robustness;
the ablation shows the mechanism. (3) For second-order vector-field estimation, six
robots off any common conic; a ring fails at any size.

**Novelty.** The estimator (formation of [4], per component) and the primitives (trench
following of [2], on derived scalars) are incremental. Tracking an OECS with a cluster
from instantaneous measurements, with the D/s1 contrast, appears new; the "first" claim
is unchecked against the corpus.

**Recommendation: Major Revision.** The core contribution is sound, the arithmetic checks
nearly everywhere, and the paper is unusually honest about where its trackers fail. Not
Minor because: D's curvature estimate is structurally biased and called finite-radius
truncation (A2.1); the estimator-to-cliff link rests on a mislabeled rotation-dominated
point (A2.2); the objectivity headline rests on one trial whose figure shows a failure
the text does not name, and the Introduction never motivates it (A2.7); one
self-contradicted citation claim (A2.3), a units error (A2.4), and two unreferenced
figures. None needs new theory; most is rewriting.

---

# Part B. Reverification of items the old reviews marked closed

Each item was checked against what the Draft 8a reviewer actually asked, not only
whether new text exists.

| Item | Original complaint (Draft 8a) | Draft 10 now | Verdict |
|---|---|---|---|
| E1 / C7 | kappa on mixed normalizations (8.26 and 564 normalized, 688 raw) | II-C states radius-normalized 8.26 and 564; IV-G reports 8.26 | **Closed.** |
| E3 | Band B called "on the separatrix" but surrounds {D = 0}, including the OW diamond | "The band surrounds {D = 0}, not the separatrix"; both thresholds named; off-structure trigger explained | **Closed.** Residual: "a flow that advects the cluster off-structure" conflicts with the no-advection premise; the command follows the flow, nothing is advected. |
| E5 | D_hat unbiasedness contradicted by the position-noise model | Scoped to measurement noise; position noise correlates components and biases D_hat | **Closed.** IV-C's "forty-eight conditions" are 8 sigma_uv values x 6 quantities, all measurement noise (`scripts/verify_estimator_bias.json`), so its scope is right. |
| E6 | Traversal-time bound false near saddles, where both branches of v_par vanish | Bound stated on segments excluding an epsilon neighborhood; transit of it "bounded in the same way as the isotropic point" | **Partly closed.** The isotropic-point bound rests on s1's constant-speed ride (c_max tanh 1). The D tracker has no constant-speed term; both branches of v_par vanish at the saddle, which is the original complaint. The analogy does not transfer. What carries D through is the x30 underestimate of lambda_1 (IV-D). The neighborhood is also said to be "where the terminal test applies," and IV-D shows that test never fires. |
| E7 | D's flow-sign reference v0^T w1 >= 0 degenerates at every saddle and where flow runs transverse, so the D/s1 asymmetry in IV-E is one of frequency, not kind | Old status mapped this to "sign test specified, terminal condition given." Neither addresses it. IV-E still presents per-cycle re-signing as a clean advantage. | **Reopened.** Draft 10 makes it worse, not better: since the D terminal test cannot fire, the cluster uses the degenerate sign at every saddle it passes, and Fig. 8(b)'s branch flip at p1* is this failure (A2.7). The old Reviewer1_Response rebuttal (the state machine switches before the sign is needed) is false for Draft 10. |
| E9 | lambda_1 < 0 < lambda_2 premise fails on the outer half; caveat only in Limitations | Caveat at the premise in III-C | **Closed.** Residual: "true H_D" and "the true (hess_det)" name different objects (A2.1). |
| E12 | e_H used, never defined | Symbol removed | **Closed as asked.** The floor it named is now "truncation," which A2.1 finds misleading. |
| D terminal test | Section III gave the D tracker no terminal condition | Eq. (d_capture), lambda_1 lambda_2 >= 0, in III-C | **Closed.** |
| "straddle retention" | Undefined | Defined in IV-E | **Closed.** Residual: never quantified (A2 test plan). |
| "trench-network distance" | Undefined | Old status said the term "no longer appears." It does, in IV-F, defined two sentences after first use. | **Closed** (it is defined). The old status note was wrong. Move the definition before first use. |
| R2-2 "four experiment families" | Count wrong | "Five experiment families" | **Closed as asked.** The paper runs about ten experiments (A2 test plan). |
| R2-2 k = 1.8 branch | Why a static gain selects a branch | States the field is time-varying, so k sets arrival time and hence the branch | **Closed.** |

**Old live items.** C2 is closed ("field and mission" is gone; see A3 for where the
mismatch moved). C6 is still live in the abstract (A2.7). Figure hygiene is still live
(Part D).

---

# Part C. Standing decisions: declined on scope

Carried over from the deleted `reviewer_status.md`. These are deliberate decisions, not
oversights. Draft 10 was narrowed to the D / s1 contrast and the controllers that ride
it. Do not re-raise.

- **A1, the gain ladder table.** Cut on purpose: nothing downstream consumes
  gamma = 8/sqrt(10), and the noise result is carried by the Monte Carlo sweep. The
  intro sentence advertising it was removed on 2026-09-23.
- **E4**, one gain per derivative order. Moot once the ladder was cut.
- **A3** experiment matrix, **A6** algorithm boxes, **A8** state-machine figure,
  **A2** parameter table (Draft 10 has Table I regardless).
- **A5** symmetric s1 runs for the three D-only experiments.
- **A9** feasibility sentence, **A10** compute cost, **A11** data availability.
- **Promoting the reachable-set estimate** and **restructuring the abstract around
  IV-E.**
- **Reviewer 2 Section 8** (writing grades) in full: its rewrites inserted em-dashes,
  "Crucially," "Ultimately," and intensifiers, against house style.
- **The rho sweep** (`rho_sweep_findings.md`, kept in this folder). 25,600 trials; no
  interior optimum; success rises monotonically with rho; closed-loop exponents 0.98 and
  0.94 against a predicted 1, 2.02 and 1.95 against a predicted 2. Draft 10 carries a
  two-sentence summary of it in IV-E. The full result is a gain-ladder result and most
  likely the seed of the next paper.

---

# Part D. Housekeeping

- `revision/ring_phase_open_loop.json` is deleted in the working tree but was not part
  of the stale-report cleanup. It backs IV-C's ring-phase numbers (slope 4.0 to 9.7,
  mean 6.83 vs 6.888, the [-0.51, 0.45] eigenpair), and `scripts/ring_phase_open_loop.py`
  still writes to that path. Restore it or regenerate it.
- Three result PNGs are modified and uncommitted: `estimator_accuracy_vs_noise`
  (with its `.meta.json`), `objectivity_traverser`, `traverse_vs_logic_c`.
- Figs. 4, 7, 8, 9 have no `.meta.json` sidecar. Their generators live in
  `VF_Robot/experiments/`, outside the pipeline:
  - Fig. 4 `flip_resolution`: `plot_flip_resolution.py`, reading CSVs from
    `mc_sweep_flip_resolution.py` and `mc_sweep_flip_resolution_sigma_p.py`. The live
    CSVs in `experiments/outputs/mc_oecs_traverse/` are untracked; only an older archive
    is in git.
  - Fig. 7 `traverse_vs_logic_c`: `main_separatrix_traverse.py`.
  - Fig. 8 `objectivity_traverser`: `traverse_objectivity_demo.py`.
  - Fig. 9 `ocean_ftle_trajectory_overlay_2km`: `main_ocean_hfr_2km_ftle_overlay.py`
    writes `ftle_trajectory_overlay_2km.png` to its own output directory; the paper copy
    was renamed and copied by hand.
