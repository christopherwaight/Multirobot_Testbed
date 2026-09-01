# Technical Audit: Paper_Draft_4A.tex

Date: 2026-07-03. Scope: Phase 1 (math + code), Phase 2 (experimental logic),
Sec. VI-A double-check, and Phase 3 (literature nuance, Section 7). All line numbers
refer to Paper_Draft_4A.tex as of this audit. Verification scripts live in the
session scratchpad (`master_verify.py`, `verify_eq11_sign.py`,
`param_sensitivity_mc.py`, `verify_error_analysis.py`, `verify_rbf_followup.py`,
`baseline_harness.py`); they are reproducible against the repo venv.

---

## 1. Verdict

The core method is sound. The estimation pipeline (Eqs. 2-9) is exact and matches the
code line for line; the minimality argument is correct as a necessity claim; Table I,
Appendix A, and the simulation results (Tables II and III) were all independently
reproduced. One stop-level defect was found and fixed with approval: the printed
orbital control law (Eq. 11) was radially unstable under the paper's own definitions.
The implementation always had the correct sign, so no experimental result is affected.

The remaining issues are: one internal inconsistency in Appendix B (FK/IK disagree by
pi in theta_c), two places where the text misdescribes what the simulation actually did
(stiction value, orbital averaging window), an unexercised claim (critical point
"types" are never classified in any experiment), and a set of smaller wording,
rounding, and notation corrections listed in Section 3.

## 2. Stop-condition disposition

**Eq. 11 sign error (FIXED, user-approved).** With r = p* - p_c (r_hat pointing toward
the critical point, line 181), the printed law `p_c_dot = k_t r_perp - k_r (r - r_d)
r_hat` gives e_r_dot = +k_r e_r: the commanded radius is a repeller (verified
symbolically and by integration; starting 5 cm outside r_d = 0.3 diverges to 7.8 m in
10 s, starting inside collapses onto the critical point). The code
(`primitives.py`, orbiter) uses the opposite, stable sign. Fix applied per approval:
the radial term in Eq. 11 (line 184) is now `+ k_r(r - r_d)\hat{\mathbf{r}}`, which
makes the line 188 claim e_r_dot = -k_r e_r true and matches the code exactly. The
paper recompiles clean (11 pages, zero undefined references).

No other finding met the stop bar. The Appendix B inconsistency (Section 3, edit B1)
is a real mathematical contradiction between two printed equations, but it is confined
to an orientation convention, affects no result, proof, or control computation, and
has a one-line fix; it is reported here rather than stopped on.

## 3. Surgical LaTeX edits

Not applied to disk (per repo policy); before/after shown for approval. Ordered by
severity.

### B1. Appendix B: FK and IK are inconsistent by pi in theta_c (lines 622-623 vs 630-648)

Eq. 27 defines `theta_c = atan2(y2 - y1, x2 - x1)` (direction robot 1 -> robot 2), but
the inverse kinematics (Eqs. 28-30) place robot 1 at +x from robot 2 in the local
frame, so the direction robot 1 -> robot 2 is the local -x axis. Composing the printed
equations, FK(IK(theta_c)) returns theta_c + pi for every configuration (verified on
200 random configurations; shape and centroid round-trip exactly). The pair as printed
is therefore not mutually inverse. Inert in practice (orientation is never actively
commanded, omega_c = 0, and the code uses FK self-consistently), but a reviewer who
composes the appendix equations will find it.

Minimal fix, Eq. 27 (line 623):

```
BEFORE:  \theta_c = \text{atan2}(y_2 - y_1, x_2 - x_1)
AFTER:   \theta_c = \text{atan2}(y_1 - y_2, x_1 - x_2)
```

(direction robot 2 -> robot 1, which is the local +x axis used by the IK). Recommend
also adding one sentence after Eq. 30: "The construction assumes robot 3 lies
counterclockwise from the robot 1 - robot 2 edge; beta from Eq. 24 lies in (0, pi), so
the mirror configuration maps to the same SAS parameters." The same convention notes
were added as comments to `kinematics.py`.

### E1. Line 86: "distinct eigenvalues" excludes the paper's own sink and source

The sink (Eq. 16) and source (Eq. 17) are star nodes with repeated eigenvalues
(-1, -1) and (+1, +1) (verified symbolically), so the qualifier contradicts two of the
six test fields, and line 88 then classes "repeated roots" as out-of-scope degenerate
cases while two in-scope fields have them.

```
BEFORE (line 86): For $2 \times 2$ systems with distinct eigenvalues and nonzero
determinant, six standard critical point types arise from distinct eigenvalue
structures, ...
AFTER:            For $2 \times 2$ systems with nonzero determinant, six standard
critical point types arise from the eigenvalue structure, ...
```

And in line 88, replace "such as repeated roots, or cases where the determinant is
zero" with "such as cases where the determinant is zero" (or explicitly note that the
sink and source used here are the repeated-eigenvalue star-node special case of
nodes, which Table I's sign conditions still cover).

Related nuance worth one sentence at line 86: "the local topology is determined by the
Jacobian" holds for hyperbolic critical points (Hartman-Grobman); the center is the
exception (a nonlinear perturbation of a linear center can be a spiral), and under
measurement noise the estimated real part alpha of a center hovers near zero, so
center-versus-weak-spiral discrimination is inherently ill-posed. This matters
because contribution 1 claims type identification.

### E2. Line 221: simulation stiction is misstated

Text says robots are modelled with "a minimum speed of 0.05 m/s needed to overcome
stiction". The simulation actually ran with the Omnibot default of 0.025 m/s
(`omnibot.py`; `OmniCluster` does not override it). Evidence: rerunning the Table II
protocol with stiction 0.025 reproduces the printed values (bias 0.0009 m, precision
0.0145 m vs printed <0.001 and 0.014-0.016); with the text's 0.05 the precision would
be 0.0455 m, three times the printed value. Decision needed: either correct the text
to 0.025 m/s (and explain it as sub-hardware, or remeasure), or regenerate Tables
II-III with 0.05. The current tables correspond to 0.025.

### E3. Line 229 vs code: alpha = 0.717 vs 0.7

Eq. 13's alpha = exp(-dt/tau) = 0.7165 at tau = 0.3 s is correct (ZOH discretization
verified symbolically), but the simulation uses a fixed alpha = 0.7, i.e. tau = 0.280 s.
Closed-loop effect is negligible (precision 0.0145 vs 0.0148), but the paper asserts
the simulation implements Eq. 13 with tau = 0.3. Either set
`momentum_alpha = exp(-0.1/0.3)` in code and re-run baselines (behavior change, not
done in this audit), or soften the text: "alpha ~= 0.72 (0.70 in the simulation)".

### E4. Line 324-325: orbital statistics window is misdescribed

Text: "The mean radial error and standard deviation were recorded over the entirety of
each run." The generating script (`generate_simulation_orbital_table_8.py`,
STABLE_START = 100) skips the first 100 steps (10 s transient) and averages steps
100-600. The script's CSV reproduces Table III digit for digit (e.g. 0.1459 -> 0.146,
std 3e-6), so the table is right and the sentence is wrong.

```
AFTER: ... the orbital controller (Eq.~11) was used for 600 timesteps. The mean
radial error and standard deviation were computed after discarding the first 100
timesteps (10 s) as a settling transient.
```

Note the inconsistency this exposes: the hardware analysis (`orbiter_plotter.m`)
computes mean(r), std(r) over the full record with no transient skip, so Table V
(full run) and Table III (steady state) are not like-for-like. Either recompute one
of them with a matched window or disclose the difference where the two are compared.

### E5. Line 263: starting distribution

Starts are uniform in the box [-0.5, 0.5] x [-0.5, 0.5] (max distance 0.71 m), not a
1 m disk. "within 1 m of the true critical point" is technically true but implies the
wrong geometry:

```
AFTER: ... initialized at a random position drawn uniformly from a 1 m x 1 m box
centered on the true critical point ...
```

### E6. Line 272 / Table II caption: define "convergence"

The Monte Carlo success criterion is "no NaN and centroid stayed within |x|,|y| < 10 m
for 100 steps"; there is no epsilon-ball convergence test. The claim "converged in
100% of the 1000 trials" is defensible because the final-position statistics (bias
<0.001 m, precision ~0.015 m) show every run ended at the critical point, but the
criterion should be stated. Suggest appending to line 263: "A trial was counted as
convergent if the trajectory remained bounded and terminated within the stiction-
limited neighborhood of the critical point; final-position statistics quantify that
neighborhood."

### E7. Line 351-352: explain the identical noise-free orbital performance

Verified mechanism, worth one sentence in place of the bare observation: on exactly
linear noise-free fields the three-point fit recovers p* exactly at every step, so the
controller input, and hence the trajectory, is independent of the field; the six
noise-free trajectories are numerically identical (max deviation 1e-9 m, dominated by
a 1e-10 regularizer in the vortex field code). The residual "std below 2e-5" is the
steady-state radius ripple, not field variation.

### E8. Line 498: mislabeled quantity and sharper mechanism

`k_t` IS the commanded orbital (tangential) speed; `k_t/r_d` is the angular rate.

```
BEFORE: ... the commanded orbital speed $k_t / r_d$ is high relative to the radial
correction gain $k_r$, and the resulting centripetal acceleration ($v^2 / r$) ...
AFTER:  ... the commanded angular rate $k_t / r_d$ is high relative to the radial
correction bandwidth $k_r$, and the centripetal demand ($k_t^2 / r$) ...
```

Quantified decomposition (point-mass replication of the full Table III bias column to
within 0.003 m): at r_d = 0.30, bias is +0.060 m with the momentum model, +0.012 m
with instantaneous dynamics at dt = 0.1 s, and +0.0001 m at dt = 1 ms. So roughly 80%
of the small-radius bias is the first-order actuator lag and 20% is 10 Hz
discretization (chord-stepping along the tangent), both of which the radial gain must
cancel. The discrete-time equilibrium radius solves
(rho - dt k_r (rho - r_d))^2 + (dt k_t)^2 = rho^2. Optional, but this turns a
qualitative paragraph into a checkable one.

### E9. Rounding and small numeric inconsistencies

- Table IV (line 429): vortex avg rest point (-0.004, 0.012) gives bias 0.0126, printed
  as 0.012 (rounds to 0.013); saddle (0.004, -0.001) gives 0.0041, printed as 0.005.
  The biases were evidently computed from unrounded data; print coordinates to 3-4
  decimals or recompute so the table is self-consistent.
- Conclusion (line 548): "position errors under 0.012 m" -- the vortex bias IS 0.012
  (0.0126 unrounded). Say "of 0.012 m or less" or "within 0.013 m".
- Fig. caption line 374: "Hue RMSE = 0.027 (9.7 deg ...)" vs body line 369 "0.17
  radians (9.7 deg)". Both are right (0.027 x 2 pi = 0.17 rad) but the caption should
  say "0.027 normalized (0.17 rad, 9.7 deg)" to stop a reviewer flagging it.
- Abstract (line 37) says 157 trials; Introduction (line 66) says 169 experiments;
  Conclusion says 157. All reconcile (157 convergence + 12 orbital) but state the
  split once in the abstract: "157 convergence trials ... and 12 orbital trials".

### E10. Notation collisions (low priority, cheap to fix)

- `r` is the orbital radial magnitude (Sec. II-D), the third triangle side (Eq. 23),
  and the polar radius (Appendix A). Rename the triangle side to `s` (one symbol, three
  equations) and Appendix A's polar radius is fine in context.
- `alpha` is the eigenvalue real part (Table I), the momentum coefficient (Eq. 13),
  and a triangle vertex angle (Eq. 25). Rename the Appendix B angles (alpha, gamma) to
  phi_1, phi_3; consider `lambda_Re` unnecessary, Table I context is clear.
- `J` is the field Jacobian (Eq. 8) and the 6x6 kinematic Jacobian (Eq. 33). Rename
  the kinematic one J_c (cluster Jacobian), matching cluster-space literature.
- Line 750, bibitem 36: `\textit;{IEEE Transactions on Robotics}` has a stray
  semicolon; renders wrong. Remove it.
- Citation bracket style is inconsistent: "[7], [8] [9] [10] [11]" (line 49),
  "[28],[29]" (line 53), "[4, 25]" (line 195) vs separate brackets elsewhere. IEEE
  wants "[8]-[11]" or comma-separated single brackets, used consistently.
- Line 192: "MultiLayer" -> "Multilayer".

### E11. Strengthenings the math now supports (optional, from this audit's verification)

- Minimality (Sec. II-C, lines 161-165): the current argument is necessity by counting.
  The audit verified both halves: (i) two robots admit two distinct affine fields with
  identical readings and different critical points (explicit counterexample), and
  (ii) three robots suffice iff non-collinear, since det(A) = 2 x (signed triangle
  area). One added sentence makes the "proof" complete: "Three non-collinear robots
  are also sufficient: det(A) equals twice the signed area of the formation triangle,
  so A is invertible exactly when the robots are not collinear." This also connects
  forward to the conditioning discussion (line 502) instead of leaving collinearity as
  a Discussion afterthought.
- Attraction (line 178): on the linear fields of Sec. IV the estimate is exact and
  hence stationary, so the stationarity proviso is automatically satisfied in the
  noise-free experiments; one sentence here inoculates the claim. Add "in the absence
  of actuation limits" if Eq. 10 is to be called globally exponential (the
  implementation saturates commanded speed).
- Estimator sensitivity: first-order error is delta_p* = -J^{-1} [delta_U(p*);
  delta_V(p*)] with delta_U(p*) = [x*, y*, 1] A^{-1} nu_u. Monte Carlo matches this
  within a few percent; the induced rms error grows linearly with the distance from
  the formation to the critical point and inversely with formation size (e.g. sigma_uv
  = 0.01: 0.008 m at the critical point, 0.059 m at 1 m range for the 0.33 m triangle).
  This one formula would let the paper state a quantitative accuracy budget instead of
  the qualitative kappa(A), kappa(J) discussion, and it predicts the observed hardware
  precision scale (0.009-0.014 m).

## 4. Experimental logic findings (Phase 2)

1. **The "types" half of contribution 1 is never exercised.** No eigenvalue
   classification of the estimated Jacobian exists anywhere in the 3-robot code, the
   Monte Carlo runner, or the MATLAB hardware analysis; nothing in Sections IV-V
   reports a classification result. Table I plus Eq. 8 make the capability plausible,
   but as validated the system estimates locations only. Either (a) add a cheap
   classification experiment (log eigenvalues of the estimated J during the existing
   runs and report the confusion matrix across the six fields; near-zero-alpha
   ambiguity for the vortex should be disclosed per E1), or (b) soften contribution 1
   and line 64 to "locations, with type available from the estimated Jacobian's
   eigenvalues".
2. **Metric definitions are sound** (bias = norm of mean final centroid, precision =
   radial std, orbital error = mean(actual - commanded)), but the orbital sign
   convention (negative = inside commanded radius) is never stated; add it to the
   Table III/V captions. The transient-window mismatch between sim and hardware
   orbital statistics is item E4.
3. **Tables II and III reproduce, from the NN pipeline.** Mini-MC (300 trials)
   reproduces the Table II noise-free rows; the archived steady-state CSV reproduces
   the Table III noise-free column exactly; and the reconstructed-field columns of
   BOTH tables match the archived NN-variant outputs digit for digit
   (`monte_carlo_results/results_8fields_1000runs_nn.csv` and
   `experiments/simulation_orbital_results_8fields_nn.csv`; e.g. saddle bias
   0.004841 / precision 0.015742 vs printed 0.0048 / 0.0157, orbital -0.0029 at
   r_d = 0.50 vs printed -0.003). The paper's numbers are real and regenerable,
   with the parameter caveats of E2/E3.
   **Reproducibility hazard:** the un-suffixed scripts
   (`monte_carlo_analysis_run_to_center_8.py`,
   `generate_simulation_orbital_table_8.py`) load the RBF reconstructions instead,
   which do NOT reproduce the paper: on the RBF saddle, attraction precision is
   0.096 (archived CSV and audit rerun agree), roughly 6% of uniform starts escape
   the saddle entirely (final positions up to 1 m out, counted as "successes" by the
   divergence-only criterion), and orbits diverge for r_d >= 0.2 (archived errors
   +0.49 and +0.72 at 0.2 and 0.3). Anyone reproducing from the default scripts will
   conclude the paper's saddle rows are wrong. A provenance comment now marks the MC
   script; recommend the same for the orbital generator, and citing the `_nn`
   variants in any data-availability statement.
4. **The simulated "noise" is bias, not variance.** The NN (MLP) reconstructions
   add a systematic estimate offset but almost no per-trial scatter: reconstructed
   precision equals noise-free precision to the third decimal (saddle 0.0157 vs
   0.0157, vortex 0.0162 vs 0.0163), because the reconstruction error is a frozen,
   smooth field warp, not stochastic sensing noise. Line 500's "sensor noise ...
   propagates through the critical point estimation equations" is true in hardware,
   but in simulation the reconstructed fields test systematic reconstruction bias
   only; per-trial variance is entirely robot-dynamics scatter. One clause in
   Sec. IV would make this precise, and it strengthens the paper's own point that
   the estimator is deterministic given the field.
   The RBF comparison (point 3) doubles as an unreported robustness result: with a
   rougher (interpolating rather than smoothing) reconstruction, saddle performance
   degrades sharply while vortex performance survives. Worth a sentence in
   Limitations if the RBF results are ever mentioned.
5. **Dead code with wrong math is quarantined, not removed.**
   `calculate_jacobian_from_readings` (primitives.py) returns plane-normal components
   without dividing by the normal's z component, so its "derivatives" are scaled by
   2 x signed triangle area, and the sign flips with robot ordering (the default
   ordering is clockwise, so signs are inverted). It feeds only `find_center` /
   `find_sink_center`, both marked "Under Development. Do not use". A warning comment
   now sits on the function; recommend deletion in a future pass (was out of the
   approved cleanup scope).

## 5. Code cleanup performed (bit-exact verified)

Scope approved: comments + minor refactors, 3-robot core path only. A fixed-seed
harness (4 scenarios x 200 steps, attraction + orbit on vortex and saddle) was captured
before edits; after edits all 12 trajectory/velocity arrays are bit-identical
(`np.array_equal`).

| File | Changes |
|---|---|
| `src/control/primitives.py` | Equation-mapped comments (Eqs. 4-11) on both critical-point primitives; warning on `calculate_jacobian_from_readings`; no-op note in `vector_sum`; near-singular-J caveat; manual clamp -> `min()` (identical semantics); RNG note on the orbiter's random-escape branch. |
| `src/robot/omnibot.py` | Eq. 12-13 mapping; alpha 0.7 vs exp(-dt/tau) provenance note; stiction ordering note (pre-clamp magnitude, inert). |
| `src/robot/omni_cluster.py` | Formation-controller comment (Sec. III-C mapping); exactness note on the while-loop angle wrap; documented that `reset()` does not clear `diagnostics`. |
| `src/control/kinematics.py` | Appendix B equation mapping; the theta_c pi-offset and chirality caveats documented at module level; note that the 6x6 inverse Jacobian is finite-difference, not the analytic form the paper describes (line 247). |
| `experiments/main_omni.py` | `except (FileNotFoundError, Exception)` -> `except Exception` (identical behavior). |
| `experiments/monte_carlo_analysis_run_to_center_8.py` | Corrected the contradictory seed comments (module-level seed at line 14 is live); documented the success criterion and start box. |
| `config/formations/equilateral_default.yaml` | Fixed stale comment claiming p should be 0.433; value is and should be 0.33. |

Also applied (approved separately): the Eq. 11 sign fix in `Paper_Draft_4A.tex`;
two-pass pdflatex build clean, 11 pages, zero undefined references.

## 6. Error Analysis double-check (Sec. VI-A, lines 492-506)

Requested follow-up pass; every claim in the subsection was tested
(`verify_error_analysis.py`, `verify_rbf_followup.py` in the scratchpad).

### E12. Line 496: "alpha = 1" is a typo, and the machine-error figure does not reproduce

By Eq. 13, alpha = 1 gives v[k+1] = v[k]: robots starting at rest never move
(verified: final distance equals start distance after 1000 steps). The intended
ideal is alpha = 0 (instantaneous velocity response). Separately, with alpha = 0,
no stiction, and no velocity cap, the centroid converges to 9.0e-7 m and floors
there, because `critical_point_plane_fitting` returns zero velocity inside a 1e-6 m
deadband; it cannot reach 1e-15 by construction. What IS machine-precision exact is
the estimate p* itself on the linear fields (verified symbolically and numerically
in Phase 1).

```
BEFORE: However, repeating experiments with no stiction, no max velocity, and
$\alpha = 1$ confirms convergence up to machine error of $10^{-15}$.
AFTER:  However, repeating experiments with no stiction, no max velocity, and
$\alpha = 0$ (instantaneous velocity response) confirms convergence to the
controller's $10^{-6}$~m command deadband, with the critical point estimate itself
exact to machine precision.
```

(If the original 1e-15 run was done without the deadband, rerunning under the
current code will not reproduce it; the suggested wording matches what the released
code does.)

### E13. Line 502: equilateral optimality of kappa(A) VERIFIED, with a caveat

Among 200,000 random formations with the same size and centroid on the sampled
point, none beat the equilateral triangle's kappa(A) = 7.42 (best random 7.43,
under both fixed-circumradius and fixed-RMS-size normalizations). The claim stands.
Caveat worth a clause: kappa(A) with the raw [x, y, 1] rows is not translation
invariant (the same equilateral formation centered at (1,1) has kappa(A) = 22.4),
so the optimality statement should say "for a formation centered on the sampled
region" or be phrased on the centered coordinates.

### E14. Line 504: the negative large-radius orbital bias is now measurable, and it
is reconstruction distortion at range, not stochastic conditioning noise

Two tests separate the candidate mechanisms. (1) Injecting unbiased Gaussian noise
on p* with the distance-growing sensitivity law leaves the closed-loop radial bias
essentially unchanged (+0.040 vs +0.042 at r_d = 0.50 even at twice the nominal
noise), so zero-mean noise does not produce the negative bias. (2) Directly probing
the NN-reconstructed fields: the apparent distance ||p*_est - p_c|| is unbiased to
within 0.03 m out to rho = 0.5, then inflates systematically (+0.13 m at rho = 0.6,
+0.33 m at 0.7 on the saddle; +0.10, +0.26 on the vortex) with fast-growing spread,
as the formation reaches the edge of the reconstructed region. An inflated radius
estimate makes the radial controller pull inward, which reproduces the archived
orbital columns going negative exactly over that range (vortex_nn -0.019 at 0.6,
-0.067 at 0.7; saddle_nn -0.049, -0.108; the paper's tables stop at 0.50 where the
effect is just beginning). Suggested replacement for the "It is suspected ..."
sentence:

```
AFTER: The negative bias at the largest commanded radii traces to degradation of
the reconstructed field near the edge of the mapped region: there the estimated
critical point distance is systematically inflated, so the radial controller pulls
the cluster inside the commanded radius. This is the same range-dependent error
amplification captured by kappa(J): far from the critical point the sampled patch
approaches uniform flow and small field errors produce large estimate shifts.
```

## 7. Literature nuance (Phase 3)

Method: abstracts and full-text keyword sweeps of the priority prior art (Brinon-
Arranz 2019 [3], Adamek 2014 [7], Michini 2014 [34], Kularatne 2017 [12], Kitts 2018
[4], Knizhnik 2022 [42], Mas/Kitts entrapment (uncited), Cochran-Krstic source
seeking (uncited), Waight IDETC 2025 [33]) extracted via PyMuPDF, checked against the
draft's specific claims.

### 7.1 The novelty claims hold

- **Line 58, "None recovers the coordinate of a critical point together with its
  identity":** verified against the cited camps. Michini [34] and Kularatne [12]
  track manifolds with local sensing (three robots in [34]) but never localize the
  saddle itself or classify types. Knizhnik [42] orbits gyres but "relies only on the
  robot's location relative to the center of the gyre": the center is given, not
  estimated from the flow. Brinon-Arranz [3] estimates gradient and Hessian of a
  SCALAR signal. The claim as worded survives.
- **Self-overlap with [33] (author's own testbed paper): none.** [33] contains zero
  occurrences of "critical point", validated only the vector-sum and vector-to-scalar
  primitives, and explicitly states that orbiting a center at fixed radius "remains
  an ongoing research challenge that this testbed is uniquely positioned to address."
  This paper is the direct answer to that sentence; citing it as such in the
  Introduction would sharpen the continuity claim at no cost.
- The specific combination claimed (formation-sampled vector measurements -> assembled
  Jacobian -> critical point coordinate + eigenvalue identity + standoff orbit of the
  estimated point) has no precedent in the surveyed folder. Each ingredient exists
  separately: formation derivative estimation [3], [7]; scalar saddle station-keeping
  [4]; orbiting known targets or gyres (Mas/Kitts entrapment, [42], Cochran-Krstic).

### 7.2 Missing citations worth adding

- **Mas and Kitts, entrapment/escorting/patrolling (cluster space, 3-robot hardware):**
  the closest orbiting precedent from the authors' own lab, currently uncited. One
  sentence in Sec. II-D or the Discussion: "Cluster-space patrolling around a known
  or externally tracked target was demonstrated in [Mas]; here the orbit center is
  estimated online from the field itself." Add a bibitem.
- **Cochran and Krstic, nonholonomic source seeking (orbit-like attractor around a
  scalar source):** optional, one clause where orbital control related work is
  discussed; it is the single-vehicle scalar-field analogue of standoff behavior
  without position information.

### 7.3 Places the paper undersells itself

- **Exactness.** Gradient/Hessian estimators in [3] and [7] carry Taylor-truncation
  bias controlled by formation radius; [3] devotes analysis to bounding it. This
  paper's estimator is EXACT on affine fields at any formation scale, with bias
  O(rho_formation^2) times the field's second derivatives on smooth nonlinear fields.
  Sec. II-B never says this; one sentence claims a concrete advantage over the
  scalar-field state of the art.
- **Error bounds.** [3] gives explicit noise-propagation bounds and uses them to
  optimally size the formation. A reviewer who knows [3] will ask for the analogue.
  The audit's first-order sensitivity law (Sec. 3, E11) is exactly that analogue and
  is already validated; adding it (with [3] cited as the scalar-field counterpart)
  pre-empts the request and quantifies the kappa(A)/kappa(J) discussion.
- **Field-independence.** The audit-verified bit-identical noise-free orbits across
  all six field types (E7) is a property no flow-riding method ([42], drifters) can
  have, since those depend on the ambient field structure by construction. Worth
  stating as a designed property, not an observation.
- **Minimum-robots consistency with [3]:** gradient estimation of one scalar in 2D
  also needs three non-collinear samplers; the paper's six-unknown counting argument
  is the vector-field generalization. A clause noting the consistency strengthens
  the minimality contribution.

### 7.4 Overstatement risks

- Contribution 1's "and types" remains the main exposure (Sec. 4.1): novel as a
  capability, unexercised as an experiment. Fix by demonstration or by wording.
- Line 66 contribution 4 reads as if hardware covered the method broadly; hardware
  covers vortex and saddle (well justified at line 397). Tightening to "...in 169
  hardware experiments on vortex and saddle fields" costs nothing and disarms a
  standard reviewer complaint.
- **Fig. 6 (testbed photo, line 364):** the image file is `testbed_with_four.png` and
  the photo appears to show four Decabots on the map, while the caption says "three
  Decabot rovers" and the paper is a 3-robot paper. Replace with a three-robot photo
  or reword the caption; as it stands it invites a "which is it?" query. (This is the
  only 4-robot remnant found; the body text is clean.)

### 7.5 Over-proven material and T-Mech length fit

- **Appendix B:** Eqs. 21-32 are standard cluster-space kinematics; the canonical
  citation is Kitts and Mas 2009 [46]. Eqs. 25-26 (vertex angles alpha, gamma) are
  never used anywhere in the paper and can be cut outright. Recommended shape: keep
  the SAS definition, the centroid equation, and the 6x6 Jacobian structure (Eq. 33),
  compress the rest to "derived in the standard way [46]" after applying fix B1.
  Frees roughly half a page.
- **Appendix A:** the six field equations could collapse into one table with columns
  (field, v(x,y), J, eigenvalues). That both saves space and doubles as the
  ground-truth table for a type-classification experiment (Sec. 4.1), tying the
  appendix to contribution 1.
- The body is otherwise appropriately lean for T-Mech: Sec. III defers kinematics
  correctly, and Eqs. 12-13 are the right level of detail for the robot layer.

## 8. Housekeeping

Uncommitted work now includes the Eq. 11 fix, the code comments, and this report.
Per repo policy, commit at the end of the session (backup first, clean history
second).
