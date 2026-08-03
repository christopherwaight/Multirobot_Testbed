# Draft_6a technical audit

Worked against `draft6a_technical_callouts.html`. Every item was checked by running the
actual estimator and controller code in `src/control/pentagon_primitives.py`, not by
re-reading the algebra. Verification scripts are in the session scratchpad; the numbers
below are reproducible from the repo.

**Verdict: 10 of 11 callouts real, 1 wrong. Two of the real ones were understated, and
chasing one of them turned up a mechanism the paper had attributed to the wrong cause.**

---

## The one that was wrong

**T5, sign error in the frame-transport term. No defect.** For `p' = Q(t)p` with `Q` a
rotation through `phi`, direct computation gives `Q_dot Q^T = Omega [[0,-1],[1,0]]`,
which is exactly what II-C prints, and it yields `omega' = omega + 2Omega`, matching (6)
and the experiment. The callout asserts the typeset convention gives `omega - 2Omega`.
It does not. No change made.

---

## The two headline findings

### T2. The s1 sign inversion is the argmax, not the seed

VI-C attributed the directional failure to the one-time flow read that seeds
`t_ref`, using a signal-to-noise argument built on the 0.12 differential-flow span
across the six sample points. That span is real (measured 0.1195) but it is not what
the sign decision consumes. The fit interpolates at the centre robot, so `v0_hat` is
that robot's reading exactly, with noise `sigma_uv` at `gamma_0 = 1`, giving SNR 71 at
`sigma_uv = 0.002`.

Causal isolation, running the real loop with one channel at a time fed clean values:

| sigma_uv | baseline | clean SEED | clean ARGMAX |
|---|---|---|---|
| 0.0015 | 73.7% | 75.0% | 100.0% |
| 0.0020 | 42.7% | 45.0% | 100.0% |
| 0.0030 | 6.0% | 6.7% | 99.3% |
| 0.0050 | 0.3% | 0.3% | 95.7% |

The seed is not the mechanism. The argmax in (30) is. On the separatrix `s_s = 0` and
`mu = 0`, so (18) collapses to `grad s1 = +/-(a5, a4)` with the eigenvectors on the
coordinate axes, and the tangent test reduces to `|a5_hat| > |a4_hat|`. The true
`a5 = u_xx` is identically zero there, `a4` collapses at both saddles, and the decision
is made with the largest noise gain of the twelve (`gamma_2/rho^2 = 449.7`). SNR 1.4 at
`sigma_uv = 0.002`, fifty times worse than the seed.

Supporting evidence, all measured: zero one-step reversals at every level (the continuity
rule cannot reverse a tangent, as it returns `t . t_ref = |t_raw . t_ref| >= 0`);
off-tangent hop rates of 1.1% at 0.001 and 16.5% at 0.002 tracking the inversion rate;
and relocating the start to (0, 0.15) moves the 50% crossing from 0.0015-0.002 to
0.0025-0.003 with nothing else changed.

**Narrative effect: this strengthens VI-C's closing claim.** The thesis was that
robustness is set by how many commitments are unrevisable. That now rests on an algebraic
asymmetry rather than an SNR estimate that does not hold up. The D tracker re-signs `w1`
against `v0_hat` every cycle, so a bad frame costs it one cycle. The s1 tracker carries
the sign in state, and one hop through an orthogonal intermediate breaks the chain
without any single comparison being reversed. The sign is laundered, not flipped.

VI-G was also relocated. Three of its four listed alternatives address the seed, which is
not the exposure. Replaced with two that reach the argmax, both objectivity-preserving.

### T1 and T11 together, plus what they led to

**T1 confirmed in closed form.** On `x = 0` the six-point fit returns
`H_hat_xx = 2 pi^6 A^2 sin^2(pi y_f)` against a true `D_xx = 2 pi^6 A^2`, so the D
tracker's transverse rate is `a_perp = csc^2(pi y_f)`. Measured: 1.02 at the crest, 4.96
at y = 0.35, 44 at y = 0.45. Table IV's "a_perp = 1.0, monotone" was the crest value.
Past the monotone bound on 36.8% of each segment.

This also gives the floor `e` in closed form on the benchmark, `e = 2 pi^6 A^2 cos^2(pi
y_f)`, now (25). III-D previously left it as `O(1)`; it is 0% of `D_xx` at the crest and
98% at |y| = 0.45.

**T11 confirmed, and the cause found.** The "fitted kappa_perp = 4.0" comes from
`experiments/oecs_estimator_check.py:111`, which uses an ideal pentagon at ring phase 0.
The formation the sweeps actually fly (`pentagon_small.yaml`) sits at phase -90 and
returns 6.56 against analytic 6.888. Rotating the ring through its 72-degree period moves
the recovered slope between 4.0 and 9.7 about a mean of 6.83, and halving rho halves the
spread. So 4.0 is one sample of an orientation-dependent quantity whose mean is the
analytic value. Table IV's two s1 rows were not two measurements of one thing.

**What this led to, which no callout raised.** The orientation dependence is the m = 5
angular harmonic of the discarded cubic remainder: ring sums kill every harmonic except
multiples of the robot count, and the cubic remainder reaches the fifth against the
second-order rows of `Phi^-1`. It does not vanish on the structure, so it displaces the
equilibrium of the transverse channel rather than spreading it. Both trackers park where
their own transverse command vanishes, which is where the *fitted* trench lies rather
than the field's.

Predicting that offset from the open-loop fit alone, with no closed loop and no gains,
against the 10,000-trial zero-noise MC rows:

| | predicted | measured |
|---|---|---|
| D tracker | 0.0019 | 0.0016 |
| s1 tracker | 0.0079 | 0.0075 |

The offset vanishes at the two ring phases that make the pentagon mirror-symmetric about
x = 0 and peaks 18 degrees away. The measured zero-noise error carries exactly that
36-degree period (0.0006 to 0.0024 across ring phase for the D tracker).

**Narrative effect, and this is the significant one.** VI-D attributed the five-fold
tracking-error gap to the missing Newton normalization. That cannot be right: the
normalization is a denominator and cannot move the zero of a numerator. The gap is an
estimator effect. VI-D now says objectivity is paid twice, once at the controller in the
forbidden Newton normalization and once at the estimator in a fitted trench that sits
further from the true one. That is a sharper claim than the original and it is measured
rather than asserted.

**Also found: Table IV's discrete bounds ignored the actuator the simulation includes.**
Closing the identified momentum model around `-k a_perp n` gives a two-state map whose
determinant is `alpha_mom` for every gain, so the spectrum is complex with modulus
`sqrt(alpha_mom)` over a wide band. The instability threshold is
`dt k a_perp (1-alpha_mom) > (1+sqrt(alpha_mom))^2`, which at this operating point is
`dt k a_perp > 11.3`, not 2. Trajectories confirm no oscillation: at the worst
orientation the D tracker holds a smooth one-sided offset with two sign changes over a
300-step run at `dt k a_perp` above 12. Table IV's "oscillatory" and "limit cycle" labels
described nothing that happens. Table IV is rebuilt around the two closed forms and the
fraction of segment past each bound.

---

## The rest

| ID | Verdict | Disposition |
|---|---|---|
| **T3** | Real | III-A's defence of the unconstrained fit was philosophical. Replaced with the physical one, that HFR surface currents are not divergence-free, plus an explicit ledger of which results need `tr(J) = 0` (all surrogate interpretations, none of the estimator). |
| **T4** | Real | (41) defined `Ro = 2 Omega / |omega|_max`, the reciprocal of convention, so "Ro = 0.20" read as rotation-dominated when the opposite was meant. Inverted to `|omega|_max / 2 Omega = 4.9`. The two "O(Ro)" claims were also overstated: the frame term `Omega omega + Omega^2` is additive and unbounded relative to `D` at the crest, which the tracker must cross. Both restated. |
| **T6** | Real | Prop 4 proved command equivariance and hedged the path claim. Integrating an equivariant command in a rotating frame is not path-equivariant; the two differ by the transport velocity. `Omega ||p_Q|| < k c_max` is now an explicit hypothesis (38), with the residue bounded by the ISS argument. |
| **T7** | Real, quantified small | `Ocean_HFR.py` applies the same degree scale to both axes, so the map is not a similarity (20.9% anisotropy at 34.2 N). `det J` scales by the positive constant `Lx Ly`, verified on the record to 1e-12 relative, so D's trench geometry is exact. The strain eigenframe is not preserved: median 1.8 deg, max 25.4 deg over the corridor, with 0.087 spurious strain per unit vorticity. Stated in V-C with the fix named. The "51 km" was the latitude scale; east-west is 42.5 km. |
| **T8** | Real | "Superlinear local convergence" deleted. On the fitted surrogate the step is exact in one stroke; against the true field the rate is set by how well `lambda_2` matches the field's curvature, which by T1 it does not. IV-D proves exponential. |
| **T10a** | Real | (21) is an average over the two components, not an identity. Stated. |
| **T10b** | Real, and the callout's guess was wrong | The six conditions outside 5% are **not** the intermediate `r ~ sigma_r` regime. They are the `r/sigma_r >= 3` cases where the predicted bias is below the resolution of 1e4 draws. All twenty-four are within one standard error. Restated that way, which is stronger. |

### Found outside the callouts

- **(35) was missing the navigation gain `k`.** The transverse law was written
  `n_dot = -c_max tanh(...)` while (33) defines `p_c_dot = k delta`, and the discrete
  condition three paragraphs later does include `k`. Fixed in (35), (37) and the
  Lyapunov rate.
- **Future Work asserted an "s1 tracker discretization limit cycle."** No such limit
  cycle exists. Replaced with ring rotation to average the truncation bias, which is what
  the measurement actually motivates.
- **VI-B claimed descent "reaches the trench network in all 20,000 runs."** The basin
  grid records `dist_core`, `min_d_top`, `min_d_bot`, not network distance, so that
  specific claim is not backed by that file. What is verified: no run settles at a gyre
  core over those 20,000, and the separate 20,000-trial acquisition sweep reaches the
  network within 0.05 in 100% of cases. Sentence narrowed to the verified statement.

### Re-verified as correct, no change

Gain ladder (22) exact (measured row norms 1, 8.433, 449.74, 224.87). Prop 3 unbiasedness.
Straddle retention 1.000 / 0.589 / 0.389. Heading-quartile isotropy 75.9-77.1%. Basin
rates 47.4% (grid, n=20,000) and 47.1% (random, n=10,000). Acquisition 100% of 20,000.
Shear residual at 1e-4. `H_s1` eigenvalues negative semidefinite. Direction errors 4.5 /
44 / 41 degrees. All of Table II's arithmetic. Saddle `H_s1 = pi^4 A I = 9.74`. The
Okubo-Weiss diamond identity. The six flow saddles.

---

## Numbers I put into the .tex

Per your rule that numbers get reviewed before they enter the paper, here is everything
new, with provenance. Closed forms are preferred over measurements wherever both were
available, since a referee can check them by hand.

| Value | Where | Source |
|---|---|---|
| `a_perp,D = csc^2(pi y_f)` | Table IV, IV-D | Closed form, confirmed numerically |
| `H_hat_xx = 2 pi^6 A^2 sin^2`, `e = 2 pi^6 A^2 cos^2` | (25) | Closed form |
| 36.8%, 10.4%, 77.7%, 0% | Table IV | Integrated from the closed forms |
| 0.30, 1.46, 2.60, 2.92 | Table IV | Closed forms at the stated y |
| 6.6, 6.888, 4.0-9.7, mean 6.83 | VI-A | Direct fit at the live formation and over ring phase |
| 71, 1.4, 449.7 | VI-C | Analytic SNR from (22) |
| 42.7 / 45.0 / 100 / 95.7% | VI-C | 300-trial isolation runs |
| 1.1%, 16.5% | VI-C | Instrumented hop counts |
| 0.0015-0.002 vs 0.0025-0.003 | VI-C | Relocation test, 300 trials/level |
| 0.0019, 0.0079 | VI-C | Open-loop equilibrium solve |
| 0.0006-0.0024, 36 deg period | VI-C | Existing 10,000-trial MC rows, re-binned |
| `(1+sqrt(alpha))^2`, 11.3 | IV-D, Table IV | Closed form from the 2x2 map |
| 20.9%, 51.4/42.5 km, 5.4/4.5 km, 0.087, 1.8/25.4 deg | V-C | Ocean map, measured on the 2 km record |
| 4.9 (Rossby) | (41) | Arithmetic |
| `[-0.72, 0.43]` vs `[18.3, 19.2]` | VI-B | `outputs/estimator_accuracy/eigvec_check.csv` |

---

## Status and the page count

Compiles clean, zero errors, zero undefined references, zero overfull boxes. **18 pages.**
You asked for 16.

Baseline before I started was 17. I added roughly 1000 net words of verified content and
cut roughly the same again from III-D, IV-B, IV-D, V-C, VI-B, VI-C, VI-D, Related Work
and Limitations, including folding the Selection Rule and Future Work subsections into
running text and removing the three-way duplication of the initialization and advection
comparison between Related Work, Table I and Limitations.

Getting from 18 to 16 needs about 2000 more words, which is 12% of the paper. That is no
longer trimming, it is deciding which argument to drop, and that is your call rather than
mine. The three candidates, in the order I would pick them:

1. **Cut the ocean experiment's start-sensitivity and FTLE-persistence detail** in VI-E
   and fold VI-E into VI-D. Roughly 0.5 page. Costs the weakest of the four validation
   conditions.
2. **Move the estimator sensitivity derivation (III-D) to an appendix**, keeping only the
   gain ladder, the floor `e`, and the orientation result in the body. Roughly 1 page.
   This is the structure Brinon-Arranz uses in the T-RO short paper, where the
   trigonometric machinery sits in an appendix and only the bounds appear in the body.
3. **Drop Table III's `sigma_p` columns** and report the D tracker's grid as a single
   `sigma_uv` column plus one sentence noting that `sigma_p = 0.01` and
   `sigma_uv = 0.01` give nearly identical rates, which the text already says. Roughly
   0.3 page. Costs a referee the ability to read the interaction directly.

I did not do any of these because each removes content rather than fixing an error, and
option 3 in particular touches data you review.

One thing I did not take you up on: you gave permission to remake figures. I checked
whether merging Figs. 4 and 5 would help and it would not. Both are wide, low-aspect
images at 0.34\textwidth, so stacking them at full column width would increase total area
from about 0.65 to 0.93 column-widths. Combining them buys only one caption.
