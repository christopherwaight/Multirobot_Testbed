# Referee report (Reviewer 1): Draft_10

**Paper:** Second-Order Cooperative Field Estimation for Multirobot Tracking of Coherent
Flow Structures
**Venue:** IEEE Systems Journal. **Recommendation:** Minor Revision.
**Technical accuracy:** 8.5/10. The derivations hold. One bound carries a spurious gain
(T1), and the objectivity result is a single trial (T2). All nine figures and Table I
are referenced.

## 1. Summary

**Gap.** Robotic trackers of flow structures either start straddling a structure whose
identity is given [26-29] or compute FTLE on board over a finite horizon [30]. None
acquires a structure it does not already occupy from instantaneous data.

**Thesis.** A linear fit [8] gives the velocity gradient at one point. A second-order fit
gives it as a field over the formation, so any scalar built from it comes with a
closed-form gradient.

| Contribution | Fills the gap? | Evidence |
|---|---|---|
| (a) Six-robot estimator, minimal, fails only on a common conic | Enables it | II-B counting argument, II-C conditioning (kappa 8.26, 564 near-conic), IV-C at 10^4 draws |
| (b) Surrogates D and s1 in closed form, noise characterized | Yes | II-D derivations, D-hat unbiasedness and s1 bias confirmed in IV-C |
| (c) Two trackers that acquire and ride without pre-straddling | Yes, on the benchmark | 6/6 matched starts (IV-D), 20,000-cell start grid, one ocean start (IV-G) |
| (d) Noise versus objectivity tradeoff | Half | Noise side at 10^4 trials per cell (IV-E). Objectivity side is one trial (IV-F) |

**Bonus contributions.**
1. Objectivity plus traversal leaves the s1 tracker only its own prior output as a sign
   reference, and that memory, not per-cycle accuracy, sets its noise robustness. The
   ablation (clean seed 44.6% to 45.8%, clean argmax to 100%) is the evidence. The most
   interesting idea in the paper.
2. One s1 tracker rides an attracting and a repelling OECS in one pass, with the tangent
   swapping eigenvectors at the isotropic point, all closed form in II-E.
3. The D tracker's noise cliff (0.0079) sits inside the band where grad D-hat stops being
   informative along the separatrix (0.0047 to 0.0085), locating the failure in the estimator.

## 2. New gap and future work

**Stated:** time-varying stability, online reshaping, a ten-robot cubic fit, Decabot then
surface vessels. The cubic fit is the natural next step. It recovers the third derivatives the
true H_D depends on (T4).

**Open gap.** A tracker that is objective and noise-robust. Filtering the carried sign over
time, or confirming it at the deeper s1 well, are the obvious candidates.

**Missing limitations.**
- The s1 tracker is objective except at its seed (Appendix), yet the Conclusion's selection
  rule calls it objective without the critical-rate caveat.
- About 11 km inter-robot links at rho = 5.4 km make "ample bandwidth" an assumption.

**Quick wins.**
1. One sentence in II-D stating what H_D-hat omits (T4), cited where IV-C reports 0.94.
2. Sweep Omega through the appendix's predicted critical rate, 0.238, for both trackers.
3. Run the D tracker in the rotating frame with the inertial flow sign on w1 (T2).
4. Plot both trackers' success curves on one axis. The D curve exists but is in no figure.
5. Backward FTLE for the attracting-structure comparison in IV-G.

## 3. Narrative

Abstract, Introduction, and Conclusion agree. Each leads with the fit, presents the two
trackers as uses of it, and ends on the tradeoff.

| Claim | Check |
|---|---|
| 4 to 5x noise tolerance | Matches IV-E (0.0079 against 0.0015 to 0.002, ratio 3.9 to 5.3). In no single figure |
| Final gaps 1.219 and 0.025 | Match IV-F. "Leaves the structure" misdescribes Fig. 8(b) (T2) |
| 1.8 and 2.4 km, 28 h | Match IV-G |
| 0.87 m/s | sqrt(2)(1.8)(0.04) x 51.4 km / 6000 s = 0.872. Correct |
| kappa 8.26, Ro 4.9, 18.3/19.2, 2.5%, critical rate 0.238, margin 1.19 | Recomputed, all correct |
| 32° to 40°, 0.0047 to 0.0085 | Match `verify_estimator_separatrix.py` output |

**Overclaims.**
- "Separatrices run along the trenches of D inside the strain-dominated regions" (II-D.1)
  is stated generally. II-D.3 says neither field is guaranteed.
- "Along which floating material collects" (Abstract) describes attracting structures. The
  ocean paths are scored against forward-FTLE ridges, which are repelling.
- "Acquisition does not depend on the start" (IV-D). The evidence is that no run of the
  20,000-cell grid settles at a gyre center, which is weaker. Which tracker, and how many
  acquired?
- "The D tracker reaching the 0.87 m/s cap exactly" (IV-G). tanh saturation never reaches
  its cap.
- "Mapping these exact curves allows teams to... [11,12,13]" (I). [10] is the search paper.
  Check that [11-13] concern coherent structures. Likewise [24] as "refined for
  oceanographic use."

## 4. Technical accuracy

Checked by hand and correct: the quadratic model, J-hat, the H_D-hat entries (as the Hessian
of D-hat), the D-hat unbiasedness pairing (Cov(a-hat, b-hat) is c Phi^-1 Phi^-T, symmetric,
so every antisymmetric pair cancels), grad s1-hat, concavity of s1-hat, the s1 bias,
D' = D + Omega omega + Omega^2 for the field as written, every II-E closed form, the
position-noise correlation (0.447 at the test point), and the appendix margins.

**T1. The practical-stability bound carries a spurious k.** III-B gives
limsup |n| ≈ delta/(k a_perp). Estimation error enters the navigation controller's output,
which k then scales, so n-dot = -k(a_perp n - delta) and the k cancels. The bound is
delta/a_perp. The rho sweep in IV-E (error set by truncation, independent of gain) is
consistent with the corrected form.

**T2. The objectivity result is one trial, and Fig. 8(b) contradicts its caption.** One
start, Omega = 0.2, 16% below the s1 tracker's own critical rate. In Fig. 8(b) the
rotating-frame D run follows the separatrix to p1* and then takes the opposite wall branch.
The 1.219 gap is that branch choice, a sign decision on w1, which the flow's transport term
sets every cycle. IV-F names both mechanisms (landscape and transport), then concludes "the
artifact is in the surrogate," and the caption credits (eq. D_not_objective) alone.

**T3. Loose ends in IV-G.** The jitter behind "84 of 100" is unstated. Where 2|mu|/r > 2
(90th percentile 3.0 to 5.9) s1 and s2 share a sign, so there is no strain saddle there at
all, which is stronger than "not tightly bounded." The branch outcome depends on k.

**T4. The fitted H_D is never said to omit third-derivative terms.** H_D-hat is the
Hessian of D-hat. The true Hessian of det J adds products of J with third derivatives,
which a quadratic fit drops at any radius. IV-D says this at the saddle, and III-B calls the
trench normal "an approximation in general," but II-D presents H_D-hat as "the Hessian."
It is the unstated cause of IV-C's 0.94 error floor at zero noise (at (-0.3, 0.1) the fit is
about diag(4.8, -4.8) against a true diag(-5.9, -15.6)). IV-C's "signed mean error of
H_D-hat zero" must then be relative to the noise-free fit. No claim changes. The headline
gradient needs only first and second derivatives, and on the benchmark the fitted
eigenframe stays on the axes with the transverse curvature positive.

**T5. Smaller points.**
- The sigma_eff check (1.6%) cannot test its approximation. |grad u| = |grad v| everywhere
  on the double gyre, the case where (eq. sigma_eff) is exact.
- The added term +Omega(-y', x') is the apparent flow of an observer rotating at -Omega.
  D' is correct for the field as written, but the text calls it rotation at Omega.
- Cruise speed is c_max tanh(1) in III-C paragraph 4 and k c_max tanh 1 in the transverse
  argument.
- Limitations places both s1 weaknesses "where S approaches isotropy." The argmax failure
  of IV-E happens at y = 0.35 (r = 0.88) and worsens toward the saddles, where S is most
  anisotropic.
- IV-E "a sign inversion" explains a floor near 50%, not success falling to 0% by
  sigma_uv = 0.004 (Fig. 7). Straddle retention also falls to 0, so the structure is lost.

**Test plan versus results.** Fig. 7(b), position noise, is never discussed. Straddle
retention is plotted, not quantified (23% against 44% success at 0.002). Tracking error
versus noise is recorded, not reported. The rho sweep has no numbers for success. The
20,000-cell grid has no setup (tracker, run length). "0 to 35 steps" to acquire has no
acquisition criterion. The appendix's critical rate is never tested.

## 5. What I don't like

- **Notation collisions.** beta is the ride flag (III-C) and a SAS angle (III-A). n is the
  lag's time index (III-A) and the transverse coordinate (III-B). r is strain magnitude and,
  in IV-F, a radius. k is the gain and the eigen-index in w_k, lambda_k. m is the
  measurement vector and a speed bound.
- **Contradiction.** II-D.1's general "separatrices run along trenches of D" against
  II-D.3's "neither field is guaranteed."
- **Mode logic scattered.** The s1 tracker's beta transitions are split across the latch
  paragraph and the hysteresis sentence. The release from capture appears only in the
  latter.
- **Figures.** Fig. 4 labels "half" where the text says "segment," and draws the ride
  along +y while flow on x = 0 runs -y. Fig. 6 axes say meters. Fig. 7's "both conditioned
  on reaching p1*" would make success 100% by definition. The Fig. 9 title shows the code
  variable TIME_WARP=6000x. Fig. 7 is the s1 tracker only, though the text says both swept
  the grid.
- **References.** [36] lacks year and URL. [16] (2015) backs a claim about both fields,
  but OECS [17] came later.
- **Small.** "Alternately" for "Alternatively" (II-B). IV-C's "eigenvalues are negative
  semidefinite" (a matrix is semidefinite). Undefined "formation collapse" (IV-E).

## 6. Writing statistics

399 sentences, 7,972 words of body text (equations, floats, and contribution list
excluded). Mean sentence 20.2 words, burstiness (CV) 0.47, longest 74 words (the
Conclusion's future-work list). MATTR(100) 0.68, high for technical prose. Top content
words: field 51, tracker 41, trench 41, gradient 40, flow 38, noise 35. Top trigram "the
[D/s1] tracker" (36). Top 4-gram "a second-order fit of" (3), the thesis restated on
purpose.

Fingerprints: ", so" 49, "same" 38, "against" 25 (every numeric comparison), "carr-" 19
(any property of a quantity). Math objects take agency: "the band test hands the speed to
the flow," "the ride carries it," "costs nothing." Flattest cadence: Position Noise 0.17,
Formation 0.23, Eulerian Surrogates 0.27, Estimator Accuracy and rotating observer 0.29.
Estimator Accuracy also has the longest mean sentence, 27.6 words. The Introduction varies
most outside the Conclusion. Tell vocabulary is nearly clean (one "Furthermore," one
"furthermore," one "Additionally").

**Register.** Section I paragraphs 3 and 4 are looser and more promotional ("strategically
deploy resources," "spewing vortices," doubled spaces). III-B through IV-F are compressed,
with claims chained by ", so." The seam is audible at the start of II-D.

## 7. Grades

| Section | Grade | Why |
|---|---|---|
| Abstract | B+ | One arc, no overclaim except "floating material collects." Clunky first sentence |
| I | B- | Concrete contributions. Promotional paragraph 3, long related-work paragraph |
| II-A, II-B | A-, B+ | Best writing in the paper. "Alternately" |
| II-C | B | Accurate. Position Noise is monotone, noise algebra in prose |
| II-D, II-E | B+ | One job per paragraph, checkable. The D overclaim in II-D.1 |
| III-A | B+ | Clear three-layer account, robot model in one paragraph |
| III-B | C+ | Trench glossed, but the convergence paragraph is one dense block with a wrong bound |
| III-C | B- | Latch described as it runs. Mode logic split, beta collision |
| IV-A, IV-B | B, B- | Checkable setup. IV-B mixes justification with parameters |
| IV-C | C+ | Longest sentences in the paper, numbers without causes (T4) |
| IV-D | B- | Explains the terminal split well. Acquisition overclaim |
| IV-E | C+ | Places its cliffs against prior work, but packs the sign-memory argument into results |
| IV-F, IV-G | B-, B- | Clear. IV-F contradicts itself on mechanism, IV-G hedged |
| IV-H, Conclusion | B, B | Honest. 74-word future-work sentence |
| Appendix | B- | Followable, the seed exemption is the best-stated caveat in the paper |

## 8. Ideas most likely to outlive the paper

1. A second-order vector fit makes any velocity-gradient scalar a navigable surface with a
   closed-form gradient. Six robots off any common conic for second order, ten off any
   common cubic for third. A ring fails at any size.
2. Objectivity plus traversal forces sign memory, and the memory sets noise robustness.

## 9. Novelty and authorship

**Novelty.** The formation is that of [4], and the trackers are trench following [2] on
derived scalars. What is new is fitting both velocity components to second order so that
gradient-derived diagnostics carry their own gradients, and acquiring a structure from
instantaneous data. Cluster tracking of an OECS appears new, unchecked against the corpus
(the paper-search server was unavailable for this review).

**AI.** In part. Sections I and II-A through II-C have human texture. III-B onward is
compressed, with math objects given agency and reasons attached by ", so." Told the whole
paper came from a strong model, I would believe it for III-B onward. Unpack each
compressed claim into a number or mechanism, or cut it. Check IEEE's current AI-disclosure
rules.

## 10. Reordering and additions

Put the H_D-hat caveat (T4) in II-D, where the Hessian is introduced. Move the sign-memory
paragraph out of IV-E into its own subsection. Merge the s1 mode logic (latch, capture,
release) into one III-C paragraph. Add one figure with both trackers' noise curves.

## 11. Recommendation: Minor Revision

The headline claim, a closed-form gradient from a second-order fit, is the best-supported
claim in the paper, and the paper is candid about where its trackers fail. T1 is a
one-symbol fix, T4 one sentence. T2, T3, T5, and Section 5 are rewriting. The Omega
sweep and flow-sign ablation would strengthen the objectivity finding but are not required
for a secondary result.
