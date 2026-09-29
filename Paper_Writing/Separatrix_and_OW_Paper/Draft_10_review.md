# Referee report (Reviewer 1): Draft_10

**Paper:** Second-Order Cooperative Field Estimation for Multirobot Tracking of Coherent
Flow Structures
**Venue:** IEEE Systems Journal. **Recommendation:** Major Revision (one item, T1; the rest
is minor).
**Technical accuracy:** 7.5/10. The derivations hold. The ocean trial's map does not match
its description (T1), and the objectivity result is one trial (T2). All eight figures and
Table I are referenced.

Sections: II-E Double-Gyre Example (with the estimator checks), III-B D tracker, III-C s1
tracker, IV-B Clean Runs, IV-C Behavior Under Noise, IV-D Rotating Observer, V Ocean, VI
Limitations and Future Work, VII Conclusion, Appendix A equivariance, Appendix B stability.

## 1. Summary

**Gap.** Robotic trackers of flow structures either start straddling a structure whose
identity is given [26-29] or compute FTLE on board over a finite horizon [30]. None
acquires a structure it does not already occupy from instantaneous data.

**Thesis.** A linear fit [8] gives the velocity gradient at one point. A second-order fit
gives it as a field over the formation, so any scalar built from it comes with a
closed-form gradient.

| Contribution | Fills the gap? | Evidence |
|---|---|---|
| (a) Six-robot estimator, minimal, fails only on a common conic | Enables it | II-B counting argument, II-C conditioning (kappa 8.26, 564 near-conic), II-E checks at 10^4 draws |
| (b) Surrogates D and s1 in closed form, noise characterized | Yes | II-D derivations; II-E confirms D-hat unbiasedness and the s1 bias (18 of 24 within 5%) |
| (c) Two trackers that acquire and ride without pre-straddling | Yes, on the benchmark | 6/6 matched starts (IV-B), noise sweep (IV-C), one ocean start (V, see T1) |
| (d) Noise versus objectivity tradeoff | Half | Noise side at 10^4 trials per point, both trackers in Fig. 6. Objectivity side is one trial (IV-D) |

**Bonus contributions.**
1. Objectivity plus traversal leaves the s1 tracker only its own prior output as a sign
   reference (III-C). The most interesting idea in the paper, though the evidence now
   shown for it is thin (T3).
2. One s1 tracker rides an attracting and a repelling OECS in one pass, the tangent
   swapping eigenvectors at the isotropic point, all in closed form in II-E.
3. A quadratic fit supports no definiteness certificate for either surrogate, which is
   why the s1 capture test is built on the gradient (III-C). This generalizes beyond the
   paper.
4. The D tracker's noise cliff (0.0077) sits inside the band where grad D-hat stops being
   informative along the separatrix (0.0047 to 0.0085), locating the failure in the
   estimator.

## 2. New gap and future work

**Stated (VI):** time-varying stability, online reshaping, a ten-robot cubic fit, Decabot then
surface vessels. The cubic fit is the natural next step. It recovers the third derivatives the
true H_D depends on (T4).

**Open gap.** A tracker that is objective and noise-robust. Filtering the carried sign over
time, or confirming it at the deeper s1 well, are the obvious candidates.

**Missing limitations.**
- The s1 tracker is objective except at its seed (Appendix A), yet the Conclusion's
  selection rule calls it objective without the critical-rate caveat.
- The ocean result rests on one start and one gain, and V-A says the gain decides which
  ridge branch the cluster meets. No robustness evidence over starts or gains is shown.

**Quick wins.**
1. Sweep Omega through Appendix A's predicted critical rate, 0.238, for both trackers.
2. Run the D tracker in the rotating frame with the inertial flow sign on w1 (T2).
3. Backward FTLE for the attracting-structure comparison in V.
4. One sentence in II-D stating what H_D-hat omits (T4).

## 3. Narrative

Abstract, Introduction, and Conclusion agree. Each leads with the fit, presents the two
trackers as uses of it, and ends on the tradeoff.

| Claim | Check |
|---|---|
| About 4x noise tolerance | Matches Fig. 6 (0.0077 against 0.0019, ratio 4.1; 3.5 under position noise) |
| Final gaps 1.219 and 0.025 | Match IV-D. "Leaves the structure" misdescribes Fig. 7(b) (T2) |
| 1.8 and 2.4 km, 28 h | Match V, but produced under a different map than V describes (T1) |
| 0.87 m/s | sqrt(2)(1.8)(0.04) x 51.4 km / 6000 s = 0.872. Holds north-south only under the map actually used (T1) |
| kappa 8.26, critical rate 0.238, margin 1.19 | Recomputed, correct |
| 0.41 vs 0.44 correlation | Prediction is 0.447 at (-0.3, 0.1). Correct |

**Overclaims.**
- "Separatrices run along the trenches of D inside the strain-dominated regions" (II-D)
  is stated generally, while the end of II-D says neither field is guaranteed.
- "Along which floating material collects" (Abstract) describes attracting structures. The
  ocean paths are scored against forward-FTLE ridges, which are repelling.
- "The D tracker makes landfall on the middle island" (V-B). In Fig. 8 it ends at
  (34.09 N, 120.23 W), about 6 km off Santa Rosa Island's north coast.
- "Mapping these exact curves allows teams to... [11,12,13]" (I). [10] is the search paper.
  Check that [11-13] concern coherent structures. Likewise [24] as "refined for
  oceanographic use."

## 4. Technical accuracy

Checked by hand and correct: the quadratic model, J-hat, the H_D-hat entries (as the Hessian
of D-hat), the D-hat unbiasedness pairing (Cov(a-hat, b-hat) is c Phi^-1 Phi^-T, symmetric,
so every antisymmetric pair cancels), grad s1-hat, concavity of s1-hat, the s1 bias,
D' = D + Omega omega + Omega^2 for the field as written, every II-E closed form, the
separatrix reduction grad s1 = ±(a5, a4) (it uses b4 = -a5, b6 = -a4 from incompressibility),
the Appendix A margins, and the Appendix B bound delta/a_perp.

**T1. The ocean trial was run on a different map than V-A describes.** V-A states that
longitude is scaled by cos(34.2°), so the map is a uniform dilation (51.4 km per unit on both
axes). Fig. 8 and the 1.8 and 2.4 km distances reproduce only with the same degree scale on
both axes, 51.4 km north and 42.5 km east (config `isotropic_map: false`). Rerun from the
same start under the map V-A describes, the D tracker still rides the ridge and reaches the
middle island's north coast, but the s1 tracker turns east along the mainland coast instead of
descending through the channel entrance. The asymmetry is expected. det J is invariant under any linear change of
coordinates, since det(A J A^-1) = det J, but its gradient, the height-ridge frame, and every
strain quantity are not, and s1 is built entirely from strain. Section V's headline claim
("both follow the same dominant transport corridor", repeated in the Abstract and
Conclusion) therefore holds only under a map the paper says it did not use. Either report
the isotropic-map result, with a start that exercises both trackers, or describe the map
used and drop the claim that the trenches and eigenframe are physical.

**T2. The objectivity result is one trial, and Fig. 7(b) contradicts its caption.** One
start, Omega = 0.2, 16% below the s1 tracker's own critical rate. In Fig. 7(b) the
rotating-frame D run follows the separatrix to p1* and then takes the opposite wall branch.
The 1.219 gap is that branch choice, a sign decision on w1 that the flow's transport term sets
every cycle. IV-D names both mechanisms (landscape and transport). The caption credits
(D_not_objective) alone and says the run "leaves the separatrix."

**T3. One claim has no evidence.** III-C states that noise robustness "depends less on
per-cycle estimation accuracy than on how long a sign decision persists." IV-C shows only
that a clean gradient and eigenframe restore 100% at sigma_uv = 0.002. That locates the
failure in tangent selection. It does not compare persistence with per-cycle accuracy. Give
it one number or state it as a hypothesis.

**T4. The fitted H_D is never said to omit third-derivative terms.** H_D-hat is the Hessian
of D-hat. The true Hessian of det J adds products of J with third derivatives, which a
quadratic fit drops at any radius. III-B and IV-B imply this ("truncation error sets the
sign"), and future work (c) depends on it, but II-D presents H_D-hat as "the Hessian." No
result changes; one sentence where (hess_det) is introduced fixes it.

**T5. Smaller points.**
- The sigma_eff check (II-E, 1.6%) cannot test the approximation. |grad u| = |grad v|
  everywhere on the double gyre, the case where (sigma_eff) is exact.
- The added term +Omega(-y', x') is the apparent flow of an observer rotating at -Omega.
  D' is correct for the field as written, but the text calls it rotation at Omega.

**Test plan versus results.** "0 to 35 steps" to acquire has no acquisition criterion. Appendix A's critical rate is never tested.

## 5. What I don't like

- **Notation collisions.** n is the lag's time index (III-A) and the transverse coordinate
  (III-B). r is strain magnitude and, in IV-D, a radius ("Omega r <= 0.10").
- **Contradiction.** II-D's general "separatrices run along trenches of D" against its own
  "neither field is guaranteed."
- **Figures.** Fig. 4 labels "half" where the text says "segment," and panel (b) draws the
  ride along +y while the flow on x = 0 runs -y. Fig. 7's caption says the D run "leaves the separatrix" (T2).
- **Continuity.** IV-D's "the condition the appendix requires" should name Appendix A.
  "Formation collapse" (IV-C) is never defined.
- **References.** [35] lacks year and URL. [16] (2015) backs a claim about both fields,
  but OECS [17] came later.
- **Small.** "Alternately" for "Alternatively" (II-B).

## 6. Writing

About 330 sentences and 6,600 words of body text (equations, floats, and contribution list
excluded). Mean sentence about 20 words. Flattest cadence: Position Noise (CV 0.17),
Formation (0.22), Eulerian Surrogates (0.27). Longest mean sentences: VI (26.1), the
Introduction (24.6), III-C (22.1).

**AI-isms.** Nearly clean on vocabulary (one "Furthermore," one "Additionally"). The tells
are structural. Math objects take agency ("the ride carries the cluster across," "both
terms of the D tracker are steered by a frame artifact"), and claims are chained by ", so."
Section I paragraphs 3 and 4 are looser and more promotional ("strategically deploy
resources," "spewing vortices," doubled spaces). The seam is audible at the start of II-D.

| Section | Grade | Why |
|---|---|---|
| Abstract | B+ | One arc. "Floating material collects" overclaims; clunky first sentence |
| I | B- | Concrete contributions. Promotional paragraph 3, long related-work paragraph |
| II-A, II-B | A-, B+ | Best writing in the paper. "Alternately" |
| II-C | B | Accurate. Position Noise is monotone, noise algebra in prose |
| II-D | B+ | One job per paragraph, checkable. The general D claim |
| II-E | B+ | Closed forms are clean. The closing checks paragraph packs four numbers into five sentences but each earns its place |
| III-A | B+ | Clear three-layer account, robot model in one paragraph |
| III-B | B | Trench frame, band, and capture follow in order. Dense |
| III-C | B- | Longest controller subsection. The sign-persistence paragraph is a results claim placed before any result (T3) |
| IV-A, IV-B | A-, B+ | Short and tight. The terminal split reads cleanly |
| IV-C | B | Setup is one long paragraph with a semicolon splice; the result paragraph is short and checkable |
| IV-D | B | Tight. "Leaves the structure" (T2) |
| V-A, V-B | B, C+ | V-A is a dense but honest unit block. V-B reads well but rests on T1 |
| VI, VII | B, B | Honest and short |
| Appendices | B- | Followable; the seed exemption is the best-stated caveat in the paper |

## 7. Housekeeping and arc

- Put the H_D-hat caveat (T4) in II-D, where the Hessian is introduced.
- The sign-persistence paragraph in III-C either needs its evidence in IV-C or should be
  rephrased as design rationale (why t_ref is the only admissible reference) without the
  robustness claim.
- V needs its map reconciled (T1) before its numbers reach the Abstract and Conclusion.

## 8. Ideas most likely to outlive the paper

1. A second-order vector fit makes any velocity-gradient scalar a navigable surface with a
   closed-form gradient. Six robots off any common conic for second order, ten off any
   common cubic for third. A ring fails at any size.
2. A quadratic fit supports gradient tests but no curvature certificate, so terminal logic
   on fitted surrogates must be gradient-based.
3. Objectivity plus traversal forces sign memory.

## 9. Novelty

The formation is that of [4], and the trackers are trench following [2] on derived scalars.
What is new is fitting both velocity components to second order so that gradient-derived
diagnostics carry their own gradients, and acquiring a structure from instantaneous data. A
search of the 133-paper reference corpus found no robotic tracking of OECS or of
rate-of-strain eigenvalues; the closest work is the straddle family [26-28] and onboard
FTLE [30].

## 10. Recommendation: Major Revision

The headline claim, a closed-form gradient from a second-order fit, is well supported, and
the double-gyre results are candid about where each tracker fails. T1 alone moves this to
major: the ocean trial is one of three result sections, it feeds the Abstract and the
Conclusion, and under the map the paper describes, one of the two trackers does not follow
the corridor. If T1 is resolved by rerunning under the stated map and reporting honestly,
the remaining items (T2 to T5, Section 5) are minor.
