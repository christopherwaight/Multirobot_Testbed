# Review: *Multirobot Tracking of Separatrices and Objective Eulerian Coherent Structures*

**Draft 8a** | Reviewer assessment for IEEE Systems Journal

**Recommendation: Major revision.** The core contribution is real, the framing is
better than most papers in this space, and the analytic work I could check is
correct. What holds it back is a set of load-bearing statements that are either
undefined, unsupported, or contradicted elsewhere in the same paper. Almost all
of them are fixable with text edits plus two or three cheap simulation runs.

---

## Part 1: What I like

### 1.1 The strongest contribution (and it is not the one you lead with)

Your intro sells contribution (a), the estimator. The estimator is good, but it
is the least surprising thing here. **The strongest contribution is (d): you take
a tradeoff that everyone else states qualitatively (objectivity costs you
something) and you localize it to a single scalar comparison.**

Section IV-E is the best passage in the paper. You show that the `s1` tracker's
noise cliff is not an estimator accuracy problem at all, but a sign inversion in
one argmax where the true `a5` is identically zero and `a4` collapses at both
saddles, giving SNR 1.4 against the seed's 71. Then you do the channel
substitution ablation (clean flow moves it 44.6% to 45.8%, clean gradient moves
it to 100%) which nails the mechanism rather than asserting it. Then you derive a
threshold from your own gain ladder and recover 84.1%.

That is: observe a failure, predict the mechanism from theory, isolate it by
ablation, fix it with a derived rather than tuned threshold. Very few robotics
papers do all four. **Restructure the abstract and contribution list around this.**

### 1.2 The splitting identity as the organizing device

`D = ω²/4 − s1²` is an elegant spine. Two primitives reading two terms of one
identity off one fit is a genuinely clean framing, and it pays off three times:
it explains why the two agree on the double gyre (`ω ≡ 0` on the trench network),
it predicts exactly how they must diverge under rotation (the artifact lives in
the spin term), and it sets up the ocean result as a test of whether agreement
survives when the coincidence is removed. This is the kind of structure reviewers
remember.

### 1.3 The rotating-frame trial

A falsifiable discriminator with a predicted mechanism (7), a predicted magnitude,
and a matching outcome (0.025 versus 1.219). It is the cleanest experiment in the
paper and it is the one that justifies the OECS framing. Novel in this literature:
I am not aware of prior multirobot work that closes the loop on an objective
structure and then demonstrates frame invariance experimentally rather than
citing it.

### 1.4 The representability argument, used symmetrically

"Traversal and capture are the same representability limit read twice" is a
strong sentence and you earn it. The concavity of `ŝ1` (rigorous, correct, and
formation-independent) plus the traceless `Ĥ_D` symmetry together explain why one
tracker captures and the other does not, without appeal to tuning. Explaining a
negative result from the model class rather than the gains is the mark of a
paper that understands its own system.

### 1.5 The reachable-set estimate

47.4% over a 20,000-cell grid, with the explicit note that prior AN primitives
report local convergence without such an estimate. This is a real methodological
upgrade to how AN primitives are characterized, and it is currently buried in
one paragraph of IV-D. Promote it.

### 1.6 The robustness insight

"Robustness is set less by the per-cycle accuracy of what a law reads than by how
far a decision is carried before it is checked against the world." Followed by:
the sign recursion is the *structural residue* of the objectivity commitment,
because objectivity rules out the fitted curvature, the measured flow, and a
world axis in turn, leaving only the law's own prior output. That is a
generalizable claim about objective controllers, not a fact about your
controller. It deserves to be in the abstract.

### 1.7 Honesty that will earn you goodwill

- Reporting `2|µ|/r` median 0.8 to 1.1 on the ocean, which qualifies your own
  splitting-identity reading of the agreement.
- "Instantaneous criteria of this kind are not in general reliable indicators of
  the material transport skeleton [16]."
- Explaining that the `D` tracker's saddle traversal works by a symmetry of the
  field rather than a property of the estimator.
- Stating that no head-to-head against [26] or [30] was run, and why.

Keep all of it. Reviewers punish papers that hide this and reward papers that
front it.

### 1.8 The math checks out

I independently verified the following and all are correct:

| Claim | Status |
|---|---|
| `det(S+W) = det S + det W` via `tr(SW) = 0` | Correct (2x2 polarization identity) |
| `Q = ¼(∇·v)² − D` (quarter convention) and `Q = −4D` (incompressible) | Both correct |
| `D = −½π⁴A²(cos2πx_f + cos2πy_f)` | Correct |
| `∇D = π⁵A²(sin2πx_f, sin2πy_f)ᵀ`, `H_D = 2π⁶A² diag(...)` | Correct |
| `s1 = −π²A|cos πx_f cos πy_f|` | Correct |
| Transverse curvature `κ⊥ = π⁴A|cos πy_f|`, giving 6.888 at y = 0.25 | Correct |
| Eq. (9), all three entries of `Ĥ_D,0` | Correct |
| Traceless `Ĥ_D` from `u_xx = u_yy`, `v_xx = v_yy` | Correct, and the double gyre does satisfy it |
| Eq. (7): `D' = D + Ωω + Ω²` | Correct, and holds without the incompressibility assumption |
| `σ²_eff = σ²_uv + ½‖J‖²_F σ²_p` and the equal-norm caveat | Correct |
| Concavity of `ŝ1 = µ − ‖(s_n, s_s)‖` | Correct |
| Ocean unit block: `√2 k c_max` = 0.87 m/s | Correct (velocity unit 8.567 m/s) |
| Lat/lon extents self-consistent at 66.8 km both axes | Correct |
| `ρ = 0.105` = 5.4 km, 5.4 grid cells across the footprint | Correct |
| 168 steps x 600 s = 28 h, 29 hourly frames | Correct |
| Rossby number 4.9, `τ = 0.28 s`, well depth `−π²A = −0.987` | All correct |
| Appendix A margin: `Ω‖p_seed‖ = 0.081`, critical `Ω = 0.238` | Correct |
| `γ₂ = 8/√10` for pentagon-plus-center | **Correct, I reproduced it exactly** |

That last one is worth stating plainly: I rebuilt your formation matrix and the
row norms of `Φ⁻¹` at unit radius come out `[1, 2/√10, 2/√10, 4/√10, 8/√10,
8/√10]`. Your `γ₂` is exact. See item **E4** for what that also reveals.

---

## Part 2: What I do not like

Severity: **[H]** blocks acceptance, **[M]** a reviewer will ask, **[L]** polish.

### 2.1 Technical errors and overreaches

---

**E1 [H] The three condition numbers are computed on different normalizations,
and as written they say the opposite of what you mean.**

Section II-C: nominal `cond(A) = 8.3`, near-degenerate `cond(A) = 564`.
Section IV-G: "the formation stays far from the conic degeneracy throughout, at
`κ(Φ) = 688`."

688 is larger than 564. A reader who takes those three numbers at face value
concludes the ocean formation is *more* degenerate than your pathological
example. I reproduced all three:

- `cond(Φ) = 8.26` at `ρ = 1` (nominal shape)
- `cond(Φ) = 563.9` at `ρ = 1` with the center robot at `0.99ρ`
- `cond(Φ) = 688.4` at `ρ = 0.105` (nominal shape, ocean radius)

So Section II-C reports the *radius-normalized* condition number and Section IV-G
reports the *raw* one. The 688 is entirely the `ρ⁻²` column scaling of the
monomial basis, not degeneracy at all. For reference the raw number at the double
gyre radius `ρ = 0.075` is 1349, and the genuinely near-degenerate case at that
radius is 87,018.

**Fix:** report `κ` on the radius-normalized `Φ` everywhere. Then the ocean
formation reports 8.26, identical to nominal, and the sentence "stays far from
the conic degeneracy" becomes true and obviously so. Add one line stating the
normalization, because a careful reader *will* recompute this.

---

**E2 [H] "inflating all noise gains by that factor" is wrong three ways.**

Section II-C, on moving the center robot to `0.99ρ`. I computed the actual row
norms:

| Coefficient | Nominal `γ` | At `0.99ρ` | Inflation |
|---|---|---|---|
| `a₁` (order 0) | 1.00 | 70.5 | 70.5x |
| `a₂`, `a₃` (order 1) | 0.632 | 0.632 | **1.0x** |
| `a₄` (xy) | 1.265 | 1.265 | **1.0x** |
| `a₅`, `a₆` (xx, yy) | 2.530 | 141.9, 140.7 | ~56x |

Three problems: (i) "that factor" is 564, but the condition number *ratio* is
68; (ii) not all gains inflate, the first-order and mixed second-order gains are
untouched to three digits; (iii) the ones that do inflate go up by 56 to 70, not
564. A condition number bounds worst-case relative amplification, it does not
multiply the individual gains.

This correction actually *helps* you. It says conic drift attacks exactly the
channels the `D` tracker's Hessian depends on and leaves the first-order channels
the `s1` tracker rides untouched, which is a nice extra piece of your
objectivity-versus-noise story.

---

**E3 [H] The band `B` in (14) is mislabeled, and the label is not just imprecise,
it names the wrong set.**

You write "The cluster is on the separatrix, `p_c ∈ B`, when `|D̂₀| < ε_raw` or ...".

`B` is a neighborhood of `{D = 0}`. On the double gyre the separatrix has `D =
−π⁴A² ≈ −0.974` along almost all of it, against `ε_raw = 10⁻³`. So `B` does not
contain the separatrix. It contains the crest at the origin, and it also contains
**the entire Okubo-Weiss diamond boundary**, which is not the separatrix at all.

Your preceding sentence makes the intent clear: you want to detect the crest,
where the along-trench gradient vanishes. Say that. "The cluster is at a crest of
the trench" or "in the vanishing-gradient band". Also add one sentence on what
happens if the band fires while the cluster is near an Okubo-Weiss boundary
during acquisition, because a reader will ask and the answer (the along-trench
speed hands off to the flow, which is harmless off-structure) is short.

---

**E4 [M] "one per derivative order" is not accurate: order 2 has two distinct
gains.**

From my reproduction: `γ(a₄) = 4/√10 = 1.265` for the mixed term, `γ(a₅) = γ(a₆)
= 8/√10 = 2.530` for the pure terms. That is a factor of 2 within one derivative
order. Section II-C hedges correctly ("for the pure second-order coefficients"),
but the contribution list in the intro drops the hedge and claims one gain per
order.

**Fix:** print the whole ladder as a three-line table (see A1 below) and reword
the contribution to "exact, heading-isotropic noise gains for every coefficient,
scaling as `ρ⁻�q` with derivative order `q`".

---

**E5 [H] The unbiasedness claim contradicts your own position-noise model.**

Section II-C: "Each term of the determinant pairs one `â` with one `b̂`, so
component independence leaves `D̂`, `∇D̂`, and `Ĥ_D` unbiased at all noise levels."

But the same subsection introduces position noise, which *correlates* the two
components wherever `∇u · ∇v ≠ 0`, and IV-C measures that correlation at 0.41.
`E[â₂b̂₃] = a₂b₃ + cov(â₂, b̂₃)`, so `D̂` is biased under position noise. The
argument holds for measurement noise only.

**Fix:** scope the sentence to measurement noise, then add the position-noise
bias term explicitly. You already have everything needed to write it. IV-C's
"signed mean error zero to within 2.7% of one standard deviation across
forty-eight conditions" then becomes evidence that the bias is small in practice
rather than an unexplained agreement with a claim that should not hold.

---

**E6 [M] The traversal-time bound assumes something false near a saddle.**

Section III-C: "both branches of (15) keep `v∥ ≥ 0` and bounded away from zero,
so traversal cannot reverse and a segment of length `L_Γ` is crossed in time at
most `L_Γ/(k m)`."

At a saddle of the double gyre, `∇D = 0` **and** the flow vanishes. So the
off-band branch `|w₁ᵀĝ₀|/|λ₁| → 0` and the on-band branch `v̂₀ᵀw₁ → 0`
simultaneously. `m` is not bounded away from zero on a segment whose endpoint is
a saddle, which is every segment you traverse.

**Fix:** state the bound for segments excluding an `ε` neighborhood of the
stagnation points, then handle the neighborhood the way you already handle the
isotropic point in III-D (bounded transit, bounded excursion). You have the
template.

---

**E7 [M] The `D` tracker's sign reference degenerates in exactly the same place
the `s1` tracker's does, which undercuts your IV-E argument.**

IV-E argues the `D` tracker is more robust because it "re-signs `w₁` against `v̂₀`
every cycle" while the `s1` tracker carries the sign in state. But the sign rule
`v̂₀ᵀw₁ ≥ 0` is undefined when `v̂₀ → 0`, which happens at every saddle, which is
where the mission ends. It also degenerates when the flow runs transverse to the
trench.

This does not kill the argument (the `D` tracker re-checks far more often, over
far more of the trajectory) but it does mean the asymmetry is one of *frequency*,
not of *kind*. State it that way, and note that the `D` tracker inherits an
implicit continuity assumption of its own near stagnation.

---

**E8 [M] The Hessian-eigenvector definition of "trench" is assumed, not stated.**

Section III-C takes `w₂` as "across the trench" and `w₁` as "along it". That
identification requires a trench definition. Under the height-ridge (Eberly)
definition it needs `g₀ ⊥ w₂` as well, and in general the Hessian eigenvector is
not the trench normal. It happens to be exact on the double gyre because `H_D` is
diagonal there, which is a property of your benchmark, not of trenches.

**Fix:** one paragraph in III-C stating which trench definition you adopt, with
a citation, plus the observation that the benchmark satisfies it exactly. This
also strengthens IV-G, where the ocean field is not axis-aligned and the
assumption is doing real work.

---

**E9 [M] The `λ₁ < 0 < λ₂` premise fails over half of every separatrix segment,
and you only admit it in Limitations.**

III-C asserts "A trench of `D` is convex across and concave along, so `λ₁ < 0 <
λ₂`". But `H_D = 2π⁶A² diag(cos2πx_f, cos2πy_f)` on `x = 0` is **positive
definite** for `|y| > 0.25`, which is the outer half of each segment, including
both saddles. Limitations (fifth item) says so, correctly, and adds that the
fitted frame stays indefinite by a field symmetry rather than an estimator
property.

That is a significant caveat on the control law's premise and it should appear at
the point where the premise is introduced, not eight pages later. As written, a
reviewer reads III-C, believes the geometry, and then discovers in Limitations
that the law operates outside its stated assumptions for most of its trajectory.

---

**E10 [M] The `[−0.51, 0.45]` versus `[18.3, 19.2]` comparison needs one more
sentence, and it is the most interesting sentence available.**

Two things. First, at the exact saddle both true eigenvalues are `2π⁶A² = 19.23`,
so `[18.3, 19.2]` must be evaluated at the cluster's actual offset. Say so.

Second and more important: the fitted `|λ₁| = 0.51` against a true 18.3 is a 36x
underestimate, and `|λ₁|` sits in the *denominator* of the off-band along-trench
speed (15). So the tracker's ability to keep moving past the saddle, rather than
stalling as **E6** would predict, is directly enabled by that underestimate. You
already say the traversal works by a field symmetry. Add that the *speed* through
the saddle is likewise an artifact. It is a better limitation than the one you
currently state, because it is quantitative.

---

**E11 [M] `ŝ1` bias: a prediction is validated but never made.**

II-C says only "`ŝ1` is biased low ... worst where the strain degenerates", with
no formula. IV-C then reports "The downward bias of `ŝ1` matched prediction within
5% in eighteen of twenty-four conditions and within one standard error in all
twenty-four" and refers to "the resolution of `10⁴` draws" at `r/σ_r ≥ 3`.

There is no prediction in the paper to match. Add the one-line result. The bias
comes from `E‖(ŝ_n, ŝ_s)‖ > ‖(s_n, s_s)‖` for a noisy 2-vector, which for
moderate SNR gives a bias of order `σ_r²/(2r)`, and that expression immediately
explains both the `1/r` exposure you cite in IV-F and the `r/σ_r ≥ 3` resolution
floor. It is two sentences and it converts an unsupported validation into a
second confirmed prediction.

---

**E12 [M] `e_H` is used twice and never defined.**

Lines: IV-C ("held there by `e_H`") and IV-F ("the unmodelled-derivative floor
`e_H`"). It appears nowhere else in the source. Define it in II-C alongside the
truncation discussion, ideally with its scaling in `ρ` so the reader can see why
it floors `Ĥ_D` at all noise levels.

---

**E13 [M] The N-scaling paragraph contradicts your own degeneracy result.**

II-B: "A ring alone is degenerate at any size." II-C: "Robots added to one ring
drive the first-order gain down as `√(2/N)` ... reaching 2.14 at twenty ...
twenty-one robots on two rings reach 1.42 where twenty-one on one ring reach
2.14."

On a single ring `½x² + ½y² = ½ρ²·1` identically, so the quadratic basis is rank
deficient for *any* N and no least-squares gain exists. The numbers must be for
ring-plus-center configurations. Say so explicitly, or the paragraph reads as
self-contradictory within two pages.

---

**E14 [M] The capture test detects a stationary point of a provably concave
surface. Explain why that is a minimum finder.**

II-D proves `ŝ1` is concave with negative semidefinite Hessian for every field
and formation. (19) then declares capture at `‖∇ŝ1‖ < g_capture`. A vanishing
gradient on a concave surface is a *maximum* of the fit. IV-D asserts "The one
terminal test a quadratic fit supports is a vanishing gradient, and a flow saddle
is a genuine two-dimensional minimum of `s1`, so that test is sufficient", which
reads as a contradiction until the reader works out that the fitted *gradient*
still tracks the true gradient even though the fitted *curvature* has the wrong
sign.

That is the actual argument and it takes one sentence. Write it. Right now this
looks like an error to any reviewer who reads II-D carefully, which is the kind
of reviewer you will get.

---

**E15 [L] Two statements need incompressibility scoping.**

- II-D: "where `D < 0` the flow is strain-dominated and nearby particles separate
  at rate `√(−D)`." True only when `ω = 0`; in general `|s1| = √(ω²/4 − D)`.
- Eq. (8) is derived from `D = ω²/4 − s1²`, which is the incompressible
  specialization of (6). IV-G explicitly reports that the ocean field is *not*
  incompressible (`2|µ|/r` median 0.8 to 1.1), so (8) does not hold there and the
  trench-tangency argument it supports should be flagged as benchmark-only.

---

**E16 [L] Justify (6) with the identity rather than the phrase.**

"The cross term `tr(SW)` vanishes, so the determinant splits exactly" is correct
but cryptic. The underlying fact is the 2x2 identity `det(A+B) = det A + det B +
tr A tr B − tr(AB)`, with `tr W = 0` and `tr(SW) = 0` for symmetric times
antisymmetric. One clause fixes it.

---

### 2.2 Contradictions

| # | Location A | Location B | Problem |
|---|---|---|---|
| **C1 [H]** | Abstract: "four to five times more measurement noise" | IV-E and Conclusion: "a full grid step later/longer" | 0.0079 vs 0.0015 to 0.002 is a factor of 4 to 5, and against the stated 0.0005 sweep spacing it is roughly **twelve** grid steps, not one. Two different characterizations of the same result, one of them arithmetically wrong. Pick the ratio; it is the honest and more impressive number. |
| **C2 [H]** | Intro contribution (d): "a selection rule for choosing between the two primitives **by field and mission**" | Conclusion: "The operator's choice therefore depends on the **observing platform, not on the flow**" | Direct contradiction. The conclusion is the correct one and it matches the abstract ("by observing platform and mission"). Fix the intro. |
| **C3 [M]** | Intro contribution (a): "exact, **heading-isotropic** noise gains" | IV-C: recovered slope moves between 4.0 and 9.7 over the 72 degree ring phase; "An individual formation orientation is [biased]" | Not actually inconsistent, since the *noise gains* are isotropic and the *truncation bias* is not, but nothing in the text says so. Add the distinction to the contribution sentence, because as written it reads as refuted by your own IV-C. |
| **C4 [M]** | III-A: the state layer "applies the hysteresis that keeps a noise-driven flicker from recycling a mode" | Limitations 2: "noise can chatter the `D` tracker across the band test boundary" | The only hysteresis specified anywhere is the `s1` tracker's `−4s_trim` / `−s_trim` latch. The `D` tracker's band (14) has none. Either scope the III-A claim or give (14) a hysteresis band. |
| **C5 [M]** | II-C: "Formation error reaches the estimate **only** here [through conditioning]" | II-C, same subsection: position noise `ξ` perturbs where the sample is taken while the fit uses nominal positions | Position error is formation error and it reaches the estimate through a completely different route, which you model. Reword to "formation *shape* error enters the conditioning only here". |
| **C6 [M]** | Intro contribution (c): "**closed-loop frame equivariance** of the `s1` tracker" | Appendix A: "Equivariant commands do not by themselves give equivariant paths"; paths agree only up to a bounded transport residue under `Ω‖p_Q‖ < k c_max`, with one non-objective seeded step | The appendix is correct and appropriately careful. The contribution claim overstates it. Reword to "frame-equivariant commands and bounded path divergence under a stated rate condition". |
| **C7 [M]** | II-C: near-degenerate `cond = 564` presented as pathological | IV-G: `κ(Φ) = 688` presented as healthy | See **E1**. |
| **C8 [L]** | Intro contribution (a): "a conic characterization of the formations that fail" | II-B: one sentence, cited to [2], no proof or construction | Overclaim relative to delivery. Either add the short argument (the six monomials fail to be poised exactly when the nodes lie on a conic, a classical bivariate-interpolation result) or downgrade the contribution wording. |

---

### 2.3 Clarity, redundancy, incomplete phrasing

**Undefined terms used as if defined:**

- `e_H` (E12 above).
- **"straddle retention"** appears three times, including in the Fig. 5 caption
  and legend, and is never defined. A reader who does not know [26] has no idea
  what a straddle is or what retaining one means. One clause: "straddle retention,
  the fraction of trials in which the formation continues to bracket the
  structure, the failure count used in [26]".
- **"the failure count of [26]"** is doing the defining work and is itself
  unclear. Failure count of what, measured how?
- **"trench-network distance"** (IV-F, "reaching a mean trench-network distance
  ten times its inertial value of 0.005") is the headline metric of the
  objectivity trial and is never defined. Distance to the nearest point of which
  set, in which norm?
- **"the gain ladder"** is invoked four times as a known object (II-C, IV-C, IV-E,
  IV-F) and is never displayed. Only `γ₂ = 8/√10` appears. See A1.

**Redundant notation:**

- The formation matrix is `Φ` in II-A and `A` in II-C, and they are the same
  matrix. II-C even says "the `6 × 6` matrix `A` stacks the basis vectors `φ(pᵢ)`
  row-wise", which is the definition of `Φ`. IV-G then reports `κ(Φ)`. Pick `Φ`
  and delete `A`.

**Vague or incomplete phrases:**

- II-C: "Only the constants are formation specific ... **the exponent is not.**"
  Dangling. "the exponent `−q` is not" would close it.
- II-E closing: "Both have companion features in the quadratic fit, making this a
  matter of navigating a scalar field invariant." Three abstractions in one
  sentence. What is a companion feature? Say what the section actually
  established: both surrogates put a transverse trench on the separatrix, and
  both trenches are recoverable in closed form from the fit, so the tracking
  problem reduces to scalar-field trench following.
- II-E: "the four crossings the domain walls carry at `x = ±1, y = ±0.5`" reads
  as four lines rather than four points. They are the domain corners. Say
  "the four domain corners".
- II-E: "Two further facts carry into Sections III and **IV-G**." The two facts
  (separatrix is a `D` trench; strain shares the grid but signs its segments) are
  used in III and in **IV-D**. IV-G is the section that *removes* the coincidence.
  Likely a wrong cross-reference.
- IV-D: "Prior AN primitives [3], [2] report local convergence without such an
  estimate." Citation order reversed; also this sentence is a strong claim
  deserving its own position rather than a trailing clause.
- IV-G: "84 of 100 jittered starts ... landing on the same middle-island branch"
  is introduced with "the `D` tracker's outcome is **insensitive** to the start".
  16% branch divergence is not insensitivity. Report it as 84% branch agreement
  and let the number speak.
- Appendix A: "At the trial's start the right side is 0.096 against 0.081."
  Ambiguous which side is which. Write "the flow term is 0.096 against a
  transport term of 0.081".
- III-B: the parameter dump is a 9-line run-on sentence with two operating
  points interleaved. This is the single least readable paragraph in the paper.
  Make it a table (A2).

**Source hygiene (from the .tex):**

- Two `\label`s on the same subsection: `sec:surrogates` and `sec:strain_field`
  both on II-D (lines 337 to 338).
- Two orphan equation labels never referenced: `eq:dg_u` (11) and
  `eq:det_analytic_main` (12). Either reference them or drop the labels.
- No `\label` on Section IV or IV-A, so nothing can cross-reference the
  simulation setup.
- Zero tables in the document.

---

### 2.4 Continuity

**Introduced with no lead-in (appear out of the blue):**

| Item | First appearance | Should be introduced in |
|---|---|---|
| `e_H` | IV-C | II-C |
| The `D` tracker capture test `λ₁λ₂ ≥ 0` | IV-D | III-C |
| "straddle retention" | IV-E and Fig. 5 caption | IV-A or IV-E lead |
| The margin-hold rule | IV-E | III-D |
| `κ(Φ)` as a monitored runtime quantity | IV-G | II-C |
| "trench-network distance" | IV-F | IV-A metrics paragraph |
| `ω_dg = 2π/10`, `ε = 0.1` | IV-D | II-E (where the unsteady member is mentioned) |

The `D` tracker capture test is the serious one. **Section III never specifies a
terminal condition for the `D` tracker.** III-A promises the state layer
"sequences acquisition, traversal, and terminal capture", III-D gives the `s1`
tracker an explicit three-part test (19), and III-C gives the `D` tracker
nothing. Then IV-D says "the capture test `λ₁λ₂ ≥ 0` cannot fire", which is the
reader's first encounter with a test that was apparently in force all along. This
is an asymmetry in how the two primitives are specified, and it lands precisely
on the comparison the paper is built around.

**Introduced with no follow-up (orphan threads):**

- **The N > 6 scaling and radial-diversity analysis (II-C).** This is the largest
  orphan in the paper: two full paragraphs establishing `√(2/N)` first-order
  scaling, a second-order floor of 2, and the 2.14 vs 1.42 one-ring vs two-ring
  comparison. Every single experiment uses six robots. Nothing in Sections III or
  IV touches it. Either add a closed-loop N sweep, or move the analysis to a
  clearly-labeled design-guidance subsection and say explicitly that it is not
  exercised here.
- **The `ε_dim` branch of (14).** Introduced with a units argument, never
  analyzed. Which branch actually fires, and how often?
- **The Decabot.** Introduced in III-B as "built for this work" with two
  citations, used only for the `α_mom = 0.7` value, then reappears only in future
  work. Fine as-is, but the phrase "built for this work" promises hardware
  results that the paper explicitly does not have.
- **"responds to losing track of the feature of interest" (III-A).** No mechanism
  anywhere, no experiment. Either specify it or cut the clause.
- **The [4] critique.** "and [4]'s fixed symmetric weights hold only for the ideal
  ring geometry" is a criticism raised in the intro and never returned to,
  particularly awkward given that you then adopt [4]'s geometry.
- **Recorded-but-unreported metrics.** IV-E: "each trial also records straddle
  retention, the failure count of [26], and tracking error." Straddle retention
  appears only in Fig. 5, never in prose. Tracking error appears only at zero
  noise (0.0075 vs 0.0016). You collected 10,000 trials per cell across a 2D grid
  and reported the noise dependence of one metric out of three.
- **Drive-by citations.** [18], [19], [20] appear once in a single "adjacent
  active directions" list; [23] and [25] similarly. Acceptable for an intro but
  they contribute nothing.

**Asymmetric experiments (`D` gets it, `s1` does not):**

- Time-varying double-gyre spot check: `D` only.
- 20,000-cell reachable-set grid and 10,000 random starts: `D` only.
- 100 jittered ocean starts: `D` only.

Each of these is a claim about one primitive presented inside a paper whose
entire structure is a controlled comparison. A reviewer will read the asymmetry
as "the `s1` tracker did worse and it was left out". Even a one-line negative
result for `s1` on each is better than silence.

---

### 2.5 Figures and tables

**All seven figures are referenced in the text.** Fig. 1 (II-E), Fig. 2 (III-A),
Fig. 3 (IV-C, IV-E), Fig. 4 (IV-D), Fig. 5 (IV-E), Fig. 6 (IV-F), Fig. 7 (IV-B,
IV-G). No orphans, no dangling `\ref`. Good.

Issues:

- **[H] Fig. 4 leaks development jargon into the rendered figure.** The plot
  titles read "Objective separatrix traverser (Primitive 11) vs Logic C
  (Primitive 7): path match" and the panels are labeled "Controller 1: D-tracker
  (Logic C)" and "Controller 2: objective s1 traverse". "Primitive 11", "Primitive
  7", and "Logic C" appear nowhere in the paper. Regenerate with the paper's own
  names.
- **[M] Fig. 2 caption promises a layer the figure may not show.** The caption
  says "with the state control layer sequencing navigation modes above them", but
  the rendered boxes are Desired Cluster Shape Parameters, Formation Controller,
  Navigation Controller, Feature Estimation, `J⁻¹`, Kinematics, `J`, Robot 1..n.
  I do not see a state control layer box. Either add it or reword the caption.
- **[M] Zero tables in a 12-page paper with two operating points, six-plus
  experiment families, and a gain ladder.** See A1 to A3.
- **[L] Fig. 3 is doing a lot of work** and its two dashed reference lines
  (error-equals-signal, the 50% traverse-success level) are the hinge of the IV-E
  argument. Consider annotating the `σ_uv = 0.0079` cliff directly on panel (a).

---

## Part 3: Narrative arc

### 3.1 The gap, and whether you fill it

**Gap as stated:** existing structure trackers either (i) commit to the
structure's identity before tracking, or (ii) require a finite integration
horizon, and both characterize offline from a reconstructed field.

**Do you fill it?** Yes, substantively. Instantaneous, one synchronous sample, no
horizon, no prior identity, and the origin-crossing result (riding an attracting
OECS onto a repelling one on the same law) is the concrete demonstration that
identity is not assumed.

**Three arc weaknesses:**

1. **The identity-agnostic result is your headline claim and it is buried.** The
   fact that a single unmodified law rides through the origin, from attracting to
   repelling, with the degenerate-frame guard never firing over a full `r_band`
   sweep, is the direct answer to gap (i). It currently sits mid-paragraph in
   IV-D. It should be a named experiment with its own figure panel.

2. **The no-advection premise is a bigger deal than Limitations makes it.** You
   criticize [26] because "its robots begin on the manifold and are advected by
   the flow". Your robots are not advected, which is a *stronger* platform
   assumption, not a weaker one. Limitations admits this and correctly notes it is
   exact for the planned Decabot but an extrapolation on the ocean. That belongs
   in the introduction, next to the critique, so the comparison is fair on first
   reading. A reviewer who finds it only on page 11 will feel handled.

3. **The estimator design story never closes.** Section II-C builds a full design
   framework (radius sets scale, robot count buys precision, radial diversity buys
   order) and calls each "a specifiable mission attribute selected against a stated
   trade". No closed-loop experiment varies any of them. The one design parameter
   that does change (`ρ` from 0.075 to 0.105 on the ocean) is changed for a stated
   reason but its effect is never measured. See A4.

### 3.2 Test plan versus results

**IV-A promises four experiment families.** The paper reports at least eleven
distinct experiments:

| Experiment | In the stated plan? | Results reported? |
|---|---|---|
| Estimator accuracy sweep (10⁴ draws/condition) | No | Yes (IV-C) |
| `s1` channel analytic check | No | Yes (IV-C) |
| Ring-phase rotation sweep (72 deg) | No | Yes (IV-C) |
| Noise-free runs, six matched starts | Yes | Yes (IV-D) |
| `r_band` sweep 0.05 to 0 | No | Yes (IV-D, one clause) |
| 20,000-cell reachable-set grid + 10,000 random starts | No | Yes (IV-D), `D` only |
| Time-varying double gyre spot check | No | Yes (IV-D), `D` only |
| 2D noise grid Monte Carlo | Yes | Yes (IV-E) |
| Channel-substitution ablation | No | Yes (IV-E) |
| Margin-hold rule evaluation | No | Yes (IV-E) |
| Rotating observer | Yes | Yes (IV-F) |
| Ocean 168-step run | Yes | Yes (IV-G) |
| FTLE four-anchor persistence check | No | Yes (IV-G) |
| 100 jittered ocean starts | No | Yes (IV-G), `D` only |

**Good news:** nothing in the plan lacks results. **Bad news:** two thirds of the
reported experiments are not in the plan, so the reader cannot tell what was
designed in advance versus what was added in response to a finding. For a paper
whose central claim is a controlled comparison, this matters. Fix with a table
(A3).

**Metrics recorded but not reported:** straddle retention across the noise grid
(prose), tracking error across the noise grid (only zero-noise is given).

### 3.3 Does the theory tie to the results?

**Ties that work well, and you should say so more loudly:**

- Gain ladder (II-C) → estimator validation (IV-C, within 2.1%) → predicted
  failure threshold (IV-E, `σ_uv ≈ 0.0079` vs `∇D̂` informative to 0.0075, within
  5%). This is a theory-to-closed-loop prediction chain and it is excellent.
- Concavity of `ŝ1` (II-D) → capture asymmetry (IV-D).
- Eq. (7) → rotating-frame departure (IV-F).
- Eq. (8) and `ω ≡ 0` on the trench network → why the two agree on the double
  gyre and need not on the ocean (IV-D, IV-G).

**Theory introduced and never used:**

- N-scaling and radial diversity (II-C). Largest orphan.
- The conic degeneracy condition (II-B) is used once as a runtime number (`κ(Φ)`)
  and the mitigation is explicitly unimplemented.
- `ε_dim` (III-C).

**Results with no theory behind them:**

- The `ŝ1` bias validation in IV-C matches a prediction that is never written
  (E11).
- The margin-hold threshold is derived from the gain ladder, but the ladder is
  never printed, so the derivation cannot be checked (A1).

### 3.4 Abstract, intro, conclusion consistency

**Matching:** the surrogate framing, the six-robot minimality, the rotating-frame
result and its two numbers, the ocean corridor result and its two distances, the
"simple, reactive, minimal" self-assessment, the platform-based selection rule
(abstract and conclusion agree, intro does not, see C2).

**Number audit:**

| Number | Abstract | Body | Conclusion | Status |
|---|---|---|---|---|
| Noise advantage of `D` tracker | "four to five times" | 0.0079 vs 0.0015 to 0.002 | "a full grid step longer" | **Mismatch (C1)** |
| Rotating-frame `D` final gap | (qualitative) | 1.219 | 1.219 | OK |
| Rotating-frame `s1` final gap | (qualitative) | 0.025 | 0.025 | OK |
| Ocean FTLE ridge distance | (qualitative) | 1.8 km / 2.4 km | 1.8 km / 2.4 km | OK |
| Ocean duration | "nearly the full record" | 28 h, 168 steps | 28 hours | OK |
| Six robots minimal | Yes | II-B | Yes | OK |
| Selection rule basis | "observing platform and mission" | IV-F | "observing platform, not the flow" | Intro disagrees (C2) |

**One stylistic note:** the abstract ends with "The controllers are admittedly
simple, reactive, and minimal in robot count". Honest, and I like it in the
conclusion, but in an IEEE abstract it reads as pre-emptive apology and it costs
you the last sentence, which is prime real estate. Move it; end the abstract on
the selection rule.

---

## Part 4: Reordering and additions

### 4.1 Reordering (in order of payoff)

**R1. Move the "Two consequences follow" paragraph out of II-C.** It discusses
the bias of `D̂` and `ŝ1`, both of which are defined in II-D. Either swap II-C and
II-D, or move that paragraph to the end of II-D. As it stands the reader meets
`ŝ1`'s bias one page before meeting `ŝ1`.

**R2. Move the III-B parameter dump into IV-A as a table.** It is simulation
configuration, not robot dynamics, and it currently interrupts the derivation of
the two primitives with two pages of numbers.

**R3. Move the margin-hold rule from IV-E into III-D.** Present it as a stated
option in the control law, then evaluate it in IV-E. Right now a new algorithmic
component is introduced in the results section and then referred to from
Limitations as if it were part of the design.

**R4. Add the `D` tracker's terminal capture test to III-C.** See continuity above.

**R5. Move the no-advection premise from Limitations to the introduction,**
adjacent to the [26] critique.

**R6. Promote the origin-crossing result in IV-D to a named experiment** with its
own paragraph heading, since it is the direct evidence for the identity-agnostic
claim.

**R7. Retitle II-E** from "The Double-Gyre Example" to something like "Benchmark
Field and Closed Forms", since it supplies the analytic constants that III-C's
convergence proof consumes rather than serving as an example.

### 4.2 Additions (highest value first)

**A1. A gain ladder table.** This is the single highest-value addition. You refer
to "the gain ladder" four times and never print it. It is six numbers:

| Coefficient | Order `q` | `γ_q` | Std. dev. |
|---|---|---|---|
| `a₁` | 0 | 1 | `σ_eff` |
| `a₂`, `a₃` | 1 | `2/√10 ≈ 0.632` | `γ₁ σ_eff / ρ` |
| `a₄` (mixed) | 2 | `4/√10 ≈ 1.265` | `γ σ_eff / ρ²` |
| `a₅`, `a₆` (pure) | 2 | `8/√10 ≈ 2.530` | `γ₂ σ_eff / ρ²` |

I verified these against your formation. Printing them makes the IV-E margin-hold
threshold checkable, makes IV-C's "matched within 2.1%" meaningful, and fixes E4.

**A2. A parameter table** with two columns (double gyre, ocean) and a units
column. Replaces the III-B run-on and makes the ocean unit conversion auditable.

**A3. An experiment matrix table** in IV-A: experiment name, purpose, which
primitive, trials, what is measured, where reported. This closes the test-plan
gap in Section 3.2 above in one float.

**A4. A closed-loop `ρ` sweep.** You claim radius is "a specifiable mission
attribute selected against a stated trade" (truncation bias against noise
suppression) and you never show the tradeoff closing the loop. A single-axis sweep
of `ρ` against far-saddle success at fixed `σ_uv`, for both primitives, should
show a minimum. Cheap to run given your existing sweep infrastructure, and it
converts II-C from design rhetoric into a validated design rule. This is the
addition that most improves the paper.

**A5. Symmetric `s1` results** for the three `D`-only experiments (time-varying,
reachable set, jittered ocean starts). Even negative results.

**A6. Algorithm boxes.** Two pseudocode blocks, one per primitive, covering the
estimator call, the mode logic (acquisition, first-contact latch, traversal,
capture, and the fallbacks), and the command. This would simultaneously fix the
missing `D` capture test, the unspecified "responds to losing track" behavior,
the state-layer hysteresis question, and the margin-hold placement. Four
continuity problems, one float.

**A7. The `ŝ1` bias formula** (E11).

**A8. A state-transition figure or inset** for the state control layer. It is
described in III-A as sequencing three modes with hysteresis and it is never
drawn.

**A9. Feasibility sentence on the ocean operating point.** You note both primitives
"spend the record near the edge of their command authority", with the `D` tracker
hitting the 0.87 m/s cap against a 0.71 m/s mean. A reader planning a real
deployment wants one sentence on what vessel class that implies and what happens
if the cap is exceeded.

**A10. Computational cost.** One sentence: a `6 × 6` factorization plus closed-form
polynomial evaluation per control cycle, cost `O(1)`, negligible against a 600 s
ocean step. It costs you a line and pre-empts a reviewer question.

**A11. Data and code availability statement.** [35] is public; say so and state
whether the simulator will be released.

---

## Part 5: References

- **No dangling citations.** All 35 entries are cited at least once, and every
  `\ref` resolves. Clean.
- **[H] [8] is "submitted for publication" and is load-bearing.** It supplies the
  three-robot critical-point estimator, the momentum model identified on the
  Decabot, and (with [31]) the control architecture. Four separate citations
  depend on an unavailable reference. Get it to arXiv before submission and cite
  the preprint, or reproduce the two facts you actually need (the `α_mom` value
  and the layered architecture) so the paper stands alone.
- **[M] Possible mis-citation in II-B.** "which fails only when they lie on a
  common conic [2]". [2] is Kitts/McDonald/Neumann on scalar-field AN primitives.
  The conic condition is the classical poisedness result for bivariate quadratic
  interpolation. Cite that literature, or [4], or give the two-line argument.
- **[L] Inconsistent citation for the same object.** II-E attributes the unsteady
  double-gyre perturbation to [11]; IV-D attributes the same field to [26]. Use
  [11] in both, optionally noting [26] uses it too.
- **[L] Reference ordering** in IV-D: "[3], [2]" should be "[2], [3]".

---

## Part 6: Prioritized fix list

### Must fix before resubmission

1. **E1 / C7** Normalize `κ(Φ)` consistently; the ocean formation currently reads
   as more degenerate than your pathological example.
2. **E2** Correct the noise-gain inflation claim (not all gains, not that factor).
3. **E3** Rename the band `B`; it currently names a set that excludes the
   separatrix and includes the Okubo-Weiss boundary.
4. **E5** Scope the `D̂` unbiasedness claim to measurement noise.
5. **C1** Reconcile "four to five times" with "a full grid step".
6. **C2** Fix contribution (d) to match the conclusion's platform-based rule.
7. **E12** Define `e_H`.
8. **Continuity** Specify the `D` tracker's terminal capture test in Section III.
9. **A1** Print the gain ladder.
10. **Fig. 4** Regenerate without "Primitive 11" / "Logic C".

### Should fix

11. **E6** Scope the traversal-time bound away from stagnation points.
12. **E9** Move the `λ₁ < 0 < λ₂` caveat into III-C.
13. **E10** Explain that the saddle traversal *speed* is a fitted-eigenvalue
    artifact.
14. **E11 / A7** Write down the `ŝ1` bias prediction you validate.
15. **E13** State that the N-scaling configurations include a center robot.
16. **E14** One sentence on why a vanishing gradient of a concave fit still finds
    a true minimum.
17. **E8** State the trench definition used in III-C.
18. **C3** Separate heading-isotropic gains from orientation-dependent truncation
    bias in the contribution list.
19. **C4** Reconcile the hysteresis claim with the `D` tracker's band.
20. **C6** Downgrade "closed-loop frame equivariance" to what Appendix A proves.
21. **R5** Move the no-advection premise to the introduction.
22. **A3** Add the experiment matrix.
23. **A4** Run the closed-loop `ρ` sweep.
24. **A6** Add the two algorithm boxes.
25. Define "straddle retention" and "trench-network distance".
26. Report straddle retention and tracking error across the noise grid, not just
    at zero noise.
27. Resolve [8]'s availability.

### Nice to have

28. **A2** Parameter table. **A5** Symmetric `s1` runs. **A8** State-machine
    figure. **A9** Feasibility sentence. **A10** Compute cost. **A11** Data
    availability.
29. **R1, R2, R3, R6, R7** Reorderings.
30. Fix duplicate labels, orphan equation labels, the II-E cross-reference to
    IV-G, the Appendix A "right side" phrasing, "the exponent is not", the
    "four crossings" wording, and the II-E closing sentence.

---

## Closing note

The reason this review is long is that the paper is good enough to be worth
picking at. The framing is genuinely novel, the analytic work is correct where I
could check it, and Section IV-E is the kind of diagnosis that most papers in
this area do not attempt. Nearly everything above is a bookkeeping failure
between what you know and what you wrote down: quantities used before definition,
one normalization reported two ways, an analysis built and never exercised.
Those are cheap to fix relative to what they cost you in reviewer confidence.

The one substantive research gap is A4, the closed-loop radius sweep. Section
II-C promises a design methodology and Section IV never tests it. That is the
addition that would move this from a good paper to a hard-to-reject one.
