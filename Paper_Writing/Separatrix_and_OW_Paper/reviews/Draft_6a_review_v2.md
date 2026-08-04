# Review of Draft 6a, version 2

*Revised after reading a second referee report on this manuscript. Read v1 first for the material that is unchanged; this document is the delta plus a merged action list.*

---

## 0. What that other report actually is

It is a referee report on an **earlier draft**, not on 6a. The tells are unambiguous:

- It quotes abstract text that is not in 6a: "six robots proven to be the minimum necessary team," "failing only at the estimator's theoretical limit," "both controllers successfully track a coherent ridge for 28 hours." 6a's abstract already reads *exactly* the way that report's F-B3 asks it to read.
- Its section and equation pointers are shifted throughout (its §VI-C is your VI-D, its eq (25) is your (42), its Table II is your Table III).
- It describes content 6a no longer has, most notably a non-similarity world-to-physical map with a 25.4° maximum angle error. 6a redefines the longitude half-extent as `0.3°/cos(34.2°)` and verifies the map is a similarity to 2 × 10⁻¹³.

**So the first thing to do with it is not act on it, it is diff it.** Roughly a third of its blocking list is already closed in 6a. Reopening any of that would be pure loss.

### Already fixed in 6a. Do not reopen.

| Their finding | Status in 6a |
|---|---|
| F-B2, Theorem 1 proved for one controller | Fixed. The proof now carries the s₁ branch explicitly (constant `c_max tanh(1)`, seed plus continuity), and adds the sentence that the direction claim rests on the exact-estimate hypothesis. |
| F-B3, three abstract overclaims | Fixed verbatim, all three. |
| F-B4, conclusion contradicts the ocean section | Fixed. Both now say a single shared start. |
| F-B6, a 24-h forward FTLE inside a 28-h record | **Fixed and now correct.** 6a anchors the 24-h field at the record start (first 25 frames) and runs the four-anchor persistence check at a 12-h horizon with anchors at 0, 5, 10, 16 h. The last one lands exactly at 28 h. Their arithmetic objection no longer applies. |
| F-B7, `√2 c_max` should be `√2 k c_max` | Fixed in the text. One residue survives, see A6 below. |
| F-A1, the title's second noun never lands | Fixed. II-D now names the upper segment an attracting OECS and the lower a repelling one, and VI-B says the tracker rides one onto the other under the same law. |
| F-T14, Table III uncited | Fixed. I checked every label and cite programmatically: no dangling `\ref`, all 40 bibitems cited, and only one unreferenced label in the whole source (`sec:time_varying`). |
| F-T16, duplicate reference at 0.91 similarity | Fixed. 6a is at 40 refs, down from 41. The closest surviving pair is [36] vs [37] at 0.72, which is two genuinely different cluster-space papers with overlapping authors. |
| F-C4, "Firstly" with no "Secondly" | Fixed. VI-H now reads First, Second, Third. |
| F-A2, selection rule buried in the ocean subsection | Fixed. It is its own subsection, VI-G. Their further suggestion about the abstract's last sentence still stands, see A11. |
| F-B5's 95th-percentile promise | Fixed by deletion. V-C now promises only the mean. |

### Where our two reports disagree with each other

We overlap on almost nothing, which is useful. Their strengths are the line-level and reproducibility layer. My strengths are the theorem statements and the unit chains. Neither of us found the other's blocking items:

- They did not find the Theorem 1 exact-estimate contradiction (v1 B1), the `(1+√α)² → 2(1+α)` algebra error (v1 B4), the ocean `s_trim` unit problem (v1 B2), or the missing seed-consistency hypothesis in Proposition 3 (v1 B5).
- I did not find the D tracker's flow-stagnation exposure (their F-T8), the 0.000-versus-0.0075 inconsistency (F-T5), or the unstated ring-phase policy (F-T3).

Treat the union as the real list.

---

## 1. Adjudication of the disputed numbers

Their report and the manuscript disagree on four numerical claims. I rebuilt the pentagon-plus-center estimator from (11) and (12) and re-ran the analytic double gyre to settle each. Ring phase below is measured with robot 1 at the stated angle from the +x axis and the remaining four at 72° increments, center robot at the origin.

### D1. The `[−0.72, 0.43]` eigenvalue pair. **The manuscript is right and their objection is wrong, but their underlying point survives and gets stronger.**

```
fitted H_D at (0, 0.45), rho = 0.075, noise-free
  ring phase  0 deg  ->  [-0.5056, +0.4486]   trace -0.057   <- their number
  ring phase 18 deg  ->  [-0.7211, +0.4345]   trace -0.287   <- YOUR number, exactly
  ring phase 36 deg  ->  [-0.5056, +0.4486]
  phase sweep        ->  lambda1 in [-0.721, -0.262],  lambda2 in [0.434, 0.449]
  Taylor limit, rho -> 0.005  ->  [-0.4707, +0.4704]   symmetric, as (42) predicts
  true H_D           ->  diag(19.23, 18.29), positive definite
```

They recomputed at phase 0, got `[−0.506, +0.449]`, and concluded your reported pair was unreproducible. It reproduces exactly, at ring phase 18°. That is not a coincidence: 18° is precisely the phase your own VI-D identifies as the peak of the orientation-dependent truncation bias. So your number is the **worst-phase** value, quoted without saying so.

Three consequences, and they matter more than the original objection:

1. Your reported pair is defensible. Say "at the mirror-antisymmetric ring phase" and give the phase-0 pair alongside it.
2. Their F-T1 structural point is correct: tracelessness is a Taylor-limit identity, and at ρ = 0.075 the induced trace ranges from −0.057 to −0.287 across phase. "Eigenvalues approximate ±|Ĥ_xx| everywhere" is true as ρ → 0 and 30% off at the working radius. This is v1 M6, and it is worse than I graded it.
3. It makes their F-T3 (unstated ring-phase policy) load-bearing rather than pedantic. See A1.

### D2. The 6.6 restoring slope. **They are right.**

```
transverse slope of (18) at (0.03, 0.25), rho = 0.075, analytic kappa = 6.8879
  phase  0 deg -> 3.997        phase 18 deg -> 7.099        phase 36 deg -> 9.664
  full sweep   -> 3.984 to 9.676, mean 6.826
  6.6 occurs at phase ~16 deg (and ~54 deg)
  phase-mean deficit = 0.90%,  not 4.7%
```

Your stated range and mean (4.0 to 9.7 about 6.83) reproduce to three digits, so the sweep is right. The headline is the problem: 6.6 is a single unstated phase near the middle of the range, and calling the gap to 6.888 a "4.7% truncation deficit" attributes to truncation something that is 90% orientation. Lead with 6.83 and 0.9%, then give the range as the orientation spread. Your very next two sentences already make this exact point, so this is only a matter of leading with the number that supports them.

### D3. The 0.0019 and 0.0079 open-loop predictions. **Both reproduce exactly. State the parking condition.**

```
fitted-trench offset at y = 0.25, rho = 0.075, |x| averaged over the 72 deg period
  D,  solving w2' * ghat0 = 0   ->  0.00198      you predict 0.0019   MATCH
  D,  solving dDhat/dx   = 0    ->  ~0.0040      2x your figure
  s1, solving P * grad(s1) = 0  ->  0.00798      you predict 0.0079   MATCH
```

This is the strongest quantitative prediction in the paper and it holds. But a referee reproducing it will naturally solve `∂D̂/∂x = 0` and land at roughly double. Nine words fixes it: "solving `w₂ᵀĝ₀ = 0` and `P∇ŝ₁ = 0` at the nominal formation, averaged over ring phase."

### D4. The 72°/36° symmetry. **They are wrong about the s₁ tracker, and the correction is the key to their best finding.**

They assert the s₁ offset is 72°-periodic only, does not share the D tracker's 36° symmetry, and "does not vanish at any phase." It does:

```
signed fitted-trench offset at y = 0.25, rho = 0.075
  phase:    0      9     13.5    18     27      36     45      54     63     72
  D:     +.00310 +.00222 +.00121  0   -.00222 -.00310 -.00216   0   +.00216 +.00310
  s1:    +.01248 +.00858 +.00461  0   -.00858 -.01248 -.00908   0   +.00908 +.01248
```

Both vanish at 18° and 54°, the two mirror-symmetric phases, and both change sign there. So |offset| is close to 36°-periodic for **both** trackers, with a small residual 72° component (0.00310 against 0.00284 at mirror-image phases, an 8% asymmetry) that your text does not mention and probably should not need to.

The correct sentence, covering both: the signed offset has period 72° and changes sign at each mirror-symmetric phase, so the magnitude carries an approximate 36° period.

**Why this matters far more than the symmetry bookkeeping:** the s₁ offset is *zero at 18° and 54°* and 0.0125 at phase 0. That is the mechanism behind their F-T5, and it turns a rounding complaint into a real problem. See A2.

---

## 2. New items I am adopting

Numbered A1 onward so they do not collide with v1's B and M numbering. Everything here survives into 6a and is absent from v1.

### A1. The ring-phase policy is never stated, and D1 proves it is load-bearing. **[promote to blocking]**

V-A gives the *initial* heading (zero for clean runs, uniform for sweeps). IV-A says the cluster heading follows the command from above. Nowhere does the paper say whether heading is held fixed during a run, slaved to the direction of travel, or left free.

Given D1, D2, and D4, at least four reported numbers are not reconstructible without it: the `[−0.72, 0.43]` pair, the 6.6 slope, the 0.0019/0.0079 predictions, and VI-E's 0.000. One line in V-A. It is the cheapest high-value fix in either report.

### A2. VI-E's "0.000" cannot be squared with VI-D's 0.0075, and now there is a mechanism. **[blocking]**

VI-D: the s₁ tracker's zero-noise tracking error is 0.0075, predicted at 0.0079 from the fit.
VI-E: the s₁ tracker's mean distance to the true trench network is "0.000 inertial."

Same tracker, same clean field, both distance-to-structure. My D4 table shows the s₁ fitted-trench offset is exactly zero at ring phases 18° and 54° and peaks at 0.0125 at phase 0. So a 0.000 is entirely consistent with the objectivity experiment having run at or near a mirror-symmetric phase, and 0.0075 is the trial mean over random headings.

If that is the explanation, **your headline objectivity number was measured at the one formation orientation where the estimator's dominant systematic error vanishes.** That does not invalidate it (the 1.219 gap is four orders above any of this), but a referee who works it out and is not told will assume the objectivity numbers were measured differently from everything else.

**Fix:** report VI-E to four decimals, name the ring phase, state whether "distance to the trench network" is the same quantity as `|x_c|`, and if the phase is incidental say so explicitly. One sentence, and it inoculates the paper's best result.

### A3. The D tracker has the same flow-stagnation exposure the paper flags against the s₁ tracker. **[blocking]**

This is the best finding in their report and I missed it entirely.

IV-D, arguing for the gradient-based capture test: *"the ambient flow also stagnates at a saddle, which would starve any flow-based tangent rule of a direction exactly where the ride needs one."*

The D tracker **is** a flow-based tangent rule. It signs `w₁` by `v̂₀ᵀw₁ ≥ 0` every cycle, and the on-band branch of (29) sets `v∥ = v̂₀ᵀw₁`, which vanishes in the same neighborhood, at the same saddles. VI-D then credits exactly that per-cycle re-signing as the D tracker's robustness advantage ("a corrupted frame costs it one cycle") without noting that the reference it re-signs against degenerates precisely where the s₁ tracker's seed would.

As written, the comparison is one-sided and the asymmetry is the load-bearing part of your noise-robustness argument. A hostile referee will go here first.

**Fix:** state what (29) does when `|v̂₀|` falls below the noise floor near a saddle, and report the D tracker's transverse behavior in that window from the clean runs. You already record closest approach, so the data exists. Handled, this makes VI-D airtight; unhandled, it is the softest joint in the paper.

Note this also interacts with my v1 B1: near the saddle the D law has a degenerate Hessian frame *and* a vanishing flow reference at the same time.

### A4. Sharpen the incompressibility caveat with an exact formula. **[supersedes v1 B7's divergence bullet]**

Their algebra is right and it is better than my "report the divergence ratio":

```
D = s1*s2 + w^2/4       general, any mu
D = w^2/4 - s1^2        requires mu = 0
exact discrepancy       = 2*mu*s1
relative error of the strain term = 2|mu|/r   to first order
```

I verified both. 6a already hedges (5) correctly ("the middle equality holding generally, and the last specializing"), so this is narrower than they filed it. What survives is VI-F, where "its magnitude in the tracking corridor was empirically small compared to the strain" is unquantified, and III-A, where you argue divergence matters enough to justify a sixth robot.

**Fix:** report the median `|µ|/r` over the SBC corridor. You compute µ every cycle via (17), so it is a logging change. Their threshold ladder is sensible: under 0.01 it is a footnote, 0.01 to 0.05 it is one sentence in VI-F, above 0.05 it needs the general form stated in II-C and the splitting-identity reading qualified.

### A5. Problem 2 says "certify" and VI-B says no certificate exists.

Problem 2: *"certify a terminal attracting core."*
VI-B: *"neither surrogate admits a definiteness certificate."*

(33) tests a resolved eigenframe, a small gradient, and a depth threshold. Those are necessary conditions for a two-dimensional minimum, not sufficient. **Fix:** "detect and hold at a terminal core of s₁ where one is reachable."

### A6. Table I's physical column understates the actual commanded cap by 2.55×.

IV-B's `√2 k c_max` is now correct in the text, but Table I still presents `c_max = 0.04 → 3.4 m/s` as the ocean physical value, and a reader takes that as the vessel requirement. The realized cap is `√2 k c_max`:

```
ocean point, k = 1.8, c_max = 0.04 world/step, 1 world unit = 51.4 km, 1 step = 600 s
  c_max            = 3.43 m/s   <- what Table I shows
  k * c_max        = 6.17 m/s
  sqrt2 * k c_max  = 8.72 m/s   <- the actual commanded cap, about 17 knots
  s1 ride speed, k * c_max * tanh(1) = 4.70 m/s  (constant while beta = 1)
```

**Fix:** add the derived cap to the caption and one clause in VI-H noting it never binds at the realized speeds, so it is an artifact of the gain choice rather than a platform claim. This ties directly into v1 B2, since a 4.70 m/s constant ride speed is also the diagnostic for whether `β` ever latched on the ocean.

### A7. Dimensional bookkeeping in (30) and (32).

`ĝ₀/λ₂` is a length and `sat()` returns a speed, so (30) carries an implicit `1 s⁻¹` absorbed into `k`. Same for `g⊥∇ŝ₁`, since `s₁` has units of inverse time. You acknowledge this in prose ("asks the cluster to close the whole remaining distance in one unit of time") and never resolve it in notation. Invisible on the non-dimensional double gyre, load-bearing in Table I's ocean column.

**Fix:** one sentence in IV-A: all gains are dimensionless in the non-dimensionalized system, and the ocean point carries an implicit per-step time unit of 600 s.

### A8. `ε` and `ω_dg` appear with no definition anywhere.

VI-C uses `ε = 0.1` and `ω_dg = 2π/10`. II-D mentions "the steady (ε = 0) member of the Shadden family" but never writes the unsteady field, so there is no expression to put `ε = 0.1` into, and `ω_dg` is entirely new. This is the clearest undefined-symbol case in the paper, and it compounds v1's finding that VI-C is an orphan (its label is the only unreferenced one in the source).

**Fix:** either write the unsteady field in two lines in II-D and give VI-C a test-plan entry and a figure inset, or cut it to a Limitations sentence. Half-promoting it is the worst option. On the merits I would keep it: mean transverse offset 0.006 against a trench swinging ±0.116 at 62% of the speed cap, with no feedforward term, is a strong result for a paper aimed at an unsteady ocean.

### A9. "A 300-step run at Δt k a⊥ above 12" is not reachable.

For the D tracker, `Δt k a⊥ = 0.3 csc²(πy_f) > 12` requires `|y| > 0.4495`, the last 5% of a separatrix segment. A 300-step run cannot sit there.

This compounds my v1 B4 addendum. The absence of an instability signature now has two independent explanations, and you should give both: the cluster is only briefly in that regime, and while it is there the proportional band `c_max/a⊥ = 0.0011` is smaller than the measured tracking error 0.0016, so the linearization the bound comes from never applies. **Fix:** "over the portion of the run where `Δt k a⊥` exceeds 12."

### A10. Smaller adoptions

- **"never directed against the ambient flow"** (after (30)) is stronger than `v∥ ≥ 0` supports. That constrains only the along-trench component; the `w₂` term is unconstrained relative to the flow. Add "the along-trench component of."
- **"verified numerically at the double-gyre saddle"** (IV-D) undersells a closed-form result. (10) gives `s₁ = −π²A|cos πx_f cos πy_f| ≥ −π²A` with equality exactly at integer `(x_f, y_f)`, so the saddle is a strict nondegenerate two-dimensional minimum analytically, with curvature `π⁴A` on both axes. I confirmed it. Replace the phrase; it removes a "we checked and it seemed fine" from a load-bearing justification.
- **The argmax SNR denominator** is unstated. 1.4 is against the combined noise of the two compared coefficients; against `σ(â₄)` alone a reader gets 3.13. Both of us independently derived the same thing, so say it in one clause.
- **(24)'s order q** is never mapped onto the three curves in Fig. 3(a). `D̂` is q = 1, `∇D̂` and `Ĥ_D` are both q = 2. Give the exponents in the caption.
- **Reference [9]** is unpublished and carries the architecture, the identified τ, the momentum model, and the 0.3 s settling claim. Add a preprint ID, or restate the two facts you actually need (τ and the settling time against transit times), each one sentence.
- **The advection premise** is stated twice in II-E and repeated near-verbatim in VI-H. Keep II-E, have VI-H cite it, and let VI-H's real content (exact for Decabot, extrapolation for the ocean) carry the paragraph.
- **VI-D quotes a σ = 0.002 figure that is not in Table III.** Move the Fig. 5 pointer ahead of the number.
- **The 62.4%-to-43.7% comparison spans 0.005 to 0.010** while the s₁ interval it is compared against spans 0.005 to 0.008.
- **Two different 20 000s** in adjacent subsections (gridded starts in VI-B, acquisition trials in VI-D). Label them and give the decomposition once.
- **III-D should close with forward pointers** to where its three constants are cashed out: the γ ladder in the s₁ argmax failure, the 72° period in the zero-noise floor, the floor `e` in traversal past a saddle. By the time VI-D says "the estimator, not the control law, sets its noise ceiling," the reader is holding three results from twelve pages earlier.

### A11. Two structural suggestions worth taking

- **Move the robot model (IV-B, (25)–(26), τ, `α_mom`) up to the end of IV-A.** IV-E's discrete-bound paragraph currently uses `α_mom` conversationally before the reader has the plant in hand. After the move, IV is self-contained: architecture, plant, two controllers, stability.
- **End the abstract on the selection rule rather than the ocean run.** Their F-A7 is right that the paper's most hedged result (one record, one start, terminal test never fires, evaluated against a field the controllers never see) currently occupies the abstract's last sentence, the conclusion's last evidence, and the final results subsection. Since VI-G already exists as its own subsection, the abstract can close on the decision procedure instead.

---

## 3. What I am *not* adopting

- **Their F-T1 as filed.** The `[−0.72, 0.43]` pair is correct at phase 18°. Report the phase, do not change the number.
- **Their F-T6 claim that s₁ lacks the 36° symmetry.** My D4 sweep shows the s₁ offset vanishes at 18° and 54° exactly as the D offset does. Their own F-T5 (offset crossing zero near phase 89°, which is 17° mod 72°) contradicts it too.
- **Their F-B6.** Correct against the old draft, resolved and now arithmetically sound in 6a.
- **Their F-B1 as blocking.** 6a already states (5)'s generality. What survives is the unquantified VI-F caveat, folded into A4.
- **Their suggestion to cut Fig. 2.** The architecture diagram is thin, but for a T-RO audience unfamiliar with cluster space it earns its half column. Cut the triplicated floor-`e` discussion instead if you need space.
- **Their F-B5 framing that this is "not bookkeeping."** Agreed on substance, and v1 already lists the same five orphaned studies. Their proposed fifth family, "Diagnostic ablations for the s₁ tangent rule," is a better container than my scattered one-liners. Take that phrasing.

---

## 4. Merged action list

**Blocking**

1. v1 B1. Theorem 1's exact-estimate hypothesis versus the positive-definite true `H_D` over the outer half of each segment. Prefer changing the guard from a sign test to a transverse-eigenvector test.
2. **A3.** The D tracker's flow-stagnation exposure at the saddle, and the one-sided VI-D comparison that rests on ignoring it.
3. v1 B2. The ocean `s_trim` unit problem. Report `ŝ₁` range, `β` latch history, realized speed, and `min‖∇ŝ₁‖` along the ocean path. A6 gives the speed diagnostic.
4. **A2.** Reconcile VI-E's 0.000 with VI-D's 0.0075, and name the ring phase.
5. v1 B3. Qualify the abstract's "fundamental tradeoff."
6. v1 B4. Correct the instability threshold to `2(1+α)`. No reported number changes.
7. **A1.** State the ring-phase policy in V-A.
8. v1 B6 and their F-C11. Say once, plainly, in V-B: same record and same metric so the numbers are interpretable against prior work, not a controlled comparison because initialization and advection differ.
9. v1 M2. The missing `k` in (37).

**Should fix**

10. v1 B5. Seed-consistency hypothesis in Prop 3, with the 0.081 against 0.107 margin reported.
11. v1 B7 and **A4**. Quantitative ocean metrics, including median `|µ|/r`.
12. **D1, D2, D3.** State the phase for the eigenvalue pair, lead with the 6.83 phase-mean and 0.9% deficit, and name the parking condition for the 0.0019/0.0079 predictions.
13. **A5, A6, A7, A9.** "Certify," Table I's derived cap, gain dimensions, and the 300-step claim.
14. **A8.** Define the unsteady field or cut VI-C, and give it a test-plan entry either way.
15. v1 items 9 through 13 unchanged: Okubo-Weiss normalization, the `√−D` material-rate qualifier, the correlation-order wording, "interior saddles," the latitude half-extent, moving (41) to II-C, rewriting the contribution list, the five unplanned studies (as a fifth test-plan family), and renaming the floor `e`.
16. **A10** line items, and **A11** structural moves.

**Unchanged from v1**

The margin gate remains the single highest-value addition. `κ(Φ)`, shape RMSE, and Table III confidence intervals remain the cheapest credibility wins.

---

## 5. One meta-observation

The two reports fail in opposite directions and that is worth knowing about your own review process. Theirs is stronger on reproducibility, because it rebuilt your estimator and swept the ring phase, which is exactly what surfaced A1, A2, and D1 through D4. Mine is stronger on the statements, because I re-derived the theorem hypotheses and the unit chains, which is what surfaced the `2(1+α)` error and the `s_trim` problem. Neither approach finds the other's failures.

They also got one thing wrong in a specific, instructive way: they recomputed at a single ring phase, found a mismatch, and filed it as your error rather than as a phase question. That is the same failure mode as your own 6.6 headline, which is a single-phase number presented as a truncation result. **The manuscript and its second referee both got caught by the same unstated convention.** Fixing A1 fixes both, which is a good argument for doing it first.
