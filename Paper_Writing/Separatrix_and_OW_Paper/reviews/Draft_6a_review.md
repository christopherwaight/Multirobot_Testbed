# Review: *Multirobot Tracking of Separatrices and Objective Eulerian Coherent Structures* (Draft 6a)

Reviewed against the `.tex` source, not the PDF text layer. Several apparent errors in the PDF extraction (the noise-gain constants, the frame convention) are extraction artifacts and are correct in the source. Where I say something is wrong, I checked it in the source and re-derived it.

**Recommendation as it stands: major revision.** The core is publishable in T-RO. The blocking problems are not new experiments, they are five places where a claim is stated more strongly than the paper's own evidence supports, plus one algebra error in the discrete stability bound. All are fixable in text plus about three cheap re-runs.

---

## 1. What I like

### 1.1 The strongest contributions, ranked as I read them

This ordering differs from the intro's, deliberately.

**1. Closed-loop objectivity, demonstrated rather than asserted.**
No prior robotic structure-tracker is frame-indifferent. The rotating-frame experiment with a matched inertial twin (gap 0.025 vs 1.219) is the single result in the paper that nobody else has. It is also the cleanest experimental design in the paper: same controller, same start, same physical field, two observers, one number. Say this louder. The abstract currently buries it behind the estimator.

**2. The splitting identity as an architectural axis.**
`D = ω²/4 − s₁²` is textbook. Using it as the design axis, two controllers reading the two terms of one identity off one fit, is not. It gives the paper an organizing spine that most multirobot papers lack, and it makes the objectivity result *structural* rather than a comparison of two unrelated algorithms. Section VI-E ("Objectivity is therefore paid twice over") is the best paragraph in the paper.

**3. Representability limits determining controller structure.**
Remark 2 (no usable Hessian of s₁ exists under a quadratic fit, because `μ − ‖(sₙ,s_s)‖` is concave for every field and every formation) and the floor `e` on `H_D` are both real results, and you use them to explain a design asymmetry rather than presenting it as tuning. "Traversal and capture are the same representability limit read twice" is the kind of sentence that gets a paper cited. I checked the concavity argument and the `D_xx` expansion; both are correct.

**4. The estimator, with minimality and the conic characterization.**
Proposition 1 is trivial linear algebra but correctly scoped (you qualify it to the *unconstrained* model and then defend that choice against the five-robot incompressible alternative, which is the right defense). Remark 1 is better than trivial: singular exactly on a common conic, six on a circle always singular, pentagon-plus-center non-singular because the unique conic through five points of a circle is that circle and a circle misses its own center. That is a clean, complete, and correct characterization, and it is strictly more general than [6]'s fixed-weight ring.

**5. The failure analysis in VI-D.**
Better than most published failure analyses. SNR 71 for the seed against 1.4 for the argmax, single-channel ablation (clean argmax restores 100% at σ=0.002 and 95.7% at 0.005), hop-rate instrumentation tracking the inversion rate, and the start relocation to (0, 0.15) as a controlled manipulation of the competing quantity. The conclusion you draw, that robustness is set by how far a decision is carried before it is re-checked against the world rather than by per-cycle accuracy, is genuinely transferable outside this paper.

### 1.2 Other things worth keeping

- Predicting the zero-noise tracking error from the open-loop fit alone (0.0019 and 0.0079 against measured 0.0016 and 0.0075). This closes the estimator-to-closed-loop loop quantitatively, and almost nobody does it.
- The orientation-dependent truncation bias: identified analytically as a harmonic-annihilation argument, then measured in closed loop with the right period, then shown to be unreachable by gain. That is a complete story.
- Honest negatives kept in: the guard never fires, the terminal test never fires on the ocean, no head-to-head was run, and the D tracker's traversal past a saddle is executed by a fitted frame outliving the true one. Reviewers reward this.
- The actuator-aware discrete bound in IV-E, tied to Table II. Very few adaptive-navigation papers acknowledge that the lag they abstract away is what relaxes the discrete limit.
- Notation discipline in III-C about where hats are written and suppressed.

---

## 2. Blocking issues

### B1. Theorem 1 assumes exact estimates; Section IV-C says exact estimates would break the D tracker on the benchmark

This is the most serious internal contradiction in the paper, and a sharp reviewer will find it in ten minutes.

Section IV-E: *"Estimates are taken exact, so hats are dropped."* Theorem 1 then claims traversal and convergence for both laws.

Section IV-C, correctly: on the double gyre `H_D = 2π⁶A² diag(cos 2πx_f, cos 2πy_f)`, so on the separatrix at |y| > 0.25 the true `H_D` is **positive definite**. With exact estimates the D law's degenerate guard (eigenvalues of one sign) fires over the outer half of every separatrix segment, and the guard is explicitly stated to carry "no stability claim." The law only works because the fitted `Ĥ_D` is traceless and therefore artificially indefinite.

So Theorem 1's hypothesis and the benchmark's geometry are mutually exclusive for the D tracker, and Table II makes it worse: `a⊥,D = κ⊥/κ̂⊥ = csc²(πy_f)` is computed from the *fitted* curvature, inside a section that just declared estimates exact.

**Fix, in order of preference:**

1. Change the guard. The frame is not actually broken at |y| > 0.25: `λ₂` is still the transverse eigenvalue (at y = 0.35, exact eigenvalues are 11.3 along and 19.2 across, correctly ordered). What fails is only the *sign test*. Replace "eigenvalues of one sign" with a test on which eigenvector is transverse, for example `|w₂ᵀ∇̂D|` maximal or `|w₁ᵀv̂₀|` maximal. Then the exact-estimate law works everywhere on the benchmark, Theorem 1's hypothesis becomes consistent with the geometry, and the traversal result stops depending on an estimator artifact. This is a small code change and one clean re-run.
2. If you keep the guard, state Theorem 1 for the class of fields on which `Σ` is concave along `Γ`, note explicitly that the double gyre is *not* in that class for D, and reframe Section VI-B's traversal as "what the fitted law does" rather than "what Theorem 1 predicts."
3. Either way, add one sentence to the Table II caption saying `a⊥,D` is evaluated on the realized fit, not on exact estimates.

**Related, same section:** Theorem 1's hypothesis `0 < κ ≤ κ⊥(ℓ)` also fails for the s₁ surrogate at the isotropic point, where `κ⊥ = g⊥π⁴A|cos πy_f| = 0` at y = 0. Table II prints that zero in its own first column. The runs cross the origin fine because the tangential channel carries them, but the theorem as stated does not cover the crossing. One sentence: the transverse rate degenerates at the isotropic point, the theorem applies on segments bounded away from it, and the constant-speed ride carries the cluster through.

### B2. The ocean s₁ claim contradicts Table I, and the units suggest the ride flag may never have latched

Section VI-F: *"Which of the two behaviors appears ... is a property of the field and not of a setting, since the terminal test is unconditional by design and has no mode to select."*

But `s_trim` is a setting, and Table I changes it from 0.05 to 0.3 between the two operating points, a factor of six. The terminal test's depth condition is `ŝ₁ < −4 s_trim`, so the threshold moved from −0.2 to −1.2. The sentence as written is not defensible.

Worse, the arithmetic. Ocean world units are L = 51.4 km and T = 600 s, so one world unit of strain rate is 1/600 = 1.67 × 10⁻³ s⁻¹.

| quantity | world units | physical |
|---|---|---|
| first-contact threshold `s_trim` | 0.3 | 5.0 × 10⁻⁴ s⁻¹ |
| capture depth `4 s_trim` | 1.2 | 2.0 × 10⁻³ s⁻¹ |
| reference strain: 0.5 m/s across the 5.4 km formation | 0.056 | 9.3 × 10⁻⁵ s⁻¹ |

The first-contact threshold sits about 5× above a representative HFR-scale strain and the capture depth about 20× above it. If the realized `|ŝ₁|` on the corridor never reached 0.3, then `β` never latched, the projector stayed the identity, and the s₁ ocean run was **pure gradient descent on a time-varying field, not the ride**. That is still an interesting result, but it is a different result from the one the section claims, and it would also explain why the s₁ path lags the D path near landfall.

There is a second, independent tell. With `β = 1` the ride speed is pinned at `k c_max tanh(1) = 4.7 m/s`, constant. Table I's note reports a realized mean near 0.5 m/s (for the D branch). If the s₁ run had ridden, its speed would have been a constant 4.7 m/s for the whole record.

**Fix (cheap, you already have the logs):** report, for the ocean s₁ run, (i) the range of `ŝ₁` along the path, (ii) whether and when `β` latched, (iii) the realized speed, and (iv) `min ‖∇ŝ₁‖` along the path against `g_capture`. Then either defend the non-firing with those numbers or restate the claim as "no point on the corridor met the depth condition at this operating point." Consider making the depth condition relative (a fraction of the running max `|ŝ₁|` seen during the ride) so that it is genuinely field-scale-free, which is what the current sentence is trying to claim.

### B3. The abstract's "fundamental tradeoff" is contradicted by the paper's own ablation

Abstract: *"demonstrate a fundamental tradeoff between tracking objectivity and noise tolerance."*

Section VI-D shows that feeding the argmax a clean gradient and eigenframe restores 100% at σ=0.002 and 95.7% at σ=0.005, where the unaided tracker is at zero. Section VI-H then lists two unimplemented fixes (margin gate, re-anchoring the sign on axis change) and says neither costs objectivity.

So the measured 4-to-5× noise gap is a property of *this tangent rule*, not of objectivity. Part of the cost genuinely is fundamental, and you already identified which part: Remark 2 forbids the Newton normalization, so the s₁ transverse channel contracts at gradient rate rather than Newton rate, and that is model-class, not tuning. Split the two.

**Fix:** abstract to "a tradeoff, in this implementation, between..." plus one clause naming the part that is structural (the forbidden Newton normalization). The conclusion already gets this right, which makes the abstract the outlier.

### B4. The discrete instability threshold is wrong

Section IV-E: *"real and positive below β = (1 − √α)² and unstable above β = (1 + √α)²."*

For `z² − (1+α−β)z + α`, the Schur condition is `|1+α−β| < 1+α`, so instability begins at **β = 2(1+α)**, not `(1+√α)²`. At `β = (1+√α)²` the roots merely coalesce at `−√α`, which is stable. Numerically at α = 0.7: β = 3.39 gives roots (−0.963, −0.727), stable; β = 3.40 gives (−1.000, −0.700), marginal; β = 3.45 gives (−1.131, −0.619), unstable. `(1+√0.7)² = 3.373`, `2(1.7) = 3.400`.

Good news: **no number in the paper changes.** The Table II caption's `Δt k a⊥ > 11.3` is what the *correct* bound gives (3.4/0.3 = 11.33; the stated bound gives 11.24), and the "fraction past instab." column is 10.40% under the correct threshold against 10.45% under the stated one, both printing as 10.4%. So this is a text fix only.

**Fix:** state the full ladder, it is more informative anyway. Real positive below `(1−√α)²`; complex with modulus `√α` between `(1−√α)²` and `(1+√α)²`; real negative between `(1+√α)²` and `2(1+α)`; unstable above `2(1+α)`.

**While you are there:** VI-D says the instability signature is absent even at `Δt k a⊥ > 12`, and does not say why. Your own numbers explain it. At `a⊥ = 37.8` the proportional band is `c_max/a⊥ = 0.0011`, smaller than the measured tracking error 0.0016, so the cluster is never inside the linear regime where the bound applies. One sentence, and the reader stops wondering whether you got lucky.

### B5. Proposition 3 needs a seed-consistency hypothesis, and the experiment is close to violating it

Prop 3 proves equivariance "by induction from the second tracking step onward" and exempts the degenerate guard. But `t_ref = v̂₀` at first contact is also frame-dependent, and unlike the guard it is *always* used. If the seed resolves to the opposite sign in the rotating frame, the whole ride goes the other way and the paths are not equivariant to a bounded residue, they are opposite. The bounded-residue claim quietly assumes the seed sign agrees across frames.

At the experiment's start (0.05, 0.40): `|v| = 0.107` and the frame transport term `Ω‖p‖ = 0.2 × 0.403 = 0.081`. The transport term is 76% of the flow speed at the seed point. The sign happened to survive, but the margin is 1.3×, not comfortable, and the paper reports neither number.

**Fix:** add the hypothesis explicitly, something like `Ω‖p_seed‖ < |v̂₀ᵀ t_raw|`, report both values at the experiment's start, and ideally sweep Ω upward to find where the seed flips. That sweep is one afternoon and it turns a soft spot into a characterized boundary. It also gives Prop 3 a testable condition rather than a footnote.

### B6. A comparison is promised in V-B and explicitly disclaimed in VI-H

Section V-B: *"the same network over the same 28-h window ... so the two methods are directly comparable on the same data."*
Section VI-H: *"No controlled head-to-head against the straddle family [17]–[19] or the onboard-FTLE strategy of [21] was run."*

The reader is told the record was chosen for comparability and then told no comparison happened. Pick one.

Given your position that a head-to-head against [17] would read as adversarial, the right move is to drop the comparability clause from V-B (keep "the same record," which establishes provenance and is a strength) and replace the comparison with a **capability table** in I-A. That is not adversarial, it is a positioning device, and you have every entry already:

| | robots | latency to first estimate | prior commitment | saddle handling | objective | advection |
|---|---|---|---|---|---|---|
| [17]–[19] straddle | 3 to N | none | manifold identity | reported failure | no | advected |
| [21] onboard FTLE | — | integration horizon | none | — | no | advected |
| this paper, D | 6 | none | none | traverses | no | commanded |
| this paper, s₁ | 6 | none | none | traverses and captures | yes | commanded |

That table does more for the intro's gap claim than the whole Related Work paragraph does.

### B7. The ocean section has no quantitative result

Section VI-F is the paper's only real-data experiment and it contains no measured accuracy number. "Follow the dominant ridge ... closely" and "its magnitude in the tracking corridor was empirically small compared to the strain" are both unquantified. The 84/100 jitter figure is about repeatability, not accuracy.

**Fix, all from data you already have:**
- Mean and max distance from each tracked path to the nearest 24-h FTLE ridge point, in km, stated against the 2 km grid and the 5.4 km formation radius. This is the number people will cite.
- The divergence ratio: `|∇·v| / ‖S‖_F` along the corridor, median and 90th percentile. One number retires the `s₂ = −s₁` substitution question.
- Realized speed distribution for both trackers, which also supports the vehicle-feasibility discussion in VI-H.

---

## 3. Math audit

### 3.1 Verified correct

I re-derived these independently. All correct, no action needed. Listing them so you know what the rest of the audit does and does not cover.

- `tr(SW) = 0` for symmetric S and antisymmetric W, hence `det J = det S + det W` exactly in 2D, hence (5).
- (4), the eigenvalue branches under `tr J = 0`.
- (6), `D' = D + Ωω + Ω²`, and the consistency of `ω' = ω + 2Ω` with the added solid-body field `Ω(−y′, x′)` in V-C.
- The frame convention in the source is `Q̇Qᵀ = Ω[[0,−1],[1,0]]`, which is right. (The PDF renders it as `QQ̇ᵀ`, which would be sign-flipped. Extraction artifact only.)
- (7) through (10). Every closed form checks: D, ∇D, `H_D`, the Okubo-Weiss diamond `x_f ± y_f ∈ ½ + ℤ`, cross-track curvature `2π⁶A²` on x = 0, the crest at the origin, `s₁ = −π²A|cos πx_f cos πy_f|`, and the six global minima at `s₁ = −π²A`.
- (13) through (16). I expanded `det Ĵ(p)` symbolically; the gradient and all three Hessian entries match, including the cancellation `a₄b₄ − a₄b₄` in the off-diagonal.
- The four discarded third-order products in `D_xx`. Exactly as stated: `D_xx = 2(u_xx v_xy − u_xy v_xx) + (u_xxx v_y + u_x v_xxy − u_xxy v_x − u_y v_xxx)`.
- (42) and the floor. On the separatrix `Ĥ_xx = 2π⁶A² sin²πy_f`, `e = 2π⁶A² cos²πy_f`, true `D_xx = 2π⁶A² = 19.23`. At |y| = 0.45, `e/D_xx = 97.6%`, reported as 98%, fine.
- (21), the averaged position-noise channel, and the predicted component correlation. At (−0.3, 0.1) the analytic `∇u·∇v/‖∇u‖‖∇v‖ = 0.4472`, reported as 0.44 predicted against 0.41 measured. Correct.
- **(22)**. `γ₁ = 2/√10` and `γ₂ = 8/√10` in the source are both correct. I derived the pentagon-plus-center gains from the ring harmonics: `Var(a₂) = 0.4 σ²/ρ²` so `γ₁ = √0.4 = 2/√10`, consistent with the stated gradient covariance `(4/10)σ²ρ⁻²I`; `Var(a₅) = 6.4 σ²/ρ⁴` so `γ₂ = 8/√10`; and the mixed coefficient `Var(a₄) = 1.6 σ²/ρ⁴` so `4/√10`. All three match. (The PDF renders these as `√2/10` and `√8/10`, which would be wrong by 4.5×. Extraction artifact only, but check the compiled PDF's `\tfrac` rendering.)
- (23). Rice mean, both limits, and `σ_r = γ₁σ_eff/(√2 ρ)` from the independence of `a₂, a₃, b₂, b₃`. Correct, including the `σ_r/(2r)` eigenframe angle from the half-angle map.
- Proposition 2. Every term of (13), (15), (16) pairs one `â` with one `b̂`, and `ηᵘ ⊥ ηᵛ`. Correct.
- (24) and `ρ̃* ∝ ν^{1/(p+1)}`, cube root at p = 2.
- Remark 2. `μ` affine minus a norm of an affine map is concave, so `Ĥ_s1 ⪯ 0` for every field and formation. Correct and strong.
- (36), the Lyapunov argument, and `V̇ ≤ −2k a⊥ tanh(1) V` inside the proportional band via concavity of tanh.
- (38), including the `artanh` form and the `δ < k c_max` proviso.
- **Table II in full.** `a⊥,D = csc²(πy_f)` follows from (42); `a⊥,s1 = g⊥π⁴A|cos πy_f|` follows from expanding `|cos πx_f|` about the separatrix. Both column values and both fractions reproduce: 1.46 at |y| = 0.35, 2.60 and 2.92 for s₁, 36.9% and 10.4% for D, 77.8% and 0% for s₁. The analytic transverse curvature 6.888 at (0.03, 0.25) also checks.
- Ocean unit chain: `ρ = 0.105 → 5.4 km`, `c_max = 0.04 → 3.43 m/s`, 168 steps × 600 s = 28 h, 29 hourly frames. All consistent.
- The `‖J‖`-scaling argument for why `σ_p = 0.01` and `σ_uv = 0.01` give nearly identical rates: at the straddling start `½‖J‖²_F = 0.77`, so `σ_p = 0.01` alone is slightly the weaker channel, and 44.5% vs 43.7% is the right ordering.
- The VI-D SNR arithmetic: flow 0.1426 at (0, 0.35) gives 71 at `σ_uv = 0.002`; `|a₄| = 1.408` against the combined noise `√(γ₅²+γ₄²)σ/ρ² = 1.006` gives 1.40; ratio 50×. All three numbers land.

### 3.2 Errors, overreach, and imprecision

| # | Location | Issue | Fix |
|---|---|---|---|
| M1 | IV-E | Instability at `(1+√α)²`. Wrong, it is `2(1+α)`. | See B4. Text only, no numbers change. |
| M2 | (37) and Thm 1(ii) | `m` is called "the realized tangential speed" and the time bound is `L_Γ/m`, but (37) omits the navigation gain k, while (34) has `ṗ_c = k δ`. Off by a factor of k. | Either fold k into (37) or write the bound as `L_Γ/(k m)`. |
| M3 | II-B | "Q = ¼(∇·v)² − D" holds only under the quarter-normalized Okubo-Weiss convention. The common convention `Q = s_n² + s_s² − ω²` with unhalved strains gives `Q = −4D` for incompressible flow. | Write Q out once, explicitly, or cite the normalization to [31]. One clause. |
| M4 | II-B | "nearby particles separate at the instantaneous exponential rate `√−D`". `√−D` is the eigenvalue rate of the frozen-time linearization, which is *not* the material stretching rate. The material rate is `s₂ = −s₁`, and the two coincide only where ω = 0. This is the paper's own thesis, so stating it loosely here undercuts II-C. | Add the qualifier. You already do it correctly in II-D at the crest. |
| M5 | III-D | "The two channels stop being interchangeable **above first order in σ_p**". The correlation `σ_p²∇u·∇v` arises from the leading-order `Jξ` term, not from a higher-order one. | Reword: position noise correlates the two component errors at `∇u·∇v/(‖∇u‖‖∇v‖)` at leading order, which measurement noise does not. |
| M6 | IV-C, VI-B | "the eigenvalues stay closely approximated by `±|Ĥ_xx|` everywhere". The one measured pair is [−0.72, 0.43], a 67% asymmetry. | Drop "closely," or report the observed asymmetry range. |
| M7 | II-D | `p*₁, p*₂` are called "the two **interior** saddles" in the same sentence that attributes four more crossings to "the domain walls." Both sit at y = ±0.5, which *is* the wall of `[−1,1] × [−0.5,0.5]`. | Call them the two separatrix saddles and the four corner crossings, or widen the stated domain. |
| M8 | V-B | "spanning 0.300° of latitude and 0.363° of longitude" with a `[−0.65, 0.65]²` box gives 25.7 km per world unit, not 51.4. The figure only works if these are **half**-extents, which the next paragraph confirms. | Write `±0.300°`. |
| M9 | III-D vs VI-D | Truncation bias has "period 72°" in III-D and "36° period" in VI-D. Both are true (signed bias 72°, magnitude 36°) but they read as a contradiction. | One clause: the signed offset has period 72° and changes sign at each mirror phase, so `|x_c|` has period 36°. |
| M10 | VI-E | "the tangent the s₁ tracker rides comes from the *first-order* fitted coefficients, one order better in noise gain." The *axis* is first-order, but the *choice between axes* is `argmax|∇ŝ₁ᵀe|`, and `∇ŝ₁` is second-order. VI-D proved that choice is the weak link. As written, VI-E scores a point that VI-D has already retired. | Restate: the axis estimate is an order better, and the selection between axes is not, which is exactly why VI-D's failure sits where it does. |
| M11 | Remark 2 | A reader will ask how gradient descent works on a surface the paper just proved concave. It works because the fit is recomputed at the new centroid each cycle and (18) is the exact gradient of the fitted field. | One sentence. Costs nothing, prevents a reviewer objection. |
| M12 | VI-D | "predicts trial-mean `|x_c|` of 0.0019 and 0.0079" is ordered D-then-s₁ while the preceding sentence lists them s₁-then-D. | Pair them explicitly. |
| M13 | Abstract | "roughly one fifth that noise level." The measured ratio is 0.0079 against 0.0015–0.002, so about 4 to 5×. | "About a factor of four to five." |
| M14 | Notation | `e` is the truncation floor, `e₁, e₂` are the strain eigenvectors, and `e` is Euler's number in `α_mom = e^{−Δt/τ}`. All three appear within a page of each other. | Rename the floor. `e_H` or `b_H`. |
| M15 | Notation | `L` is the length scale in (24), the trench length `L_Γ` in Thm 1, and 51.4 km in V-B. | Disambiguate at least the first two. |

---

## 4. Narrative arc

### 4.1 The gap, and whether it is filled

**Gap as stated:** prior trackers either commit to the structure's identity in advance (straddle family) or pay an integration latency (onboard FTLE). Clear, specific, correctly cited, and true.

**Is it filled?** Yes for latency, unambiguously. Yes for identity commitment, with one honest caveat you should surface: the D law does not commit to attracting vs repelling, but it does require an indefinite fitted Hessian, which is itself a structural precondition (see B1). Say "no commitment to attracting or repelling identity," which is what you actually deliver and is still a real advance.

**What the gap statement misses:** the intro sets instantaneity up as pure gain. It is not. Instantaneous criteria are not in general reliable indicators of the material transport skeleton, which you say honestly in VI-F but only after the reader has spent fifteen pages assuming otherwise. Move one sentence of that caveat into I-A. It costs nothing and it inoculates you against the reviewer who raises it as a discovery.

**Missing from the gap statement entirely:** objectivity. No prior robotic tracker is frame-indifferent. That is your strongest claim to novelty and it appears in the intro only as the third of four contributions, phrased defensively ("stays valid when a team cannot certify its own orientation"). Lead with it.

### 4.2 Does the contribution list make it clear how the gap is filled?

Partly. Four contributions are listed: estimator, D tracker, s₁ tracker, validation. Problems:

- Contribution 4, "validation," is not a contribution, it is an obligation. Spending a quarter of the list on it dilutes the other three.
- The **finding** is missing. The abstract's headline (objectivity vs noise tolerance, with a diagnosed mechanism) is not in the intro's list at all. That finding, plus the selection rule in VI-G, is more novel than the D tracker.
- "Galilean-invariant" appears once, in contribution 2, and is never used again. Either define it against objectivity in II-C, which would be genuinely useful (D survives constant-velocity frame changes, fails under rotation, by (6)), or delete the term.
- "Hybrid controller" in contribution 2 is never formalized. There is a switching structure (the band test) but no dwell-time or chattering analysis, and VI-H concedes chatter is possible. "Hybrid" invites a hybrid-systems reviewer. Drop it or defend it.

**Suggested list:** (1) estimator with minimality, conic characterization, and exact heading-isotropic gains; (2) two trackers reading the two terms of one splitting identity from one fit; (3) closed-loop frame equivariance, proved and demonstrated, with the D counterexample; (4) the tradeoff and its diagnosed mechanism, plus the selection rule.

### 4.3 Test plan against results

Every one of the four families and all four supporting checks report. Gaps run the other direction: results with no plan entry.

| Test plan item (V-C) | Result | Status |
|---|---|---|
| 1. Clean runs, six starts, acquisition step, closest approach, settle vs continue, pairwise gap | VI-B | complete |
| 2. Monte Carlo noise grid, straddle retention, tracking error, which saddle | VI-D, Table III, Fig 5 | complete for s₁ |
| 2. "Where failures occur, the sweeps supply noise-free inputs to individual channels" | VI-D, s₁ only | **partial**: promised generally, delivered only for s₁. Either do the D-channel ablation or scope the sentence to the s₁ tracker. |
| 2. "each trial records which saddle it settles at" | used for s₁ only | **partial**: the D tracker never captures, so the quantity is undefined for it. Say so. |
| 3. Objectivity | VI-E, Fig 6 | complete |
| 4. Ocean, plus 10×10 jitter for D | VI-F, Fig 7 | complete but unquantified, see B7. No jitter test for s₁, asymmetric. |
| 5a. Estimator sweep, four s₁-channel quantities | VI-A | complete |
| 5b. `r_band` sweep 0.05 to 0 | VI-B | complete |
| 5c. Acquisition sweep, 20 000 trials | VI-D | complete. Does not say **which tracker**. Fix. |
| 5d. Time-varying spot check | VI-C | complete |

**Results with no test-plan entry** (add each to V-C, one line apiece):

- The 20 000 gridded starts and 10 000 random starts establishing the reachable set and the 47.4% / 47.1% arrival rates (VI-B). This is a substantial experiment appearing only in Results.
- The start relocation to (0, 0.15) as a controlled manipulation (VI-D). It is one of your best pieces of evidence and it appears from nowhere.
- The hop-rate and one-step-reversal instrumentation (VI-D).
- The open-loop prediction of closed-loop tracking error from the fit alone (VI-D). Also one of your best results, also unplanned.
- The initial-heading quartile analysis over 40 000 trials (VI-D).

### 4.4 Theory to results traceability

Almost everything lands. Three exceptions.

**(24) is never tested.** The optimal-radius result `ρ̃* ∝ ν^{1/3}` is derived, used qualitatively ("no single radius suits both"), and then dropped. VI-A halves ρ once to confirm linear-in-radius scaling of the truncation spread, which tests the truncation term but not the optimum. Either verify the cube root open-loop (cheap: sweep ρ at three noise levels, plot the error minimum) or state that (24) is used only to establish that the two optima differ.

**Remark 1 is never measured.** The conic-degeneracy condition is stated, used to justify the formation, listed first among the limitations, and never observed. Report `κ(Φ)` range over the runs. You have it in every log. Two numbers and one sentence, and Remark 1 stops being decorative.

**(41) appears in the middle of Results.** The identity `∂D/∂n = (ω/2)(∂ω/∂n) − 2s₁(∂s₁/∂n)` is pure theory, follows directly from (5), and explains why the two trackers coincide on the benchmark and diverge on the ocean. It belongs in II-C as a third consequence of (5). Right now it arrives in VI-B introduced by "Differentiating **it** across the structure," with no antecedent for "it," in a paragraph about the degenerate-frame guard. See §5.

**Everything else traces cleanly**, and some of it beautifully: (22) to VI-A's 2.1%; Prop 2 to the signed-mean-error check; Remark 2 to the measured negative-semidefinite `Ĥ_s1`; the floor `e` to (42) to the [−0.72, 0.43] vs [18.3, 19.2] pair to the traversal-past-saddle behavior; (6) to the 1.219 gap; III-D's orientation dependence to VI-D's 36° period. That last chain, from a harmonic-annihilation argument to a measured closed-loop tracking error, is the best theory-to-experiment thread in the paper. Do not let it get trimmed.

### 4.5 Are all results discussed?

Yes, with one orphan. **VI-C (time-varying spot check)** is reported and then never referenced again. Its label `sec:time_varying` is the only unreferenced label in the source. It is not in the abstract, not in the conclusion, and VI-H's future-work item about a time-derivative correction does not cite it. Either cross-reference it from VI-H and add half a sentence to the conclusion, or fold it into VI-H as evidence motivating the future work.

---

## 5. Continuity

**Introduced with no lead-in:**

- **(41) in VI-B.** "Differentiating **it**" has no antecedent. The nearest noun phrase is the degenerate-frame guard. Move the derivation to II-C and leave a back-reference here.
- **"the middle island"** (V-B, VI-F, Fig 7 caption, conclusion). Used four times, never named. Name it once.
- **"Decabot"** appears as a Table I column header on page 7 before IV-B introduces it. Gloss it in the Table I caption.
- **`s_trim = 0.3`** for the ocean appears in Table I with no discussion anywhere, despite being a 6× change from the double-gyre value and despite VI-F resting a claim on the terminal test not firing. See B2.

**Introduced with no follow-up:**

- **"Galilean-invariant"** (I, once).
- **"hybrid controller"** (I, once).
- **Remark 1's conditioning limit.** Forward-referenced to VI-H, which says monitoring `κ(Φ)` is "not implemented here." So the forward reference resolves to a non-result.
- **VI-C**, as above.
- **The five-robot constrained estimator** (III-A). Well argued, then never revisited. One sentence in VI-H or the conclusion ("a divergence-constrained five-robot variant is available where incompressibility is defensible") would close it.
- **"straddle retention ... used here as an internal metric only"** (V-C). It then does real work in VI-D and Fig 5. Drop "internal metric only," it undersells your own data.

**Figures and tables: all seven figures and all three tables are referenced in the text, and there are no dangling `\ref`s or uncited bibitems.** I checked programmatically. Two notes:

- Fig 5's axes are labeled `σ_uv (m/s)` and `σ_p (m)`, but II-D states the double-gyre experiments are non-dimensional throughout. Strip the units from the axis labels.
- Fig 3's caption ties the dash-dotted line to Table III(a). Good, keep doing that.

---

## 6. Clarity, redundancy, and phrasing

- **VI-G: "Read together, the four conditions give a selection rule."** They are four experiment families or four operating conditions, not conditions in the mathematical sense used everywhere else in the paper.
- **VI-E: "mean distance to the true trench network of 0.000 inertial."** Give the precision, or write `< 5 × 10⁻⁴`.
- **VI-D footnote 1** is attached to a sentence about the s₁ tracker but reads as though it is about the D tracker's box. Rewrite as: the s₁ sweep's widened box recovers 30 of 54 trials (0.5% of the cell) that the narrower box would have scored as exits. Also add the differing exit box to the Table III caption, since it is the one place the two trackers are not scored identically.
- **VI-D:** the D tracker's floor "in the low tens of percent" is explained as an unbiased random walk; the s₁ tracker's 0.2–0.7% floor is not explained. One clause: the s₁ law settles wherever it stops, so it does not random-walk into the far saddle.
- **Table III** reports point estimates with no uncertainty. At 10 000 trials the 95% half-width is under 1%. State it once in the caption.
- **IV-D:** `β c_max tanh(1)` will read as a magic constant. One clause: the ride speed is the saturation of a unit command.
- **Three verbatim quotations from [17]** (two in I-A, one in V-B). Under IEEE norms and given your own preference for non-adversarial positioning, paraphrase all three. Quoting a competitor's stated limitation verbatim, twice in one paragraph, reads more pointedly than the surrounding prose does.
- **Section I-A** runs long and mixes three functions: LCS background, the straddle critique, and the estimation lineage. Consider splitting the straddle critique into its own short paragraph so the capability table from B6 has a home.

**Redundancy candidates**, if you are still trimming toward 16 pages:

- The floor `e` is explained three times (III-D, IV-C, VI-D). Keep the III-D derivation and the VI-D closed form, compress the IV-C restatement to a cross-reference.
- The saddle-crossing behavior is described in II-D, IV-C, IV-E, and VI-B. Two of those can become cross-references.
- Table II's prose introduction in IV-E partly duplicates the caption.

---

## 7. Abstract, intro, conclusion consistency

**Numbers all reconcile.** 2.1% coefficient match, within 5% cliff-to-threshold agreement (5.3% strictly), 0.025 vs 1.219 gaps, 84 of 100 jittered starts, 28 hours, six-robot minimality. Checked each against the body.

**Mismatches:**

| Item | Abstract | Body | Action |
|---|---|---|---|
| "fundamental tradeoff" | asserted | VI-D ablation and VI-H fixes contradict it | B3 |
| "roughly one fifth" | 1/5 | ≈ 1/4.5 | M13 |
| objectivity ranking | third of four | your strongest result | promote in abstract and intro |
| the tradeoff finding | in abstract | absent from the intro's contribution list | add as contribution 4 |
| VI-C time-varying | absent | reported | mention in one clause or fold into VI-H |
| "both controllers follow the same dominant ridge for 28 hours" | reads as full agreement | they separate near landfall | "for nearly the full record" |

The conclusion is the most disciplined of the three. It correctly attributes the s₁ ceiling to the tangent-sign mechanism rather than to objectivity, correctly reports the non-firing of the terminal test, and correctly scopes the ocean result. Bring the abstract up to the conclusion's standard, not the other way around.

---

## 8. Reordering and additions

**Reorder:**

1. **(41) from VI-B to II-C**, as the third consequence of (5). It is the paper's own explanation for why VI-B's agreement is a coincidence, so it should precede the demonstration, not interrupt it.
2. **The reachable-set experiment** from VI-B into V-C.5, with the results staying in VI-B.
3. **Table II's "fraction past bound" columns**: either move to VI or state in the caption that they are analytic predictions at the operating point evaluated on the realized fit. Right now an analysis section contains what look like results.
4. **The capability table (B6) into I-A**, ahead of the straddle critique.
5. Optional: swap II-D and II-E so the problem statement follows the surrogates directly and the double gyre illustrates the stated problems. Low priority, moderate churn.

**Add, in descending value per hour of work:**

1. **Implement the s₁ margin gate** from VI-H (hold the tangent when `||∇ŝ₁ᵀe₁| − |∇ŝ₁ᵀe₂||` falls below `γ₂σ_eff/ρ²`) and re-run one noise axis. You already predict it will work, you already have the noise floor in closed form, and it converts the paper's weakest result into a designed defense with a theoretical trigger. This is the highest-value addition available and it is a few lines of code plus one sweep. If the schedule truly forbids it, at least report what the margin distribution looks like at σ = 0.002 so the reader can see the gate would fire.
2. **The ocean quantitative metrics** (B7). Zero new simulation.
3. **The ocean `β` and `ŝ₁` diagnostics** (B2). Zero new simulation.
4. **`κ(Φ)` over the runs** (§4.4). Zero new simulation.
5. **The Ω sweep for the seed-flip boundary** (B5). One afternoon, and it upgrades Prop 3 from a proof with an unstated hypothesis to a proof with a measured margin.
6. **The capability table** (B6).
7. **Formation shape RMSE distribution during runs.** You already compute it for the collapse guard at 0.30. Reporting the realized distribution retires the "does the cluster layer actually hold formation" question before a reviewer asks it.

---

## 9. Priority action list

**Must fix before submission**

1. B1. Reconcile Theorem 1's exact-estimate hypothesis with the D tracker's dependence on the fitted frame. Prefer changing the guard.
2. B2. Retract or defend "a property of the field and not of a setting," with ocean `ŝ₁`, `β`, and speed numbers.
3. B3. Qualify the abstract's "fundamental tradeoff," and name the part that is structural.
4. B4. Correct the instability threshold to `2(1+α)`. No numbers change.
5. B6. Remove the "directly comparable" promise or deliver the capability table.
6. M2. The missing `k` in (37).

**Should fix**

7. B5. Seed-consistency hypothesis in Prop 3, with the two numbers at the experiment's start.
8. B7. Quantitative ocean metrics.
9. M3, M4, M5, M7, M8. Okubo-Weiss normalization, the `√−D` material-rate qualifier, the correlation-order wording, "interior saddles," and the latitude half-extent.
10. Move (41) to II-C. Fix the "Differentiating it" antecedent.
11. Rewrite the contribution list: drop "validation," add the tradeoff and the selection rule, delete or defend "Galilean-invariant" and "hybrid."
12. Add the five unplanned experiments to V-C. Say which tracker the acquisition sweep used.
13. M14. Rename the floor `e`.

**Nice to have**

14. The margin gate implemented and swept.
15. `κ(Φ)`, shape RMSE, and Table III confidence intervals.
16. Fig 5 axis units, Table III exit-box footnote, VI-C cross-reference, "middle island" named.
17. Paraphrase the three quotations from [17].
18. Trim the triplicated floor and saddle-crossing discussions if you are still cutting length.

---

## 10. Closing note

The paper's real argument is not "here are two trackers." It is: *one local quadratic fit exposes both terms of a splitting identity, and which term you steer on determines whether your path is a property of the flow or a property of your observer.* Everything strong in the draft serves that argument, and almost everything I flagged above is a place where the writing claims either more or less than that argument earns. The estimator is sound, the controllers are sound, the failure analysis is better than the field's norm, and the objectivity experiment is the sort of result that gets a paper remembered. Fix the five overreaches, correct the one algebra slip, put numbers on the ocean, and this is a strong T-RO paper.
