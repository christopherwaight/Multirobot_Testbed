# Plan 0.2: Themes the newer papers add to the thesis

Date: 2026-11-19. Read for this note:

- `Separatrix_and_OW_Paper/Draft_14.tex`, with `Plans/Draft_14_changes.md`
- the revision, `cwaight_systems_paper_with_revisions2.tex` (file of 2026-11-19 09:54). I took "the revisions document" to mean the revised paper. `Response_to_Reviewers.tex` is reference only.
- IDETC 2025, `sources/idetc2025_testbed.txt`
- the thesis chapters as built on 2026-11-18, plus Plan0 and Plan0.1

Nothing in the thesis was edited. Every item below is a proposal.

## Short answer

Yes. The fit-order ladder (Ch. 4) is still the spine. The three papers now also share structure that the thesis states only inside one chapter, or not at all. The strongest three:

1. **One flow, three rungs.** Ch. 6's gyre cell is one gyre of Ch. 7's double gyre, so the thesis already has one benchmark flow seen at every order. The first-order basin boundary (det J = 0) meets the separatrix at the crest where the D tracker loses its along-trench slope.
2. **The cluster steers to a feature of the fit, not of the flow.** Every rung has a stated condition for when the two coincide, and a quantity the fit can compute to check it.
3. **Speed over loop gain.** The friction ring, the orbit limit cycle, drift trailing, the rotating-frame hold lag and the time-periodic outrun all follow one law. Its traces reach back to the IDETC drift.

What the thesis already has: the ladder (Sec. 4.3), Ch. 8 (scalar landscapes, sampling geometry, noise growth with order, scalability), and the primitive library (Table 10.1). The themes below are not in it, or appear only inside one paper's chapter.

---

## Backbones: structure the whole thesis can hang on

### B1. One flow, three rungs

The revision's nonlinear test field, v = 0.1π(cos πx sin πy, −sin πx cos πy), is the left gyre of Draft_14's steady double gyre with A = 0.1, recentered on the gyre center. I checked the shift: putting x_f = 0.5 + X and y_f = 0.5 + Y in Draft_14's field gives the revision's field term for term. So both rungs already ran on the same flow:

| Rung | What it did on this flow | Source |
|---|---|---|
| 0 | V2S cannot tell the center from the saddles (speed vanishes at both); lost every gyre orbit | revision VI-C, Table baselines |
| 1 | Initial sign of det Ĵ decided which critical point the cluster reached, in all 2000 trials; orbits lost where det J = 0 (the diamond \|x\| + \|y\| = 0.5) | revision IV-C, IV-D |
| 2 | D tracker and s1 tracker ride the separatrix between the two wall saddles | Draft_14 IV |

These two papers view the same function, D = det J, at two orders. At first order the cluster reads one value of it, and its sign sorts the starts into basins. At second order it reads the landscape, and the trenches are separatrices. The two meet at one point. In Draft_14's coordinates D = 0 on the separatrix only at the origin, the crest. In the revision's coordinates the diamond det J = 0 meets each cell wall at its midpoint. That is the same point. The curve that bounds the first-order basins crosses the separatrix exactly where the D tracker's Newton step stalls and it switches to riding the measured flow.

- **Where:** Ch. 3 benchmark section (one sentence: Ch. 6's gyre cell is one gyre of Ch. 7's double gyre). A composite figure opening Ch. 8: the D landscape with the det J = 0 diamond, the Ch. 6 basin starts, and a Ch. 7 ride. The same figure could serve as the "estimation-hierarchy diagram" Plan0 suggests for Ch. 1.
- **Cost:** text plus one figure. If the gyre-cell trial outputs behind `fig_nonlinear_triangles_2026.pdf` are saved, the figure is plotting only. If not, it needs a three-robot sim, and that needs your OK first.

### B2. The cluster steers to a feature of the fit, not of the flow

Each rung navigates to something the fit defines and then states when that equals the feature in the flow.

| Rung | Navigates to | Equals the true feature when | Fails visibly on | Source |
|---|---|---|---|---|
| 0 | minimum of the fitted magnitude plane | speed grows equally in every direction | anisotropic node (V2S reached p\* in 6% of trials); anisotropic saddle (Ch. 8) | revision VI-C, ch08 |
| 1 | zero of the fitted affine field, p̂\* − p\* = −Ĵ⁻¹ v̂(p\*) | the field is affine from the cluster out to p\* | gyre and rotational saddle: estimate 2.6 to 4.9 mm off; basin set by det Ĵ sign | revision II-B, IV-C |
| 2 | trench of the fitted D or s1 | rotation on the separatrix is small against strain (necessary, not sufficient) | bent gyre, Bickley jet: "each tracker rides a trench of its own field" | Draft_14 II-D, IV-E |

Each rung also has a quantity the fit can compute about itself:

| Indicator | Order | Flags | Tested? |
|---|---|---|---|
| σ_min(Ĵ) | 1 | near-singular critical point estimate | stated; deliberately not used as a gate (revision III-D) |
| eigenvalue margin to the classifier band (about 6°) | 1 | type label near a boundary | implicit in the classification results |
| \|ω̂\|/2r̂ | 1 | trench unlikely to mark a separatrix | yes: retention falls from 97% near 0.1 to at most 9% above 0.5 (Draft_14 Table II, prelim) |
| least-squares residual | any, needs more than n_k robots | measurement noise level | future work (Draft_14 VI) |

Plan0 item 3E asks the thesis to "distinguish separatrices from their surrogates". This answers that at every rung, not just the second. It also supplies the inputs that "adaptive switching between estimator orders" (Plan0, optional) would need.

- **Where:** Ch. 4, a short section after the ladder ("Fitted features and true features") with the first table; the indicator table in Ch. 8 or Ch. 10.
- **Cost:** text and two tables, all from existing results.

### B3. Lower rungs run inside higher ones

The D tracker is a composition of all three rungs. In your pseudocode register:

```
D tracker, one control cycle
  Fit the quadratic to six readings: coefficients of order 0, 1, 2
  If stop latched:
      command the attraction law of Ch. 6                     (order 1)
  Else if the fitted flow has a saddle inside the formation:  (orders 0, 1)
      latch stop
  Else:
      w1, w2 ← eigenvectors of the fitted Hessian of D        (order 2)
      sign w1 so that v̂0 · w1 ≥ 0                             (order 0)
      across ← Newton step on D along w2
      along  ← crest ? T (v̂0 · w1) : Newton step along w1
      command k [ sat(along) w1 + sat(across) w2 ]
```

Draft_14 gives the reason for one of these choices: the stop test "uses only the zeroth- and first-order coefficients, the least noisy the fit returns". The s1 tracker shows the opposite case. It picks between e1 and e2 (order 1) using ∇s1 (order 2) and takes its sign from memory. Reversals concentrate where the order-2 along-trench slope goes to zero. The future-work fix in Draft_14 VI (the eigenvector nearest the previous tangent) moves that choice down to order 1.

Two ideas come out of this:

- **A candidate design rule:** make each decision with the lowest-order coefficients that can make it, since coefficient noise grows as ρ^(−q). This is my synthesis, not a tested result. The D tracker's along-trench direction w1 is itself an order-2 quantity, so the rule explains the sign and stop decisions, not the whole noise gap between the trackers.
- **Topological skeleton:** critical points are nodes and separatrices are edges. On the benchmark the separatrix x = 0 is a heteroclinic connection: the flow runs down it from p2\* to p1\*. The D tracker travels that edge and captures the node at its end with Ch. 6's test and law. Sec. 9.2.5 ("Additional Topological Features") lists heteroclinic orbits as future work needing a second-order fit. On the steady benchmark, Ch. 7 has already done one. Scope it to that field, since Draft_14 IV-E shows the time-periodic and jet cases do not hold up as cleanly. The visualization literature calls this the topological skeleton. Verify a citation with `paper-search` before using the term (Tricoche 2001 is in `Reference Papers/`).

- **Where:** a "uses from lower rungs" column in the ladder table (Table 4.1), a "calls" column in Table 10.1, and a rewrite of Sec. 9.2.5.
- **Cost:** text only.

---

## Support structures: quantitative laws that hold on every rung

### S1. Speed over loop gain

Whenever some speed reaches the loop without being cancelled, the cluster settles offset from its target by that speed divided by the loop gain. If the speed exceeds the saturated command, the cluster loses the target.

| Speed the loop does not cancel | Loop gain | Result | Source |
|---|---|---|---|
| dead zone v_c | k | friction ring v_c/k: 0.025 m (Ch. 6), 0.008 (Ch. 7 hold) | revision III-D; Draft_14 III-D |
| outward drift from late turns, T V²/r | g k_r | limit cycle, r\* − r_d = (T V²/r\*)/(g k_r) | revision eq. orbit_cf (rearranged) |
| critical point drifting at speed V | g k | trailed by V/(g k) | revision VI-D, limitations |
| rotating-frame transport, 0.5\|Ω\| | g k | hold trails by 0.5\|Ω\|/(g k); s1 hold lost once 0.5\|Ω\| > g k c_max | Draft_14 IV-D |
| moving separatrix, ε = 0.25 | saturated, g k c_max = 0.103 | outruns both channels | Draft_14 IV-E |
| rung 0: no radial loop | 0 | vector-sum orbit drifts outward without bound | IDETC 4.2 |
| rung 0: V2S commands under the dead zone | | settles about 0.2 m from center; IDETC itself blames static friction | IDETC 4.2 |

The ceiling on the same gain is shared. Both papers run on the same robot layer, so both quote gk < 1/(gL): 21.1, or 16.1 with the lag. At first order the designer sets the loop gain (k). At second order the field sets part of it: k times true over fitted curvature for the D tracker, and k g⊥ ∂²s1/∂n² for the s1 tracker. That is why the s1 path wiggles below y = −0.19, where its gain passes the bound (Draft_14 III-D, IV-B).

- **Where:** a new Ch. 8 section, or the end of Sec. 2.6 (robot dynamics) with each chapter pointing back. Either way it absorbs Plan0.1 item 3 (orbit drift described two ways), because the drift row is the IDETC observation and the limit-cycle row is its explanation.
- **Note:** the dead zone mismatch (0.05 vs 0.025 m/s) is Plan0.1 decision D2. The law holds either way.
- **Cost:** text and one table, all from existing numbers.

### S2. A signal-to-noise ratio for each rung

Each rung's signal is the next derivative of the field times the footprint raised to that order. Each paper found its own threshold:

| Rung | Signal against noise | Threshold found | Source |
|---|---|---|---|
| 0 | \|v\| against sensing error | small vectors gave large direction shifts on the old RGB testbed; saturation below 0.2 dropped from training | IDETC 1, 3.3 |
| 1 | σ_min(J) R against σ_v | covariance prediction holds when σ_min(J) R ≥ 3σ_v | revision IV-C |
| 2 | second derivative × ρ² against 2.53 σ_uv | equal near σ_uv = 0.007; by 0.01 neither tracker keeps the straddle in more than 60% of trials | Draft_14 IV-C |

The rung-0 row has a sharp consequence: direction from a reading fails exactly where |v| → 0, that is, at the critical points rung 1 is built to find. Ch. 8's "Noise Growth with Fit Order" already has the ρ^(−q) scaling. This adds each rung's threshold in one table.

- **Caveats:** rung 0 has no measured threshold, so do not invent one. The rung-1 condition is a validity condition for the covariance, not a failure point. The Draft_14 row's 60% is prelim.
- **Cost:** text and one table.

### S3. The formation is the instrument and the yardstick

Every paper judges success by containment:

- IDETC: the target "falls within the boundaries of the triangle"
- revision: a trial succeeds if p\* lies inside the formation (100% on hardware)
- Draft_14: straddle retention; crest test ‖Ĥ⁻¹ĝ‖ < c_crest ρ; stop test ‖Ĵ⁻¹v̂‖ < ρ; capture ratio q = ρ‖∇ŝ1‖/|ŝ1|

Every threshold is in formation radii, which is why one controller runs on a 1.6 m map and, after a uniform dilation, on a 51.4 km channel (Draft_14 V-A). A table of the dimensionless groups behind every result:

| Group | Governs |
|---|---|
| ρ × curvature / slope | truncation bias |
| S2's signal-to-noise ratio | noise failure |
| g k L | stability |
| v_c / (k ρ) | friction ring in formation radii |
| \|ω\| / 2r | trench-to-separatrix scope |
| Ω / strain rate | frame effects |
| feature speed / (g k c_max) | tracking a moving feature |

From stated numbers (my arithmetic, not in any paper): the friction ring is about 0.14 of the triangle's center-to-robot distance (0.025 / 0.173 m) and about 0.11 of the pentagon radius (0.008 / 0.075).

- **Where:** Ch. 4 or Ch. 8, one short section and the table. This doubles as the "engineering design-guidelines subsection" Plan0 suggests for the end of Ch. 8.
- **Cost:** text and one table.

---

## Method

### M1. Isolation by substitution

All three papers earn their claims the same way: swap one part and hold the rest fixed.

| Substitution | Isolates | Paper |
|---|---|---|
| robot R4 held out of calibration training | whether the calibration generalizes | IDETC |
| analytical vs reconstructed field | fixed map distortion vs time-varying noise | revision |
| matched sim of every hardware run | what the model reproduces (saddle attraction; about half the orbit spread) | revision |
| V2S alone, V2S + oracle, V2S + speed estimate | the missing coordinate vs the steering direction | revision |
| own-trench vs separatrix scoring; analytic trenches | field vs tracker | Draft_14 |
| measurement and position noise swept separately | which noise matters | Draft_14 |
| rotating observer | objectivity | Draft_14 |
| 100-run start perturbation | whether the headline run was selected | Draft_14 |
| oracle fit | fit vs field | Draft_14, planned, not run |

- **Where:** a new section in Ch. 2. "Verification Tools" is currently hardware only, and this would make it the methods chapter its title promises.
- **Cost:** text and one table.

### M2. The testbed-to-model arc, and the modular architecture paying off

The three papers form a sequence:

1. **IDETC** found an outward drift "not predicted in simulation studies". Its future work promised digital twins of the Decabots and controller substitution on a fixed lower architecture.
2. **The revision** delivered both. It fit the robot model (g, L, τ, v_c; RMS 0.029 m/s), used it to explain the drift as a limit cycle, and compared V2S against the Jacobian method with the lower layers held fixed.
3. **Draft_14** runs the six-robot double gyre on the same model ("one time unit equal to one second of the Decabot's identified dynamics") and reuses the same stability bound.

The architecture's IDETC promise is cashed out twice.

- **Where:** one sentence each in the Ch. 2 opening, the Ch. 4 architecture paragraph and the Ch. 5 closing (a backward reference).
- **Side effect:** porting Draft_14 moves Ch. 7 onto the refit model. That removes most of Plan0.1 item 2b's two-model problem.

---

## Smaller commonalities

**C1. Classification at every rung.**

- Rung 0 cannot assign a type: the magnitude looks the same at a saddle and a center (revision VI-C).
- Rung 1 assigns the type from the eigenvalues of Ĵ.
- Rung 2's s1 tracker classifies what it rides by which strain eigenvector it follows. At sea it followed e2 (attracting OECS) on 34 of 42 steps, checked against backward FTLE. Ch. 7 already has these numbers but does not tie them to Ch. 6's classifier.

At both rungs the hard cases sit on the classifier's boundaries. Centers (tr J = 0) got only 45% strict-center labels on hardware. At isotropic points (r = 0) the tangent swaps and reversals start.

**C2. Frame and memory.**

- Ch. 8's Table 8.1 marks which fields are objective. The primitives deserve the same column.
- Rung 1 is not objective either. From D′ = D + Ωω + Ω², an irrotational saddle has D′ = D + Ω², so an observer rotating fast enough reads an incompressible saddle as a center. Draft_14 IV-D already measured the other half: the frame's transport velocity moves the fitted stagnation point by 0.5|Ω|/(π²A). The saddle-as-center step is my derivation, so check it before use.
- "Instantaneous" needs one scope sentence. Estimation is memoryless. Control keeps a latch (the stop test) and, in the s1 tracker, one sign bit, which buys objectivity and causes the reversals.
- Draft_14 III-C's three-level objectivity statement answers Plan0 item 3C.

---

## Trends across the three papers

These are for the Ch. 1 organization paragraph and Ch. 10 "Final Thoughts".

1. **From reporting to explaining to predicting.** IDETC reports an unexpected drift. The revision explains it with a closed form that matches simulation within 3 mm. Draft_14 predicts where both trackers must fail (σ_uv ≈ 0.007) before showing the Monte Carlo.
2. **From best case to stress test.** 10 runs per case (IDETC), then 459 hardware runs with matched sims (revision), then sweeps of up to 10^4 trials, rotating frames, three harder fields and a perturbation replay of the headline run (Draft_14).
3. **Every claim carries its scope condition:** "where the affine model remains accurate out to p\*", "local and linearized", "where strain dominates rotation", "in the tested environments".
4. **From printed map to measured ocean.** Only rung 2 has left the lab, and only in simulation. An optional idea that needs your OK, since it is a new sim: run the rung-1 laws on the same Santa Barbara Channel record. That would make the ocean a second shared flow for B1.

---

## What Draft_14 forces in the thesis anyway

These are not themes. I found them while reading and edited nothing.

1. **Ch. 7 results.** Ch. 7 mirrors Draft_13, whose result figures predate the 2026-10-28 formation sign fix and ran on the legacy robot model (`Draft_14_changes.md`, item 1). CLAUDE.md says to treat such results as suspect.
2. **Abstract.** "1.1 to 5.8 times more measurement noise" and "both primitives successfully follow the dominant transport corridor" are Draft_13-era. Draft_14 V-B shows the s1 tracker's ocean run is the one of 101 tested settings that went downstream.
3. **Ch. 10.** The tracker recommendation is Draft_13's. Draft_14's conclusion replaces it with the trade-off (rotation favors s1; noise and direction favor D).
4. **Unsupported sentences still in the thesis.**
   - Ch. 7 intro still says [26] reports veering near a saddle in flow-tank simulations. Draft_14 removed this because it is not in [26].
   - Sec. 9.1.2 still says "one simulator cannot host both models". Draft_14 judged this false and replaced it.
5. **Ch. 9 future work, partly answered.**
   - "Time-Varying Fields": Draft_14 IV-E ran the time-periodic gyre.
   - "Additional Topological Features": see B3.
6. **Timing.** Draft_14 is awaiting your review, and its Monte Carlo numbers are still `\prelim` pending the overnight runs (IV-C, IV-E, Table II, basin shading). Port it after both.
   - Can go ahead now, since they rest on closed forms or `\rev` numbers: B1's math, S1, S3 and M2.
   - Wait for the overnight runs: B2's retention numbers and S2's Draft_14 row.
7. **Assembler.** In `sources/assemble/port.py`, `'sep'` still points at `Separatrix_and_OW_Paper/Draft_13.tex`. That file now sits in `Archived/` (an unstaged move). `all.py` will fail on every chapter that ports from `'sep'` until the path changes.

---

## Other updates

- [ ] **Update the IDETC-derived passages to match the current simulator** (noted 2026-11-19 at Chris's request).
  - **Why.** Chapters 1, 2 and 5 carry IDETC text written before the Decabots had a model. The simulator is now the 2026-11-16 fit:
    - command gain 0.86
    - delay 0.055 s
    - lag 0.02 s
    - dead zone 0.025 m/s
    - cap 0.3 m/s
  - **How good the fit is.** Simulated without feedback of the measured speed, its velocity error is 0.029 m/s over 718 robot runs, against 0.047 m/s for the October model. Source: `SIM_RERUN_PLAN_2026-11-16.txt`, Secs. 1 and 1b.
  - **The rule.** IDETC results stay as measured. Only the explanation and the description of the simulator change. Match each passage on its exact text, not on line numbers.
  - **Ch. 5, vector-sum orbit.** It blames the outward drift on "momentum effects not captured in previous simulation models". The current model attributes the drift to delay and lag, through the closed form in Sec. 6.4.4. Same edit as Plan0.1 items 3a and 3b, including "so momentum carried the cluster steadily outward" in Sec. 5.6.
  - **Ch. 5, V2S randomized starts.** It says "commands less than 0.05 m/s that are not sufficient for the robots to overcome static friction". The current dead zone is 0.025 m/s; see Plan0.1 item 2c and decision D2.
  - **Ch. 2.**
    - Testbed limitations: "a limited velocity range of 0.05 to 0.3 m/s".
    - Saturation-to-velocity section: saturation below 0.3 (about 0.09 m/s) "not ... strong enough to overcome static friction".
    - The current model moves any command above 0.025 m/s, and its gain is 0.86, not 1. If the IDETC-era robots really behaved differently (before the maintenance of Sec. 2.4.4), say so once with the date rather than overwrite (D2).
  - **Ch. 1 and Ch. 5, simulation-to-reality framing.**
    - Passages: "The gap between simulation and reality highlights the need...", "only validated in simulation", "previously validated in simulation".
    - Add one backward-looking sentence where it fits: the gap the IDETC paper found is now modeled.
    - State what the model still misses: the estimate lead at small orbit radii (model 1.0 to 1.5 cm short), the vortex attraction error (hardware 0.4 to 1.5 cm farther), the four-robot orbit shortfall, and the shape of the low-speed response (no commands below about 0.06 m/s in the campaign).
  - **IDETC future work on digital twins.** This is now done. It is not currently ported into Ch. 9. If any chapter picks it up, mark it done rather than future.
  - **Before Ch. 2 cites the fit.** Copy the fit scripts and outputs (`load_runs.py`, `fit.py`, `fit2.py`, `closed_loop.py`, `*_out.txt`) from the scratchpad path named in `SIM_RERUN_PLAN_2026-11-16.txt` into the repo. A reboot can clear that folder.

---

## Recommendation

Adopt B1, B2 and S1 first. Each spans all three papers, and each needs only text plus one table or figure from existing results. Together they let Ch. 8 say more than "the rungs differ in cost": one flow seen at three orders, one rule for when the fitted feature is the real one, and one law for steady offsets. B3, S2, S3 and M1 follow from them. C1 and C2 are a paragraph each.

Per Waight_style, the next step is a layout showing which section each theme lands in and in what order, for your yes before any prose.
