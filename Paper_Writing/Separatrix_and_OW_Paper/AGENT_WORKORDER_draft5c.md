# Draft 5c revision work order

Source manuscript: `Draft_5c.tex` (18.5 pp, 16,119 prose words). Target: **16.2 pp**.
Companion human artifact: `referee_report_draft5c.html`. Task IDs below map to its M/C numbers.

---

## 0. Rules of engagement

1. **All dispositions are settled with the author. Do not relitigate.** If a task looks wrong, flag it and continue; do not substitute your own judgement about whether a cut should happen.
2. Do not invent numbers. Any figure not already in the manuscript or stated here must be marked `AUTHOR` and left as a placeholder.
3. Do not paraphrase cited works from memory. Tasks touching refs [9], [13], [15] require reading the source PDF.
4. Compile after every tier. Record page count. If a tier overshoots its budget by more than 0.3 col, stop and report.
5. Preserve all `\label` / `\ref` integrity. Deletions that orphan a reference are the primary failure mode of this revision.

---

## 1. Guardrails: do not touch

| Item | Loc | Why |
|---|---|---|
| Table II panels (b) and (c), and the prose comparing them | VI-B-3 | Best quantitative content in the paper. The (b)-vs-(c) contrast is the finding. |
| Three phrases in VI-D: "without the terminal test firing"; "a genuine two-dimensional minimum of s1 was never reached"; "the tracked value rather than a certified settle" | VI-D | Task T30 compresses this paragraph. These three must survive verbatim or in equivalent wording. Deleting them is the only version of this revision that reads as concealment. |
| Eq (17) and the sentence "set to zero by construction, not made small by a good fit" | III-B | Load-bearing for Remark 2 and for M5's rationale. |
| s1 terminal test (30) and its Lyapunov certificate (37) | IV-C, IV-D | NOT the park step. It fires, it is demonstrated, and after M5 it is the only terminal behaviour in the paper. |
| Degenerate branch of (27) | IV-B | Rewrite as a guard (T22). Do not delete: the law is undefined when Hessian eigenvalues share a sign. |
| Remark 1, Remark 2 | III-C, III-D | Together they are the paper's central claim post-M5. |
| Figs 1, 2, 4, 6, 7, 11 | throughout | All earn their space. Fig 2(b) in particular: its truncation floors are why T31 scopes rather than deletes that analysis. |
| Intro contribution 3's scoping clause ("on the double gyre it captures... on a field without a stationary point close enough... it instead traverses") | I | Already correctly scopes capture. Survives M5 unchanged. |

---

## 2. Task graph

```
T1..T9    (mechanical)        no deps
T10..T16  (blocking)          no deps; T11 must precede T40+
T20..T27  (M5 cut)            T20..T26 sequential, single sitting
T30..T37  (ocean/positioning) T30 after T20 (VI-D compression assumes capture cut landed)
T40..T47  (compression)       after T20 and T30
T50       ([14] arXiv)        AUTHOR, start immediately, external lead time
T60..T62  (verification)      gates between tiers
```

---

## 3. Tasks

### Tier 1: mechanical (~40 min)

| ID | Key | Action | Loc | Done when | Model |
|---|---|---|---|---|---|
| T1 | C3 | Retitle section to "Cooperative Second-Order Estimation" | III heading | Heading matches Abstract/contributions/IV/VIII usage of *cooperative* | Sonnet |
| T2 | C1 | "twelve shape variables" → "twelve cluster variables in place of the twelve Cartesian coordinates, nine of them shape" | IV-A | Agrees with V-A's "nine formation shape terms" | Sonnet |
| T3 | C2 | "ρ = 0.075 throughout this paper" → "throughout the double-gyre experiments" | III-D | No longer contradicts ocean ρ = 0.105 | Sonnet |
| T4 | C4 | Delete the word "nearest" | Fig 5 caption | Caption matches VI-B-1 and V-D-2 | Sonnet |
| T5 | C5 | Delete the unused eigenvalue-gap clause "λ2 − λ1 ≥ γ > 0..." | Assumption 1, IV-D | γ appears nowhere in the file | Sonnet |
| T6 | C7 | "This is the layer where..." → "The adaptive navigation layer is where the two trackers differ." | IV-A | Antecedent unambiguous | Sonnet |
| T7 | C10 | "a measurable cost in robustness" → "a measurable cost in convergence rate and in noise tolerance at a single decision point" | Abstract | Matches what VII-C actually reports | Sonnet |
| T8 | §07 | Fix 4 comma splices + 1 dangling fragment. Replacement text supplied in the HTML report §07 for: IV intro, II-F, VII-C, VII-A, VIII. Also break the 70+ word sentences at II-E and VI-B-1. | multiple | No sentence in the listed set exceeds 45 words; no comma splices | Sonnet |
| T9 | M7 | Delete "weakly divergent" and the "fitted divergence is a live diagnostic" sentence. Add: (a) three-sentence five-robot counting (∇·v = 0 plus its two first derivatives removes 3 of 12 coefficients → 5 robots give 10 equations for 9 unknowns); (b) tie-back to the Introduction's no-presupposition line. **No appendix.** | III-A, I | No divergence claim remains unsupported; sixth robot justified by refusal to presuppose | Opus |

### Tier 2: blocking (~3 h)

| ID | Key | Action | Loc | Done when | Model |
|---|---|---|---|---|---|
| T10 | M1 | Rewrite VIII's s1 ocean sentence to a traverse. Delete "loop", "converged to and held", "1.4 km", "11.1 km". Headline becomes: rode the same ridge network for 28 h from the objective strain term alone. **Do not add a VII-E limitations clause** (VI-D already carries the fact four times). | VIII | Abstract, VI-D, VIII agree that the ocean s1 run traversed | Opus |
| T11 | M2 | Unit pass. Strip `m`, `m/s`, `s^-1`, `s^-2`, `m^2`, `s^-1 m^-1` from every double-gyre number **including Table I's Decabot column**. EXCLUSION LIST: Ocean (phys.) column, and the realized 0.5 m/s in V-C. Add non-dimensional note to Abstract. | global | No SI unit attached to a double-gyre quantity | Sonnet |
| T12 | M2 | Add dual normalisation, two clauses: (a) σ_uv against peak flow πA ≈ 0.314 (so 0.01 → 3.2%, 0.002 → 0.6%); (b) σ_uv against the differential span across the formation, already given as ≈ 0.12 in VI-B-3 (so 0.002 → 1.7%). (b) is the correct denominator for the tangent-seed failure. | V-D-2, VI-B-3 | Both denominators stated; seed fragility reported against (b) | Opus |
| T13 | M3 | Add advection as a scope boundary in VII-E, two sentences: exact for the Decabot testbed (robots not immersed; field encoded and read, per [33]), extrapolative for the ocean. One Future Work line. **Do not cite [9] as precedent** (their robots ARE advected, their eq 1a). **Do not report ‖v‖ vs command speed.** | VII-E, VIII | Advection named as a limitation; no partial measurement offered | Opus |
| T14 | C8 | Supply the count of s1 runs the |y| ≤ 0.60 exit box rescues relative to the D tracker's 0.52, or the s1 success rate under 0.52. | V-D-2 | Parenthetical present | `AUTHOR` (needs sweep data) |
| T15 | C9 | Add a clause separating estimator isotropy (Lemma 1, holds) from outcome isotropy (does not hold off-structure, where heading alone decides ~60% at zero noise). | VI-A | The 40% figure no longer sits unqualified beside the isotropy claim | Opus |
| T16 | C6 + §05 | Close the loop on Problems 1 and 2: one referring clause each in IV-B, IV-C, VIII. Add 12-word hysteresis gloss to the −4s_trim condition in (30): capture arms at 4× the first-contact depth and releases at 1×, so noise near contact cannot arm and disarm the test. | II-F, IV-B, IV-C, VIII | Both problems referenced after statement; the 4 is explained | Opus |

### Tier 3: the M5 cut (~half a day, single sitting)

Highest blast radius. Items 20.8 to 20.11 are the ones that leave dangling references.

| ID | Action | Loc | Model |
|---|---|---|---|
| T20.1 | Delete Theorem 1 (terminal capture) and its proof | IV-D, App A | Opus |
| T20.2 | Delete Theorem 2 part (iii) and proof Steps 4 and 5. Theorem 2 becomes two parts: transverse convergence, traversal in the flow direction. | IV-D, App A | Opus |
| T20.3 | Delete the park step, eq (39), and all of Section VII-B including its Lyapunov argument | VII-B | Opus |
| T20.4 | Delete Fig 10 and its file. **Not supplementary** (illustrates a mode that no longer exists) | VII-B | Opus |
| T20.5 | Delete every occurrence of `D_capture` | V-D-1, VII-B | Sonnet |
| T20.6 | Delete the `s_capture` row from Table I; check the caption does not still promise it | Table I | Sonnet |
| T20.7 | Delete the "park-versus-traverse pair runs both settings of D_capture" sentence | V-D-1 | Sonnet |
| T20.8 | Delete contribution 2's clause "with terminal capture available by tightening one threshold" | I | Opus |
| T20.9 | Delete "whose terminal-capture case is one threshold away" | VIII | Opus |
| T20.10 | Delete Problem 1's terminal-capture clause. Problem 1 becomes pure traversal. | II-F | Opus |
| T20.11 | Reword VII-C's "more reliably determined target". Post-cut the D tracker has no target; its advantages are Newton-rate transverse convergence and no dependence on a seeded tangent. | VII-C | Opus |
| T22 | Rewrite the degenerate branch of (27) as a one-sentence well-definedness guard, no certificate | IV-B | Opus |
| T23 | Fold Remark 2's capture consequence into III-D; repoint surviving citations (M5 removes 3 of its 5) | III-D, VI-A, VI-B | Opus |
| T24 | Add Future Work sentence: TRAP cores, eddy cores, non-critical-point targets go to the follow-on paper | VIII | Opus |
| T25 | Move Lemma 1's ring-sum derivation to supplementary; statement and both consequences stay in III-D | App A | Sonnet |
| T26 | Optional: demote Prop 2 to a remark (drops formal environments 10 → 8) | III-A | Sonnet |

**Verification gate G1:** grep for `capture`, `park`, `Theorem 1`, `D_{capture}`, `s_{capture}`. Every surviving hit must belong to the s1 terminal test (30)/(37), the Future Work sentence, or Remark 1/2. Compile and confirm zero undefined references.

### Tier 4: ocean reframe and positioning (~half a day)

| ID | Key | Action | Loc | Model |
|---|---|---|---|---|
| T30 | M4d | Reframe VI-C and VI-D as **demonstrations**, not validation. Claim becomes: the method runs unmodified on measured data for 28 h and produces a path following the dominant FTLE ridge. Collapse the twelve-start spread (1.37–11.06 km, median 3.3) and the 10×10 jitter grid to one sentence each on start insensitivity. Delete the 11.1 km. **Honour guardrail on VI-D's three qualifying phrases.** | VI-C, VI-D | Opus |
| T31 | M4a | Delete the 6 km coverage comparison (48–58% vs 67%). Replace with a sampling argument: at ρ = 0.105 the ring is 5.4 km, spanning ~5 cells of the 2 km grid, the minimum for a six-point quadratic to see curvature rather than interpolation artifact; at ρ = 0.075 it spans under 4. **Do not simply delete** — it is currently the only justification for ρ = 0.105. | V-C | Opus |
| T32 | M4b | Scope the truncation-bias analysis to the double gyre, where M3 is analytic. Add one sentence: no bias bound claimed on the ocean field, where ρ was set by T31. **Keep the analysis and Fig 2(b).** | III-D | Opus |
| T33 | M4c | Scope out ocean-noise quantification, citing [9]'s own scoping (their Sec. IV closing paragraph: quantifying uncertainty in real ocean data and its impact on tracking is challenging and outside scope, hence noise characterised on the analytic model). **Do not attempt to estimate σ_uv** — the exactly-determined fit leaves zero residual. | V-C | Opus |
| T34 | C12 | Fig 9 caption: drop the 11.1 km; recaption the 92 white stars as context at the final frame rather than a distance target | Fig 9 | Sonnet |
| T35 | M6a | State that [9] uses the same radar record over the same window: Santa Barbara Channel HF radar, 2 km grid, 28 h, 16–17 May 2012, same network. Currently unmentioned. | V-C or VI-C | Opus |
| T36 | M6b | Replace VII-A with an assumptions table vs [9]–[11] and [13]. Rows: initialization requirement; structure identity known in advance; sensing model (single synchronous sample vs measurement history); advection; what is estimated (velocity / gradient / curvature); terminal behaviour. **Every cell sourced from the cited papers' stated assumptions — read them, do not recall them.** ~0.3 col. | VII-A | Opus |
| T37 | M6c | Add citation of [9]'s own reported limitation: robots "tend to veer away from the tracked LCS as they approach a local saddle/hyperbolic point", observed in flow tank and ocean runs. Contrast with D tracker traversing the crossing and s1 tracker settling at it. | VII-A or VII-C | Opus |
| T38 | fig | Merge Figs 8 and 9 into one two-panel ocean figure with shared colorbar. Delete Fig 3 (generic three-layer architecture from prior work); keep its 195-word paragraph. | Figs 3, 8, 9 | `AUTHOR` (plot scripts) + Sonnet for caption |

### Tier 5: compression (~a day)

Ledger items 5, 6, 8, 9, 10, 11, 12, 13, 14 from the HTML report §12. Exact keep-lists supplied there.

| ID | Action | Col | Model |
|---|---|---|---|
| T40 | Fold V-D-2's 584-word scoring rationale into VI-B-3; delete V-D-1's recording list | 0.35 | Sonnet |
| T41 | Objectivity argument: 11 statements → 4. Keep: II-D eq (6), Prop 3 statement, VI-B-2 measurement, one clause in VIII. | 0.35 | Sonnet |
| T42 | Compress II-E's degeneracy-line analysis. Keep (10), the swap sentence, the nongenericity caveat. | 0.25 | Opus |
| T43 | Off-structure acquisition sweep → two sentences (finding: acquisition is not the noise-limited phase) | 0.22 | Sonnet |
| T44 | VII-C → five-line "which tracker to pick" paragraph (also completes T20.11) | 0.22 | Opus |
| T45 | Delete II-B; move its FTLE-horizon clause to II-A | 0.20 | Sonnet |
| T46 | Related-work artificial-vector-field paragraph → one sentence. Tangent-seed caveat 8 places → 3 (keep VI-B-3, VII-E, Prop 3's one-clause exception). | 0.25 | Sonnet |
| T47 | §05 residuals: relocate the "6.9" comparison from Remark 1 to VI-B-3 alongside the 4.0 slope and state the evaluation point (`AUTHOR` for the point); drop "and order" from Remark 1; delete the flow-projection counterfactual; delete "negligible at 10 Hz"; merge κ(Φ)/det(Φ) into one quantity and say whether it was logged; shorten the TRAP definition; cut the basin coincidence paragraph to one sentence. **Keep the ε = 0.1 spot check (two sentences) — only time-varying evidence in the paper.** | 0.20 | Opus |
| T48 | Line level: Table II and Fig 7 captions to content-only; Abstract to ~210 words (M4 and M5 already pay half of it) | 0.25 | Sonnet |

### Author-only

| ID | Action | Note |
|---|---|---|
| T50 | Get [14] to arXiv | **Start now.** Only item with external lead time. Carries eq (38), the 0.3 s settling separation behind (31), and the prior-art framing. Alternative: inline both facts so they stand alone. |
| T51 | Optional, cheap: two or three more Ω points on Fig 6 so objectivity is a trend, not a point | |

---

## 4. Verification gates

| Gate | After | Check |
|---|---|---|
| G0 | T1–T9 | Compile clean. `grep -c` for γ, "weakly divergent", "twelve shape" all return 0. |
| G1 | Tier 3 | See above. Plus: Theorem numbering renumbers cleanly; Assumption 1 still supports the surviving Theorem 2 proof. |
| G2 | Tier 4 | No distance claim survives in VI-D/VIII/Abstract for the s1 ocean run. Assumptions table cells traceable to source PDFs. |
| G3 | Tier 5 | Page count ≤ 16.5. Prose share: Results up, Appendix ≤ 8%. |
| G4 | final | Read Abstract → Contributions → VI → VIII in sequence. Every claim in the first and last must appear in VI. |

Budget: **5.29 col freed, 0.70 spent, net 4.59 ≈ 2.3 pp → 16.2 pp.**

---

## 5. Model routing

| Route to | Task classes | Rationale |
|---|---|---|
| **Sonnet** | T1–T8, T11, T20.5–20.7, T25, T26, T34, T40, T41, T43, T45, T46, T48 | Deterministic edits with a stated target string or an explicit keep-list. Verifiable by grep or word count. Cheapest place to spend. |
| **Opus** | T9, T10, T12, T13, T15, T16, T20.1–20.4, T20.8–20.11, T22–T24, T30–T33, T35–T37, T42, T44, T47 | Three markers: new technical prose (T31's sampling argument, T12's normalisation), cross-section reasoning where a deletion changes what a distant paragraph may claim (all of T20), and accuracy against external sources (T36, T37 — misrepresenting a cited paper's assumptions is a serious error). |
| **Fable** | none this pass | Correct call to hold. Its value is catching what everyone else missed, and that pays off against stable text. 2.3 pages are about to move; running it now spends credits on prose that will not survive. Deploy it on the full compiled draft two revisions out, briefed to hunt for exactly the failure this report made: a recommendation that contradicts the manuscript's existing text. |

**Do not route T36 or T37 to any model without the source PDFs in context.** Both require reading [9] (Michini et al., IEEE T-RO 30(3), 2014) and [13] (Kularatne and Hsieh, Auton. Robots 41(8), 2017).

---

## 6. Known open items

| Item | Blocking? | Owner |
|---|---|---|
| C8 exit-box count | No | AUTHOR |
| Evaluation point for the analytic curvature 6.9 | No | AUTHOR |
| [14] availability | Yes, for submission | AUTHOR |
| Fig 8+9 merge plotting | No | AUTHOR |

---

## 7. What this revision is for

One sentence, so the agent optimises toward it rather than toward word count:

> After the cut, the paper claims that the D tracker traverses and the s1 tracker traverses and can certify a terminal core, because s1's minimum is recoverable from a gradient while D's needs a Hessian sign the estimator cannot deliver.

Any edit that weakens the legibility of that sentence is the wrong edit, whatever it saves.
