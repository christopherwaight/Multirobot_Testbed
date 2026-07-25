# Draft_5c revision progress tracker

Executing `AGENT_WORKORDER_draft5c.md` per the approved plan at
`~/.claude/plans/read-agent-workorder-draft5c-and-parsed-quasar.md`.
Two author overrides in effect: Fig. 3 (three-layer architecture) is KEPT
(target page count ~16.35 pp, not 16.2); approval model is direct-edit
with logged diffs, not per-edit blocking.

Update this file after every task, not just every tier, so an interrupted
session can resume here.

Legend: [ ] not started, [~] in progress, [x] done, [A] blocked on AUTHOR item.

---

## Tier 1: mechanical (T1-T9) -- COMPLETE

- [x] T1 (C3): retitle Section III heading (L560) "Distributed" -> "Cooperative Second-Order Estimation"
- [x] T2 (C1): L951 "twelve shape variables" -> "twelve cluster variables in place of the twelve Cartesian coordinates, nine of them shape"
- [x] T3 (C2): L862 rho=0.075 "throughout this paper" -> "throughout the double-gyre experiments"
- [x] T4 (C4): Fig 5 caption (L1805 area) -- deleted "nearest"; reads "captures at the saddle its tangent orientation resolves toward"
- [x] T5 (C5): deleted gamma eigenvalue-gap clause from Assumption 1 (was L1255-56) -- confirmed dead in Thm 2 proof before deleting
- [x] T6 (C7): "This is the layer" -> "The adaptive navigation layer is where the two trackers..."
- [x] T7 (C10): Abstract "cost in robustness" -> "cost in convergence rate and in noise tolerance at a single decision point"
- [x] T8 (S07): 5 sites fixed -- IV intro comma splice (rewrote per HTML report's exact replacement text), II-E "that does," splice, VI-B-1 71-word run-on split, VII-E dangling fragment, VII-A "paid by"->"paid in" (this last one is inside VII-A prose that T36 will later replace with the assumptions table, so it may become moot -- harmless either way)
- [x] T9 (M7): deleted "weakly divergent"/"live diagnostic" defence; replaced with presupposition framing tying back to sec:intro's "neither pre-straddling...nor accumulation" line (L137-138). NOTE: the five-robot incompressibility counting sentence already existed at L610-613 pre-revision -- T9 only needed to replace the defence sentence, not add new derivation, confirming the plan's finding.

**Gate G0: PASSED.** Compiled clean (0 undefined refs, 0 errors), 20 pages (unchanged, expected -- Tier 1 is mechanical only). grep counts: gamma=0, "weakly divergent"=0, "twelve shape"=0.

## Tier 2: blocking (T10-T16) -- COMPLETE

- [x] T10 (M1): VIII s1 ocean sentence rewritten. Headline: "the s1 tracker, started on that same ridge, rode the same network structure for 28 hours from the objective strain term alone." Deleted "loop", "converged to and held", "1.4 km". No VII-E limitations clause added. Bonus: fixed the unreferenced "does not need the map" flourish (S07-adjacent, HTML report's own suggested fix) -> "does not need a precomputed FTLE field."
- [x] T11 (M2): unit pass done -- L921-922 (sigma_uv informative-to values), L1545/1550 (MC sweep grids), L1750-51 (tracking error), Table I Description column (dropped s^-2/m^2/s^-1/s^-1m^-1 parentheticals from 5 rows). Ocean (phys.) column and realized 0.5 m/s (L1515) untouched per exclusion list. JUDGMENT: left Delta t=0.1s and tau~=0.28s (robot dynamics/timestep, not field quantities) with s units -- these describe simulator cadence, not the double-gyre field itself.
- [x] T12: dual normalisation added at VI-B-3 (was ~L1880): "against peak flow, pi*A ~= 0.314, sigma_uv=0.002 is a modest 0.6%; against this 0.12 differential span... the same noise is 1.7%". Folded in the T8/M2 "systematically rather than symmetrically" gloss from the same paragraph since it's the same sentence cluster (referee report flagged this exact clause in S02/S07).
- [x] T13 (M3): advection scope-boundary added to VII-E, two sentences: exact for Decabot per \cite{33}, extrapolation for ocean. Did NOT cite [9] as precedent (per guardrail -- their robots are advected, verified against their eq 1a). Did NOT report ||v|| vs command speed.
- [x] T14 (C8): FOUND the count in mc_sweep_oecs_traverse.py docstring (not a search miss) -- 30 of 54 sigma_p=0.005 failures under y_exit=0.52 are recovered by widening to 0.60, all 54 exited at y in [0.520,0.529]. Reported to author for approval before writing (per confirmed 3-way rule step 1) -- APPROVED. Written into V-D-2 parenthetical.
- [x] T15 (C9): isotropy clause added in VI-A (~L1768 area): separates "outcome isotropy" (does not hold off-structure, decided by initial heading) from Lemma 1's "estimator isotropy" (holds, confirmed above).
- [x] T16: Problem 1/Problem 2 loop closure -- 2 of 3 referring clauses added (IV-B "this is Problem 1", IV-C "solves Problem 2"). THIRD clause (VIII) deliberately deferred to Tier 3/T20.9, since VIII's capture sentence is being rewritten there anyway and editing it twice was wasteful. Hysteresis gloss on -4s_trim added at eq (30)'s discussion (arms at 4x first-contact depth, releases at 1x).

**Gate (tier-local compile check): PASSED** after a clean-aux rebuild. Transient compile failure on first attempt was traced to STALE aux/log files from before this session (last modified before any edit) -- not a content defect from my edits; `rm -f Draft_5c.aux Draft_5c.log ...` + two-pass rebuild fixed it. Zero undefined refs, zero multiply-defined labels, 20 pages (unchanged as expected -- Tier 2 nets out to roughly neutral length, additions offsetting the ocean-sentence trim).

## Tier 3: the M5 cut (T20-T26) -- COMPLETE

- [x] T20.1: deleted Theorem 1 (was L1236-1249) + lead-in (L1232-35) + trailing gloss (L1251-52) + proof (was L2372-2422). thm:attract and eq:VA_dot labels removed entirely (confirmed self-contained, no external refs).
- [x] T20.2: Theorem 2 -> two parts (i)/(ii) only; part (iii) folded into a new prose sentence right after the theorem stating the true-Hessian fact without invoking Thm 1. Proof Steps 4-5 deleted, Steps 1-3 + \end{IEEEproof} kept.
- [x] T20.3: deleted eq (39) (park step) + all of Section VII-B (was L2094-2136) in one block, including the "Remark~\ref{rem:eigvec} identifies..." lead-in and the full parking-certificate paragraph.
- [x] T20.4: Fig 10 deleted as part of the VII-B block deletion above (\includegraphics + caption + label all removed together). Confirmed park_vs_traverse.png now correctly UNREFERENCED via verify-figures.sh (10 figures found, 0 missing).
- [x] T20.5: every D_capture occurrence was inside VII-B, gone with that deletion. Verified via grep: zero D_{capture}/D_{text{capture}} survivors.
- [x] T20.6: s_capture row deleted from Table I. ALSO deleted the entire s_capture MECHANISM (not just the row) since it's a separate optional early-park path for the s1 tracker, distinct from D_capture but equally dead/unset -- confirmed via referee report row 5 ("unset in both columns, dead parameter surface") and traced its 5 usage sites (control law, Table I intro sentence, Table I row, VI-D sentence, appendix Prop 3 proof Step 2). All 5 sites cleaned; ride-flag logic now correctly says "Two situations" not "Three". Caption checked, does not promise s_capture.
- [x] T20.7: park-versus-traverse pair sentence deleted from V-D-1, replaced with one clause on the s1 terminal test being unconditional.
- [x] T20.8: contribution 2's "with terminal capture available by tightening one threshold" deleted from Introduction.
- [x] T20.9: VIII's capture sentence rewritten to carry the paper's central post-cut claim (work order Sec 7's sentence): D solves Problem 1 (Newton step + Lyapunov cert + traversal theorem), s1 solves Problem 2 (terminal test from objective gradient, because s1's minimum is a first derivative where D's needs an unreliable Hessian sign) -- cites Remarks 1 and 2 directly. This also closed T16's deferred third Problem 1/2 referring clause.
- [x] T20.10: Problem 1's terminal-capture clause deleted from II-F; Problem 1 is now pure traversal.
- [x] T20.11: VII-C's "more reliably determined target" reworded to "Newton-rate transverse convergence and its independence from a seeded tangent direction" (D tracker has no target post-cut). Bonus: fixed the adjacent S07-flagged "Zero-noise reach costs nothing... what it costs there is rate" sentence in the same edit since it's the same paragraph.
- [x] T22: degenerate branch of (27) rewritten to a one-sentence-cluster well-definedness guard -- "the trench frame is undefined... so (27) needs a fallback... without a stability claim." No theorem, no certificate reference.
- [x] T23: checked -- Remark 2 (rem:eigvec) already carries the capture-misfire mechanism generically in its own closing sentence ("near the wells... signs become unreliable") without naming deleted machinery. All 5 \ref{rem:eigvec} sites confirmed live and non-orphaned. No edit needed; content was already correctly scoped.
- [x] T24: DONE in Tier 2 (see that section's log).
- [x] T25: Lemma 1's ring-sum derivation compressed to a proof sketch IN PLACE in Appendix A (not moved to a separate supplementary file -- none exists in this repo, and creating one is new infrastructure; asked author, confirmed "cut to sketch" is the right call). Dropped the eigenvector expansion mechanics, kept the symmetry argument and the four numeric ring sums the rest of the proof needs.
- [ ] T26 (optional, SKIPPED BY DEFAULT): Prop 2 (conic criterion) NOT demoted to a remark. Referee report S01 explicitly lists minimality/conic/pentagon as a strength to keep, and T26 is marked optional in the work order. Corollary 1 (pentagon) reads more naturally following a Proposition than a Remark. Judgment call, not executed.

Also: fixed the D tracker's L1636-area capture-test description (kept, legitimate -- describes the still-live lambda1*lambda2 sign guard, not deleted machinery). Reworded 2 instances of "No start parks at a gyre core" -> "No start settles at a gyre core" (basin_map caption + body) since "parks" is now confusable with the deleted park-step term even though it was being used as a plain verb. Added \label{sec:failure_modes} to VII-D (was the only unlabeled subsection).

**Gate G1: PASSED.**
- grep -iE "capture\|park": every survivor belongs to eq:capture_test/g_capture (the s1 terminal test, correctly kept per guardrail) or is a legitimate "s1 tracker captures" statement.
- grep "park" as whole word: zero hits.
- grep D_{capture}/s_{capture} bare forms: zero hits.
- grep "Theorem 1"/thm:attract in source: zero hits; compiled PDF shows a single clean "Theorem 1 (Separatrix network traversal)" with correct renumbering, no orphaned "Theorem 2" anywhere.
- Compile: clean, 0 undefined refs, 0 multiply-defined labels (verified via log grep after a clean-aux rebuild).
- verify-figures.sh: 10/10 figures resolve, park_vs_traverse.png correctly unreferenced.
- No hardcoded Section~\mbox{VII-*} refs exist anywhere, so VII-B's deletion (which shifts VII-C->VII-B, VII-D->VII-C) orphans no hardcoded cross-reference.
- Page count: 19 (down from 20 after Tier 2).

## Tier 4: ocean reframe and positioning (T30-T38)

- [ ] T30 (M4d): VI-C/VI-D -> demonstrations; guardrail phrases preserved
- [ ] T31 (M4a): rho=0.105 sampling argument replaces 6km comparison
- [ ] T32 (M4b): scope truncation-bias analysis to double gyre
- [ ] T33 (M4c): scope out ocean-noise quantification, cite [9]
- [ ] T34 (C12): Fig 9 caption -- drop 11.1km, recaption stars
- [ ] T35 (M6a): same radar record sentence
- [ ] T36 (M6b): VII-A -> assumptions table vs [9]-[11],[13]
- [ ] T37 (M6c): cite [9]'s saddle-veering limitation
- [ ] T38 (partial): merge Figs 8+9 only. Fig 3 KEPT per author override.

**Gate G2:** [ ] no ocean distance claim in VI-D/VIII/Abstract for s1 run; assumptions table cells traceable to source PDFs

## Tier 5: compression (T40-T48)

- [ ] T40: fold V-D-2 scoring rationale into VI-B-3
- [ ] T41: objectivity argument 11 -> 4 statements
- [ ] T42: compress II-E degeneracy-line analysis
- [ ] T43: off-structure acquisition sweep -> two sentences
- [ ] T44: VII-C -> five-line "which tracker to pick"
- [ ] T45: delete II-B, move FTLE-horizon clause to II-A
- [ ] T46: related-work AVF paragraph -> one sentence; tangent-seed caveat 8->3
- [ ] T47: Section 05 residual cuts (6.9 relocation needs AUTHOR eval point?)
- [ ] T48: Table II / Fig 7 captions to content-only; Abstract ~210 words

**Gate G3:** [ ] page count <= 16.5 (Fig 3 retained)
**Gate G4:** [ ] Abstract -> Contributions -> VI -> VIII consistency read

---

## AUTHOR items raised

(none yet -- T14's number was recovered from a script docstring, not flagged)

## Judgment calls logged

1. **T11 unit pass scope:** left `\Delta t = 0.1$~s` and `\tau \approx 0.28$~s`
   with seconds units. These describe the simulator's control cadence and the
   momentum filter's identified time constant (alpha_mom, per CLAUDE.md's
   notation table), not a double-gyre field quantity. Stripping units here
   would misrepresent the robot dynamics as non-dimensional when they are
   calibrated against real hardware (0.1 s at 10 Hz, matching the Decabot
   identification). Flag for author review if this reads wrong.
2. **Line ~1524 ("covers its path"):** confirmed this is the pre-existing
   apparent-missing-token the plan flagged. Applied the three-way rule: left
   as is. The sentence's job (speed cap is rarely approached, 0.5 m/s is a
   realistic cruise speed not a demonstration of the cap) stands without a
   distance figure, and M4(d) is cutting ocean distance claims elsewhere in
   this same revision anyway, so restoring a number here would work against
   the revision's own direction.
3. **T20.9's VIII referring-clause for Problems 1/2** deferred to Tier 3
   rather than done in T16, to avoid editing the same "terminal-capture case
   is one threshold away" sentence twice (once for loop-closure, again to
   delete the capture clause).
4. **T24 (Future Work TRAP/eddy-cores sentence, nominally Tier 3) was done
   during T13's Tier-2 pass**, in the same paragraph as the advection
   Future Work line, to only touch the "Four/Five extensions" count once.
   Marked complete under Tier 3 too; not being redone.
5. **Bonus fix beyond scope:** VIII's "does not need the map" flourish (no
   stated referent) was fixed while rewriting the same sentence for T10,
   using the HTML report's own suggested replacement ("a precomputed FTLE
   field"). Not a numbered task but directly adjacent and cost nothing extra.
6. **T25 (Lemma 1's ring-sum proof "to supplementary"):** no supplementary-
   material file or convention exists anywhere in this repo. Asked the author
   rather than inventing one; confirmed decision was to compress the proof to
   a sketch in place in Appendix A, not create new infrastructure
   (`Draft_5c_supplementary.tex`). Dropped the eigenvector-expansion mechanics,
   kept the symmetry argument and the four numeric ring sums the rest of the
   proof needs.
7. **T26 (Prop 2 -> remark) skipped by default.** It is marked optional in the
   work order, and the referee report's own Strengths section (S01) lists
   minimality/conic-criterion/pentagon together as content to keep, not
   demote. Corollary 1 (pentagon nonsingular) also reads more naturally
   following a Proposition than a Remark. No author check-in needed for an
   optional item with a stated default of "keep."
8. **T20.6 was broader than its literal wording.** The work order's item 6
   only says "delete the s_capture row from Table I." But s_capture is a full
   second capture mechanism (an optional early-park path for the s1 tracker,
   separate from D_capture), not just a table row -- deleting only the row
   would have left dead code paths in the control-law prose, the VI-D
   results sentence, and the appendix equivariance proof. Traced and cleaned
   all 5 usage sites, consistent with the referee report's own framing of
   s_capture as "dead parameter surface" throughout, not merely in the table.

## Compile log (page count per tier)

| After | Pages | Notes |
|---|---|---|
| baseline | 18.5 (per work order, word-count-derived estimate) | not the same measure as pdfinfo |
| baseline (actual) | 20 | independently compiled 2026-07-25, clean (0 undefined refs, 0 multiply-defined labels). Discrepancy with 18.5 is expected: work order's figure is a word-count/col conversion estimate, not a real pdflatex page count. Track deltas from this 20-page baseline, targeting proportionally (~2.3pp cut from work order becomes ~2.5pp from this baseline; with Fig 3 kept, expect landing near 17.7-18.0 rather than 16.35). Will reassess after Tier 3 (largest single cut) with real numbers. |
| after Tier 1 | 20 | mechanical edits only, no length change expected or observed |
| after Tier 2 | 20 | blocking-item fixes roughly length-neutral (deletions offset additions); real cut begins Tier 3 |
| after Tier 3 | 19 | M5 cut landed: Theorem 1 + proof, Thm 2(iii) + proof steps, all of VII-B + Fig 10, s_capture mechanism, D_capture, 4 downstream clauses. 1 page recovered so far; Tiers 4-5 (ocean reframe, Fig 8+9 merge, compression) still to come. |
