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

## Tier 2: blocking (T10-T16)

- [ ] T10 (M1): VIII s1 ocean sentence rewrite
- [ ] T11 (M2): unit pass, double-gyre numbers + Table I Decabot column
- [ ] T12: dual normalisation (peak flow + differential span)
- [ ] T13 (M3): advection scope boundary in VII-E
- [ ] T14 (C8): exit-box count -- CSV search, else scope in prose (no default placeholder)
- [ ] T15 (C9): isotropy clause
- [ ] T16 (C6 + hysteresis gloss): Problems 1/2 loop closure, -4s_trim gloss

**Gate G1 (tier gate for T2, per work order "G0" after T1-T9 covers T1-T9 only; treating T10-16 as tier-local):** [ ] compile clean

## Tier 3: the M5 cut (T20-T26)

- [ ] T20.1: delete Theorem 1 + proof
- [ ] T20.2: delete Theorem 2(iii) + proof Steps 4-5
- [ ] T20.3: delete park step (39) + all of VII-B
- [ ] T20.4: delete Fig 10 + file ref
- [ ] T20.5: delete every D_capture occurrence
- [ ] T20.6: delete s_capture row, Table I
- [ ] T20.7: delete park-vs-traverse pair sentence
- [ ] T20.8: delete contribution 2's capture clause
- [ ] T20.9: delete VIII "one threshold away"
- [ ] T20.10: delete Problem 1's capture clause
- [ ] T20.11: reword VII-C "more reliably determined target"
- [ ] T22: degenerate branch of (27) -> one-sentence guard
- [ ] T23: fold Remark 2 capture consequence into III-D
- [ ] T24: Future Work sentence (TRAP/eddy cores to follow-on paper)
- [ ] T25: Lemma 1 ring-sum proof -> supplementary
- [ ] T26 (optional): Prop 2 -> remark

**Gate G1 (work order's actual G1):** [ ] grep capture/park/Theorem 1/D_capture/s_capture -- every survivor belongs to (30)/(37)/Future Work/Remark 1-2. Compile clean, zero undefined refs, theorem renumbering clean.

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

(none yet)

## Judgment calls logged

(none yet)

## Compile log (page count per tier)

| After | Pages | Notes |
|---|---|---|
| baseline | 18.5 (per work order, word-count-derived estimate) | not the same measure as pdfinfo |
| baseline (actual) | 20 | independently compiled 2026-07-25, clean (0 undefined refs, 0 multiply-defined labels). Discrepancy with 18.5 is expected: work order's figure is a word-count/col conversion estimate, not a real pdflatex page count. Track deltas from this 20-page baseline, targeting proportionally (~2.3pp cut from work order becomes ~2.5pp from this baseline; with Fig 3 kept, expect landing near 17.7-18.0 rather than 16.35). Will reassess after Tier 3 (largest single cut) with real numbers. |
| after Tier 1 | 20 | mechanical edits only, no length change expected or observed |
