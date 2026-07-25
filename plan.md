# Separatrix / Okubo-Weiss Paper: Test Plan and Completion Tracker

Working notes for the T-RO separatrix paper. The live draft is
Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex (NOT 2A/4A/5A/5b; those
are superseded). If a session is cut off, read this file top to bottom and
resume at the most recent HANDOFF.

## HANDOFF 2026-07-25: Referee-report revision -- COMPLETE (commits bef8c8e..c4e2702)

Executed AGENT_WORKORDER_draft5c.md (derived from referee_report_draft5c.html)
end to end, all five tiers. Full detail, per-task log, judgment calls, and
AUTHOR-flagged items are in Paper_Writing/Separatrix_and_OW_Paper/
REVISION_5c_PROGRESS.md -- read that file, not this summary, before touching
the paper again.

**What changed:** M5 (largest cut) removed all D-tracker terminal-capture
machinery -- Theorem 1, Theorem 2's part (iii), all of former Section VII-B
(Park versus Traverse) including Fig 10 and eq (39), the D_capture parameter,
and a separate dead s_capture mechanism for the s1 tracker (unset in both
operating points, traced through all 5 of its usage sites). The s1 tracker's
own terminal test (eq 30) is untouched and is now the paper's only terminal
behavior. Conclusion rewritten to state the post-cut claim explicitly: D
traverses, s1 traverses and can certify a terminal core, because s1's
minimum is a first derivative of the fit where D's needs a Hessian sign the
estimator cannot deliver reliably. Fixed the two blocking defects: Section
VIII's ocean claim contradicted VI-D (now consistent, all three of VI-D's
guardrail phrases survive verbatim); stripped SI units from double-gyre
quantities that the paper declares non-dimensional. Ocean sections (V-C,
VI-C, VI-D) reframed as demonstrations rather than validation, with a new
6-row assumptions table (Section VII-A) replacing an unsupported comparison
to Michini/Kularatne, every cell checked against the actual PDFs.

**Author override:** Figure 3 (three-layer architecture) was KEPT against
the work order's own recommendation to cut it -- explicit instruction.

**Where it landed:** 19 pages (compiled from a true 20-page baseline; the
work order's "18.5 pp" was a word-count estimate, not a real pdflatex
count), not the work order's 16.2 pp target. Per explicit instruction
mid-session, completing every action item took priority over the exact page
count once the corrected baseline made 16.2 unreachable with Fig 3 kept.
Two open AUTHOR items: (1) T38, merging Figs 8+9 into one two-panel ocean
figure, needs new plotting code (separate scripts, different datasets) --
not done; (2) a kappa(Phi)-monitoring sentence in Failure Modes was
corrected to say "not implemented" after finding no evidence it's actually
logged -- flag if that's wrong. Full item-by-item disposition in
REVISION_5c_PROGRESS.md.

Compiled clean, zero undefined references, zero multiply-defined labels,
all 10 figures resolve. A hardcoded-section-reference audit (prompted by
deleting Section II-B, which shifted every later II-subsection up one
letter) found and fixed 3 stale `Section~\mbox{...}` prose references that
a compile-clean log would not have caught on its own.

## HANDOFF 2026-07-25 (earlier): Test Plan + Results rewrite -- COMPLETE (commit f614c33)

Sections V-D and VI rewritten against the current controllers. Section IV
was already current and was not touched.

### Settled decisions (do NOT relitigate)
- **Scoring is far-saddle only, for BOTH trackers.** D and s1 are two
  surrogates for one object (the separatrix), so the fair test holds the
  object fixed and varies only the surrogate. A near-saddle settle is a
  DIRECTIONAL FAILURE of the ride, not a substituted easier task. The
  rationale is in rescore_single_target.py's docstring (2026-07-21).
  mc_sweep_oecs_traverse.py scores either-saddle; that was the right lens
  for characterizing CAPTURE in isolation and is NOT the paper's metric.
  The rescore corrects it. Table II(b) = far saddle, (c) = either saddle
  as a diagnostic.
- **The s1 failure above the threshold is a SIGN INVERSION, not decay to
  chance.** far_saddle_sign_rate goes 100.0 -> 91.9 -> 0.1% across
  sigma_uv = 0, 0.001, 0.005 (7/10000 far-saddle at 0.005). Meanwhile
  90.9% of those same runs still settle cleanly at a saddle. The team
  rides competently in the wrong direction. Do not describe this as an
  "even bet" or a "smooth decline"; that was the old, weaker framing.
- **No selector-mode names in the paper body.** Section IV derives both
  controllers as single velocity commands, so FLOW/SLIDE/ATTRACT and
  ACQUIRE/PARK/CAPTURE/CROSS/TRACK have no definition there. They are
  purged from Results, Discussion, Table I, and both appendix proofs.
  separatrix_clean_runs.py --mode-strip recovers the old two-panel Fig. 3
  if ever wanted.
- **Fig. 9 (ocean s1) is the s2 start deliberately.** 11.06 km is the
  WIDEST of twelve starts; it is shown for the traverse it produces, and
  VI-D reports the full range (1.37 to 11.06 km, median 3.3) so it is not
  read as typical. The 1.37 km run is the default start, which matches the
  D tracker's own initial condition.

### Data provenance (all verified 2026-07-25)
- The paper's 10k numbers come from the run that finished 04:01 today,
  commit bd6fd0f, 10,000 trials/cell: outputs/mc_oecs_traverse/
  {trials_fixed,summary_fixed,flip_resolution,flip_resolution_sigma_p}.csv
  plus summary_single_target.csv (the rescore).
- rescore_single_target.py NOW WRITES A PROVENANCE HEADER (commit, source
  file, trials/cell). It did not before, which is why that file's trial
  count was previously ambiguous. Re-run it after any new sweep.
- All 150 Table II cells and every inline figure were checked against
  their CSVs programmatically. Zero mismatches after fixing 43.6 -> 43.5%.
- The f002866 objectivity fix changed NOTHING in the demo/ocean results:
  traverse_objectivity.csv (0.0247/1.2188) and the s2 ocean run (11.06 km)
  are bit-identical pre- and post-fix. All five paper figures are
  byte-identical to what was already committed. The regeneration was
  verification, not repair.

### Open items
- 27 experiment scripts repointed to Draft_5c.tex. mc_sweep_ow.py and
  ow_clean_runs.py are marked "Paper: NONE" (the OW boundary-tracker
  thread was cut from the paper), not relabeled.
- Six "et al." bibitems ([2],[3],[4],[5],[25],[32]) still need full
  author-list expansion if the venue enforces it. Known, deliberate gap.
- Pre-existing pytest collection errors in tests/test_clean_dual_jacobian.py
  and tests/test_gain_sweep.py (parametrized helpers named test_* with no
  fixtures). Unrelated to the paper; 26 other tests pass.

## Key files

- Live draft: `Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex`
- 4A fact-checked reference copy (full provenance comments, diverged in
  content from 5c): `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_4A.tex`
- Math walkthrough: `Paper_Writing/Separatrix_and_OW_Paper/teaching_notes.tex`
- Simulation scripts: `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/`
- MC sweep outputs (paper's numbers, commit bd6fd0f): `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/outputs/mc_oecs_traverse/`

## Constraints and style rules (do not violate)

- No emojis, no em-dashes, no AI voice. Sutton and Barto pseudocode style
  for explanations; IEEE control-law/equation form in the paper itself.
- Show diffs before writing to paper; user reviews all numbers before .tex entry.
- Inline numeric citations, hand-written thebibliography. No BibTeX.
- alpha collision: alpha_eig (eigenvalue real part) vs alpha_mom (momentum). Disambiguate.
