# Separatrix / Okubo-Weiss Paper: Test Plan and Completion Tracker

Working notes for the T-RO separatrix paper. The live draft is
Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex (NOT 2A/4A/5A/5b; those
are superseded). If a session is cut off, read this file top to bottom and
resume at the most recent HANDOFF.

## HANDOFF 2026-07-25: Test Plan + Results rewrite -- COMPLETE (commit f614c33)

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
