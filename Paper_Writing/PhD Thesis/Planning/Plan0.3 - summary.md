# Plan 0.3: what was done, what was not, and what needs you

Date: 2026-11-19 (night). Companion files: `Plan0.3 - applied_changes.md` (every edit, old and new text) and `Plan0.3 - blue_numbers.md` (every `\prelim` number). Build: 165 pages, no undefined references, two overfull boxes that were already there.

## Done

| Tier | Items |
|---|---|
| 5 A | Chapter 7 re-ported onto Draft_14 (refit robot model, Sec. IV-E "Beyond the Steady Gyre", ocean replay). `port.py` now reads `Draft_14.tex`; `bib_sources.tex` carries Draft_14's 37 references; `\prelim` defined (blue). Chapters 1, 3, 4, 9, 10 re-anchored. |
| 5 B | Appendix D rewritten as three levels (features objective, command covariant where the sign test is decisive, trajectory equivariant only empirically). Appendix E states its idealizing assumptions and the 1/(gL) limit. |
| 5 D | Sec. 8.1 closed-loop numbers rerun on the refit model (X1). |
| 5 E | Re-scoped: the 239 attraction trials are 2025 data, so the offline classification table already uses the right logs (78 + 80 of 80 + 80 three-robot trials; the table says why). No rerun needed. |
| 4 | Ch. 8: one flow at three orders (new figure), signal-to-noise per rung, steady offsets (speed over loop gain), fitted versus true features and indicators, lower rungs inside higher ones, classification and frames, design guidelines. |
| 3 | Thesis statement, contributions (seven), abstract, Ch. 2 consolidation and "Verification by Substitution", Ch. 9 updates, Ch. 10 conclusion and Table 10.1/10.2, Ch. 3 wording. |
| 2, 1 | "this paper" wording, section references, caption fixes, glossary, alpha_mom, 169 to 459 runs, orbit-drift wording, formation precision, Arduino/Edison, confidence intervals, availability statement, front matter. |

## Experiments

- X1: attraction law settles at the 0.025 m friction ring (median 0.0226 m, max 0.0237 m); old run 0.0126 m. Anisotropic-saddle bias unchanged (median 0.0475 m).
- X3: the ten 2025 vector-sum logs give a response delay of 0.164 s (median; 0.136 s with a fitted center) against 0.125 s from the refit. This supports the refit timing. It does not decide the dead zone.
- X4: `experiments/thesis_ch8_one_flow.py`; the shifted cell field equals the left gyre of the double gyre to machine precision.

## Deviations from the approved plan

- B2's table and B3's column went into Chapter 8, not Chapter 4 and Table 4.1, because in Chapter 4 they would point forward to Chapters 6 and 7. Table 10.1 got the "Uses" column.
- The loop-gain ceiling 1/(gL) is stated in Chapter 7 and used in Chapter 8, not restated in Sec. 2.6.
- The old two-model problem (K2b) mostly vanished: after the port, every chapter uses the refit model.

## Needs you

1. **Defense date and reader names** on the approval page, and the department's page-numbering convention (I inferred it from the SCU template and the example thesis).
2. **0.3 m/s versus the logs.** Secs. 2.2 and 2.5 (IDETC text) say the robots top out at 0.3 m/s. The ten 2025 vector-sum logs show cluster speeds of 0.37 to 0.40 m/s. I left the IDETC text as published.
3. **The revision paper is still changing.** It was edited at 22:27, 22:57, 23:01 and 23:09 tonight, and each time a port anchor in Chapter 6 or 4 moved. After you freeze it, rerun `python3 sources/assemble/all.py`; a failed anchor aborts the build and names the sentence.
4. **Draft_14 is read live.** The same applies there. After the overnight runs, change `\prelim` to `\rev` in Draft_14, then rerun the assembler; update the three copied numbers in `ch08.py`.
5. **Your veto list** (see the ledger): "Dr. Chris Kitts" to "Kitts and colleagues"; 2.3 m^2 to 1.6 m by 1.6 m maps; the scoped "only validated in simulation" in Sec. 1.2.2 (I did not open the cited works); the seven-item contributions list.
6. **`paper-search` failed to connect**, so no new related-work claim was added and the "topological skeleton" term is not used.

## Not done

- A full straight-through read of the 165 pages. I read the new text, the joins at the start of Chapter 7, and the rendered pages of the new tables.
- Plan0.2 trends in Chapter 1; they are in Chapter 10 only.
