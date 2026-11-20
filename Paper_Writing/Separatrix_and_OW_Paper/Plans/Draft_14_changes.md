# Draft_14: what changed, and why

Draft_14.tex is a copy of Draft_13.tex with the changes below. Draft_13 and its figures are
untouched, because the PhD thesis assembler mirrors Draft_13 text verbatim. New and changed
numbers are wrapped in `\rev{}` and traced in `Draft_14_numbers.md`. New figures carry a
`_d14` suffix. Run code lives in `VF_Robot/experiments/draft14/`.

## Two things found while exploring, not raised by any review

1. Every result figure of Draft_13 (Figs. 5 to 9) predates the 2026-10-28 formation sign fix,
   and Figs. 5 to 8 ran on the legacy alpha 0.7 robot model. All double-gyre results were
   re-run on the 2026-11-16 model (`_mc_common.ROBOT_2026`). The ocean trial replays
   exactly on current code (same landfall times and ride lengths), so the ocean figure (Fig. 10 in Draft_14) stands.
2. The ocean figure's shared start is a selected run for the s1 tracker. The October jitter study
   (`experiments/ocean_small_rho/README.md`) already showed it; Draft_14 replays it on
   current code and explains it (Section V-B).

## Critique by critique

### llm_review_of_draft_13.txt

| # | Critique | Action |
|---|---|---|
| 1 | Separate novelty from [3], [5], [10]; stress acquisition without pre-straddling | Contributions rewritten as First/Second/Third/Finally, each against the prior work it extends; intro states that [26], [27], [29] assume a straddling start (verified in paper-search) |
| 2 | Second analytic field with curved trenches, shear, smooth crest | New Section IV-E "Beyond the Steady Gyre": time-periodic, bent (constructed), Bickley jet (Rypina et al. 2007, verified via Crossref), Fig. 9 strip and Table II; trackers follow their own trenches, which mark the separatrix only where strain outweighs rotation |
| 3 | Validate the fit against analytic values; radius sweep; propagate covariance | II-C now gives the closed-form noise gains; II-E checks the fit against the closed forms (truncation errors, 2.53 sigma/rho^2 within 1 %, and the structural gap in the fitted H_D); radius sweep from S3 and S4 at rho x 2/3 and x 1.5 (IV-C) |
| 4 | Stability is conditional; singularities; Eq. (19) | "Guaranteeing local convergence" removed; new III-D cites [10]'s Jury/ISS bound and states each channel's gain, where it exceeds the bound, and why saturation keeps it bounded; eigenvalue floor and least-squares fallback stated; Eq. (19) now said to certify a saddle of the fitted linearization |
| 5 | Objectivity of the whole algorithm | III-C states how the stored tangent behaves under a rotating frame (turns by Omega dt per step) and that only the seed is non-objective; abstract and conclusion keep the seed caveat |
| 6 | No physical validation; no baseline; more distant starts | Head-to-head deferred (see below) with a corrected rationale in VI; acquisition basin shading added behind Fig. 5 (noise-free map over the basin) |
| 7 | Ocean tuned against its own reference; percentile is not a ridge; backward FTLE | 100-run perturbation replay reported in V-B with its mechanism; band renamed "high-FTLE band"; 12-h backward FTLE check of the attracting segment (numbers in text, the ocean figure unchanged) |
| minor | D sign in divergent flow | Eq. (D_parts): D = mu^2 - r^2 + omega^2/4, and what D < 0 means with divergence |
| minor | Fig. 7 marks success but text compares retention | Fig. 7 now marks retention crossings |
| minor | Fonts in Figs. 4 and 7 | Fig. 4 redrawn at column width (fonts roughly double); Fig. 7 fonts enlarged, rows taller |
| minor | Code or pseudocode | No pseudocode (paper register); code-availability sentence |
| minor | Units in Table I | k in 1/time, c_max a length, g_perp in length^2 time; crest branch scaled by one time unit T; Table I moved after the trackers so every parameter is defined first |
| minor | Transport claims vs limitations | Scope sentence in II-D, abstract, contributions and VI now say the trenches mark a separatrix where strain outweighs rotation |

### Opus_review_of_draft13.txt

| # | Critique | Action |
|---|---|---|
| 1 | Keep to what is new | III-A cut to the paper's own blocks and the twelve pentagon variables; II-B points to [10, Prop. 1]; II-C opener rewritten |
| 2 | Re-run on the 2026-11-16 model | Done for Figs. 5 to 8 and the new fields. Pilot: g_perp 1 vs 0.5 changes retention by at most 2.3 points, so g_perp stays 1. Rotating rates moved to +-0.10 and +-0.18 because g k c_max = 0.103 no longer holds the saddle at 0.23 |
| 3 | Stability language up to [10] | III-D, as above |
| 4 | 2.53 sigma/rho^2 as the spine of IV-C; radius sweep | Gain stated in II-C, validated in II-E, used as the prediction in IV-C; radius sweep added |
| 5 | Oracle | See "Not done" below if it did not fit |
| 6 | [10] title, [2] year, "team", abstract opener, verb mix | Title fixed; [2] is vol. 20 no. 2, 2015 (correct here; [10] has 2014, left alone because [10] is frozen); "team" removed; abstract opens on the capability; verb mix gone |

### Opus_self_plagiarism_check_draft13.txt

| # | Finding | Action |
|---|---|---|
| 1 | Identical Fig. 2 and copied caption sentence | Skipped at your request |
| 2 | III-A architecture paragraph | Rewritten from scratch in fewer words |
| 3 | II-C noise opener | Rewritten |
| 4 | Abstract opener | Opens on the capability |

### Proposal_for_edits.pdf (Nov 6 notes)

| Item | Action |
|---|---|
| Three test fields, scope condition, online ratio | IV-E, Fig. 9 and Table II; ratio defined in II-D, used in IV-E and V-B |
| Four scope edits (Part III, 13) | Applied in spirit: II-D scope sentence, abstract, VI |
| Open item: own-trench scoring | Done for the bent gyre (`mc_bent_double_gyre.py` retained_own) |
| Open item: Bickley citation | Rypina et al., J. Atmos. Sci. 64(10):3595-3610, 2007, verified |
| Open item: constructed field | Bent gyre described as constructed |

## Decisions I made (you asked me to reason them through)

- **Draft_14, not Draft_13.** Draft_13 feeds the thesis mirror.
- **Numbers into .tex.** CLAUDE.md says not to; you asked me to work, so every number is
  `\rev{}`-marked and traced. Nothing replaces Draft_13 until you approve.
- **g_perp stays 1.** The pilot showed no measurable effect of halving it.
- **Omega = +-0.10, +-0.18.** Keeps the contrast about objectivity, not the s1 hold's speed.
- **Fig. 7 at 2000 trials per cell.** 10^4 would take about 34 h here; seeds match, so a
  later 10^4 run extends the same trials.
- **Ocean: keep the ocean figure (Fig. 10) and its headline, explain the split.** Your direction.
- **No [26] reimplementation.** A fair comparison needs an advection model for both and
  could only use the three straddling starts; a strawman would hurt more than help. The
  false "one simulator cannot host both" sentence is replaced.
- **No second ocean record.** About a page and new data; the perturbation replay answers the
  tuning-independence question more directly.
- **Code availability without a URL.** Whether the repo is public is your call.

## Claims I could not verify and changed

- Intro: "that work reports veering away from the structure near a saddle in simulations on
  measured flow-tank data" ([26]) is removed; I could not find it in [26]'s text. The
  ocean sentence about [26]'s robots veering south stays, on your word (2026-11-19).

## State at hand-off (2026-11-19, afternoon)

- `Draft_14.tex` compiles to **11 pages** (three pdflatex passes), no errors, no undefined
  references, and the same four box warnings Draft_13 has. `Draft_14.pdf` and a latexdiff
  against Draft_13 (`Draft_14_diff.pdf`, `Draft_14_diff.tex`) sit next to it; the diff's
  reference list shows one recovered markup error, cosmetic only.
- `scripts/_common.py` TEX_FILE now points to Draft_14. CLAUDE.md still names Draft_13 in its
  Separatrix section; update it if you adopt Draft_14.
- Citations are renumbered by first appearance (37 references, all cited).
- Fig. 7 runs at 1000 trials per cell (about 40 min on this machine once free). For the
  submission figure, `WORKERS=7 sh experiments/draft14/run_noise_sweeps.sh` with
  `--trials 10000` and new tags extends the same seeds overnight.
- Style: mean sentence 22.5 words, 4.9 % over 40, no banned words, no em-dashes.

### What the 11-page fit cost

Body text went from 9,223 words (first full Draft_14) to about 7,370; Draft_13 was 7,559.
Figures were not shrunk: Figs. 1-3, 5, 6, 8, 10 keep their Draft_13 sizes, Fig. 4 prints
larger, Fig. 7 keeps its Draft_13 height with larger fonts, and Fig. 9 is new. To make room:

- Five short unreferenced equations went inline; Table I's robot rows merged; Table II pairs
  D/s1 per cell; captions were shortened to what the panels do not already show.
- Cut outright: the 1260-trial hold check (legacy model, unverified on the new one); the
  position-noise variance and correlation sentence; the along-trench concavity sentence of
  the double gyre; the S5 "reach but lose the straddle" example; the separate reversal
  hazard example from the rotating-frame text; the I-to-IV transition "Recent work has begun".
- Condensed throughout without changing claims or numbers. Your sentences were kept where
  they carried a fact; where two said the same thing, one went.

Anything cut from Draft_13 shows in red in the latexdiff. Material added during this
pass and then trimmed for length is not in the diff; the numbers behind it remain in
`Draft_14_numbers.md`.

## Not done, and why

- **Oracle runs (Opus item 5).** Not run. The own-trench scoring and the analytic-trench
  figure (Fig. 9) answer most of the same question, whether losses come from the field or
  from the fit, and the page budget had no room for the result. A script-only version
  (monkeypatch the fit to return the analytic Taylor coefficients) remains a short job.
- **Head-to-head against [26]-[29] (LLM item 6).** Deferred, with the reason now stated
  correctly in Section VI.
- **Second ocean record, physical experiments.** Deferred (Section VI).
- **Fig. 7 at 10^4 trials per cell.** Running it is an overnight job; the text's numbers come
  from 1000 per cell.
- **Fig. 2.** Untouched at your request. Its caption still ends with the sentence the
  self-plagiarism check flagged as copied from [10].

## Blue numbers (added 2026-11-19, evening)

Numbers in Section IV-C and the Fig. 7 caption come from reduced-size runs (Fig. 7 sweep at
1000 trials per cell, radius and mechanism runs at 2000, g_perp pilot at 1000). They are
wrapped in `\prelim{}`, which prints blue, until the 10^4 overnight runs replace them.
Commands and the follow-up steps are in `overnight_runs.txt` next to the draft
(`experiments/draft14/run_overnight.sh`, `experiments/draft14/noise_summary.py`).
- Later the same evening: Table II, the Section IV-E numbers drawn from it, and the basin numbers
  (Fig. 5 caption, IV-B) also went blue, so that after `overnight_runs.txt` every Monte Carlo
  number rests on 10^4 trials. CLAUDE.md now names Draft_14.

## Second review round (2026-11-19, late)

Five points, all text only, no new numbers:
1. Objectivity of the s1 tracker is stated at three levels (III-C): the inputs are objective;
   the command is equivariant only where the sign test is decisive (it can flip where
   |t_ref . t_raw| < sin(Omega dt), as at an eigenframe swap); the path matches across frames
   empirically (IV-D, because the across-trench feedback absorbs the transport velocity).
   Abstract and contributions now say "built from objective quantities".
2. Trench-to-structure correspondence is scoped ("lie closest ... where strain dominates
   rotation", "in the tested environments"). A small |omega|/2r is necessary, not sufficient
   (II-D), and the ratio is not objective: the frame adds 2 Omega to omega but not to r (II-D),
   so IV-E trusts it only with an inertially referenced heading.
3. Abstract's ocean sentence: both acquire the corridor, only D keeps direction under small
   changes to the start.
4. Conclusion recommends by trade-off (rotation favors s1; noise and direction favor D).
5. Small items: sat_v(0) = 0; Cov(theta_hat) = sigma^2 (Phi^T Phi)^-1, u and v independent;
   an overdetermined fit's residual carries both noise and truncation error (II-B); the
   future-work residual sentence is corrected (it estimates noise where truncation is small);
   the stability result is "local and linearized, not a guarantee for the full switched,
   saturated controller" (III-D).

To stay at 11 pages, about 30 paragraphs that ended in a short last line were tightened by a
few words each, with no facts removed. Clauses dropped as redundant: "with no changes to the
primitives" (V-B), the Section II-D back-pointer on s1 concavity (III-C), and "not a
consequence" (V-B). "Clockwise for Omega > 0" (IV-D) and the centralization detail in VI were
cut, then restored at Chris's request; the paper still fits 11 pages.
