# TASK (Session 1 of 2): Major Revision of the IEEE Systems Journal Paper

## Objective

Revise `Paper_Writing/Systems Paper Revision/cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex` so that it answers every concern of the Associate Editor and Reviewers 1 to 3, within a **hard cap of 12.0 pages** in IEEEtran journal format (references and biographies included, last page balanced). The submitted paper was 11 pages. The revision file currently builds to 12 pages before any of the new content.

The response letter is a separate session (`REVISION_PROMPT_2_letter.md`). This session ends with the paper, a change log and a number ledger that the letter session will use.

---

## Before you start

1. Read `CLAUDE.md` and `Waight_style.md` at the repo root. They override defaults.
2. Ask Chris to commit the working tree, or confirm a commit exists, so there is a rollback point. Ask him not to edit the revision file in the IDE while you work.
3. Files drift because several agents and Chris edit in parallel. Before every edit, re-grep the exact text you are replacing. Match on exact text, never line numbers. If a match fails, stop that item and report it.
4. Paths below are relative to the repo root. Build none from `/Users/...`.

---

## Operating rules

### Files you may change

- `.../cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex`, including its preamble (you may add `\usepackage{xcolor}` and the marker macros below).
- New figure files copied into `.../cwaight_systems_paper/`. Copy them, never move them.
- Two new handoff files: `Queued_Edits/revision_change_log.md` and `Queued_Edits/revision_numbers_ledger.md`.

### Files you must not change

- The frozen submitted paper `Paper_Writing/Systems Paper/cwaight_systems_paper/cwaight_systems_paper.tex`. The PhD thesis scripts read it.
- Every note and draft listed in the reference map. Read them and do not edit them.
- Anything under `trunk/`. Do not run simulations.

### Style (`Waight_style.md`, strictly)

- No emojis, no em-dashes, no colons in running prose, no AI-voice vocabulary.
- No forward-pointing scaffolding. In related work, say what others did and did not do, then stop.
- No closing flourishes.
- Explain the physical mechanism behind a result, and keep every claim inside its evidence.
- Citations are hand-written `\bibitem{N}` with integer keys in `\begin{thebibliography}{99}`, cited with `\cite`. Append new entries with the next free N. Never introduce BibTeX, natbib or author-year keys. Do not characterize what a cited work concludes unless you have checked it (the `paper-search` MCP server indexes the reference corpus).

### Numbers

Chris normally approves every number before it enters the .tex. For this round he authorizes the sources below, provided **every such number is visibly marked in the PDF**, so none can ship by accident. Put these in the preamble.

```latex
\usepackage{xcolor}
\newcommand{\syn}[1]{\textcolor{red}{#1}}            % placeholder data, replaced by the campaign
\newcommand{\rev}[1]{\textcolor{blue}{#1}}           % real result awaiting Chris's review
\newcommand{\open}[1]{\textcolor{magenta}{[#1]}}     % protocol setting not yet decided
% For submission: redefine all three as {#1} (and \open as {}) after the placeholders are gone.
```

Wrap every authorized number in one of the macros and add a `% [SYNTHETIC]` or `% [REVIEW]` comment on the line, as below.

| Source | Marker | Notes |
|---|---|---|
| `Queued_Edits/V_VIA_synthetic_draft/` (its `results/` CSVs, `numbers.tex`, tables, figures) | `\syn{}` | Orbit rows are fabricated. The 3-robot attraction rows are real old logs, but the campaign replaces them, so mark them `\syn{}` too. **Never use the 4-robot rows.** |
| Hyperparameter study v3: `hyperparameter_study.tex`/`.md`, `results_v3/report_numbers.txt`, `results_v3/v3_numbers.txt`, `results_v3/tables.md`, `results_v3/start_distance_bins.txt` | `\rev{}` | Simulation only. Do not use any number that compares with hardware vortex runs (E10hw, the kinetic-friction fit, the "157 runs" comparisons), because those used 4-robot vortex logs. |
| `revision_metrics/tables/*_revised.tex` and `revision_metrics/results/` | `\rev{}` for simulation, `\syn{}` for hardware | Simulation uses the Coulomb robot model and post-dates the 2026-10-28 formation sign fix. |
| `vib_baselines_note/vib_baselines.tex` (table and text) | `\rev{}` | Leave its bracketed `[PENDING the new campaign]` sentence out. |
| `error_note/error_analysis.tex` classification percentages | `\rev{}` | Report them with the classifier band. |

Hard rules:
- Do not claim an experiment was run unless it was. No repeated hardware orbit, hardware oracle run, hardware classification or 4-robot run exists yet.
- Write the hardware sections as the campaign will read, with undecided settings as `\open{}`. Those are formation side (0.33 or 0.30 m), attraction run length, orbit run length, the number of radii and repeats, and the invalid-estimate behavior.
- Exclude the hardware-validation paragraph of `robot_model_note/robot_model.tex` (the "vortex, 79 runs" paragraph). It used 4-robot vortex data.
- Remove or replace every superseded number in the abstract, introduction and conclusion. That covers 157 trials, 100%, 0.021 m, "12 orbital trials", "six commanded radii", "169 hardware experiments", and "less than 0.25 m error at small radii". Do not present a small-radius orbit as accurate radius tracking (R1.4, R3.5).

---

## Reference map

### Reviews and tracker

- `Paper_Writing/Systems Paper Revision/Reviewer_Feedback.txt`: the reviews, verbatim.
- The live tracker is the Checkpoint 1 artifact, https://claude.ai/artifact/AYRPHnRGad1f4GiUQYkURc. Read its `actions` collection with ArtifactData. It holds 67 actions keyed AE.1-1 ... R3.5-4, each with section, text and evidence. Some evidence fields are stale. The local `Paper 1 Reviewer Matrix.html` is an offline copy with no actions, so do not rely on it.
- `draft_outline_note/revised_draft_outline.tex`: a section-by-reviewer map, the new and regenerated figures, and the claims to rewrite.

### Drafted text (adopt, refine or re-derive, then record what you did in the change log)

- `error_note/error_analysis.tex`: II-B error bound, existence test, covariance, VI-A edits, the singular versus non-hyperbolic distinction, and the classification band. Its "Changes elsewhere" list is part of the work.
- `minimality_note/minimality.tex`: Proposition 1 for II-C. Rename its **w** to **m**₁, since **w** now names the barycentric weights.
- `stability_note/stability.tex`: new III-D, with the sampled closed loop, the Jury bound on kΔt, Proposition 2, near-singular handling, the orbit bound, the r = 0 guard, and the VI-A last-paragraph replacement. Its near-singular and r = 0 text was aligned with the campaign on 2026-11-01. Keep it aligned.
- `robot_model_note/robot_model.tex`: the lag plus Coulomb robot model for III-A, with its parameter table. Omit its hardware-validation paragraph.
- `cwaight_systems_paper/multilayer_architecture_onepager.tex`: Section III condensed. It cites the reused work ([5], [8], [50], [10], [7], [9]), marks the new blocks in the caption, and states the robot model with ν for robot velocity. Its own reference list is only a compile aid. Do not copy it.
- `Queued_Edits/prose_framing_batch.tex`: items 1 to 9 and notes N1 to N7. Do not apply item 10.
  - Item 7 (metric definitions) is superseded by the V-B text of the V_VIA draft.
  - Item 3 (classification) needs the R1.7 work below.
- `vib_baselines_note/vib_baselines.tex`: replaces Section VI-B and its figure. **Do not use `Queued_Edits/VI-B_comparison_rewrite.tex`. It is superseded.**
- `Queued_Edits/V_VIA_synthetic_draft/V_VIA_synthetic_draft.tex`: V-B metric definitions and test plan, V-C, V-D and two VI-A paragraphs. Its header lists the ripple, and its `figures/` hold the two new hardware figures. Use the 3-robot content only.

### Supporting results

- `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/hyperparameter_study/`:
  - the `hyperparameter_study.tex` report;
  - `figures_v3/fig3_size_curvature.png`, `fig4_noise.png`, `fig5_estimators.png` and `fig6_random_index.png`;
  - `results_v3/`, including `start_distance_bins.txt`.
- `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/revision_metrics/tables/`: revised Tables II to V, the baselines table and the classification table.
- `trunk/hardware_campaign_2026-11/CAMPAIGN_PROMPT.md`: the campaign protocol and the open decisions. Read it for what the hardware section may say.
- `Synthetic_data/README.md`: how the placeholder data were made.

---

## Order of application and notation

Apply in this order, because later notes assume earlier ones:
1. error note
2. minimality note
3. stability note
4. robot model and one-pager (Section III)
5. prose batch
6. VI-B baselines
7. V and VI-A from the V_VIA draft
8. the new Section IV material

Unify notation as you go:
- **Robot velocity:** use **ν**, so it does not clash with the field **v**.
- **Per-axis state:** write it [e_a, ν_a]ᵀ.
- **Discrete index:** use [j], not [k], so it does not clash with the gain k.
- **Friction level:** use v_c everywhere. The stability note's v_min becomes v_c (prose batch N7).
- **Speed limits:** v_max is the 0.4 m/s robot top speed. The 0.3 m/s per-axis centroid command limit is a separate quantity. Fix the stability note's "v_max per axis" wording accordingly.

---

## Stage 1: Free up space (target 10.5 to 11.0 pages before new content)

1. **Section VI-B.** Replace it with `vib_baselines.tex`, which deletes the `orbit_demonstration.png` figure, and follow its "Changes elsewhere" list.
2. **Appendix B.** Condense the SAS forward and inverse kinematics and the Jacobian by about 70%. Inline only what Section III-B needs, cite [50] for the cluster space formulation, and remove the appendix.
   - The thesis reads the frozen paper, so this does not break the thesis build.
   - Keep the heading convention consistent with the Python and Simulink code. Check `cluster_builder/clusterbuilder.py` if a sign or angle convention is in doubt.
3. **Appendix A.** Decide whether the canonical field definitions stay as a short appendix or move into Table I or Section IV. Inline them only if that removes the `\appendices` block and saves 15 to 20 lines without hurting readability. State the rationale in the change log.
4. **Figures.** Shrink Fig. 2 (`control_architecture_2.png`) and the SAS figure (`SAS_Robots.png`), or set them side by side as the one-pager does.
5. **Section VI-D (Future Work).** Cut items this revision resolves, namely actuator lag, curvature effects and the stability analysis. Keep every limitation of the static laboratory setup in VI-C, because AE.1 and R2.5 ask for them. Rewrite, rather than delete, the classification paragraph that says classification "is not exercised in the present experiments" (see R1.7 below).
6. Compile twice and confirm 11.0 pages or fewer.

## Stage 2: Technical revisions (target 11.8 to 12.0 pages)

1. **Section I.**
   - Contributions (prose batch item 1) and novelty beyond "two affine fits" (item 2).
   - The classification wording "indicate the type" (item 3).
   - The roadmap sentence from the VI-B note.
   - The distributed-versus-centralized wording is already in the file. Check that it survives.
2. **Section II.**
   - II-B: the error bound and existence test from the error note.
   - II-C: Proposition 1 (minimality).
   - II-D: the closed loop of the attraction law on a static field, ė = −k e and e(t) = e(0)e^{−kt} for a fixed estimate (prose batch item 9).
   - **Do not write ė = (J − kI)e here.** That equation describes a flow that advects the robots. It belongs only in the real-flow limitation (prose batch item 5).
3. **Section III.**
   - Use the one-pager for the architecture, with its citations and the reused-versus-new marking.
   - Put the robot model in III-A.
   - Add the stability note as III-D: sampled loop, kΔt bound, Proposition 2, near-singular handling, orbit bound and the r = 0 guard.
4. **Section IV (simulation).**
   - **Existing tables.** Replace Tables II and III with the revised `revision_metrics` tables. The old ones predate the formation sign fix and the robot model. Add the concession that affine fields do not test extrapolation (prose batch item 6). Add the MLP-fields-as-fixed-distortion wording (item 8 with notes N1 to N3).
   - **New subsection, nonlinear benchmarks.** This answers AE.1b, R1.1 and R3.2, and it comes from the hyperparameter study:
     - curvature on the gyre cell and the curved saddle (E5);
     - more than one critical point, using the sign of det Ĵ at the start against the type reached (E5 basins and Fig. 6c);
     - formation size against localization error, with the side³ growth (E4);
     - start distance (`start_distance_bins.txt`). On the gyre the plane fit reaches the center from every start within 0.3 m, and beyond the det J = 0 boundary it reaches a neighboring saddle;
     - fixed distortion separated from time-varying noise (E6, friction off);
     - the achievable orbit radius against lag and gains (E2 and the closed form).
     Budget about one figure (`fig3_size_curvature.png` or a two-panel crop) and one compact table.
   - **Classification in simulation** (R1.7, R1.3): use `table_classification_revised.tex` and state the classifier band ε. Report the coarse classes too, because the center is fragile (non-hyperbolic).
5. **Section V (hardware).**
   - V-B to V-D from the V_VIA draft, three robots only: metric definitions, success at 3, 4 and 5 cm, mean error beside bias, settle time, localization error, effort, failure.
   - Add the VI-B note's point that convergence should require rest, or report final speed.
   - Add the classification metric.
   - Add the orbit bias with between-trial and within-run spread, the practical radius range, and orbit failures.
   - Copy the two figures from `V_VIA_synthetic_draft/figures/`. Regenerate or drop the current `fig:vortextomiddle` (`allrunsadjusted.png`) and `fig:Tomiddlewithtime` (`fig9.png`), which were drawn from 4-robot vortex data. If kept, mark them `\open{regenerate from campaign}`.
6. **Section VI.**
   - **VI-A:** the error note's VI-A edits, N3, the two V_VIA paragraphs and the stability note's last-paragraph replacement.
   - **VI-B:** the baselines note.
   - **VI-C:** larger teams (prose batch item 4) and the four real-flow challenges (item 5).
   - **VI-D:** Future Work as trimmed in Stage 1.
7. **R1.7 classification, end to end.**
   - The introduction says "indicate".
   - Section IV reports the simulation accuracy.
   - Section V defines the hardware classification metric, with the values `\syn{}` or `\open{campaign}`.
   - The Future Work paragraph no longer says classification was never exercised.
8. **Abstract and conclusion.** Rewrite them to the new evidence, with every number marked.

## Page budget

If the paper runs over 12.0 pages, cut in this order and log each cut:
1. numerical-check details and proof details that the letter can carry instead;
2. any duplicated statement of a result;
3. a second figure in the nonlinear subsection;
4. Appendix A, moved inline;
5. prose in VI-A that the new propositions make redundant.

Never cut an answer to a reviewer request to make space. Flag the conflict instead.

## Stage 3: Compile, verify, hand off

1. Compile twice (BasicTeX).
   ```bash
   export PATH="/Library/TeX/texbin:$PATH"
   cd "Paper_Writing/Systems Paper Revision/cwaight_systems_paper"
   pdflatex -interaction=nonstopmode cwaight_systems_paper_with_revisions.tex
   pdflatex -interaction=nonstopmode cwaight_systems_paper_with_revisions.tex
   ```
   Report errors, undefined references, new warnings and the page count. Check that page 12 is balanced (`\balance` from `balance.sty` is already loaded).
2. Grep the PDF text and the .tex for the superseded numbers listed under "Numbers". None may remain unmarked.
3. Write `Queued_Edits/revision_change_log.md`. For each of the 67 tracker actions, give the action ID, what changed, and where (section, equation, table or figure label). Record what you adopted, refined or re-derived from each draft, and any action left unanswered with the reason.
4. Write `Queued_Edits/revision_numbers_ledger.md`. List every `\syn{}`, `\rev{}` and `\open{}` occurrence with its section, its value and the exact source file it came from.
5. Report to Chris in prose and tables: page count, which actions are answered, what was cut, and every decision you made that he should check.
