# TASK (Session 1 of 2): Surgical Revision of the IEEE Systems Journal Paper

## Objective

Answer every concern of the Associate Editor and Reviewers 1 to 3 (67 tracker actions) by editing `Paper_Writing/Systems Paper Revision/cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex` **surgically**. The submitted text stays, and it changes only where an action requires it.

Two measured limits:
- **Pages.** 12.0 at most, a hard cap in IEEEtran journal format, references and biographies included, last page balanced.
- **Survival.** At least 80% of the submitted words must survive. 70% is the floor. Measure it with `python3 "Paper_Writing/Systems Paper Revision/survival.py"`. It counts submitted body words that reappear in order, in runs of four or more identical words. Comments, tabular cells and the bibliography are excluded.

The response letter is a separate session (`REVISION_PROMPT_2_letter.md`). This session ends with the paper, a change log and a numbers ledger.

## Why this prompt exists

A first pass on 2026-11-02 rewrote most of the paper, and only 46% of the submitted words survived. The editor and reviewers compare against the version they read. A rewrite reads as a new submission, makes every claim in the letter hard to check, and invites fresh comments on text they had already accepted. That pass is saved in `Queued_Edits/rewrite_v1/` as a donor of new material. **It is not the base.**

---

## Starting state (already set up)

- **The working file.** It was reset to commit d3f1be8, which is the submitted paper plus Chris's approved edits:
  - hats on estimated quantities;
  - the exact error relation `eq:scope`;
  - the "cooperative, spatially distributed" wording;
  - the fifth limitation, on centralized computation;
  - the hardware attraction table in metres.

  It builds to 12 pages with no errors and 97.2% survival. **The paper is already at the cap**, so every addition must be paid for with space.
- **The donor, `Queued_Edits/rewrite_v1/`**, holds:
  - the v1 `.tex` and PDF;
  - `revision_change_log_v1.md`, which maps all 67 actions to v1 locations;
  - `revision_numbers_ledger_v1.md`, which traces every marked number to its source;
  - the v1 letter;
  - a diff PDF of v1 against the submitted paper.
- **Tools** in `Systems Paper Revision/`:
  - `survival.py`.
  - `make_diff.sh`, which builds `cwaight_systems_paper/cwaight_systems_paper_diff.pdf` with latexdiff. In it, added text is green and underlined, removed text is grey and struck through, and changed tables show only the new version.
- **Figures that v1 already made**, in `cwaight_systems_paper/`:
  - `fig_nonlinear_size_curvature.png`, panels (a) and (c) of the study's Fig. 3;
  - `fig_hw_attraction_ecdf.pdf` and `fig_hw_orbit_radius.pdf`, both three-robot only and drawn from placeholder data.

## Before you start

1. Read `CLAUDE.md` and `Waight_style.md` at the repo root. They override defaults.
2. Confirm with Chris that a commit exists as a rollback point, and that he is not editing the file in the IDE.
3. Before every edit, re-grep the exact text you are changing. Match on exact text, never line numbers. If a match fails, stop that item and report it.
4. Build no path from `/Users/...`.

---

## Editing rules (the point of this session)

1. **Every edit answers a tracker action.** Put the action ID in the change log. With no action, make no edit. The one exception is an outright error in the submitted text; log that as `ERR`.
2. **Use the smallest edit that answers the action.** In order of preference:
   - (a) change a number, symbol or word in place;
   - (b) add a clause or sentence inside an existing paragraph;
   - (c) add a new paragraph, equation or subsection for content that did not exist;
   - (d) delete a sentence that is wrong, superseded or needed for space;
   - (e) replace a whole paragraph, only when most of its sentences are wrong, and log why.
3. **Never reword a correct sentence.** That rules out changes for style, flow, concision or tone, and changes made only so new text fits beside it. Write the new sentence around the old one. The submitted wording stays even where you would phrase it differently, and even where it bends the medium style rules. For example, the roadmap paragraph stays, and only its section list is updated.
4. **Do not move, split or merge existing paragraphs.** `survival.py` counts a moved paragraph as deleted. If an action seems to need a move, ask Chris.
5. **Notes and drafts supply content, not replacement text.** Where a note offers a rewritten version of an existing paragraph, keep the existing sentences and insert only what the note adds or corrects. Where a note supplies genuinely new material, such as a proposition, a bound or a subsection, write it as new text.
6. **New text is compact and follows `Waight_style.md`.**
   - Keep propositions under 150 words and proofs short.
   - Numerical-check detail goes to the letter, not the paper.
   - Introduce new symbols for new content. Rename an existing symbol only on a clash (see Notation).

## Paying for space

Budget about 2,000 to 2,500 words of new prose in total. `survival.py` prints the revised body's word count; keep it at or below about 8,500. For scale, v1 fit 9,249 prose words into 12 pages, but only by cutting six figures and both appendices.

Free space in this order. After each step, log the page and survival effect.
1. **Remove figures whose data the revision supersedes.** Edit any sentence that refers to them minimally.
   - `allrunsadjusted.png` (`fig:vortextomiddle`) and `fig9.png` (`fig:Tomiddlewithtime`) both show four-robot vortex runs.
   - `orbit_demonstration.png` (`fig:primitive_comparison`) shows the vector-sum comparison, which the VI-B note removes.
   - `orbit040_analytical_vs_real.png` comes from the old orbit runs with mixed gains, which the campaign redoes.
2. **Shrink figures,** or set Fig. 2 (`control_architecture_2.png`) and the SAS figure side by side.
3. **Remove figures that repeat what the text states,** namely `color_sensor_hsv_predictions.png` and `figure_5.png`. Keep the sentences that state the result.
4. **Fold Appendix A into a fourth column of Table I** (the canonical field definitions).
5. **Trim the new blocks you wrote.**
6. **Delete whole sentences that are superseded or redundant.**
7. **Last resort: Appendix B.** Delete the equations and sentences that III-B does not need, without rewording, or remove the appendix and point to [50]. This is the item that decides whether you reach 80%, so run `survival.py` before and after.

Never cut an answer to a reviewer request, and never reword to save a line. If they conflict, flag the conflict to Chris.

## Survival checks

- Run `survival.py` and `sh make_diff.sh` at the end of every stage. Read the diff PDF for reworded sentences you did not intend.
- **Expected losses:**
  - superseded result sentences in Sections IV and V;
  - the old VI-B, "Comparison with Other Primitives";
  - the Future Work items this revision resolves;
  - Appendix A;
  - possibly Appendix B.
- **Per-section guide:**
  - The abstract, I, II, III, V-A, V-B, VI-A, VI-C and the Conclusion should each stay at or above 85%.
  - IV and V-C/D should stay at or above 40%. Keep the sentence frames and swap the numbers.
  - VI-B is exempt.
- **Stopping rule.** Stop and send Chris the per-section table before continuing if either happens:
  - overall survival drops below 75% before the final page-fit stage;
  - it drops below 70% at any point.

---

## Numbers

Chris normally approves every number before it enters the .tex. For this round he authorizes the sources below, provided every such number is visibly marked in the PDF. Put these in the preamble.

```latex
\usepackage{xcolor}
\newcommand{\syn}[1]{\textcolor{red}{#1}}            % placeholder data, replaced by the campaign
\newcommand{\rev}[1]{\textcolor{blue}{#1}}           % real result awaiting Chris's review
\newcommand{\open}[1]{\textcolor{magenta}{[#1]}}     % protocol setting not yet decided
% For submission: redefine all three as {#1} (and \open as {}) after the placeholders are gone.
```

Wrap every authorized number in a macro, and add a `% [SYNTHETIC]` or `% [REVIEW]` comment on its line. Changing a submitted number to a new value is an edit of type (a), so log it.

| Source | Marker | Notes |
|---|---|---|
| `Queued_Edits/V_VIA_synthetic_draft/results/` (CSVs, `numbers.tex`, table bodies) | `\syn{}` | The orbit rows are fabricated. The three-robot attraction rows are real old logs, but the campaign replaces them, so mark them `\syn{}` too. **Never use the four-robot rows.** The draft's own `figures/` include four-robot curves, so use the three-robot versions already in `cwaight_systems_paper/`. |
| Hyperparameter study v3: `hyperparameter_study.tex`/`.md`, `results_v3/report_numbers.txt`, `v3_numbers.txt`, `tables.md`, `start_distance_bins.txt` | `\rev{}` | Simulation only. Use no number that compares with hardware vortex runs (E10hw, the kinetic-friction fit, the "157 runs" comparisons), because those came from four-robot logs. |
| `revision_metrics/tables/*_revised.tex` and `revision_metrics/results/` | `\rev{}` for simulation, `\syn{}` for hardware | The simulations use the Coulomb robot model and post-date the 2026-10-28 formation sign fix. |
| `vib_baselines_note/vib_baselines.tex` (table and text) | `\rev{}` | Leave out its bracketed `[PENDING the new campaign]` sentence. |
| `error_note/error_analysis.tex` classification percentages | `\rev{}` | Report them with the classifier band. |

The donor's `revision_numbers_ledger_v1.md` already traces most of these numbers. Check each one against its source as you port it. Numbers that come from the drafted notes rather than from the sources above (such as γ = 1.0003 from the stability note, or 2×10⁻⁵ m/s from prose batch N1) get `\rev{}`, and each one is listed in the ledger as outside the authorized list.

Hard rules:
- Do not claim an experiment was run unless it was. No repeated hardware orbit, hardware oracle run, hardware classification or four-robot campaign run exists yet.
- Write the hardware sections as the campaign will read, with each undecided setting as `\open{}`:
  - formation side, 0.33 or 0.30 m;
  - attraction run length;
  - orbit run length;
  - the number of radii and repeats;
  - the behavior when an estimate is invalid.
- Exclude the hardware-validation paragraph of `robot_model_note/robot_model.tex` (the "vortex, 79 runs" paragraph), because it used four-robot data.
- Replace or mark every superseded number in the abstract, introduction and conclusion:
  - 157 trials;
  - 100%;
  - 0.021 m;
  - "12 orbital trials";
  - "six commanded radii";
  - "169 hardware experiments";
  - "less than 0.25 m error at small radii".

  Do not present a small-radius orbit as accurate radius tracking (R1.4, R3.5).

## Notation

- **Robot velocity:** **ν**, so it does not clash with the field **v**.
- **Per-axis state:** [e_a, ν_a]ᵀ.
- **Discrete index:** [j], so it does not clash with the gain k.
- **Friction level:** v_c everywhere. The stability note's v_min becomes v_c.
- **Speed limits:**
  - v_max is the 0.4 m/s robot top speed.
  - The 0.3 m/s per-axis centroid command limit on hardware is a separate quantity.
  - The simulation command limit is 0.5 m/s (`revision_metrics/sim.py`). State both limits where they apply.
- **Clash fixes carried over from v1:**
  - χ for the curvature coefficient of the curved saddle;
  - ε_c for the classifier band;
  - **m**₁ in place of the minimality note's **w**.

## Reference map

- **Reviews:** `Systems Paper Revision/Reviewer_Feedback.txt`, verbatim.
- **Tracker:** the Checkpoint 1 artifact, https://claude.ai/artifact/AYRPHnRGad1f4GiUQYkURc.
  - Read its `actions` collection with ArtifactData. It holds 67 actions, AE.1-1 to R3.5-4.
  - Some evidence fields are stale.
  - The local `Paper 1 Reviewer Matrix.html` has no actions.
- **Content sources** (read-only; record in the change log what you used from each):
  - **`draft_outline_note/revised_draft_outline.tex`:** the section-by-reviewer map.
  - **`error_note/error_analysis.tex`:** the II-B error bound, existence test and covariance, the VI-A edits, the classification band, and its "Changes elsewhere" list.
  - **`minimality_note/minimality.tex`:** Proposition 1.
  - **`stability_note/stability.tex`:** the content of III-D. Its near-singular and r = 0 text was aligned with the campaign on 2026-11-01. Keep it aligned.
  - **`robot_model_note/robot_model.tex`:** the lag plus Coulomb model for III-A, without its hardware-validation paragraph.
  - **`cwaight_systems_paper/multilayer_architecture_onepager.tex`:** the citations to reused work and the reused-versus-new marking. Its prose is a rewrite of Section III, so take only those two things from it.
  - **`Queued_Edits/prose_framing_batch.tex`:** items 1 to 9 and notes N1 to N7, but not item 10.
    - Item 7 (metrics) is superseded by the V_VIA draft's definitions.
    - Item 3 needs the R1.7 work below.
  - **`vib_baselines_note/vib_baselines.tex`:** replaces Section VI-B. **Do not use `Queued_Edits/VI-B_comparison_rewrite.tex`.**
  - **`Queued_Edits/V_VIA_synthetic_draft/V_VIA_synthetic_draft.tex`:** the metric definitions, test plan, V-C, V-D and two VI-A paragraphs. Use the three-robot content only.
- **Supporting results:**
  - `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/hyperparameter_study/`, which holds the report, `figures_v3/` and `results_v3/`;
  - `.../experiments/revision_metrics/tables/`;
  - `trunk/hardware_campaign_2026-11/CAMPAIGN_PROMPT.md`, which sets what the hardware section may say;
  - `Synthetic_data/README.md`.

## Using the donor

Take these from `rewrite_v1`:
- its genuinely new blocks, trimmed to the budget:
  - the II-B bound text;
  - Proposition 1 and its proof;
  - the robot-model equations;
  - III-D;
  - the metric-definitions paragraph;
  - the nonlinear and classification subsections;
  - VI-B;
  - the revised tables with their markers;
  - Table I's canonical-field column;
- its ledger, for number sources;
- its change log, as a checklist of the 67 actions.

Do not take from it any paragraph that exists in the submitted paper. For those, start from the base text.

Leave behind these v1 decisions:
- the deleted roadmap;
- the rewritten abstract, contributions, conclusion and VI-A;
- the condensed architecture prose;
- the replaced Minimum Robot Requirements text (2.8% survival);
- the hardware attraction rows merged into Table II. Keep the hardware attraction table as its own table (`tab:estimation_results`) and swap its numbers.

---

## Work by stage

Run `survival.py` and `make_diff.sh`, and compile twice, at the end of each stage.

**Stage 1. Space.** Work through steps 1 to 4 of "Paying for space". Report the page count and survival.

**Stage 2. Sections II and III.**
- **II-B:** keep everything. After `eq:scope`, add the error bound, existence test and covariance from the error note as new text.
- **II-C:**
  - Keep the existing paragraph.
  - Add Proposition 1 and a short proof after it.
  - Delete only a sentence the proposition makes false.
- **II-D:** add the static-field closed loop, ė̂ = −kê with exponential decay for a fixed estimate (prose batch item 9). **Do not write ė = (J − kI)e here.** That equation describes a flow that advects the robots, so it belongs only in the real-flow limitation.
- **III, general:**
  - Keep the architecture paragraphs.
  - Add the one-pager's citations to existing sentences.
  - Add the reused-versus-new marking to the Fig. 2 caption.
- **III-A:**
  - Keep the text.
  - Insert the lag plus Coulomb model, with its equations, where the old text describes the robot response.
  - Replace only the sentences about the old model that are now wrong.
  - Give the parameter values in the text, with no parameter table.
- **III-D (new):**
  - the sampled closed loop;
  - the kΔt bound;
  - Proposition 2;
  - near-singular handling with `\open{}` for the invalid-estimate behavior;
  - the orbit bound;
  - the r = 0 guard.

  Start from the donor's III-D and trim it.

**Stage 3. Section IV.**
- **Metric definitions:** insert a short paragraph at first use, shared by simulation and hardware, with E_u as the effort metric.
- **Tables II and III:** replace them with the revised `revision_metrics` tables. Keep the sentences around them and swap their numbers. Delete only the sentences whose claims no longer hold.
- **Prose batch items:** add item 6 (affine fields do not test extrapolation) and item 8 with N1 to N3 (the MLP fields as fixed distortion).
- **New subsection, nonlinear fields** (AE.1b, R1.1, R3.2), from the hyperparameter study:
  - curvature on the gyre cell and the curved saddle (E5);
  - multiple critical points, with the sign of det Ĵ at the start against the type reached;
  - formation size against error, growing as side³ (E4);
  - start distance (`start_distance_bins.txt`);
  - fixed distortion separated from time-varying noise (E6);
  - the achievable orbit radius against lag and gains (E2 and the closed form).

  Use one figure (`fig_nonlinear_size_curvature.png`) and at most one compact table.
- **Classification in simulation** (R1.7, R1.3): use `table_classification_revised.tex`. State the band ε_c, and give the coarse classes too, because the center is non-hyperbolic.

**Stage 4. Section V.** Three robots only.
- **V-A:** change only the wording that is now wrong. Per N1 to N3, the field is a fixed distortion and the field speed is 0.3 m/s.
- **V-B:**
  - Keep the test-plan sentences.
  - Amend the settings with `\open{}`.
  - Point to the metric definitions.
  - Add that convergence requires rest, or that the final speed is reported (from the VI-B note).
  - Add the hardware classification metric, as `\open{campaign}`.
- **V-C and V-D:**
  - Keep the sentence frames and swap the numbers for `\syn{}` three-robot values.
  - Delete the sentences that describe four-robot vortex results.
  - Add the orbit bias with its between-trial and within-run spread, the practical radius range, and the orbit failures.
  - Use `fig_hw_attraction_ecdf.pdf` if space allows.

**Stage 5. Section VI, introduction, abstract, conclusion.**
- **VI-A:** apply, as inserted or replaced sentences:
  - the error note's VI-A edits;
  - N3;
  - the two V_VIA paragraphs;
  - the stability note's last-paragraph replacement.

  Do not rewrite VI-A wholesale; v1 kept only 18% of it. Where the submitted VI-A cites a negative orbital bias at 0.5 m, check that claim against the revised Table III and fix that sentence only.
- **VI-B:** replace it with the baselines note, and follow its "Changes elsewhere" list.
- **VI-C:**
  - Amend the existing limitations in place.
  - Append larger teams (item 4) and the four real-flow challenges (item 5).
  - Keep every limitation of the static laboratory setup (AE.1, R2.5).
- **VI-D:**
  - Delete the items this revision resolves: actuator lag, curvature effects and the stability analysis.
  - Amend the sentence that says classification "is not exercised" in place.
- **Introduction:**
  - Amend each contribution sentence in place, using prose batch item 1 as a content source.
  - Add the novelty sentence beyond "two affine fits" (item 2).
  - Change "identify" to "indicate" for the type (item 3).
  - Update the roadmap's section list.
  - Check that the distributed-versus-centralized wording survives.
- **Abstract and Conclusion:**
  - Swap the superseded numbers.
  - Add at most two sentences to the abstract and one to the conclusion for the new evidence.
- **R1.7 end to end:**
  - The introduction says "indicate".
  - IV reports the simulation accuracy.
  - V defines the hardware metric as `\open{campaign}`.
  - VI-D no longer says classification was never exercised.

**Stage 6. Fit, verify, hand off.**
1. Apply steps 5 to 7 of "Paying for space" if the paper is over 12.0 pages.
2. Compile twice with BasicTeX:
   ```bash
   export PATH="/Library/TeX/texbin:$PATH"
   cd "Paper_Writing/Systems Paper Revision/cwaight_systems_paper"
   pdflatex -interaction=nonstopmode cwaight_systems_paper_with_revisions.tex
   pdflatex -interaction=nonstopmode cwaight_systems_paper_with_revisions.tex
   ```
   Report errors, undefined references, new warnings and the page count. Check that page 12 is balanced.
3. Grep the PDF text and the .tex for the superseded numbers. None may remain unmarked.
4. Run `survival.py` and `make_diff.sh` one last time.
5. Write `Queued_Edits/revision_change_log.md`, in two parts.
   - **Part 1, one row per edit:** the action ID (or ERR), the edit type (a to e), the location, and the reason for each (d) or (e).
   - **Part 2, one row per tracker action:** where it is answered, or why it is not.

   List the donor blocks you reused.
6. Write `Queued_Edits/revision_numbers_ledger.md`. List every `\syn{}`, `\rev{}` and `\open{}` with its section, value and source file.
7. Report to Chris in prose and tables:
   - the page count;
   - the `survival.py` table;
   - the actions answered;
   - the cuts and their page effects;
   - every decision he should check.

## Files

- **You may change:**
  - the working `.tex`, including its preamble;
  - new figure files copied (never moved) into `cwaight_systems_paper/`;
  - the change log and the ledger;
  - the build and diff outputs.
- **You must not change:**
  - the frozen `Paper_Writing/Systems Paper/cwaight_systems_paper/cwaight_systems_paper.tex`, which the thesis scripts read;
  - any note or draft in the reference map;
  - `Queued_Edits/rewrite_v1/`;
  - `survival.py`, `make_diff.sh` and `diff_preamble.tex`;
  - anything under `trunk/`.

  Do not run simulations.
- **Citations:** hand-written `\bibitem{N}` with integer keys, with new entries appended at the next free N. Never use BibTeX, natbib or author-year keys. Do not characterize what a cited work concludes unless you have checked it. The `paper-search` MCP server indexes the reference corpus.
