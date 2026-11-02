# TASK (Session 2 of 2): Response to Reviewers

## Objective

Write a standalone LaTeX response letter, `Paper_Writing/Systems Paper Revision/Response_to_Reviewers.tex`, for the IEEE Systems Journal major revision of manuscript ISJ-RE-26-22639.

Run this session only after Session 1 (`REVISION_PROMPT_1_paper.md`) has finished and Chris has reviewed its report. Session 1 revised the paper surgically. The submitted text was kept and changed only where a tracker action required it. The letter should make each of those targeted changes easy to find and check.

## Inputs

- **`Reviewer_Feedback.txt`:** the reviews, quoted verbatim in the letter.
- **The revised paper,** `cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex`, and its compiled PDF. Take section, equation, table and figure numbers from the compiled `.aux` or the PDF, never from memory.
- **The marked-up PDF,** `cwaight_systems_paper/cwaight_systems_paper_diff.pdf`. Rebuild it first with `sh make_diff.sh` from `Systems Paper Revision/`. It shows every change against the submitted paper.
- **`Queued_Edits/revision_change_log.md`:**
  - Part 1 lists every edit with its tracker action and edit type.
  - Part 2 lists where each of the 67 actions is answered.
- **`Queued_Edits/revision_numbers_ledger.md`:** every marked number and its source.
- **The live tracker,** https://claude.ai/artifact/AYRPHnRGad1f4GiUQYkURc, read with ArtifactData from its `actions` collection. Every one of its 67 actions must be answered or explicitly declined.
- **The donor letter,** `Queued_Edits/rewrite_v1/Response_to_Reviewers_v1.tex`. It was written for a first revision that rewrote most of the paper, and that revision has been replaced.
  - **Reuse its verbatim quotes,** already split into the tracker's lettered sentences. Check each one against `Reviewer_Feedback.txt`.
  - **Reuse its answers to technical questions** where the paper still makes the same change.
  - **Re-derive every location and every "we changed" statement** from the new change log, because the paper changed differently.
- **Response text already drafted** in `vib_baselines_note/vib_baselines.tex` (the "Response letter (vector-sum only...)" block), and in the error, stability and minimality notes where they address a reviewer directly.
- **`CLAUDE.md` and `Waight_style.md`** at the repo root.

## Files

Create only `Response_to_Reviewers.tex` and its build outputs, plus the rebuilt diff PDF. Do not edit the paper. If the letter needs a change in the paper, list it at the end of your report for Chris.

## Structure

1. **Opening.** A short note to the Associate Editor, four to six sentences with no flourish. It summarizes the main changes and says that the submitted text was retained and changed only where a comment required it.

   Ask Chris whether he will upload the marked-up PDF. If he will, add one sentence saying that the changes are marked in it.
2. **One section for the Associate Editor and one per reviewer.** For each comment:
   - quote the reviewer's text verbatim, split into the lettered sentences the tracker uses;
   - give the response, saying what was changed and why, and naming the physical mechanism where the answer is technical;
   - give the location of each change (section, equation, table, figure) using the paper's actual numbers;
   - where the change is an inserted sentence or short paragraph, quote the new text, so the reviewer can check it without hunting.
3. **Requests not adopted.** Say so, and give the reason. Setting a question aside is different from never addressing it, so say which it is.
4. **Shared requests.** Several reviewers ask for the same work: nonlinear benchmarks, stability, metrics and the three-robot minimum. Answer each fully once, and refer to that answer elsewhere.

## Wording rules

- **Style.** `Waight_style.md` applies. No em-dashes, no colons in running prose, no AI-voice vocabulary, and no flattery of the reviewers beyond one plain thanks.
- **Describing the edits.** Say what was added, corrected or removed. Do not describe sections as "rewritten" or "restructured" unless the change log shows an edit of type (e) there.
- **Hardware campaign.** Use the wording Chris agreed:

  > "All hardware experiments were repeated in a new campaign designed around the reviewers' requests. All hardware results in the revised manuscript come from this campaign and supersede the earlier results."

  The campaign has not run yet. Keep the sentence, wrapped in the same red `\syn{}` or magenta `\open{}` markers the paper uses, so the letter cannot be sent before the data exist.
- **Old versus new hardware numbers.** Do not attribute differences to recalibration, map wear or anything else without evidence. The known cause of the vortex difference is that the submitted vortex rows and both vortex convergence figures came from four-robot runs. The v1 letter disclosed this. Keep the disclosure, and list it in your report as a decision for Chris to confirm.
- **Comparison primitives.** Say "the vector-sum comparison was removed". Never imply that a modified vector-to-scalar orbit was simulated or tested, because it was not.

  R1.5 asks to clarify that baseline. Answer it this way: the submitted VI-B described a vector-to-scalar primitive, the figure showed only vector-sum and the proposed controller, no results for it were reported, and the matched baselines replaced it.
- **Numbers.** Every number in the letter must appear in the paper or the ledger, with the same marker. Introduce no new number.
- **Classification (R1.7).** State what was evaluated, namely simulation accuracy with the classifier band, and that the hardware metric is pending the campaign. Then state how the claim was qualified.
- **No promises.** Do not promise future work to win a point.

## Compile and report

Compile twice with BasicTeX (`export PATH="/Library/TeX/texbin:$PATH"`), and report errors and warnings.

Then report to Chris in prose and tables:
- each tracker action, and where the letter answers it;
- any comment answered only partly;
- every sentence in the letter that depends on the campaign;
- any change the paper needs;
- the decisions he should confirm, including the four-robot disclosure and the marked-up-PDF sentence.
