# TASK (Session 2 of 2): Response to Reviewers

## Objective

Write a standalone LaTeX response letter, `Paper_Writing/Systems Paper Revision/Response_to_Reviewers.tex`, for the IEEE Systems Journal major revision (manuscript ISJ-RE-26-22639). Run this session only after Session 1 (`REVISION_PROMPT_1_paper.md`) has finished and Chris has looked at its report.

## Inputs

- `Reviewer_Feedback.txt`: the reviews, quoted verbatim in the letter.
- The revised paper `cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex` and its compiled PDF. The letter cites its section, equation, table and figure numbers, so read them from the compiled `.aux` or PDF, not from memory.
- `Queued_Edits/revision_change_log.md` and `Queued_Edits/revision_numbers_ledger.md`, written by Session 1.
- The live tracker, https://claude.ai/artifact/AYRPHnRGad1f4GiUQYkURc (`actions` collection, read with ArtifactData). It holds 67 actions. Every action must be answered or explicitly declined in the letter.
- Response text already drafted in `vib_baselines_note/vib_baselines.tex` (the "Response letter (vector-sum only...)" block) and in the error, stability and minimality notes where they address a reviewer directly.
- `CLAUDE.md` and `Waight_style.md` at the repo root.

## Files

Create only `Response_to_Reviewers.tex`, plus its build outputs. Do not edit the paper. If the letter needs a change in the paper, list it at the end of your report for Chris.

## Structure

1. A short opening to the Associate Editor that summarizes the main changes in four to six sentences, with no flourish.
2. One section for the Associate Editor and one per reviewer. For each comment:
   - quote the reviewer's text verbatim, split into the lettered sentences the tracker uses;
   - give the response, saying what was changed and why, and naming the physical mechanism where the answer is technical;
   - give the location of each change (section, equation, table, figure) with the paper's actual numbers.
3. Where a request was not adopted, say so and give the reason. Setting a question aside is different from never addressing it. Say which it is.
4. Where several reviewers ask for the same work (nonlinear benchmarks, stability, metrics, three-robot minimum), answer it fully once and refer to that answer elsewhere.

## Wording rules

- `Waight_style.md` applies. No em-dashes, no colons in running prose, no AI-voice vocabulary, no flattery of the reviewers beyond one plain thanks.
- **Hardware campaign.** Use the wording Chris agreed: "All hardware experiments were repeated in a new campaign designed around the reviewers' requests. All hardware results in the revised manuscript come from this campaign and supersede the earlier results." The campaign has not run yet. Keep the sentence, wrapped in the same red `\syn{}` or magenta `\open{}` markers the paper uses, so it cannot be sent before the data exist.
- **Old versus new hardware numbers.** Do not attribute differences to recalibration, map wear or anything else without evidence. The known cause of the vortex difference is that the published vortex rows used four-robot runs.
- **Comparison primitives.** Say "the vector-sum comparison was removed". Never imply that a modified vector-to-scalar orbit was simulated or tested. It was not. R1.5's request to clarify that baseline is answered by explaining that the matched baselines replaced it.
- **Numbers.** Every number in the letter must appear in the paper or in the ledger, with the same marker. Introduce no new number.
- **Classification (R1.7).** State what was evaluated (simulation accuracy with the classifier band; hardware metric pending the campaign), and how the claim was qualified.
- Do not promise future work to win a point.

## Compile and report

Compile twice with BasicTeX (`export PATH="/Library/TeX/texbin:$PATH"`) and report errors and warnings. Report to Chris in prose and a table:
- each tracker action and where the letter answers it;
- any comment answered only partly;
- every sentence in the letter that depends on the campaign;
- any paper change the letter needs.
