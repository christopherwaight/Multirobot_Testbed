# Paper_Draft_5A Condensation Changelog

Goal: 12 pages -> 11 pages for IEEE T-Mech. Reference version: Paper_Draft_4A.tex (untouched).
Baseline: 12 pages. Page 12 holds refs [28]-[53] plus both biographies (~1.4 columns), so ~0.7 page of savings is required.

| Location | Change | Rationale | Page fraction saved |
|---|---|---|---|
| whole text | Step 0: converted 35 hardcoded citation clusters [N] to \cite{N}; thebibliography block byte-identical | Reference removals now renumber automatically; verified all 53 \bibcite{N}=N in .aux, still 12 pages, no undefined cites. cite package now sorts/compresses in-bracket lists per IEEE convention (e.g. [2], [3], [4] -> [2]-[4]; [32], [23], [33], [34] -> [23], [32]-[34]) | ~0 (a few lines) |
| all 30 equations | Added \label{eq:N} to the 28 unlabeled equations; converted every hard prose reference (Eq.~(4), Eq.~11, Eqs.~(2)--(3), ...) to \ref-based soft references with identical rendering | IEEE how-to requires soft cross references; equation surgery later in this effort would silently break hard-coded numbers. Verified all 30 labels resolve to their original numbers | 0 |
| Appendix B, mirror-configuration note | BUGFIX: "beta from Eq. (24)" now references the beta definition, Eq. (23); (24) is the theta_c definition | Off-by-one cross reference. NOTE: this bug is also present in Paper_Draft_4A.tex (line 649), which is out of scope here; fix it there separately | 0 |
| Sec. III overview | Trimmed layer (2)/(3) descriptions duplicated verbatim by subsections III-A and III-B | Genuine redundancy; details preserved in the subsections | ~3 lines |
| Sec. III-A first sentence | Tightened robot-controller sentence duplicated by overview item (3) | Redundancy | ~1 line |
| Sec. III-B closing paragraphs | Merged the forward/inverse kinematics naming paragraph with the Jacobian paragraph; "the appendix" now says "Appendix B" (there are two appendices) | Redundancy + precision | ~3 lines |
| Sec. IV opening | Fixed verbless sentence ("For all trials in simulations, a multirobot cluster...") while tightening | Grammar bug + length | ~1 line |
| Sec. IV-B orbital discussion | Folded "This identical performance led to only one column..." into the preceding sentence | Redundancy | ~1 line |
| Sec. V-C first paragraph | Replaced protocol restatement with "Following the test plan above"; protocol already fully specified in V-B including lost-trial accounting | Verbatim duplication between Test Plan and Results | ~3 lines |
| Sec. VI-A kappa(J) paragraph | Merged two adjacent sentences that both said kappa(J) captures error amplification; fixed comma splice in the det(J)->0 sentence | Redundancy + grammar | ~2 lines |
| Sec. VI-A noise paragraph | Tightened four short sentences into two; content unchanged | Wordiness | ~1 line |
| Sec. II-A | Merged trivial "Each vector carries a magnitude and direction" into the field definition; condensed mission-motivation paragraph | Redundancy with itself and with the Introduction | ~3 lines |
| Sec. VI-B | Merged saddle-comparison paragraph into the drift paragraph, dropped "We note that" | Paragraph merge | ~1 line |
| Sec. V-A | BUGFIX: "housing for to minimize external lighting conditions" -> "housing to minimize external lighting effects" | Typo | 0 |
| Sec. V-D | Fixed parenthetical punctuation "(more than 3 complete orbits. See Table...)" | Grammar | 0 |
| 6 captions (Table I, Figs. control-arch/testbed/HSV/time-plot/primitive-comparison) | Tightened wording; fixed sloppy "Vs" caption; all numbers retained | Verbose captions; IEEE self-containment preserved | ~4 lines |
| Eqs. (1), (6), (7) [old numbering] | Inlined into prose: v(p*)=0 definition, the substituted zero conditions, and Jp*=-h; all were unreferenced one-line steps. Equation count 30 -> 27; soft refs renumber automatically. Control laws untouched | Trivial displayed equations; key results (A matrix, J/h definition, p*=-J^-1 h) remain displayed | ~8 lines |
| whole text | US spelling standardization (modelled/colour/summarises/travelled/neighbouring) | IEEE house style consistency | 0 |
| Introduction | Approved lighter-touch rewrite, ~1165 -> ~950 words. Paragraph flow, all examples, all citation brackets, contributions paragraph, and roadmap preserved verbatim; removed connective filler and one sentence that restated its predecessor | User-approved 2026-07-08 ("i like it"); protected flow untouched | ~0.25 page |
