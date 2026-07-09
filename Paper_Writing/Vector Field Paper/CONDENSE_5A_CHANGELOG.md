# Paper_Draft_5A Condensation Changelog

Goal: 12 pages -> 11 pages for IEEE T-Mech. Reference version: Paper_Draft_4A.tex (untouched).
Baseline: 12 pages. Page 12 holds refs [28]-[53] plus both biographies (~1.4 columns), so ~0.7 page of savings is required.

| Location | Change | Rationale | Page fraction saved |
|---|---|---|---|
| whole text | Step 0: converted 35 hardcoded citation clusters [N] to \cite{N}; thebibliography block byte-identical | Reference removals now renumber automatically; verified all 53 \bibcite{N}=N in .aux, still 12 pages, no undefined cites. cite package now sorts/compresses in-bracket lists per IEEE convention (e.g. [2], [3], [4] -> [2]-[4]; [32], [23], [33], [34] -> [23], [32]-[34]) | ~0 (a few lines) |
| all 30 equations | Added \label{eq:N} to the 28 unlabeled equations; converted every hard prose reference (Eq.~(4), Eq.~11, Eqs.~(2)--(3), ...) to \ref-based soft references with identical rendering | IEEE how-to requires soft cross references; equation surgery later in this effort would silently break hard-coded numbers. Verified all 30 labels resolve to their original numbers | 0 |
| Appendix B, mirror-configuration note | BUGFIX: "beta from Eq. (24)" now references the beta definition, Eq. (23); (24) is the theta_c definition | Off-by-one cross reference. NOTE: this bug is also present in Paper_Draft_4A.tex (line 649), which is out of scope here; fix it there separately | 0 |
