# unsteady_dg: status and queue

Robustness study for the Separatrix / Okubo-Weiss paper (Draft_13): do the D and
s1 trackers follow the separatrix on fields without the steady double gyre's
special structure? Write-up for the author:
`Paper_Writing/Separatrix_and_OW_Paper/Draft_13_unsteady_dg_proposed.tex` (standalone,
19 pages; Part III is proposed paper text, nothing applied to Draft_13.tex).

## Done (2026-11-06)

- Time-periodic gyre (`mc_unsteady_double_gyre.py`, tags n1000, eps025_vmax008)
- Bent gyre, constructed field (`mc_bent_double_gyre.py`, tag n500)
- Bickley jet, co-moving steady frame (`mc_bickley_jet.py`, tag n1000)
- Scope ratio |w|/2r along the true separatrix (`scope_ratio.py`) and fitted along
  rides (`path_ratio.py`, tag n50)
- Figures for the notes (`fig_notes.py` -> Paper_Writing/.../figures/notes_unsteady/)

Finding: each tracker follows a trench of its own field (D or s1). The trenches
mark the separatrix only where strain outweighs rotation on it. Retention against
the separatrix falls with the fitted ratio: below ~0.1 fine, ~0.3 partial, >0.5 none.

## Queue (started 2026-11-06, author away; session may be interrupted)

1. **Own-trench scoring** (`own_trench.py`). Rerun bent (delta 0.1, 0.2) and
   Bickley trials with path recording, same seeds as the full runs, and score
   straddle retention and error against each tracker's OWN true trench instead of
   the separatrix. High own-trench retention where separatrix retention is low
   means "wrong target, not lost target". Output: `outputs/own_trench_<tag>.csv`.
   Status: see bottom of this file.
2. **D tracker S6/S7 failure on the bent gyre** (`diag_bent_s6.py`). From S6 and
   S7 the D tracker fails every trial at both bends (descends to the bottom wall,
   leaves the basin along it). Log the fitted Hessian eigenvalues, crest test,
   and Newton step per step, and compare with the steady gyre from the same start.
3. **Bickley jet citation.** The paper-search corpus has no primary source.
   Find candidates, verify bibliographic details, and list them for the author.
   Do not add a bibitem without the author's approval. The bent gyre is a
   constructed field; whether it belongs in the paper is the author's call.

## Results log
