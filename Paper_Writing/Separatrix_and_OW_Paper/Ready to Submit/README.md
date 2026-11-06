# Ready to Submit

**Second-Order Cooperative Field Estimation and Tracking of Coherent Flow Structures Using Multirobot Systems**
Christopher Waight and Christopher A. Kitts. Target: IEEE Systems Journal.

Copied from `../Draft_13.tex` on 2026-11-05. Edit `Draft_13.tex`, not this copy, and recopy when it changes.

## Contents

| File | What it is |
|---|---|
| `ready_to_submit.pdf` | Compiled paper, 11 pages, US letter |
| `ready_to_submit.tex` | LaTeX source (IEEEtran journal class, inline `thebibliography`, no .bib) |
| `figures/` | The nine figures the source uses, and only those |
| `ready_to_submit_source.zip` | `ready_to_submit.tex` and `figures/`, for a source upload |

| Fig. | File |
|---|---|
| 1 | `double_gyre_fields.png` |
| 2 | `control_architecture.png` |
| 3 | `pentagon_formation.png` |
| 4 | `s1_channels.png` |
| 5 | `traverse_vs_logic_c_seven.png` |
| 6 | `separatrix_formation_seven.png` |
| 7 | `noise_seven_starts_col.png` |
| 8 | `objectivity_rotating.png` |
| 9 | `ocean_progression_2km.png` |

## Build

```bash
export PATH="/Library/TeX/texbin:$PATH"
pdflatex ready_to_submit.tex && pdflatex ready_to_submit.tex
```

Packages: amsmath, amsfonts, amssymb, array, stfloats, url, graphicx, cite. All ship with TeX Live and BasicTeX.

## Checks done on 2026-11-05

- Builds standalone with no errors, undefined references, or overfull boxes.
- No placeholder markers or internal notes in the source.
- Every simulation result re-run on the post-sign-fix code; numbers and figures unchanged.
