# Monte Carlo sweep archive: 1000-trial baseline and 10,000-trial final

Protected copy of the Monte Carlo noise-sweep results behind the Separatrix/OW/OECS
paper (`Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_4A.tex`). Files
here are read-only (chmod 444). Do not overwrite; regenerate into the live
`experiments/outputs/mc_*/` folders and re-copy if a run needs to be redone.

## Contents

Each tracker has `summary_*_1000.csv` / `trials_*_1000.csv` (1000 trials/cell) and
`summary_*_10000.csv` / `trials_*_10000.csv` (10,000 trials/cell, the paper's final
numbers as of 2026-07-16).

- `separatrix/` -- Logic C separatrix tracker, fixed-start (0, 0.35) and random-start
  basin sweeps.
- `ow/` -- Okubo-Weiss boundary tracker (Logic G Newton-contour pentagon), fixed-start
  (-0.5, 0.25).
- `oecs/` -- OECS core-seek tracker (`oecs_trap_step`), fixed-start (0, 0.35).
- `mc_analysis_1000.txt` / `mc_analysis_10000.txt` -- text tables from
  `experiments/analyze_mc_sweeps.py` (separatrix + OW only; OECS is not read by that
  script, see its own `summary_fixed_10000.csv`).
- `compare_1000_10000.py` -- script used to diff every cell between the 1000-trial and
  10,000-trial runs (per-cell percentage-point deltas, cliff location, isotropy).

## Provenance

- **1000-trial separatrix fixed-start baseline**: originally generated 2026-07-03
  (git commit `1fd7b6c`), accidentally overwritten on disk by a 5-trial smoke test in
  commit `d36db4e` (2026-07-09). Restored from git commit `4de5e97` on 2026-07-16
  (`git show 4de5e97:trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/outputs/mc_separatrix/{summary,trials}_fixed.csv`).
  Verified: header `trials_per_cell: 1000`, 50007 trial rows, and cells match the
  values printed in the paper (e.g. sigma_uv=0, sigma_p=0.005/0.01/0.02/0.05 ->
  64.7/45.4/36.0/18.8%).
- **1000-trial separatrix random-basin, OW, OECS**: intact on disk throughout, no
  restore needed (dated 2026-07-03 basin/OW, 2026-07-09 OECS).
- **10,000-trial final runs**: generated 2026-07-16, commit `148df0a`, via
  `experiments/mc_10000_logs/run_all.log`. Auto-extension (doubling top sigma_uv up to
  3 times while worst-cell success > 10%) fired for separatrix fixed-start (extended to
  sigma_uv=0.8) and OECS (extended to sigma_uv=0.8); separatrix random-basin and OW
  finished on the base 7x5 grid, no extension needed.

## Cross-check summary (10,000-trial vs 1000-trial baseline)

All zero-noise cells: 100% success, as expected. The 50% success cliff (sigma_p=0)
remains at sigma_uv=0.01 for both the separatrix and OW trackers, unchanged from the
1000-trial baseline, consistent with the paper's claim that the estimator (not the
control law) sets the noise ceiling. Heading isotropy quartiles stayed flat (separatrix
75.9-77.1%, OECS not separately isotropy-checked but shares the estimator).

Per-cell deltas across all ~175 compared (sigma_uv, sigma_p) cells, all four
sweep/metric combinations: max |delta| = 3.3 percentage points (one OECS cell at
sigma_uv=0.1, sigma_p=0.01), everything else smaller. This is consistent with binomial
sampling noise at n=10,000 (expected std error ~0.5pp near 50% success; a few cells
exceeding 3pp by chance across ~175 comparisons is unremarkable, not a flag for a code
or methodology change). No cliff location moved. Full per-cell diff reproducible via
`compare_1000_10000.py`.

## Regenerate command

```bash
cd trunk/Python_Simulations/Vector_Fields/VF_Robot
venv/bin/python3 experiments/mc_sweep_separatrix.py --trials 10000 --workers 8 --starts fixed
venv/bin/python3 experiments/mc_sweep_separatrix.py --trials 10000 --workers 8 --starts random
venv/bin/python3 experiments/mc_sweep_ow.py         --trials 10000 --workers 8
venv/bin/python3 experiments/mc_sweep_oecs.py       --trials 10000 --workers 8
venv/bin/python3 experiments/analyze_mc_sweeps.py   > experiments/outputs/mc_analysis_10000.txt
```
