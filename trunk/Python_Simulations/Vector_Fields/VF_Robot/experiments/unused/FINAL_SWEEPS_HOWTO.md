# Final 10,000-trial Monte Carlo runs (manual, run when ready)

Written 2026-07-03. The paper currently reports the 1000-trial-per-cell
sweeps (committed in `outputs/mc_separatrix/` and `outputs/mc_ow/`,
tables in `outputs/mc_analysis_1000.txt`). If you want the tighter
confidence intervals (about plus or minus 1 point instead of 3 at the
50 percent level), run the 10,000-trial versions below. Nothing else in
the pipeline changes.

## Commands

```bash
cd trunk/Python_Simulations/Vector_Fields/VF_Robot

# 1. Archive the 1000-trial CSVs first (the runs overwrite them):
cp experiments/outputs/mc_separatrix/trials_fixed.csv \
   experiments/outputs/mc_separatrix/trials_fixed_1000.csv
cp experiments/outputs/mc_separatrix/summary_fixed.csv \
   experiments/outputs/mc_separatrix/summary_fixed_1000.csv
cp experiments/outputs/mc_ow/trials_fixed.csv \
   experiments/outputs/mc_ow/trials_fixed_1000.csv
cp experiments/outputs/mc_ow/summary_fixed.csv \
   experiments/outputs/mc_ow/summary_fixed_1000.csv

# 2. Separatrix (roughly 45-70 min at 8 workers):
venv/bin/python3 experiments/mc_sweep_separatrix.py \
    --trials 10000 --workers 8 --starts fixed

# 3. Okubo-Weiss (roughly 30-50 min):
venv/bin/python3 experiments/mc_sweep_ow.py --trials 10000 --workers 8

# 4. Regenerate the review tables:
venv/bin/python3 experiments/analyze_mc_sweeps.py \
    > experiments/outputs/mc_analysis_10000.txt
```

## What to update in the paper afterwards

- `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex`,
  Testing Plan subsection (`sec:test_plan`): change "1000 trials per
  cell" to "10\,000 trials per cell".
- Any success-rate numbers already quoted in Results text or tables:
  re-read them from the new `mc_analysis_10000.txt` (expect shifts of a
  couple of points at most; the cliff location at sigma_uv = 0.01
  should not move).
- Commit the new CSVs and the .tex change together so the numbers and
  their source stay in one commit.

## Sanity checks on the new output

- Zero-noise cells: separatrix success_traverse = 100 percent, OW
  success_track = 100 percent. If not, something changed in the code;
  diff against commit 4de5e97 before trusting anything.
- Cliff at sigma_p = 0 should remain at sigma_uv = 0.01 for both
  controllers (the estimator-sets-the-ceiling result).
- Heading quartiles should stay flat (isotropy).
