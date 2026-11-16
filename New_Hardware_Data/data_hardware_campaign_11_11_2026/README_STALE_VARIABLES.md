# ⚠ READ FIRST: these run files contain STALE variables

**Found 2026-11-15 by Claude (AI assistant).**

Every `.mat` file in this campaign was saved with a plain `save` of the whole MATLAB workspace. So besides what the run actually logged, each file also contains variables left over from **earlier sessions**. Those leftovers are NOT this run's data. Some even have the same names a different model uses.

The variables the model writes are overwritten on every run, so **the run's own data is correct**. The danger is reading a leftover by mistake.

## What's in which files (all 158 files checked)
| Files | Stale variables | Where they came from |
|---|---|---|
| **3-robot saddle, r_d 0.01–0.3** (40 files) | **4-robot** `pose_r4`, `field_cp_jacobian(s)`, `orbit_radius_jacobian(s)`, `FK_results`, `*_r4`, `*_jacobians` (13 variables), + 24 old-name variables | the 2026-11-13 4-robot session |
| **4-robot vortex** (52 files) | **3-robot** `field_cp`, `orbit_radius`, `simfield_hue/sat`, `cmd_*_vectorsum/ascent/descent` (10 variables), + 24 old-name variables | the 3-robot session that came before |
| **3-robot vortex** (60 files) | 24 old-name variables from 2025, e.g. `robot1_pose`, `p`, `q`, `beta` (601 rows), `hue` / `saturation` (timeseries), `detected_critical_point`, `ans` | an old 2025 workspace |
| **3-robot saddle, r_d 0.4** (saved after the MATLAB restart at about 22:10) | 7 empty placeholders (0 rows) plus `ans` | the setup scripts |

**Duplicate:** `Vortex/three_robot/vortex_001/vortex_4robots_radius_of_0_point_01_run1.mat` is a misplaced copy of the 4-robot file with the same name in `Vortex/four_robot/robot4_vortex_001/`.

## Rules for reading these files
1. **Robot count:** take it from the run's own command log, `size(cmd_robots_selected, 2) / 3`, or from the file name. **Never decide it from whether `pose_r4` exists**, because 3-robot saddle files contain a stale `pose_r4`.
2. **Read only the variables the run's model logs.** They're listed in `run_config.log_legend`.
   - 3 robots: `field_cp`, `orbit_radius`.
   - 4 robots: `field_cp_jacobian`, `orbit_radius_jacobian`.
3. **Use `trunk/robots_3/load_run.m`.** It was fixed on 2026-11-15 to use rule 1; before the fix it would have treated saddle files as 4-robot. `trunk/campaign_results/scripts/campaign_report.m` uses the file name and was always safe.

## Cleaning
`trunk/campaign_results/scripts/clean_campaign_files.m` writes **clean copies** that keep only what each run logged. The originals are never modified, and every copy is checked to be bit-identical for the variables it keeps.
```matlab
clean_campaign_files(src)                 % dry run: lists what would be dropped
clean_campaign_files(src, dst, true)      % writes the copies + CLEANING_MANIFEST.csv
```

**Clean copies exist:** `../data_hardware_campaign_11_11_2026_clean/`, written 2026-11-15 (171 files) and verified. Re-run the command above to refresh it after more runs.

## Prevention (for future sessions)
Run `clear` at the start of every session, **or** save only the logged variables:
```matlab
save(f, '-regexp', '^(pose_|sensor_|field_|orbit_|cluster_state|setpoint|err_|cmd_|simfield_|diag_|timing|tout|run_config)')
```
