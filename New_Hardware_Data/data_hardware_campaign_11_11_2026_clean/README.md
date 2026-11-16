# Clean copies of `data_hardware_campaign_11_11_2026`

**Made 2026-11-15 by Claude (AI assistant)** with `trunk/campaign_results/scripts/clean_campaign_files.m`.

- **Same folders and file names** as the original campaign folder, which is **unchanged**. Keep it as the raw record.
- **Each file holds only what that run logged:** the variables in its model's `run_config.log_legend`, plus `tout` and `run_config`. For 4-robot files, the debug `FK_results` is also kept.
- **Removed:** stale workspace leftovers from other sessions and models. See `../data_hardware_campaign_11_11_2026/README_STALE_VARIABLES.md`.
- **Verified:**
  - every kept variable is bit-identical to the original;
  - every kept time series has `numel(tout)` rows;
  - no expected variable was missing.
- **Manifest:** `CLEANING_MANIFEST.csv` lists every file, its robot count, and what was dropped.
- **Includes:** the failed run `Robot4_vortex_050_runs_off_the_mat.mat`, and the misplaced duplicate `Vortex/three_robot/vortex_001/vortex_4robots_radius_of_0_point_01_run1.mat`, because they're in the original.
- **Only `.mat` files are copied.** The `ANALYSIS.txt` notes stay in the original folder.

**To refresh after more runs**, re-run (from `trunk/campaign_results/scripts`):
```matlab
clean_campaign_files('C:\Users\cwaight\Desktop\MultiRobot\data_hardware_campaign_11_11_2026', ...
                     'C:\Users\cwaight\Desktop\MultiRobot\data_hardware_campaign_11_11_2026_clean', true)
```
It rewrites every file, which is fine because it always starts from the originals.
