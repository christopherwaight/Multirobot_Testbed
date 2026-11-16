# Campaign analysis scripts (2026-11-11 to 2026-11-13)

Written by Claude (AI assistant) while analysing the hardware campaign. **All of them are read-only:** they load run files and print results, and none modifies data or models.
- **Main script:** **`campaign_report.m`** produces every table in `../2026-11-13_vortex_orbit_campaign.md`.
- **The rest:** the focused analyses behind specific claims, kept so each claim can be re-checked.
- **Ranking:** start with `campaign_report.m`; the others are listed in order of importance.

```matlab
addpath('trunk/campaign_results/scripts'); addpath('trunk/robots_3');   % load_run.m lives in robots_3
T = campaign_report('data_hardware_campaign_11_11_2026');
```
- **Arguments:** most scripts take a folder of run files and/or the true centre `[0.839 0.856]` (Motive x, z).
- **Folder layout:** some expect a flat folder (one `.mat` per run); some expect one sub-folder per condition. Each script's header says which.

| Script | What it shows | Used in |
|---|---|---|
| *(field reconstruction)* | Not written yet. The plan and its pitfalls are in `../2026-11-15_field_reconstruction_and_parameters.md`, Part A. | anticipated-vs-measured maps |
| `fit_coulomb_delay.m` | Robot response model: Python Omnibot "coulomb" structure + input delay + gain, fitted per robot and pooled; speed-cap check. Main result of `../2026-11-15_robot_response_model.md` | robot model for the sims |
| `fit_coulomb.m` | First (no-delay) fit + steady-state speed-vs-command bins + quick delay test | robot model, §3.2 |
| `delay_locate.m` | Shows the delay is between command and motion, not inside the controller | robot model, §3.4 |
| `clean_campaign_files.m` | Clean copies of run files containing only what each run logged (removes stale workspace variables; see `data_hardware_campaign_11_11_2026/README_STALE_VARIABLES.md`). Dry run by default; never modifies the originals. | data hygiene |
| `campaign_report.m` | Everything: inventory, battery screen, true centre (and map-slide check), per-condition radius and estimate statistics, estimate vs distance, plant fit, per-robot sensor accuracy | the whole write-up |
| `cp_error_vs_distance.m` | Estimate error vs distance in the data, plus a noise model (measured reading noise on an ideal vortex) reproducing the trend | §4.2 |
| `cp_distance_bias.m` | Why the distance to the estimate looks longer at small radii (√(r² + 2σ²)) and shorter at large radii | §3.1, §4.2 |
| `campaign_lag.m` | Outlier rate, robust jitter, the estimate's offset in the cluster frame (the lead along the motion), sensor-delay test, closed-loop radius with the measured offset | §3.1 |
| `campaign_sanity.m` | First per-condition pass: sensor fit R², saturation vs distance, plant re-fit, closed-loop radius prediction | §3.1 |
| `why_4R_worse.m` | 4R leave-one-robot-out on the logged data; triangle vs square under the same noise | §5 |
| `cp_vs_dist_3R4R.m` | 3R vs 4R estimate quality at equal distance (superseded by report section D) | §5 |
| `calib_fix_test.m` | Recomputes the estimate with per-robot bias and strength corrections | §5 |
| `compare_3R_4R.m` | Early per-condition 3R vs 4R table (superseded by report section C) | §5 |
| `fit_true_cp.m` | True centre by four methods (linear field, direction-only, magnitude-only, with sensor offsets), plus per-condition and per-run fits | §6 |
| `cp_drift_vs_radius.m` | Map slide vs radius-dependent fit: inner vs outer readings, early vs late runs | §6 |
| `robot_accuracy.m` | Per-robot reading accuracy ranking (superseded by report section F) | §8 |
| `net_trained_on.m` | The sensor net on each robot's calibration and held-out files | §8 |
| `robot_health_4R.m` | Per robot, per run: speed reached (battery gauge), direction error, fitted gain | §7.2 |
| `confirm_4R_ls.m` | Proves the 4R runs used the 4-point least-squares estimate | §9 |
| `check_4R_run.m` | Diagnoses robot order, shape, and command vs motion per robot (the wrong-order test run) | §9 |
| `check_4R_data.m` | Logging and sanity check of 4R run files | §1 |
| `check_runs.m` | Logging and sanity check of 3R run files (first new-logging runs) | §1 |
| `r04_detail.m` | Per-run radius over time at r_d 0.4 (steadiness) | §4.3 |
| `orbit_predict.m` | 2026-11-11 plant fit and closed-loop radius predictions for different Orbiter laws | §3.1, §7.1 |
| `kt_compare.m` | k_t 0.20 vs 0.15 prediction. **Never run** (cancelled); kept for reference | – |

**Note on `campaign_report.m`:** a couple of the per-condition and per-run centre fits can hit `fminsearch`'s evaluation limit. They do so in a nuisance parameter (the angle bias of a robot that isn't in that subset), not in the centre. The centre values were identical, to 0.1 mm, between runs with a 40,000 and a 200,000 evaluation limit.
