# Hardware campaign, 2025-10-09 to 2025-10-10: 4-robot vortex

These are copies made on 2026-11-15; the original files are still in `trunk/robots_3/vortex_tests/`. That folder name is misleading, because **every run in it is 4 robots** (`robot1..4_pose`, 4 hue columns). `MANIFEST_copied_files.tsv` lists each file's source path, original timestamp and SHA-256, and every copy was hash-checked against its source.

This campaign started more than 2 months after the July/August one, so it is kept as a separate campaign.

## Convergence (10 s per run; same 8 start positions × 10 runs in both sets)
| Folder | Runs | Estimator steered to |
|---|---|---|
| `convergence/least_squares_run1-80` | 79 | `detected_critical_point` |
| `convergence/subgroup_average_run101-180` | 80 | `detected_critical_point_avg` |

Both estimates are logged in every run, but only one drove the controller. To tell which, I compared the cluster's position over the last 3 s with each estimate (scratchpad `which_cp.m`, 2026-11-15). Median end distances:
- **runs 1–80:** 0.015 m to `cp`, 0.055 m to `cp_avg`;
- **runs 101–180:** 0.041 m to `cp`, 0.017 m to `cp_avg`.

The labels "least squares" and "subgroup average" come from the variable names: `_avg` is taken to be the average of the 3-robot subgroups. **Confirm this before publishing.**

**Excluded:** two files in this set are not run logs.
- `run35.mat` holds only the real-time pacer's state.
- `run81.mat` is a saved workspace from `all_data_combiner.m` (combiner variables plus 272 s of concatenated data).

So `least_squares_run1-80` has 79 runs.

## Orbit (60 s per run)
| Folder | Files | Date |
|---|---|---|
| `orbit/set_0xx` | orbit001, 010, 020 … 070 | 10-09 |
| `orbit/set_1xx` | orbit101, 110, 120 … 160 | 10-09 |
| `orbit/set_2xx` | orbit201, 210, 220, 230, 233, 240, 250 | 10-10 (also logs `FK_results`, `angle_error_diagnostics`) |

- **Estimator:** the 0xx and 1xx sets probably follow the same split as the convergence runs (least squares, then subgroup average), but **the data doesn't confirm it**. In both sets the radius measured to `cp` is steadier, and in 2xx the two estimates agree too closely to tell apart.
- **r_d:** the last two digits look like r_d (010 means 0.1 m), but that is unconfirmed. The measured radii run from about 0.27 to 0.69 m.
- **`orbit233`:** it's unclear what this one is.

These runs use old origin-centred coordinates and old variable names, and there is no `run_config`.
