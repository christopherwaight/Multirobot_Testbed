# Hardware campaign, 2025-07-31 to 2025-08-06

These are copies made on 2026-11-15; the original files are still where they were. `MANIFEST_copied_files.tsv` lists each file's source path, original timestamp and SHA-256, and every copy was hash-checked against its source.

| Folder | Date | Source | Files |
|---|---|---|---|
| `Saddle/three_robot/convergence` | 2025-07-31 | `trunk/robots_3/saddle_tests/run*.mat` | 78 (run43 and run64 never existed) |
| `Saddle/three_robot/orbit` | 2025-07-31 | `.../saddle_tests/radius*.mat` | 7 (radius001, 01 to 06) |
| `Vortex/four_robot/convergence` | 2025-08-05 | `trunk/robots_4/run*.mat` | 80 |
| `Vortex/four_robot/orbit` | 2025-08-05 | `trunk/robots_4/radius*.mat` | 5 (radius001, 01 to 04) |
| `Vortex/three_robot/convergence` | 2025-08-06 | `trunk/robots_3/vortex_tests/3-robot-tests/run*.mat` | 80 |
| `Vortex/three_robot/orbit` | 2025-08-06 | `.../3-robot-tests/radius*.mat` | 6 (radius001, 01 to 05) |

- **Convergence runs:** each lasts 10 s (101 samples). The original plotters treat them as 8 start positions × 10 runs (runs 1–10, 11–20, …).
- **Orbit runs:** each lasts 60 s. The file name gives r_d, so `radius01` means 0.1 m and `radius001` means 0.01 m.
- **Field check for the 4-robot runs (2026-11-15):** they are on the vortex. They hold 4 poses and 4 hue columns, and their end-of-run hue readings match the October 4-robot vortex runs at the same positions.
- **Estimator for the 4-robot runs:** a single 4-point least-squares fit, not two averaged 3-robot subgroups. I rebuilt the critical point from the logged poses, hue and saturation (scratchpad `ls_or_dual.m`, 2026-11-15), using the decode hue → angle 2π·h with magnitude (sat − 0.3)/0.7.
  - **Least squares:** reproduces the logged `detected_critical_point` exactly at the median step.
  - **Every subgroup-average version:** misses by 2.7–8 cm at the median.
- **⚠ The 4-robot runs had CROSSED SENSOR WIRES (mux), found 2026-11-15.** The controller ran on scrambled inputs.
  - **How it was found:** each logged hue/sat column was tested against every robot's expected field at its own position (scratchpad `mux_pairing.m`, `mux_pairing2.m`, `mux_impact.m`).
  - **What each column actually holds:**
    - `hue` column 3 = robot 3's **saturation** (it tracks robot 3's distance from the centre, r = 0.86–0.98);
    - `saturation` column 2 = robot 3's **hue** (1–3° direction error against robot 3);
    - `saturation` column 3 = robot 2's **saturation** (r = 0.95–0.98 with robot 2's distance).
  - **Unaffected:** robots 1 and 4, and robot 2's hue, are correct.
  - **Control used them:** the logged critical point is reproduced exactly (0.0000 m) by least squares on the columns *as logged*, and misses by 4.6 cm with the corrected columns. So the controller really ran with robot 3's direction replaced by a saturation value, and with robots 2 and 3 using the wrong magnitudes.
  - **Cost:** over the last 3 s of each run, the estimate was 3.1 cm from the true centre (median; 90th percentile 5.6 cm). With correct wiring the same data gives 1.4 cm (90th percentile 3.8 cm).
  - **Consequence:** the August 3-robot vs 4-robot comparison is NOT a fair team-size comparison. The 4-robot cluster still had the lower typical error (2.3 vs 3.8 cm) **despite** one robot effectively feeding garbage. That's a robustness result for least squares, not a clean "4 beats 3".
  - **The 3-robot runs** in this campaign pass the same test cleanly: each column matches its own robot within 1°.
  - **The November 2026 4-robot model is correctly wired:** its `sensor_hue`/`sensor_sat` and `field_uv` columns each match their own robot within 2–3°. So the mux was fixed some time between August 2025 and November 2026. It was not fixed by Claude; the 2026-11-12 build diff vs `chris_4R_all_primitives.slx` contains no rewiring.
- **The other old 4-robot model bugs did not affect these runs** (checked from the logged poses, 2026-11-15, scratchpad `aug4R_wiring.m`). The bugs are listed in `trunk/robots_4/README_four_robot.md`, Phase 2.
  - **φ wrap bug:** φ, recomputed with the old forward-kinematics formula, stayed between 0.85 and 1.99 rad in all 85 runs (run medians 1.570 ± 0.05). It never left [−π, π], so the 2π error jump never happened.
  - **Robot order:** the diagonals 1–3 and 2–4 are 0.40 m and the sides 0.28 m (ratio √2), so the square was correct, and φ = +π/2 matches the setpoint.
  - **Decode mirror:** it doesn't move the critical point, but don't use these runs for field-type classification.
  - **NaN critical-point steps:** none.
  - **Caveat:** the August model file isn't on disk (`chris_4R_all_primitives.slx` was saved on 2025-10-10). So this is a check of the bugs' symptoms in the data, not of the August code itself.
- **This campaign's 4-robot setup differs from 2026-11:**
  - a 0.40 m diagonal square (0.30 m in 2026-11);
  - the old Orbiter law;
  - the θc controller switched on.

  Compare 3-robot vs 4-robot within this campaign, not against 2026-11 orbit radii.
- **Coordinates:** these runs use the old, origin-centred coordinates and the old variable names (`robotN_pose`, `hue`, `saturation`, `detected_critical_point`). There is no `run_config`, and the Orbiter gains differ from the 2026-11 campaign.
- **Left out on purpose:** the map scans (`scanning_entire_vortex_mat.mat`, `sampling_everywhere_vortex.mat`, `vector_field_grids.mat`).
