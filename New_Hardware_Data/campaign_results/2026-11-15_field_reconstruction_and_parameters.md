# Notes, 2026-11-15: reconstructing the measured field, and the new robot-model parameters

**Written by Claude (AI assistant) at the user's request, as notes to think about later. Nothing has been implemented.**

---

# Part A: reconstructing the field (anticipated vs measured)

## A1. Goal
- **What to make:** a map showing where the robots' sensors (through the net) read the floor field correctly and where they don't. Anticipated field = the printed design; measured field = what the robot believed.
- **What to show:** direction error (hue) and magnitude error (saturation) over the **whole** map, corners included.
- **What it's for:** sensor-error figures in the paper and thesis, and explaining the estimator's behaviour far from the centre (campaign write-up §4). This is probably what the 2025 "sampling everywhere" / "scanning entire mat" scripts were for.

## A2. The problem with this campaign alone: it's a disc
- **Coverage:** the 2026-11 robots stayed within about **0.7 m** of the vortex centre (p99 0.73, max 0.88 m). So a map from this campaign alone is a **circle**, with empty corners.
- **Measured with 10 cm cells,** cells visited in the corner band of a square of each half-width:

| Data | ±0.5 m square | ±0.7 m | ±0.9 m |
|---|---|---|---|
| 2026-11 vortex campaign (all) | 48/48 | 66/84 | **19/136** |
| **2025-09-16 `sampling_everywhere_vortex.mat` (3R)** | 48/48 | **84/84** | **90/136** |
| 2025-09-05 `scanning_entire_vortex_mat.mat` (4R) | 48/48 | 78/84 | 50/136 |
| 2025-08 vortex runs, 3R + 4R (all) | 48/48 | 80/84 | 61/136 |

**The corners come from 2025, mainly `trunk/robots_3/vortex_tests/3-robot-tests/sampling_everywhere_vortex.mat`:** 3 robots, 500 s, robots reaching ±0.8 m from the centre.

## A3. What each dataset recorded
| | 2026-11 campaign | 2025 runs and scans |
|---|---|---|
| raw colour sensor counts | **yes**: `sensor_rgbk_raw_rN` [K R G B] and `sensor_rgbk_norm_rN` (net input) | **no**. `blue_in`/`green_in` exist but are empty leftovers, and `Hue_grid_raw`/`Sat_grid_raw` are gridded *measured hue*, not raw colour. |
| hue / sat after the net | `sensor_hue`, `sensor_sat` | `hue`, `saturation` (timeseries) |
| decoded vectors | `field_uv` | none (decode them yourself) |
| positions | `pose_rN` | `robotN_pose` |
| settings | `run_config` | none |

**The net is identical.** The August 2025 4-robot `run1.mat` saved the net it ran with. On 2,000 random inputs it gives **exactly** the same outputs as today's `trunk/Deep_Learning/july_tuning/everything_nets_final.mat` (saved 2025-07-22; SHA-256 A153C4ED2D06). So 2025 `hue`/`saturation` and 2026 `sensor_hue`/`sensor_sat` are directly comparable. **Raw colours are only needed to try a *different* calibration**, and for 2025 that isn't possible.

## A4. Things that must be handled
1. **Different map placement in each session.**
   - The 2025 vortex was centred near the Motive origin, about (0.001, 0.008). In 2026 it's at (0.839, 0.856), within ±1.5 cm.
   - Fit each session's **centre, rotation and mirror** (see A5) and work in map coordinates. Don't assume Motive coordinates line up.
2. **Is it the same physical map?** Probably the same old vortex mat ("it's old"), but this is **unconfirmed**. If it was re-laid, its rotation can differ (A5 handles that). If it was reprinted, its scale can differ (A5, item 4).
3. **Work with raw hue/sat values, not decoded vectors.**
   - Hue→vector conventions changed over time: the 2026-11-11 decode fix, the saddle map not being mirrored, and the old Simulink decode.
   - A measured hue angle 2πh relates to the designed field angle ψ by `2πh = s·ψ + offset`, with s = ±1 (mirror) and a constant offset (map rotation plus encoding). Fit s and the offset per session; no decode convention is needed.
4. **Magnitude is only relative,** and the map's scale matters.
   - **Printed design** (`Plots_to_print/.../VectorField_plot_creator.ipynb`, cell 11, vortex strips): `u = −(y−0.5)`, `v = x−0.5` on a unit square, hue `h = (atan2(v,u)+π)/2π`, saturation `S = 0.3 + 0.7·m/m_max`.
     - m_max is the corner value, √0.5 map units.
     - So the anticipated S depends on the distance from the centre **in map units**, which needs the printed scale (metres per map unit).
   - **Getting the scale:** either measure the mat (strip width ×3 = side length), or fit one scale factor per session from the saturation-vs-distance slope.
   - The saddle map seems to be physically smaller (the user measured it), so its scale differs.
5. **Don't use the logged `hue_anticipated` / `sat_anticipated` (2025) or `simfield_*` (2026) as "truth".**
   - They come from the in-model analytic simulator: a vortex at (0, 0) with the old conventions.
   - On 2026-11-11 its logging mux also had robot 3's hue and sat swapped (since fixed).
   - **Compute the anticipated field from the design (item 4) with the fitted placement instead.**
6. **The 2025 4-robot data has a crossed sensor mux,** affecting the August runs and the 2025-09-05 scan:
   - `hue` column 3 = robot 3's **saturation**;
   - `saturation` column 2 = robot 3's **hue**;
   - `saturation` column 3 = robot 2's **saturation**.

   **Remap** before use: robot 3 hue = sat col 2, robot 3 sat = hue col 3, robot 2 sat = sat col 3. Or just leave these files out; `sampling_everywhere` (3R) already covers the corners. The 2025 3R files and all 2026 files are correctly paired (checked).
7. **Stale variables** in the 2026 files: use the clean copies in `data_hardware_campaign_11_11_2026_clean/` or `load_run.m` (see `README_STALE_VARIABLES.md`). Never decide the robot count from `pose_r4`.
8. **Per-robot differences.**
   - Accuracy ranking (2026-11): pacific-blue ≈ celeste (about 5°) > tidal (about 5.7°) > **redwood** (about 6.8–7.8°, ~20% low magnitude). Redwood was **never trained on**, deliberately; it is the IDETC unseen-robot case.
   - **Make per-robot maps,** plus a pooled map of the trained robots only.
   - Redwood also seems to have about −7° of Motive heading offset; that only matters if heading is used.
9. **Sensor position vs rigid-body origin.** If the colour sensor isn't at the OptiTrack rigid-body origin, every reading is taken a few cm from `pose`, in a direction that turns with heading. **Unchecked.** Test it by fitting a body-frame (dx, dy) sensor offset per robot as part of A5; it should come out as about 0 if the sensor sits at the origin.
10. **Batteries** shouldn't affect the colour readings (motors only). Still note the runs with low batteries.
11. **The saddle map is separate:** it's printed un-mirrored, at a different scale, centred at about (0.830, 0.798). Treat it as its own map with its own fit, and note that its 2026 coverage is also a disc.

## A5. Suggested pipeline (not implemented)
1. **Collect** every (robot, time) sample: position in Motive, measured hue h, measured sat S, robot ID, session ID.
2. **Fit placement per session** with a robust direction-only fit, like the campaign's true-centre fit (`fit_true_cp.m`): centre (x0, z0), mirror s, and hue offset. Optionally add the sensor offset (A4.9).
3. **Fit the scale per session** from S vs distance, using the design S = 0.3 + 0.7·d/d_max. Or measure the mat.
4. **Map everything** into design coordinates (x, y in [0, 1]) and compute the anticipated hue and sat from the design formulas.
5. **Compute errors:** direction error = wrap(2πh − anticipated angle) in degrees; magnitude error = S − S_anticipated.
6. **Grid** them (2–5 cm cells): median error and count per cell, per robot and pooled, as heatmaps over the printed map. Mark cells with few samples.
7. **Sanity checks:**
   - the 2026 and 2025 maps should agree where they overlap (the inner disc);
   - a session-to-session difference there means a placement or scale fit problem, not the sensor.
8. **Optional:** with the 2026 raw RGBK, re-run any *new* calibration over the 2026 disc and compare it with the net, inside the disc only.

## A6. Files
- **2025 corners:**
  - `trunk/robots_3/vortex_tests/3-robot-tests/sampling_everywhere_vortex.mat` (3R, 2025-09-16, 5,001 steps);
  - `trunk/robots_4/scanning_entire_vortex_mat.mat`, which is the same file as `robots_3/saddle_tests/scanning_entire_vortex_mat.mat` despite the folder name; it's a 4R vortex scan from 2025-09-05 with the crossed mux.
- **2025 runs:** `data_hardware_campaign_07_31_2025/Vortex/...`.
- **Not the vortex map:** `trunk/robots_3/sinking_vortex/lots_of_points_1/2.mat` (2025-09-17/18) are 3R scans of the **sinking vortex** map, a different field.
- **2026:** `data_hardware_campaign_11_11_2026_clean/`.
- **Coverage script used here:** scratchpad `coverage.m` (cell counts in A2).

---

# Part B: the new robot-model parameters (summary)

**Full write-up, including how they were found:** `trunk/campaign_results/2026-11-15_robot_response_model.md`. Scripts: `trunk/campaign_results/scripts/fit_coulomb_delay.m`, `fit_coulomb.m` and `delay_locate.m`.

**Measured model (2026-11 campaign, 550 robot-runs, healthy batteries):**

`v(k) = 0.11·v(k−1) + 0.89·(0.85·u(k−1))`

That is, a **one-step (0.1 s) delay**, almost no lag, about 85% of the commanded speed, no kinetic friction, and saturation at about 0.3 m/s.

| Parameter (dt 0.1 s) | Python `OmniCluster` now | Proposed | Confidence |
|---|---|---|---|
| input delay | none | **1 step** | high: the fit improves 30%, and the delay is between command and motion, not in the controller |
| α (lag) | 0.70 (τ 0.28 s) | **≈ 0.1** (τ ≈ 0.05 s) | high (per robot 0.06–0.17) |
| command gain | 1.0 (no parameter) | **≈ 0.85** | high (per robot 0.85–0.89; battery-dependent, 0.8–0.93 healthy) |
| kinetic friction v_k | 0.025 m/s | **≈ 0** (≤ 0.01) | medium-high (speed/command flat at 0.80–0.86 from 0.10 to 0.25 m/s) |
| static stiction v_s | 0.025 m/s | **unknown**; keep 0.025 for now | none (commands were never below about 0.1 m/s) |
| speed cap | 0.4 m/s | **0.3 m/s** | medium (never above 0.41; median 0.21–0.27 at commands of 0.35–0.45) |
| no-delay fallback | – | α 0.47, gain 0.89, v_k ≈ 0 | free-run 0.037 vs 0.032 with delay, 0.045 for Python now |

**Important correction:** the earlier "momentum ≈ 0.40–0.44" (2026-11-11 fit and first campaign fit) was the lag term **absorbing the one-step delay**. It's superseded in `CLAUDE.md` and `README_hardware.md` §5.7.

**Still to do:**
- a 5-minute low-speed stiction test (steps of 0.01–0.08 m/s from rest);
- if adopted, add a gain and a delay to `Omnibot`, then re-run the Monte Carlo baselines;
- re-predict the measured orbit radii with the new model as the closed-loop check;
- re-run `fit_coulomb_delay.m` when the campaign ends;
- sims must also use the hardware Orbiter gains, k_t 0.15 and k_r 1 (README_hardware §5.4, §5.7).
