# Hardware campaign 2026-11-13: orbiting a vortex with 3 and 4 robots

Notes for the paper and the thesis. Written by Claude (AI assistant) during and after the session, from the logged `.mat` files.

**Where the numbers come from:** every number in sections 2–9 is produced by one script, `scripts/campaign_report.m`. Its full output is in `campaign_report_output_2026-11-13.txt`. Anything that comes from another analysis says which script.

**Status:** the campaign is still running. When this was written, the 4-robot r_d = 0.4 m set had only 2 runs and there was no 4-robot r_d = 0.5 m set. Re-run `campaign_report` when more data arrives and update the tables.

> 🏆 **See also: [`ROBOT_REPORT_CARD.md`](ROBOT_REPORT_CARD.md), which robot is best and why.** Short version: pacific-blue is best overall, celeste is second (and the most predictable), then tidal, then redwood, which the net never saw.

---

## 0. Headline results (paper-ready statements)

1. **Orbit tracking:** a 3- or 4-robot cluster that reads the field only through colour sensors locates a printed vortex and orbits it. Across 102 runs, the field fit read the field as a vortex (or a near-vortex spiral) in 99–100% of steps.
2. **Tracking accuracy depends on the orbit radius:**
   - **Smallest orbit:** about 7 cm, whatever is asked below that. This is a dynamics floor (§3).
   - **r_d = 0.1–0.2 m:** within 1–4 cm of the target with 3 robots. r_d = 0.1 m gives 0.130 ± 0.002 m over 10 runs.
   - **r_d ≥ 0.3 m:** the orbit overshoots by 4–10 cm, because the critical-point estimate gets worse as it extrapolates further (§4).
3. **Critical-point estimate:**
   - **Near the centre:** accurate to about 3 cm (median), with no outliers.
   - **Far from the centre:** the error grows to 10–15 cm at 0.4–0.5 m. It also becomes *biased toward the cluster*, so the controller over-estimates how close it is and drifts outward.
   - **Repeatability:** averaged over a run, the estimate lands within 1–6 cm of the true centre.
4. **3 vs 4 robots:**
   - **Random error:** the 4-robot least-squares estimate has fewer outliers at moderate distances (0.20–0.35 m), so redundancy averages out random reading error.
   - **Bias:** it has a *larger* pull toward the cluster beyond about 0.2 m, and averaging cannot remove a bias the robots share.
   - **Net:** the same radius as 3 robots at r_d ≤ 0.1 m; larger overshoot at r_d ≥ 0.2 m (§5).
5. **The net transfers to a robot it never saw.** The colour-sensor net was trained on 3 robots only, and redwood was deliberately held out (IDETC).
   - **Direction:** redwood reads the field at 6.8° median error, against 5.1–5.7° for the trained robots.
   - **Strength:** about 10% low.
   - **Formation:** it still works, and the 4-robot fit is better *with* redwood than without (§8).
6. **Robot response:** the robots follow a velocity command like a first-order lag with "momentum" 0.44 that reaches 0.85 of the commanded speed (R² 0.94–0.96). This matters for the simulation-to-hardware comparison (§7).
7. **Practical lessons:**
   - Batteries visibly fade within about 10 minutes, and the logs show it per robot (§7).
   - In the 4-robot model, the robots must be placed so that 1–3 and 2–4 are the diagonals (§9).

---

## 1. Setup

| Item | Value |
|---|---|
| Field | Printed vortex floor map (hue = direction, saturation = magnitude; `VectorField_plot_creator.ipynb`). Centre in Motive (x, z) ≈ **(0.839, 0.856)** for this placement (§6). |
| Robots | Decabots: celeste (r1), tidal (r2), pacific-blue (r3), redwood (r4, 4-robot runs only). OptiTrack Motive pose, 10 Hz control. |
| Sensor net | One shared net `trunk/Deep_Learning/july_tuning/everything_nets_final.mat`, trained on celeste, tidal and pacific-blue. **redwood never trained** (by design). |
| 3-robot model | `trunk/robots_3/three_robot_cluster_all_primitives.slx`. All 3-robot runs: file SHA 90e6b8d33b89 (unsaved-edit flag set, but no parameter differed from the file in the runs checked). |
| 4-robot model | `trunk/robots_4/four_robot_cluster_all_primitives.slx`. All 4-robot runs: file SHA 574b25c01ce3 (user-saved after the φ-wrap fix and Orbiter gain sync). |
| Critical-point estimate | 3 robots: exact linear (affine) fit through 3 readings. 4 robots: least-squares affine fit through 4 readings (`Jacobian` primitive; confirmed in §9). |
| Orbiter (both) | v = k_t·r̂⊥ + k_r(r − r_d)·r̂, with **k_t = 0.15 m/s** and **k_r = 1 s⁻¹**, radial speed clipped to ±0.3 m/s, direction +1. r̂ is the unit vector to the estimated centre. No θc control (`thC_gain` = 0). |
| Formation setpoints | 3 robots: p = q = 0.30 m, β = 60° (equilateral, side 0.30 m). 4 robots: d1 = d2 = 0.30 m diagonals, r1 = r2 = 0.5, φ = +90° (square, side 0.21 m). |
| Conditions | r_d = 0.01, 0.1, 0.2, 0.3, 0.4, 0.5 m (3 robots); 0.01, 0.1, 0.2, 0.3, 0.4 m (4 robots, 0.4 in progress). 10 runs each. |
| Run length | 3 robots: 30 s (a few 25 s). 4 robots: 20–30 s (see §2). StopTime was 51 s, but runs were stopped at 20–30 s. |
| Data | `data_hardware_campaign_11_11_2026/` (sub-folders per condition: `vortex_001`, `vortex010`, … for 3 robots; `robot4_vortex_001`, … for 4 robots). Files are named `vortex_<N>robots_radius_of_0_point_<d>_run<k>.mat`. Load with `trunk/robots_3/load_run.m`. Column meanings are in `run_config.log_legend`. |

**Logging:** this was the first campaign with the full logging suite (`README_hardware.md` §6). Every file passed these checks:
- every new variable is present, N rows = `numel(tout)`, and there are no NaNs;
- timing is 0.100 s per step, i.e. real time;
- `run_config` records the model SHA, the gains, r_d and the switches;
- every file name matches the r_d in its `run_config`.

---

## 2. Data inventory and exclusions

| Condition | Runs | Weak-battery runs | Duration (s) | Time |
|---|---|---|---|---|
| 3R r_d 0.01 | 10 | 0 | 30 | 20:37–20:46 |
| 3R r_d 0.10 | 10 | 0 | 30 | 20:51–21:00 |
| 3R r_d 0.20 | 10 | 0 | 25/30 | 21:01–21:09 |
| 3R r_d 0.30 | 10 | 0 | 30 | 21:12–21:27 |
| 3R r_d 0.40 | 10 | 0 | 30 | 21:31–21:40 |
| 3R r_d 0.50 | 10 | 0 | 30 | 21:42–21:52 |
| 4R r_d 0.01 | 10 | **5** (runs 3–7) | 20/30 | 22:02–22:15 |
| 4R r_d 0.10 | 10 | 0 | 20 | 22:16–22:22 |
| 4R r_d 0.20 | 10 | 0 | 25 | 22:25–22:32 |
| 4R r_d 0.30 | 10 | 0 | 28 | 22:33–22:44 |
| 4R r_d 0.40 | 2 (in progress) | 0 | 30 | 22:46–22:49 |

- **Weak-battery runs are excluded** from the condition statistics, the plant fit and the estimator tables. "Weak" means some robot reached < 0.7 of its commanded speed. All five are redwood, in `vortex_4robots_radius_of_0_point_01_run3..7`, with redwood at 0.69, 0.61, 0.42, 0.38 and 0.34 (§7).
- **Earlier diagnostic runs are not part of the campaign:**
  - **`variables1.mat` (21:57):** a 5 s 4-robot test with the robots in the wrong order (§9).
  - **The 2026-11-11 runs:** a different map placement and Orbiter gains; used only for the first plant fit.
- **Duplicate copy:** `vortex_4robots_radius_of_0_point_01_run1.mat` is also in `vortex_001/`, a 3-robot folder. It's identical, and the report reads it once.
- **Stale variables:** the files were saved from the whole workspace, so they also contain old-name variables from earlier sessions with different row counts (`cluster_pose`, `robot1_pose`, `p`, `q`, …). Ignore them. `load_run.m` replaces them with the new logs and keeps the old copies as `*_STALE`.
- **Durations:** at r_d ≥ 0.3 m a lap takes 15–25 s, so 20–30 s runs give only about 1–1.5 laps. Those radii are not fully steady (§4.3).

---

## 3. Orbit radius vs requested radius

The radius below is measured from the cluster centroid to the true centre (§6), averaged over t > 10 s, for healthy runs. "± sd" is across runs.

| r_d (m) | 3R radius (m) | 3R r − r_d | 4R radius (m) | 4R r − r_d | 3R speed (m/s) | 4R speed (m/s) |
|---|---|---|---|---|---|---|
| 0.01 | 0.071 ± 0.013 | +0.061 | 0.072 ± 0.015 (n 5) | +0.062 | 0.147 | 0.138 |
| 0.10 | **0.130 ± 0.002** | +0.030 | 0.129 ± 0.007 | +0.029 | 0.130 | 0.128 |
| 0.20 | 0.208 ± 0.015 | **+0.008** | 0.237 ± 0.030 | +0.037 | 0.131 | 0.130 |
| 0.30 | 0.340 ± 0.029 | +0.040 | 0.396 ± 0.014 | +0.096 | 0.137 | 0.134 |
| 0.40 | 0.477 ± 0.026 | +0.077 | 0.506 ± 0.013 (n 2) | +0.106 | 0.140 | 0.132 |
| 0.50 | 0.518 ± 0.018 | +0.018 | – | – | 0.151 | – |

Other columns in the report output, section C: last-5-s radius, wobble, what the controller believed (`orbit_radius`), estimate jitter and outliers, laps, and shape error.

- **The formation held its shape in every condition:** the RMS error of p, q (3 robots) or d1, d2 (4 robots) was 0.5–1.6 cm against 0.30 m (2.4 cm for the two 4R r_d 0.4 runs).
- **Speed:** the centroid moved at 0.13–0.15 m/s. The command is |v| = √(k_t² + radial²); the radial term (k_r = 1) adds noticeably when the cluster is off the circle, and the robots reach about 0.85 of the command (§7).

### 3.1 Small radii: a dynamics floor of about 7 cm
- **The cause:** with a 10 Hz loop and robots that lag their command (momentum about 0.4), a cluster moving tangentially at k_t cannot hold an orbit smaller than about v·dt/√(1 − (1 − k_r·dt)²) plus lag terms. So r_d = 0.01 and 0.1 give 0.071 and 0.130 m.
- **The closed-loop prediction is short:** with the fitted plant (§7) and the measured estimate jitter, it gives 0.045–0.050 m at r_d 0.01 and 0.116 m at r_d 0.1 (`campaign_sanity.m`). The measured orbits are about 1–2 cm larger.
- **The extra 1–2 cm:** the live estimate runs **1.5–2 cm ahead of the cluster along its direction of motion**. Adding that offset to the simulation reproduces the measurements: 0.068 vs 0.069 and 0.134 vs 0.129 (`campaign_lag.m`).
- **Not sensor delay:** shifting the readings 0–4 steps relative to the poses fits best at 0 steps (`campaign_lag.m`). The cause of the lead is not identified.
- **What the controller sees at r_d 0.01** (§4.2): the distance to the estimate looks *longer* than the true distance in 70% of steps (0.081 vs 0.071 m). Zero-mean scatter σ around the true centre inflates distance to √(r² + 2σ²) = 0.082 (`cp_distance_bias.m`). This alone would pull the orbit *in*, and the lead pushes it out by more.
- **For the paper:** small-radius tracking is limited by the loop rate and the robot lag, not by sensing. A faster loop or a lower k_t would shrink the floor; a comparison of k_t is in `README_hardware.md` §5.4.

### 3.2 Large radii: overshoot from estimate bias
At r_d ≥ 0.3 m the orbit settles 4–10 cm outside the requested circle.
- **The pattern:** the controller's own radius estimate (`orbit_radius`, measured to the live estimate) stays close to r_d (3R: 0.322 at r_d 0.3, 0.422 at 0.4). The true radius is larger. So the controller thinks it is on the circle while it is outside it.
- **Why:** the estimate sits between the cluster and the true centre (§4.2). The controller underestimates its distance and drifts out, which puts the robots farther from the centre, where the bias is larger. This feedback loop amplifies small estimator differences.
- **An odd point:** 3R r_d 0.5 (0.518 m, +0.018) overshoots *less* than r_d 0.4 (+0.077). Possible reasons:
  - **Not steady:** with ≤ 1.2 laps per run, the orbit hasn't settled.
  - **Map edge:** at 0.5 m plus the formation spread, the robots may reach the edge of the printed map, where the field model no longer holds. Check the map's printed extent before interpreting this.

---

## 4. The critical-point estimate

### 4.1 Accuracy vs distance from the centre
This is the estimate the controller used, against the true centre, binned by how far the cluster was from the centre. Healthy runs, t > 5 s; report section D.

| Cluster distance (m) | 3R median error | 3R > 10 cm | 3R radial bias | 4R median error | 4R > 10 cm | 4R radial bias |
|---|---|---|---|---|---|---|
| 0.00–0.10 | 0.035 | 0.0% | +0.009 | 0.026 | 0.7% | −0.002 |
| 0.10–0.15 | 0.030 | 2.7% | −0.009 | 0.030 | 2.3% | −0.007 |
| 0.15–0.20 | 0.040 | 9.4% | −0.012 | 0.045 | 8.5% | −0.006 |
| 0.20–0.25 | 0.050 | 15.1% | −0.012 | 0.054 | **5.8%** | **−0.028** |
| 0.25–0.30 | 0.069 | 22.5% | −0.034 | 0.068 | **16.9%** | **−0.046** |
| 0.30–0.35 | 0.080 | 34.0% | −0.035 | 0.069 | **23.2%** | **−0.047** |
| 0.35–0.40 | 0.099 | 49.0% | −0.052 | 0.102 | 52.0% | −0.069 |
| 0.40–0.45 | 0.112 | 58.1% | −0.048 | 0.146 | 74.7% | −0.112 |
| 0.45–0.55 | 0.143 | 76.3% | −0.099 | 0.177 | 83.8% | −0.141 |
| 0.55–0.70 | 0.162 | 80.0% | −0.125 | 0.299 (few) | 89.6% | −0.258 |

Radial bias is the component of the estimate error along the line from the cluster to the true centre. **Negative means the estimate lies on the cluster's side, i.e. "too close".**

### 4.2 Why the estimate degrades with distance: interpolation vs extrapolation
- **Near the centre the fit interpolates.** The robots straddle the zero of the field, so the fit lands inside the formation. The error floor is about (direction noise ≈ 7°) × (robot spacing ≈ 0.17 m) ≈ 2–3 cm.
- **Far from the centre the fit extrapolates.** The zero lies outside the formation, and two things grow with distance d:
  - in a vortex the vector length grows with r, so a fixed direction error is a larger *vector* error (∝ d);
  - an error in the fitted slope, measured across a 0.2–0.3 m formation, is multiplied by d when projected back to the centre.

  The error therefore grows roughly as (direction error) × d × (1 + d / formation size).
- **A noise model reproduces the trend.** It uses the measured reading noise (direction 6.9° median, 17° p90; strength noise 5 cm-equivalent) on an ideal vortex with the same formation and fit, and gives about 4 → 8 cm from the centre out to 0.3 m. The data show 2.4 → 7.0 cm (`cp_error_vs_distance.m`).
- **The sensors are not worse far out:** reading accuracy is the same everywhere, and the geometry magnifies it.
- **Outliers:** at r_d 0.2 with 3 robots, 11% of steps were off by more than 10 cm, and a few were metres off. A nearly flat fitted field sends the zero far away.
- **The radial bias** has two parts with opposite signs:
  - **Small distances (≲ 0.1 m): the estimate looks *farther* away.** Zero-mean scatter makes the distance √(r² + 2σ²) on average, pure geometry (`cp_distance_bias.m`).
  - **Large distances (≳ 0.2 m): the estimate sits *closer* (toward the cluster).** This is systematic, not random. Candidates:
    - the strength offset all robots share (they read |field| ≈ 0.858·r − 0.025, so every reading is slightly weak, which tilts a linear fit toward the formation);
    - the noisy slope of the fit tending to look steeper;
    - direction-dependent reading error.

    This is not proven.
- **This is kept raw on purpose.** No gate or filter is applied to the estimate; possible fixes are documented in `README_hardware.md` §5.8.

### 4.3 Steadiness at large radii
At r_d ≥ 0.3 m one lap takes 15–25 s (0.38–0.62 laps per 10 s), so a 20–30 s run is 1–1.5 laps.
- **Example, 3R r_d 0.4:** the 10-s window means moved between 0.45 and 0.54 m within single runs (`r04_detail.m`, early runs).
- **Example, 4R r_d 0.4:** the last-5-s radius is 0.565 m, against a 0.506 m average, so it was still growing.
- **Recommendation:** for steady-state numbers at r_d ≥ 0.3 m, use 60–90 s runs (3–4 laps) and measure after the first lap.

---

## 5. 3 robots vs 4 robots

Measured with the same Orbiter, gains, map placement and centre (the 3R and 4R sessions agree on the centre to 0.2 cm, §6).
- **Near the centre (r_d ≤ 0.1 m): no difference in radius.** 0.071 vs 0.072; 0.130 vs 0.129. Here the radius is set by the Orbiter dynamics (§3.1), which are identical.
- **Averaging does reduce random error.**
  - **Fewer outliers at moderate distances:** at the same distance from the centre, 4 robots have fewer steps off by more than 10 cm (5.8% vs 15.1% at 0.20–0.25 m; 16.9% vs 22.5%; 23.2% vs 34.0%), with about the same median error (§4.1).
  - **The first 4R r_d 0.2 runs** looked better than 3R (0.209 m, 4.8% outliers, less wobble), but over all 10 runs 4R r_d 0.2 came out at 0.237 m.
- **But 4 robots carry a larger shared pull toward the cluster beyond about 0.2 m:** −2.8 vs −1.2 cm at 0.20–0.25 m, −4.7 vs −3.5 cm at 0.30–0.35 m, −11.2 vs −4.8 cm at 0.40–0.45 m. Least squares averages out random error but not a bias the robots share. Through the feedback in §3.2, this gives the larger overshoot at r_d ≥ 0.2 m (0.237 vs 0.208; 0.396 vs 0.340).
- **What it is NOT** (tested):
  - **Not redwood.** Leaving any one robot out of the 4-robot fit makes the estimate *worse* (`why_4R_worse.m`). At 4R r_d 0.3 (the first 6 runs), the median error is 0.125 m with all four and 0.133–0.186 m without one. Leaving redwood out makes the pull toward the cluster larger (−0.125 vs −0.077 m).
  - **Not mainly redwood's calibration.** Correcting every robot's measured angle bias and strength scale (redwood −8°, 0.70) before the fit reduces the 4-robot pull only from −8.9 to −7.0 cm at 0.35–0.45 m, and changes nothing for 3 robots (`calib_fix_test.m`). Note: those per-robot corrections came from an earlier pass over fewer runs; see §8 for the final values.
  - **Not formation geometry under random noise.** A square with 0.30 m diagonals (sides 0.21 m) and a triangle with 0.30 m sides give the same error with the same reading noise: median 0.091 vs 0.089 m at 0.3 m (`why_4R_worse.m`, noise model).
- **Open:** why the 4-robot least-squares fit is pulled harder toward the cluster than the 3-robot exact fit. Next checks:
  1. A noise model that includes the shared strength offset and direction-dependent error, for both formations.
  2. The same formation size for both (e.g. a 4-robot square with 0.42 m diagonals, i.e. 0.30 m sides).
  3. Longer runs.

---

## 6. Where the true centre is (and the map didn't move)

The vortex centre was fitted from all readings (direction only, ignoring saturation; a constant angle bias per robot; over 90,000 readings from 102 runs). Report section B; method checks in `fit_true_cp.m` and `cp_drift_vs_radius.m`.
- **Centre:** **(0.8390, 0.8556) m** in Motive (x, z).
  - **3R runs only:** (0.8376, 0.8546).
  - **4R runs only:** (0.8398, 0.8563).
  - **So the map did not move between sessions** (0.2 cm apart).
- **Statistical error:** about 1 mm (102 per-run fits, std 1.0 cm).
- **Systematic error: about ±1.5 cm.**
  - **The symptom:** different conditions (different radii) give centres up to 2–3 cm apart, and inner vs outer readings of the *same* runs differ by 1–3 cm.
  - **The cause:** direction-dependent reading error (up to about 10–15° at some field directions), which weights differently depending on where the robots sample.
  - **Agreement with other methods:** a linear-field fit, a magnitude-only fit, and a fit with per-robot sensor offsets all agree within about 3 cm.
- **Live estimator vs true centre:** run-median estimates land 1–6 cm from the true centre (report section C, "est offset").
- **For thesis-grade ground truth,** measure the printed centre with an OptiTrack marker at the start and end of each session. A data-derived centre cannot beat the ±1.5 cm systematic limit, and neither would an RBF or NN fit, because it would learn the same sensor errors.

---

## 7. How the robots respond, and batteries

### 7.1 Robot response (plant) fit
The centroid velocity was fitted against the Orbiter's centroid command (report section E):

| Data | a (momentum) | b | Speed reached b/(1 − a) | R² |
|---|---|---|---|---|
| 2026-11-11 (3 runs, older map placement and gains) | 0.398 | 0.471 | 0.78 | 0.98 |
| 2026-11-13, 3R healthy runs (60 runs) | 0.435 | 0.482 | **0.85** | 0.94 |
| 2026-11-13, 4R healthy runs (37 runs) | 0.440 | 0.474 | **0.85** | 0.96 |

- **3 and 4 robots respond identically:** momentum about 0.44, speed reached about 0.85.
- **The Python simulator assumes otherwise:** `omnibot.py` uses momentum 0.7 and speed reached 1.0. **For sim-vs-hardware comparisons, re-run the Python sims with momentum ≈ 0.44, gain ≈ 0.85, and Orbiter k_t 0.15, k_r 1** (`README_hardware.md` §5.7). That was planned for after the campaign: re-fit with all of its data first.
- **Per robot** (speed reached, median of runs [range]; direction of motion vs command):
  - celeste 0.82 [0.78–0.89], −0.8°
  - tidal 0.85 [0.79–0.88], −2.5°
  - pacific-blue 0.91 [0.85–0.96], −0.5°
  - redwood 0.85 [0.76–0.94], **−7.1°**
- **Redwood's constant −7° direction offset** didn't change with new batteries, which points to its Motive heading zero being about 5° off (`README_hardware.md` §5.1). Re-zero it before more 4-robot runs.

### 7.2 Battery fade
Redwood's speed reached in successive 4R r_d 0.01 runs (22:02 → 22:11) was 0.81, 0.76, 0.69, 0.61, 0.42, 0.38, 0.34. With new batteries (run 8, 22:14) it was **0.92**. The other robots stayed at 0.8–0.93.
- **Shape effect:** the diagonal containing redwood (d2) stretched from 0.31 to 0.34 m, then returned to 0.298 m.
- **Lesson:** the start-of-day voltage check doesn't catch fade during a session. The logs give a per-robot battery gauge, median |actual| / |commanded| velocity: healthy 0.8–0.93, swap below about 0.7 (`README_hardware.md` §5.9, `robot_health_4R.m`).

---

## 8. Sensor accuracy per robot, and the unseen robot

Readings were compared with the ideal vortex about the true centre, using all runs and only readings more than 5 cm from the centre (report section F).

| Robot | Trained on? | Median direction error | p90 | Constant bias | Direction-dependent (peak-to-peak) | Strength slope (shared 0.858) | Strength RMS |
|---|---|---|---|---|---|---|---|
| pacific-blue | yes | **5.1°** | 12.2° | −3.0° | 5.6° | 0.87 | 0.051 |
| celeste | yes | 5.2° | 10.9° | −4.8° | 8.3° | 0.89 | 0.052 |
| tidal | yes | 5.7° | 14.5° | −5.4° | 9.4° | 0.86 | 0.075 |
| **redwood** | **no** | **6.8°** | 14.5° | **−7.2°** | 9.9° | **0.77** | 0.072 |

- **Ranking (best to worst): pacific-blue ≈ celeste > tidal > redwood.** The three trained robots are within 0.6° of each other. Tidal has the noisiest strength reading.
- **Redwood generalizes.** It is the only robot the net never saw, and it is about 1.5° worse in median direction. It has the largest constant bias (−7°) and reads strength about 10% low (slope 0.77 vs 0.86–0.89).
  - **On the calibration bench:** redwood's held-out file has a p90 hue error of 30.7°, against 5–12° for the trained robots (`net_trained_on.m`).
  - **In the formation:** the 4-robot fit is still better with redwood than without it (§5).
- **Sensing and driving are separate:** redwood's sensing bias (−7°, hue is an absolute colour) is unrelated to its motion offset (−7°, the Motive zero). Two independent issues happen to have similar size.
- **Training provenance:**
  - The January `comprehensive_search.m` trained on 3 robots.
  - The July script *currently on disk* lists `redwood_cal` in training, but that is **not** the version that produced the net (confirmed by the user).
  - The net file is dated 2025-07-22 09:26; the script was edited at 09:31.
- **How these numbers changed:** an early pass over the first 84 runs gave pacific-blue 4.9°, celeste 5.2°, tidal 5.7° and redwood 7.8°. With all 102 runs the ranking is unchanged, and redwood's gap is smaller (6.8°).

---

## 9. 4-robot specifics

- **Which estimate was used:** all `vortex_4robots_*` runs used the `Jacobian` primitive, a 4-point least-squares fit (`Manual Switch1` = 1 → `Mux1`).
  - Recomputing the 4-point fit from the logged poses and readings reproduces `field_cp_jacobian` to 1e-15 m.
  - The robot commands equal `cmd_robots_jacobian` exactly.
  - The Orbiter command is explained by that estimate with R² = 1.0000 (`confirm_4R_ls.m`).
  - The averaging primitive (`Jacobians`, two 3-robot fits) was not used.
- **Robot order:** the kinematics need 1–3 and 2–4 to be the diagonals. Going around the square: celeste → tidal → pacific-blue → redwood. Numbered clockwise in Motive x–z, that gives φ = +90°, matching the setpoint.
  - **The first test:** the robots were placed redwood → pacific-blue → celeste → tidal. The model's "diagonals" were sides, so r1 = −4.8 and r2 = −2.8, and the commands ran 1–5 m/s (`check_4R_run.m`).
  - **Fix:** swap celeste and tidal (`README_hardware.md` §5.10).
- **φ wrapping:** the angle between the diagonals and its error are wrapped to [−π, π] (Phase 2 of `trunk/robots_4/README_four_robot.md`). In the campaign, φ stayed at 1.51–1.59 rad.

---

## 9b. Failed run: 4R r_d 0.5 (`Robot4_vortex_050_runs_off_the_mat.mat`, 23:04, aborted at 22.8 s)

This is the only 4R r_d 0.5 attempt; it is not in the campaign tables. Reconstructed from the logs:
1. **0–12 s: a normal large orbit.**
   - The centroid was 0.37–0.61 m from the centre and the shape held (d1, d2 about 0.30).
   - **The outer robots were already 0.65–0.75 m out,** which is r_d plus the 0.15 m formation spread plus overshoot.
   - **The estimate jumped:** at 4 s and 13 s the controller's distance to it read 0.85 m, against a true 0.47–0.56 m. Each jump gave a burst of 0.32–0.37 m/s commands.
2. **13–14 s: pacific-blue stalled for about 1.5 s.** It was commanded at 0.34 m/s but moved at 0.01–0.02 m/s. The square distorted (d1 0.42, r2 0.63), then recovered.
3. **18 s: another estimate jump.** The controller read 0.99 m and the cluster was pushed out to 0.745 m. Celeste and tidal reached **0.83–0.87 m** from the centre.
4. **19–22.8 s: celeste stopped dead at about 0.79 m from the centre.** It was commanded at 0.23–0.45 m/s and moved at about 0 m/s; tidal also stalled briefly.
   - **The formation tore:** d1 0.58, d2 0.61, φ 0.88 rad, against setpoints 0.30 / 0.30 / 1.57.
   - The run was then stopped.
- **Interpretation:** the robots stalled at about 0.8 m from the centre, most likely at the **edge of the printed mat**, with wheels catching or slipping on the edge.
  - **Sensor readings at that point** still looked like the map (saturation about 0.70, normal clear-channel counts), so the sensors were still over printed area. The sensor sits away from the wheels.
  - **Contributing:**
    - at r_d 0.5 the 4R formation puts its outer robots at about 0.65 m even on a perfect orbit;
    - the large-distance estimate jumps (§4) push the cluster further out;
    - one robot stalling breaks the formation.
- **Usable workspace:** treat r ≲ 0.65–0.7 m from the centre as the limit for any robot on this map placement. So r_d + formation spread + about 10 cm of overshoot must stay below that. For the 4R square that means r_d ≤ 0.4; for the 3R triangle, r_d 0.5 is marginal (see the odd 3R r_d 0.5 result in §3.2).

## 10. Open questions and next steps

1. **True centre:** measure the printed centre with a marker. That will settle the ±1.5 cm systematic uncertainty and give the estimator accuracy against ground truth.
2. **Longer runs at r_d ≥ 0.3 m** (60–90 s), to get steady-state radii (§4.3).
3. **Map extent at r_d 0.5:** check that the robots stayed on the printed map (§3.2).
4. **Why 4 robots are pulled harder toward the cluster** (§5): run the noise model with the shared strength offset, and test a 4-robot square with 0.30 m sides.
5. **Lead of the estimate along the motion** (§3.1): the cause is unidentified; sensor delay has been ruled out.
6. **After the campaign:**
   - re-fit the robot response per robot, excluding weak-battery runs;
   - re-run the Python sims with momentum ≈ 0.44, gain ≈ 0.85, k_t 0.15, k_r 1;
   - compare the sims with these tables.
7. **Housekeeping:**
   - re-zero redwood's Motive heading;
   - set `run_notes` before runs (it was empty in every run);
   - `clear` the workspace or save only the logs, to avoid stale variables;
   - keep 4-robot files out of the 3-robot folders.

---

## 11. Suggested figures

1. **Measured radius vs r_d** for 3R and 4R, with error bars (sd across runs), the y = x line, and the predicted dynamics floor (§3).
2. **Estimate error and radial bias vs cluster distance from the centre** for 3R and 4R (§4.1). This is the key figure for the interpolation vs extrapolation argument.
3. **One example trajectory per condition:** the centroid path, the true centre, and the live estimate cloud. One small and one large r_d show the contrast.
4. **Per-robot reading error vs field direction** (direction-dependent error), with redwood as the unseen robot (§8).
5. **Redwood's battery fade:** speed reached vs run number, with the battery swap marked (§7.2).
6. **The robot response fit:** measured vs predicted centroid velocity, R² 0.94–0.96 (§7.1).

## 12. Reproduce

```matlab
addpath('trunk/campaign_results/scripts'); addpath('trunk/robots_3');
T = campaign_report('data_hardware_campaign_11_11_2026');   % all tables in this document
S = load_run('data_hardware_campaign_11_11_2026/vortex040/vortex_3robots_radius_of_0_point_4_run1.mat');  % one run
```
- **Script index:** `scripts/README.md` describes each script. Most take a folder of run files and the centre `[0.839 0.856]` as arguments.
- **Snapshot used:** the numbers here come from a snapshot taken at about 22:52 (102 runs). The SHA-256 manifest of that snapshot was in the session scratchpad and is not kept. The run files in `data_hardware_campaign_11_11_2026/` are the originals.

## Glossary
- **Radius:** distance from the cluster centroid (`cluster_state(:,1:2)`) to the true centre, mean over t > 10 s.
- **Wobble:** standard deviation of that radius within one run.
- **Estimate / live estimate:** the critical-point estimate the controller used at each step (`field_cp` for 3 robots, `field_cp_jacobian` for 4 robots).
- **Estimate jitter:** 1.4826 × the median distance of the live estimate from its run median (a robust standard deviation).
- **Estimate offset:** distance from the run-median estimate to the true centre.
- **Radial bias:** component of (estimate − true centre) along the line from the cluster to the true centre. Negative means toward the cluster.
- **Speed reached (battery gauge):** a robot's median |actual velocity| / |commanded velocity|, using `pose_rN` and `cmd_robots_selected`.
- **Plant:** centroid v(k) = a·v(k−1) + b·u(k), with u the commanded centroid velocity. Speed reached = b/(1 − a).
