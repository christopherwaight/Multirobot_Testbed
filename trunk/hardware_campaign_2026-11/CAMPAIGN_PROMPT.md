# Hardware campaign, November 2026: instructions for the lab-computer agent

Paste this whole file into a Claude Code session on the lab computer, the one that runs MATLAB, Simulink and OptiTrack. It was written on 2026-10-30 from the analysis computer, and every fact in it was checked against the logs and code in this repository.

## 1. Your role

You help Chris prepare and run a new hardware campaign for the vector field paper's major revision (IEEE Systems Journal). Chris runs the robots. You prepare the Simulink model, the logging, the run scripts and the post-run checks.

Ground rules:

- **Chris does not write code.** You make every edit. Before building anything non-trivial, show the change as pseudocode or math in plain English and wait for a yes.
- **Work on copies.** Never modify the original `.slx` files. They are the record of the old runs. Copy to a new name, for example `three_robot_cluster_campaign2026.slx`, and edit the copy.
- **Make only the changes listed in Section 4.** Anything that changes controller behavior beyond that list needs Chris's approval first, because the paper must describe exactly what ran.
- **Do not modernize the MATLAB code** or restructure the models beyond what is needed.
- **Never overwrite a log.** Aborted runs are kept and marked aborted in the manifest.
- **Ask before committing `.mat` files to git.** They are large. `.m` files, this prompt and the manifest can be committed.
- **Use repository-relative paths.** Never hard-code a user's home path.

## 2. The method being tested

A cluster of **three** omnidirectional Decabots on a printed vector-field map (1.6764 m square, OptiTrack positions in metres) runs this loop at 10 Hz:

1. **Sense.** Each robot reads the printed color under it. An MLP maps RGB to hue and saturation, which decode to a field vector (u_i, v_i).
2. **Estimate.** Plane fit `A θ_u = z_u`, `A θ_v = z_v` with `A = [x_i y_i 1]`, then Ĵ = [a b; d e], ĥ = [c; f] and p̂* = −Ĵ⁻¹ĥ. This is the `Criticalpointfinder` block.
3. **Navigate.** The attraction law is ṗ_c = k (p̂* − p_c). The orbit law is ṗ_c = k_t t + k_r (r − r_d) n, with r = ‖p̂* − p_c‖ and n the unit vector from p_c toward p̂*. This is the `Orbiter` block.
4. **Hold formation.** The cluster-space controller turns the centroid command and the shape errors (p, q, β) into robot velocities through the inverse Jacobian (`Forward_Kinematics` and `InverseJabobian` blocks).
5. **Actuate.** Robots respond with a first-order lag (τ ≈ 0.28 s) and Coulomb friction (v_c ≈ 0.025 m/s). Background is in `Paper_Writing/Systems Paper Revision/robot_model_note/robot_model.tex`.

The analysis computer runs `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/revision_metrics/`. It computes, per run:
- final centroid error, bias of the mean rest point, scatter;
- localization error ‖p̂* − p*‖;
- convergence (first-hit and permanent-settle) times at 3, 4 and 5 cm;
- failure;
- commanded and realized control effort;
- signed and absolute radial error with the first 10 s excluded;
- controller-only radial error ‖p_c − p̂*‖ − r_d;
- lap-by-lap orbit statistics.

The logging in Section 5 exists to feed exactly these metrics.

## 3. Why there is a new campaign

The reviewers asked for:
- repeated trials;
- clearly defined convergence, localization, radial-tracking, control-effort and failure metrics;
- matched fixed-radius baselines;
- per-trial error distributions;
- consistent transient exclusion;
- a separation of fixed map distortion from temporal sensor noise.

A data audit on 2026-10-30 found that the old logs cannot support those requests cleanly:

- **Robot count is not recorded anywhere a reader would see it.**
  - In `trunk/robots_3/vortex_tests/`, every top-level log is a four-robot formation, about a 0.24 m square: `run1–81`, `run101–180`, `orbit0XX/1XX/2XX`.
  - The three-robot vortex runs sit in the `3-robot-tests/` subfolder.
  - File names never say which is which. **Every new file name must carry the robot count.**
- **Formation size differs between fields and from the paper.** The saddle runs are about a 0.30 m triangle and the three-robot vortex runs about 0.25 m. The paper and the simulation use 0.33 m.
- **The command was never logged.** The logged `desired_velocities` is a linear function of position (R² 0.98–0.99), probably a scaled field reading, and not the command sent to the cluster. So commanded control effort and the gains cannot be checked from the old logs. The realized motion does match k ≈ 1 s⁻¹.
- **Gains live inside function bodies and are not logged.**
  - The 3R `Orbiter` hard-codes `K_p = 0.2` and `orbit_speed = 0.15`.
  - The 4R model uses `K_p = 0.1`.
  - The paper and simulation use k_r = 0.5 s⁻¹ and k_t = 0.2 m/s.
- **Orbit runs are 600 cycles (60 s) with one run per radius.** The paper said 3000 cycles.
- **No per-cycle wall-clock time is logged.** Chris suspects some cycles ran longer than 0.1 s.
- **Singular estimates are not handled or logged.** `Criticalpointfinder` returns NaN when |det Ĵ| < 1e-10. Up to 39 of 601 orbit estimates landed off the map, and nothing records it.

## 4. Model changes

### 4.1 Cluster kinematics and the "θ + π" fix

**The Python bug.** In `VF_Robot/src/robot/omni_cluster.py` (`OmniCluster.move`), forward kinematics measured the heading θ_c from robot 1 to robot 2. Inverse kinematics, which the inverse Jacobian differentiates, placed the triangle at θ_c + π. Evaluating the Jacobian at the forward-kinematics heading negated the p, β and q shape feedback. Shape errors then grew as e^{0.44 t}, and the formation broke after about 60 s of motion. The fix, dated 2026-10-28, passes `theta_c + pi`. The same fix went into `pentagon_cluster.py`.

**The Simulink 3R model does not have this bug.** In `three_robot_cluster_all_primitives.slx`, both kinematics blocks measure θ_c from the centroid to robot 1, from the +y axis:
- `Forward_Kinematics` computes `theta_c = atan2((2/3)x1 − (1/3)(x2+x3), (2/3)y1 − (1/3)(y2+y3))`.
- `InverseJabobian` places robot 1 at `xc + (r/3) sin θ_c`, `yc + (r/3) cos θ_c`, with `r = sqrt((q + p cos β)² + (p sin β)²)`. Here r/3 is the centroid-to-vertex distance.

**Do not port the Python fix.** Verify instead with a formation-hold test:
1. Start the robots about 10% off the desired side lengths.
2. Command zero centroid velocity for 30 s and log p, q, β.
3. The shape errors must decay. If any grows, stop and tell Chris.

### 4.2 Gains and limits: one set, shared with the simulation

Move every gain out of the function bodies into model-workspace parameters or Constant blocks, so that each is logged (Section 5) and visible in one place.

Ask Chris to choose the values. Recommended, to match the paper and the simulation:

| Parameter | Recommended | Old 3R Simulink | Python simulation |
|---|---|---|---|
| attraction gain k | 1.0 s⁻¹ | about 1 (realized) | 1.0 |
| orbit tangential speed k_t | 0.2 m/s | 0.15 | 0.2 |
| orbit radial gain k_r | 0.5 s⁻¹ | 0.2 | 0.5 |
| centroid command limit | decide: ±0.3 m/s per axis (hardware) or 0.5 m/s magnitude (simulation) | ±0.3 per axis | 0.5 magnitude |
| orbit radial limit | decide: 0.3 m/s on the radial term (simulation) or ±5 m on the radius error (Simulink) | ±5 m on error | 0.3 m/s |
| formation side | decide: 0.33 m (paper and simulation) or 0.30 m (old saddle runs) | 0.25 vortex, 0.30 saddle | 0.33 |

Whatever Chris chooses, use the same values for **both fields and every run**. Record them in Section 5's metadata. The simulation will be rerun to match, so consistency matters more than the particular numbers.

### 4.3 Singular and off-map estimates

`Criticalpointfinder` currently returns NaN when |det Ĵ| < 1e-10, and nothing downstream handles it.

- Add outputs: `a, b, c, d, e, f`, `detJ`, `sigma_min_J` (smallest singular value of Ĵ) and a validity flag `est_valid`.
- When the estimate is invalid, Chris chooses the behavior: hold the last valid estimate (recommended), or command zero centroid velocity. Log the flag either way.
- **Do not add a conditioning gate** (rejecting estimates with small σ_min) unless Chris decides to. The paper states that no gate was applied. If a gate is added, it becomes part of the method and must be described.

### 4.4 Orbiter r = 0 guard

`Orbiter` divides by `current_radius` with no guard.
- If `current_radius < 1e-6`, use a fixed unit vector n = [1, 0]ᵀ. This matches the convention described in the paper's stability section. Confirm with Chris.
- Log `r_guard` (1 when the guard fires).

### 4.5 Which estimate the controller uses

The old four-robot logs carried both `detected_critical_point` and `detected_critical_point_avg`.
- Find out which one feeds the navigation blocks in the 3R copy.
- The paper describes the instantaneous estimate. If an average is used, report the window to Chris and let him decide.
- Log both signals and a constant flag saying which one was used.

### 4.6 Timing

Log, every cycle:
- the wall-clock time;
- the measured period;
- an overrun flag (period > 0.12 s).

The sampling period enters the stability bound (k Δt < 2(1 + α)/(1 − α)), and overruns distort the orbit speeds. The old models had real-time pacing variables (`elapsedRealTime`, `differenceInSeconds`, `mySimTimePerRealTime`). Reuse whatever the pacing block exposes.

### 4.7 Online critical-point type classification (logged only)

Reviewer 1, Comment 7, noted that "critical point classification is not experimentally evaluated" and asked to validate it or qualify the claim. Add a MATLAB Function block, `Classifier`. It takes Ĵ from `Criticalpointfinder` every cycle and outputs:

- `J_trace` = a + e, `J_det` = ae − bd, `J_disc` = trace² − 4 det;
- `cp_type`, an integer code following Table I of the paper:
  - 1 saddle (det < 0);
  - 6 center (det > 0 and |trace| ≤ ε_c √det), checked first;
  - 2 sink and 3 source (det > 0, not center, and disc ≥ 0 or √(−disc) ≤ ε_c √det; trace < 0 or > 0);
  - 4 stable spiral and 5 unstable spiral (det > 0, any other case; trace < 0 or > 0);
  - 0 invalid (est_valid = 0).

Rules for this block:

- **It must not feed the controller.** It only labels what the cluster senses as it moves, so the experiments stay identical to the method in the paper.
- **The band ε_c is Chris's choice.** It is one test on the eigenvalue angle: within asin(ε_c/2) (about 6° for ε_c = 0.2) of the imaginary axis is a center, and of the real axis a node.
  - A center has trace exactly zero, and the canonical sink and source (J = ∓I) have disc exactly zero. Without the band, noise or round-off decides between neighboring classes.
  - The analysis code uses the same rule (`metrics.classify` in `revision_metrics`). Recommend ε_c = 0.2.
- **Make ε_c a logged parameter.** The raw trace and determinant are logged too, so any band can be reapplied offline.

### 4.8 The 4-robot model

- **The Python `QuadCluster`** (`src/robot/quad_cluster.py`) has not changed since commit 485d14d. It uses a diagonal parameterization (d1, d2, r1, r2, φ) and was checked unaffected by the θ + π bug. No upgraded 4-robot definition exists in the repository.
- **The Simulink 4R model** (`trunk/robots_4/chris_4R_all_primitives.slx`) uses the same parameterization, a `Theta_controller`, a four-point least-squares `Criticalpointfinder`, and `K_p = 0.1`, `orbit_speed = 0.15`.
- **The kinematics match the Python code; the control around them does not.**
  - **Kinematics.** The forward and inverse kinematics formulas are identical in Simulink and Python.
  - **Heading.** Simulink rotates the square with a `Theta_controller`, which points diagonal 1–3 along the direction of travel and doubles the wrapped error. Python holds the heading (ω_c = 0).
  - **Estimator.** Simulink fits one least-squares plane through all four robots. Python's default four-robot primitives estimate from two three-robot triangles; only `four_planar_center_finder` and `center_orbiter_quad_planar` match Simulink.
  - **Orbit gains.** Simulink uses k_r 0.1 and k_t 0.15 with no radial limit. Python uses 0.5 and 0.2 with a 0.3 m/s radial limit.
  - **Robot model.** Python's `QuadCluster` still uses the legacy robot model (α 0.7, legacy stiction, 0.3 m/s cap), not the Coulomb model.
  - **Formation.** The hardware ran a square of about 0.24 m side (0.33 m diagonals). The Python configs are `quad_square` (0.30 m diagonals) and `quad_default` (0.433 m by 0.25 m).
  - All of these must be aligned before any four-robot simulation is compared with four-robot hardware.
- **The paper is three-robot only.** If Chris also runs four-robot experiments for other work:
  - use the 4R model copy;
  - put `4R` in every file name;
  - store the data in a separate folder;
  - never mix the two in one folder.

## 5. What to log, and why

Log every signal below at the 10 Hz control rate. All signals must have the same length and share one cycle index. Save each run with `save(fname, ..., '-v7')`, plus a `meta` struct.

### 5.1 Per-cycle signals

| Variable name | Contents (units, frame) | Why it is needed |
|---|---|---|
| `k` | cycle index | alignment |
| `t_sim` | Simulink time (s) | nominal time base |
| `t_wall`, `dt_wall`, `overrun` | wall-clock time (s), measured period (s), flag | true sampling period for the stability bound and orbit speeds |
| `robot{i}_pose` (i = 1..3) | x, y (m), θ (rad), OptiTrack frame | every position metric; robot-level realized effort |
| `mocap_frame`, `robot{i}_tracked` | OptiTrack frame number, tracking-valid flag | detect dropouts and occlusion |
| `cluster_pose` | x_c, y_c (m), θ_c (rad) from forward kinematics | centroid error, convergence, orbit radius |
| `shape`, `shape_des` | p, q (m), β (rad); desired values | formation-error failure criterion; 0.33 m check |
| `rgbk{i}` | raw sensor R, G, B, K per robot | offline replay of the estimator |
| `hue{i}`, `sat{i}` | MLP outputs per robot | separating sensor-model error from map distortion |
| `field{i}` | decoded u_i, v_i per robot | temporal noise σ_v; validating the error bound and covariance |
| `Jhat`, `hhat` | a, b, d, e and c, f | conditioning and the error-bound validation |
| `detJ`, `sigma_min_J`, `est_valid` | determinant, smallest singular value, flag | nearly singular cases (reviewer request) |
| `J_trace`, `J_det`, `J_disc`, `cp_type` | classifier outputs (Section 4.7) | experimental validation of type classification (Reviewer 1, Comment 7): per-cycle accuracy against the map's true type, time to first correct label, and fraction of time correct after 10 s |
| `phat_raw`, `phat_avg`, `phat_used` | estimates (m) | localization error; controller-only radial error |
| `v_nav_pre`, `v_nav_post`, `sat_active` | centroid command before and after the limit (m/s), flag | commanded control effort; saturation share |
| `orbit_r_est`, `orbit_n`, `orbit_t`, `v_tan`, `v_rad`, `r_guard` | orbit internals | separating lag offset from estimation offset |
| `v_cluster_cmd` | full cluster-space command [ẋ_c, ẏ_c, θ̇_c, ṗ, q̇, β̇] | formation effort |
| `robot{i}_cmd` | velocity command sent to each robot (v_x, v_y, ω) | robot-level commanded effort; friction evidence (commands below v_c producing no motion) |
| `mode` | primitive ID (attract, orbit, oracle, hold, sysid) | grouping and checks |
| `comm_ok` or a packet counter | Wi-Fi delivery per robot, if available | explaining dropouts |

### 5.2 Per-run metadata (`meta` struct in the same .mat, plus one manifest row)

- **Identity:** campaign ID, date and time, operator.
- **Run definition:**
  - field (`vortex` or `saddle`);
  - task (`attract`, `orbit`, `attract_oracle`, `orbit_oracle`, `hold`, `sysid`, `noise`);
  - robot count;
  - Decabot IDs in slots 1 to 3;
  - start group and start pose;
  - r_d and rotation direction;
  - repetition number.
- **Controller settings:**
  - all gains and limits from Section 4.2, plus formation gains;
  - estimator settings: which estimate is used, the averaging window, singular handling, the r = 0 rule, and the classifier band ε_c.
- **Software:** model file name, an MD5 checksum of the .slx, MATLAB version, control rate.
- **Calibration:**
  - OptiTrack calibration date and residual;
  - measured map-origin offset and orientation check (Section 6.1).
- **Outcome:** battery voltages if available; completed, or aborted with the reason; free-text notes.

### 5.3 File names and folders

- **Folder:** `trunk/hardware_campaign_2026-11/data/<YYYY-MM-DD>/`.
- **Name pattern:** `<field>_<task>_<N>R_<param>_rep<NN>.mat`. Examples:
  - `vortex_attract_3R_start05_rep07.mat`
  - `saddle_orbit_3R_rd030_rep02.mat`
  - `vortex_orbitoracle_3R_rd010_rep01.mat`
  - `saddle_sysid_3R_robot2_rep01.mat`
- **Manifest:** append one row per run, aborted runs included, to `trunk/hardware_campaign_2026-11/manifest.csv`. Columns are file, field, task, robots, param, rep, start time, duration, outcome and notes.

## 6. Protocol

### 6.1 Every session, before data

1. **OptiTrack calibration.** Record the date and the residual.
2. **Origin check.** Place one robot's marker on the printed critical point and log 10 s. The mean position is the map-origin offset; record it. Earlier data assumed p* = (0, 0) without checking.
3. **Orientation check.** Place a robot on two marked points of the printed map, one on each printed axis, and log each for 5 s. This confirms axis direction and catches any mirroring or rotation; Chris has suspected the frame is turned.
4. **Timing check.** Run 60 s idle and report the mean, maximum and number of overruns of `dt_wall`.
5. **Formation-hold check** (Section 4.1), once per session.

### 6.2 Experiments: three robots, both fields, the same model copy and parameters throughout

| Block | Design | Length per run | Why |
|---|---|---|---|
| A. Attraction | 8 start locations × 10 reps per field (the old corner and edge-midpoint starts, about 0.6 m from p*) | 20 s (200 cycles) | per-trial distributions; convergence and failure metrics. 20 s separates slow settling from parking. |
| B. Orbit | r_d ∈ {0.01, 0.10, 0.20, 0.30, 0.40, 0.50} m × 3 reps (5 if time allows) per field. Start each run at the same point, 0.30 m from p* on a fixed bearing, with one rotation direction. | 180 s (1800 cycles) | repeated trials; at least 8 laps at 0.50 m for lap statistics; 10 s transient excluded |
| C. Oracle baselines | Same as A and B, but the controller uses the measured true origin instead of p̂*. Attraction: 8 starts × 3 reps per field. Orbit: 6 radii × 2 reps per field. | as A, B | matched fixed-radius baseline; splits estimation error from friction and lag |
| D. Robot model ID | Per robot, per session: step commands of 0.01, 0.02, 0.03, 0.05, 0.10, 0.20 and 0.30 m/s, 3 s each, robot alone | about 5 min | confirms τ and the Coulomb threshold v_c behind the stability analysis |
| E. Sensor noise | Formation parked at 5 positions per field, 30 s each | 30 s | temporal noise σ_v per robot, separated from fixed map distortion |

**Randomize the order.** Interleave start groups, radii and fields across the session rather than running 10 repetitions back to back. In the old vortex data, runs 41–50 (one start corner, run consecutively) all parked 59–80 mm from p*. With consecutive runs, a start-location effect cannot be told apart from drift over the session (battery, lighting, slip). Pre-generate a randomized run list and record its seed.

### 6.3 After every run (automatic check script)

Write `check_run.m`. It loads the file just saved and prints:
- the number of cycles and `dt_wall` statistics (mean, max, overruns);
- the maximum formation side error (flag if it exceeds 10%);
- the number of invalid or off-map estimates;
- r_guard counts;
- the final centroid error (attraction) or the mean radius after 10 s (orbit);
- the meta fields.

It flags anything unusual before the next run starts. Missing signals are a stop condition, because a run without its logs cannot be used.

## 7. Matching the simulation

Report the final values of each of these to Chris. The analysis computer will rerun the Python simulation (`experiments/revision_metrics/sim.py`) with them:

- k;
- the command-limit scheme and value;
- k_t and k_r;
- the radial-limit scheme and value;
- the formation side and formation gains;
- the control period;
- the estimate used and the singular-estimate rule;
- the classifier band ε_c;
- the r = 0 rule;
- the start positions;
- the run lengths;
- the transient window;
- the measured τ and v_c.

## 8. Hand-back

1. Commit the edited `.slx` copy, `check_run.m`, any run-list scripts, `manifest.csv` and this prompt. Ask Chris before committing `.mat` files.
2. Write `trunk/hardware_campaign_2026-11/README.md`. It lists:
   - what ran;
   - the parameter values;
   - session dates;
   - aborted runs and their reasons;
   - anything that differs from this plan.
3. On the analysis computer, `revision_metrics/hardware.py` will be updated to read the new file names and variables (Section 5). Keep the variable names exactly as listed in 5.1.
