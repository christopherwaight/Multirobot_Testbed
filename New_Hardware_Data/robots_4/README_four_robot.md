# 4-Robot Model: `four_robot_cluster_all_primitives.slx`

**Built by Claude (AI assistant) on 2026-11-12 at the user's request.**
- **Source:** made from `chris_4R_all_primitives.slx`, the existing 4-robot model, which is left **unchanged**.
- **What was done:** the changes made to the 3-robot model `trunk/robots_3/three_robot_cluster_all_primitives.slx` on 2026-11-11/12 were ported over and extended to 4 robots.
- **Where the tag is:** inside the model, as a yellow note at the top of the root diagram and in the model's Description.
- **Rebuild:** `tools/build_four_robot_model.m`.

Read `trunk/robots_3/README_hardware.md` first. It explains the floor maps, the hue decoding, the logging and the lessons learned; all of that applies here too.

---

## Phase 1: what the 4-robot model now has

Robots: **r1 = celeste, r2 = tidal, r3 = pacific-blue, r4 = redwood** (OptiTrack `rigidBodyNames` order; also stored in every run as `run_config.robots`).

| Change (same as the 3-robot model) | Where |
|---|---|
| Hue → vector decode fix: v = −m·cos(2πh), via a Gain `z_sign` = −1 | all 8 decoders: `Jacobian` and `Jacobians` → `Critical_point_predicter/Subsystem…Subsystem3` |
| Orbiter: Python law structure, **`k_t` 0.15, `k_r` 1** (the user's gains, 2026-11-12), radial speed clip ±0.3, direction flag +1. **Identical code to the 3-robot Orbiter**: one source file, `trunk/robots_3/tools/orbiter_python_match.m`, pushed in by `trunk/robots_3/tools/sync_orbiter_code.m`. Backup before the gain change: `four_robot_cluster_all_primitives_BACKUP_pre_orbiter_gains_20261112.slx`. | `Jacobian/MATLAB Function`, `Jacobians/MATLAB Function` (+ their `Constant`) |
| No θc spin, like Python: Gain `thC_gain` = 0 before the inverse Jacobian (set it to 1 to restore) | `Cluster_Controller_setpoint`, `Jacobian`, `Jacobians` |
| Logging suite (cyan `LOG` blocks, Array format) + `run_config` via StopFcn | everywhere; see the table below |

Not changed in Phase 1: the kinematics (see Phase 2, φ wrapping only), the desired orbit radii (`Jacobian/Constant1` = 0.7, `Jacobians/Constant1` = 0.6), the switch positions, the setpoints and the neural nets.

**Primitives in this model.** The 4-robot model never had the 3-robot model's `vector_sum`, `steepest_ascent`, `steepest_descent` or the analytic-field simulator, so they aren't here. It has:
- **`Cluster_Controller_setpoint`:** holds a quadrilateral at a set point.
- **`Jacobian`:** 4-point least-squares critical-point fit, then orbit.
- **`Jacobians`:** averages two 3-robot fits (robots 1-2-3 and 1-3-4), then orbits. It also contains a `Theta_controller` that turns the cluster toward the critical point, which `thC_gain` = 0 now switches off.

The root switches currently select **`Jacobians`**: `Manual Switch` sw=0 → `Manual Switch1` sw=0 → `Mux2`.

### Logged variables (4 robots)
| Variable | Cols | What it is |
|---|---|---|
| `pose_r1..r4` | 3 | OptiTrack [x z heading] |
| `sensor_rgbk_raw_r1..r4` | 4 | raw Decabot counts, in the Decabot's output order **[K(clear) R G B]** |
| `sensor_rgbk_norm_r1..r4` | 4 | normalised [R G B K], the neural-net input |
| `sensor_hue`, `sensor_sat` | 4 | per robot, 0..1 |
| `field_uv` | 8 | decoded [u1 v1 … u4 v4] |
| `field_jacobian` | 8 | 4-point least-squares fit [J11 J12 J21 J22 trace det curl type]; type 1 vortex, 2 spiral, 3 node, −1 saddle, 0 degenerate |
| `field_cp_jacobian`, `field_cp_jacobians` | 2 | each primitive's critical point |
| `orbit_radius_jacobian`, `orbit_radius_jacobians` | 2 | [radius, desired − radius] |
| `cluster_state` | 8 | [xc zc θc d1 d2 r1 r2 φ] |
| `setpoint` | 8 | set point controller's desired [xc zc θc d1 d2 r1 r2 φ] |
| `err_setpoint`, `err_jacobian`, `err_jacobians` | 8 | errors [xc zc θc d1 d2 r1 r2 φ] |
| `cmd_cluster_<prim>` | 8 | what enters the inverse Jacobian [v_xc v_zc ω_c v_d1 v_d2 v_r1 v_r2 v_φ] |
| `cmd_robots_<prim>`, `cmd_robots_selected` | 12 | per-robot global velocities [vx vz ω] × 4 |
| `cmd_body_r1..r4` | 3 | body-frame command sent to each robot |
| `diag_angle_error_jacobians` | 3 | the `Jacobians` primitive's old diagnostics (was `angle_error_diagnostics`) |
| `timing` | 2 | [sim time, wall-clock posix s] |
| `run_config` | struct | settings snapshot; see the 3-robot README §6 |

`<prim>` is one of `setpoint`, `jacobian`, `jacobians`. `trunk/robots_3/load_run.m` loads these files too and adds the old names (`robot4_pose`, `detected_critical_point`, `detected_critical_point_avg`, …).

**Removed:** `cluster_position`, which duplicated `cluster_state(:,1:2)`.

**Debug leftover, kept on purpose:** the `Jacobians` forward kinematics still writes a debug variable `FK_results` to the workspace (Phase 2 was deliberately kept to angle wrapping only). It stops after 201 steps, so for runs longer than 20 s it won't match `tout`. Ignore it.

### Verified offline (2026-11-12)
- **Connection diff vs `chris_4R_all_primitives.slx`:** only the changes listed above, plus 190 connections into LOG blocks.
- **Compile + 2 s run of a throwaway copy, with OptiTrack and the Decabots stubbed out** (no hardware contacted): all logs present, N rows each; `run_config` lists the 4 robots; θc commanded = 0 in all three primitives; the pacer held about 0.095 s per step.
- **The 4-point Jacobian logger** recovers a linear field exactly (1e-14) and classifies a pure vortex as type 1.
- **Not yet run on the robots.**

---

## First hardware runs (2026-11-13)

### Robot order matters: celeste opposite pacific-blue, tidal opposite redwood
- **Rule:** the kinematics assume robots 1→3 and 2→4 are the **diagonals**. Going around the square, the robots must be in the order celeste → tidal → pacific-blue → redwood.
- **First test run (`variables1.mat`, 21:57):** the robots were placed redwood → pacific-blue → celeste → tidal.
  - The model's "diagonals" were really sides, so r1 = −4.8 and r2 = −2.8.
  - The resulting shape errors produced commands of 1–5 m/s, and the robots lurched around at their 0.3 m/s limit.
  - The wiring was fine: each robot followed its own command.
- **Fix:** physically swap celeste and tidal.
  - **Direction:** this numbers the robots clockwise in Motive x–z, which gives φ = +π/2, matching the setpoint.
  - **The other swap,** pacific-blue ↔ redwood, also fixes the diagonals but gives φ = −π/2. The setpoint's sign would then have to flip.
- **Quick check before a run:** in `cluster_state`, r1 and r2 should be within [0, 1] (≈ 0.5) and φ (column 8) ≈ +1.57.

### Which critical-point method ran
All the `vortex_4robots_*` runs use the **`Jacobian` primitive**: a 4-robot least-squares fit (4 equations, 3 unknowns).
- **Root switches:** `Manual Switch` = 0, `Manual Switch1` = 1 → `Mux1`.
- **Checked on run 2:**
  - Recomputing the 4-point least-squares fit from the logged poses and `field_uv` reproduces `field_cp_jacobian` to 1e-15.
  - The robot commands equal `cmd_robots_jacobian` exactly.
  - The Orbiter command is explained by `field_cp_jacobian` with R² = 1.0000, against 0.78 for the averaging primitive's `field_cp_jacobians`.
- **Cosmetic:** the finder's console print says "from all 4 points 1,2,3". That label is a typo; the fit uses all 4.

### Results, r_d 0.01 (runs 1–2: 30 s; runs 3–8: 20 s)
- **Shape held:** d1 0.29–0.32, d2 0.30–0.34, r1/r2 0.49–0.51, φ 1.51–1.59 (setpoint 0.30 / 0.30 / 0.5 / 0.5 / 1.571).
- **Radius about the true centre (0.840, 0.856):** 0.056–0.074 m. The 3-robot model gave 0.069 at the same r_d.
- **Critical-point estimate:** the 4-point fit's jitter is 1–2 cm with no outliers, steadier than the 3-robot 3-point fit (2–3 cm).
- **Field:** read as a vortex 70–79% of steps; the rest read as spiral, i.e. a near-vortex.

### Redwood is the unseen robot: the sensor net was never trained on it (by design)
- **The setup:** the shared net (`trunk/Deep_Learning/july_tuning/everything_nets_final.mat`) was trained on celeste, tidal and pacific-blue only. Deploying it on redwood, a robot it never saw, was the point of the IDETC paper.
- **Caution:** the July `comprehensive_search.m` currently on disk lists `redwood_cal` in its training set, but that is not the version that produced the net.
- **How the unseen robot compares** (all 2026-11-13 runs, readings against the ideal vortex about the true centre):

| Robot | Trained on? | Median direction error | Worst 10% | Constant bias | Strength scale (ideal ≈ 0.87) |
|---|---|---|---|---|---|
| pacific-blue | yes | 4.9° | 12.0° | −2.3° | 0.87 |
| celeste | yes | 5.2° | 10.4° | −4.8° | 0.88 |
| tidal | yes | 5.7° | 15.7° | −5.3° | 0.83 |
| **redwood** | **no** | **7.8°** | 15.9° | **−8.0°** | **0.70** |

- **The result:** the net transfers to an unseen robot with about 2–3° more median error, a larger constant bias, and an under-read strength (about 20% low). On its held-out calibration file, redwood's worst 10% of readings are off by 30.7° (5–12° for the trained robots).
- **Impact on the 4-robot runs:** small. The 4-point least-squares fit is still better with redwood than without it. Correcting redwood's bias and scale offline removes only about a fifth of the 4-robot estimate's pull toward the cluster at large radii.

### Redwood's battery: visible in the data, fixed by new batteries
The table shows how much of its commanded speed each robot reached (median actual/commanded, after 5 s).

| Run (time) | celeste | tidal | pacific-blue | **redwood** | d2 (diagonal 2–4, includes redwood) |
|---|---|---|---|---|---|
| 1 (22:02) | 0.83 | 0.86 | 0.89 | **0.81** | 0.311 |
| 2 (22:06) | 0.81 | 0.81 | 0.90 | **0.76** | 0.311 |
| 3 (22:07) | 0.81 | 0.79 | 0.89 | **0.69** | 0.321 |
| 4 (22:08) | 0.79 | 0.82 | 0.90 | **0.61** | 0.313 |
| 5 (22:09) | 0.81 | 0.79 | 0.89 | **0.42** | 0.336 |
| 6 (22:10) | 0.82 | 0.82 | 0.93 | **0.38** | 0.323 |
| 7 (22:11) | 0.82 | 0.81 | 0.92 | **0.34** | 0.325 |
| *new batteries in redwood* | | | | | |
| 8 (22:14) | 0.83 | 0.83 | 0.89 | **0.92** | 0.298 |

- **What happened:** redwood's speed fell steadily from 0.81 to 0.34 over about 10 minutes; the other three stayed flat at 0.8–0.9.
  - **Shape:** the diagonal that includes redwood (d2) drifted long, from 0.31 to 0.34.
  - **New batteries:** redwood went to 0.92, the best of the four, and d2 came back to 0.298.
  - **Same in the response fit:** fitting each robot's response (gain = b/(1 − a)) shows the same pattern, 0.85 → 0.38 → 0.94.
- **Not the battery: a direction offset.** Redwood moves 7–11° off its command, both before and after the new batteries; the others are 1–7° off. A constant few-degree extra offset that doesn't change with battery points to redwood's **Motive heading zero**, slightly off (about 5°). Re-zero it when convenient (`Optitrack_demo/OptiTrack_SOP.md`).
- **A live battery check:** each run's per-robot ratio |actual velocity| / |commanded velocity| (from `pose_rN` and `cmd_robots_selected`) works as a battery gauge.
  - **Healthy:** 0.8–0.93.
  - **Swap batteries:** below about 0.7.
  - **Script:** the scratchpad script `robot_health_4R.m` computes it.
- **For the analysis:** runs 3–7 had a weak redwood (ratio 0.34–0.69). Flag them, or leave them out of any robot-response fit and of comparisons that depend on speed.

---

## Phase 2: the 4-robot kinematics vs the Python code (APPLIED 2026-11-12: angle wrapping only)

**What was applied** (user's choice: "just angle wrapping"), via `tools/apply_phi_wrap_only.m`:
- **φ wrap in the FK:** `phi = mod(phi + pi, 2*pi) - pi;` added right after the existing φ line in the `Cluster_Controller_setpoint` and `Jacobian` forward kinematics. This is the same line the `Jacobians` FK already had. Nothing else in the FK code changed.
- **φ-error wrap:** an orange `wrap_phi_err` MATLAB Function before the `th1_err` outport in `Cluster_Controller_setpoint/Compute_error`, `Jacobian/Formation_error` and `Jacobians/Formation_error`.
- **Not changed:**
  - the `FK_results` debug output;
  - the critical-point finders' per-step `fprintf`;
  - the inverse Jacobians.
- **Backup:** `four_robot_cluster_all_primitives_BACKUP_pre_phase2_20261112.slx` (hash 1591F98D251E). Installed model hash 2A0AD490C876.
- **Verified on a stubbed throwaway copy** (no hardware):
  - **Connection diff:** only the 3 wrap blocks were added.
  - **Compile and 2 s run:** OK, all logs N rows. φ reads −1.571 (was +4.71), and the φ error is within [−π, π].
  - **Re-run:** a second run of the script changes nothing.
- **Fuller alternative, NOT applied:** `tools/apply_quad_python_match.m`. It also replaces the FK code, adds the parallel-diagonal guard, removes the debug code and silences the prints.

The findings and the original plan follow, for reference.

Python reference: `from_mac/VF_Robot/VF_Robot/cluster_builder/src/control/quad_kinematics.py` and `src/robot/quad_cluster.py`.

### What I found
1. **The core maths is correct and matches Python.** I tested the block code numerically on 2,000 random quadrilaterals (scratchpad `quad_check.m`):
   - **Round trip:** forward kinematics of the Python inverse kinematics gives the shape back to 1.8e-15.
   - **Inverse Jacobian:** the block's analytic inverse Jacobian equals the numerical derivative of Python's inverse kinematics to 2e-9.
   - **Consistency:** forward Jacobian × inverse Jacobian = I to 9e-9.

   So the forward kinematics, inverse kinematics, Jacobian and inverse Jacobian are *not* where the problems are.
2. **Bug: φ (angle between the diagonals) isn't wrapped.**
   - **Where:** the `Cluster_Controller_setpoint` and `Jacobian` forward kinematics compute `φ = atan2(diag 1) − atan2(diag 2)` without wrapping, so φ can be anywhere in (−2π, 2π). 456 of 2,000 random shapes came out of range, and in the stubbed run a −90° shape was reported as **+270°**.
   - **Error never wrapped either:** `Compute_error` / `Formation_error` don't wrap the φ error. Only θc's error is wrapped. Python wraps both (`psi` in the FK, and `error_psi` in `quad_cluster.py`).
   - **Effect:** when the diagonals' angles straddle ±180°, the φ error jumps by 2π (6.3 rad). With gain 1, that commands a violent rotation of the second diagonal even when the shape is already right.
3. **Debug leftovers.**
   - The `Jacobians` forward kinematics has dead "unwrap θc" code (it can never run) and an `assignin('base', 'FK_results', …)` that runs every step.
   - Both critical-point finders `fprintf` every 0.1 s, which floods the console and costs real time.
4. **Check before the first run: which way the robots are numbered.**
   - The set point has φ = **+π/2** (`Cluster Shape Variables/Theta1`). With the quad's robots numbered **counter-clockwise** in Motive's x–z (1→2→3→4), the true φ is **−π/2**, which is what the stubbed test layout gives.
   - If the physical numbering is like that, the controller sees a 180° φ error and tries to mirror the formation, the same kind of handedness problem the 3-robot model had (s = −1).
   - **What to do:** on the first run, compare `cluster_state(:,8)` with the setpoint, and flip the setpoint's sign (or the robot order) if they disagree.
5. **Matches Python already:**
   - centroid, d1, d2, r1, r2 and θc = angle of diagonal 1→3;
   - gain 1 on all shape errors (Python `position_gain` 1.0 on d1, d2, r1, r2, psi);
   - no θc control (done in Phase 1).

   Python also unwraps θc from step to step (`prev_theta_c`). That isn't needed here: θc isn't controlled, and its error is wrapped anyway.

### Plan (ready to run; tested on a scratch copy)
`tools/apply_quad_python_match.m four_robot_cluster_all_primitives.slx` does three things. It is safe to re-run.
1. **Forward kinematics:** replaces the three forward-kinematics blocks with `tools/fk_quad_python_match.m`. It has the same ports and outputs, φ wrapped to [−π, π], Python's parallel-diagonal guard, and the debug code removed.
2. **Error wrap:** wraps the φ error to [−π, π] (a `wrap_phi_err` block) in all three error blocks.
3. **Quiet:** silences the per-step prints in both critical-point finders.

The inverse-Jacobian blocks stay as they are, since they're already correct.

**Scratch-copy test results:**
- kinematics still exact (round trip 1.8e-15; J·J⁻¹ = I to 1e-8), and φ never out of range;
- the connection diff shows only the three wrap blocks;
- in the stubbed run φ = −1.571 (correct, was 4.71) and the φ error is wrapped;
- `FK_results` is gone.

**To apply (after you say OK):**
1. Back up the model.
2. Run the script on `four_robot_cluster_all_primitives.slx`.
3. Check point 4 above on the first hardware run.
