# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PhD thesis research code for a multi-robot (swarm/cluster) testbed. Small ground robots read a printed floor map with color sensors; the color (HSV hue = direction, saturation = magnitude) encodes a vector or scalar field. Robot clusters then run control "primitives" to find and follow field features: vortices, saddles, sinks/sources, separatrices, ridges, trenches and contours.

It is not a packaged software project. There is no build system, test suite, linter or dependency manifest. Code is run interactively, either in MATLAB/Simulink or as Python scripts and notebooks. The directory has both `.git` and `.svn` metadata, and `git` is not on PATH in this Windows shell.

## Running things

- **Simulink models (`*.slx`)**: open them in MATLAB and run them from there. Before running any model that uses learned sensor calibration, run that folder's `neural_network_loader.m` first. It loads `everything_nets_final.mat` into the base workspace, and MATLAB Function blocks inside the models pull the nets out with `evalin('base', ...)`. `trunk/Deep_Learning/Tutorials/calling_trained_net_example.m` shows this pattern.
- **MATLAB path**: the models call functions in `trunk/utilities/`, `trunk/utilities/behaviors/`, `trunk/utilities/behaviors/WeightingFunctions/` and `trunk/utilities/testbed/`. Add these to the path (for example `addpath(genpath('trunk/utilities'))`).
- **Python vector-field simulator**: run it from inside the simulator folder, because imports are relative to that folder (`from env... `, `from primitives...`):
  ```
  cd "trunk/Python Simulations/Vector_Fields/VF_3_robot"
  python main.py        # or main_lite.py
  ```
  It needs numpy, matplotlib, scipy and torch. Choose the field and mode by editing the flags and imports at the top of `main.py` (`USE_BLENDED`, `USE_NN_ONLY`, `USE_RBF_ONLY`, `USE_ANALYTICAL`, and `from env.<Field> import <fn> as enviro`). There is no CLI.

## Architecture

### MATLAB / Simulink (hardware + simulation)
- `trunk/utilities/testbed/`: hardware interface classes. These are `matlab.System` blocks running in interpreted mode. `InterpretedSystemBase` is the abstract base. `Decabot` talks to a robot over DataTurbine (RBNB) or a direct TCP socket. `OptiTrackRigidBody` reads motion capture. `LoopPacer`, `msfun_realtime_*` and `realtime_pacer_lib.mdl` pace the simulation to wall-clock time. These blocks interface with Java, so they cannot be code-generated, and they declare output sizes and types explicitly.
- `trunk/utilities/behaviors/`: swarm behaviors (`SwarmSim*.m`). Each returns a velocity vector `Vf = [Vfx, Vfy, Vft]` for one robot, given the flattened `RobotParams` vector (4 entries per robot), the robot index, the sensor range and a **weighting function handle**. `SwarmSimSum.m` adds up all behavior outputs into the command. To add a behavior, add it to both that function's inputs and its sum.
- `trunk/utilities/behaviors/WeightingFunctions/`: interchangeable weight functions. Signature: `W = f(NRobot, RobotParams, SensorRange, SysParam)`, returning per-robot weights. These are what turn generic attract/disperse/orbit into specific behaviors.
- `trunk/robots_3/README_hardware.md` is the main write-up of the 3-robot hardware model. It covers the data flow, how the floor maps encode hue and saturation (and why they're printed mirrored), the hue→vector conventions and the 2026-11-11 vortex/saddle fix, the orbit-radius floor, and what each logged variable is. Read it before changing the hardware model or analysing hardware runs.
- **Orbiter (3-robot and 4-robot, since 2026-11-12):**
  - **The law:** Python's structure, `v = k_t·r̂⊥ + k_r(r − r_d)·r̂`, where r̂ is the unit vector to the critical point. Radial speed is clipped to ±0.3 and the direction flag is +1.
  - **Gains:** the user's choice, **`k_t` 0.15, `k_r` 1**. Python's defaults (0.2 / 0.5) are not used; the slower orbit holds the radius better.
  - **No θc control:** `thC_gain` = 0 in every primitive, matching Python's `omega_c = 0`.
  - **One source file:** `trunk/robots_3/tools/orbiter_python_match.m` holds the code and gains for both models. Push it into a model with `tools/sync_orbiter_code.m <model.slx>`, so the 3-robot and 4-robot Orbiters stay identical.
  - **Python sims must be re-run with k_t 0.15, k_r 1** (README_hardware.md §5.4, §5.7).
- **Measured robot response (re-fit 2026-11-15 from the campaign logs, supersedes the 2026-11-11 "momentum 0.4" fit):**
  - **Model:** each robot follows its command after a **one-step (0.1 s) delay**, almost without lag: `v(k) = 0.11·v(k−1) + 0.89·0.85·u(k−1)`. So the gain is ≈ 0.85 and kinetic friction ≈ 0.
  - **Speed cap:** speed saturates at about 0.3 m/s.
  - **Stiction:** not yet measured.
  - **Details:** `README_hardware.md` §5.7. The Python simulator assumes momentum 0.7 and full speed (`omnibot.py`). Python simulations should be re-run with the measured values **after the hardware data campaign**, re-fitting them first from all the campaign's logs. That same re-run should use the hardware Orbiter gains (k_t 0.15, k_r 1), not Python's defaults. Method and numbers: `README_hardware.md` §5.7.
- **4-robot model:** `trunk/robots_4/four_robot_cluster_all_primitives.slx` was built by Claude on 2026-11-12 from `chris_4R_all_primitives.slx` (left unchanged). It carries the 3-robot changes: decode fix, Python Orbiter, no spin, logging suite. Its kinematics match Python. Since 2026-11-12, φ (the angle between the diagonals) and its error are wrapped to [−π, π], via `tools/apply_phi_wrap_only.m`; the user chose angle wrapping only. Still check on the first run whether φ's sign matches the setpoint (+π/2). See `trunk/robots_4/README_four_robot.md`.
- 🏆 **Robot Report Card:** `trunk/campaign_results/ROBOT_REPORT_CARD.md` (the user is keen on this one). It ranks the robots: **pacific-blue** best (sensor 5.1°, highest and most consistent speed; brief wheel or mat stalls near the centre), **celeste** second (5.2°, most predictable motion), **tidal** third (its battery died on 11/15), **redwood** last (never trained on: about 7–8°). Update it when new data changes the ranking.
- **Campaign results:** `trunk/campaign_results/2026-11-13_vortex_orbit_campaign.md` is the write-up of the 2026-11-13 vortex-orbit campaign (102 runs, 3R and 4R).
  - **Main script:** `trunk/campaign_results/scripts/campaign_report.m` regenerates every table. Re-run it and update the write-up when more campaign data arrives.
  - **Key numbers:** true centre (0.839, 0.856) for that map placement; plant momentum ≈ 0.44, speed reached ≈ 0.85; sensor ranking pacific-blue ≈ celeste > tidal > redwood (the unseen robot).
- **Hardware lessons from 2026-11-13** (`README_hardware.md` §5.9–5.10):
  - **Batteries fade within a session.** Redwood's speed reached fell 0.81 → 0.34 over about 10 min, then 0.92 after new batteries. Use each robot's |actual|/|commanded| velocity ratio from the logs as a battery gauge (healthy 0.8–0.93, swap below about 0.7). Flag or exclude low-battery runs from robot-response fits.
  - **4-robot placement:** going around the square, celeste → tidal → pacific-blue → redwood. Placing them wrong turns the kinematics' diagonals into sides.
- Experiment folders are grouped by cluster size: `robots_1/`, `robots_3/`, `robots_4/`, plus `robot_racer_july_tuning_nns/`. Each holds its Simulink models, the `.mat` run logs (`run*.mat`) and its plotting and analysis scripts. `archives/` and `artifacts/` hold older models.
- `ClusterSpace_Simulations/`: pure-simulation cluster-space control models (continuous and discrete 3-robot).
- **Sensor net used on the hardware:** the 3-robot and 4-robot models both load the single shared net `trunk/Deep_Learning/july_tuning/everything_nets_final.mat` (`everything_hue_net`, `everything_sat_net`).
  - **Trained on:** celeste, tidal and pacific-blue only.
  - **Redwood was deliberately never trained on.** It is the unseen-robot test from the IDETC paper. The July `comprehensive_search.m` on disk lists `redwood_cal` in its training set, but that is not the version that produced the net.
  - **On the floor (2026-11-13):** redwood reads worst (7.8° median direction error vs 4.9–5.7°, magnitude about 20% low). That is the expected generalization gap, not a bug.
- `Deep_Learning/`: color-sensor-to-field calibration. Each robot is named after a color (celeste, pacific-blue, redwood, tidal, pink). Each has `*_cal.csv` / `*_ver.csv` data and per-robot nets, grouped into tuning campaigns (`january_`, `march_`, `july_tuning`). `robot_calibration_tool/hsv_rgb_sampler.slx` collects the raw samples.
- `Optitrack_demo/`: standalone mocap demo. `Optitrack_demo/OptiTrack_SOP.md` is the lab procedure for hardware runs: prechecks, Motive camera calibration and ground plane, Decabot Helper setup and voltage checks, rigid-body creation and naming, the `javaaddpath C:\DecaMATLAB\SocketShim.jar` step, and end-of-day steps. Read it before debugging anything on the hardware side.

### Python (`trunk/Python Simulations/`)
- `Vector_Fields/VF_3_robot/` is the main simulator. `Vector_Field_4_Robot*` are 4-robot variants with the same layout.
  - `env/`: analytical fields. `VF_bases.py` defines the basis flows (vortex, sink, source, saddle, doublet, and others). The field modules (`Sinking_Vortex.py`, `Saddle.py`, ...) build fields as sums of those bases. `grid_setup.py` defines the plotting grid.
  - `clusters/robot_cluster.py`: `RobotCluster` holds the robot formation and gets field readings in one of four ways: an analytical function, a PyTorch NN, a scipy RBF interpolator, or a blend of NN and RBF. It loads trained models (`hue_model.pth`, `sat_model.pth`, `model_info.pkl`, `*_rbf_interpolator.pkl`) from a hard-coded `*_predictors/` sibling folder, so switching fields means editing that path.
  - `*_predictors/`: trainer scripts plus saved models for each field type.
  - `primitives/control_primitives.py`: control laws with the signature `f(cluster)`. Examples: critical-point finding, orbiting, center finding, ROI finding. They are passed to `exec_sim.simulation.execute_simulation(cluster, primitive, title)`, which draws the field and steps `cluster.move(primitive)` for a fixed number of steps.
  - NN outputs follow the floor-map encoding: the hue model outputs (sin, cos) of the direction, and the saturation model's output is mapped to magnitude through `(sat+1)/2` and then `(sat-0.3)/0.7`.
- `Archive/` and `Separatrix_Control_testing/Failed_experiments/` hold older notebooks and experiments that did not pan out. Do not use them as reference implementations.

## Conventions / gotchas
- **⚠ Hardware run `.mat` files contain STALE variables.** They are whole-workspace saves, so 3-robot saddle files carry 4-robot `pose_r4` / `field_cp_jacobian`, 4-robot files carry 3-robot `field_cp` / `orbit_radius`, and all of them carry 2025 old-name variables.
  - **Robot count:** get it from `size(cmd_robots_selected, 2)/3` or the file name, never from whether `pose_r4` exists.
  - **What to read:** only the variables in that run's `run_config.log_legend`.
  - **How to load:** use `trunk/robots_3/load_run.m`.
  - **Clean copies:** `trunk/campaign_results/scripts/clean_campaign_files.m`.
  - **Details:** `data_hardware_campaign_11_11_2026/README_STALE_VARIABLES.md`.
- `slprj/` folders and `*.slx.autosave` / `*.asv` files are generated by Simulink and MATLAB. Do not edit them.
- Many scripts keep alternative configurations as commented-out blocks (for example the gain combinations in `SwarmSimSum.m` and the field compositions in `env/VF_env.py`). Experiments are switched by commenting code in or out, not through parameters.
- Folder and file names contain spaces (`Python Simulations`, and many notebooks). Quote paths.
- **OptiTrack rigid-body zero must match each robot's forward axis.** The models turn global velocity commands into body-frame commands using each robot's OptiTrack heading (`Robot_signal_processing_*`, `Global to Body Rotation Matrix`). If a rigid body's yaw zero in Motive doesn't match the robot's forward axis, that robot drives at a constant angle away from its command, and the cluster won't hold its shape or converge. Seen on 2026-11-11 in `robots_3/three_robot_cluster_all_primitives.slx`: celeste was off by about +155°, tidal −80°, pacific-blue −150°. It was fixed by re-zeroing the rigid bodies in Motive (residual afterwards was within about 10°). The zero is set when the rigid body is created: the robot's blue wheel must point +x (see `Optitrack_demo/OptiTrack_SOP.md`). Rigid-body names must match the model's OptiTrack `rigidBodyNames` and Decabot Helper exactly (lowercase, e.g. `pacific-blue`). Symptom check: compare the direction each robot is commanded with the direction it actually moves, using the logged `robot*_pose`. If the angle between them is a constant per robot, re-zero that robot in Motive.
