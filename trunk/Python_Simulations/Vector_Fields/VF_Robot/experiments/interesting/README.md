# Interesting experiments

Moved here 2026-10-09. Not used by the three papers, but either future work they name (OW boundary, D=0, TRAP cores, time-varying fields, type classification, sim-to-real/robot dynamics, robot count/fit order) or a case where it was unclear whether a paper still relies on it.

Scripts here compute the VF_Robot root as two directories up from their own file and import `experiments.*`, so after the move they need one more `dirname` (or to be run from a copy back in `experiments/`) before they run again. Their outputs are still in `../outputs/`.

| File | Why it is here |
|---|---|
| `main_logic_g_quad.py` | OW boundary / D=0: 4-robot det(J)=0 Okubo-Weiss contour tracker (Logic G) |
| `main_logic_g_pentagon.py` | OW boundary / D=0: 6-robot det(J)=0 contour tracker, heuristic Logic G |
| `main_logic_g_newton_pentagon.py` | OW boundary / D=0: 6-robot det(J)=0 contour tracker, Newton step |
| `mc_sweep_ow.py` | OW boundary: Monte Carlo of the OW boundary tracker cut from the separatrix paper |
| `ow_clean_runs.py` | OW boundary: clean runs and ow_trajectories figure from the cut OW thread |
| `main_separatrix_oecs.py` | TRAP cores: Primitive 10 (oecs_trap_step) on the double gyre |
| `main_ocean_hfr_2km_oecs.py` | TRAP cores: Primitive 10 core-seek on the Santa Barbara Channel field |
| `oecs_objectivity_demo.py` | TRAP cores: objectivity demo, Logic C vs Primitive 10 |
| `mc_sweep_oecs.py` | TRAP cores: Monte Carlo of the OECS core-seeker |
| `oecs_estimator_check.py` | TRAP/OECS: open-loop check of the rate-of-strain estimator quantities |
| `traverse_rotated_check.py` | OW paper limitation: s1 tracker when the strain eigenframe is tilted (nonzero shear) |
| `unsteady_gyre_check.py` | Time-varying field (OW and critical-point paper future work): D tracker on the unsteady gyre |
| `ral_experiment_b.py` | Time-varying fields (critical-point paper future work): dynamic gyre, wobbling vortex |
| `main_ocean_hfr_2km_critical_point.py` | Critical-point paper future work: 3-robot critical point seeking in the time-varying, advecting ocean field |
| `classify_from_hardware_logs.py` | Critical-point paper future work: critical point type classification from the hardware logs |
| `identifiability_check.py` | OW paper future work (robot count): can 5 robots identify the quadratic under incompressibility |
| `mc_sweep_radius.py` | Unsure: closed-loop formation radius sweep (reviewer item); Draft_12 argues the rho trade-off without it |
| `estimator_accuracy_sweep.py` | Unsure: estimator error vs noise and radius; backs the rho^2 noise argument but no longer a figure |
| `plot_estimator_accuracy.py` | Unsure: plots estimator_accuracy_sweep.py output (keep together) |
| `omega_seed_sweep.py` | Unsure: rotating-frame seed condition sweep; Draft_12 states the seed condition, unclear if these numbers are used |
| `thesis_ch8_radius_sweep.py` | PhD thesis ch8 (not one of the 3 papers): fit order vs noise, relates to cubic-fit future work |
| `thesis_ch8_magnitude_saddle.py` | PhD thesis ch8 (not one of the 3 papers): descent on |v| as a scalar landscape |
| `hop_rate_instrumentation.py` | Unsure: pre-rewrite Primitive 11 diagnostics cited in primitive11_simplification/README.md (does not run since 2026-10-09) |
| `margin_gate_sweep.py` | Unsure: pre-rewrite Primitive 11 diagnostics cited in primitive11_simplification/README.md (does not run since 2026-10-09) |
| `main_mod_scal_newt_step.py` | Unsure: pentagon scalar saddle seeking (Primitive 5, bimodal Gaussian) |
| `main_mod_dauph.py` | Unsure: pentagon scalar saddle seeking (Case 6, Dauphin-style) |
| `sim2real_comparison_all_multimode.py` | IDETC future work (sim-to-real gap / digital twins): latest multi-field-mode sim2real comparison |
| `sim2real_multimode_results` | IDETC future work: output of sim2real_comparison_all_multimode.py |
| `sim2real_with_center_tracking.py` | IDETC future work (sim-to-real gap): sim2real with center-estimate tracking |
| `sim2real_center_tracking_results` | IDETC future work: output of sim2real_with_center_tracking.py |
| `test_momentum_velocity_effects_fixed.py` | IDETC future work (robot velocity/acceleration dynamics): momentum and velocity-limit effects |
| `momentum_velocity_analysis_fixed` | IDETC future work: output of test_momentum_velocity_effects_fixed.py |
| `analyze_individual_robot_motions.py` | IDETC future work (robot dynamics): per-robot motion during orbits |
| `individual_motion_analysis` | IDETC future work: output of analyze_individual_robot_motions.py |
