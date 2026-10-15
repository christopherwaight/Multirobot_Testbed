# Unused experiments

Moved here 2026-10-09. None of these feeds the IDETC 2025 testbed paper, the critical point (Systems) paper, or the Separatrix/OW paper (Draft_12), and none is future work named in those papers. Most are superseded drafts (Draft_5c/6a era), parameter searches whose result is already fixed, or older 3/4-robot variants.

Scripts here compute the VF_Robot root as two directories up from their own file and import `experiments.*`, so after the move they need one more `dirname` (or to be run from a copy back in `experiments/`) before they run again. Their outputs are still in `../outputs/`.

| File | Why it is here |
|---|---|
| `FINAL_SWEEPS_HOWTO.md` | Draft_5c-era 10k sweep instructions (mc_separatrix / mc_ow) |
| `run_10k_objective.sh` | Draft_5c-era 10k sweep driver |
| `analyze_mc_sweeps.py` | Draft_6a success-rate tables, replaced by the Draft_12 noise sweep |
| `basin_map.py` | Draft_5c basin map, not in Draft_12 |
| `plot_basin_map.py` | Draft_5c basin map figure, not in Draft_12 |
| `baseline_fixed_weights.py` | Draft_5c fixed-weight baseline, not in Draft_12 |
| `mc_sweep_acquisition.py` | Draft_5c acquisition-under-noise sweep, not in Draft_12 |
| `mc_sweep_separatrix.py` | Draft_5c D tracker table, replaced by mc_sweep_noise_both_trackers.py |
| `mc_sweep_oecs_traverse.py` | Draft_5c s1 tracker table, replaced by mc_sweep_noise_both_trackers.py |
| `mc_sweep_flip_resolution.py` | Draft_5c flip-resolution sweep, replaced by mc_sweep_noise_both_trackers.py |
| `mc_sweep_flip_resolution_sigma_p.py` | Draft_5c flip-resolution sigma_p sweep, replaced |
| `park_vs_traverse_demo.py` | Draft_5c park-vs-traverse figure, not in Draft_12 |
| `separatrix_clean_runs.py` | Draft_5c separatrix_trajectories figure, not in Draft_12 |
| `traverse_flowband_shrink.py` | Draft_5c flow-band table for the removed CROSS mode |
| `main_separatrix_v6r.py` | Early Logic C runner, superseded by main_separatrix_traverse.py |
| `_ftle_grid_sweep.py` | One-off FTLE rendering parameter sweep |
| `_generate_ocean_ftle_reference.py` | Old 6 km FTLE reference image |
| `_generate_ocean_ftle_matched.py` | Old 6 km FTLE matched rendering |
| `_generate_ocean_ftle_2km_matched.py` | Old 2 km FTLE reference image (overlay now drawn by the progression script) |
| `main_ocean_hfr_v6r.py` | 6 km ocean run, superseded by the 2 km shared-start run |
| `main_ocean_hfr_v6r_timevarying.py` | 6 km time-varying ocean run, superseded |
| `main_ocean_hfr_2km.py` | Early 2 km D tracker run, superseded by main_ocean_hfr_2km_shared_start.py |
| `main_ocean_hfr_2km_ftle_overlay.py` | Draft_5c ocean overlay figure, superseded |
| `main_ocean_hfr_2km_traverse.py` | Early 2 km s1 tracker run, superseded by the shared start |
| `ocean_hfr_2km_branch_sensitivity.py` | Draft_5c branch-sensitivity figure, not in Draft_12 |
| `ocean_hfr_2km_panel_progression.py` | Draft_5c panel progression, superseded by main_ocean_hfr_2km_progression.py |
| `ocean_shared_start_search.py` | Shared-start search (old anisotropic map) |
| `ocean_shared_start_closest_approach.py` | Shared-start search variant (old map) |
| `ocean_shared_start_visualize_top2.py` | Shared-start search visualizer (old map) |
| `ocean_candidateA_finemc_search.py` | Candidate A fine search v1 (old map) |
| `ocean_candidateA_finemc_v2.py` | Candidate A fine search v2 (old map) |
| `ocean_candidateA_finemc_v3.py` | Candidate A fine search v3 (old map) |
| `ocean_square_map_start_search.py` | Start search on the isotropic map (start already chosen) |
| `ocean_rank_candidates.py` | Ranks the start search (start already chosen) |
| `ocean_candidate_overlays.py` | Contact sheet for start candidates |
| `ocean_candidate_jitter.py` | Start jitter check for candidates |
| `ocean_candidate_time_panels.py` | Time panels for candidates |
| `ocean_param_search.py` | Operating-point search (gain/radius already chosen) |
| `ocean_param_mc.py` | Operating-point Monte Carlo (already chosen) |
| `ocean_rank_params.py` | Ranks the operating-point search |
| `ocean_legacy_corridor_reference.py` | Freezes the pre-isotropic-map ocean paths (its JSON stays in outputs/) |
| `ocean_diagnostics.py` | Draft_6a per-step ocean instrumentation |
| `precompute_ocean_hfr_2km_cache.py` | Cache builder for the Manim visualization, not a paper |
| `double_gyre_curl_div.py` | One-off curl/divergence plot |
| `create_simulation_orbital_table.py` | Deprecated 6-field orbital LaTeX table |
| `create_simulation_orbital_table_8.py` | LaTeX formatter reading the RBF orbital CSV; paper table uses the NN data |
| `generate_simulation_orbital_table.py` | 6-field orbital run, superseded by the _8_nn version |
| `generate_simulation_orbital_table_8.py` | RBF orbital run; paper Table III uses the NN version |
| `simulation_orbital_results.csv` | Output of generate_simulation_orbital_table.py |
| `simulation_orbital_results_8fields.csv` | Output of the RBF orbital run |
| `simulation_orbital_table.tex` | Output of create_simulation_orbital_table.py |
| `orbital_8fields_output.log` | Log of the RBF orbital run |
| `monte_carlo_analysis_run_to center.py` | 100-trial run-to-center MC, superseded |
| `monte_carlo_analysis_run_to_center_8.py` | RBF run-to-center MC; per its own note it does not reproduce paper Table II |
| `monte_carlo_8fields_output.log` | Log of the RBF run-to-center MC |
| `monte_carlo_analysis_rbf_field.py` | RBF vortex MC, superseded |
| `generate_table1_data.py` | Old Table 1 with 3- and 4-robot clusters |
| `generate_analytical_orbit_data.py` | Old Table 4 analytical orbit data |
| `analytical_orbit_results.csv` | Output of generate_analytical_orbit_data.py |
| `radius_sweep_comparison.py` | 3- vs 4-robot orbiter radius sweep |
| `radius_sweep_field_comparison.py` | Orbiter radius sweep across field modes |
| `radius_sweep_results` | Output of the radius sweeps |
| `field_comparison_consolidated.py` | NN/RBF vs analytical field figure, not in the paper (paper uses measurement_error_comparison.png) |
| `both_field_imperfection_analysis.py` | reconstruction_comparison figure, not in the paper |
| `reconstruction_comparison.png` | Output of both_field_imperfection_analysis.py |
| `reconstruction_comparison1.png` | Older copy of the reconstruction figure |
| `saddle_field_imperfection_analysis.py` | Saddle NN/RBF imperfection analysis |
| `vortex_field_imperfection_analysis.py` | Vortex NN/RBF imperfection analysis |
| `diagnose_saddle_raw_transform.py` | Saddle raw-data coordinate diagnosis |
| `diagnose_saddle_rbf_failure.py` | RBF saddle failure diagnosis |
| `saddle_rbf_diagnosis` | Output of diagnose_saddle_rbf_failure.py |
| `saddle_diagnosis_complete.log` | Saddle diagnosis log |
| `saddle_diagnosis_final.log` | Saddle diagnosis log |
| `saddle_diagnosis_full.log` | Saddle diagnosis log |
| `saddle_diagnosis_output.log` | Saddle diagnosis log |
| `saddle_tuning.py` | Gain tuning for scalar_newton_with_rotation |
| `saddle_point_tuning_experiment.py` | Gain tuning for scalar_newton_with_rotation |
| `saddle_point_fine_tuning_multifield.py` | Fine tuning for scalar_newton_with_rotation |
| `diagnostics_analytical.csv` | Dec 2025 sim2real diagnostics |
| `diagnostics_neural_network.csv` | Dec 2025 sim2real diagnostics |
| `diagnostics_rbf_interpolator.csv` | Dec 2025 sim2real diagnostics |
| `sim2real_comparison.py` | First sim2real comparison, superseded by the multimode version |
| `sim2real_comparison_vortex1.png` | Output of sim2real_comparison.py |
| `sim2real_comparison_all.py` | sim2real comparison, superseded |
| `sim2real_comparison_all_fixed.py` | sim2real comparison, superseded |
| `sim2real_results` | Output of the superseded sim2real scripts |
| `analyze_sim2real_results.py` | Summary of sim2real results |
| `analyze_sim2real_results_simple.py` | Summary of sim2real results (no pandas) |
| `analyze_primitives.sh` | Shell summary of sim2real results |
| `analyze_real_world_primitives.sh` | Shell summary of sim2real results |
| `analyze_results.sh` | Shell summary of sim2real results |
| `primitive_analysis_results.txt` | Output of the analyze_*.sh scripts |
| `real_world_primitive_analysis.txt` | Output of the analyze_*.sh scripts |
| `test_momentum_velocity_effects.py` | Broken first version (its _fixed successor is in interesting/) |
| `momentum_velocity_analysis` | Output of the broken first version |
