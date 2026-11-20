# Draft_14 numbers ledger

Every number wrapped in `\rev{}` in `Draft_14.tex` is listed here with its source.
Paths are relative to `trunk/Python_Simulations/Vector_Fields/VF_Robot/` unless they start
with `Paper_Writing/`. "Closed form" means checked by hand or by the one-line computation
given. Robot model "2026" is `experiments/_mc_common.ROBOT_2026`.

## Section II (estimator)

| Number | Where | Source |
|---|---|---|
| 0.632, 1.265, 2.53 (noise gains, heading independent) | II-C Measurement Noise | closed form: sqrt(diag(Phi^-1 Phi^-T)) for the pentagon-plus-center at rho = 1, three headings; printed in the session (gains 1, 0.632, 0.632, 1.265, 2.53, 2.53, cond 8.26) |
| 1.4, 2.8, 8.2, 4.0 % (noise-free truncation, rho 0.075) | II-E check paragraph | `experiments/draft14/outputs/estimator_check.txt` (J, D, gD, gs1 max rows) |
| scaling rho^2 for J and D, rho for gradients | II-E | same file: ratios 0.45 and 2.2 (J, D) vs 0.65 and 1.6 (gD, gs1) across rho 0.05, 0.075, 0.1125 |
| within 1 % (pure 2nd-derivative spread vs 2.53 sigma/rho^2) | II-E | same file, rho 0.05 and 0.075 rows (1.021 vs 1.012 ... 2.251 vs 2.249) |
| 0.43 vs 19.2 (fitted vs true across-trench curvature at y = -0.45) | II-E | `experiments/draft14/outputs/hessian_check.txt`; 19.2 = 2 pi^6 A^2 |
| 3.5 deg (eigenvector error, \|y\| <= 0.40, 8 headings) | II-E | same file |

## Section III (architecture and closed loop)

| Number | Where | Source |
|---|---|---|
| g 0.86, L 0.055 s, tau 0.02 s | III-A, Table I | [10] revision, Robot Controller Layer; `_mc_common.ROBOT_2026` |
| 21.1, 16.1 (loop-gain bounds) | III-D | [10] revision, Sec. III-D (`\rev{21.1}`, `\rev{16.1}` there); 1/(0.86 * 0.055) = 21.1 |
| 0.14 (D gain over bound near saddles) | III-D | closed form: 3/sin^2(pi y_f) = 16.1 at \|y\| = 0.358, i.e. 0.142 from the saddle |
| 0.19, 29 (s1 gain) | III-D | closed form: 3 pi^4 A \|cos pi y_f\| = 29.2 max; equals 16.1 at \|y\| = 0.186 |
| 0.008 (friction ring) | III-D | closed form: v_c/k = 0.025/3 |

## Section IV-B (clean runs, 2026 model)

| Number | Where | Source |
|---|---|---|
| acquisition 0 to 10 steps (unchanged) | IV-B | `experiments/outputs/oecs/traverse_vs_logic_c_seven_d14.csv` (t_band 0, 5, 10) |
| D settles 0.005 to 0.006, s1 within 0.009 of p1* | IV-B | same CSV: D final_y -0.4936 to -0.4951; s1 final_y -0.5078 to -0.5090 |
| s1 wiggle median 0.004 below y = -0.19 (S6, S7), D under 0.001 | IV-B | `experiments/draft14/outputs/clean_ride_offsets.txt` (0.0036, 0.0037 vs 0.0008, 0.0006) |

## Pilot (not in the paper text unless noted)

| Number | Where | Source |
|---|---|---|
| g_perp 1 vs 0.5 retention differs by at most 2.3 points | IV-C (planned) | `experiments/draft14/outputs/pilot_gperp.csv`, 2026 rows, S3/S6 at sigma 0.002/0.005 |

## Section V (ocean)

| Number | Where | Source |
|---|---|---|
| jitter: D reaches southern channel 99/100, lands 66/100 (median 23.7 h); s1 0/100 south, 92/100 land north (median 6.9 h) | V-B | `experiments/draft14/outputs/ocean_jitter_why.csv`, summary in `ocean_jitter_why_summary.txt` |
| s1 turns upstream in 98/100 at median 2.5 h; ratio 0.46 in the hour before; D first 7 h ratio 0.27, upstream 7 % | V-B | `ocean_jitter_why_summary.txt` |
| backward band distances 1.7 / 6.5 km (s1, 7-14 / 14-21 h), D 5.6 km (7-14 h); s1 2.0 km to forward band 14-21 h (1.95) | V-B | `experiments/draft14/outputs/ocean_backward_ftle.txt` |
| perturbed rides' median distance to the 95th-pct band: D 1.5 (1.45), s1 0.9 km; nominal 1.60 / 2.10 reproduces Draft_13 | V-B | `ocean_jitter_why_summary.txt` (last block) |
| 99, 66 (D south / island), 98 (s1 upstream), 2.5 h, 0.46, 0.27, 6.9 h, 101 settings | V-B | `ocean_jitter_why_summary.txt`; the nominal (jitter 0) plus 100 perturbations |
| Draft_13 V-B numbers kept (80 %, 0.15-0.20, 1.3 km, 34/42, 37/42, 49 %, 70 %, 12/27, 2.1 vs 8.4 km, 1.6/2.1 km, 14 h, 23 h, 21 h, 3.8 km) | V-B | `experiments/ocean_section_v_metrics.log`; the Draft_14 replay's nominal row reproduces the trajectories exactly (land 23.0 h, ride 8.2/3.0 km), so the October sign fix did not change them |

## Acquisition basin (Fig. 5 shading)

| Number | Where | Source |
|---|---|---|
| 8000 starts per tracker, 0.1 cells | Fig. 5 caption | `experiments/outputs/mc_zone/zone_trials_d14.csv` (mc_zone_of_convergence.py --trials 8000 --robot 2026 --tag _d14) |
| >= 90 % out to 0.40 (D 0.40, s1 0.45) near y = 0; 0.25 at \|y\| = 0.2 | IV-B | `experiments/draft14/outputs/zone_d14_summary.txt` |

## Section IV-C (noise, 2026 model, 1000 trials per cell)

Sources: `experiments/outputs/mc_noise_both/noise_both_S<i>_d14.csv`, radius cells `noise_both_S3_r050.csv`, `S4_r050`, `S3_r1125`, `S4_r1125`; summary `experiments/draft14/outputs/noise_d14_summary.txt` (log-interpolated 50 % crossings); mechanism `experiments/outputs/mc_mechanism/d_fallback_d14.csv`, `s1_reversal_d14.csv`, `s1_hazard_d14.csv` (2000 trials per start).

| Number | Source detail |
|---|---|
| 1000 trials per point | run_noise_sweeps.sh |
| 60 % (max retention at sigma_uv 0.01, any start) | S5 D 60.2 % |
| 5.0, 4.7 (D sigma50 ratio rho 0.05 to 0.075, S3, S4) | 0.00832/0.00167, 0.00934/0.00198 |
| D never below 50 % at rho 0.1125 | S3_r1125, S4_r1125 D rows |
| 2.6 (s1 sigma50 ratio rho 0.075 to 0.1125) | 0.00395/0.00152, 0.00707/0.00270 |
| 4.4 to 5.6, 3.5, 1.3 to 1.5 (sigma_uv ratios) | S1 4.43, S2 5.56, S3 5.47, S4 3.45, S6 1.45, S7 1.25 |
| 2.8 to 4.9, 1.3 to 1.4 (sigma_p ratios) | S1 3.44, S2 4.05, S3 4.85, S4 2.81; S6 1.39, S7 1.34 |
| curves cross near 0.005 (S6, S7) | retention D/s1 at 0.004: 47.9/40.3, 53.9/42.2; at 0.006: 16.5/27.6, 24.0/29.2 |
| 2.3 points (g_perp halved) | `experiments/draft14/outputs/pilot_gperp.csv` |
| 96 to 100 % of s1 misses end at p2* (sigma 0.002) | s1_reversal_d14 failures_at_top 0.958 to 0.996 |
| 0.13 vs 0.002 to 0.004 reversal hazard | s1_hazard_d14 (0.4-0.5 bin 0.134; -0.1 to 0.2 bins 0.0024 to 0.0041) |
| (cut for length) 77 %, 33 % (S5, sigma 0.01, s1); 16 points (D success minus retention) | noise_both_S5_d14; not in the final text |
| 95, 86, 32, 38, 85, 81 (S6, S7 definite Hessian at sigma 0.005) | d_fallback_d14 |
| 75, 84 (S1, S2) | d_fallback_d14 |

## Section IV-D (rotating observer, 2026 model, 10^4 trials per frame)

Source: `experiments/outputs/oecs/objectivity_rotating_summary_d14.csv`, log `experiments/draft14/outputs/rotating_d14.log`.

| Number | Source detail |
|---|---|
| 0.18 rate; 0.103 = 0.86*3*0.04; 0.21 = 0.103/0.5; 13 % margin = 1 - 0.09/0.103 | closed form |
| s1 offset 0.010 inertial, at most 0.015 rotating; ends 0.012 to 0.015 from p1* | abs_offset_median, final_dist_median |
| D offset 0.004, 0.035, 0.063; spread < 0.007 | abs_offset_median; offset_p5..p95 at 0.18 (0.058 to 0.065) |
| 28 to 32 % lose the straddle at 0.18 | straddle_lost_frac 0.284, 0.317 |
| 0.032, 0.058 predicted trench shift | Omega/(pi^3 A) |
| 0.072 (D final at 0.10), 0.012 (s1), 0.091 (stagnation shift at 0.18), 0.138 (D final at 0.18) | final_dist_median; 0.5*0.18/(pi^2 A) |

## Section IV-E (beyond the steady gyre, 2026 model, noise-free)

Sources: `experiments/unsteady_dg/outputs/trials_d14.csv` (time-periodic, 500 per start), `bent_d14.csv` (300 per start, with own-trench scoring), `bickley_d14.csv` (500 per start), `path_ratio_d14.csv` (50 per start); summary `experiments/draft14/outputs/beyond_summary.txt`.

| Number | Source detail |
|---|---|
| Table II rates | beyond_summary.txt (pooled success, retained, retained_own, reversal) |
| Table II ratios | path_ratio_d14 medians; steady from the eps = 0 rows (0.006, 0.023) |
| trench offsets 0.041, 0.078 (D), 0.085, 0.155 (s1); separatrix speeds 0.063, 0.157 | field-only, from the Nov 6 notes (Proposal_for_edits.pdf, Table N1; `trench_offset.py`, `fig_notes.py`); independent of the robot model |
| 70 vs 29 % (s1 own vs separatrix retention, delta 0.1) | bent_d14 |
| 97, 29 to 67, 9 (retention by ratio band) | Table II |
| 0.007 (Bickley s1 end offset from its trench) | beyond_summary.txt |
| 61 %, all losses within 0.1 of the crest (TP eps 0.1, s1) | beyond_summary.txt (1220 trials, 100 %) |
| 3.2 rho, 2.4 rho (S6, S7 at delta 0.1) | closed form: separatrix x = -0.1 cos(pi y) |
| D from S6/S7 ends at the corner saddle (1, -0.5) in every trial | beyond_summary.txt (timeouts, final x 0.994, y -0.4995) |


## Notes on the final text (2026-11-19)

- Citation numbers were renumbered by first appearance after these entries were written; the
  sources above are unaffected.
- Table II now pairs D/s1 in each cell; the numbers are the same as in the IV-E rows above.
- Table I merges the robot model into one row (g, L, tau = 0.86, 0.055, 0.02; ocean 1, 0, 0).
- Numbers carried over unchanged from Draft_13 (kappa 8.26 and 564, the closed forms, the
  Draft_13 Section V-B values) are not wrapped in `\rev{}`.
- Section IV-C numbers are now `\prelim{}` (blue) rather than `\rev{}`: reduced-size runs,
  to be replaced from `experiments/draft14/outputs/noise_summary_10k.txt` after
  `overnight_runs.txt` is run. `experiments/draft14/noise_summary.py` regenerates them.
