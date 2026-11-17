# Revision results report, 2026-11-16

This report covers every hardware and simulation result behind the vector field revision after the 2026-11-16 robot model refit:
gain 0.86, delay 0.055 s, lag 0.02 s, dead zone 0.025 m/s, cap 0.3 m/s. The sims use the hardware controller settings: formation side 0.30 m,
k = 1 with a 0.3 m/s per-axis clip, formation gains 1, and orbit gains k_t 0.15 and k_r 1. They also use only the 2026 reconstructed
fields. The orbit transient trim is 3 s throughout. File locations are in `analysis/README.md`.

Status at writing: the revision-metrics simulations, hardware scoring, paired runs, limit cycle and theory checks are
done, and their numbers are in the tex. The v4 hyperparameter sweep finished at 23:51 on 2026-11-16 (section 9).

## 1. Hardware attraction (2025 runs, re-scored)

Truth p* for each session is the zero of the field fitted to that session's own readings. Vortex 4R uses only the
Oct 2025 least-squares runs. No four-robot saddle attraction was run. The saddle 3R set fills the missing runs 43
and 64 with extra1 and extra2 (Extras/HANDLING_NOTES.txt). The model refit (D7) does not use the extras.

| | Vortex 3R | Saddle 3R | Saddle 3R, no extras | Vortex 4R |
|---|---|---|---|---|
| Runs | 80 | 80 | 78 | 79 |
| Bias (m) | 0.007 | 0.002 | 0.001 | 0.024 |
| sigma (m) | 0.036 | 0.025 | 0.024 | 0.014 |
| Mean final error ± sd (m) | 0.034 ± 0.014 | 0.023 ± 0.009 | 0.023 ± 0.008 | 0.026 ± 0.009 |
| Median [p95] (m) | 0.035 [0.058] | 0.022 [0.038] | 0.022 [0.037] | 0.025 [0.043] |
| Localization error (m) | 0.015 | 0.014 | 0.014 | 0.025 |
| T_conv, median (s) | 3.9 | 3.4 | 3.4 | 2.8 |
| E_u (m²/s) | 0.187 | 0.205 | 0.205 | 0.218 |
| Success 3/4/5 cm (%) | 39/69/91 | 80/98/99 | 81/99/100 | 65/86/100 |
| p* inside final triangle (%) | 100 | 100 | 100 | 100 |
| Final speed, median (m/s) | 0.0013 | 0.0011 | 0.0011 | 0.0009 |

No trial left the map. With the extras in, saddle success at 5 cm drops from 100 to 99%: one of the two refilled
runs ends beyond 5 cm. The vortex 4R bias of 0.024 m is larger than its spread, so the four-robot cluster comes to rest
consistently to one side of the fitted zero. The bias matches its localization error (0.025 m), which points to the estimate.

## 2. Hardware orbits (2026 runs, ten per radius, 3 s trim)

Radial bias ± between-run spread s_b, then within-run spread s_w (m):

| r_d | Vortex 3R | Vortex 4R | Saddle 3R | Saddle 4R |
|---|---|---|---|---|
| 0.01 | +0.060 ± 0.011 | +0.057 ± 0.011 | +0.066 ± 0.015 | +0.057 ± 0.014 |
| 0.10 | +0.028 ± 0.002 | +0.028 ± 0.007 | +0.036 ± 0.011 | +0.027 ± 0.006 |
| 0.20 | +0.007 ± 0.014 | +0.036 ± 0.025 | −0.001 ± 0.005 | +0.031 ± 0.025 |
| 0.30 | +0.043 ± 0.027 | +0.098 ± 0.020 | +0.033 ± 0.016 | +0.040 ± 0.019 |
| 0.40 | +0.073 ± 0.023 | +0.098 ± 0.008 | +0.032 ± 0.010 | −0.013 ± 0.073 |
| 0.50 | +0.020 ± 0.009 | -- | +0.005 ± 0.011 | -- |
| s_w range | 0.014 to 0.076 | 0.017 to 0.057 | 0.012 to 0.083 | 0.022 to 0.076 |

- The practical range, |bias| ≤ 0.25 r_d, starts at r_d = 0.20 m for 3R in both fields and for saddle 4R. Vortex 4R
  passes at 0.20 and 0.40 m but fails at 0.10 and 0.30 m.
- s_w exceeds s_b from r_d = 0.10 m in every column.
- No orbit run left the map.
- The controller-only bias at r_d = 0.01 m (radius about p-hat*) is 0.071 m in the vortex 3R, against a true bias of 0.060 m.
- Run length is 20 to 30 s. From r_d = 0.30 m the 3R runs complete one full lap. At 0.01 m they complete up to eight. One vortex 4R
  run at 0.40 m completes no full lap.
- Classification of J-hat on every cycle: saddle 3R 99.95%, saddle 4R 99.6%. Vortex 3R: 99.9% rotational, 44.7% as a
  center. Vortex 4R: 99.6% and 46.5%. The center lies on the boundary between the two spiral types.
- Parked noise (the extras' last 5 s): the reading changes by 2.1e-4 m/s per cycle on average, and the hue sd is 0.14 deg.

## 3. Orbit transient trim (R1.4)

The trim is 3 s, down from 10 s. Hardware runs show no start-up transient because they start near their circle. The 23 to 44 mm
level in `hardware/outputs/transient_choice.md` is the radius swinging once per lap. In simulation from 5 cm off the circle,
the radial transient falls to 2 mm by 3 s. Mean hardware bias per group, by trim (m):

| | 2 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| Vortex 3R | 0.0387 | 0.0387 | 0.0385 | 0.0382 |
| Vortex 4R | 0.0633 | 0.0635 | 0.0644 | 0.0642 |
| Saddle 3R | 0.0281 | 0.0284 | 0.0292 | 0.0283 |
| Saddle 4R | 0.0284 | 0.0284 | 0.0277 | 0.0276 |

Moving the trim between 2 and 10 s changes the group bias by at most 0.9 mm. The settling analysis (trims of 0, 2, 5 and 10 s,
`settling_2026.csv`) puts the largest change at 4 mm.

## 4. Simulation, Table II (3R, matched settings, 1000 attraction and 100 orbit runs per field)

| | Analytical (6) | Recon. vortex | Recon. saddle | Gyre | Irrot. saddle | Rot. saddle |
|---|---|---|---|---|---|---|
| Mean final error (m) | 0.0226 | 0.0201 | 0.0247 | 0.0238 | 0.0221 | 0.0234 |
| Localization error (m) | < 0.0001 | 0.0101 | 0.0134 | 0.0043 | 0.0049 | 0.0026 |
| Success 3/4/5 cm (%) | 100 | 91/100/100 | 68/100/100 | 99/100/100 | 100 | 100 |
| Orbit bias at r_d 0.01 (m) | +0.042 | +0.039 | +0.034 | +0.044 | +0.042 | +0.043 |
| Orbit bias at r_d 0.50 (m) | +0.004 | +0.027 | +0.019 | -- | -- | -- |
| Classification, noise 0.02 m/s (%) | 86 to 100 | 52 | 100 | 83 | 100 | 100 |

Compared with the submitted model (Coulomb, side 0.33, k_t 0.2, k_r 0.5), the small-radius orbit bias drops from about
0.13 m to 0.04 m. The analytical fields reach the 0.25 r_d criterion from r_d = 0.10 m instead of 0.30 m. On the 2026
nets, every reconstructed-field trial ends within 5 cm, but noise-free vortex classification falls to 73% (91% in the previous table, which used the 2025 net
and the old model). White noise up to 0.03 m/s changes the reconstructed fields' final error by at most 3.0 mm.

## 5. V2S comparison (Table IV)

Three claims in Sec. VI-B changed (queued item O):
- V2S alone no longer loses orbits on the linear fields. It drifts outward, with a radial error of 0.14 m at r_d = 0.3 m.
- V2S with its own speed estimate now orbits inside r_d, with an RMSE of 0.075 m against 0.018 m for the Jacobian at
  r_d = 0.1 m. The old text said its RMSE was lower.
- With the oracle on the anisotropic node, 88%, 46% and 30% of orbits are lost at r_d 0.1, 0.3 and 0.5 m. The old text said most were lost.

The numbers that kept their sentence are in the tex: overshoot 0.003 m, effort ratio 1.6, oracle within 0.004 m, final speed
0.21 m/s, node success 6%, node median miss 0.64 m, estimate error 0.50 m, gyre agreement 97%.

## 6. Hardware against simulation, run by run (paired runs from the hardware starts)

Attraction, mean final error (m):

| | Hardware | Linear sim | Recon. sim |
|---|---|---|---|
| Vortex 3R | 0.034 | 0.023 | 0.019 |
| Saddle 3R | 0.023 | 0.023 | 0.024 |
| Vortex 4R | 0.026 | 0.022 | 0.018 |

The saddle matches. The hardware vortex ends 0.4 to 1.5 cm farther out than the sims. Orbit bias at r_d = 0.01 m runs 1.4 to 1.8 cm
above the reconstructed sim in every column: 0.057 to 0.066 m on hardware against 0.039 to 0.052 m in the reconstructed sim. The
rest of the gap varies by radius. The 2025 vortex 3R attraction formation measured 0.26 m per side against 0.30 m in the sims, which may
account for part of the vortex gap. The paired vortex attraction runs could be repeated at 0.26 m. Full rows are in `simulation/paired/paired_orbits_summary.csv`.

## 7. Theory (`simulation/theory_check.py`)

- Attraction stability: k < 1/(gL) = 21.1 /s. The eigenvalues of the full model give 21.14. Simulation is stable at k = 21 and unstable
  at 22. The submitted lag-only model gave 113 /s. At k = 1 the poles are 0.91 and 0.05.
- Orbit floor: r* = (r_d + sqrt(r_d² + 4C))/2 with C = T V²/(g k_r), V = g k_t, T = L + tau + dt/2 = 0.125 s. It is within
  3.0 mm of simulation for r_d from 0 to 0.5 m. A fitted T = 0.111 s brings that to 0.6 mm.
- r_min (r_d = 0): 0.047 m at the defaults, against 0.049 m from the closed form. It is 0.031 m with k_t = 0.1, 0.032 m without delay and lag, and 0.034 m with k_r = 2.

## 8. Limit cycle (vortex, r_d = 0.10 m)

Hardware: ten 3R runs start inside the cycle (4) and outside it (6), from 0.057 to 0.156 m. All converge to one closed
curve with mean radius 0.127 m and a peak-to-peak swing of 0.043 m. The across-run spread is 2.9 mm. Simulation on the reconstructed vortex
gives a cycle of mean radius 0.111 m and swing 0.029 m. Starts from inside and outside meet within 0.03 mm. The two
curves agree in size but only modestly in shape (correlation 0.38, median gap 13 mm). The figure is
`limit_cycle/figures/fig_limit_cycle.pdf`. It is proposed, not inserted (queued item P).

## 9. v4 sweep (finished 2026-11-16 23:51, 8635 cells; `simulation/v4_numbers.py`)

- Noise covariance check (E6ideal attraction, linear fields, ideal robot, lambda R / sigma_n >= 3, converged): 66
  cells, measured over predicted mean error has median 1.003 and range 0.98 to 1.05, so the prediction is within 5% (95%
  of cells within 4%). The tex now reads 5%. In v3 it read 4%.
- Orbit closed form, T = L + tau + dt/2, across the E2 sweeps on the linear fields (gains, delay, lag, sample period,
  dead zone; 1594 cells kept, 2 excluded where k_t sat inside the dead zone): median gap 1.2 mm, 90th percentile 4.5 mm,
  largest 12.9 mm at the extreme k_t and k_r settings. The tex now reads a median of 1.2 mm.
- E7 (random global design) stores per-trial results, so `summary.csv` omits it, and no paper number uses it.

## 10. What changed in the paper and letter

- Applied (`paper/apply_paper_edits.py`, map in `paper/numbers_map.csv`): 85 numeric edits marked `\rev{}`; Table II,
  the hardware table (now 3R and 4R columns) and the V2S table regenerated; five figures replaced (hardware
  trajectories and distance, reconstruction, convergence band, nonlinear triangles, V2S effort); protocol numbers in
  Sec. IV (side 0.30, k_t 0.15, k_r 1, 0.3 m/s per axis, 3 s trim, 300 steps). Both captions corrected to match their figures.
  Paper compiles with no errors and no new warnings. Survival is 60.1% (target 60%).
- Letter (`paper/apply_letter_edits.py`): 12 numeric edits. `check_letter_quotes.py`: 46 quotes, 0 mismatches.
- Prose batch (`paper/queued_prose_batch.md`), approved and applied 2026-11-16 by `paper/apply_prose_batch.py`; item F and
  the sweep sentence of item M followed after the v4 sweep. Before that it was queued for approval: robot model text and equations (III-A, III-D), seven
  sentences whose claims changed, the test-plan wording, the abstract 4R sentence, the limit-cycle paragraph, and five
  letter items.
- No `\syn{}` or `\open{}` markers remain in the tex. Survival is now 59.3% (target 60%, floor 55%). The abstract runs about 270 words.
