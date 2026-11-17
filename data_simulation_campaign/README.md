# Simulation campaign, 2026-11-16

Every simulation for the vector field paper's revision, re-run with the robot model fitted to the November 2026 hardware campaign. Decisions and the reasoning behind them are in `../SIM_RERUN_PLAN_2026-11-16.txt`.

## Settings

| | Value |
|---|---|
| Robot | gain 0.86, input delay 0.055 s, lag tau 0.02 s (alpha = e^(-dt/tau)), deadband 0.025 m/s, speed cap 0.3 m/s |
| Formation | equilateral triangle, side 0.30 m |
| Orbit gains | k_t 0.15 m/s, k_r 1 /s (the 2026-11 hardware) |
| Attraction | k = 1 /s, command clipped to 0.3 m/s per axis (the hardware's 0.3 m error clip) |
| Formation gains | 1 on p, q and beta (the hardware); theta_c not controlled |
| Orbit transient | 3 s excluded (analysis/hardware/outputs/transient_choice.md) |
| Sample period | 0.1 s unless a cell says otherwise |
| Reconstructed fields | the 2026 rebuilds only: `vortex_nn_2026`, `saddle_nn_2026` in vecsim (`vortex_predictors_2026/`, `saddle_predictors_2026/`). The names `vortex_nn`, `saddle_nn` still load the 2025 nets, for the v3 results |

## Running and resuming

From the repository root:

```
nohup caffeinate -is bash data_simulation_campaign/run_campaign.sh > data_simulation_campaign/logs/campaign.log 2>&1 &
```

- **Resume:** run the same command again. Finished steps are skipped (`logs/<step>.done`), and the sweep skips every cell already saved.
- **Laptop closed:** the runs pause while it sleeps and continue on wake. If they were killed (restart, logout), resume as above. At most the cells in progress are lost.
- **Progress:** `tail -f data_simulation_campaign/logs/campaign.log` (steps) and `tail -f data_simulation_campaign/hyperparameter_v4/progress.log` (sweep cells).
- **Expected time:** about 4 to 5 hours on this laptop (7 cores for the sweep, 1 for the tables).

## Layout

| Path | What it holds |
|---|---|
| `hyperparameter_v4/manifest.csv` | one row per cell: experiment, task, estimator, field, and every setting |
| `hyperparameter_v4/cells/<hash>.npz` | per-trial results of one cell (file name in the manifest) |
| `hyperparameter_v4/summary.csv` | one row per cell of summary metrics, written when the sweep finishes |

| `logs/` | one log per step, plus the campaign log |
| `load_campaign.py` | loader (below) |

The revision scripts write to `analysis/simulation/revision_metrics/` (set by REVISION_OUT in run_campaign.sh). The previous outputs stay in `trunk/.../experiments/revision_metrics/` and its `archive_2026-10-30_coulomb_side0.33/`.

## Reading the data

```python
import sys; sys.path.insert(0, "data_simulation_campaign")
import load_campaign as lc
C = lc.cells("E2delay")                                   # cells with settings, tau and summary metrics
T = lc.trials(C.iloc[0])                                  # trials of one cell
T = lc.trials_of("E3delay", field="vortex", delay=0.055)  # trials of all matching cells
```

Use the VF_Robot venv (`trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3`).

## Experiments in the sweep

| Experiment | Question | Varies |
|---|---|---|
| E1hw, E1ideal | Baseline accuracy on every field and estimator, with the hardware robot model and with ideal robots | field, estimator (fit, oracle, magnitude), task, r_d |
| E2tau, E2delay, E2gain, E2kt, E2kr, E2dt, E2vs | How closely the cluster can hold a requested orbit radius (0.1 mm to 10 m), and what sets the small-radius floor | one robot or controller setting at a time |
| E3delay, E3tau, E3gain, E3vs, E3dt | How fast and how accurately attraction converges, and where it goes unstable (k < 1/(g L) = 21 /s) | attraction gain k against delay, lag, gain, threshold, sample period |
| E4hw, E4ideal | Formation size from 0.01 to 1 m | side, field, estimator |
| E5hw, E5ideal, E5basin | Curved (nonlinear) saddles and the gyre: orbit accuracy and attraction basins | curvature, family, side |
| E6hw, E6ideal | Sensor noise | noise level, side, field |
| E7 | Random global design: every setting drawn per trial (delay per batch) | everything |
| E8a, E8o | Transients: attraction overshoot and orbit radial settling, with and without speed caps | (delay, tau) pairs, sample period, k or k_r |
| E9a, E9o | Raised speed caps | v_max, k, k_t, radial clamp |
| E10, E10hw | Low-speed rules (legacy, scaled, deadband, Coulomb), including the 157 hardware attraction starts | rule, sample period, threshold |
| E11, E11hw | Sensitivity of the headline results to the fitted robot parameters | delay 0.04 to 0.075 s, tau 0 to 0.03 s, gain 0.80 to 0.93, threshold 0.020 to 0.035 m/s |

Attraction trials start uniformly in a box around the critical point with random headings and run 60 s (10 s for hardware starts). Orbit trials start on the requested circle and are scored over 1.5 revolutions after a 20 s settling window.
