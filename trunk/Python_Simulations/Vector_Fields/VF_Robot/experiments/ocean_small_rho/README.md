# Smaller formations on Santa Barbara (Fig. 9), 2026-10-12/13

Question: does a smaller formation, with a different start and speed, follow
the main FTLE ridge in Fig. 9 better than the current operating point
(rho 5.04 km)? The old search (`../unused/ocean_param_mc.py`) never sampled
below scale 0.6 (rho 3.2 km) and scored against the pre-isotropic-map corridor.

Nothing here touches the paper or `figures/`. Renders are in `review/`, each
a byte-for-byte run of `main_ocean_hfr_2km_progression.py` with the constants
of `main_ocean_hfr_2km_shared_start.py` overridden (`review/paper_current/`
reproduces the installed Fig. 9 pixel for pixel). `review/<name>_ridge.png`
adds the extracted ridge as a dotted cyan line on panel (d).

## Method

```
ridge <- least-cost path through the 24-h FTLE field, channel entrance to island   (main_ridge.npy, 59.5 km)
repeat for 5.75 h, 8 workers:                                                       (529,238 trials)
    draw start within 2.5 km of ridge arc 3..30 km, scale 0.30..1.10 (rho 1.6..5.9 km),
         heading 0..360, gain k 1..3, speed cap c 0.020..0.060
    run D and s1 trackers 168 steps, stop at landfall (any robot over land)
    per tracker: ride = longest unbroken downstream stretch within 1.5 km of the ridge (km of ridge)
                 dev  = mean centroid distance to the ridge over the run (km)
rank = max over trackers of (59.5 - ride + 5 dev); a landfall counts only south of 34.10 N
jitter top 50 distinct: start within 1 km, heading +-15 deg, scale, k, c +-5%, 100 runs each
```

Two earlier scores were dropped after rendering their leaders (`v0_loose_score/`,
`v1_cov_score/`). A third run (`v2_contaminated_D/`) was invalid because of the
latch bug below.

## Results

| Run | rho km | k | c | k c | start | ride D / s1 km | dev D / s1 km | landfall D / s1 |
|---|---|---|---|---|---|---|---|---|
| current Fig. 9 | 5.04 | 1.94 | 0.0333 | 0.065 | 34.3849, -120.4055 | 8.2 / 3.0 | 2.07 / 2.77 | 23.0 h / at sea |
| **C_both_land** | 3.32 | 2.45 | 0.0336 | 0.082 | 34.2785, -120.4476 | 18.5 / 18.3 | 0.57 / 1.41 | 21.2 h / 22.5 h |
| A_tight | 1.72 | 1.45 | 0.0244 | 0.035 | 34.2542, -120.4400 | 16.8 / 15.6 | 0.80 / 0.53 | at sea / at sea |
| D_both_land_tight | 3.48 | 1.99 | 0.0415 | 0.083 | 34.1952, -120.4339 | 13.8 / 16.6 | 0.75 / 0.74 | 20.8 h / 12.8 h |
| B_long | 3.77 | 2.89 | 0.0324 | 0.094 | 34.2389, -120.4598 | 26.8 / 17.3 | 0.84 / 1.04 | 27.0 h / at sea |

Full parameters (heading, exact scale) are in `ridge_mc.csv` (rows 528473,
65086, 12222, 234892) and in the render log line of each `review/` folder.

- C: both trackers ride the red ridge together from 34.28 N through the
  bifurcation, then split, D down the southeast branch and s1 down the
  southwest one, and both reach land.
- A: tightest to the ridge, about a third of the current size, but slow, so
  it ends at 34.11 N, before the bifurcation.
- D: tight, but starts at 34.20 N, below most of the straight ridge.
- B: s1 sits near the start for 21 h.

Smaller formations make up most of the trade-off front between ride and dev:
rho 1.6 to 3.8 km, median 2.95 km among the top 300. The best rank in each
rho bin is flat, though (46.7 to 50.7 from 1.5 to 6 km), so a smaller size
helps but does not decide the result.

## Every good run is fragile, including the current one

Under the jitter above, no candidate holds up. The median s1 ride falls to 1
to 5 km, and the best candidate beats the current Fig. 9 in 18% of its
jittered runs (C in 2%). The current settings are no better. Jittered the
same way (`ridge_jitter_paper_current.csv`), D still reaches the island in 66%
of runs (median 23.7 h), but 92% of s1 runs turn north and hit the mainland
at about 7 h. From most starts on the upper ridge the flow carries both
trackers northwest to the mainland, not down the channel. Any shared-start
figure on this field is a selected run.

## Latch bug in PentagonCluster.reset (fixed 2026-10-13)

`separatrix_logic_c_step` latches `cluster._logicc_captured`, added in the
2026-10-09 rewrite. `PentagonCluster.reset` clears only `_oecs_prev_tangent`,
`_oecs_banded`, and `_oecs_captured`, so a reused cluster starts every D run
after the first capture already captured. No paper figure is affected: Figs.
5 to 8, `_mc_common.run_trial`, and the primitive11 screens all build a fresh
cluster per run, and Fig. 9 runs D first. `_ocean_run_common.run_traj` reuses
clusters, and so did this search until it cleared the flag itself
(`Trial.run`). Fixed 2026-10-13 at Chris's request: `reset` now clears
`_logicc_captured` too. A D run after a latched one now matches a fresh run
exactly, and `tests/test_oecs_separatrix.py` passes. No figure is stale.

## Files

| File | What |
|---|---|
| `extract_main_ridge.py`, `main_ridge.npy` | ridge extraction and polyline |
| `ocean_ridge_mc.py`, `ridge_mc.csv`, `ridge_mc.log` | the search |
| `ridge_jitter.py`, `ridge_jitter*.csv`, `ridge_jitter.log` | ranking, jitter and refinement |
| `jitter_paper_current.py` | the same jitter on the current Fig. 9 point |
| `render_candidates.py`, `review/` | Fig. 9 renders |

## Annealing on the upper ridge (2026-10-13)

Chris wanted the start on the upper ridge (120.28 W along the mainland coast
down to about 34.39 N), rho no smaller than the grid allows, and rides of 20 h
or more. `ocean_ridge_anneal.py` (pseudocode in its docstring) runs simulated
annealing with random restarts on the extended ridge `main_ridge_ne.npy`
(69.1 km), rho 2.0 to 4.0 km (a 4 km footprint spans two cells of the
2.0 x 1.9 km HFR grid). Cost adds a lost-time charge for mainland landfall and
2 km per hour for reaching the island before 24 h. Two runs, pooled:
`ridge_anneal_v1_nodur.csv` (36 min, no duration term) and `ridge_anneal.csv`
(3.4 h, 265 chains, seeded from the first), 123,148 trials.
Distinct finalists are in `anneal_finalists.csv`.

| Run | rho km | k | c | heading | start | ride D / s1 km | dev D / s1 km | landfall |
|---|---|---|---|---|---|---|---|---|
| **anneal_best** | 2.35 | 2.467 | 0.0489 | 204.0 | 34.3910, -120.4150 | 24.0 / 20.3 | 0.82 / 1.24 | neither, 28 h |
| anneal_28h | 2.12 | 2.452 | 0.0597 | 205.5 | 34.4130, -120.4062 | 15.8 / 20.6 | 1.39 / 2.21 | neither, 28 h |
| anneal_check | 2.26 | 2.573 | 0.0516 | 204.0 | 34.3949, -120.4148 | 13.4 / 21.1 | 1.56 / 1.83 | D 19.7 h, s1 16.0 h |

anneal_best: D rides on the ridge from the start through the bifurcation and
down the southeast branch to near the island at 28 h. s1 rides to about
34.15 N and holds by the southwest branch. Panel (d) of these, C, and the
current figure side by side is `review/contact_sheet_28h.png`
(`contact_sheet.py`). Its k c is 0.121, about 1.9 times the current 0.065.
