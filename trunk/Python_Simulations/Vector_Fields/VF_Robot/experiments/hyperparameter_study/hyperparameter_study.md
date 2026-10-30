# Hyperparameter study for Section IV (3-robot cluster, simulation)

2026-10-29, version 3. Simulation only, run after the 2026-10-28 formation sign fix, with the robot model committed on 2026-10-29 (first-order lag with tau = 0.28 s and alpha recomputed from the sample period, Coulomb friction v_c = 0.025 m/s, measured in a dedicated testbed experiment, with equal static and kinetic levels, 0.4 m/s per-robot cap). Version 2 used a dead zone on the command and a 0.3 m/s cap. This version also reads the hardware runs in metres (version 2 read map units, which understated the hardware rest distance by a factor of 1.68), adds a sensitivity study on tau and kinetic friction (E11), and compares the model with the hardware orbit runs. Numbers are for review and none are in the paper.

## What was run

All runs use a vectorized copy of the paper's simulator (`vecsim.py`). It reproduces `OmniCluster`, `Omnibot` and the Eq. 10 and Eq. 11 laws to within 1e-11 m of the reference code over 400 steps, to 4e-7 m on the MLP fields, and to 7e-12 m under every stiction rule including Coulomb friction with kinetic below static (`validate_vecsim.py`). Version 3 ran 5,403 designed configurations plus 1,000 random batches, 28.8 million trials, in 104 minutes on 7 cores, and E11 ran 1.7 million more.

```
for each configuration (field, estimator, task, side, tau, dt, v_c, v_max, k, k_t, k_r, r_d, sigma_n, stiction rule):
    repeat N times:
        draw start (attraction: uniform in a 1 m box; orbit: on the commanded circle) and heading
        run 60 s (attraction) or w0 + W s (orbit)
        record per-trial metrics
    summarize, and compute the theory prediction for the same configuration
```

| Exp. | Question (reviewer) | Factors and levels | Trials |
|---|---|---|---|
| E1 | Per-trial errors, convergence and failure (AE, R1.6, R3) | 10 fields x 3 estimators x {hardware, non-idealities off}, attraction, and orbit at r_d 0.01 to 0.5 | 9.6M (10^5 per attraction cell, 10^4 per orbit cell) |
| E2 | Achievable orbit radius, actuator lag, gains (R1.4, R1.2) | r_d 1e-4 to 10 m (14 levels) x alpha 0 to 0.95, k_t 0.05 to 0.5, k_r 0.1 to 5, dt 0.02 to 0.2 at fixed tau, v_c 0 to 0.1 | 1.5M |
| E3 | Attraction gain, lag, friction (AE, R1.2) | k 0.1 to 100 x alpha, v_c x alpha, dt x k | 2.5M |
| E4 | Formation size (R1.1, R3) | side 0.01 to 1 m x 6 fields x {fit, magnitude} | 2.1M |
| E5 | Curvature and multiple critical points (R1.1, R3) | eps -4 to 4, irrotational and rotational saddles, side, r_d, basins | 0.8M |
| E6 | Temporal noise versus fixed distortion (R1.3, R1.6) | sigma_n 0 to 0.1 x side x 6 linear fields, plus the 2 reconstructed fields | 1.6M |
| E7 | Global check | 1,000 batches of 10^4 trials, every parameter drawn at random per trial | 10M |
| E8 | Overshoot and undershoot | tau 0 to 1 s x dt 0.01 to 0.2 x k 0.3 to 30 x caps on/off x v_c {0, 0.025}; orbit transient from r0 0.05 and 0.6 x k_r 0.25 to 5 | 0.31M |
| E9 | Raised speed limits | v_max 0.1, 0.3, 1, 3 m/s and none x alpha x k 1 to 300; orbit v_max x k_t 0.1 to 2 x r_d x radial clamp | 0.16M |
| E10 | Stiction rule | 4 rules x dt 0.01 to 0.2 at tau 0.28 x v_c; and the 157 hardware start positions x 50 headings, 10 s runs | 0.27M |
| E11 | Robot model sensitivity | tau {0.20, 0.28} x v_c {0.025, 0.035} x v_k {v_c, v_c / 2, 0}; attraction, orbit, radius sweep and hardware starts on 5 fields | 1.7M |

The 10 fields are the paper's six linear fields, a single double-gyre cell (center target, saddles at the corners), a curved saddle (v = -y + eps y^3 on the inflow axis, irrotational or rotational form), and the two MLP fields reconstructed from testbed data. The estimators are the plane fit (paper), the true critical point (an oracle that isolates the controller), and a magnitude-only baseline that senses |v| and feeds the same laws. Attraction counts as converged when the final distance to the true critical point is under 5 cm. Orbit statistics start after w0 = max(20 s, 5/k_r, 10 tau) and run for 1.5 revolutions (40 to 300 s). A trial fails if it escapes, its formation side error exceeds 10%, or it completes under a quarter revolution.

## 1. The robot model is a lag with Coulomb friction

Each robot is a first-order lag with Coulomb friction behind it. While it moves, tau dv/dt = u - v - v_c v/|v| with |v| <= v_max. A parked robot stays parked until |u| > v_c. Held constant over each sample, with alpha = exp(-dt / tau),

```
v-      = alpha v[n] + (1 - alpha) u[n]
v[n+1]  = 0                                    if |v-| <= (1 - alpha) v_c
        = v- (1 - (1 - alpha) v_c / |v-|)      otherwise,   then the cap |v| <= v_max
```

which is exact at the sample instants whenever u is parallel to v. Alpha is a property of the sample period and tau is the physical constant (alpha = 0.700 at dt = 0.1 s and 0.965 at dt = 0.01 s). Friction enters through the same (1 - alpha) factor as the command, so its coefficient follows the sampling rate and the model behaves the same at every dt. A constant command drives the speed to |u| - v_c, and a robot moving straight at the target parks once k |e| <= v_c. The friction level v_c = 0.025 m/s was measured in a dedicated testbed experiment and is used throughout.

Four discrete rules were compared (Table 1). The legacy rule zeroes the lagged velocity below v_s, so its start threshold on the command is v_s / (1 - alpha), which grows as dt shrinks until nothing moves. The scaled rule starts correctly but stops only below (1 - alpha) v_s and creeps in at small dt. The dead zone (version 2) switches the command off at |e| = v_s / k and lets the robot coast on its lag, so it rests about v_s / (2k) short. Coulomb friction keeps opposing the motion until the robot stops, so it rests at v_c / k on the approach side, twice as far.

**Table 1.** Stiction rules at tau = 0.28 s, v_c = 0.025 m/s, k = 1, vortex, 1,000 trials per cell. Mean final distance to the critical point in mm, all trials.

| rule | moves from rest when | dt 0.01 | dt 0.02 | dt 0.05 | dt 0.1 | dt 0.2 | orbit at k_t 0.05 starts | hardware starts, 10 s |
|---|---|---|---|---|---|---|---|---|
| legacy | abs(u) >= v_s / (1 - alpha) | 380.1 | 104.8 | 20.5 | 13.4 | 12.4 | only at dt 0.2 | 12.0 |
| scaled | abs(u) >= v_s | 0.3 | 0.6 | 1.6 | 3.3 | 5.8 | at every dt | 3.1 |
| dead zone (v2) | abs(u) >= v_s | 12.3 | 12.3 | 12.3 | 12.1 | 11.8 | at every dt | 11.5 |
| Coulomb (v3) | abs(u) > v_c | 24.9 | 24.9 | 25.0 | 24.9 | 24.9 | at every dt | 24.9 |
| hardware | | | | | | | | 23.4 |

The 157 hardware runs (79 vortex, 78 saddle, 10 s each) end 23.4 mm from the critical point on average (vortex 23.5, saddle 23.3, median 23.3, 95th percentile 40.2). Re-running the same starts with 50 random headings each, Coulomb friction gives 24.9 mm on the linear fields and 27.8 mm on the reconstructed fields, against 11.5 mm for the dead zone. The saddle runs check the stop side. Of the hardware saddle runs 94% stop on the approach side, 18.2 mm back along the approach line on average, and the model stops every trial on the approach side, 24.9 mm back. The hardware vortex runs carry a fixed 21 mm offset in one direction (-6.0, +19.9 mm). The reconstructed vortex field also puts the rest point about 21 mm off, but toward (+17, -13) mm, 145 degrees away. The hardware and reconstruction frames probably differ by a turn, since the map was printed in the x-z plane, which can mirror it, and OptiTrack can add a rotation. Of the eight 90-degree rotations and mirrors, only the 180-degree rotation that `combined_trajectory_plotter.m` applies to the vortex runs brings both fields closer, leaving 35 degrees and 13 mm between the hardware and simulated offsets for the vortex and 42 degrees and 8 mm for the saddle (`results_v3_sens/frames.csv`). The exact frame chain is not settled here. The hardware spread is wider than any single-v_c model (Fig. 7c).

`OmniCluster` now defaults to this model (commit 7e91436). Passing `stiction_model="legacy"`, `momentum_alpha=0.7` and `max_velocity=0.3` reproduces runs made before 2026-10-29.

![Figure 7. (a) Mean final distance against sample period at fixed tau = 0.28 s, filled k = 1, hollow k = 10. The dead zone and Coulomb rules are flat at k = 1. (b) Share of orbit trials that start. The legacy rule fails at small dt. (c) Final-distance distributions from the 157 hardware start positions (in metres), hardware against each rule.](figures_v3/fig7_stiction_models.png)

## 2. Overshoot and undershoot as functions of alpha, sample rate and gain

With an exact estimate the cluster moves on a straight line, so attraction reduces to a sampled second-order loop. Writing everything in tau and dt

```
alpha  = exp(-dt / tau)
T_eff  = T_lag + dt/2 = (dt/2) coth(dt / (2 tau)),   T_lag = alpha dt / (1 - alpha)      -> tau as dt -> 0
poles    z^2 - (1 + alpha - dt (1 - alpha) k) z + alpha = 0
k_c    = (1 - sqrt alpha) / (dt (1 + sqrt alpha)) = tanh(dt / (4 tau)) / dt     -> 1 / (4 tau)     overshoot begins
k_max  = 2 (1 + alpha) / (dt (1 - alpha))      = 4 T_eff / dt^2              -> infinity       unstable beyond
overshoot ratio, dt -> 0     M = exp(-pi zeta / sqrt(1 - zeta^2)),   zeta = 1 / (2 sqrt(k T_eff))
```

The exact overshoot ratio comes from iterating the two-pole loop. In the continuous limit overshoot depends on k tau alone, and the sample period adds a small correction through T_eff and the discrete poles. At fixed tau the sample rate hardly matters (k = 3 and tau = 0.28 s give 12.9% at dt = 0.01 and 14.4% at dt = 0.2, Fig. 8b). At fixed alpha it matters a great deal, because alpha = 0.7 means tau = 0.28 s at dt = 0.1 and tau = 0.028 s at dt = 0.01. With caps and friction off, the simulator matches the discrete prediction to 0.000 percentage points in all 121 stable cells and diverges in all 125 cells above k_max. At the paper's settings k_c = 0.890 /s and k_max = 113 /s, so k = 1 sits just past the onset with a predicted overshoot of 0.013%.

**Table 2.** Overshoot in % of the start distance at k = 3, caps and friction off, 500 trials per cell.

| tau (s) | dt 0.01 | dt 0.02 | dt 0.05 | dt 0.1 | dt 0.2 |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.5 |
| 0.28 | 12.9 | 12.9 | 13.0 | 13.3 | 14.4 |
| 0.5 | 24.5 | 24.5 | 24.6 | 24.6 | 25.1 |
| 1.0 | 38.8 | 38.8 | 38.8 | 38.9 | 39.6 |

Friction turns the end of the approach into an undershoot. While the robot moves toward the target, Coulomb friction is a constant along the motion, so the moving loop is the frictionless loop aimed at a shifted target b = v_c / k on the approach side. The robot stops at each velocity zero and parks there if it lies within b of the target. Each swing starts from rest, so the swing map and its first step are

```
x'      = sign(x) b - M (x - sign(x) b)        for |x| > b,   M = sampled overshoot ratio
rest    = first x with |x| <= b                 one swing: v_c / k - M (d0 - v_c / k) short of the target
```

On an overdamped approach (M = 0) the robot creeps to v_c / k exactly, whatever tau is. At the paper's settings M = 0.013%, so the rest point is 24.9 mm short against v_c / k = 25 mm. With caps off the swing map matches the simulated median rest point to within 0.1 mm for k up to 3 and 0.6 mm at k = 10 and 30, at every tau from 0 to 1 s. A longer tau or a higher gain overshoots the shifted target and parks closer, or past the target (tau = 1 s at k = 1 rests 2.0 cm past, Fig. 8c).

The orbit's radial loop is the same sampled loop with k replaced by k_r, so the same formula predicts radial overshoot when an orbit starts off r*. Over 147 caps-off cells with |r0 - r*| >= 0.1 m the median gap is 0.0 percentage points, the 90th percentile is 5.1, and the correlation is 0.96. At the paper's k_r = 0.5 < k_c there is none. Starting outside with a long lag, the radius first dips below r* (11 mm at tau = 1 s and k_r = 0.25), because the outward push that sets r* builds up only as the robots reach speed. Friction lowers that push, so the dip is smaller than version 2's 46 mm.

![Figure 8. (a) Overshoot against k for four sample periods at tau = 0.28 s, caps and friction off. Markers are simulation, solid lines the discrete theory, dashed the continuous limit, dotted the k_max of dt = 0.2. (b) Overshoot at k = 3 over tau and dt. (c) Rest point along the approach line with v_c = 0.025 and dt = 0.1, caps off, negative is short of the target. Dotted lines are the Coulomb swing map. (d) Orbit radial overshoot against the sampled-loop prediction with k replaced by k_r.](figures_v3/fig8_overshoot_undershoot.png)

## 3. Raising the speed limits

The robots' cap is not what limits accuracy. Raising v_max from 0.1 to 3 m/s, or removing it, changes the attraction error little below k_max (k = 1 gives 25.0 mm at every cap, k = 30 gives 0.16 to 0.41 mm). Above k_max the cap decides the outcome. Without one the loop diverges at k = 300. With one it settles into a limit cycle whose final distance is v_max dt / 2 (ratio median 1.02, range 0.96 to 1.09, n = 16), so the error grows with the cap, 5.1, 15.2, 53 and 153 mm at v_max = 0.1, 0.3, 1 and 3 m/s. The same saturation explains why random-design convergence falls only gradually past k_max, to 70% at four to ten times k_max (Fig. 6a).

For orbits the closed form of Section 4 holds with the cap raised up to k_t = 2 m/s (radius error 1.071 m simulated against 1.069 m at r_d = 1). The cap also scales the radial correction, and the closed form includes that, so with the 0.3 m/s cap and k_t from 0.5 to 2 m/s it gives 0.054 to 0.093 m against 0.054 to 0.088 m simulated, where version 2's formula, which only capped the speed, gave 0.049 m. Without a cap the error grows as k_t squared, so a higher cap buys a faster orbit, not a more accurate one.

![Figure 9. (a) Median final error against k for five speed caps, alpha = 0.7, v_c = 0.025. The dashed line is k_max. (b) Limit-cycle size above k_max against v_max dt / 2. (c) Orbit radius error against k_t at r_d = 1 m with raised caps. Lines are the closed form of Section 4.](figures_v3/fig9_speed_limits.png)

## 4. The orbit radius error has a closed form

The tangential command rotates at V / r. The robot lag delays it, which turns part of the command outward, and each straight sampled step adds (V dt)^2 / (2r). Friction takes v_c off the speed the robots reach. In the sampled update friction and the speed cap both scale the whole vector alpha v + (1 - alpha) u, by a factor g each step, which shortens the lag (the pole moves from alpha to g alpha) and passes only G = V / k_t of the radial command. The radial loop balances at

```
V = min(k_t - v_c, v_max),   g = V / (alpha V + (1 - alpha) k_t),   G = V / k_t,   T_lag(a) = a dt / (1 - a)
r* = (r_d + sqrt(r_d^2 + 4C)) / 2,     C = V^2 (T_lag(g alpha) + dt/2) / (G k_r)
error r* - r_d  ->  C / r_d  for large r_d,     ->  r_min = sqrt(C)  as r_d -> 0
```

Without friction or cap g = G = 1 and C = V^2 T_eff / k_r, version 2's form. As dt -> 0 it becomes C = V^2 tau / k_r, the continuous balance, where the command is V + v_c along the motion plus tau Omega V across it, and matching that to k_t and k_r (r_d - r) gives k_t = V + v_c and k_r (r - r_d) = tau V^2 / r. Keeping the frictionless lag instead misses by up to 4 mm on the hardware-like cells and by 82% (95th percentile, relative) on random trials with friction. The estimate does not enter, so the result is the same for every linear field and for the oracle in every field. At the paper's gains the simulation gives +0.129, 0.093, 0.067, 0.051, 0.040 and 0.033 m at r_d = 0.01 to 0.5 and the formula +0.127, 0.092, 0.066, 0.051, 0.040 and 0.033. Version 2, without friction and with the 0.3 m/s cap, gave +0.146 to +0.042, which is the paper's current Table V. Across 1,971 designed cells inside the validity region (r_d from 1e-4 to 10 m) the median gap between formula and simulation is 0.03 mm and the largest is 5.3 mm (0.088 against 0.093 m, the capped orbit at k_t = 2 m/s and r_d = 1 m). Across 16,991 random trials inside the region it holds within 4.3% at the 95th percentile. The region is a lag phase (V / r) T_eff under 0.5 rad and k_r times the error under the 0.3 m/s radial clamp. Outside it the orbit ends up larger than the formula says (Fig. 1c, gray). Since T_eff is close to tau, C is about V^2 tau / k_r and the sample rate barely matters (r_d = 0.3 gives 0.049, 0.050, 0.051 and 0.053 m for dt = 0.02, 0.05, 0.1 and 0.2 s). An orbit starts if and only if k_t > v_c, which held in all 3,592 orbit cells and all 1,087,573 random orbit trials.

**Table 3.** Smallest orbit the controller can hold (closed form, dt = 0.1).

| settings | C (m^2) | r_min (m) | r_d for error <= 5 cm | r_d for error <= 1 cm |
|---|---|---|---|---|
| hardware (alpha 0.70, k_t 0.2, k_r 0.5, v_c 0.025) | 0.0178 | 0.133 | 0.31 | 1.77 |
| version 2 model (no friction, 0.3 m/s cap) | 0.0227 | 0.151 | 0.40 | 2.26 |
| lag off (alpha 0) | 0.0035 | 0.059 | 0.02 | 0.34 |
| k_t 0.1 | 0.0034 | 0.058 | 0.02 | 0.33 |
| k_r 2 | 0.0044 | 0.067 | 0.04 | 0.43 |
| k_t 0.1 and k_r 2 | 0.0008 | 0.029 | any | 0.07 |
| tau 0.20 (hardware transient fit) | 0.0129 | 0.114 | 0.21 | 1.28 |

![Figure 1. (a) Orbit radius error against commanded radius for five lag levels, vortex. Lines are the closed form. (b) The same for tangential and radial gains at alpha = 0.7. (c) Closed form against simulation for designed cells and random trials, gray where the validity conditions fail.](figures_v3/fig1_orbit_radius.png)

## 5. Attraction accuracy and the gain

At hardware settings all six linear fields converge in 100% of 100,000 trials each and end 24.9 mm from the critical point (Table A), which is v_c / k. Gain trades stop distance for overshoot. At alpha = 0.7, k = 3 cuts the mean final error to 6.5 mm with a 95th-percentile overshoot of 2.3 cm, and k = 10 cuts it to 0.40 mm with 3.7 cm. At k = 0.3 the rest radius v_c / k = 8.3 cm lies outside the 5 cm test, so only 0.8% of trials converge.

![Figure 2. (a) Median overshoot past the target against k, with dotted lines at k_c for each alpha. (b) Median stop distance against v_c at k = 1, with the 1-D model as lines and v_c / k dashed. (c) Median final error in mm over alpha and k, with k_c in yellow and k_max in red.](figures_v3/fig2_attraction.png)

## 6. Formation size and curvature

These runs turn lag, friction and the cap off, so they do not depend on the robot model. On curved fields the estimator's fixed point sits off the critical point by an amount that grows as side^3, because the cubic terms of the field do not cancel over an equilateral triangle. It is 0.16 mm at a 0.1 m side and 6.6 mm at 0.33 m in the gyre, and the fixed-point calculation matches (Fig. 3a, crosses). The reconstructed vortex has a floor that does not shrink with size, 17 to 21 mm at sides from 0.01 to 0.33 m, and the reconstructed saddle falls only from 13 to 5 mm over the same range. That is the fixed spatial distortion of the MLP reconstruction, and it shows up as bias (Table A). For orbits, curvature acts through the estimator error d, split into radial and tangential parts, which adds <d_r> + k_t <d_t> / (k_r r) to the radius. In the gyre the estimate lands past the center, so orbits come in by 7 cm at r_d = 0.3 (Fig. 3b). The two curved saddles have the same cubic on the axis but behave differently (Fig. 3c, d). In the irrotational saddle the error is +2 eps r^3 on one axis and -2 eps r^3 on the other, so it averages out. The mean radius moves under 2 mm for |eps| <= 1, and the orbit becomes an ellipse with wobble 2|eps| r^3 sqrt(k_r^2 + Omega^2) / sqrt(k_r^2 + 4 Omega^2), with Omega = k_t / r. That predicts 0.036 m against 0.038 m simulated at eps = 1, and +eps and -eps give identical numbers. In the rotational saddle the error has the same sign on both axes, so the mean shifts, from +0.013 m at eps = 0 to -0.047 m at eps = 1 and +0.18 m at eps = -1. Orbits are lost once they reach a point where det J = 0, which also bounds the usable radius in the gyre (r_d >= 0.4 fails, Table B).

![Figure 3. (a) Median final error against formation side, crosses from the fixed-point calculation. (b) Orbit radius error against side at r_d = 0.3 with lag and friction off. (c) Mean radius error against eps for the two curved saddles. (d) Orbit ellipticity against eps. Hollow markers on the top edge are lost orbits.](figures_v3/fig3_size_curvature.png)

## 7. Sensor noise

If the plane is fitted about the centroid, the slope errors and the centroid-value error are uncorrelated for any triangle, because the robot offsets sum to zero. That answers R1.3's concern about correlated J_hat and h_hat errors. To first order

```
Cov(p_hat) = (sigma_n^2 / 3) (1 + 2 |e|^2 / R^2) J^{-1} J^{-T},     R = side / sqrt 3,   e = distance to p*
```

At the critical point this does not depend on formation size. Fed through the sampled loop (a discrete Lyapunov equation), it gives the closed-loop mean error with friction off to a median ratio of 1.004 over 66 cells (range 0.98 to 1.04, Fig. 4), as long as lambda R / sigma_n >= 3, where lambda = sqrt|det J|. Below that the noise across the triangle swamps the field's slope and errors grow quickly. The spirals (lambda = 0.43) are 2.3 times noisier than the other linear fields for that reason.

With friction, moderate noise helps. Noise in the estimate pushes the command past v_c, so parked robots break away and creep closer. At sigma_n = 0.01 the final error drops from 24.9 mm to 1.8 to 14.8 mm on the linear fields at a 0.33 m side. Whether the hardware does the same is untested. At sigma_n = 0.1 convergence falls to 93% (vortex) and 24% (sinking vortex). The same dither lowers the reconstructed fields' error with friction on (vortex 28.2 to 21.6 mm and saddle 25.4 to 14.5 mm between sigma_n = 0 and 0.03), so the separation of fixed distortion from temporal noise (R1.6) rests on the friction-off runs, where the reconstructed fields' error stays at 21.3 to 22.5 mm (vortex) and 14.1 to 15.8 mm (saddle) up to sigma_n = 0.03.

![Figure 4. Mean final error against sensor noise for three formation sides, lag and friction off. The line is the covariance fed through the loop, and dotted lines mark lambda R / sigma_n = 3 for each side.](figures_v3/fig4_noise.png)

## 8. Estimator comparison and critical-point type

On every linear field the plane fit and the oracle give identical trajectories (largest orbit difference 0.000 mm), so all of the orbit error in the paper's table is controller dynamics, not estimation (R3). In the gyre the fit pulls orbits inward relative to the oracle (-0.044 against +0.051 m at r_d = 0.3). The magnitude-only baseline ends 2.1 mm from the target on the unit-strength fields (6.3 mm on the spirals), closer than the fit, because gradient descent on |v| has a much higher effective gain near the target, which shrinks the friction band v_c / k. It cannot hold an orbit smaller than about one formation radius. At hardware settings r_d = 0.01 and 0.1 both collapse to a 0.032 m circle, and it loses most gyre orbits at r_d >= 0.4.

The attraction law follows -J^{-1} F, a sampled Newton flow, which cannot cross det J = 0. So the sign of det J_hat at the start equals the index of the critical point reached (saddle -1, otherwise +1). Over 451,508 gyre and rotational-saddle trials that reached a critical point it was right 99.89% of the time, and in all 180,771 that started at least one formation circumradius from det J = 0 (Fig. 6c).

![Figure 5. (a) Median final error by field and estimator at hardware settings. (b, c) Orbit radius error against r_d in the gyre and the reconstructed saddle, bars are within-trial standard deviation.](figures_v3/fig5_estimators.png)

![Figure 6. Random design. (a) Convergence against k / k_max. (b) Share of trials that move at all against the first command over v_c. (c) Share where the sign of det J_hat at the start matches the type of critical point reached, by start distance from det J = 0.](figures_v3/fig6_random_index.png)

## 9. Robot model sensitivity and the hardware runs (E11)

The robot model note leaves two parameters uncertain. A fit to the hardware start transients gives tau about 0.20 s against the 0.28 s used here, and static friction is assumed equal to kinetic. E11 (`sensitivity_v3.py`) reruns the hardware-settings attraction, the paper's orbits, the radius sweep and the 157 hardware starts for every combination below, each next to its closed form.

```
for tau in {0.20, 0.28}, static v_c in {0.025, 0.035}, kinetic v_k in {v_c, v_c / 2, 0}:
    attraction (plane fit) on vortex, saddle, gyre and both reconstructed fields, N 10^4; vortex at k 0.3, 3, 10
    orbit at r_d 0.01 to 0.5 on the same fields, N 10^3; radius sweep 1e-4 to 10 m on the vortex, N 200
    the 157 hardware starts x 50 headings, 10 s
```

Kinetic friction sets where the cluster rests and static friction only decides whether it starts. On the linear fields the rest distance is v_k / k to within 0.05 mm for both values of tau, and it is zero when v_k = 0 even with static friction present. The orbit closed form with V = min(k_t - v_k, v_max) holds across all 312 linear-field orbit cells (median gap 0.06 mm, largest 0.39 mm), and an orbit starts in every cell where k_t > v_c.

Both uncertain parameters shrink the orbit error, tau through T_eff and v_k through V^2.

**Table 4.** Mean orbit radius error in m, vortex, plane fit, k_t 0.2, k_r 0.5, 1,000 trials per cell.

| model | r_d 0.01 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| tau 0.28, no friction | +0.146 | +0.109 | +0.081 | +0.062 | +0.050 | +0.042 |
| tau 0.28, v_k 0.025 (v3 model) | +0.129 | +0.093 | +0.067 | +0.051 | +0.040 | +0.033 |
| tau 0.28, v_k 0.035 | +0.122 | +0.086 | +0.061 | +0.046 | +0.037 | +0.030 |
| tau 0.20, no friction | +0.123 | +0.087 | +0.062 | +0.047 | +0.037 | +0.031 |
| tau 0.20, v_k 0.025 | +0.109 | +0.074 | +0.052 | +0.038 | +0.030 | +0.025 |

The friction level stays at the 0.025 m/s measured in a dedicated experiment. As a check, the hardware attraction runs can be compared with v_k through two bias-free measures, the precision about the mean rest point and the median rest point along the approach line. The hardware saddle runs give 23.8 mm and -19.5 mm, which the reconstructed saddle reproduces at v_k = 0.020 to 0.025 m/s (19.5 to 24.0 mm and -18.7 to -23.5 mm, smallest KS distance 0.14 at 0.0225), consistent with the measured value. The hardware vortex runs give 14.3 mm and -12.9 mm, which the reconstructed vortex reproduces at a lower v_k of 0.0125 to 0.015 m/s (13.6 to 15.7 mm and -10.8 to -13.3 mm). That difference is left open (`results_v3_sens/hw_fit.csv`).

The hardware orbit runs (`trunk/robots_3`, metres) reproduce the paper's hardware orbit table exactly from the full record. Each run is 600 control cycles, 60 s at the nominal 0.1 s step, and covers 2.7 to 17.7 orbits. The paper's text gives 3000 cycles, and a correction is queued for it. Table 5 sets them against the model on the reconstructed fields.

**Table 5.** Orbit radius error in m, hardware after 20 s (full record in parentheses) against the model on the reconstructed fields (v_c = v_k = 0.025), 1,000 trials per cell.

| r_d | vortex, hardware | vortex model, tau 0.28 / 0.20 | saddle, hardware | saddle model, tau 0.28 / 0.20 |
|---|---|---|---|---|
| 0.01 | +0.269 (+0.246) | +0.133 / +0.111 | +0.115 (+0.114) | +0.131 / +0.112 |
| 0.1 | +0.192 (+0.190) | +0.094 / +0.078 | +0.072 (+0.072) | +0.089 / +0.075 |
| 0.2 | +0.150 (+0.143) | +0.089 / +0.069 | +0.058 (+0.058) | +0.067 / +0.047 |
| 0.3 | +0.091 (+0.087) | +0.087 / +0.076 | +0.095 (+0.089) | +0.062 / +0.051 |
| 0.4 | +0.038 (+0.032) | +0.063 / +0.056 | +0.100 (+0.090) | +0.039 / +0.033 |
| 0.5 | -0.035 (-0.035) | +0.026 / +0.020 | +0.068 (+0.060) | +0.000 / -0.004 |

The saddle orbits at r_d <= 0.2 sit between the two values of tau. The speeds here take the logged 0.1 s step as the real cycle. On that time base the saddle orbits run at 0.21 to 0.23 m/s, faster than k_t = 0.2 allows, and the closed form without friction gives T_eff of 0.13 to 0.18 s. If some cycles ran longer than 0.1 s, the apparent speed overstates the real one by the same factor. The model's 0.175 m/s would then correspond to cycles of 0.12 to 0.13 s, and the same orbits give T_eff of 0.20 to 0.24 s, close to the 0.20 s transient fit. At larger radii the hardware saddle orbits are larger and much more variable than the model (within-run sd up to 0.09 m). The vortex orbits are two to three times the model at small radii and move at only 0.13 to 0.14 m/s on the logged time base, against 0.175 m/s in the model. A slower orbit should be smaller, and longer cycles would make the real speed lower still, so something other than the lag, most likely the estimate in the real field, sets the vortex orbit radius. Neither difference is resolved here.

![Figure 10. (a) Mean final distance from the 157 hardware starts against kinetic friction, tau 0.20 (markers) and 0.28 (lines), with the hardware means dotted. (b) Orbit radius error on the vortex for tau and v_k, lines the closed form. (c) Hardware orbit runs against the model on the reconstructed fields.](figures_v3/fig10_robot_model.png)

## Tables

**Table A.** Attraction at hardware settings (Coulomb friction), 100,000 trials per field. Per-trial final distance in m. Conv. is the share of trials within 5 cm.

| field | conv. % | mean | median | p95 | bias | precision | t to 5 cm (s) | oracle median | magnitude conv. % / median |
|---|---|---|---|---|---|---|---|---|---|
| six linear fields | 100.0 | 0.0249 | 0.0250 | 0.0250 | 0.0001 | 0.0249 | 2.1 | 0.0250 | 100 / 0.0021 (spirals 0.0062 to 0.0063) |
| gyre | 59.7 | 0.0262 | 0.0252 | 0.0336 | 0.0001 | 0.0266 | 1.8 | 0.0250 | 59.9 / 0.0100 |
| curved saddle, eps 1 | 99.9 | 0.0240 | 0.0234 | 0.0310 | 0.0001 | 0.0244 | 2.1 | 0.0250 | 99.8 / 0.0073 |
| vortex (reconstructed) | 95.3 | 0.0285 | 0.0314 | 0.0456 | 0.0191 | 0.0244 | 2.3 | 0.0250 | 100 / 0.0329 |
| saddle (reconstructed) | 100.0 | 0.0252 | 0.0260 | 0.0437 | 0.0117 | 0.0248 | 2.3 | 0.0250 | 100 / 0.0165 |

**Table B.** Orbit, plane fit, mean radius error in m (between-trial sd, within-trial sd), 10,000 trials per cell. "Off" turns lag, friction and the speed cap off.

| field | settings | r_d 0.01 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| six linear fields, and oracle anywhere | hardware | +0.129 | +0.093 | +0.067 | +0.051 | +0.040 | +0.033 |
| same | off | +0.059 | +0.031 | +0.018 | +0.013 | +0.010 | +0.008 |
| gyre | hardware | +0.127 (.000, .003) | +0.080 (.000, .009) | +0.023 (.000, .016) | -0.044 (.001, .023) | 44% kept | 4% kept |
| curved saddle, eps 1 | hardware | +0.129 (.000, .003) | +0.092 (.000, .012) | +0.067 (.002, .030) | unstable, 90% kept | lost | lost |
| vortex (reconstructed) | hardware | +0.133 (.003, .018) | +0.094 (.003, .016) | +0.089 (.002, .022) | +0.087 (.006, .026) | +0.063 (.009, .029) | +0.026 (.009, .035) |
| saddle (reconstructed) | hardware | +0.131 (.006, .008) | +0.089 (.004, .010) | +0.066 (.007, .018) | +0.062 (.008, .033) | +0.039 (.009, .044) | +0.000 (.008, .052) |

The between-trial standard deviations are near zero. For a deterministic field, repeating an orbit condition changes almost nothing, and the spread reviewers saw is within-trajectory ripple (R3).

## Caveats

- The sampled Coulomb update is exact when the command is parallel to the velocity. At a reversal with kinetic friction below static it can carry a robot through zero velocity that the continuous model would stick, so the swing map is not used for v_k < v_c.
- The hardware orbit speeds differ from the model's in both fields, and the vortex orbit files do not log the commanded velocities, so the gains used on hardware could not be checked. The orbit logs use the nominal 0.1 s step as their time base, and real cycles may have run longer.
- The truth for the reconstructed fields is the printed map's critical point at the origin. The MLP fields never reach zero (minimum |v| is about 0.05).
- One formation-control gain set (0.5 and 0.01) and a 0.3 m/s radial clamp throughout.
- `results_v3/tables.md` holds every per-field and per-estimator orbit cell, `results_v3/v3_numbers.txt` and `results_v3/report_numbers.txt` hold the checks quoted above, and `results_v3_sens/` holds E11.

## Files (VF_Robot/experiments/hyperparameter_study/)

`vecsim.py` (simulator, all four stiction rules), `validate_vecsim.py`, `design.py` (`build_v3`), `run_study.py --v3` (resumable), `theory.py` (every formula above, including the Coulomb swing map), `analyze.py` and `analyze_v2.py` (figures and tables, `--res results_v3`), `report_numbers.py`, `sensitivity_v3.py` (E11, `--fit` for the hardware friction scan, `--frames` for the frame check), `load.py`, `build_report.py` (this page), `results_v3/`, `results_v3_sens/`, `figures_v3/`. Version 2 (dead zone) is kept in `results_v2/`, `figures_v2/` and `hyperparameter_study_v2.*`, and version 1 (legacy rule) in `results/` and `figures/`.
