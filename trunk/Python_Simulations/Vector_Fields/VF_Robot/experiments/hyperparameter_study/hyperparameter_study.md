# Hyperparameter study for Section IV (3-robot cluster, simulation)

2026-10-29, version 2. Simulation only, run after the 2026-10-28 formation sign fix. This version replaces the paper simulator's stiction rule with a dead zone on the command, which is the exact sampled form of a continuous actuator model, and adds overshoot, undershoot and speed-limit experiments. Numbers are for review and none are in the .tex.

## What was run

All runs use a vectorized copy of the paper's simulator (`vecsim.py`). It reproduces `OmniCluster`, `Omnibot` and the Eq. 10 and Eq. 11 laws to within 1e-11 m of the reference code over 400 steps, to 4e-7 m on the MLP fields, and to 1e-12 m under each of the three stiction rules (`validate_vecsim.py`). Version 2 ran 5,339 designed configurations plus 1,000 random batches, 28.7 million trials in all, in 76 minutes on 7 cores. Each configuration is a batch of trials with random starts and random formation headings.

```
for each configuration (field, estimator, task, side, tau, dt, v_s, v_max, k, k_t, k_r, r_d, sigma_n, stiction rule):
    repeat N times:
        draw start (attraction: uniform in a 1 m box; orbit: on the commanded circle) and heading
        run 60 s (attraction) or w0 + W s (orbit)
        record per-trial metrics
    summarize, and compute the theory prediction for the same configuration
```

| Exp. | Question (reviewer) | Factors and levels | Trials |
|---|---|---|---|
| E1 | Per-trial errors, convergence and failure (AE, R1.6, R3) | 10 fields x 3 estimators x {hardware, non-idealities off}, attraction, and orbit at r_d 0.01 to 0.5 | 9.6M (10^5 per attraction cell, 10^4 per orbit cell) |
| E2 | Achievable orbit radius, actuator lag, gains (R1.4, R1.2) | r_d 1e-4 to 10 m (14 levels) x alpha 0 to 0.95, k_t 0.05 to 0.5, k_r 0.1 to 5, dt 0.02 to 0.2 at fixed tau, v_s 0 to 0.1 | 1.5M |
| E3 | Attraction gain, lag, stiction (AE, R1.2) | k 0.1 to 100 x alpha, v_s x alpha, dt x k | 2.5M |
| E4 | Formation size (R1.1, R3) | side 0.01 to 1 m x 6 fields x {fit, magnitude} | 2.1M |
| E5 | Curvature and multiple critical points (R1.1, R3) | eps -4 to 4, irrotational and rotational saddles, side, r_d, basins | 0.8M |
| E6 | Temporal noise versus fixed distortion (R1.3, R1.6) | sigma_n 0 to 0.1 x side x 6 linear fields, plus the 2 reconstructed fields | 1.6M |
| E7 | Global check | 1,000 batches of 10^4 trials, every parameter drawn at random per trial | 10M |
| E8 | Overshoot and undershoot | tau 0 to 1 s x dt 0.01 to 0.2 x k 0.3 to 30 x caps on/off x v_s {0, 0.025}; orbit transient from r0 0.05 and 0.6 x k_r 0.25 to 5 | 0.31M |
| E9 | Raised speed limits | v_max 0.1, 0.3, 1, 3 m/s and none x alpha x k 1 to 300; orbit v_max x k_t 0.1 to 2 x r_d x radial clamp | 0.16M |
| E10 | Stiction rule | rule x dt 0.01 to 0.2 at tau 0.28 x v_s; and the 157 hardware start positions x 50 headings, 10 s runs | 0.16M |

The 10 fields are the paper's six linear fields, a single double-gyre cell (center target, saddles at the corners), a curved saddle (v = -y + eps y^3 on the inflow axis, irrotational or rotational form), and the two MLP fields reconstructed from testbed data. The estimators are the plane fit (paper), the true critical point (an oracle that isolates the controller), and a magnitude-only baseline that senses |v| and feeds the same laws. Attraction counts as converged when the final distance to the true critical point is under 5 cm. Orbit statistics start after w0 = max(20 s, 5/k_r, 10 tau) and run for 1.5 revolutions (40 to 300 s). A trial fails if it escapes, its formation side error exceeds 10%, or it completes under a quarter revolution.

## 1. Stiction has to enter through the sampling rate

The robot model is a first-order lag, tau dv/dt = -v + u. Held constant over each sample, it is exactly

```
v[n+1] = alpha v[n] + (1 - alpha) u[n],      alpha = exp(-dt / tau)
```

so alpha is a property of the sample period and tau is the physical constant. The paper's alpha = 0.7 at dt = 0.1 s is tau = 0.28 s. The same robot sampled at dt = 0.01 s has alpha = 0.965. Three discrete stiction rules were compared.

```
legacy     v <- alpha v + (1 - alpha) u;        if |v| < v_s:             v <- 0
scaled     v <- alpha v + (1 - alpha) u;        if |v| < (1 - alpha) v_s: v <- 0
dead zone  v <- alpha v + (1 - alpha) DZ(u),    DZ(u) = 0 if |u| < v_s, else u
```

The legacy rule is the paper simulator's. A robot at rest moves only if (1 - alpha)|u| >= v_s, so the threshold it applies to the command is v_s / (1 - e^(-dt/tau)), about v_s tau / dt. That is 8.3 cm/s at dt = 0.1 and 0.71 m/s at dt = 0.01, above the 0.5 m/s command cap, so at the faster rate nothing ever leaves rest. Scaling the threshold by (1 - alpha) fixes the start, but a moving robot then stops only when its velocity falls below (1 - alpha) v_s, which goes to zero with dt, so it creeps in and the stop distance depends on the sample rate the other way. The dead zone is the exact sampled form of tau dv/dt = -v + DZ(u). Its stiction term enters through the same (1 - alpha) factor as the command, so the coefficient is a function of the sampling rate by construction and the continuous behavior is the same at every dt. A robot starts when |u| >= v_s, the command switches off inside |e| < v_s / k, and the robot coasts down on its lag.

**Table 1.** Stiction rules at tau = 0.28 s, v_s = 0.025 m/s, k = 1, vortex, 1,000 trials per cell. Mean final distance to the critical point in mm, all trials.

| rule | moves from rest when | dt 0.01 | dt 0.02 | dt 0.05 | dt 0.1 | dt 0.2 | orbit at k_t 0.05 starts | hardware starts, 10 s |
|---|---|---|---|---|---|---|---|---|
| legacy | abs(u) >= v_s / (1 - alpha) | 380.1 | 104.9 | 20.6 | 13.5 | 12.6 | only at dt 0.2 | 12.5 |
| scaled | abs(u) >= v_s | 0.3 | 0.6 | 1.7 | 3.3 | 5.8 | at every dt | 3.2 |
| dead zone | abs(u) >= v_s | 12.4 | 12.4 | 12.4 | 12.2 | 11.9 | at every dt | 11.7 |
| hardware | | | | | | | | 14.0 |

The 157 hardware runs (79 vortex, 78 saddle, 10 s each) end 14.0 mm from the critical point on average (median 13.9, 95th percentile 24.0). Re-running the same starts with 50 random headings each, the dead zone gives 11.7 mm on the linear fields (median 11.5, 95th percentile 12.8) and 20.0 mm on the reconstructed fields. The saddle runs are the cleanest check. The dead zone stops every simulated trial short of the target on the approach side, 11.5 mm back along the approach line, and 94% of the hardware saddle runs stop on the approach side, 10.8 mm back along it. The hardware vortex runs carry a fixed 12.4 mm offset in one direction (-3.6, +11.9 mm), which no stiction rule produces and which looks like a field-center or calibration offset. The hardware spread is wider than any single-v_s model (Fig. 7c), which points to robot-to-robot differences in v_s that the model leaves out.

Everything below uses the dead zone. `Omnibot` and `OmniCluster` now take `stiction_model` ("legacy", "deadzone", "scaled"), but the default is still "legacy", so every existing script reproduces. Switching the default is one line in each file and changes every simulation result in the repo, so it waits for your approval.

![Figure 7. (a) Mean final distance against sample period at fixed tau = 0.28 s, filled k = 1, hollow k = 10. At k = 1 only the dead zone is flat. (b) Share of orbit trials that start. The legacy rule fails at small dt. (c) Final-distance distributions from the 157 hardware start positions, hardware against each rule.](figures_v2/fig7_stiction_models.png)

## 2. Overshoot and undershoot as functions of alpha, sample rate and gain

With an exact estimate the cluster moves on a straight line, so attraction reduces to a sampled second-order loop. Writing everything in tau and dt

```
alpha  = exp(-dt / tau)
T_eff  = T_lag + dt/2 = (dt/2) coth(dt / (2 tau))              -> tau as dt -> 0
poles    z^2 - (1 + alpha - dt (1 - alpha) k) z + alpha = 0
k_c    = (1 - sqrt alpha) / (dt (1 + sqrt alpha)) = tanh(dt / (4 tau)) / dt     -> 1 / (4 tau)     overshoot begins
k_max  = 2 (1 + alpha) / (dt (1 - alpha))      = 4 T_eff / dt^2              -> infinity       unstable beyond
overshoot ratio, dt -> 0     M = exp(-pi zeta / sqrt(1 - zeta^2)),   zeta = 1 / (2 sqrt(k T_eff))
```

The exact overshoot ratio comes from iterating the two-pole loop. In the continuous limit overshoot depends on k tau alone, and the sample period adds a small correction through T_eff and the discrete poles. At fixed tau the sample rate hardly matters (k = 3 and tau = 0.28 s give 12.9% at dt = 0.01 and 14.4% at dt = 0.2, Fig. 8b). At fixed alpha it matters a great deal, because alpha = 0.7 means tau = 0.28 s at dt = 0.1 and tau = 0.028 s at dt = 0.01. With caps and stiction off, the simulator matches the discrete prediction to 0.000 percentage points in all 121 stable cells and diverges in all 125 cells above k_max. At the paper's settings k_c = 0.89 /s, so k = 1 sits just past the onset with a predicted overshoot of 0.01%.

**Table 2.** Overshoot in % of the start distance at k = 3, caps and stiction off, 500 trials per cell.

| tau (s) | dt 0.01 | dt 0.02 | dt 0.05 | dt 0.1 | dt 0.2 |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.5 |
| 0.28 | 12.9 | 12.9 | 13.0 | 13.3 | 14.4 |
| 0.5 | 24.5 | 24.5 | 24.6 | 24.6 | 25.1 |
| 1.0 | 38.8 | 38.8 | 38.8 | 38.9 | 39.6 |

Stiction turns the end of the approach into an undershoot. The command switches off once |e| < v_s / k, and the robot then coasts T_eff times its speed. On an overdamped approach (4 k T_eff < 1) the speed at the switch-off point follows the slow pole, which gives

```
d_stop = (v_s / k) (1 + sqrt(1 - 4 k T_eff)) / 2          short of the target, on the approach side
```

At k = 1 this matches the simulated median to within 2.4% for alpha 0, 0.3, 0.5 and 0.6 and v_s from 0.005 to 0.1 m/s (largest gap 1.7 mm, at v_s = 0.1). Underdamped, the robot can come to rest anywhere in the disk |e| <= v_s / k and on either side of the target, but never outside it. The paper's settings give 4 k T_eff = 1.13, just underdamped. The simulation rests 12.2 mm short, close to the critical-damping value v_s / (2k) = 12.5 mm. Raising k shrinks the disk as 1/k, and a longer tau can carry the rest point past the target (tau = 0.5 s at k = 1 rests 1.9 cm past, Fig. 8c).

The orbit's radial loop is the same sampled loop with k replaced by k_r, so the same formula predicts radial overshoot when an orbit starts off r*. Over 147 caps-off cells with |r0 - r*| >= 0.1 m the median gap is 0.0 percentage points, the 90th percentile is 7.6, and the correlation is 0.93. Starting outside, the simulation follows the prediction (median ratio 0.99). Starting inside, the overshoot is smaller (median ratio 0.65). At the paper's k_r = 0.5 < k_c there is none. One transient lies outside this theory. Starting outside with a long lag, the radius first dips below r*, by 46 mm at tau = 1 s and k_r = 0.25 (r0 = 0.6, r* = 0.577, minimum 0.531), where the loop predicts no overshoot. The outward push that sets r* builds up only as the robots reach speed, so for the first few tau the radial loop aims at r_d rather than r*.

![Figure 8. (a) Overshoot against k for four sample periods at tau = 0.28 s, caps and stiction off. Markers are simulation, solid lines the discrete theory, dashed the continuous limit, dotted the k_max of dt = 0.2. (b) Overshoot at k = 3 over tau and dt. (c) Rest point along the approach line with v_s = 0.025 and dt = 0.1, negative is short of the target. Dotted lines are the overdamped closed form. (d) Orbit radial overshoot against the sampled-loop prediction with k replaced by k_r.](figures_v2/fig8_overshoot_undershoot.png)

## 3. Raising the speed limits

The robots' 0.3 m/s cap is not what limits accuracy. Raising v_max to 1 or 3 m/s, or removing it, changes the attraction error little below k_max (k = 1 gives 12.0 to 13.6 mm, k = 30 gives 0.3 to 0.5 mm). Above k_max the cap decides the outcome. Without one the loop diverges at k = 300. With one it settles into a limit cycle whose final distance is v_max dt / 2 (ratio median 0.99, range 0.96 to 1.09, n = 16), so the error grows with the cap, 4.9, 14.5, 49 and 145 mm at v_max = 0.1, 0.3, 1 and 3 m/s. The same saturation explains why random-design convergence falls only gradually past k_max, to 81% at about seven times k_max (Fig. 6a).

For orbits the closed form of Section 4 uses V = min(k_t, v_max). With the cap raised it holds to k_t = 2 m/s (1.088 m simulated against 1.086 m at r_d = 1). With the 0.3 m/s cap and k_t above it, the orbit comes out larger than the formula (0.054 to 0.088 m against 0.049 m at r_d = 1), because the radial correction has to share the capped speed. The error grows as k_t squared, so a higher cap buys a faster orbit, not a more accurate one.

![Figure 9. (a) Median final error against k for five speed caps, alpha = 0.7, v_s = 0.025. The dashed line is k_max. (b) Limit-cycle size above k_max against v_max dt / 2. (c) Orbit radius error against k_t at r_d = 1 m with raised caps. Lines are the closed form with V = min(k_t, v_max).](figures_v2/fig9_speed_limits.png)

## 4. The orbit radius error has a closed form

The tangential command k_t rotates at k_t / r. The robot lag delays it by T_lag = alpha dt / (1 - alpha), which turns part of k_t outward, and each straight sampled step adds (k_t dt)^2 / (2r). The radial loop balances both at

```
r* = (r_d + sqrt(r_d^2 + 4C)) / 2,     C = V^2 T_eff / k_r,   V = min(k_t, v_max)
error r* - r_d  ->  C / r_d  for large r_d,     ->  r_min = sqrt(C)  as r_d -> 0
```

The estimate does not enter, so the result is the same for every linear field and for the oracle in every field, and stiction does not enter once the orbit is moving. At the paper's gains it gives +0.146, 0.109, 0.081, 0.063, 0.050 and 0.042 m at r_d = 0.01 to 0.5, the paper's Table V to three decimals. Across 1,874 designed cells inside the validity region (r_d from 1e-4 to 10 m) the median gap between formula and simulation is 0.04 mm and the largest is 3.1 mm, at k_t = 2 m/s where r* is about 2 m. Across 16,879 random trials inside the region it holds within 4.2% at the 95th percentile. The region is a lag phase (k_t / r) T_eff under 0.5 rad and k_r times the error under the 0.3 m/s radial clamp. Outside it the orbit ends up larger than the formula says (Fig. 1c, gray). Since T_eff is close to tau, C is about k_t^2 tau / k_r and the sample rate barely matters (r_d = 0.3 gives 0.062, 0.062, 0.063 and 0.064 m for dt = 0.02, 0.05, 0.1 and 0.2 s). With the dead zone an orbit starts if and only if k_t >= v_s, which held in all 3,562 orbit cells and all 1,087,245 random orbit trials.

**Table 3.** Smallest orbit the controller can hold.

| settings | C (m^2) | r_min (m) | r_d for error <= 5 cm | r_d for error <= 1 cm |
|---|---|---|---|---|
| hardware (alpha 0.7, k_t 0.2, k_r 0.5) | 0.0227 | 0.151 | 0.40 | 2.26 |
| lag off (alpha 0) | 0.0040 | 0.063 | 0.03 | 0.39 |
| k_t 0.1 or k_r 2 | 0.0057 | 0.075 | 0.06 | 0.56 |
| k_t 0.1 and k_r 2 | 0.0014 | 0.038 | any | 0.13 |

![Figure 1. (a) Orbit radius error against commanded radius for five lag levels, vortex. Lines are the closed form. (b) The same for tangential and radial gains at alpha = 0.7. (c) Closed form against simulation for designed cells and random trials, gray where the validity conditions fail.](figures_v2/fig1_orbit_radius.png)

## 5. Attraction accuracy and the gain

At hardware settings all six linear fields converge in 100% of 100,000 trials each and end 12.2 mm from the critical point (Table A). Under the legacy rule 1.2 to 1.6% failed the 5 cm test, because starts between 5 and 8.3 cm = v_s / ((1 - alpha) k) sat below its inflated start threshold and never moved. The dead zone moves any start beyond v_s / k = 2.5 cm. Gain trades stop distance for overshoot. At alpha = 0.7, k = 3 cuts the mean final error to 2.1 mm with a 95th-percentile overshoot of 2.4 cm, and k = 10 cuts it to 1.3 mm with 3.0 cm. Gains of 0.3 /s and below stop outside the 5 cm test from almost every start, since their stop distance is about 0.9 v_s / k.

![Figure 2. (a) Median overshoot past the target against k, with dotted lines at k_c for each alpha. (b) Median stop distance against v_s at k = 1, with the 1-D model as lines and v_s / k dashed. (c) Median final error in mm over alpha and k, with k_c in yellow and k_max in red.](figures_v2/fig2_attraction.png)

## 6. Formation size and curvature

On curved fields the estimator's fixed point sits off the critical point by an amount that grows as side^3, because the cubic terms of the field do not cancel over an equilateral triangle. It is 0.16 mm at a 0.1 m side and 6.6 mm at 0.33 m in the gyre, and the fixed-point calculation matches (Fig. 3a, crosses). The reconstructed fields have a floor that does not shrink with size, 17 to 21 mm for the vortex and 9 to 13 mm for the saddle. That is the fixed spatial distortion of the MLP reconstruction, and it shows up as bias (21 mm and 12 mm at hardware settings, Table A). For orbits, curvature acts through the estimator error d, split into radial and tangential parts, which adds <d_r> + k_t <d_t> / (k_r r) to the radius. In the gyre the estimate lands past the center, so orbits come in by 7 cm at r_d = 0.3 (Fig. 3b). The two curved saddles have the same cubic on the axis but behave differently (Fig. 3c, d). In the irrotational saddle the error is +2 eps r^3 on one axis and -2 eps r^3 on the other, so it averages out. The mean radius moves under 2 mm for |eps| <= 1, and the orbit becomes an ellipse with wobble 2|eps| r^3 sqrt(k_r^2 + Omega^2) / sqrt(k_r^2 + 4 Omega^2), with Omega = k_t / r. That predicts 0.036 m against 0.038 m simulated at eps = 1, and +eps and -eps give identical numbers. In the rotational saddle the error has the same sign on both axes, so the mean shifts, from +0.013 m at eps = 0 to -0.048 m at eps = 1 and +0.18 m at eps = -1. Orbits are lost once they reach a point where det J = 0, which also bounds the usable radius in the gyre (r_d >= 0.4 fails, Table B).

![Figure 3. (a) Median final error against formation side, crosses from the fixed-point calculation. (b) Orbit radius error against side at r_d = 0.3 with lag and stiction off. (c) Mean radius error against eps for the two curved saddles. (d) Orbit ellipticity against eps. Hollow markers on the top edge are lost orbits.](figures_v2/fig3_size_curvature.png)

## 7. Sensor noise

If the plane is fitted about the centroid, the slope errors and the centroid-value error are uncorrelated for any triangle, because the robot offsets sum to zero. That answers R1.3's concern about correlated J_hat and h_hat errors. To first order

```
Cov(p_hat) = (sigma_n^2 / 3) (1 + 2 |e|^2 / R^2) J^{-1} J^{-T},     R = side / sqrt 3,   e = distance to p*
```

At the critical point this does not depend on formation size. Fed through the sampled loop (a discrete Lyapunov equation), it gives the closed-loop mean error to within 4% (median ratio 1.005 over 66 cells, Fig. 4) as long as lambda R / sigma_n >= 3, where lambda = sqrt|det J|. Below that the noise across the triangle swamps the field's slope and errors grow quickly. The spirals (lambda = 0.43) are 2.3 times noisier than the other linear fields for that reason. With the dead zone, moderate noise helps. At sigma_n = 0.01 it dithers the command across the threshold and the final error drops from 12.2 mm to 3 to 7 mm. Whether the hardware does the same is untested. At sigma_n = 0.1 convergence falls to 87% (vortex, 0.33 m side) and 22% (sinking vortex). The reconstructed fields' error does not change with added noise up to 0.03, which separates their fixed distortion from temporal noise (R1.6).

![Figure 4. Mean final error against sensor noise for three formation sides, lag and stiction off. The line is the covariance fed through the loop, and dotted lines mark lambda R / sigma_n = 3 for each side.](figures_v2/fig4_noise.png)

## 8. Estimator comparison and critical-point type

On every linear field the plane fit and the oracle give identical trajectories, so all of the orbit error in the paper's table is controller dynamics, not estimation (R3). In the gyre the fit pulls orbits inward relative to the oracle (-0.040 against +0.063 m at r_d = 0.3). The magnitude-only baseline ends 1.0 mm from the target on the unit-strength fields (5.6 mm on the spirals), closer than the fit, because gradient descent on |v| has a much higher effective gain near the target, which shrinks the dead-zone disk. It cannot hold an orbit smaller than about one formation radius. At hardware settings r_d = 0.01 and 0.1 both collapse to a 0.04 m circle, and it loses most gyre orbits at r_d >= 0.3.

The attraction law follows -J^{-1} F, a sampled Newton flow, which cannot cross det J = 0. So the sign of det J_hat at the start equals the index of the critical point reached (saddle -1, otherwise +1). Over 452,634 gyre and rotational-saddle trials that reached a critical point it was right 99.89% of the time, and in all 181,034 that started at least one formation circumradius from det J = 0 (Fig. 6c).

![Figure 5. (a) Median final error by field and estimator at hardware settings. (b, c) Orbit radius error against r_d in the gyre and the reconstructed saddle, bars are within-trial standard deviation.](figures_v2/fig5_estimators.png)

![Figure 6. Random design. (a) Convergence against k / k_max. (b) Share of trials that move at all against the first command over v_s. (c) Share where the sign of det J_hat at the start matches the type of critical point reached, by start distance from det J = 0.](figures_v2/fig6_random_index.png)

## Tables

**Table A.** Attraction at hardware settings with the dead zone, 100,000 trials per field. Per-trial final distance in m. Conv. is the share of trials within 5 cm.

| field | conv. % | mean | median | p95 | bias | precision | t to 5 cm (s) | oracle median | magnitude conv. % / median |
|---|---|---|---|---|---|---|---|---|---|
| six linear fields | 100.0 | 0.0122 | 0.0121 to 0.0122 | 0.0140 | 0.0000 to 0.0001 | 0.0123 | 1.8 | 0.0121 to 0.0122 | 100 / 0.0010 (spirals 0.0056) |
| gyre | 59.9 | 0.0144 | 0.0135 | 0.0220 | 0.0000 | 0.0151 | 1.4 | 0.0122 | 60.5 / 0.0072 |
| curved saddle, eps 1 | 99.8 | 0.0115 | 0.0110 | 0.0180 | 0.0000 | 0.0122 | 1.8 | 0.0122 | 99.8 / 0.0072 |
| vortex (reconstructed) | 100.0 | 0.0239 | 0.0249 | 0.0373 | 0.0211 | 0.0139 | 1.9 | 0.0122 | 100 / 0.0326 |
| saddle (reconstructed) | 100.0 | 0.0159 | 0.0156 | 0.0313 | 0.0118 | 0.0134 | 1.9 | 0.0121 | 100 / 0.0162 |

**Table B.** Orbit, plane fit, mean radius error in m (between-trial sd, within-trial sd), 10,000 trials per cell. "Off" turns lag, stiction and the speed cap off.

| field | settings | r_d 0.01 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| six linear fields, and oracle anywhere | hardware | +0.146 | +0.109 | +0.081 | +0.063 | +0.050 | +0.042 |
| same | off | +0.059 | +0.031 | +0.018 | +0.013 | +0.010 | +0.008 |
| gyre | hardware | +0.142 (.000, .005) | +0.091 (.000, .011) | +0.030 (.001, .018) | -0.040 (.001, .025) | 44% kept | 4% kept |
| curved saddle, eps 1 | hardware | +0.146 (.000, .006) | +0.108 (.000, .015) | +0.081 (.002, .035) | unstable, 87% kept | lost | lost |
| vortex (reconstructed) | hardware | +0.150 (.002, .017) | +0.110 (.004, .017) | +0.107 (.002, .023) | +0.098 (.007, .026) | +0.070 (.009, .030) | +0.031 (.010, .036) |
| saddle (reconstructed) | hardware | +0.148 (.005, .009) | +0.103 (.003, .010) | +0.083 (.007, .020) | +0.073 (.009, .034) | +0.046 (.007, .045) | +0.006 (.008, .053) |

The between-trial standard deviations are near zero. For a deterministic field, repeating an orbit condition changes almost nothing, and the spread reviewers saw is within-trajectory ripple (R3).

## Caveats

- The dead zone lets a robot coast down on its lag after the command switches off. A Coulomb stop on velocity would end the coast sooner and would also block the noise dithering in Section 7. The hardware stop distance and stop side agree with the dead zone as modeled.
- The truth for the reconstructed fields is the printed map's critical point at the origin. The MLP fields never reach zero (minimum |v| is about 0.05).
- One formation-control gain set (0.5 and 0.01) and a 0.3 m/s radial clamp throughout.
- `results_v2/tables.md` holds every per-field and per-estimator orbit cell, including the magnitude baseline and the "off" settings. `results_v2/v2_numbers.txt` holds the checks quoted in Sections 1 to 3.

## Files (VF_Robot/experiments/hyperparameter_study/)

`vecsim.py` (simulator, all three stiction rules), `validate_vecsim.py`, `design.py` (all factor levels, `build_v2`), `run_study.py --v2` (resumable), `theory.py` (every formula above), `analyze.py` and `analyze_v2.py` (figures and tables), `load.py`, `build_report.py` (this page), `results_v2/` (per-trial `.npz`, `summary.csv`, `tables.md`, 1.5 GB), `figures_v2/`. Version 1, with the legacy rule, is kept in `results/` and `figures/`. The `stiction_model` option is in `src/robot/omnibot.py` and `src/robot/omni_cluster.py`.
