# 3-robot plane-fit accuracy on linear, gyre and curved-saddle fields

2026-10-28. Simulation only, noise off, default gains (attraction k = 1 /s, orbit k_t = 0.2 m/s and k_r = 0.5 /s). Monte Carlo cells hold 100 runs with a random start and formation heading. These numbers are for review and do not go into any .tex.

The code is in `VF_Robot/experiments/`. `plane_fit_field_test.py` and `vector_to_scalar_field_test.py` run one configuration (flags `--field --eps --side --alpha --stiction`), `mc_field_tests.py` writes the tables to `outputs/mc_field_tests/summary*.txt`, and `index_prediction_check.py` runs the last section. All of them share `_field_tests_common.py`, whose docstring defines the fields.

## Bug fixed first

`OmniCluster` (3-robot) and `PentagonCluster` (6-robot) evaluated the inverse Jacobian at a heading pi away from the one it differentiates, which flipped the sign of the shape feedback. Round-off errors grew as e^{0.44 t} and the formation fell apart after about 60 s (3-robot) or 30 to 50 s (6-robot) of motion. Both are fixed, so every earlier 3- and 6-robot simulation needs a re-run. Across the 17,500 runs below the worst side-length error is 1e-9.

## The model behind the numbers

alpha is the robot lag, a single knob with time constant -dt / ln alpha. The orbit law holds the estimated radius rho = |p_hat - c| at r_d.

```
Orbit, mean radius
    d      <- p_hat - p_true            split into radial d_r and counterclockwise d_t
    r*      = r_d + e_dyn(alpha, r*) + <d_r> + (k_t / (k_r r*)) <d_t>      <.> = mean over orbit angle
    e_dyn   = [k_t sin(lag) + dt G k_t^2 / (2 r)] / (k_r cos(lag))         lag, G of the robot filter at k_t / r
Orbit, wobble
    harmonic n of the angle-dependent part of d passes with gain 1 / sqrt((k_r s)^2 + (n k_t / r)^2)
Attraction, final error
    max(estimator fixed-point bias, stiction stop)       stop = v_s / k with no lag, about half at alpha 0.7
Attraction, destination type
    sign det J_hat at the start = sign det J at the critical point reached
```

The predicted mean radius lands within 2 mm of the simulation in 84 of the 116 orbit cells that ran, and within 15 mm in 103. The 13 misses are orbits that reach a det J = 0 point and the magnitude-only cases flagged below. All 9 cells where the model predicted no stable orbit had none.

## What each knob does (vector plane fit)

| knob | linear center or saddle (estimate exact) | single gyre | curved saddles |
|---|---|---|---|
| lag alpha, orbit | Pushes the radius out. r_d = 0.2 gives 0.219, 0.248, 0.281, 0.393 for alpha 0.01, 0.5, 0.7, 0.9 (predicted 0.219, 0.248, 0.280, 0.392). | Same push minus the estimator pull. r_d = 0.2 gives 0.184 (alpha 0.01) and 0.222 (alpha 0.7). | Same push. |
| lag alpha, attraction | No effect without stiction. With stiction the robot coasts and stops closer. | Same. | Same. |
| stiction v_s | Orbit unchanged to 3 decimals. Attraction stops 0, 0.44, 1.26, 2.75 cm out for v_s = 0, 0.01, 0.025, 0.05 (alpha 0.7). A start within v_s / ((1 - alpha) k) = 8.3 cm never moves. At alpha 0.9 and v_s 0.025 an orbit never starts. | Stops 1.2 to 1.4 cm out. | Same. |
| formation side | No effect. | Attraction bias 0.2 mm (0.10 m side) and 6.6 mm (0.33 m), growing as side cubed. Orbit radius moves 0.013 m at most. | The 0.33 m triangle fails once it reaches a det J = 0 point (potential saddle, eps = +-1, r_d 0.3: 13 to 15% of runs fail and the rest are 0.2 m off). |
| orbit radius r_d | Error falls as 1/r. | The estimate overshoots by about pi^2 r^3 / 3, so radii are squeezed to 0.12 to 0.28 for r_d 0.1 to 0.4. At r_d 0.4 the orbit reaches the det J = 0 diamond and only 34 to 52% of runs survive. | Wobble grows as r^3. |

## Curvature sweep, v = -y + eps y^3 on the inflow axis (side 0.10, alpha 0.01, no stiction, r_d 0.3)

| eps | -2 | -1 | -0.5 | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| potential (irrotational), mean r | captured | 0.315 | 0.313 | 0.313 | 0.313 | 0.315 | captured |
| potential, radius std (predicted) | | 0.031 (0.029) | 0.015 (0.014) | 0 | 0.015 (0.014) | 0.031 (0.029) | |
| cellular (rotational), mean r (predicted) | runaway | 0.479 (0.491) | 0.370 (0.370) | 0.313 | 0.278 (0.277) | 0.253 (0.252) | 0.219 (0.218) |

"Captured" means the ellipse reaches det J = 0 at radius 1/sqrt(3 |eps|) = 0.41 and the cluster ends up orbiting a neighboring saddle. Both fields have the same cubic on the axis and give opposite answers. In the irrotational saddle the estimate error is +2 eps r^3 on one axis and -2 eps r^3 on the other, which averages to zero around an orbit (the log of the estimated radius is harmonic). The mean radius stays put, the orbit turns into an ellipse, and eps and -eps give identical numbers because they differ by a 90 degree rotation plus flow reversal. In the rotational saddle the error is -2 eps r^3 on both axes, so the mean shifts inward for eps > 0 (sin, gyre-like) and outward for eps < 0 (sinh). The gyre is the same sin-product flow seen from its center, and its orbits come in small for the same reason.

## Magnitude only (|v| sensed, no direction)

- Round cones, meaning J^T J is a multiple of I (linear saddle, gyre center, potential saddle). With the 0.10 m side, orbit radii are within 0.005 m of the vector method, and attraction ends 0.8 mm out against 12 mm for vector. Gradient descent on |v| has an effective gain of order sigma / R (about 17 /s) against k = 1 for the vector law, so stiction stops it closer. Raising k would do the same for the vector law.
- With the 0.33 m side it cannot orbit inside about one circumradius (0.19 m). r_d = 0.1 collapses to a 0.04 m circle, and in the gyre r_d = 0.2 does too.
- Anisotropic cone (shear saddle u = x + y, v = -y). It fails. Orbit wobble is 0.07 to 0.17 m, and attraction ends 3 to 4.5 cm out, with only 39% (0.10 m side) and 7% (0.33 m) of runs inside 5 cm. The vector method is exact in the same field.

## Predicting the destination type from the first fit

Each cell shows matched runs over runs that reached a critical point (100 starts each).

| | gyre | cellular eps 1 | cellular eps 2 | Duffing (saddle + 2 foci) |
|---|---|---|---|---|
| side 0.10, alpha 0.01, no stiction | 100/100 | 100/100 | 100/100 | 100/100 |
| side 0.20, alpha 0.7, v_s 0.025 | 97/97 | 98/98 | 99/99 | 95/95 |
| side 0.33, alpha 0.7, v_s 0.025 | 96/96 | 99/99 | 94/95 | 38/40 |

At 0.20 m the 11 runs that reached nothing all started 5 to 8 cm from a critical point and never moved (stiction start). At 0.33 m the three opposite-type arrivals started within two circumradii of det J = 0. The 60 Duffing runs stalled at |x| = 0.32 because the triangle spans the saddle-to-focus gap and its plane fit has a false zero there.

The barrier is a property of continuous Newton flow and is probably known in numerical analysis (Branin's method reverses direction at det J = 0 to cross it). I have not checked those papers, and they are not in the corpus. The index of a triangle's linear interpolant is standard in vector field topology (Tricoche 2001, in the corpus). Nothing in the corpus uses it to predict where a robot formation will end up. In practice it is a check before committing (attract only when sign det J_hat matches the type wanted), plus a flag if the sign flips mid-run. Before it can be stated as a result it needs a bound on formation size relative to the distance to det J = 0 and to the next critical point, since every failure above comes from a triangle that is too large.

Noise is off, there is one gain set, and each cell has 100 runs. The full grid at 10^4 runs per cell would take about 17 h on this machine, and `--only` runs single groups.
