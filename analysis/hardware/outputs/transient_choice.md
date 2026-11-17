# Orbit transient trim (Step 2b), 2026-11-16

Mean |r(t) - steady mean| (mm) per 1 s bin, t = 0 to 20 s; steady mean = mean radius after 10 s.

|  | 0 | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| hw vortex 3R | 41 | 38 | 38 | 41 | 34 | 34 | 35 | 35 | 38 |
| hw vortex 4R | 34 | 32 | 28 | 33 | 35 | 34 | 28 | 36 | 23 |
| hw saddle 3R | 37 | 37 | 37 | 36 | 36 | 35 | 36 | 36 | 40 |
| hw saddle 4R | 38 | 35 | 36 | 32 | 35 | 40 | 36 | 37 | 42 |
| sim vortex (linear) | 31 | 11 | 4 | 2 | 1 | 0 | 0 | 0 | 0 |
| sim vortex_nn_2026 | 36 | 20 | 17 | 19 | 20 | 19 | 18 | 20 | 19 |
| sim saddle_nn_2026 | 36 | 23 | 20 | 22 | 22 | 22 | 19 | 23 | 23 |

The simulation starts 5 cm inside or outside r_d (r_d 0.01 to 0.5 m, 20 runs each, hardware gains).

- **Hardware:** no start-up transient. The runs start near their circle, and the 23 to 44 mm level is
  the lap-periodic swing of the radius.
- **Simulation:** the transient falls to 2 mm by 3 s (linear field). On the reconstructed fields it
  reaches their steady swing by about 2 s.
- **Effect on bias:** across all 2026 runs, the mean radial bias with trims of 0, 2, 5 and 10 s
  differs by at most 4 mm (settling_2026.csv).
- **Choice:** T_tr = 3 s (was 10 s), for simulation and hardware. Statistics at 2, 5 and 10 s go in
  the report for R1.4.
