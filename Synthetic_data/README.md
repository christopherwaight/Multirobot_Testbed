# Synthetic_data

Placeholder data for building the data pipeline and post-processing. Everything under `synthetic/` is fabricated. Do not quote a number from it in any paper or note. Replace it with real logs and keep the file layout and variable names, so the pipeline does not change.

Regenerate with `trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3 Synthetic_data/generate_synthetic.py` (fixed seed, paths built from `__file__`). `manifest.csv` lists every file with an `origin` column that reads `real` or `SYNTHETIC`. The `.mat` files are git-ignored by the repo rules, so only this README, the script and the manifest are committable.

## Layout

| Folder | Origin | Contents |
|---|---|---|
| `real_copy/attraction_3robot/{vortex,saddle}/` | real, unmodified | `run1..80` (vortex), 78 runs (saddle, 43 and 64 absent). Copied from `trunk/robots_3`, three-robot logs only. |
| `real_copy/orbit_3robot/{vortex,saddle}/` | real, unmodified | The orbit logs that exist (`radius001`, `radius01..05`, plus `radius06` for the saddle). Old mixed-gain runs, kept for calibration. |
| `synthetic/orbit_3robot/{vortex,saddle}/` | SYNTHETIC | 8 radii x 10 trials per field, files `rdXXX_trialKK.mat` where XXX is the radius in mm (010 is 0.01 m, 500 is 0.50 m). |
| `synthetic/attraction_4robot/{vortex,saddle}/` | SYNTHETIC | One file per real three-robot run, same run number. |
| `synthetic/orbit_4robot/{vortex,saddle}/` | SYNTHETIC | Same radius and trial grid as the three-robot orbit. |

Commanded radii are 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50 m. The first, third, fifth, sixth, seventh and eighth match radii that were run on hardware. 0.05 and 0.15 are new.

## File format

Metres, critical point at the origin, 10 Hz, as in the real logs, so `revision_metrics/hardware.py` loads them as is. Three-robot files carry `tout`, `cluster_pose` (x, y, heading, p, q, beta, vx, vy, omega), `robot1..3_pose`, `detected_critical_point`, `desired_velocities`. Four-robot files follow the existing four-robot logs, with `tout`, `cluster_position` (robot mean), `robot1..4_pose`, `detected_critical_point`, `detected_critical_point_avg`. The `hue_`, `sat_`, `Jhat` and gradient-descent channels are not generated.

## How the synthetic runs are made

**Orbit.** For each field the script reads the real orbit logs and takes, per commanded radius, the mean radial bias, the radial standard deviation, the scatter of the critical point estimate, and the tangential speed k_t (about 0.14 m/s vortex, 0.20 m/s saddle, clockwise). These are interpolated to the eight radii. A trial then draws

- a radius that settles from near the command with a 3 s time constant, plus the interpolated bias with a trial-to-trial spread of 15 percent of its size,
- ripple split equally between a once-per-revolution term (field-fit error) and a 2 s correlated term,
- angular rate k_t over the current radius, with a 2 percent trial spread,
- a per-trial centre offset for the estimate, and a correlated estimate noise on `detected_critical_point`,
- robots at the formation offsets read from the real logs (three robots) or a 0.24 m square (four robots), with 1 mm tracker noise and 4 mm correlated shape error.

Four-robot orbits use 0.8 times the three-robot bias and scatter. That factor is an assumption, not a measurement.

**Four-robot attraction.** Each real three-robot centroid path is time-stretched by a random factor near 1.05, given a start jitter that fades out, a small parked-offset noise that grows with convergence, and correlated tracker noise. A fourth robot is added and the reported centroid is the robot mean. The estimate trace is scaled by 0.85. Final-distance means land at 0.037 m (vortex) and 0.025 m (saddle), the same as the real three-robot runs, by construction.

## Caveats

- The orbit calibration comes from the old mixed-gain orbit logs, which are being redone. The synthetic biases (0.12 to 0.17 m at small radii) inherit that and are larger than the paper's tables imply.
- `desired_velocities` is the finite-difference velocity of the noise-free centroid times 1.1, with a zero third column. It is not a controller output.
- Real orbit logs contain occasional estimate blow-ups from an ill-conditioned fit (vortex `radius04`). The synthetic runs do not, so a pipeline outlier filter will not be exercised by them.
- Heading is nearly constant with small noise. Real logs may wander more.
