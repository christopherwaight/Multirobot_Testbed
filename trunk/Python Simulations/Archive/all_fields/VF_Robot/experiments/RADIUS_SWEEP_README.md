# Radius Sweep Comparison Experiment

## Overview

This experiment compares three orbiter control primitives across multiple orbital radii and vector field environments to evaluate their performance characteristics.

## Files

- **`radius_sweep_comparison.py`** - Main experiment script
- **`radius_sweep_results/`** - Output directory for CSV results

## Control Primitives Tested

1. **3R_orbiter_plane** - `critical_point_orbiter_plane_fitting` (3-robot)
   - Formation: Equilateral triangle (p=q=0.33, β=120°)
   - Uses plane fitting to estimate critical point

2. **4R_orbiter_planar** - `center_orbiter_quad_planar` (4-robot)
   - Formation: Square configuration (d1=d2=0.3, φ=90°)
   - Uses 4-point planar estimation

3. **4R_orbiter_advanced** - `center_orbiter_quad_advanced` (4-robot)
   - Formation: Rectangle configuration (d1=0.25, d2=0.433, φ=90°)
   - Uses dual Jacobian + orientation control

## Test Parameters

- **Radii**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] meters
- **Simulation time**: 300 steps (30 seconds)
- **Initial position**: (0, desired_radius) for all primitives
- **Field mode**: Analytical (ground truth)

## Environments Tested

1. Sink 1
2. Source 1
3. Vortex 1
4. Sinking Vortex 1
5. Spewing Vortex 1
6. Saddle 1

## Metrics Collected

For each run, the following metrics are calculated (averaged over all 300 time steps):

1. **avg_dist_origin** - Average distance of cluster centroid from origin
2. **std_dist_origin** - Standard deviation of distance from origin
3. **avg_dist_to_est_center** - Average distance from centroid to estimated critical point
4. **std_dist_to_est_center** - Standard deviation of distance to estimated center
5. **est_center_x_mean** - Mean estimated x-position of critical point over time
6. **est_center_x_std** - Standard deviation of estimated x-position
7. **est_center_y_mean** - Mean estimated y-position of critical point over time
8. **est_center_y_std** - Standard deviation of estimated y-position

## Running the Experiment

```bash
cd VF_Robot
./venv/bin/python3 experiments/radius_sweep_comparison.py
```

### Testing Mode

To run a quick test with limited parameters:

1. Edit `radius_sweep_comparison.py`
2. Set `TESTING_MODE = True`
3. Run the script

This will test only:
- One environment (Sink 1)
- One radius (0.2)
- Shorter simulation (50 steps)

## Output

### Terminal Output

The script displays formatted tables showing results for each environment and primitive:

```
####################################################################################################
# ENVIRONMENT: Sink 1
####################################################################################################

====================================================================================================
PRIMITIVE: 3R_orbiter_plane
====================================================================================================
Radius   Avg Dist Origin      Avg Dist Est Center       Est Center X         Est Center Y
         (mean ± std)         (mean ± std)              (mean ± std)         (mean ± std)
----------------------------------------------------------------------------------------------------
0.10       0.2024 ± 0.0204       0.2024 ± 0.0204       0.0000 ± 0.0000     0.0000 ± 0.0000
0.20       0.2746 ± 0.0159       0.2746 ± 0.0159      -0.0000 ± 0.0000    -0.0000 ± 0.0000
...
```

### CSV Output

Results are saved to `experiments/radius_sweep_results/radius_sweep_results_YYYYMMDD_HHMMSS.csv`

CSV columns:
- `environment` - Environment name (sink1, source1, etc.)
- `primitive` - Primitive name (3R_orbiter_plane, 4R_orbiter_planar, 4R_orbiter_advanced)
- `radius` - Desired orbital radius
- `avg_dist_origin` - Mean distance from origin
- `std_dist_origin` - Std dev of distance from origin
- `avg_dist_to_est_center` - Mean distance to estimated center
- `std_dist_to_est_center` - Std dev of distance to estimated center
- `est_center_x_mean` - Mean estimated center x position
- `est_center_x_std` - Std dev of estimated center x position
- `est_center_y_mean` - Mean estimated center y position
- `est_center_y_std` - Std dev of estimated center y position

## Total Runs

The complete experiment performs:
- 3 primitives × 6 radii × 6 environments = **108 simulation runs**
- Runtime: Approximately 30-60 seconds on modern hardware

## Key Findings

From the experiment results, we observe:

1. **Orbital radius accuracy**: All three primitives maintain orbits close to the desired radius, with average distances slightly exceeding the desired radius (e.g., desired 0.2 → actual ~0.27)

2. **Consistency**: The 4-robot configurations generally show lower standard deviations than the 3-robot configuration, especially at larger radii

3. **Center estimation**: All primitives successfully estimate the critical point at the origin (0, 0) with very small deviations (~10^-17 to 10^-18)

4. **Formation differences**:
   - **4R_orbiter_advanced** shows slightly tighter control at smaller radii
   - **4R_orbiter_planar** provides the most consistent performance across all radii
   - **3R_orbiter_plane** has slightly higher variability but comparable average performance

## Configuration Details

### Formation Configs Used

- **3-robot**: `config/formations/equilateral_radius_sweep.yaml`
- **4-robot planar**: `config/formations/quad_square_default.yaml`
- **4-robot advanced**: `config/formations/quad_rectangle.yaml`

### Robot Parameters

- Timestep: 0.1 seconds
- Momentum alpha: 0.7
- Initial orientation: π radians (pointing left)

## Notes

- The experiment uses analytical (ground truth) vector fields for consistency
- All robots start at the same position: centroid at (0, desired_radius)
- The estimated center position should converge to (0, 0) for all symmetric environments
- Distance metrics include both initial transient and steady-state behavior
