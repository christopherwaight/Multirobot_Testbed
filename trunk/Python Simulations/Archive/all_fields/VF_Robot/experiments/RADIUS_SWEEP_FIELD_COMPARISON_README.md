# Radius Sweep Field Mode Comparison Experiment

## Overview

This experiment compares four orbiter control primitives across multiple orbital radii (0.1 to 0.6 meters) and **three field approximation modes** (Analytical, RBF, NN) to evaluate how field representation affects orbital control performance.

## Files

- **`radius_sweep_field_comparison.py`** - Main experiment script
- **`radius_sweep_results/`** - Output directory for CSV results

## Control Primitives Tested

1. **3R_orbiter_plane** - `critical_point_orbiter_plane_fitting` (3-robot)
   - Formation: Equilateral triangle (p=q=0.33, β=120°)
   - Uses plane fitting to estimate critical point and orbit around it

2. **4R_orbiter_planar** - `center_orbiter_quad_planar` (4-robot)
   - Formation: Square configuration (d1=d2=0.3, φ=90°)
   - Uses 4-point planar estimation and orbits around center

3. **4R_orbiter_advanced** - `center_orbiter_quad_advanced` (4-robot)
   - Formation: Rectangle configuration (d1=0.25, d2=0.433, φ=90°)
   - Uses dual Jacobian + orientation control and orbits around center

4. **3R_vector_sum** - `vector_sum` (3-robot)
   - Formation: Equilateral triangle (p=q=0.33, β=120°)
   - Simple field following (baseline control - just moves with vector sum)
   - Does NOT orbit - provides comparison for non-orbiting behavior

## Field Modes Tested

1. **Analytical** - Ground truth mathematical field functions
2. **RBF** - Radial Basis Function interpolator approximation (no NN)
3. **NN** - Neural network approximation only (no RBF)

**No blending** is used - each mode is tested independently.

## Test Parameters

- **Radii**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] meters
- **Simulation time**: 300 steps (30 seconds)
- **Timestep**: 0.1 seconds
- **Initial position**: (0, desired_radius) for all primitives

## Environments Tested

1. **Sink 1** - Uses sinking_vortex_predictors for ML models
2. **Source 1** - Uses sinking_vortex_predictors for ML models
3. **Vortex 1** - Uses vortex_predictors for ML models
4. **Sinking Vortex 1** - Uses sinking_vortex_predictors for ML models
5. **Spewing Vortex 1** - Uses sinking_vortex_predictors for ML models
6. **Saddle 1** - Uses saddle_predictors for ML models

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
python3 experiments/radius_sweep_field_comparison.py
```

### Testing Mode

To run a quick test with limited parameters:

1. Edit `radius_sweep_field_comparison.py`
2. Set `TESTING_MODE = True`
3. Run the script

This will test only:
- One environment (Vortex 1)
- First primitive only (3R_orbiter_plane)
- Two radii (0.2, 0.3)
- Two field modes (analytical, rbf)
- Shorter simulation (50 steps)

## Output

### Terminal Output

The script displays formatted tables showing results for each environment, primitive, and field mode:

```
####################################################################################################
# ENVIRONMENT: Vortex 1
####################################################################################################

===============================================================================================================
PRIMITIVE: 3R_orbiter_plane | FIELD MODE: ANALYTICAL
===============================================================================================================
Radius   Avg Dist Origin      Avg Dist Est Center       Est Center X         Est Center Y
         (mean ± std)         (mean ± std)              (mean ± std)         (mean ± std)
---------------------------------------------------------------------------------------------------------------
0.10       0.2024 ± 0.0204       0.2024 ± 0.0204       0.0000 ± 0.0000     0.0000 ± 0.0000
0.20       0.2746 ± 0.0159       0.2746 ± 0.0159      -0.0000 ± 0.0000    -0.0000 ± 0.0000
...
```

### CSV Output

Results are saved to `experiments/radius_sweep_results/radius_sweep_field_comparison_YYYYMMDD_HHMMSS.csv`

CSV columns:
- `environment` - Environment name (sink1, source1, vortex1, etc.)
- `primitive` - Primitive name (3R_orbiter_plane, 4R_orbiter_basic, etc.)
- `field_mode` - Field approximation mode (analytical, rbf, nn)
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
- **4 primitives × 3 field modes × 6 radii × 6 environments = 432 simulation runs**
- Runtime: Approximately 10-20 minutes on modern hardware

## Research Questions

This experiment helps answer:

1. **How does field approximation affect orbital control?**
   - Do NN and RBF approximations maintain similar orbital accuracy as analytical fields?
   - Which field mode provides the most stable orbits?

2. **Are ML-based fields sufficient for real-world deployment?**
   - Can RBF or NN models replace analytical fields without degrading performance?
   - Which approximation method is more robust across different radii?

3. **Do different primitives react differently to field approximation errors?**
   - Are 4-robot configurations more robust to ML approximation errors?
   - Does the advanced primitive with orientation control compensate better for field errors?

4. **How does performance scale with orbital radius?**
   - Do errors compound at larger radii?
   - Is there an optimal radius for ML-based field approximations?

## Expected Findings

We expect to observe:

1. **Analytical mode**: Provides baseline "ground truth" performance
2. **RBF mode**: Generally accurate but may have interpolation artifacts at boundaries
3. **NN mode**: May show slight biases but potentially smoother across the domain

Differences between modes will reveal:
- The impact of field approximation errors on control performance
- Whether ML models are accurate enough for closed-loop control
- Trade-offs between RBF and NN approaches

## Configuration Details

### Formation Configs Used

- **3-robot**: `config/formations/equilateral_radius_sweep.yaml`
- **4-robot basic/planar**: `config/formations/quad_square_default.yaml`
- **4-robot advanced**: `config/formations/quad_rectangle.yaml`

### ML Model Directories

The experiment automatically selects the appropriate predictor directory for each environment:
- Vortex environments → `vortex_predictors/`
- Saddle environments → `saddle_predictors/`
- Other environments → `sinking_vortex_predictors/`

## Notes

- All robots start at the same initial position: centroid at (0, desired_radius)
- The estimated center position should converge to (0, 0) for all symmetric environments
- Distance metrics include both initial transient and steady-state behavior
- Each field mode uses the same robot controller - only the field representation changes

## Comparison with Original Radius Sweep

The original `radius_sweep_comparison.py` tests:
- 3 primitives (not 4)
- Only analytical fields
- **108 total runs**

This new experiment:
- 4 primitives
- 3 field modes (analytical, rbf, nn)
- **432 total runs** (4× more comprehensive)
