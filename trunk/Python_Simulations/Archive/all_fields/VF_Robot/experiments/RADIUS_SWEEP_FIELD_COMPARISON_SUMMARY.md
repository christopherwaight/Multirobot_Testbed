# Radius Sweep Field Comparison - Quick Reference

## What This Experiment Does

Tests **4 control primitives** across **6 radii** using **3 field modes** (Analytical, RBF, NN) on **6 environments**.

**Total: 432 simulation runs**

## Four Control Primitives

1. **3R_orbiter_plane** - 3-robot orbiter (plane fitting)
2. **4R_orbiter_planar** - 4-robot orbiter (planar estimation)
3. **4R_orbiter_advanced** - 4-robot orbiter (dual Jacobian + orientation control)
4. **3R_vector_sum** - 3-robot baseline (simple field following, NO orbiting)

## Three Field Modes

1. **Analytical** - Ground truth (mathematical functions)
2. **RBF** - RBF interpolator only (no NN)
3. **NN** - Neural network only (no RBF)

No blending is used.

## Running the Experiment

### Full Experiment (432 runs, ~15 minutes)
```bash
cd "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python Simulations/Vector_Fields/VF_Robot"
./venv/bin/python3 experiments/radius_sweep_field_comparison.py
```

### Quick Test (4 runs, ~10 seconds)
Edit `radius_sweep_field_comparison.py` and set:
```python
TESTING_MODE = True
```

Then run the same command.

## Output Files

- **CSV**: `experiments/radius_sweep_results/radius_sweep_field_comparison_YYYYMMDD_HHMMSS.csv`
- **Terminal**: Real-time formatted tables

## CSV Columns

- `environment` - sink1, source1, vortex1, sinking_vortex1, spewing_vortex1, saddle1
- `primitive` - 3R_orbiter_plane, 4R_orbiter_planar, 4R_orbiter_advanced, 3R_vector_sum
- `field_mode` - analytical, rbf, nn
- `radius` - 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
- `avg_dist_origin` - Mean distance from (0,0)
- `std_dist_origin` - Std dev of distance from origin
- `avg_dist_to_est_center` - Mean distance to estimated critical point
- `std_dist_to_est_center` - Std dev of distance to estimated center
- `est_center_x_mean` - Mean x-position of estimated center
- `est_center_x_std` - Std dev of estimated center x
- `est_center_y_mean` - Mean y-position of estimated center
- `est_center_y_std` - Std dev of estimated center y

## Key Differences from Original Radius Sweep

**Original (`radius_sweep_comparison.py`)**:
- 3 primitives (3R_orbiter, 4R_planar, 4R_advanced)
- Analytical fields only
- 108 total runs

**This Experiment (`radius_sweep_field_comparison.py`)**:
- 4 primitives (adds vector_sum baseline)
- 3 field modes (analytical, rbf, nn)
- 432 total runs (4× more comprehensive)

## What to Look For in Results

### Orbiting Primitives (3R_orbiter_plane, 4R_orbiter_planar, 4R_orbiter_advanced)
- `avg_dist_origin` should be close to desired radius
- `std_dist_origin` shows orbital stability (lower = more stable)
- `est_center_x_mean` and `est_center_y_mean` should be near (0, 0)

### Vector Sum Primitive (3R_vector_sum)
- Does NOT orbit, just follows field
- Will likely converge to critical point or exhibit different behavior
- Provides baseline for comparison

### Field Mode Comparison
- **Analytical** = ground truth performance
- **RBF** vs **NN** = which ML approximation is more accurate?
- Look for differences in std dev (stability) between modes

## Test Run Verification

The test successfully ran and produced:
```
Total runs completed: 4
Results saved to: experiments/radius_sweep_results/radius_sweep_field_comparison_20251109_233104.csv
```

Sample output:
- Analytical mode: avg_dist_origin = 0.249 (r=0.2), 0.336 (r=0.3)
- RBF mode: avg_dist_origin = 0.266 (r=0.2), 0.332 (r=0.3)
- Both successfully orbit with small deviations from desired radius

## Documentation

See `RADIUS_SWEEP_FIELD_COMPARISON_README.md` for comprehensive documentation.
