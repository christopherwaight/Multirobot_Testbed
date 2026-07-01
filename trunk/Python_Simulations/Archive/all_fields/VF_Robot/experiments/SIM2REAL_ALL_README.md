# Sim2Real Comparison All - Usage Guide

## Overview
This comprehensive sim-to-real comparison system processes multiple robot configurations and radius values, comparing simulated trajectories with real robot data.

## Step-by-Step Instructions

### 1. Export MATLAB Data

Navigate to the MATLAB data directory:
```bash
cd trunk/robots_3/vortex_tests/
```

Run the MATLAB export script:
```matlab
% In MATLAB
export_all_orbit_trajectories
```

This will:
- Process all orbit*.mat files (orbit001.mat through orbit250.mat)
- Extract cluster trajectory data (x, y positions over time)
- Export to CSV files in `real_robot_trajectories/` folder
- Generate a summary file with configuration details

### 2. Copy Data to Python Directory

Copy the exported data folder:
```bash
cp -r real_robot_trajectories/ "../../../Python Simulations/Vector_Fields/VF_Robot/real_robot_data/"
```

Or manually copy the `real_robot_trajectories/` folder to:
`trunk/Python Simulations/Vector_Fields/VF_Robot/real_robot_data/`

### 3. Run the Comparison

Navigate to the Python experiments directory:
```bash
cd "trunk/Python Simulations/Vector_Fields/VF_Robot/experiments"
```

Run the comprehensive comparison:
```bash
python3 sim2real_comparison_all.py
```

## Configuration Details

### Orbit Naming Convention
- **0XX series** (e.g., orbit001, orbit040, orbit070)
  - 3-robot equilateral triangle formation
  - Radius = XX/100 meters (001 = 0.01m, 040 = 0.40m)
  - Uses `critical_point_orbiter_plane_fitting` control primitive

- **1XX series** (e.g., orbit101, orbit140, orbit160)
  - 4-robot square formation (d1=0.3, d2=0.3)
  - Radius = (XX-100)/100 meters (101 = 0.01m, 140 = 0.40m)
  - Uses `center_orbiter_quad` control primitive

- **2XX series** (e.g., orbit201, orbit240, orbit250)
  - 4-robot advanced formation (d1=0.433, d2=0.25)
  - Radius = (XX-200)/100 meters (201 = 0.01m, 240 = 0.40m)
  - Uses `center_orbiter_quad_advanced` control primitive

## Output Files

The script generates:

1. **CSV Results File**: `sim2real_results/sim2real_comparison_results.csv`
   - Contains all statistics for each configuration
   - Columns include:
     - config_name, robot_type, radius_m
     - avg_dist_origin, std_dist_origin
     - avg_dist_center, std_dist_center
     - avg_x_est, std_x_est, avg_y_est, std_y_est
     - rmse
     - real_avg_dist, real_center_x, real_center_y

2. **Comparison Plots**: `sim2real_results/orbit*_comparison.png`
   - Visual comparison of simulated vs real trajectories
   - Statistics panel with detailed metrics

## Statistics Calculated

For each configuration, the script calculates:

- **Average distance from origin** ± standard deviation
  - How well the robot maintains the desired radius

- **Average distance from estimated center** ± standard deviation
  - Consistency of the orbital path

- **Average estimated center position** (x, y) ± standard deviation
  - Where the robot thinks the center is

- **RMSE** (Root Mean Square Error)
  - Overall position accuracy metric

## Simulation Parameters

Default settings (can be modified in `sim2real_comparison_all.py`):
- **NUM_TRIALS**: 4 simulations per configuration
- **SIM_TIME**: 600 timesteps (60 seconds at 0.1s/step)
- **Starting position**: (0, radius) for each trial
- **Environment**: vortex1 (analytical field)

## Customization

To modify the comparison:

1. **Change number of trials**:
   ```python
   NUM_TRIALS = 10  # Increase for more statistical significance
   ```

2. **Change simulation duration**:
   ```python
   SIM_TIME = 1000  # 100 seconds
   ```

3. **Use different field modes**:
   - Modify `sim2real_comparison_all.py` to use NNField or RBFField
   - Change the field creation line in `run_simulation()`

4. **Add more metrics**:
   - Modify `calculate_statistics()` function
   - Add new columns to the results dictionary

## Troubleshooting

### "No trajectory files found"
- Ensure you've run the MATLAB export script first
- Check that the `real_robot_trajectories/` folder exists
- Verify the path in `REAL_DATA_DIR` variable

### "Formation config file not found"
- Check that these files exist:
  - `config/formations/equilateral_default.yaml`
  - `config/formations/quad_square.yaml`
  - `config/formations/quad_default.yaml`

### Memory issues with many configurations
- Reduce `MAX_PLOTS_PER_CONFIG` to save fewer plots
- Process configurations in batches by modifying the main loop

## Analysis Tips

1. **Compare across radii**: Look for trends as radius increases
2. **Compare robot types**: See which formation performs best
3. **Check convergence**: Verify simulations reach steady state
4. **Validate centers**: Compare estimated vs true centers

## Example Analysis

After running, you can analyze results:

```python
import pandas as pd

# Load results
df = pd.read_csv('sim2real_results/sim2real_comparison_results.csv')

# Group by robot type
by_type = df.groupby('robot_type')['rmse'].agg(['mean', 'std'])
print(by_type)

# Find best performing radius for each type
best_radius = df.loc[df.groupby('robot_type')['rmse'].idxmin()]
print(best_radius[['robot_type', 'radius_m', 'rmse']])
```