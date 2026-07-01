# Real Robot Data

This directory contains trajectory data from physical robot experiments for sim-to-real comparison.

## Data Files

### orbit040_trajectory.csv

**Source**: `trunk/robots_3/vortex_tests/orbit040.mat`

**Experiment Parameters**:
- Number of robots: 3
- **Desired orbit radius: 0.4 meters (actual, not scaled)**
- Control primitive: Orbiter (plane fitting method)
- Vector field: Vortex (fixed vortex, V ∝ r)
- Duration: ~60 seconds (601 samples)
- Timestep: 0.1 seconds

**Data Format**:
- Column 1: `time_s` - Time in seconds
- Column 2: `x_m` - X position of cluster centroid in meters
- Column 3: `y_m` - Y position of cluster centroid in meters

**Preprocessing**:
- Cluster centroid calculated as mean of 3 robot positions
- **No scaling applied - robot was already operating at correct physical scale**
- **Control radius: 0.4m (physical experiment)**
- Origin at (0, 0) - vortex center

**Usage**:
```python
import pandas as pd
real_data = pd.read_csv('real_robot_data/orbit040_trajectory.csv')
time = real_data['time_s'].values
x = real_data['x_m'].values
y = real_data['y_m'].values
```

## Export Script

Data exported using: `trunk/robots_3/vortex_tests/export_orbit040_to_csv.m`
