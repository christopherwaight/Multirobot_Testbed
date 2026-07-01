# Velocity Plotting Guide

This guide explains how to generate and interpret robot velocity plots in the VF_Robot simulation system.

## Overview

The velocity plotting system tracks each robot's velocity throughout the simulation and generates time-series plots showing:
- **Velocity magnitude** (speed in m/s) for each robot over time
- **Velocity components** (vx and vy in m/s) for detailed analysis

## Configuration

### Enable Velocity Plotting

In `experiments/main_omni.py`, set:

```python
SAVE_VELOCITY_PLOTS = True  # Enable velocity plot generation
VELOCITY_PLOT_DIR = "velocity_plots"  # Output directory
```

### Time Scale

- **Timestep**: 0.1 (default) = 0.1 seconds per simulation step
- **Distances**: All distances are in meters
- **Velocities**: All velocities are in m/s

To change the timestep, modify the cluster initialization:
```python
cluster = OmniCluster(config_path, field, timestep=0.1)
```

## Usage

### Multi-Environment Mode (Recommended)

When running in `multi_env` mode, velocity plots are automatically generated for each environment:

```python
# In experiments/main_omni.py
SIMULATION_MODE = "multi_env"
SAVE_VELOCITY_PLOTS = True
```

Run the simulation:
```bash
cd VF_Robot
python3 experiments/main_omni.py
```

**Output**: One velocity plot per environment in `velocity_plots/` directory
- Format: `{environment_name}_velocity_{mode}.png`
- Example: `sink1_velocity_analytical.png`

### Single Environment Mode

For single simulations:

```python
# In experiments/main_omni.py
SIMULATION_MODE = "single"
ENVIRONMENT = "sinking_vortex1"
SAVE_VELOCITY_PLOTS = True
```

**Output**: `velocity_plots/sinking_vortex1_velocity_analytical.png`

### Standalone Plotting Script

Use the standalone script to run a custom simulation and generate plots:

```bash
cd VF_Robot
python3 experiments/plot_velocities.py
```

This script:
1. Runs a sample simulation (sinking_vortex1)
2. Generates both magnitude and component plots
3. Demonstrates the velocity plotting API

## Plot Interpretation

### Velocity Magnitude Plot

The velocity magnitude plot shows the speed (|v| = √(vx² + vy²)) of each robot over time.

**Structure**:
- **Top N subplots**: Individual robot velocities (Robot 1, Robot 2, Robot 3, Robot 4)
- **Bottom subplot**: All robots overlaid for comparison

**What to look for**:
- **Steady-state behavior**: Does velocity stabilize over time?
- **Oscillations**: Are there periodic fluctuations?
- **Convergence**: Do velocities decrease as robots approach critical point?
- **Relative speeds**: Which robot moves fastest/slowest?

**Typical patterns**:
- **Sink fields**: Velocities increase as robots approach the sink
- **Vortex fields**: Velocities stabilize in circular orbits
- **Saddle points**: Velocities may oscillate near unstable equilibrium

### Velocity Components Plot

Shows vx (x-direction) and vy (y-direction) components separately.

**Structure**:
- Each robot has 2 subplots: vx (left column) and vy (right column)
- Horizontal dashed line at y=0 for reference

**What to look for**:
- **Sign changes**: Indicate direction reversals
- **Magnitude asymmetry**: vx ≠ vy suggests anisotropic motion
- **Oscillation patterns**: Different frequencies in x vs y
- **Correlation**: Do vx and vy oscillate together or independently?

**Typical patterns**:
- **Orbital motion**: Sinusoidal vx and vy with 90° phase shift
- **Radial motion**: vx and vy maintain constant ratio
- **Complex fields**: Irregular patterns with multiple frequencies

## API Reference

### Plot Robot Velocities

```python
from src.simulation.velocity_plotter import plot_robot_velocities

plot_robot_velocities(
    velocity_history,      # List of velocity arrays [(timesteps, num_robots, 2)]
    timestep=0.1,          # Time step in seconds
    title="Robot Velocities",  # Plot title
    save_path=None,        # Path to save (or None to display)
    num_robots=3           # Number of robots (3 or 4)
)
```

### Plot Velocity Components

```python
from src.simulation.velocity_plotter import plot_velocity_components

plot_velocity_components(
    velocity_history,      # List of velocity arrays
    timestep=0.1,          # Time step in seconds
    title="Velocity Components",  # Plot title
    save_path=None,        # Path to save (or None to display)
    num_robots=3           # Number of robots
)
```

### Get Velocity History from Cluster

```python
# After running simulation
velocity_history = cluster.get_velocity_history()

# velocity_history is a list of arrays
# Each element: shape (num_robots, 2) with [[vx1, vy1], [vx2, vy2], ...]
# Length: number of simulation time steps
```

## Data Format

### Velocity History Structure

```python
velocity_history = [
    np.array([[vx1_t0, vy1_t0],   # Robot 1 at time 0
              [vx2_t0, vy2_t0],   # Robot 2 at time 0
              [vx3_t0, vy3_t0]]), # Robot 3 at time 0
    np.array([[vx1_t1, vy1_t1],   # Robot 1 at time 1
              [vx2_t1, vy2_t1],
              [vx3_t1, vy3_t1]]),
    # ... one entry per simulation step
]
```

### Time Calculation

```python
time_seconds = timestep * step_index
total_duration = timestep * len(velocity_history)
```

Example: 150 steps at 0.1s/step = 15 seconds total

## Examples

### Example 1: Generate Plots for All Variant 1 Environments

```python
# In main_omni.py
NUM_ROBOTS = 3
SIMULATION_MODE = "multi_env"
FIELD_MODE = "analytical"
CONTROL_PRIMITIVE_3 = "critical_point_orbiter_plane_fitting"
SAVE_VELOCITY_PLOTS = True
```

Run: `python3 experiments/main_omni.py`

**Output**: 6 velocity plots (sink1, source1, vortex1, sinking_vortex1, spewing_vortex1, saddle1)

### Example 2: Compare Field Approximation Methods

Run 4 simulations with different field modes:

```bash
# Set FIELD_MODE to "analytical", run, check velocity_plots/
# Set FIELD_MODE to "nn", run, check velocity_plots/
# Set FIELD_MODE to "rbf", run, check velocity_plots/
# Set FIELD_MODE to "blended", run, check velocity_plots/
```

Compare files:
- `sink1_velocity_analytical.png`
- `sink1_velocity_nn.png`
- `sink1_velocity_rbf.png`
- `sink1_velocity_blended.png`

### Example 3: Custom Analysis Script

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Vortex import vortex1
import src.control.primitives as ocp
import numpy as np

# Setup
field = AnalyticalField(vortex1)
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

# Run simulation
for _ in range(200):
    cluster.move(ocp.critical_point_orbiter_plane_fitting)

# Get data
velocity_history = cluster.get_velocity_history()
vel_array = np.array(velocity_history)  # Shape: (200, 3, 2)

# Compute statistics
vel_magnitudes = np.linalg.norm(vel_array, axis=2)  # Shape: (200, 3)
mean_vel = np.mean(vel_magnitudes, axis=0)  # Mean velocity for each robot
max_vel = np.max(vel_magnitudes, axis=0)

print(f"Mean velocities: {mean_vel}")
print(f"Max velocities: {max_vel}")

# Plot
from src.simulation.velocity_plotter import plot_robot_velocities
plot_robot_velocities(velocity_history, save_path="my_analysis.png")
```

## Troubleshooting

**Problem**: Empty velocity history

**Solution**: Make sure simulation runs before accessing velocity history:
```python
# Wrong
cluster = OmniCluster(...)
velocity_history = cluster.get_velocity_history()  # Empty!

# Correct
cluster = OmniCluster(...)
for _ in range(150):
    cluster.move(control_primitive)
velocity_history = cluster.get_velocity_history()  # Populated
```

**Problem**: Velocity plots directory not found

**Solution**: The directory is created automatically. Check write permissions.

**Problem**: Unrealistic velocity values

**Solution**: Check timestep and momentum_alpha settings. High momentum or large timestep can cause instability.

## Advanced Usage

### Custom Plot Styling

Modify `src/simulation/velocity_plotter.py` to customize:
- Line colors and styles
- Figure sizes
- Grid appearance
- Legend placement
- Axis labels and units

### Batch Processing

Generate plots for multiple configurations:

```python
configurations = [
    ("sink1", "analytical"),
    ("sink1", "nn"),
    ("vortex1", "analytical"),
    # ...
]

for env, mode in configurations:
    # Run simulation with env and mode
    # Generate velocity plot
    # Compare results
```

## Related Documentation

- **Main documentation**: `VF_Robot/CLAUDE.md`
- **Parent project**: `Python Simulations/CLAUDE.md`
- **Control primitives**: `src/control/primitives.py`
- **Cluster classes**: `src/robot/omni_cluster.py`, `src/robot/quad_cluster.py`
