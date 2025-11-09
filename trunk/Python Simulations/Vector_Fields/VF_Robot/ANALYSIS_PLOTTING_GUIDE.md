# Analysis Plotting Guide

**For future sessions**: This guide explains how to add new analysis plots to the simulation system.

## Current Infrastructure

### Existing Velocity Plotting System

**Location**: `src/simulation/velocity_plotter.py`

**What it does**:
- Tracks robot velocities at each timestep
- Generates plots showing velocity magnitude vs time
- Automatically saves to `velocity_plots/` directory

**How it works**:
1. `OmniCluster` and `QuadCluster` store velocity history in `self.velocity_history`
2. Each timestep, velocities are appended in `move()` method
3. After simulation, `plot_robot_velocities()` generates plots

**Key code locations**:
- Data collection: `src/robot/omni_cluster.py` line 174-175, `src/robot/quad_cluster.py` line 217-218
- Plotting function: `src/simulation/velocity_plotter.py`
- Integration: `experiments/main_omni.py` lines 405-420 (multi_env mode)

## How to Add New Analysis Plots

### Step 1: Identify What to Track

**Examples of useful data to track**:
- Estimated critical point location over time
- Distance to critical point over time
- Jacobian eigenvalues over time
- Curl and divergence over time
- Formation shape parameters (p, q, β for 3-robot or d1, d2, r1, r2, φ for 4-robot)
- Individual robot trajectories
- Control commands (vx_c, vy_c) vs actual velocities
- Radius from center (for orbital primitives)

### Step 2: Add Data Collection to Cluster Classes

**Template** (add to `omni_cluster.py` or `quad_cluster.py`):

```python
# In __init__:
self.your_data_history = []

# In move() method, AFTER commanding robots:
your_data = calculate_your_data(self)  # Your calculation here
self.your_data_history.append(your_data.copy())

# Add getter method:
def get_your_data_history(self):
    """Get history of your data as list/array."""
    return self.your_data_history

# In reset() method:
self.your_data_history = []
```

### Step 3: Create Plotting Function

**Template** (add to `src/simulation/velocity_plotter.py` or create new file):

```python
def plot_your_data(data_history, timestep=0.1, title="Your Data",
                   save_path=None):
    """
    Plot your data vs time.

    Args:
        data_history: List of data values for each timestep
        timestep: Time step in seconds (default 0.1s)
        title: Plot title
        save_path: Path to save (if None, displays instead)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if len(data_history) == 0:
        print("No data to plot")
        return

    # Convert to numpy array
    data_array = np.array(data_history)
    num_steps = len(data_array)

    # Create time array in seconds
    time = np.arange(num_steps) * timestep

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot your data
    ax.plot(time, data_array, 'b-', linewidth=2, label='Your Data')

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Your Data Units', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
```

### Step 4: Integrate into Simulation

**Add to `experiments/main_omni.py`** in the simulation loop (e.g., in `run_multi_environment_simulation()`):

```python
# After running simulation
if SAVE_YOUR_PLOTS:
    your_data_history = cluster.get_your_data_history()
    if len(your_data_history) > 0:
        plot_filename = f"analysis_plots/{env_key}_your_data_{mode_str.lower()}.png"
        plot_title = f"{env_name} - Your Data ({mode_str})"

        plot_your_data(
            your_data_history,
            timestep=cluster.timestep,
            title=plot_title,
            save_path=plot_filename
        )
```

## Specific Example: Critical Point Estimation

### What to Track

For primitives like `critical_point_plane_fitting`, track:
- Estimated critical point location (x, y)
- Distance from cluster centroid to estimated center
- True center (0, 0 for most environments)

### Code Implementation

**1. Add to control primitive** (`src/control/primitives.py`):

```python
def critical_point_plane_fitting(cluster):
    # ... existing code ...

    # Calculate critical point
    critical_point = np.linalg.solve(J, rhs)

    # Store for tracking (add this)
    cluster._last_estimated_center = critical_point.copy()

    # ... rest of code ...
    return vx_c, vy_c
```

**2. Track in cluster** (`src/robot/omni_cluster.py`):

```python
# In __init__:
self.estimated_center_history = []
self._last_estimated_center = None

# In move() method, after cluster.move(control_primitive):
if hasattr(self, '_last_estimated_center') and self._last_estimated_center is not None:
    self.estimated_center_history.append(self._last_estimated_center.copy())
else:
    # Primitive didn't estimate a center this step
    self.estimated_center_history.append(np.array([np.nan, np.nan]))

# Getter:
def get_estimated_center_history(self):
    return self.estimated_center_history

# In reset():
self.estimated_center_history = []
self._last_estimated_center = None
```

**3. Create plotter**:

```python
def plot_critical_point_estimation(center_history, centroid_history,
                                   true_center=np.array([0, 0]),
                                   timestep=0.1, title="Critical Point Estimation",
                                   save_path=None):
    """
    Plot estimated critical point location over time.

    Shows:
    - Estimated center trajectory
    - Actual center (ground truth)
    - Error over time
    """
    import numpy as np
    import matplotlib.pyplot as plt

    centers = np.array(center_history)
    centroids = np.array(centroid_history)
    time = np.arange(len(centers)) * timestep

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Trajectory of estimated center
    ax1 = axes[0, 0]
    ax1.plot(centers[:, 0], centers[:, 1], 'b-', alpha=0.7, label='Estimated Center')
    ax1.scatter(true_center[0], true_center[1], color='red', s=200, marker='*',
               label='True Center', zorder=5)
    ax1.scatter(centers[0, 0], centers[0, 1], color='green', s=100, marker='o',
               label='Initial Estimate', zorder=5)
    ax1.scatter(centers[-1, 0], centers[-1, 1], color='orange', s=100, marker='x',
               label='Final Estimate', zorder=5)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Estimated Center Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Error over time
    ax2 = axes[0, 1]
    errors = np.linalg.norm(centers - true_center, axis=1)
    ax2.plot(time, errors, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Estimation Error (m)')
    ax2.set_title('Distance from True Center')
    ax2.grid(True, alpha=0.3)

    # Plot 3: X and Y components over time
    ax3 = axes[1, 0]
    ax3.plot(time, centers[:, 0], 'b-', label='Estimated X', linewidth=2)
    ax3.plot(time, centers[:, 1], 'g-', label='Estimated Y', linewidth=2)
    ax3.axhline(y=true_center[0], color='b', linestyle='--', alpha=0.5, label='True X')
    ax3.axhline(y=true_center[1], color='g', linestyle='--', alpha=0.5, label='True Y')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Center Components vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distance cluster->center over time
    ax4 = axes[1, 1]
    distances = np.linalg.norm(centroids - centers, axis=1)
    ax4.plot(time, distances, 'purple', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Cluster Distance to Estimated Center')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Critical point plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
```

## Common Plot Templates

### 1. Single Value Over Time

```python
# Use for: distance, error, magnitude, etc.
time = np.arange(len(data)) * timestep
plt.plot(time, data, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Your Metric')
plt.grid(True, alpha=0.3)
```

### 2. Multiple Values on Same Plot

```python
# Use for: comparing different metrics, multiple robots
for i, dataset in enumerate(datasets):
    plt.plot(time, dataset, label=labels[i], linewidth=2)
plt.legend()
```

### 3. 2D Trajectory

```python
# Use for: position tracking, center estimation
plt.plot(x_data, y_data, 'b-', linewidth=2)
plt.scatter(x_data[0], y_data[0], color='green', s=100, marker='o', label='Start')
plt.scatter(x_data[-1], y_data[-1], color='red', s=100, marker='x', label='End')
plt.axis('equal')
plt.legend()
```

### 4. Subplots for Different Aspects

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left
axes[0, 0].plot(...)
axes[0, 0].set_title('Aspect 1')

# Top right
axes[0, 1].plot(...)
axes[0, 1].set_title('Aspect 2')

# ... etc
plt.tight_layout()
```

## Quick Reference: Data Access

### Available from Cluster

```python
# Current state
cluster.get_robot_positions()          # (x1, y1, x2, y2, x3, y3) or (..., x4, y4)
cluster.get_centroid()                 # np.array([x_c, y_c])
cluster.get_current_formation()        # dict with p, q, β (or d1, d2, r1, r2, φ)
cluster.sample_field_at_robots()       # [(u1, v1), (u2, v2), ...]

# History
cluster.get_center_history()           # Array of centroid positions
cluster.get_velocity_history()         # List of velocity arrays
cluster.timestep                       # Time step (0.1 seconds)

# Individual robots
for robot in cluster.robots:
    robot.get_position()               # (x, y)
    robot.get_velocity()               # (vx, vy)
```

### Calculate Common Metrics

```python
# Distance to origin (sink/source center)
centroid = cluster.get_centroid()
distance = np.linalg.norm(centroid)

# Velocity magnitude
vx, vy = robot.get_velocity()
speed = np.sqrt(vx**2 + vy**2)

# Jacobian from primitives (3-robot)
from src.control.primitives import calculate_jacobian_from_readings
curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian_from_readings(cluster)

# Direction to center
direction = -centroid / np.linalg.norm(centroid)  # Normalize toward origin
```

## Output Directory Structure

Create subdirectories for different plot types:

```
VF_Robot/
├── velocity_plots/           # Velocity magnitude plots (existing)
├── center_estimation_plots/  # Critical point tracking (example)
├── formation_plots/          # Formation parameters over time
├── jacobian_plots/           # Curl, divergence tracking
└── trajectory_plots/         # Individual robot paths
```

## Tips for Good Plots

1. **Always include time on X-axis** for time-series data
2. **Use consistent units**: meters for distance, m/s for velocity, seconds for time
3. **Add grid**: `plt.grid(True, alpha=0.3)` for readability
4. **Label everything**: xlabel, ylabel, title, legend
5. **Use appropriate line styles**: solid for actual, dashed for target/reference
6. **Color coding**: Use consistent colors (e.g., blue=robot1, red=target, green=start)
7. **Save at 150 DPI**: Good balance of quality and file size
8. **Close plots after saving**: `plt.close(fig)` to avoid memory issues

## Testing New Plots

Create a standalone test script:

```python
#!/usr/bin/env python3
"""
Test script for new plot type
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
import src.control.primitives as ocp

# Create cluster
field = AnalyticalField(sink1)
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

# Run simulation with data collection
for step in range(100):
    cluster.move(ocp.critical_point_plane_fitting)

# Get data
your_data = cluster.get_your_data_history()

# Generate plot
from src.simulation.your_plotter import plot_your_data
plot_your_data(your_data, save_path="test_output.png")

print("Test plot generated: test_output.png")
```

## Future Plot Ideas

Based on user's example:

1. **Critical point estimation over time** ✓ (template above)
2. **Formation error over time** (p_error, q_error, β_error)
3. **Jacobian eigenvalues** (stability analysis)
4. **Curl vs Divergence** (scatter plot or time series)
5. **Control command vs actual velocity** (commanded vs achieved)
6. **Orbit quality metrics** (radius variation, circularity)
7. **Energy metrics** (kinetic energy, potential energy if applicable)
8. **Multi-robot coordination** (inter-robot distances)
9. **Field gradient magnitude** (how strong is the field at cluster location)
10. **Estimation confidence** (if using multiple methods, show agreement)

## Remember

- **Units**: Always meters, m/s, seconds
- **Timestep**: Default is 0.1 seconds
- **Hardware limits**: Max velocity 0.3 m/s, stiction 0.025 m/s
- **Pattern**: Collect data in cluster, plot in plotter, integrate in main_omni.py
- **Test first**: Create standalone test before integrating
- **Document**: Add docstrings explaining what's being plotted and why

---

**When adding new plots, follow this checklist:**
- [ ] Identify what data to track
- [ ] Add data collection to cluster class
- [ ] Create plotting function
- [ ] Test with standalone script
- [ ] Integrate into main_omni.py
- [ ] Update this guide with your new plot type

## IMPORTANT NOTE: Estimated vs True Distance

When working with **noisy fields** (NN, RBF, or real robot data), you often want to track:

**Estimated distance to critical point** (what the robot *thinks*)
- Not the true distance to (0, 0)
- Use the distance calculation from the control primitive itself

### Example: Tracking Estimated Distance

```python
# In control primitive (e.g., critical_point_plane_fitting):
def critical_point_plane_fitting(cluster):
    # ... calculate critical point ...
    critical_point = np.linalg.solve(J, rhs)
    
    # Get current centroid
    centroid = cluster.get_centroid()
    
    # Calculate estimated distance (what robot thinks)
    estimated_distance = np.linalg.norm(critical_point - centroid)
    
    # Store for tracking
    cluster._last_estimated_center = critical_point.copy()
    cluster._last_estimated_distance = estimated_distance
    
    # ... rest of primitive ...
```

Then in cluster's `move()` method:

```python
# Track estimated distance (what the primitive calculated)
if hasattr(self, '_last_estimated_distance'):
    self.estimated_distance_history.append(self._last_estimated_distance)
else:
    self.estimated_distance_history.append(np.nan)
```

This is **more useful than true distance** because:
- Shows what the robot is actually responding to
- Reveals estimation errors in noisy fields
- Shows how estimation changes as robot moves
- Works even when true center is unknown

### Plot Both for Comparison

```python
def plot_distance_comparison(estimated_distances, true_distances, 
                             timestep=0.1, title="Distance Comparison"):
    """Compare estimated vs true distance to critical point."""
    
    time = np.arange(len(estimated_distances)) * timestep
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, estimated_distances, 'b-', linewidth=2, 
             label='Estimated (what robot thinks)')
    plt.plot(time, true_distances, 'r--', linewidth=2, 
             label='True (ground truth)', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance to Critical Point (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
```

**Key insight**: For NN/RBF fields, the estimated distance should converge even if it's not perfectly accurate!

