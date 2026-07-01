# Cluster Orbit Demonstrator

## Overview

Visual demonstration tool that showcases four different control primitives orbiting around a critical point in a vortex field. Each primitive is displayed in a separate subplot with vector field background, robot trajectories, and formation dynamics.

## Purpose

This demonstrator provides a clear visual comparison of how different control strategies handle orbital motion around a critical point, making it easy to:
- Compare 3-robot vs 4-robot formations
- Observe different estimation methods (plane fitting vs planar vs dual Jacobian)
- Evaluate trajectory smoothness and orbital accuracy
- Understand formation maintenance during orbital motion

## Files

- **`cluster_orbit_demonstrator.py`** - Main demonstration script
- **`cluster_orbit_plots/`** - Output directory for generated PNG files

## Control Primitives Demonstrated

### 1. 3R Vector Sum
- **Type**: 3-robot formation
- **Formation**: Equilateral triangle (p=q=0.33, β=120°)
- **Strategy**: Simple vector field following (baseline)
- **Behavior**: Follows the flow without explicit orbit control

### 2. 3R Orbiter (Plane Fitting)
- **Type**: 3-robot formation
- **Formation**: Equilateral triangle (p=q=0.33, β=120°)
- **Strategy**: Estimates critical point using plane fitting
- **Behavior**: Actively orbits around estimated center

### 3. 4R Orbiter (Planar)
- **Type**: 4-robot formation
- **Formation**: Square configuration (d1=d2=0.3, φ=90°)
- **Strategy**: Uses 4-point planar estimation
- **Behavior**: Orbits with improved center estimation from 4 sample points

### 4. 4R Orbiter (Advanced)
- **Type**: 4-robot formation
- **Formation**: Rectangle configuration (d1=0.25, d2=0.433, φ=90°)
- **Strategy**: Dual Jacobian estimation + orientation control
- **Behavior**: Orbits with both position and orientation control

## Configuration

- **Environment**: Vortex 1 (pure rotational flow field)
- **Orbital radius**: 0.5 meters
- **Simulation time**: 120 steps (12 seconds)
- **Initial position**: All clusters start at (0, 0.5)
- **Field mode**: Analytical (ground truth)

## Running the Demonstrator

```bash
cd VF_Robot
./venv/bin/python3 experiments/cluster_orbit_demonstrator.py
```

### Customization

To modify parameters, edit the configuration section in `cluster_orbit_demonstrator.py`:

```python
# Simulation parameters
SIM_TIME = 120           # Number of simulation steps
ORBITAL_RADIUS = 0.5     # Desired orbital radius
INITIAL_X = 0.0          # Starting x position

# Environment
ENVIRONMENT_FUNC = vortex1
ENVIRONMENT_NAME = "Vortex 1"
```

You can also change the environment to test in different vector fields:
```python
from src.fields.environments.Sinking_Vortex import sinking_vortex1
ENVIRONMENT_FUNC = sinking_vortex1
ENVIRONMENT_NAME = "Sinking Vortex 1"
```

## Output Files

The script generates 5 PNG files:

### 1. Combined View
**Filename**: `orbit_demonstration_YYYYMMDD_HHMMSS.png`
- 2×2 grid showing all four primitives
- Size: 16×16 inches (high resolution for presentations)
- Perfect for comparative analysis

### 2-5. Individual Plots
Four separate plots, one for each primitive:
- `vector_sum_YYYYMMDD_HHMMSS.png`
- `orbiter_plane_YYYYMMDD_HHMMSS.png`
- `orbiter_quad_planar_YYYYMMDD_HHMMSS.png`
- `orbiter_quad_advanced_YYYYMMDD_HHMMSS.png`

Each individual plot:
- Size: 10×10 inches
- High resolution (150 DPI)
- Suitable for publications or detailed examination

## Plot Elements

Each subplot contains:

### Vector Field Background
- Black arrows showing the underlying vector field
- Helps visualize the flow that robots are navigating

### Robot Trajectories
- **Colored dashed lines**: Individual robot paths (blue, orange, green, red)
- **Circles**: Starting positions of each robot
- **Squares**: Ending positions of each robot

### Centroid Trajectory
- **Black solid line**: Path of formation center
- **Green star**: Centroid starting position
- **Red X**: Centroid ending position

### Critical Point
- **Magenta pentagon**: Location of the critical point (origin)
- The target around which robots should orbit

### Legend
- Located in upper right corner
- Identifies all trajectory types and markers

## Interpreting the Results

### What to Look For:

1. **Orbital Stability**
   - Does the centroid trajectory form a circular/elliptical orbit?
   - Is the orbit centered on the critical point (magenta pentagon)?

2. **Formation Maintenance**
   - Do individual robot trajectories (colored lines) stay parallel?
   - Is spacing between robots consistent throughout the orbit?

3. **Trajectory Smoothness**
   - Are paths smooth curves or jagged?
   - Do robots exhibit oscillations or steady motion?

4. **Convergence Behavior**
   - Does the orbit radius stabilize at the desired 0.5m?
   - How quickly does each primitive reach steady-state?

### Expected Behaviors:

- **Vector Sum**: May spiral or drift, as it doesn't actively maintain orbit
- **3R Orbiter**: Should establish circular orbit around estimated center
- **4R Orbiters**: Generally smoother orbits due to more sample points
- **4R Advanced**: May show orientation alignment with radial direction

## Use Cases

### Research & Development
- Compare new control algorithms against baselines
- Visualize effect of formation geometry on performance
- Debug trajectory issues or convergence problems

### Education & Presentations
- Demonstrate multi-robot coordination concepts
- Show difference between formation types
- Illustrate vector field navigation principles

### Documentation
- Include in papers or reports
- Create figure panels for publications
- Generate reference trajectories

## Performance Notes

- **Runtime**: Approximately 5-10 seconds for all 4 simulations
- **File sizes**: ~200-250 KB per individual plot, ~900 KB for combined view
- **Memory usage**: Minimal (suitable for batch processing)

## Troubleshooting

### Issue: Plots look cluttered
**Solution**: Reduce simulation time or increase figure size:
```python
SIM_TIME = 60  # Shorter trajectories
FIGURE_SIZE = (20, 20)  # Larger canvas
```

### Issue: Trajectories extend off-plot
**Solution**: The grid is automatically set by `make_environment_grid()`. For larger orbits, you may need to modify the grid range in `src/fields/environments/grid_setup.py`.

### Issue: Vector field arrows too dense
**Solution**: Modify the quiver plot parameters:
```python
ax.quiver(X, Y, u_grid, v_grid, color='black', alpha=0.2, scale=20)
#                                                   ^^^^         ^^
#                                            reduce opacity  increase scale
```

## Example Workflow

1. **Run demonstration**:
   ```bash
   ./venv/bin/python3 experiments/cluster_orbit_demonstrator.py
   ```

2. **Check output**:
   ```bash
   open experiments/cluster_orbit_plots/orbit_demonstration_*.png
   ```

3. **Compare individual primitives**:
   ```bash
   open experiments/cluster_orbit_plots/orbiter_*.png
   ```

4. **For presentations**: Use the combined `orbit_demonstration_*.png`

5. **For detailed analysis**: Use individual plots

## Future Enhancements

Potential additions to this demonstrator:
- Animation output (GIF or MP4)
- Real-time plotting during simulation
- Quantitative metrics overlay (orbit error, formation error)
- Multiple orbital radii in single plot
- Side-by-side environment comparisons
