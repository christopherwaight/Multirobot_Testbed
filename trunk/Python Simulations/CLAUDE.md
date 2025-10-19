# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-robot simulation testbed for exploring vector field navigation and critical point detection. The robots use formations (clusters) to sample vector fields and estimate field properties like critical points (sinks, sources, vortices, saddles) using only local measurements.

The core innovation is using **momentum-based physics** for robot clusters combined with machine learning (neural networks and RBF interpolators) to approximate complex vector fields from discrete samples.

## Repository Structure

```
Python Simulations/
├── Vector_Fields/          # Vector field navigation simulations
│   ├── VF_3_robot/        # Main 3-robot simulation (ACTIVELY USED)
│   ├── Vector_Field_4_Robot/              # 4-robot variant
│   └── Vector_Field_4_Robot_experimental/ # Experimental 4-robot work
├── Scalar Fields/         # Scalar field optimization (gradient ascent/descent)
├── Separatrix_Control_testing/  # Separatrix navigation experiments
└── Archive/               # Old implementations and experiments
```

### VF_3_robot Structure (Main Working Directory)

The primary simulation code lives in `Vector_Fields/VF_3_robot/`:

```
VF_3_robot/
├── main.py                    # Entry point - run simulations here
├── main_lite.py              # Simplified version
├── clusters/                 # Robot cluster implementations
│   └── robot_cluster.py      # Core RobotCluster class
├── primitives/               # Control algorithms (THE BRAIN)
│   └── control_primitives.py # All control strategies
├── env/                      # Vector field environments
│   ├── VF_bases.py          # Basic field types (source, sink, vortex, etc.)
│   ├── VF_env.py            # Field composition
│   ├── Sinking_Vortex.py    # Combined sink+vortex fields
│   ├── Saddle.py            # Saddle point fields
│   └── grid_setup.py        # Visualization grid
├── exec_sim/                # Simulation execution
│   └── simulation.py        # Main simulation loop
└── sinking_vortex_predictors/  # ML models (NN & RBF)
    ├── nn_trainer_sinking_vortex.py
    ├── rbf_trainer_sinking_vortex.py
    └── synthetic_generator.py
```

## Running Simulations

### Basic Execution
```bash
cd "Vector_Fields/VF_3_robot"
python main.py
```

### Configuration Options (in main.py)
```python
USE_BLENDED = True    # Blend RBF + NN (90/10 default)
USE_NN_ONLY = False   # Use only neural network
USE_RBF_ONLY = False  # Use only RBF interpolator
USE_ANALYTICAL = False # Use ground truth analytical functions

RBF_WEIGHT = 0.9  # Blending ratio (0.9 = 90% RBF, 10% NN)
```

### Selecting Environment
In `main.py`, change the import:
```python
from env.Sinking_Vortex import sinking_vortex1 as enviro  # Vortex+sink
from env.Saddle import true_saddle as enviro              # Saddle point
from env.Sink import sink3 as enviro                      # Pure sink
```

### Selecting Control Primitive
In `main.py`, change the control primitive passed to `execute_simulation()`:
```python
execute_simulation(cluster, cp.critical_point_orbiter_plane_fitting, 'Title')
```

Available control primitives (from `control_primitives.py`):
- `critical_point_plane_fitting` - Find critical points using plane fitting
- `critical_point_cross_product` - Find critical points using cross products
- `critical_point_orbiter_plane_fitting` - Orbit around critical points
- `critical_point_orbiter_cross_product` - Alternative orbiter
- `vector_sum` - Simple vector field following
- `find_center` - Weighted combination of rotation/radial components
- `find_center2` - Parameter-free center finding using energy descent
- `eigenstep` - Move along most stable eigenvector direction

## Key Architecture Concepts

### 1. Robot Cluster Physics
The `RobotCluster` class (`clusters/robot_cluster.py`) implements **second-order dynamics** with momentum:

```python
desired_velocity = (desired_centre - cluster_centre) / step_size
velocity = momentum_alpha * velocity + (1 - momentum_alpha) * desired_velocity
velocity *= damping
cluster_centre += velocity * step_size
```

Parameters:
- `momentum_alpha = 0.7` - Higher = more inertia
- `damping = 1.0` - Energy dissipation
- `step_size = 0.1` - Base movement increment

### 2. Critical Point Estimation Theory
The mathematics behind center estimation is documented in `Explanation II.txt` (LaTeX source). Key principles:

**Direction to Center:**
- **Gradient method**: Direction `= -g/||g||` where `g = J^T v` (gradient of ½||v||²)
- **Curvature method**: Direction `= -s_perp/||s_perp||` where `s_perp` is normal acceleration

**Distance to Center:**
- **Radial cue**: `r = V²/||g||` (exact for V∝r and V∝1/r)
- **Curvature cue**: `r = V²/||s_perp||` (for circular/vortical motion)
- **Constant-speed vortex**: `r = V/|ω|`
- **Constant-speed sink**: `r = V/|div v|`

### 3. Jacobian Calculation from Three Points
The `calculate_jacobian()` function uses plane fitting with 3 robot positions to estimate:
```
J = [du/dx  du/dy]
    [dv/dx  dv/dy]
```

This enables computing:
- `curl = dv/dx - du/dy` (rotation)
- `divergence = du/dx + dv/dy` (expansion/contraction)

### 4. Machine Learning Architecture

**Vector Field Representation:**
Vector fields are encoded as `(hue, saturation)` where:
- `hue` → direction (angle), represented as `(sin(θ), cos(θ))`
- `saturation` → magnitude (scaled to [0,1])

**Neural Network Models:**
- Located in `sinking_vortex_predictors/`
- Two separate networks: `hue_model` (2 outputs: sin/cos) and `sat_model` (1 output)
- Default architecture: `[16,8,4]` hidden layers for hue, `[15,10,5]` for saturation
- Activation: `tanh`

**RBF Interpolator:**
- Radial basis function interpolator for smooth field approximation
- Trained on same hue/saturation representation
- Often more accurate than NN for smooth fields

**Blended Mode:**
Combines RBF and NN predictions:
```python
blended_sin = rbf_weight * rbf_sin + nn_weight * nn_sin
blended_cos = rbf_weight * rbf_cos + nn_weight * nn_cos
blended_sat = rbf_weight * rbf_sat + nn_weight * nn_sat
```

### 5. Training ML Models

**Generate Training Data:**
```bash
cd Vector_Fields/VF_3_robot/sinking_vortex_predictors
python synthetic_generator.py  # Creates CSV with (x,y,hue,sat) samples
```

**Train Neural Network:**
```bash
python nn_trainer_sinking_vortex.py
# Outputs: hue_model.pth, sat_model.pth, model_info.pkl
```

**Train RBF Interpolator:**
```bash
python rbf_trainer_sinking_vortex.py
# Outputs: vortex_rbf_interpolator.pkl
```

**Model Files Required:**
- `hue_model.pth` - NN weights for direction
- `sat_model.pth` - NN weights for magnitude
- `model_info.pkl` - Architecture specifications
- `vortex_rbf_interpolator.pkl` - RBF interpolator

## Vector Field Primitives

Available in `env/VF_bases.py`:
- `source(x, y, xc, yc)` - Radial outward flow (V ∝ 1/r²)
- `sink(x, y, xc, yc)` - Radial inward flow
- `free_vortex(x, y, xc, yc)` - Circular flow with V ∝ 1/r
- `fixed_vortex(x, y, xc, yc)` - Circular flow with V ∝ r
- `saddle(x, y, xc, yc)` - Hyperbolic saddle point
- `doublet(x, y, xc, yc)` - Source-sink pair
- `uni_xflow(x, y)` - Uniform horizontal flow
- `uni_yflow(x, y)` - Uniform vertical flow
- `boundary_layer(x, y)` - Exponentially decaying flow

Compose custom fields in `env/VF_env.py` by summing primitives.

## Dependencies

Core packages (inferred from imports):
```
numpy
matplotlib
torch (PyTorch)
scipy (for RBFInterpolator)
pandas (for ML training)
scikit-learn (for train/test split)
```

Install with:
```bash
pip install numpy matplotlib torch scipy pandas scikit-learn
```

## Common Workflows

### Experiment with New Control Algorithm
1. Add new function to `primitives/control_primitives.py`
2. Function signature: `def my_algorithm(cluster) -> np.ndarray`
3. Return the new cluster center position
4. Update `main.py` to call your primitive: `execute_simulation(cluster, cp.my_algorithm, 'My Title')`

### Create New Vector Field Environment
1. Add environment function to `env/` directory (see `Sinking_Vortex.py` as template)
2. Function signature: `def my_field(x, y) -> (u, v)` where x,y can be scalars or arrays
3. Use `VF_bases.py` primitives to compose your field
4. Import in `main.py` and set `USE_ANALYTICAL = True`

### Modify Robot Formation
Edit `reset()` in `clusters/robot_cluster.py`:
```python
self.robot_offsets = np.array([
    [0, off_size],              # robot 1 position relative to center
    [-off_size*√(1/3), 0],      # robot 2
    [off_size*√(1/3), 0]        # robot 3 (forms equilateral triangle)
])
```

### Train Models for New Field Type
1. Create new predictor directory (e.g., `saddle_predictors/`)
2. Copy `synthetic_generator.py` and modify to generate your field
3. Copy and adapt `nn_trainer_*.py` and `rbf_trainer_*.py`
4. Update `RobotCluster.__init__()` to load from new directory

## Key Implementation Details

### Noise in Readings
Robot readings include small Gaussian noise: `± 0.00` (configurable in `bot_readings()`)

### Coordinate System
- Origin typically at (0, 0)
- Default simulation domain: approximately [-0.6, 0.6] × [-0.6, 0.6]
- Grid visualization: defined in `env/grid_setup.py`

### Simulation Parameters
In `exec_sim/simulation.py`:
- `sim_time = 150` - Number of simulation steps
- Trajectory plotted with `centre_points` array

### Visualization
- Blue/Yellow/Green markers = individual robots
- Black line with circles = cluster center trajectory
- Background quiver plot = vector field

## Troubleshooting

**"Model files not found"**: Train models first using the trainer scripts in the appropriate predictor directory.

**Singular matrix errors**: Occurs when robot formation is degenerate (collinear points). The code falls back to `vector_sum()` in these cases.

**Oscillating behavior**: Reduce `step_size` or increase `momentum_alpha` in `robot_cluster.py` for more damping.

**Poor ML predictions**: Retrain with more diverse training data or adjust network architecture in the trainer scripts.

## Research Context

This codebase implements distributed sensing and control for multi-robot systems exploring unknown vector fields. The robots use:
- **Formation-based sampling** to estimate local field derivatives (Jacobian)
- **Theoretical estimators** derived from fluid dynamics (see `Explanation II.txt`)
- **Machine learning** to approximate fields when analytical forms are unknown
- **Physics-based control** with momentum for smooth, realistic motion

The primary goal is to locate and characterize critical points (equilibria) of 2D vector fields using only local measurements from a moving robot cluster.
