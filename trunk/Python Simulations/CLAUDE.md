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
├── main.py                    # Entry point - unified simulation runner
├── main_lite.py              # Simplified version (deprecated)
├── clusters/                 # Robot cluster implementations
│   └── robot_cluster.py      # Core RobotCluster class
├── primitives/               # Control algorithms (THE BRAIN)
│   └── control_primitives.py # All control strategies
├── env/                      # Vector field environments
│   ├── VF_bases.py          # Basic field types (source, sink, vortex, etc.)
│   ├── VF_env.py            # Field composition
│   ├── Sinking_Vortex.py    # Combined sink+vortex fields
│   ├── Spewing_Vortex.py    # Combined source+vortex fields
│   ├── Saddle.py            # Saddle point fields
│   ├── Sink.py, Source.py, Vortex.py  # Individual field types
│   └── grid_setup.py        # Visualization grid
├── exec_sim/                # Simulation execution
│   └── simulation.py        # Main simulation loop
├── sinking_vortex_predictors/  # ML models for sinking vortex fields
│   ├── nn_trainer_sinking_vortex.py
│   ├── rbf_trainer_sinking_vortex.py
│   └── synthetic_generator.py
├── saddle_predictors/       # ML models for saddle point fields
│   ├── nn_trainer_saddle.py
│   └── rbf_trainer_saddle.py
└── vortex_predictors/       # ML models for pure vortex fields
    ├── nn_trainer_vortex.py
    └── rbf_trainer_vortex.py
```

## Running Simulations

### Basic Execution
```bash
cd "Vector_Fields/VF_3_robot"
python main.py
```

### Configuration Options (in main.py)

The simulation has three modes controlled by `SIMULATION_MODE`:
- `"single"` - Run one simulation with one environment
- `"compare"` - Compare 4 modes (Blended, NN, RBF, Analytical) side-by-side
- `"multi_env"` - Run all 18 environments in a 3x6 grid

**Field Approximation Modes:**
```python
USE_BLENDED = False       # Use blended RBF/NN approach
USE_NN_ONLY = False      # Use only neural network
USE_RBF_ONLY = False     # Use only RBF interpolator
USE_ANALYTICAL = True    # Use ground truth analytical functions
RBF_WEIGHT = 0.9         # Blending ratio (0.9 = 90% RBF, 10% NN)
```

**Environment Selection:**
```python
ENVIRONMENT = "vortex1"  # Options: sink1-3, source1-3, vortex1-3,
                         #          sinking_vortex1-3, spewing_vortex1-3,
                         #          saddle1-3
```

**Control Primitive Selection:**
```python
CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"
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
- `direction_follow_with_center_attraction` - Follow field direction with attraction to center
- `center_attraction` - Move toward estimated center
- `find_sink_center` - Specialized for finding sink centers

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

The default configuration loads models from `sinking_vortex_predictors/`. To train models for different field types, use the corresponding predictor directory.

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

**Model Files Required (in predictor directory):**
- `hue_model.pth` - NN weights for direction
- `sat_model.pth` - NN weights for magnitude
- `model_info.pkl` - Architecture specifications
- `vortex_rbf_interpolator.pkl` - RBF interpolator

**Note:** By default, `RobotCluster` loads models from `sinking_vortex_predictors/`. To use different field-specific models, modify the `load_nn_models()` and `load_rbf_models()` methods in `robot_cluster.py` to point to `saddle_predictors/` or `vortex_predictors/`.

## Vector Field Primitives

Available in `env/VF_bases.py`:
- `source(x, y, xc, yc)` - Radial outward flow (V ∝ 1/r²)
- `sink(x, y, xc, yc)` - Radial inward flow (V ∝ 1/r²)
- `free_vortex(x, y, xc, yc)` - Circular flow with V ∝ 1/r
- `fixed_vortex(x, y, xc, yc)` - Circular flow with V ∝ r
- `saddle(x, y, xc, yc)` - Hyperbolic saddle point
- `doublet(x, y, xc, yc)` - Source-sink pair
- `uni_xflow(x, y)` - Uniform horizontal flow
- `uni_yflow(x, y)` - Uniform vertical flow
- `boundary_layer(x, y)` - Exponentially decaying flow

### Pre-Built Environment Configurations

The `env/` directory includes multiple pre-configured environments with 3 variants each:
- **Sink** (`Sink.py`): sink1, sink2, sink3 - Pure radial inward flow fields
- **Source** (`Source.py`): source1, source2, source3 - Pure radial outward flow fields
- **Vortex** (`Vortex.py`): vortex1, vortex2, vortex3 - Pure rotational flow fields
- **Sinking Vortex** (`Sinking_Vortex.py`): sinking_vortex1-3 - Combined rotation + inward flow
- **Spewing Vortex** (`Spewing_Vortex.py`): spewing_vortex1-3 - Combined rotation + outward flow
- **Saddle** (`Saddle.py`): saddle1, saddle2, saddle3 - Hyperbolic saddle points

Compose custom fields in `env/VF_env.py` by summing primitives from `VF_bases.py`.

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
4. Update `main.py`:
   - Add your primitive name to `control_primitive_map` dictionary
   - Set `CONTROL_PRIMITIVE = "my_algorithm"`
   - Run with `python main.py`

### Create New Vector Field Environment
1. Add environment function to `env/` directory (see `Sinking_Vortex.py` as template)
2. Function signature: `def my_field(x, y) -> (u, v)` where x,y can be scalars or arrays
3. Use `VF_bases.py` primitives to compose your field
4. Update `main.py`:
   - Import your environment: `from env.MyField import my_field1`
   - Add to `environment_map` dictionary
   - Set `ENVIRONMENT = "my_field1"` and `USE_ANALYTICAL = True`

### Run Quick Experiment
For single simulation with specific settings:
```python
# In main.py
SIMULATION_MODE = "single"
ENVIRONMENT = "sinking_vortex1"
CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"
USE_ANALYTICAL = True  # or USE_BLENDED/USE_NN_ONLY/USE_RBF_ONLY
```

### Compare ML Models vs Analytical
```python
# In main.py
SIMULATION_MODE = "compare"  # Shows Blended, NN, RBF, Analytical side-by-side
ENVIRONMENT = "vortex1"
RUN_BLENDING_TEST = True  # Verify blending math
```

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
1. Use existing predictor directory (`sinking_vortex_predictors/`, `saddle_predictors/`, or `vortex_predictors/`)
2. Modify `synthetic_generator.py` to generate your specific field configuration
3. Run training scripts: `python nn_trainer_*.py` and `python rbf_trainer_*.py`
4. Trained models will be saved in the same directory
5. Note: `RobotCluster` defaults to loading from `sinking_vortex_predictors/`

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
