# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3-robot vector field navigation simulation testbed. The robots form a triangular cluster that samples vector fields to estimate critical points (sinks, sources, vortices, saddles) using only local measurements.

The architecture implements **distributed robot control** with **formation control**:

```
┌─────────────────────────────────────┐
│ OmniCluster                          │
│ - Reads formation config (p,q,β)    │
│ - Computes forward/inverse kinematics│
│ - Formation error → shape velocities │
│ - Inverse Jacobian → robot velocities│
└─────────┬───────────────────────────┘
          │ commands
          ├──────────┬──────────┐
          ▼          ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Omnibot 1│ │Omnibot 2│ │Omnibot 3│
    │(x,y,v)  │ │(x,y,v)  │ │(x,y,v)  │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
         └───────────┴───────────┘
                     │
              samples environment
                     ▼
          ┌──────────────────────┐
          │ VectorField (plugin) │
          │ - Analytical/NN/RBF  │
          │ - Returns (u,v)      │
          └──────────────────────┘
```

## Running Simulations

```bash
python3 main_omni.py
```

## Key Files

- **`main_omni.py`** - Main entry point for simulations
- **`omnibot.py`** - Individual robot with momentum-based dynamics
- **`omni_cluster.py`** - Cluster coordinator with formation control
- **`fields.py`** - Field abstraction (AnalyticalField, NNField, RBFField, BlendedField)
- **`kinematics.py`** - Forward/inverse kinematics and Jacobian calculations
- **`omni_primitives.py`** - Control primitives (return velocities)
- **`omni_simulation.py`** - Simulation execution
- **`formations/*.yaml`** - Formation configuration files
- **`env/`** - Vector field environments

## Configuration in main_omni.py

**Simulation Mode** (`SIMULATION_MODE`):
- `"single"` - Run one simulation with one environment
- `"compare"` - Compare 4 modes (Blended, NN, RBF, Analytical) side-by-side
- `"multi_env"` - Run all 18 environments in 3×6 grid

**Field Mode** (`FIELD_MODE` for single mode):
- `"analytical"` - Use ground truth analytical functions
- `"nn"` - Use neural network approximation
- `"rbf"` - Use RBF interpolator approximation
- `"blended"` - Use weighted blend of RBF and NN

**Environment Selection** (`ENVIRONMENT`):
- Sink: `"sink1"`, `"sink2"`, `"sink3"`
- Source: `"source1"`, `"source2"`, `"source3"`
- Vortex: `"vortex1"`, `"vortex2"`, `"vortex3"`
- Sinking Vortex: `"sinking_vortex1"`, `"sinking_vortex2"`, `"sinking_vortex3"`
- Spewing Vortex: `"spewing_vortex1"`, `"spewing_vortex2"`, `"spewing_vortex3"`
- Saddle: `"saddle1"`, `"saddle2"`, `"saddle3"`

**Formation Configuration** (`FORMATION_CONFIG`):
```python
FORMATION_CONFIG = "formations/equilateral_default.yaml"
# Available: equilateral_default.yaml, equilateral_small.yaml
```

**Control Primitives** (`CONTROL_PRIMITIVE`):
- `"vector_sum"` - Simple vector field following
- `"critical_point_plane_fitting"` - Find critical points
- `"critical_point_orbiter_plane_fitting"` - Orbit around critical points
- `"find_center"` - Weighted combination of rotation/radial
- `"find_sink_center"` - Specialized for sink centers

**Predictor Directory** (`PREDICTOR_DIR`):
- `"sinking_vortex_predictors"` - Default
- `"saddle_predictors"` - For saddle fields
- `"vortex_predictors"` - For vortex fields

## Formation Configuration Files

Formation configs are YAML files in `formations/` directory:

```yaml
formation:
  p: 0.433          # Distance robot 1 to robot 2
  q: 0.433          # Distance robot 2 to robot 3
  beta_degrees: 120.0  # Angle at robot 2
  position_gain: 1.0   # Formation control gain
```

**SAS Parameterization:** (p, q, β) uniquely defines a triangle.

## Omnibot Class

Individual robot with momentum-based dynamics:

```python
from omnibot import Omnibot

robot = Omnibot(x=0.0, y=0.0, timestep=0.1, momentum_alpha=0.7)

# Command velocity
robot.command_velocity(vx_cmd, vy_cmd)

# Momentum update:
# velocity = alpha * velocity_old + (1-alpha) * velocity_cmd
# position += timestep * velocity

# Sample field
u, v = robot.sample_field(field)
```

**Key parameters:**
- `timestep = 0.1` - Time step for integration
- `momentum_alpha = 0.7` - Momentum coefficient (higher = more inertia)

## VectorField Classes

Unified interface for all field types:

```python
from fields import AnalyticalField, NNField, RBFField, BlendedField
from env.Sinking_Vortex import sinking_vortex1

# Analytical field
field = AnalyticalField(sinking_vortex1)

# Neural network field
field = NNField(predictor_dir='sinking_vortex_predictors')

# RBF field
field = RBFField(predictor_dir='saddle_predictors')

# Blended field (90% RBF, 10% NN)
field = BlendedField(predictor_dir='vortex_predictors', rbf_weight=0.9)

# All fields have unified interface:
u, v = field.get_value(x, y)
```

## OmniCluster Class

Manages 3 robots with formation control:

```python
from omni_cluster import OmniCluster
from fields import AnalyticalField
from env.Sinking_Vortex import sinking_vortex1

field = AnalyticalField(sinking_vortex1)
cluster = OmniCluster('formations/equilateral_default.yaml', field)

# Get current formation
formation = cluster.get_current_formation()
# Returns: {x_c, y_c, theta_c, p, q, r, beta, alpha, gamma}

# Sample field at all robots
readings = cluster.sample_field_at_robots()  # [(u1,v1), (u2,v2), (u3,v3)]

# Move using control primitive
cluster.move(control_primitive_function)
```

**Formation Control Algorithm:**
1. Computes current formation using forward kinematics
2. Computes error: `desired_shape - current_shape`
3. Generates shape velocities: `v_shape = gain * error`
4. Converts to robot velocities using inverse Jacobian
5. Commands individual robots

## Kinematics Functions

```python
from kinematics import (forward_kinematics, inverse_kinematics,
                       compute_inverse_jacobian,
                       shape_velocities_to_robot_velocities)

# Forward kinematics: robot positions → shape parameters
formation = forward_kinematics(x1, y1, x2, y2, x3, y3)
# Returns: {x_c, y_c, theta_c, p, q, r, beta, alpha, gamma}

# Inverse kinematics: shape parameters → robot positions
x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c, p, beta, q)

# Compute inverse Jacobian (numerical differentiation)
J_inv = compute_inverse_jacobian(p, beta, q, theta_c)

# Convert shape velocities to robot velocities
vx1, vy1, vx2, vy2, vx3, vy3 = shape_velocities_to_robot_velocities(
    J_inv, vx_c, vy_c, omega_c, vp, vbeta, vq
)
```

**Inverse Kinematics Implementation:**
- Places robot 2 at origin in local frame
- Places robot 1 at distance `p` along +x axis
- Places robot 3 at distance `q` at angle `beta`
- Centers triangle, then applies rotation and translation

## Control Primitives

All primitives return `(vx_c, vy_c)` - desired centroid velocity:

```python
import omni_primitives as ocp

def my_primitive(cluster):
    # Your control logic
    # Access: cluster.get_robot_positions()
    #         cluster.sample_field_at_robots()
    #         cluster.get_current_formation()
    #         cluster.get_centroid()

    return vx_c, vy_c  # Return desired centroid velocity
```

**Available primitives** in `omni_primitives.py`:
- `vector_sum(cluster)` - Move in direction of summed field vectors
- `critical_point_plane_fitting(cluster)` - Move toward critical point estimated by plane fitting
- `critical_point_orbiter_plane_fitting(cluster, desired_radius)` - Orbit around critical point
- `find_center(cluster)` - Weighted rotation/radial combination
- `find_sink_center(cluster)` - Move with/against flow based on divergence

**Helper function:**
- `calculate_jacobian_from_readings(cluster)` - Estimates Jacobian matrix from 3-robot formation

## Vector Field Environments

**Base primitives** (`env/VF_bases.py`):
- `source(x, y, xc, yc)` - Radial outward flow (V ∝ 1/r²)
- `sink(x, y, xc, yc)` - Radial inward flow (V ∝ 1/r²)
- `free_vortex(x, y, xc, yc)` - Circular flow (V ∝ 1/r)
- `fixed_vortex(x, y, xc, yc)` - Circular flow (V ∝ r)
- `saddle(x, y, xc, yc)` - Hyperbolic saddle point
- `doublet(x, y, xc, yc)` - Source-sink pair
- `uni_xflow(x, y)`, `uni_yflow(x, y)` - Uniform flows
- `boundary_layer(x, y)` - Exponentially decaying flow
- `rankine(x, y, xc, yc)` - Rankine vortex

**Pre-configured environments:**
Each file in `env/` contains 3 variants (e.g., Sink.py has sink1, sink2, sink3):
- **Sink.py** - Pure radial inward flow
- **Source.py** - Pure radial outward flow
- **Vortex.py** - Pure rotational flow
- **Sinking_Vortex.py** - Combined rotation + inward flow
- **Spewing_Vortex.py** - Combined rotation + outward flow
- **Saddle.py** - Hyperbolic saddle points

## Machine Learning Models

### Vector Field Representation

Fields are encoded as `(hue, saturation)`:
- `hue` → direction (angle), represented as `(sin(θ), cos(θ))` to avoid discontinuity
- `saturation` → magnitude (scaled to [0,1])

### Neural Networks

**Architecture** (defined in `*_predictors/nn_trainer_*.py`):
- **Hue model**: Predicts `(sin, cos)` - default layers `[16,8,4]`
- **Saturation model**: Predicts magnitude - default layers `[15,10,5]`
- Activation: `tanh`

**Model files** (in predictor directories):
- `hue_model.pth` - NN weights for direction
- `sat_model.pth` - NN weights for magnitude
- `model_info.pkl` - Architecture specifications

### RBF Interpolators

**Model files:**
- `vortex_rbf_interpolator.pkl` - Contains `hue_rbf` and `sat_rbf` interpolators

### Blended Mode

Combines RBF and NN predictions at the hue/saturation level:
```python
blended_sin = rbf_weight * rbf_sin + nn_weight * nn_sin
blended_cos = rbf_weight * rbf_cos + nn_weight * nn_cos
blended_sat = rbf_weight * rbf_sat + nn_weight * nn_sat
```

Default: `rbf_weight = 0.9` (90% RBF, 10% NN)

## Training ML Models

### 1. Generate Training Data
```bash
cd sinking_vortex_predictors
python3 synthetic_generator.py
```
Creates CSV file with `(x, y, hue, sat)` samples.

### 2. Train Neural Network
```bash
python3 nn_trainer_sinking_vortex.py
```
Outputs: `hue_model.pth`, `sat_model.pth`, `model_info.pkl`

### 3. Train RBF Interpolator
```bash
python3 rbf_trainer_sinking_vortex.py
```
Outputs: `vortex_rbf_interpolator.pkl`

**Note:** Training data coordinates are negated (`-x`, `-y`) to match simulation coordinate system.

### Model Directories
- `sinking_vortex_predictors/` - Models for sinking vortex fields (default)
- `saddle_predictors/` - Models for saddle point fields
- `vortex_predictors/` - Models for pure vortex fields

## Common Workflows

### Quick Single Simulation
```python
# In main_omni.py
SIMULATION_MODE = "single"
FIELD_MODE = "analytical"
ENVIRONMENT = "sinking_vortex1"
CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"
FORMATION_CONFIG = "formations/equilateral_default.yaml"
```

### Compare All Field Approximation Methods
```python
SIMULATION_MODE = "compare"
ENVIRONMENT = "vortex1"
RBF_WEIGHT = 0.9
```
Shows Blended, NN, RBF, and Analytical side-by-side.

### Create New Formation Config

Create YAML file in `formations/`:

```yaml
# formations/my_formation.yaml
formation:
  p: 0.5            # Side length 1-2
  q: 0.3            # Side length 2-3
  beta_degrees: 90.0   # Angle at robot 2
  position_gain: 1.0   # Formation control gain
```

Then use:
```python
cluster = OmniCluster('formations/my_formation.yaml', field)
```

### Create New Control Primitive

1. Add function to `omni_primitives.py`:
```python
def my_algorithm(cluster):
    # Your algorithm
    vx_c, vy_c = 0.0, 0.0  # Calculate desired velocity
    return vx_c, vy_c
```

2. Register in `main_omni.py`:
```python
control_primitive_map = {
    "my_algorithm": ocp.my_algorithm,
    # ...
}
```

3. Select in config:
```python
CONTROL_PRIMITIVE = "my_algorithm"
```

### Create New Vector Field Environment

1. Add environment function to `env/` directory (see `Sinking_Vortex.py` as template):
```python
# env/MyField.py
from env.VF_bases import sink, free_vortex

def my_field1(x, y):
    u1, v1 = sink(x, y, 0.0, 0.0)
    u2, v2 = free_vortex(x, y, 0.0, 0.0)
    return u1 + u2, v1 + v2
```

2. Import in `main_omni.py`:
```python
from env.MyField import my_field1
```

3. Add to `environment_map`:
```python
environment_map = {
    "my_field1": (my_field1, "My Field 1"),
    # ...
}
```

4. Use it:
```python
ENVIRONMENT = "my_field1"
```

### Train Models for Different Field Type

1. Navigate to target predictor directory
2. Modify `synthetic_generator.py` to generate appropriate field data
3. Run trainers:
```bash
python3 nn_trainer_*.py
python3 rbf_trainer_*.py
```
4. Specify `predictor_dir` when creating field:
```python
field = NNField(predictor_dir='my_new_predictors')
```

## Critical Point Estimation Mathematics

The robots use triangular formation to estimate the Jacobian matrix:
```
J = [du/dx  du/dy]
    [dv/dx  dv/dy]
```

**Derived quantities:**
- `curl = dv/dx - du/dy` - Rotation measure
- `divergence = du/dx + dv/dy` - Expansion/contraction measure

**Estimation method:**
Plane fitting using 3-robot positions. Fits planes to u and v components:
- `U(x,y) = ax + by + c`
- `V(x,y) = dx + ey + f`

Solves for critical point where U=V=0.

## Coordinate System

- Origin at `(0, 0)`
- Default simulation domain: `[-0.6, 0.6] × [-0.6, 0.6]`
- Vector fields return `(u, v)` components (not magnitude/angle)
- Robot readings include small Gaussian noise: `± 0.00` (configurable)

## Dependencies

```bash
pip install numpy matplotlib torch scipy pandas scikit-learn pyyaml
```

## Troubleshooting

**"Model files not found"**: Train models first using trainer scripts in appropriate predictor directory.

**Singular matrix errors**: Occurs with degenerate robot formation (collinear points). Code falls back to simpler methods.

**Oscillating behavior**: Reduce `timestep` or increase `momentum_alpha` in robot initialization.

**Poor ML predictions**: Retrain with more/diverse data or adjust network architecture in trainer scripts.

**YAML import error**: Install pyyaml: `pip install pyyaml`

**Formation not maintained**: Check `position_gain` in formation config. Default is 1.0. Increase for tighter formation control.

**Import errors after cleanup**: The old architecture files (main3.py, robot_cluster.py, etc.) have been removed. Use `main_omni.py` and the new OmniCluster architecture.
