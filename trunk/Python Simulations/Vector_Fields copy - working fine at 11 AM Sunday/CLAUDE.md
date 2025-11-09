# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This directory contains multi-robot vector field navigation simulations. The robots form clusters (3-robot or 4-robot formations) that sample vector fields to estimate critical points (sinks, sources, vortices, saddles) using only local measurements.

## Directory Structure

```
Vector_Fields/
└── VF_Robot/           # Multi-robot simulation (PRIMARY, ACTIVELY MAINTAINED)
    ├── src/            # Modular source code
    │   ├── robot/      # Robot and cluster implementations (3 & 4 robots)
    │   ├── control/    # Control primitives (3 & 4 robots)
    │   ├── fields/     # Field abstractions and environments
    │   └── simulation/ # Simulation execution
    ├── experiments/    # Main entry points
    ├── config/         # Formation configurations (YAML)
    └── *_predictors/   # ML models for different field types
```

## Quick Start

### For 3-Robot Simulations (Recommended)

```bash
cd VF_Robot
python3 experiments/main_omni.py
```

**See `VF_Robot/CLAUDE.md` for comprehensive documentation** including:
- Detailed architecture with formation control using SAS parameterization (p, q, β)
- All available control primitives
- ML model training instructions
- Creating custom environments and formations
- Configuration options

### For 4-Robot Simulations

```bash
cd VF_Robot
# In experiments/main_omni.py, set NUM_ROBOTS = 4
python3 experiments/main_omni.py
```

**Key features of 4-robot mode:**
- Uses two triangular sub-formations (top and bottom)
- Control primitives use dual Jacobian estimation for robust center finding
- Same modern architecture as 3-robot mode

## Architecture

### VF_Robot (Modern Unified Architecture)
- **Modular source structure** under `src/`
- **Formation control** via YAML config files:
  - 3-robot: SAS parameterization (p, q, β)
  - 4-robot: Diagonal parameterization (d1, d2, r1, r2, φ)
- **Omnibot class** with momentum-based dynamics
- **OmniCluster** (3-robot) and **QuadCluster** (4-robot) with forward/inverse kinematics
- **Unified VectorField interface** (AnalyticalField, NNField, RBFField, BlendedField)
- **Control primitives return velocities** `(vx_c, vy_c)` or `(vx_c, vy_c, omega_c)`

## Common Commands

### Run Simulation (Single Environment)
```bash
cd VF_Robot
python3 experiments/main_omni.py
```

Edit `experiments/main_omni.py` to configure:
- `NUM_ROBOTS = 3` (or 4 for 4-robot mode)
- `SIMULATION_MODE = "single"`
- `FIELD_MODE = "analytical"` (or "nn", "rbf", "blended")
- `ENVIRONMENT = "sinking_vortex1"` (or sink1-3, vortex1-3, etc.)
- `CONTROL_PRIMITIVE_3 = "critical_point_orbiter_plane_fitting"` (for 3-robot)
- `CONTROL_PRIMITIVE_4 = "dual_jacobian_center_finder"` (for 4-robot)

### Compare Field Approximation Methods
```bash
cd VF_Robot
# In experiments/main_omni.py, set:
# SIMULATION_MODE = "compare"
python3 experiments/main_omni.py
```

Shows side-by-side comparison of Blended, NN, RBF, and Analytical field modes.

### Train ML Models
```bash
cd VF_Robot/sinking_vortex_predictors
python3 synthetic_generator.py        # Generate training data
python3 nn_trainer_sinking_vortex.py  # Train neural network
python3 rbf_trainer_sinking_vortex.py # Train RBF interpolator
```

## Key Concepts

### Critical Point Estimation
Both simulators estimate critical points using triangular robot formations to compute the Jacobian matrix:
```
J = [du/dx  du/dy]
    [dv/dx  dv/dy]
```

From the Jacobian:
- **Curl** = dv/dx - du/dy (rotation measure)
- **Divergence** = du/dx + dv/dy (expansion/contraction)

### Machine Learning Field Approximation
Vector fields are encoded as **(hue, saturation)**:
- **hue** → direction as (sin(θ), cos(θ))
- **saturation** → magnitude scaled to [0,1]

Three approximation methods:
- **Neural Network** - Two networks (hue_model, sat_model)
- **RBF Interpolator** - Radial basis function interpolation
- **Blended** - Weighted combination (default: 90% RBF, 10% NN)

### Vector Field Environments

Pre-configured environments (in `VF_Robot/src/fields/environments/`):
- **sink1, sink2, sink3** - Pure inward radial flow
- **source1, source2, source3** - Pure outward radial flow
- **vortex1, vortex2, vortex3** - Pure rotational flow
- **sinking_vortex1-3** - Combined rotation + inward flow
- **spewing_vortex1-3** - Combined rotation + outward flow
- **saddle1, saddle2, saddle3** - Hyperbolic saddle points

## Development Workflow

### Adding New Control Primitives

**For 3-robot:**
1. Add function to `src/control/primitives.py`
2. Function signature: `def my_algorithm(cluster) -> (vx_c, vy_c)`
3. Register in `experiments/main_omni.py` in `control_primitive_map_3`

**For 4-robot:**
1. Add function to `src/control/quad_primitives.py`
2. Function signature: `def my_algorithm(cluster) -> (vx_c, vy_c)` or `(vx_c, vy_c, omega_c)`
3. Register in `experiments/main_omni.py` in `control_primitive_map_4`

### Creating New Environments

1. Add to `src/fields/environments/` (see existing files as templates)
2. Function signature: `def my_field(x, y) -> (u, v)`
3. Import in `experiments/main_omni.py`
4. Add to `environment_map`

## Dependencies

Install dependencies:
```bash
cd VF_Robot
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib torch scipy pandas scikit-learn pyyaml
```

## Important Notes

1. **VF_Robot is the unified codebase** - Supports both 3-robot and 4-robot formations via `NUM_ROBOTS` config
2. **VF_4_Robot has been archived** - Legacy 4-robot code moved to `../Archive/VF_4_Robot` (all functionality merged)
3. **Detailed documentation exists** in `VF_Robot/CLAUDE.md` - Consult for architecture details
4. **ML models are field-specific** - Use appropriate predictor directory (sinking_vortex_predictors, saddle_predictors, vortex_predictors)
5. **Formation control**:
   - 3-robot: SAS parameterization (p, q, β) via YAML configs
   - 4-robot: Diagonal parameterization (d1, d2, r1, r2, φ) via YAML configs
6. **Both modes use momentum-based robot dynamics** for smooth, realistic motion

## Migration from VF_4_Robot

The VF_4_Robot directory has been **archived** to `../Archive/VF_4_Robot`. All its functionality has been integrated into VF_Robot:

**What was merged:**
- 4-robot kinematics with diagonal parameterization
- Dual Jacobian estimation (using top and bottom triangles)
- All 4-robot control primitives
- Formation configurations

**How to use 4-robot mode:**
```python
# In VF_Robot/experiments/main_omni.py
NUM_ROBOTS = 4
```

The merged implementation provides:
- ✅ Same dual Jacobian functionality
- ✅ All control primitives ported
- ✅ Cleaner modular architecture
- ✅ Unified codebase for both 3 and 4 robots
- ✅ Config-based formation control

## Research Context

This codebase implements distributed sensing and control for multi-robot systems exploring unknown vector fields. The core research questions:
- How can robots estimate critical points using only local measurements?
- How accurate are ML approximations vs analytical fields?
- What control strategies work best for different field types?
- How does formation geometry affect estimation accuracy?

The robots use:
- **Formation-based sampling** to estimate local field derivatives
- **Theoretical estimators** derived from fluid dynamics
- **Machine learning** to approximate fields when analytical forms are unknown
- **Physics-based control** with momentum for smooth motion
