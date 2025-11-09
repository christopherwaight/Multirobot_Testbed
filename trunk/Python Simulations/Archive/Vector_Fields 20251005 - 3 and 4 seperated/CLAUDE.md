# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This directory contains multi-robot vector field navigation simulations. The robots form clusters (3-robot or 4-robot formations) that sample vector fields to estimate critical points (sinks, sources, vortices, saddles) using only local measurements.

## Directory Structure

```
Vector_Fields/
├── VF_3_Robot/          # 3-robot simulation (PRIMARY, ACTIVELY MAINTAINED)
│   ├── src/            # Modular source code
│   │   ├── robot/      # Robot and cluster implementations
│   │   ├── control/    # Control primitives
│   │   ├── fields/     # Field abstractions and environments
│   │   └── simulation/ # Simulation execution
│   ├── experiments/    # Main entry points
│   ├── config/         # Formation configurations (YAML)
│   └── *_predictors/   # ML models for different field types
│
└── VF_4_Robot/         # 4-robot simulation (EXPERIMENTAL)
    ├── clusters/       # 4-robot cluster implementation
    ├── primitives/     # 4-robot control primitives
    ├── env/            # Vector field environments
    └── exec_sim/       # Simulation execution
```

## Quick Start

### For 3-Robot Simulations (Recommended)

```bash
cd VF_3_Robot
python3 experiments/main_omni.py
```

**See `VF_3_Robot/CLAUDE.md` for comprehensive documentation** including:
- Detailed architecture with formation control using SAS parameterization (p, q, β)
- All available control primitives
- ML model training instructions
- Creating custom environments and formations
- Configuration options

### For 4-Robot Simulations (Experimental)

```bash
cd VF_4_Robot
python3 main4.py
```

**Key differences in 4-robot:**
- Uses two triangular sub-formations (top and bottom)
- Control primitives use dual Jacobian estimation
- Less actively maintained than 3-robot version

## Architecture Comparison

### VF_3_Robot (Modern Architecture)
- **Modular source structure** under `src/`
- **Formation control** via YAML config files with SAS parameterization
- **Omnibot class** with momentum-based dynamics
- **OmniCluster class** with forward/inverse kinematics
- **Unified VectorField interface** (AnalyticalField, NNField, RBFField, BlendedField)
- **Control primitives return velocities** not positions

### VF_4_Robot (Legacy Architecture)
- **Flat structure** with top-level modules
- **Hardcoded formation** in RobotCluster class
- **RobotCluster class** manages 4 robots
- **Direct function calls** for field evaluation
- **Control primitives may return positions or velocities**

## Common Commands

### Run 3-Robot Simulation (Single Environment)
```bash
cd VF_3_Robot
python3 experiments/main_omni.py
```

Edit `experiments/main_omni.py` to configure:
- `SIMULATION_MODE = "single"`
- `FIELD_MODE = "analytical"` (or "nn", "rbf", "blended")
- `ENVIRONMENT = "sinking_vortex1"` (or sink1-3, vortex1-3, etc.)
- `CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"`

### Compare Field Approximation Methods
```bash
cd VF_3_Robot
# In experiments/main_omni.py, set:
# SIMULATION_MODE = "compare"
python3 experiments/main_omni.py
```

Shows side-by-side comparison of Blended, NN, RBF, and Analytical field modes.

### Train ML Models for 3-Robot
```bash
cd VF_3_Robot/sinking_vortex_predictors
python3 synthetic_generator.py        # Generate training data
python3 nn_trainer_sinking_vortex.py  # Train neural network
python3 rbf_trainer_sinking_vortex.py # Train RBF interpolator
```

### Run 4-Robot Simulation
```bash
cd VF_4_Robot
python3 main4.py
```

Configure in `main4.py`:
- `SIMULATION_MODE = "single"` (or "compare", "multi_env")
- `ENVIRONMENT = "vortex1"`
- `CONTROL_PRIMITIVE = "dual_jacobian_center_finder"`

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

Pre-configured environments (available in both VF_3_Robot and VF_4_Robot):
- **sink1, sink2, sink3** - Pure inward radial flow
- **source1, source2, source3** - Pure outward radial flow
- **vortex1, vortex2, vortex3** - Pure rotational flow
- **sinking_vortex1-3** - Combined rotation + inward flow
- **spewing_vortex1-3** - Combined rotation + outward flow
- **saddle1, saddle2, saddle3** - Hyperbolic saddle points

## Development Workflow

### When to Use VF_3_Robot
Use for all new development and experiments:
- Modern modular architecture
- Better documented
- Formation control via config files
- Cleaner separation of concerns
- Active maintenance

### When to Use VF_4_Robot
Only for specific 4-robot formation experiments:
- Dual Jacobian estimation
- Larger formation sampling
- Legacy code reference

### Adding New Control Primitives

**For VF_3_Robot:**
1. Add function to `src/control/omni_primitives.py`
2. Function signature: `def my_algorithm(cluster) -> (vx_c, vy_c)`
3. Register in `experiments/main_omni.py` control_primitive_map

**For VF_4_Robot:**
1. Add function to `primitives/control_primitives.py`
2. Function signature depends on primitive type
3. Register in `main4.py` control_primitive_map

### Creating New Environments

**For VF_3_Robot:**
1. Add to `src/fields/environments/` (see existing files as templates)
2. Import in `experiments/main_omni.py`
3. Add to `environment_map`

**For VF_4_Robot:**
1. Add to `env/` directory
2. Import in `main4.py`
3. Add to `environment_map`

## Dependencies

Both simulators require:
```bash
pip install numpy matplotlib torch scipy pandas scikit-learn pyyaml
```

VF_3_Robot has a `requirements.txt`:
```bash
cd VF_3_Robot
pip install -r requirements.txt
```

## Important Notes

1. **VF_3_Robot is the primary codebase** - Use this for new work
2. **VF_4_Robot is experimental** - Legacy architecture, less maintained
3. **Detailed documentation exists** in `VF_3_Robot/CLAUDE.md` - Consult for architecture details
4. **ML models are field-specific** - Use appropriate predictor directory (sinking_vortex_predictors, saddle_predictors, vortex_predictors)
5. **Formation control in VF_3_Robot** uses SAS parameterization (p, q, β) via YAML configs
6. **Both use momentum-based robot dynamics** for smooth, realistic motion

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
