# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Overview

This is the **experimental code module** for multi-robot vector field exploration. It contains the core simulation infrastructure (robot dynamics, formation control, kinematics, fields, control primitives) that has been extracted into a reusable library structure.

**Key Purpose:** Provide modular, reusable components for 3-robot and 4-robot formation control with distributed field sensing and critical point detection.

**Location Context:** `Paper_Writing/experimental/` — parallel to the main VF_Robot implementation in `trunk/Python Simulations/Vector_Fields/VF_Robot/`, but with cleaner separation of concerns and refined architecture.

---

## Architecture Overview

The codebase is organized as a Python package with clear layer separation:

```
src/
├── robot/               # Individual robot dynamics and cluster management
│   ├── omnibot.py      # Single omnidirectional robot with momentum dynamics
│   ├── omni_cluster.py # 3-robot cluster with formation control
│   └── quad_cluster.py # 4-robot cluster with formation control
│
├── control/            # Formation control and inverse kinematics
│   ├── kinematics.py   # 3-robot forward/inverse kinematics
│   ├── primitives.py   # 3-robot control algorithms
│   ├── quad_kinematics.py   # 4-robot forward/inverse kinematics
│   └── quad_primitives.py   # 4-robot control algorithms
│
├── fields/             # Vector and scalar field abstractions
│   ├── field_types.py  # Base classes: Field, VectorField, AnalyticalField, NNField, RBFField, ScalarField
│   └── environments/   # Pre-built field implementations (Sink, Vortex, Sinking_Vortex, Saddle, etc.)
│
└── simulation/         # Execution and visualization
    ├── runner.py       # Main simulation loop and plotting
    ├── velocity_plotter.py  # Velocity analysis visualization
    └── center_estimation_plotter.py  # Center tracking visualization

config/
└── formations/         # YAML formation configuration files (3-robot and 4-robot)
```

### Data Flow

```
Input: YAML formation config + VectorField + Control primitive
  ↓
OmniCluster/QuadCluster (reads config, initializes robots in formation)
  ↓
Simulation Loop:
  1. Sample field at each robot position
  2. Call control primitive with cluster state → desired centroid velocity
  3. Formation control (error correction via proportional gains)
  4. Inverse kinematics → individual robot velocities
  5. Each robot applies momentum dynamics + physical constraints
  ↓
Output: Trajectories, centroid path, final positions
```

---

## Core Components

### Robot Dynamics (omnibot.py)

**Single robot model:**
- State: position (x, y), velocity (vx, vy)
- Momentum dynamics: `v = α·v_old + (1-α)·v_cmd`
- Physical constraints:
  - Max velocity: 0.3 m/s
  - Stiction threshold: 0.025 m/s (below this, friction stops the robot)

**Key Parameters:**
- `momentum_alpha = 0.7` (higher = more inertia, closer to hardware behavior)
- `timestep = 0.1` s (10 Hz control rate)

### Cluster Control (omni_cluster.py, quad_cluster.py)

**OmniCluster (3-robot):**
- Formation parameterization: SAS (p, q, β) — side-angle-side triangle
- Reads desired formation from YAML config
- Computes forward kinematics to extract current (x_c, y_c, θ_c)
- Formation error → shape velocities via proportional control
- Inverse Jacobian converts centroid velocity + shape velocities to robot commands

**QuadCluster (4-robot):**
- Formation parameterization: diagonals (d1, d2, r1, r2, φ)
- Similar structure but with 4 robots forming overlapping triangles

### Field Abstraction (field_types.py)

**Unified interface:**
```python
field.get_value(x, y)  # Returns (u, v) for VectorField or φ for ScalarField
```

**Types:**
- `AnalyticalField(func)` — Mathematical function (fastest, exact)
- `NNField(...)` — Neural network approximation (hue + saturation models)
- `RBFField(...)` — RBF interpolator (smooth, well-conditioned)
- `ScalarField(func)` — Scalar potential function

### Control Primitives (primitives.py, quad_primitives.py)

All primitives follow the signature:
```python
def primitive(cluster) -> (vx_c, vy_c):  # Desired centroid velocity
    return vx_c, vy_c
```

**Available 3-robot primitives:**
- `vector_sum` — Simple field following
- `critical_point_plane_fitting` — Find zeros of u and v using plane fitting
- `critical_point_cross_product` — Alternative plane-fitting method
- etc. (see primitives.py for full list)

**Available 4-robot primitives:**
- `dual_jacobian_center_finder` — Use both top and bottom triangles
- etc. (see quad_primitives.py for full list)

---

## Key Configuration

### Formation Config (YAML)

**3-Robot Example** (`config/formations/equilateral_default.yaml`):
```yaml
formation:
  p: 0.33              # Distance robot 1→2
  q: 0.33              # Distance robot 2→3
  beta_degrees: 60.0   # Angle at robot 2
  position_gain: 0.5   # Proportional gain for p, q error
  angle_gain: 0.01     # Proportional gain for β error
```

**4-Robot Example** (`config/formations/quad_default.yaml`):
```yaml
formation:
  type: "quadrilateral"
  d1: 0.433            # Diagonal 1-3 length
  d2: 0.25             # Diagonal 2-4 length
  r1: 0.5              # Intersection ratio diagonal 1
  r2: 0.5              # Intersection ratio diagonal 2
  phi_degrees: 90.0
  position_gain: 0.5
  angle_gain: 0.01
```

---

## Common Development Tasks

### Running a Simulation

This module is designed to be imported and used from other scripts. Example usage pattern:

```python
from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.control.primitives import critical_point_plane_fitting
from src.simulation.runner import execute_omni_simulation

# Create field
field = AnalyticalField(sinking_vortex1)

# Create cluster with formation config
cluster = OmniCluster('config/formations/equilateral_default.yaml', field)

# Run simulation and plot
execute_omni_simulation(cluster, critical_point_plane_fitting, 
                        title="Critical Point Finding", sim_time=500)
```

### Adding a New Control Primitive

1. Add function to `src/control/primitives.py` (or `quad_primitives.py` for 4-robot):
```python
def my_algorithm(cluster):
    """
    Control primitive description.
    
    Args:
        cluster: OmniCluster instance
    
    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    positions = cluster.get_robot_positions()
    readings = cluster.sample_field_at_robots()
    
    # Your algorithm
    vx_c, vy_c = 0.0, 0.0
    return vx_c, vy_c
```

2. Test by importing and using in a simulation script (see example above).

### Creating a New Vector Field

1. Add function to appropriate file in `src/fields/environments/`:
```python
def my_field(x, y):
    """Field description."""
    u = ...  # Compute u component
    v = ...  # Compute v component
    return u, v
```

2. Wrap in AnalyticalField:
```python
field = AnalyticalField(my_field)
```

3. Or create a dedicated environment file following the pattern of existing ones (Sink.py, Vortex.py, etc.).

### Adding a New Formation Configuration

Create a YAML file in `config/formations/` with the same structure as existing configs. Path resolution in OmniCluster automatically finds configs relative to project root.

---

## Important Design Patterns

### Formation Control Architecture

The cluster control follows a strict two-tier approach:

1. **Centroid Control** (from control primitive):
   - Primitive computes desired `(vx_c, vy_c)` based on field sampling
   
2. **Formation Maintenance** (internal to cluster):
   - Cluster tracks formation error: `(Δp, Δq, Δβ)`
   - Generates shape velocities proportionally: `v_shape = gain * error`
   - Combines centroid velocity + shape velocities
   - Converts to robot velocities via inverse Jacobian

This separation allows primitives to focus on high-level navigation without worrying about formation mechanics.

### Momentum Dynamics

Momentum is applied at the individual robot level (omnibot.py), not at the cluster level. This is physically realistic but can make the system slightly underdamped if `alpha` is too high.

**Tuning:**
- High `alpha` (0.8+): More inertia, smoother trajectories, but overshoots targets
- Low `alpha` (0.3-0.5): Tighter control, faster response, but jittery
- Default `alpha = 0.7`: Balanced for paper results

### Field Sampling Patterns

All field access goes through `cluster.sample_field_at_robots()`:
```python
readings = cluster.sample_field_at_robots()  # List of (u, v) tuples, one per robot
```

This abstraction allows easy field switching (analytical ↔ NN ↔ RBF) without changing primitive code.

---

## Troubleshooting

### "ModuleNotFoundError" when running simulations

This module is a library, not a standalone app. Always import and run from external scripts. Verify imports use relative paths from the experimental directory:
```python
from src.robot.omni_cluster import OmniCluster
from src.fields.environments.Sinking_Vortex import sinking_vortex1
```

### Robots not maintaining formation

Check formation control gains in YAML:
- If formation drifts, increase `position_gain` (default 0.5)
- If formation oscillates, decrease `position_gain` or increase `angle_gain`
- Ensure formation parameters (p, q, β) are geometrically valid (triangle inequality)

### "Singular matrix" errors in kinematics

Formation has become degenerate (collinear robots). Cluster.move() catches this and falls back to `vector_sum`. Usually self-corrects as robots move.

### Slow formation convergence with low momentum

If `momentum_alpha` is too low (< 0.5), robots may overshoot correction commands repeatedly. Increase to 0.7 or check if control gains are too aggressive.

### Field values are NaN or infinite

Check for division-by-zero in field functions. Most field implementations include small epsilon values (1e-10) to prevent this, but custom fields need explicit handling:
```python
r = np.sqrt((x-xc)**2 + (y-yc)**2) + 1e-10  # Prevent division by zero
```

---

## Related Directories

- **Main VF_Robot implementation:** `trunk/Python Simulations/Vector_Fields/VF_Robot/` — full-featured version with ML training, analysis scripts, and experiment runners
- **Scalar field simulations:** `trunk/Python Simulations/Scalar Fields/` — Newton's method for saddle finding
- **Hardware validation:** `trunk/robots_3/`, `trunk/robots_4/` — physical robot experiment data and MATLAB analysis
- **Papers:** `Paper_Writing/Vector Field Paper/Paper_Draft_3A.tex` — canonical paper with 169 hardware experiments

---

## Notes

- This module prioritizes clean architecture and code reuse over feature completeness
- It serves as the foundation for the main VF_Robot implementation in `trunk/`
- All robot physics (momentum, stiction, max velocity) match hardware parameters
- Formation control gains are tuned for stable convergence; adjust for your specific use case
