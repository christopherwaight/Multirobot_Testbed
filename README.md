# Multi-Robot Field Navigation Testbed

A research platform for distributed multi-robot systems that explore and navigate vector and scalar fields using formation-based control.

## Overview

This repository contains simulation and hardware validation code for multi-robot critical point detection in 2D fields. Robots form geometric clusters, sample the field simultaneously at multiple locations, estimate the Jacobian via plane fitting, and navigate to or orbit around critical points (sources, sinks, vortices, saddles).

**Core contribution:** Robots locate and navigate to field critical points using only local measurements -- no prior field knowledge, no measurement history, and no global positioning required beyond a shared coordinate frame.

---

## Repository Structure

```
Multirobot_Testbed/
├── trunk/
│   ├── Python_Simulations/
│   │   ├── Vector_Fields/VF_Robot/     # Main simulation codebase
│   │   └── Scalar Fields/              # Newton's method saddle finding
│   ├── robots_3/                       # 3-robot hardware experiment data
│   ├── robots_4/                       # 4-robot hardware experiment data
│   └── Deep_Learning/                  # MATLAB sensor calibration
├── Paper_Writing/
│   ├── Vector Field Paper/             # Main paper (canonical: Paper_Draft_3c.tex)
│   ├── Scalar Field Paper/             # Saddle point paper (draft)
│   └── Master's Thesis/
└── docs/                               # Architecture, control theory, hardware notes
```

---

## Python Simulation

### Entry Point

```bash
cd "trunk/Python_Simulations/Vector_Fields/VF_Robot"
source venv/bin/activate
python3 experiments/main_omni.py
```

### Key Source Files

| File | Role |
|------|------|
| `src/robot/omnibot.py` | Robot dynamics: momentum model, stiction, velocity limits |
| `src/robot/omni_cluster.py` | 3-robot cluster |
| `src/robot/quad_cluster.py` | 4-robot cluster |
| `src/control/primitives.py` | 3-robot control laws |
| `src/control/quad_primitives.py` | 4-robot control laws |
| `src/control/kinematics.py` | 3-robot kinematics |
| `src/fields/environments/` | 13 field definitions (sink, source, vortex, saddle, spirals, etc.) |
| `src/simulation/runner.py` | Simulation loop |
| `cluster_builder/clusterbuilder.py` | Formation geometry tool: SAS parameterization, symbolic Jacobian, YAML visualization |

### Configuration

Edit `experiments/main_omni.py`:

```python
NUM_ROBOTS = 3
FIELD_MODE = "analytical"          # "nn", "rbf", or "blended"
ENVIRONMENT = "sinking_vortex1"    # sink1-3, source1-3, vortex1, saddle, etc.
CONTROL_PRIMITIVE_3 = "critical_point_orbiter_plane_fitting"
```

### Scalar Field Simulation

```bash
cd "trunk/Python_Simulations/Scalar Fields"
python3 path_quality_analysis.py
```

---

## Hardware

**Robots:** Decabots (omnidirectional platforms with RGB color sensors)

**Testbed:** 1.6 x 1.6 m arena, OptiTrack motion capture, 10 Hz control rate (SCU Robotics Systems Laboratory)

**Field representation:** Printed floor maps using HSV color encoding. Hue encodes flow direction, saturation encodes magnitude.

**MATLAB analysis:**

```matlab
% 3-robot results
cd trunk/robots_3/vortex_tests
orbiter_plotter        % visualize orbits

cd trunk/robots_3
calculate_resting_point_stats

% 4-robot results
cd trunk/robots_4
allrunplotter
```

---

## Research Results

### Vector Field Paper (Paper_Draft_3c.tex, canonical)

**Hardware validation (169 experiments):**
- 157 convergence trials: 100% success rate
- 12 orbital trials: radius maintained within tolerance
- 6 field types: sink, source, vortex, saddle, sinking vortex, spewing vortex
- Average error: 0.012 m (vortex), 0.005 m (saddle)

**Simulation validation:**
- 8 fields, 1000 Monte Carlo trials per field, 100% convergence

### Scalar Field Paper (draft)

- Newton's method with formation rotation control
- With rotation (k_r = 0.3): 100% success across all orientations
- Without rotation: 60% success (fails at 45 deg and 67.5 deg)
- Optimal gain discovery: gains below 0.1 are worse than no rotation

---

## Key Concepts

### Why Three Robots

A linear field model has 6 coefficients (3 per component). Each robot provides one measurement per component. Three robots give exactly 3 equations per component -- a determined system. Fewer robots leave it underdetermined; more robots allow least-squares overdetermination.

### SAS Formation Parameterization

3-robot clusters are described by six state variables: (x_c, y_c, theta_c, p, q, beta), where p is robot 1 to 2 distance, q is robot 2 to 3 distance, and beta is the included angle at robot 2. The `clusterbuilder` tool generates valid formation configurations and computes symbolic Jacobians from this parameterization.

### Control Laws

The centroid velocity command is:

```
p_c_dot = k * r + k_t * r_hat_perp + k_r * (r_d - ||r||) * r_hat
```

where r = p* - p_c is the radial vector to the estimated critical point, r_d is the desired orbital radius, and k_t controls orbit direction.

---

## Publications

**Vector field navigation:**
- "Adaptive Navigation of Multirobot Systems to Critical Points in 2D Vector Fields"
- Christopher Waight, Christopher A. Kitts
- Status: Near submission

**Scalar field saddle finding:**
- Newton's method with 4-robot formations and rotation control
- Status: Draft in progress

---

## Dependencies

**Python** (venv at `trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/`):
- numpy, matplotlib, scipy, torch, scikit-learn, pandas, pyyaml

**MATLAB:**
- Deep Learning Toolbox
- Image Processing Toolbox

---

## Documentation

- `CLAUDE.md` -- notation, file map, and workflow reference for AI assistants
- `docs/architecture.md` -- simulation architecture
- `docs/control.md` -- control theory and field types
- `docs/hardware.md` -- hardware and MATLAB workflow
- `docs/troubleshooting.md` -- common errors

---

## Contact

Christopher Waight, Christopher A. Kitts (Advisor)
Santa Clara University, Robotics Systems Laboratory
