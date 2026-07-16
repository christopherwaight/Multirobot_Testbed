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
│   │   └── Archive/                    # Deprecated experiments (incl. old Scalar Fields)
│   ├── robots_3/                       # 3-robot hardware experiment data
│   ├── robots_4/                       # 4-robot hardware experiment data
│   └── Deep_Learning/                  # MATLAB sensor calibration
├── Paper_Writing/
│   ├── Vector Field Paper/             # Main paper (canonical: Paper_Draft_5A.tex)
│   ├── Separatrix_and_OW_Paper/        # Separatrix / Okubo-Weiss paper (Paper_Draft_2A.tex)
│   └── Master's Thesis/
└── docs/                               # Notation reference, architecture/control/hardware notes
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
| `src/fields/environments/` | 12 field-definition modules (sink, source, vortex, saddle, spirals, double gyre, separatrix, ocean HFR, etc.) |
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

### Separatrix / Okubo-Weiss Simulation

The six-robot separatrix and Okubo-Weiss tracking experiments live under
`trunk/Python_Simulations/Separatrix_Control_testing/`. See
`Paper_Writing/Separatrix_and_OW_Paper/` and the repo-root `plan.md` for the current
campaign.

> Note: the earlier scalar-field / Newton saddle-finding line of work is deprecated.
> Its simulation code is archived at `trunk/Python_Simulations/Archive/Scalar Fields/`.

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

### Vector Field Paper (Paper_Draft_5A.tex, canonical)

**Hardware validation (169 experiments):**
- 157 convergence trials: 100% success rate
- 12 orbital trials: radius maintained within tolerance
- 6 field types: sink, source, vortex, saddle, sinking vortex, spewing vortex
- Average error: 0.012 m (vortex), 0.005 m (saddle)

**Simulation validation:**
- 8 fields, 1000 Monte Carlo trials per field, 100% convergence

### Separatrix / Okubo-Weiss Paper (Paper_Draft_2A.tex, active draft)

- Six-robot formation (regular pentagon plus center robot) estimates the field
  Jacobian and traverses the separatrix / Okubo-Weiss trench network using only local
  measurements.
- Body complete (18 pages); target venue IEEE Transactions on Robotics (T-RO).
- Verified symbolically, in Monte Carlo, and in closed-loop simulation, and validated
  on a real ocean HFR field. See the repo-root `plan.md` for the working tracker.

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

**Separatrix / Okubo-Weiss navigation:**
- Distributed separatrix and Okubo-Weiss trench tracking with a six-robot formation
- Christopher Waight, Christopher A. Kitts
- Status: Draft complete (Paper_Draft_2A.tex), T-RO target

---

## Dependencies

**Python** (venv at `trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/`):
- numpy, matplotlib, scipy, torch, scikit-learn, pandas, pyyaml

**MATLAB:**
- Deep Learning Toolbox
- Image Processing Toolbox

---

## Documentation

- `CLAUDE.md` -- style rules, notation hard rules, file map, and workflow for AI assistants
- `docs/notation.md` -- full symbol reference (field, estimation, control, dynamics, SAS)
- `docs/architecture.md`, `docs/control.md`, `docs/hardware.md`, `docs/troubleshooting.md`
  -- placeholder stubs, not yet written

---

## Contact

Christopher Waight, Christopher A. Kitts (Advisor)
Santa Clara University, Robotic Systems Laboratory
