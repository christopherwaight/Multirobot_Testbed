# Multi-Robot Field Navigation Testbed

Simulation, hardware, and paper sources for multirobot adaptive navigation in 2D vector fields, from Santa Clara University's Robotic Systems Laboratory.

A cluster of robots samples a field at several points at once, fits a local model of the field, and steers on that model. It needs no prior map of the field and no measurement history. Three robots fit a plane and find critical points (sources, sinks, vortices, saddles). Six robots fit a quadratic, which makes the Jacobian determinant and its gradient available, and track the separatrices and coherent structures that divide a flow.

## Repository layout

```
Multirobot_Testbed/
├── trunk/
│   ├── Python_Simulations/
│   │   ├── Vector_Fields/VF_Robot/      # Main simulation codebase (3-, 4-, and 6-robot clusters)
│   │   ├── Vector_Fields/ocean_data/    # Santa Barbara Channel HF radar data and plots
│   │   ├── Separatrix_Control_testing/  # Notebook prototypes for the six-robot work
│   │   └── Archive/                     # Retired experiments, including scalar fields
│   ├── robots_3/, robots_4/             # 3- and 4-robot hardware data and MATLAB analysis
│   └── Deep_Learning/                   # MATLAB color-sensor calibration
├── Paper_Writing/
│   ├── Systems Paper/                   # Vector field paper (3- and 4-robot, hardware)
│   ├── Separatrix_and_OW_Paper/         # Separatrix paper (6-robot, simulation), Draft_12.tex
│   ├── PhD Thesis/                      # Generated from the papers by sources/assemble/
│   └── mcp_paper_search/                # Vector-store search over the reference corpus
├── Manim_Viz/                           # Manim animations of the fields and controllers
├── docs/notation.md                     # Symbol reference
├── CLAUDE.md                            # Working rules for AI agents
└── Waight_style.md                      # Writing guide for paper and thesis prose
```

## Running the simulation

```bash
cd trunk/Python_Simulations/Vector_Fields/VF_Robot
venv/bin/python3 experiments/main_omni.py
```

`main_omni.py` is configured by the constants in its CONFIGURATION SECTION, such as the number of robots, the field, and the control primitive. Key modules:

| Path | Role |
|---|---|
| `src/robot/` | Robot dynamics (`omnibot.py`) and the 3-, 4-, and 6-robot clusters |
| `src/control/` | Primitives and kinematics per cluster size. `pentagon_primitives.py` holds the six-robot trackers |
| `src/fields/environments/` | 13 analytical and data-driven fields, including the double gyre and ocean HF radar |
| `src/simulation/runner.py` | Simulation loop |
| `cluster_builder/clusterbuilder.py` | Formation geometry, SAS parameterization (p, q, beta), symbolic Jacobians |

Tests run from VF_Robot with `venv/bin/python3 -m pytest cluster_builder/tests/ tests/test_oecs_separatrix.py`.

## Hardware

Decabots, omnidirectional robots with RGB color sensors, run in a 1.6 m by 1.6 m arena under OptiTrack motion capture at 10 Hz. The field is a printed floor map in which hue encodes flow direction and saturation encodes magnitude. MATLAB analysis scripts live beside the data in `trunk/robots_3/` and `trunk/robots_4/`.

## Papers

Both target IEEE Systems Journal. Authors are Christopher Waight and Christopher A. Kitts.

- **Adaptive Navigation of Multirobot Systems to Critical Points in 2D Vector Fields in Simulation and Experiment.** Three- and four-robot clusters, validated in 169 hardware experiments. The cluster converged in all 157 convergence trials, with average position errors of 0.012 m or less. Source in `Paper_Writing/Systems Paper/cwaight_systems_paper/`.
- **Second-Order Cooperative Field Estimation for Multirobot Tracking of Coherent Flow Structures.** A six-robot pentagon-plus-center formation fits a quadratic field model and tracks separatrices on a double gyre benchmark and on Santa Barbara Channel HF radar data. Source in `Paper_Writing/Separatrix_and_OW_Paper/`.

## Dependencies

Python 3.13 with numpy, scipy, matplotlib, pandas, pyyaml, torch, and scikit-learn, installed in the VF_Robot venv. LaTeX builds use BasicTeX. The MATLAB code needs the Deep Learning and Image Processing toolboxes.

## Contact

Christopher Waight and Christopher A. Kitts (advisor), Robotic Systems Laboratory, Santa Clara University.
