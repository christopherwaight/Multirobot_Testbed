# VF_Robot - Multi-Robot Vector Field Navigation

Multi-robot (3 or 4) vector field navigation simulation testbed with formation control.

## Features

- **3-robot and 4-robot formations** with configurable geometry
- **Multiple vector field types**: sinks, sources, vortices, saddles
- **ML-based field approximation**: Neural networks, RBF interpolators, blended models
- **Realistic robot physics**: Momentum, max velocity (0.3 m/s), stiction (0.025 m/s)
- **Velocity tracking and plotting**: Automatic generation of velocity vs time plots
- **Multiple control primitives**: Center finding, orbiting, field following

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Simulations

**Using virtual environment (recommended):**

```bash
# From the VF_Robot/ directory
./venv/bin/python3 experiments/main_omni.py
```

**Or activate the virtual environment first:**

```bash
source venv/bin/activate  # macOS/Linux
python3 experiments/main_omni.py
```

### Outputs

- **Trajectory plots**: Displayed on screen showing robot paths and vector field
- **Velocity plots**: Automatically saved to `velocity_plots/` directory (if `SAVE_VELOCITY_PLOTS = True`)
  - Format: `{environment}_velocity_{mode}.png`
  - Shows velocity magnitude vs time for each robot
  - Useful for analyzing approach behavior, convergence, and stability

**View velocity plots:**
```bash
open velocity_plots/  # Opens directory with all plots
```

## Project Structure

```
VF_Robot/
├── src/                          # Core library code
│   ├── robot/                    # Robot components
│   │   ├── omnibot.py           # Individual robot (3 or 4)
│   │   ├── omni_cluster.py      # 3-robot cluster manager
│   │   └── quad_cluster.py      # 4-robot cluster manager
│   ├── control/                  # Control algorithms
│   │   ├── primitives.py        # 3-robot control strategies
│   │   ├── quad_primitives.py   # 4-robot control strategies
│   │   ├── kinematics.py        # 3-robot kinematics
│   │   └── quad_kinematics.py   # 4-robot kinematics
│   ├── fields/                   # Field abstractions
│   │   ├── field_types.py       # Field classes (Analytical, NN, RBF, Blended)
│   │   └── environments/        # Vector field environments
│   │       ├── Sink.py, Source.py, Vortex.py
│   │       ├── Sinking_Vortex.py, Spewing_Vortex.py
│   │       ├── Saddle.py
│   │       └── VF_bases.py      # Base field primitives
│   └── simulation/              # Simulation execution
│       ├── runner.py            # Main simulation loop
│       └── velocity_plotter.py  # Velocity plotting utilities
│
├── config/                      # Configuration files
│   └── formations/              # Formation configs (YAML)
│
├── experiments/                 # Experiment scripts
│   └── main_omni.py            # Main entry point
│
├── velocity_plots/             # Output directory for velocity plots
├── sinking_vortex_predictors/  # ML models for sinking vortex
├── saddle_predictors/          # ML models for saddle points
├── vortex_predictors/          # ML models for vortex
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── CLAUDE.md                  # Comprehensive documentation
```

## Configuration

Edit `experiments/main_omni.py` to configure:

- **NUM_ROBOTS**: `3` or `4` (robot count)
- **SIMULATION_MODE**: `"single"`, `"compare"`, or `"multi_env"`
- **FIELD_MODE**: `"analytical"`, `"nn"`, `"rbf"`, or `"blended"`
- **ENVIRONMENT**: e.g., `"sink1"`, `"vortex2"`, `"sinking_vortex3"`
- **CONTROL_PRIMITIVE**: e.g., `"critical_point_orbiter_plane_fitting"` (3-robot) or `"dual_jacobian_center_finder"` (4-robot)
- **FORMATION_CONFIG**: Path to formation YAML file
- **SAVE_VELOCITY_PLOTS**: `True` to save velocity plots, `False` to skip

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run simulation: `python3 experiments/main_omni.py`
3. Adjust config in `experiments/main_omni.py` as needed

## Documentation

See `CLAUDE.md` for comprehensive documentation including:
- Detailed architecture explanation
- Control primitives reference
- ML model training instructions
- Creating custom environments and formations

## Development

When adding new modules to `src/`, use relative imports:
- Within same package: `from .module import Class`
- From parent package: `from ..package.module import Class`

External scripts (like `experiments/main_omni.py`) should use absolute imports:
- `from src.robot.omni_cluster import OmniCluster`
