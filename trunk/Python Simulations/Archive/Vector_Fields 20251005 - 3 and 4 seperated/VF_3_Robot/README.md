# VF_3_robot_test - Multi-Robot Vector Field Navigation

3-robot vector field navigation simulation testbed with formation control.

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
# From the VF_3_robot_test/ directory
./venv/bin/python3 experiments/main_omni.py
```

**Or activate the virtual environment first:**

```bash
source venv/bin/activate  # macOS/Linux
python3 experiments/main_omni.py
```

**Note:** The script automatically adds the project root to the Python path, so it works regardless of where you run it from.

## Project Structure

```
VF_3_robot_test/
├── src/                          # Core library code
│   ├── robot/                    # Robot components
│   │   ├── omnibot.py           # Individual robot
│   │   └── omni_cluster.py      # Robot cluster manager
│   ├── control/                  # Control algorithms
│   │   ├── primitives.py        # Control strategies
│   │   └── kinematics.py        # Formation kinematics
│   ├── fields/                   # Field abstractions
│   │   ├── field_types.py       # Field classes (Analytical, NN, RBF, Blended)
│   │   └── environments/        # Vector field environments
│   │       ├── Sink.py, Source.py, Vortex.py
│   │       ├── Sinking_Vortex.py, Spewing_Vortex.py
│   │       ├── Saddle.py
│   │       └── VF_bases.py      # Base field primitives
│   └── simulation/              # Simulation execution
│       └── runner.py            # Main simulation loop
│
├── config/                      # Configuration files
│   └── formations/              # Formation configs (YAML)
│
├── experiments/                 # Experiment scripts
│   └── main_omni.py            # Main entry point
│
├── sinking_vortex_predictors/  # ML models for sinking vortex
├── saddle_predictors/          # ML models for saddle points
├── vortex_predictors/          # ML models for vortex
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── CLAUDE.md                  # Detailed documentation
```

## Configuration

Edit `experiments/main_omni.py` to configure:

- **SIMULATION_MODE**: `"single"`, `"compare"`, or `"multi_env"`
- **FIELD_MODE**: `"analytical"`, `"nn"`, `"rbf"`, or `"blended"`
- **ENVIRONMENT**: e.g., `"sink1"`, `"vortex2"`, `"sinking_vortex3"`
- **CONTROL_PRIMITIVE**: e.g., `"critical_point_orbiter_plane_fitting"`
- **FORMATION_CONFIG**: Path to formation YAML file

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
