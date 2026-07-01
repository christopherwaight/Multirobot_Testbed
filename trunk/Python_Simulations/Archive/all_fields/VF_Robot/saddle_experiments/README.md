# Saddle Point Finding Experiments

This directory contains standalone experiments for saddle point finding using Newton's method with 4-robot formations.

## Quick Start

### Run the Saddle Point Finding Experiment

```bash
cd saddle_experiments
python3 saddle_experiment.py
```

This will:
1. Create a 4-robot square formation
2. Place it in a bimodal Gaussian scalar field (saddle at origin)
3. Use Newton's method with Hessian estimation to find the saddle
4. Generate visualization plots and save diagnostics

### Expected Output

The script will display 6 plots showing:
- 2D contour view with trajectory
- 3D surface view with trajectory
- Distance to saddle vs iteration (log scale)
- Gradient magnitude vs iteration (log scale)
- Hessian eigenvalues vs iteration
- X and Y positions vs iteration

Results are saved to `saddle_experiments/results/`:
- `saddle_finding_with_rotation.png` - Visualization plots
- `diagnostics.txt` - Text file with iteration history

### Configuration Options

Edit `saddle_experiment.py` to customize:

```python
# Simulation parameters
SIM_TIME = 500  # Increase to 1000-2000 for full convergence
USE_ROTATION_CONTROL = True  # Set False to disable rotation control
START_POSITION = (0.5, -2.0)  # Starting position (away from saddle)

# Robot dynamics
TIMESTEP = 0.1  # Integration timestep
MOMENTUM_ALPHA = 0.7  # Momentum coefficient (0.7 = 70% momentum)

# Visualization
SHOW_PLOTS = True  # Display plots interactively
SAVE_PLOTS = True  # Save plots to file
```

## Understanding the Results

### Convergence Metrics

**Distance to Saddle:**
- Should decrease monotonically toward zero
- Final distance < 0.01 indicates good convergence

**Gradient Magnitude:**
- Should approach zero at critical point
- |∇z| < 0.001 indicates critical point found

**Hessian Eigenvalues:**
- At a saddle point: one positive, one negative eigenvalue
- λ₁ < 0 < λ₂ confirms saddle detection

### Newton's Method with Rotation Control

When `USE_ROTATION_CONTROL = True`:
- Formation rotates to align with Newton step direction
- Improves gradient/Hessian estimates along approach axis
- Generally faster and more accurate convergence

When `USE_ROTATION_CONTROL = False`:
- Formation maintains fixed orientation
- Still converges but may take longer
- More susceptible to orientation-dependent errors

## The Bimodal Gaussian Saddle Field

The default field is defined as:
```python
f(x,y) = log(exp(G₁) + exp(G₂))
where G₁ = -((x+2)² + y²)/2
      G₂ = -((x-2)² + y²)/2
```

This creates:
- Two Gaussian peaks at (-2, 0) and (+2, 0)
- Saddle point exactly at (0, 0)
- Smooth landscape ideal for Newton's method

## Troubleshooting

**"No module named 'src'":**
- Make sure you're running from the `saddle_experiments/` directory
- The script auto-adds parent directory to Python path

**Plots don't show:**
- Check `SHOW_PLOTS = True` in configuration
- Try running with `python3 -i saddle_experiment.py` for interactive mode

**Poor convergence:**
- Increase `SIM_TIME` (try 1000-2000 steps)
- Enable rotation control: `USE_ROTATION_CONTROL = True`
- Verify formation config: `config/formations/quad_square_default.yaml` exists

**Formation breaks apart:**
- Check `position_gain` in formation config (should be ~1.0)
- Verify timestep isn't too large (0.1 is recommended)

## Advanced Usage

### Testing Different Fields

Modify the import to test other scalar fields:

```python
# Try a paraboloid minimum
from src.fields.scalar_environments.Paraboloid import paraboloid_min1
field = AnalyticalScalarField(paraboloid_min1)

# Try another saddle configuration
from src.fields.scalar_environments.Saddle import bimodal_saddle2
field = AnalyticalScalarField(bimodal_saddle2)
```

### Testing 3-Robot Gradient Descent

For gradient-based methods (not saddle finding):

```python
from src.robot.omni_cluster import OmniCluster
import src.control.scalar_primitives as scp

# Use 3-robot formation
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

# Use gradient descent
cluster.move(scp.gradient_descent)
```

## Rotation Control Comparison

### Why Rotation Control Matters

Newton's method for saddle finding requires accurate Hessian estimation from 4-robot measurements. Formation rotation can improve convergence by:

1. **Aligning with approach direction** - Formation samples gradient along approach axis
2. **Improving numerical conditioning** - Better orientation reduces estimation errors
3. **Faster convergence** - More direct path to saddle point

### Running the Comparison

**Option 1: Compare both methods side-by-side**
```bash
python3 compare_rotation_methods.py
```

This runs both methods from the same starting position and generates:
- Side-by-side trajectory plots
- Convergence comparison (distance, gradient)
- Rotation behavior analysis
- Performance metrics table

**Option 2: Run methods individually**
```bash
# Baseline (NO rotation control)
python3 saddle_experiment_NO_rotation.py

# Enhanced (WITH rotation control)
python3 saddle_experiment_WITH_rotation.py
```

### Understanding the Results

**Rotation Control OFF (baseline):**
- Formation maintains roughly constant orientation
- May take longer to converge
- Path may be less direct
- Plot 6 shows formation angle (should be roughly constant)

**Rotation Control ON (enhanced):**
- Formation actively rotates to align with Newton step direction
- Generally faster convergence
- More direct path to saddle
- Plot 6 shows formation tracking Newton direction (blue tracks red)
- Plot 7 shows rotation error decreasing over time

**Comparison Metrics:**
- **Final Distance**: How close to exact saddle point (0,0)
- **Final |∇z|**: Gradient magnitude at final position
- **Steps to Converge**: Iterations needed to reach dist < 0.1
- **Path Length**: Total distance traveled
- **Path Efficiency**: Straight-line distance / path length (1.0 = perfect)
- **Total Rotation**: How much formation rotated (in degrees)

### Expected Performance Differences

**WITH rotation control typically shows:**
- 10-30% faster convergence (fewer steps)
- 5-15% shorter path length
- 10-20% better path efficiency
- Smoother trajectory with fewer oscillations

**Trade-offs:**
- Rotation control adds complexity (omega_c calculation)
- May cause additional rotation that affects formation sampling
- Not always beneficial for non-saddle critical points

### Configuration

Both rotation methods use the same core parameters:

```python
# In saddle_experiment_*.py
SIM_TIME = 750-1000  # Steps (WITH may need fewer)
TIMESTEP = 0.1
MOMENTUM_ALPHA = 0.7
ROTATION_GAIN = 0.3  # Only for WITH rotation (controls aggressiveness)
```

**Tuning ROTATION_GAIN:**
- Lower (0.05-0.1): Gentle rotation, more stable
- Medium (0.2-0.3): Balanced rotation, recommended
- Higher (0.4-0.5): Aggressive rotation, may cause oscillation

## Files in this Directory

- `saddle_experiment.py` - Main experiment script (configurable rotation control)
- `saddle_experiment_NO_rotation.py` - Baseline Newton WITHOUT rotation
- `saddle_experiment_WITH_rotation.py` - Enhanced Newton WITH rotation
- `compare_rotation_methods.py` - Side-by-side comparison of both methods
- `README.md` - This file
- `results/` - Output directory (created automatically)
  - `saddle_finding_with_rotation.png` - Main experiment visualization
  - `saddle_NO_rotation.png` - Baseline method visualization
  - `saddle_WITH_rotation.png` - Enhanced method visualization
  - `rotation_comparison.png` - Side-by-side comparison plots
  - `diagnostics.txt` - Iteration history and final metrics

## Related Documentation

- `../CLAUDE.md` - Full VF_Robot documentation
- `../src/control/scalar_quad_primitives.py` - Newton's method implementation
- `../src/fields/scalar_environments/Saddle.py` - Saddle field definitions
