#!/usr/bin/env python3
"""
generate_simulation_orbital_table_8_nn.py - Generate Simulation Orbital Control Data (NN VERSION)

Runs analytical (no noise) simulations for orbital control at various radii
across 8 field types (6 analytical + 2 NN) to create comparison table for the paper.

Tests:
- 8 field types: sink, source, vortex, saddle, sinking_vortex, spewing_vortex, vortex_nn, saddle_nn
- 1 robot config: 3-robot only
- 8 radii: 0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 m

Output: simulation_orbital_results_8fields_nn.csv with columns:
  - field_type: sink, source, vortex, saddle, sinking_vortex, spewing_vortex, vortex_nn, saddle_nn
  - robot_type: 3-robot
  - commanded_radius: target radius in meters
  - actual_radius: mean achieved radius
  - radius_error: actual - commanded
  - std_radius: standard deviation of radius
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Set random seed for repeatability
np.random.seed(42)

# Import robot and field components
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField, NNField
import src.control.primitives as ocp
import src.control.quad_primitives as qcp

# Import all 6 canonical fields
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Saddle import saddle1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Spewing_Vortex import spewing_vortex1

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
STABLE_START = 100  # Skip first 100 timesteps for statistics (transient)

# Test radii (0.01 to 0.70 m)
TEST_RADII = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Get absolute paths to predictor directories for RBF fields
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
vortex_predictor_dir = os.path.join(project_root, 'vortex_predictors')
saddle_predictor_dir = os.path.join(project_root, 'saddle_predictors')

# Field configurations (8 fields: 6 analytical + 2 NN)
FIELD_CONFIGS = {
    'sink': sink1,
    'source': source1,
    'vortex': vortex1,
    'saddle': saddle1,
    'sinking_vortex': sinking_vortex1,
    'spewing_vortex': spewing_vortex1,
    'vortex_nn': vortex_predictor_dir,  # NN approximation
    'saddle_nn': saddle_predictor_dir,  # NN approximation
}

# Formation configuration files (3-robot only)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")

# Output file
OUTPUT_FILE = os.path.join(script_dir, "simulation_orbital_results_8fields_nn.csv")

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_orbital_simulation(cluster_class, formation_config: str, control_func,
                          field_func, desired_radius: float, verbose: bool = False) -> Dict:
    """
    Run a single orbital control simulation with analytical or RBF field.

    Args:
        cluster_class: OmniCluster or QuadCluster
        formation_config: Path to YAML formation config
        control_func: Control primitive function
        field_func: Field function (callable) or RBF predictor directory (string)
        desired_radius: Target orbital radius
        verbose: Print progress messages

    Returns:
        Dictionary with simulation results
    """
    # Create field (analytical or RBF)
    if isinstance(field_func, str):
        # RBF field - field_func is a directory path
        field = NNField(field_func)
    else:
        # Analytical field - field_func is a callable
        field = AnalyticalField(field_func)

    # Create cluster
    cluster = cluster_class(formation_config, field)

    # Start at (0, radius) - on positive y-axis at desired radius
    start_x = 0.0
    start_y = desired_radius
    cluster.reset(x_c=start_x, y_c=start_y)

    # Storage for trajectory
    trajectory = []

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        trajectory.append([x_c, y_c])

        # Move cluster using control primitive
        cluster.move(lambda c: control_func(c, desired_radius=desired_radius))

        # Progress indicator
        if verbose and step % 100 == 0:
            r = np.sqrt(x_c**2 + y_c**2)
            print(f"    Step {step:3d}: r = {r:.4f} m (target: {desired_radius:.4f} m)")

    # Convert to numpy array
    trajectory = np.array(trajectory)

    # Calculate statistics (skip transient period)
    stable_trajectory = trajectory[STABLE_START:] if len(trajectory) > STABLE_START else trajectory
    distances = np.sqrt(stable_trajectory[:, 0]**2 + stable_trajectory[:, 1]**2)

    actual_radius = np.mean(distances)
    std_radius = np.std(distances)
    radius_error = actual_radius - desired_radius

    return {
        'actual_radius': actual_radius,
        'radius_error': radius_error,
        'std_radius': std_radius,
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("SIMULATION ORBITAL CONTROL DATA GENERATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Fields: {', '.join(FIELD_CONFIGS.keys())}")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s ({SIM_TIME} steps)")
    print(f"  Test radii: {TEST_RADII}")
    print(f"  Starting position: (0, radius) for each test")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 70)
    print()

    # Storage for all results
    all_results = []

    # Total combinations (8 fields * 1 robot config * 8 radii = 64 tests)
    total_tests = len(FIELD_CONFIGS) * len(TEST_RADII)
    current_test = 0

    # =========================================================================
    # RUN ALL COMBINATIONS
    # =========================================================================
    for field_name, field_func in FIELD_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"FIELD: {field_name.upper()}")
        print(f"{'=' * 70}")

        # =====================================================================
        # 3-ROBOT CONFIGURATION
        # =====================================================================
        print(f"\n  3-Robot Configuration")
        print(f"  {'-' * 50}")

        for radius in TEST_RADII:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] Testing radius {radius:.2f} m...")

            results = run_orbital_simulation(
                OmniCluster,
                FORMATION_3ROBOT,
                ocp.critical_point_orbiter_plane_fitting,
                field_func,
                radius,
                verbose=False
            )

            all_results.append({
                'field_type': field_name,
                'robot_type': '3-robot',
                'commanded_radius': radius,
                'actual_radius': results['actual_radius'],
                'radius_error': results['radius_error'],
                'std_radius': results['std_radius']
            })

            print(f"      ✓ actual={results['actual_radius']:.4f}m, "
                  f"error={results['radius_error']:+.4f}m, std={results['std_radius']:.6f}m")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')
    print(f"✓ Results saved to: {OUTPUT_FILE}")

    # Print summary statistics
    print()
    print("Summary (3-robot only):")
    print("-" * 50)

    subset = results_df[results_df['robot_type'] == '3-robot']
    mean_error = subset['radius_error'].mean()
    std_error = subset['radius_error'].std()
    max_std = subset['std_radius'].max()
    print(f"  3-robot: mean error = {mean_error:+.4f} ± {std_error:.4f} m, max std = {max_std:.6f} m")

    # Check if results vary across fields (analytical vs RBF)
    print()
    print("Checking variation across fields:")
    print("-" * 50)

    for radius in TEST_RADII:
        # Get all results for 3-robot at this radius across all 8 fields
        matching = results_df[(results_df['robot_type'] == '3-robot') &
                             (results_df['commanded_radius'] == radius)]

        if len(matching) > 0:
            actual_radii = matching['actual_radius'].values
            field_std = np.std(actual_radii)
            if field_std > 1e-6:  # If variation exists between analytical and RBF
                print(f"  radius {radius:.2f}m: variation across fields (std={field_std:.6f})")
                for _, row in matching.iterrows():
                    print(f"    {row['field_type']:20s}: {row['actual_radius']:.6f} m")

    # If no "DIFFERENT" messages printed above, all are identical
    print()
    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated {len(all_results)} data points (8 fields × 8 radii = 64 tests)")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Next step: Run create_simulation_orbital_table_8.py to generate LaTeX table")


if __name__ == "__main__":
    main()
