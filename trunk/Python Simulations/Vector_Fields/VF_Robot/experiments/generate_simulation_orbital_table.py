#!/usr/bin/env python3
"""
generate_simulation_orbital_table.py - Generate Simulation Orbital Control Data

Runs analytical (no noise) simulations for orbital control at various radii
across all 6 canonical field types to create comparison table for the paper.

Tests:
- 6 field types: sink, source, vortex, saddle, sinking_vortex, spewing_vortex
- 2 robot configs: 3-robot and 4-robot-square
- 8 radii: 0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 m

Output: simulation_orbital_results.csv with columns:
  - field_type: sink, source, vortex, saddle, sinking_vortex, spewing_vortex
  - robot_type: 3-robot, 4-robot-square
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

# Import robot and field components
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
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

# Test radii for both configurations (0.01 to 0.70 m)
TEST_RADII = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Field configurations
FIELD_CONFIGS = {
    'sink': sink1,
    'source': source1,
    'vortex': vortex1,
    'saddle': saddle1,
    'sinking_vortex': sinking_vortex1,
    'spewing_vortex': spewing_vortex1,
}

# Formation configuration files
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")

# Output file
OUTPUT_FILE = os.path.join(script_dir, "simulation_orbital_results.csv")

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_orbital_simulation(cluster_class, formation_config: str, control_func,
                          field_func, desired_radius: float, verbose: bool = False) -> Dict:
    """
    Run a single orbital control simulation with analytical field.

    Args:
        cluster_class: OmniCluster or QuadCluster
        formation_config: Path to YAML formation config
        control_func: Control primitive function
        field_func: Field function (e.g., vortex1, sink1)
        desired_radius: Target orbital radius
        verbose: Print progress messages

    Returns:
        Dictionary with simulation results
    """
    # Create analytical field (no noise)
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

    # Total combinations
    total_tests = len(FIELD_CONFIGS) * 2 * len(TEST_RADII)
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

        # =====================================================================
        # 4-ROBOT SQUARE CONFIGURATION
        # =====================================================================
        print(f"\n  4-Robot Square Configuration")
        print(f"  {'-' * 50}")

        for radius in TEST_RADII:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] Testing radius {radius:.2f} m...")

            results = run_orbital_simulation(
                QuadCluster,
                FORMATION_4ROBOT_SQUARE,
                qcp.center_orbiter_quad,
                field_func,
                radius,
                verbose=False
            )

            all_results.append({
                'field_type': field_name,
                'robot_type': '4-robot-square',
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
    print("Summary by Robot Type:")
    print("-" * 50)

    for robot_type in ['3-robot', '4-robot-square']:
        subset = results_df[results_df['robot_type'] == robot_type]
        if len(subset) > 0:
            mean_error = subset['radius_error'].mean()
            std_error = subset['radius_error'].std()
            max_std = subset['std_radius'].max()
            print(f"  {robot_type:20s}: mean error = {mean_error:+.4f} ± {std_error:.4f} m, max std = {max_std:.6f} m")

    # Check if results are identical across fields
    print()
    print("Checking consistency across fields:")
    print("-" * 50)

    for robot_type in ['3-robot', '4-robot-square']:
        for radius in TEST_RADII:
            # Get all results for this robot_type and radius across all fields
            matching = results_df[(results_df['robot_type'] == robot_type) &
                                 (results_df['commanded_radius'] == radius)]

            if len(matching) > 0:
                actual_radii = matching['actual_radius'].values
                if np.std(actual_radii) > 1e-6:  # If variation exists
                    print(f"  {robot_type} @ {radius:.2f}m: DIFFERENT across fields (std={np.std(actual_radii):.6f})")
                    print(f"    Values: {actual_radii}")

    # If no "DIFFERENT" messages printed above, all are identical
    print()
    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated {len(all_results)} data points")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Next step: Run create_simulation_orbital_table.py to generate LaTeX table")


if __name__ == "__main__":
    main()
