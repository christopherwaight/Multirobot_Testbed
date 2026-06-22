#!/usr/bin/env python3
"""
generate_analytical_orbit_data.py - Generate Analytical Orbital Control Data

Runs analytical (no noise) simulations for orbital control at various radii
to compare with real robot data and ML-based approximations.

Generates data for Table 4 comparison in the paper.

Output: analytical_orbit_results.csv with columns:
  - config_name: orbit001, orbit010, etc.
  - robot_type: 3-robot, 4-robot-square, 4-robot-dual
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
from src.fields.environments.Vortex import vortex1

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
STABLE_START = 100  # Skip first 100 timesteps for statistics (transient)

# Test radii for each configuration
RADII_3ROBOT = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
RADII_4ROBOT_SQUARE = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
RADII_4ROBOT_DUAL = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50]

# Formation configuration files
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# Output file
OUTPUT_FILE = os.path.join(script_dir, "analytical_orbit_results.csv")

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_orbital_simulation(cluster_class, formation_config: str, control_func,
                          desired_radius: float, verbose: bool = False) -> Dict:
    """
    Run a single orbital control simulation with analytical field.

    Args:
        cluster_class: OmniCluster or QuadCluster
        formation_config: Path to YAML formation config
        control_func: Control primitive function
        desired_radius: Target orbital radius
        verbose: Print progress messages

    Returns:
        Dictionary with simulation results
    """
    # Create analytical field (no noise)
    field = AnalyticalField(vortex1)

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
        'trajectory': trajectory  # Keep full trajectory for potential plotting
    }

def generate_config_name(robot_type: str, radius: float) -> str:
    """
    Generate configuration name in orbit### format.

    Args:
        robot_type: '3-robot', '4-robot-square', or '4-robot-dual'
        radius: Target radius in meters

    Returns:
        Config name like 'orbit040', 'orbit140', 'orbit240'
    """
    # Convert radius to integer (multiply by 100)
    radius_int = int(radius * 100)

    # Add series offset
    if robot_type == '3-robot':
        config_num = radius_int  # 0XX series
    elif robot_type == '4-robot-square':
        config_num = 100 + radius_int  # 1XX series
    else:  # 4-robot-dual
        config_num = 200 + radius_int  # 2XX series

    return f"orbit{config_num:03d}"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("ANALYTICAL ORBITAL CONTROL DATA GENERATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Environment: vortex1 (analytical, no noise)")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s ({SIM_TIME} steps)")
    print(f"  Starting position: (0, radius) for each test")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 70)
    print()

    # Storage for all results
    all_results = []

    # =========================================================================
    # 3-ROBOT CONFIGURATION
    # =========================================================================
    print("3-Robot Configuration")
    print("-" * 50)

    for radius in RADII_3ROBOT:
        print(f"  Testing radius {radius:.2f} m...")

        results = run_orbital_simulation(
            OmniCluster,
            FORMATION_3ROBOT,
            ocp.critical_point_orbiter_plane_fitting,
            radius,
            verbose=False
        )

        config_name = generate_config_name('3-robot', radius)

        all_results.append({
            'config_name': config_name,
            'robot_type': '3-robot',
            'commanded_radius': radius,
            'actual_radius': results['actual_radius'],
            'radius_error': results['radius_error'],
            'std_radius': results['std_radius']
        })

        print(f"    ✓ {config_name}: actual={results['actual_radius']:.4f}m, "
              f"error={results['radius_error']:+.4f}m, std={results['std_radius']:.4f}m")

    print()

    # =========================================================================
    # 4-ROBOT SQUARE CONFIGURATION
    # =========================================================================
    print("4-Robot Square Configuration")
    print("-" * 50)

    for radius in RADII_4ROBOT_SQUARE:
        print(f"  Testing radius {radius:.2f} m...")

        results = run_orbital_simulation(
            QuadCluster,
            FORMATION_4ROBOT_SQUARE,
            qcp.center_orbiter_quad,
            radius,
            verbose=False
        )

        config_name = generate_config_name('4-robot-square', radius)

        all_results.append({
            'config_name': config_name,
            'robot_type': '4-robot-square',
            'commanded_radius': radius,
            'actual_radius': results['actual_radius'],
            'radius_error': results['radius_error'],
            'std_radius': results['std_radius']
        })

        print(f"    ✓ {config_name}: actual={results['actual_radius']:.4f}m, "
              f"error={results['radius_error']:+.4f}m, std={results['std_radius']:.4f}m")

    print()

    # =========================================================================
    # 4-ROBOT DUAL JACOBIAN CONFIGURATION
    # =========================================================================
    print("4-Robot Dual Jacobian Configuration")
    print("-" * 50)

    # Note: Using the advanced configuration but with the specialized control
    for radius in RADII_4ROBOT_DUAL:
        print(f"  Testing radius {radius:.2f} m...")

        # For dual Jacobian, we use the advanced formation
        results = run_orbital_simulation(
            QuadCluster,
            FORMATION_4ROBOT_ADVANCED,
            qcp.center_orbiter_quad_advanced,
            radius,
            verbose=False
        )

        config_name = generate_config_name('4-robot-dual', radius)

        all_results.append({
            'config_name': config_name,
            'robot_type': '4-robot-dual',
            'commanded_radius': radius,
            'actual_radius': results['actual_radius'],
            'radius_error': results['radius_error'],
            'std_radius': results['std_radius']
        })

        print(f"    ✓ {config_name}: actual={results['actual_radius']:.4f}m, "
              f"error={results['radius_error']:+.4f}m, std={results['std_radius']:.4f}m")

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
    print("Summary by Configuration:")
    print("-" * 50)

    for robot_type in ['3-robot', '4-robot-square', '4-robot-dual']:
        subset = results_df[results_df['robot_type'] == robot_type]
        if len(subset) > 0:
            mean_error = subset['radius_error'].mean()
            std_error = subset['radius_error'].std()
            print(f"  {robot_type:20s}: mean error = {mean_error:+.4f} ± {std_error:.4f} m")

    print()
    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated {len(all_results)} data points")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Next step: Run analyze_table4_v3_with_analytical.py to create plots")


if __name__ == "__main__":
    main()