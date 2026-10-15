#!/usr/bin/env python3
"""
test_momentum_velocity_effects_fixed.py - FIXED Test of Momentum and Velocity Effects

Previous version had monkey-patching issues. This version directly modifies
robot instances after creation to ensure parameters are actually changed.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.gridspec as gridspec

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

# Quick test parameters
SIM_TIME = 200  # timesteps (20 seconds)
TIMESTEP = 0.1  # seconds
STABLE_START = 50  # Skip first 50 timesteps for statistics

# Test radii - key points
RADII_TO_TEST = [0.01, 0.1, 0.3, 0.5]

# Parameter combinations to test
PARAMETER_SETS = [
    {'name': 'Baseline', 'momentum_alpha': 0.7, 'max_velocity': 0.3},
    {'name': 'No_Momentum', 'momentum_alpha': 0.0, 'max_velocity': 0.3},
    {'name': 'Low_Momentum', 'momentum_alpha': 0.3, 'max_velocity': 0.3},
    {'name': 'No_Vel_Limit', 'momentum_alpha': 0.7, 'max_velocity': 10.0},
    {'name': 'Neither', 'momentum_alpha': 0.0, 'max_velocity': 10.0},
]

# Formation configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# Output
OUTPUT_DIR = os.path.join(script_dir, "momentum_velocity_analysis_fixed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DIRECT PARAMETER MODIFICATION (No Monkey-Patching)
# ============================================================================

def modify_robot_parameters(cluster, momentum_alpha: float, max_velocity: float):
    """
    Directly modify robot parameters in the cluster.
    This avoids monkey-patching issues.
    """
    # Modify each robot in the cluster
    for robot in cluster.robots:
        # Set momentum alpha
        robot.momentum_alpha = momentum_alpha

        # Store the max velocity for our custom command_velocity
        robot.custom_max_velocity = max_velocity

        # Save original command_velocity method
        robot.original_command_velocity = robot.command_velocity

        # Create new command_velocity method with custom max
        def make_custom_command_velocity(robot_instance, max_vel):
            def custom_command_velocity(vx_cmd, vy_cmd):
                # Update velocity using momentum
                velocity_cmd = np.array([vx_cmd, vy_cmd])
                robot_instance.velocity = robot_instance.momentum_alpha * robot_instance.velocity + \
                                         (1 - robot_instance.momentum_alpha) * velocity_cmd

                # Apply custom velocity limit
                velocity_magnitude = np.linalg.norm(robot_instance.velocity)
                if velocity_magnitude > max_vel:
                    robot_instance.velocity = (robot_instance.velocity / velocity_magnitude) * max_vel

                # Apply stiction
                STICTION_THRESHOLD = 0.025
                if velocity_magnitude < STICTION_THRESHOLD:
                    robot_instance.velocity = np.array([0.0, 0.0])

                # Update position
                robot_instance.position += robot_instance.timestep * robot_instance.velocity

            return custom_command_velocity

        # Replace command_velocity with our custom version
        robot.command_velocity = make_custom_command_velocity(robot, max_velocity)

def restore_robot_parameters(cluster):
    """
    Restore original robot parameters.
    """
    for robot in cluster.robots:
        if hasattr(robot, 'original_command_velocity'):
            robot.command_velocity = robot.original_command_velocity
            robot.momentum_alpha = 0.7  # Default value

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_simulation_with_params(cluster_class, formation_config: str, control_func,
                               desired_radius: float, param_set: Dict) -> Dict:
    """
    Run simulation with specified parameters.
    """
    # Create field
    field = AnalyticalField(vortex1)

    # Create cluster
    cluster = cluster_class(formation_config, field)

    # Modify robot parameters AFTER cluster creation
    modify_robot_parameters(cluster, param_set['momentum_alpha'], param_set['max_velocity'])

    try:
        # Start at (0, radius)
        start_x = 0.0
        start_y = desired_radius
        cluster.reset(x_c=start_x, y_c=start_y)

        # Storage
        positions = []
        velocities = []
        commands = []

        # Run simulation
        for step in range(SIM_TIME):
            # Get centroid
            x_c, y_c = cluster.get_centroid()
            positions.append([x_c, y_c])

            # Track velocities
            robot_vels = [robot.get_velocity() for robot in cluster.robots]
            vel_mags = [np.linalg.norm(v) for v in robot_vels]
            velocities.append(vel_mags)

            # Get control command
            control_output = control_func(cluster, desired_radius=desired_radius)
            if len(control_output) == 2:
                vx_c, vy_c = control_output
            else:
                vx_c, vy_c, omega_c = control_output

            commands.append([vx_c, vy_c])

            # Move cluster
            cluster.move(lambda c: control_func(c, desired_radius=desired_radius))

        # Analyze results
        positions = np.array(positions)
        velocities = np.array(velocities)
        commands = np.array(commands)

        # Skip transient
        stable_positions = positions[STABLE_START:]
        stable_velocities = velocities[STABLE_START:]

        # Calculate metrics
        radii = np.sqrt(stable_positions[:, 0]**2 + stable_positions[:, 1]**2)
        mean_radius = np.mean(radii)
        std_radius = np.std(radii)
        radius_error = mean_radius - desired_radius

        # Velocity metrics
        max_commanded = np.max(np.sqrt(commands[:, 0]**2 + commands[:, 1]**2))
        mean_velocity = np.mean(stable_velocities)
        max_velocity_achieved = np.max(stable_velocities)

        # Count saturations
        saturation_count = 0
        for vel_set in stable_velocities:
            if np.any(vel_set >= param_set['max_velocity'] * 0.95):
                saturation_count += 1

        saturation_rate = saturation_count / len(stable_velocities) if len(stable_velocities) > 0 else 0

        return {
            'mean_radius': mean_radius,
            'std_radius': std_radius,
            'radius_error': radius_error,
            'max_commanded_velocity': max_commanded,
            'mean_velocity': mean_velocity,
            'max_velocity_achieved': max_velocity_achieved,
            'saturation_rate': saturation_rate
        }

    finally:
        # Restore original parameters
        restore_robot_parameters(cluster)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("MOMENTUM AND VELOCITY EFFECTS ANALYSIS (FIXED)")
    print("=" * 70)
    print(f"Parameter sets to test:")
    for pset in PARAMETER_SETS:
        print(f"  {pset['name']:15s}: α={pset['momentum_alpha']:.1f}, v_max={pset['max_velocity']:.1f} m/s")
    print(f"Radii: {RADII_TO_TEST}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    print()

    # Storage
    all_results = {}
    csv_data = []

    # Test configurations
    configurations = [
        ('3-robot', OmniCluster, FORMATION_3ROBOT, ocp.critical_point_orbiter_plane_fitting),
        ('4-robot-square', QuadCluster, FORMATION_4ROBOT_SQUARE, qcp.center_orbiter_quad),
        ('4-robot-dual', QuadCluster, FORMATION_4ROBOT_ADVANCED, qcp.center_orbiter_quad_advanced)
    ]

    # Run all experiments
    for config_name, cluster_class, formation_config, control_func in configurations:
        print(f"\n{config_name} Configuration")
        print("-" * 50)

        for param_set in PARAMETER_SETS:
            print(f"\n  {param_set['name']:15s} (α={param_set['momentum_alpha']:.1f}, v_max={param_set['max_velocity']:.1f})")

            for radius in RADII_TO_TEST:
                print(f"    r={radius:.2f}m: ", end='', flush=True)

                try:
                    # Run simulation
                    results = run_simulation_with_params(
                        cluster_class,
                        formation_config,
                        control_func,
                        radius,
                        param_set
                    )

                    # Store results
                    key = (config_name, radius, param_set['name'])
                    all_results[key] = results

                    # Add to CSV data
                    csv_row = {
                        'configuration': config_name,
                        'radius': radius,
                        'param_set': param_set['name'],
                        'momentum_alpha': param_set['momentum_alpha'],
                        'max_velocity': param_set['max_velocity'],
                        **results
                    }
                    csv_data.append(csv_row)

                    print(f"error={results['radius_error']:+.4f}m, "
                          f"v_mean={results['mean_velocity']:.2f}m/s, "
                          f"sat={results['saturation_rate']*100:.0f}%")

                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Save CSV results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    csv_file = os.path.join(OUTPUT_DIR, 'momentum_velocity_results_fixed.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"Saved CSV: {csv_file}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Check if parameters actually had an effect
    print("\n1. PARAMETER EFFECTS CHECK:")
    print("-" * 40)

    for config in ['3-robot', '4-robot-square', '4-robot-dual']:
        print(f"\n{config} at r=0.1m:")

        # Compare baseline vs no momentum
        baseline_key = (config, 0.1, 'Baseline')
        no_momentum_key = (config, 0.1, 'No_Momentum')
        no_limit_key = (config, 0.1, 'No_Vel_Limit')

        if baseline_key in all_results and no_momentum_key in all_results:
            baseline = all_results[baseline_key]
            no_momentum = all_results[no_momentum_key]
            print(f"  Baseline:     error={baseline['radius_error']:+.4f}m, v={baseline['mean_velocity']:.2f}m/s")
            print(f"  No Momentum:  error={no_momentum['radius_error']:+.4f}m, v={no_momentum['mean_velocity']:.2f}m/s")

            if abs(baseline['radius_error'] - no_momentum['radius_error']) < 0.001:
                print(f"  → Momentum has NO effect")
            else:
                print(f"  → Momentum DOES affect performance")

        if baseline_key in all_results and no_limit_key in all_results:
            baseline = all_results[baseline_key]
            no_limit = all_results[no_limit_key]
            print(f"  No Vel Limit: error={no_limit['radius_error']:+.4f}m, v={no_limit['mean_velocity']:.2f}m/s")

            if no_limit['mean_velocity'] > 0.35:
                print(f"  → Velocity limit WAS restricting performance")
            else:
                print(f"  → Velocity limit was NOT a constraint")

    print("\n2. CROSS-CONFIGURATION COMPARISON AT r=0.1m:")
    print("-" * 40)

    for param_name in ['Baseline', 'No_Vel_Limit']:
        print(f"\n{param_name}:")
        for config in ['3-robot', '4-robot-square', '4-robot-dual']:
            key = (config, 0.1, param_name)
            if key in all_results:
                r = all_results[key]
                print(f"  {config:15s}: error={r['radius_error']:+.4f}m, v={r['mean_velocity']:.2f}m/s")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()