#!/usr/bin/env python3
"""
test_momentum_velocity_effects.py - Test Impact of Momentum and Velocity Limits

This experiment tests whether poor 4-robot dual Jacobian performance is due to:
1. Momentum effects (alpha parameter)
2. Velocity saturation (MAX_VELOCITY limit)
3. Fundamental control algorithm issues

Uses safe monkey-patching to temporarily modify parameters without changing core files.
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

# Import the modules we'll monkey-patch
import src.robot.omnibot as omnibot_module

# ============================================================================
# CONFIGURATION
# ============================================================================

# Quick test parameters (shorter than full analysis)
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
OUTPUT_DIR = os.path.join(script_dir, "momentum_velocity_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MONKEY-PATCHING FUNCTIONS
# ============================================================================

def safe_modify_robot_parameters(momentum_alpha: float, max_velocity: float):
    """
    Safely modify robot parameters using monkey-patching.
    Returns a function to restore original values.
    """
    # Save original MAX_VELOCITY from Omnibot class
    original_max_velocity = 0.3  # Default value in omnibot.py

    # Save original momentum_alpha default
    # This is trickier - we need to modify the __init__ method
    original_init = omnibot_module.Omnibot.__init__

    # Create modified __init__ that uses our momentum_alpha
    def modified_init(self, x=0.0, y=0.0, timestep=0.1, momentum_alpha_override=momentum_alpha):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha_override  # Use our value
        self.field = None

    # Apply modifications
    omnibot_module.Omnibot.__init__ = modified_init

    # Modify MAX_VELOCITY in command_velocity method
    original_command_velocity = omnibot_module.Omnibot.command_velocity

    def modified_command_velocity(self, vx_cmd, vy_cmd):
        """Modified command_velocity with custom MAX_VELOCITY"""
        # Update velocity using momentum
        velocity_cmd = np.array([vx_cmd, vy_cmd])
        self.velocity = self.momentum_alpha * self.velocity + (1 - self.momentum_alpha) * velocity_cmd

        # Apply custom velocity limit
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > max_velocity:
            self.velocity = (self.velocity / velocity_magnitude) * max_velocity

        # Apply stiction
        STICTION_THRESHOLD = 0.025
        if velocity_magnitude < STICTION_THRESHOLD:
            self.velocity = np.array([0.0, 0.0])

        # Update position
        self.position += self.timestep * self.velocity

    omnibot_module.Omnibot.command_velocity = modified_command_velocity

    # Return restoration function
    def restore():
        omnibot_module.Omnibot.__init__ = original_init
        omnibot_module.Omnibot.command_velocity = original_command_velocity

    return restore

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_quick_simulation(cluster_class, formation_config: str, control_func,
                         desired_radius: float, param_set: Dict) -> Dict:
    """
    Run a quick simulation with specified parameters.
    """
    # Apply parameter modifications
    restore_func = safe_modify_robot_parameters(
        param_set['momentum_alpha'],
        param_set['max_velocity']
    )

    try:
        # Create field
        field = AnalyticalField(vortex1)

        # Create cluster
        cluster = cluster_class(formation_config, field)

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
                omega_c = 0
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

        # Count saturations (when actual < commanded significantly)
        saturation_count = 0
        for i in range(len(commands)):
            cmd_mag = np.sqrt(commands[i, 0]**2 + commands[i, 1]**2)
            if cmd_mag > param_set['max_velocity'] * 0.95:
                saturation_count += 1

        saturation_rate = saturation_count / len(commands)

        return {
            'mean_radius': mean_radius,
            'std_radius': std_radius,
            'radius_error': radius_error,
            'max_commanded_velocity': max_commanded,
            'mean_velocity': mean_velocity,
            'max_velocity_achieved': max_velocity_achieved,
            'saturation_rate': saturation_rate,
            'positions': positions,
            'velocities': velocities,
            'commands': commands
        }

    finally:
        # Always restore original parameters
        restore_func()

# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

def create_comparison_plot(all_results: Dict, output_file: str):
    """
    Create comprehensive comparison plot.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)

    configs = ['3-robot', '4-robot-square', '4-robot-dual']
    radii = RADII_TO_TEST

    # Row 1: Radius Error for each configuration
    for col, config in enumerate(configs):
        ax = fig.add_subplot(gs[0, col])

        for param_set in PARAMETER_SETS:
            errors = []
            for radius in radii:
                key = (config, radius, param_set['name'])
                if key in all_results:
                    errors.append(all_results[key]['radius_error'])
                else:
                    errors.append(np.nan)

            ax.plot(radii, errors, marker='o', label=param_set['name'], linewidth=2)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Commanded Radius (m)')
        ax.set_ylabel('Radius Error (m)')
        ax.set_title(f'{config} - Radius Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Maximum Commanded Velocity
    for col, config in enumerate(configs):
        ax = fig.add_subplot(gs[1, col])

        for param_set in PARAMETER_SETS:
            max_cmds = []
            for radius in radii:
                key = (config, radius, param_set['name'])
                if key in all_results:
                    max_cmds.append(all_results[key]['max_commanded_velocity'])
                else:
                    max_cmds.append(np.nan)

            ax.plot(radii, max_cmds, marker='s', label=param_set['name'], linewidth=2)

        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Original Limit')
        ax.set_xlabel('Commanded Radius (m)')
        ax.set_ylabel('Max Commanded Velocity (m/s)')
        ax.set_title(f'{config} - What Algorithm Wants')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 2.0])  # Reasonable scale

    # Row 3: Saturation Rate
    for col, config in enumerate(configs):
        ax = fig.add_subplot(gs[2, col])

        for param_set in PARAMETER_SETS:
            sat_rates = []
            for radius in radii:
                key = (config, radius, param_set['name'])
                if key in all_results:
                    sat_rates.append(all_results[key]['saturation_rate'] * 100)
                else:
                    sat_rates.append(np.nan)

            ax.plot(radii, sat_rates, marker='^', label=param_set['name'], linewidth=2)

        ax.set_xlabel('Commanded Radius (m)')
        ax.set_ylabel('Saturation Rate (%)')
        ax.set_title(f'{config} - Velocity Saturation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    # Row 4: Performance Summary (absolute error)
    for col, config in enumerate(configs):
        ax = fig.add_subplot(gs[3, col])

        # Bar chart showing absolute error for each param set at r=0.1
        param_names = []
        abs_errors = []
        colors = []

        for param_set in PARAMETER_SETS:
            key = (config, 0.1, param_set['name'])
            if key in all_results:
                param_names.append(param_set['name'].replace('_', '\n'))
                abs_errors.append(abs(all_results[key]['radius_error']))

                # Color based on performance
                if abs(all_results[key]['radius_error']) < 0.05:
                    colors.append('green')
                elif abs(all_results[key]['radius_error']) < 0.1:
                    colors.append('orange')
                else:
                    colors.append('red')

        bars = ax.bar(param_names, abs_errors, color=colors, alpha=0.7)
        ax.set_ylabel('Absolute Error at r=0.1m (m)')
        ax.set_title(f'{config} - Performance at r=0.1m')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, abs_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Overall title
    fig.suptitle('Momentum and Velocity Limit Effects on Orbital Control',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_file}")

    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("MOMENTUM AND VELOCITY EFFECTS ANALYSIS")
    print("=" * 70)
    print(f"Parameter sets to test:")
    for pset in PARAMETER_SETS:
        print(f"  {pset['name']:15s}: α={pset['momentum_alpha']:.1f}, v_max={pset['max_velocity']:.1f} m/s")
    print(f"Radii: {RADII_TO_TEST}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    print()

    # Storage for all results
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
            print(f"\n  Parameter Set: {param_set['name']}")
            print(f"    α={param_set['momentum_alpha']}, v_max={param_set['max_velocity']}")

            for radius in RADII_TO_TEST:
                print(f"      r={radius:.2f}m: ", end='', flush=True)

                try:
                    # Run simulation
                    results = run_quick_simulation(
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
                        **{k: v for k, v in results.items() if k not in ['positions', 'velocities', 'commands']}
                    }
                    csv_data.append(csv_row)

                    print(f"error={results['radius_error']:+.4f}m, sat={results['saturation_rate']*100:.1f}%")

                except Exception as e:
                    print(f"FAILED: {e}")

    # Save CSV results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    csv_file = os.path.join(OUTPUT_DIR, 'momentum_velocity_results.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"Saved CSV: {csv_file}")

    # Create comparison plot
    plot_file = os.path.join(OUTPUT_DIR, 'momentum_velocity_comparison.png')
    create_comparison_plot(all_results, plot_file)

    # Analysis and insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Analyze 4-robot dual at r=0.1 (where it fails badly)
    test_radius = 0.1
    print(f"\n4-Robot Dual Jacobian at r={test_radius}m:")
    print("-" * 50)

    for param_set in PARAMETER_SETS:
        key = ('4-robot-dual', test_radius, param_set['name'])
        if key in all_results:
            r = all_results[key]
            print(f"{param_set['name']:15s}: error={r['radius_error']:+.4f}m, "
                  f"max_cmd={r['max_commanded_velocity']:.2f}m/s, "
                  f"sat={r['saturation_rate']*100:.1f}%")

    # Check which parameter helps most
    baseline_key = ('4-robot-dual', test_radius, 'Baseline')
    if baseline_key in all_results:
        baseline_error = abs(all_results[baseline_key]['radius_error'])

        print(f"\nImprovement over baseline (smaller is better):")
        for param_set in PARAMETER_SETS:
            if param_set['name'] == 'Baseline':
                continue
            key = ('4-robot-dual', test_radius, param_set['name'])
            if key in all_results:
                new_error = abs(all_results[key]['radius_error'])
                improvement = (baseline_error - new_error) / baseline_error * 100
                if improvement > 0:
                    print(f"  {param_set['name']:15s}: {improvement:.1f}% better")
                else:
                    print(f"  {param_set['name']:15s}: {-improvement:.1f}% worse")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    # Determine what's limiting performance
    no_limit_key = ('4-robot-dual', test_radius, 'No_Vel_Limit')
    no_momentum_key = ('4-robot-dual', test_radius, 'No_Momentum')
    neither_key = ('4-robot-dual', test_radius, 'Neither')

    conclusions = []

    if no_limit_key in all_results and abs(all_results[no_limit_key]['radius_error']) < 0.05:
        conclusions.append("✓ Velocity saturation IS the primary problem")
    else:
        conclusions.append("✗ Velocity saturation is NOT the only problem")

    if no_momentum_key in all_results and abs(all_results[no_momentum_key]['radius_error']) < 0.05:
        conclusions.append("✓ Momentum filtering causes instability")
    else:
        conclusions.append("✗ Momentum is NOT the main issue")

    if neither_key in all_results and abs(all_results[neither_key]['radius_error']) < 0.05:
        conclusions.append("✓ Algorithm works when unconstrained")
    elif neither_key in all_results:
        conclusions.append("✗ Algorithm has fundamental issues even when unconstrained")

    for conclusion in conclusions:
        print(f"  {conclusion}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()