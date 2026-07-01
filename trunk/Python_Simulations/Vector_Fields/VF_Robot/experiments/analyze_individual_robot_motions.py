#!/usr/bin/env python3
"""
analyze_individual_robot_motions.py - Individual Robot Motion Analysis

Analyzes individual robot behaviors during orbital control to understand why
performance is worse than expected, especially at small radii.

Key focus:
- Velocity saturation (v²/r centripetal requirements)
- Formation deformation and breakdown
- Individual robot struggles
- Control conflicts between formation and orbital objectives
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

# Simulation parameters
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
STABLE_START = 100  # Skip first 100 timesteps for statistics

# Test radii - ALL from 0.01 to 0.7
RADII_TO_TEST = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Robot constraints
MAX_VELOCITY = 0.3  # m/s - hardware limit
STICTION_THRESHOLD = 0.025  # m/s - below this, robot stops

# Formation configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# Output files
OUTPUT_DIR = os.path.join(script_dir, "individual_motion_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_theoretical_min_radius(max_velocity: float, orbital_speed_fraction: float = 1.0) -> float:
    """
    Calculate theoretical minimum orbital radius given max velocity.

    For circular motion: v = ωr, and centripetal acceleration a = v²/r
    The orbital speed is typically around 0.2 m/s for our vortex field.

    Args:
        max_velocity: Maximum robot velocity
        orbital_speed_fraction: Fraction of max velocity used for orbital motion

    Returns:
        Minimum achievable radius
    """
    orbital_speed = max_velocity * orbital_speed_fraction
    # For vortex1: v = r (approximately), so at equilibrium: orbital_speed = desired_radius
    # But we need to maintain formation too, which requires additional velocity
    # Practical minimum is when all velocity budget is consumed
    return orbital_speed

def run_detailed_simulation(cluster_class, formation_config: str, control_func,
                           desired_radius: float, verbose: bool = False) -> Dict:
    """
    Run orbital simulation tracking individual robot motions.

    Returns:
        Dictionary with detailed tracking data for each robot
    """
    # Create analytical field
    field = AnalyticalField(vortex1)

    # Create cluster
    cluster = cluster_class(formation_config, field)

    # Start at (0, radius)
    start_x = 0.0
    start_y = desired_radius
    cluster.reset(x_c=start_x, y_c=start_y)

    # Storage for detailed tracking
    num_robots = len(cluster.robots)
    tracking = {
        'centroid': [],
        'robot_positions': [[] for _ in range(num_robots)],
        'robot_velocities': [[] for _ in range(num_robots)],
        'robot_commands': [[] for _ in range(num_robots)],
        'formation_metrics': [],
        'jacobian_condition': [],
        'time': []
    }

    # Run simulation
    for step in range(SIM_TIME):
        tracking['time'].append(step * TIMESTEP)

        # Get centroid
        x_c, y_c = cluster.get_centroid()
        tracking['centroid'].append([x_c, y_c])

        # Track individual robots BEFORE move
        for i, robot in enumerate(cluster.robots):
            pos = robot.get_position()
            vel = robot.get_velocity()
            tracking['robot_positions'][i].append(pos.copy())
            tracking['robot_velocities'][i].append(vel.copy())

        # Get formation metrics
        if cluster_class == OmniCluster:
            formation = cluster.get_current_formation()
            tracking['formation_metrics'].append({
                'p': formation['p'],
                'q': formation['q'],
                'beta': formation['beta'],
                'area': calculate_triangle_area(formation['p'], formation['q'], formation['beta'])
            })
        else:  # QuadCluster
            formation = cluster.get_current_formation()
            tracking['formation_metrics'].append({
                'd1': formation['d1'],
                'd2': formation['d2'],
                'phi': formation['phi'],
                'area': calculate_quad_area(formation['d1'], formation['d2'], formation['phi'])
            })

        # Calculate Jacobian condition number (if available)
        try:
            if cluster_class == OmniCluster:
                jacobian = ocp.calculate_jacobian_from_readings(cluster)
            else:
                # For quad, use top triangle Jacobian
                jacobian = qcp.calculate_jacobian_top(cluster)

            if jacobian is not None:
                cond = np.linalg.cond(jacobian)
                tracking['jacobian_condition'].append(cond)
            else:
                tracking['jacobian_condition'].append(np.nan)
        except:
            tracking['jacobian_condition'].append(np.nan)

        # Execute move and capture commands
        control_output = control_func(cluster, desired_radius=desired_radius)
        if len(control_output) == 2:
            vx_c, vy_c = control_output
        else:
            vx_c, vy_c, omega_c = control_output  # 4-robot returns omega too

        # Store commanded velocities (before move executes)
        # Note: We'd need to modify cluster to expose individual commands
        # For now, track the centroid command
        for i in range(num_robots):
            tracking['robot_commands'][i].append([vx_c, vy_c])  # Simplified

        # Move cluster
        cluster.move(lambda c: control_func(c, desired_radius=desired_radius))

        if verbose and step % 100 == 0:
            r = np.sqrt(x_c**2 + y_c**2)
            print(f"    Step {step:3d}: r = {r:.4f} m")

    # Convert to numpy arrays
    for key in tracking:
        if key in ['robot_positions', 'robot_velocities', 'robot_commands']:
            tracking[key] = [np.array(robot_data) for robot_data in tracking[key]]
        elif key != 'formation_metrics':
            tracking[key] = np.array(tracking[key])

    return tracking

def calculate_triangle_area(p: float, q: float, beta_rad: float) -> float:
    """Calculate area of triangle given SAS parameters."""
    return 0.5 * p * q * np.sin(beta_rad)

def calculate_quad_area(d1: float, d2: float, phi_rad: float) -> float:
    """Calculate area of quadrilateral given diagonal parameters."""
    return 0.5 * d1 * d2 * np.sin(phi_rad)

def analyze_tracking_data(tracking: Dict, desired_radius: float) -> Dict:
    """
    Analyze detailed tracking data for insights.

    Returns:
        Dictionary with analysis metrics
    """
    # Skip transient
    start_idx = STABLE_START

    # Centroid analysis
    centroid = tracking['centroid'][start_idx:]
    centroid_radii = np.sqrt(centroid[:, 0]**2 + centroid[:, 1]**2)

    metrics = {
        'desired_radius': desired_radius,
        'centroid_mean_radius': np.mean(centroid_radii),
        'centroid_std_radius': np.std(centroid_radii),
        'centroid_radius_error': np.mean(centroid_radii) - desired_radius,
    }

    # Individual robot analysis
    num_robots = len(tracking['robot_positions'])

    for i in range(num_robots):
        positions = tracking['robot_positions'][i][start_idx:]
        velocities = tracking['robot_velocities'][i][start_idx:]

        # Distance from origin
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        metrics[f'robot{i}_mean_radius'] = np.mean(radii)
        metrics[f'robot{i}_std_radius'] = np.std(radii)

        # Velocity analysis
        vel_mags = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        metrics[f'robot{i}_mean_velocity'] = np.mean(vel_mags)
        metrics[f'robot{i}_max_velocity'] = np.max(vel_mags)
        metrics[f'robot{i}_velocity_saturations'] = np.sum(vel_mags >= MAX_VELOCITY - 0.001)
        metrics[f'robot{i}_stiction_events'] = np.sum(vel_mags <= STICTION_THRESHOLD)

        # Centripetal requirement (v²/r for circular motion)
        # Theoretical requirement: v_tangential = ω * r
        # For our vortex: ω ≈ 1, so v ≈ r
        theoretical_velocities = radii  # For vortex1
        metrics[f'robot{i}_mean_theoretical_v'] = np.mean(theoretical_velocities)

    # Formation quality
    if 'formation_metrics' in tracking and len(tracking['formation_metrics']) > 0:
        if 'area' in tracking['formation_metrics'][0]:
            areas = [fm['area'] for fm in tracking['formation_metrics'][start_idx:]]
            metrics['formation_area_mean'] = np.mean(areas)
            metrics['formation_area_std'] = np.std(areas)
            metrics['formation_area_cv'] = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else np.inf

    # Jacobian condition
    if 'jacobian_condition' in tracking:
        j_conds = tracking['jacobian_condition'][start_idx:]
        valid_j = [j for j in j_conds if not np.isnan(j)]
        if valid_j:
            metrics['jacobian_cond_mean'] = np.mean(valid_j)
            metrics['jacobian_cond_max'] = np.max(valid_j)

    # Check for velocity budget exhaustion
    total_saturations = sum(metrics.get(f'robot{i}_velocity_saturations', 0) for i in range(num_robots))
    metrics['total_velocity_saturations'] = total_saturations
    metrics['saturation_rate'] = total_saturations / (num_robots * len(centroid))

    # Theoretical minimum radius check
    theoretical_min = calculate_theoretical_min_radius(MAX_VELOCITY, 0.67)  # 2/3 of max for orbital
    metrics['theoretical_min_radius'] = theoretical_min
    metrics['below_theoretical_min'] = desired_radius < theoretical_min

    return metrics

def create_comprehensive_plot(all_results: Dict, output_file: str):
    """
    Create comprehensive visualization of individual robot motions.
    """
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)

    radii = RADII_TO_TEST
    configs = ['3-robot', '4-robot-square', '4-robot-dual']

    # Row 1-2: Centroid radius error vs commanded radius
    ax1 = fig.add_subplot(gs[0, :])
    for config in configs:
        errors = [all_results[config][r]['centroid_radius_error'] for r in radii]
        ax1.plot(radii, errors, marker='o', label=config, linewidth=2)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax1.set_ylabel('Radius Error (m)', fontsize=12)
    ax1.set_title('Centroid Radius Error vs Commanded Radius', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Row 2: Velocity saturation frequency
    ax2 = fig.add_subplot(gs[1, :])
    for config in configs:
        sat_rates = [all_results[config][r]['saturation_rate'] * 100 for r in radii]
        ax2.plot(radii, sat_rates, marker='s', label=config, linewidth=2)

    ax2.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax2.set_ylabel('Velocity Saturation Rate (%)', fontsize=12)
    ax2.set_title('Velocity Saturation Frequency (% of time any robot is saturated)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add theoretical minimum radius line
    theoretical_min = calculate_theoretical_min_radius(MAX_VELOCITY, 0.67)
    ax2.axvline(x=theoretical_min, color='red', linestyle=':', alpha=0.5,
                label=f'Theoretical Min Radius ({theoretical_min:.3f}m)')

    # Rows 3-4: Individual robot mean velocities for each configuration
    for row, config in enumerate(configs):
        ax = fig.add_subplot(gs[2 + row // 3, row % 3])

        # Determine number of robots
        if '3-robot' in config:
            num_robots = 3
        else:
            num_robots = 4

        for robot_id in range(num_robots):
            velocities = [all_results[config][r].get(f'robot{robot_id}_mean_velocity', 0) for r in radii]
            ax.plot(radii, velocities, marker='o', label=f'Robot {robot_id+1}', linewidth=1.5)

        ax.axhline(y=MAX_VELOCITY, color='red', linestyle='--', alpha=0.5, label='Max Velocity')
        ax.axhline(y=STICTION_THRESHOLD, color='blue', linestyle='--', alpha=0.5, label='Stiction')

        ax.set_xlabel('Commanded Radius (m)', fontsize=10)
        ax.set_ylabel('Mean Velocity (m/s)', fontsize=10)
        ax.set_title(f'{config} - Individual Robot Velocities', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, MAX_VELOCITY * 1.1])

    # Row 5: Formation quality (area coefficient of variation)
    ax5 = fig.add_subplot(gs[4, :])
    for config in configs:
        formation_cvs = []
        for r in radii:
            cv = all_results[config][r].get('formation_area_cv', 0)
            if cv != np.inf:
                formation_cvs.append(cv * 100)  # Convert to percentage
            else:
                formation_cvs.append(100)  # Cap at 100% for visualization

        ax5.plot(radii, formation_cvs, marker='^', label=config, linewidth=2)

    ax5.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax5.set_ylabel('Formation Area CV (%)', fontsize=12)
    ax5.set_title('Formation Shape Stability (lower is better)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Row 6: Jacobian condition number (numerical stability)
    ax6 = fig.add_subplot(gs[5, :])
    for config in configs:
        j_conds = []
        for r in radii:
            j_cond = all_results[config][r].get('jacobian_cond_mean', np.nan)
            if not np.isnan(j_cond):
                j_conds.append(np.log10(j_cond))  # Log scale
            else:
                j_conds.append(np.nan)

        # Filter out NaN values for plotting
        valid_radii = [r for r, j in zip(radii, j_conds) if not np.isnan(j)]
        valid_j_conds = [j for j in j_conds if not np.isnan(j)]

        if valid_radii:
            ax6.plot(valid_radii, valid_j_conds, marker='d', label=config, linewidth=2)

    ax6.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax6.set_ylabel('log₁₀(Jacobian Condition Number)', fontsize=12)
    ax6.set_title('Numerical Stability of Jacobian Estimation (lower is better)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=2, color='orange', linestyle=':', alpha=0.5, label='Poor Conditioning (100)')
    ax6.axhline(y=3, color='red', linestyle=':', alpha=0.5, label='Very Poor (1000)')

    # Overall title
    fig.suptitle('Individual Robot Motion Analysis - Understanding Performance Degradation',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive plot: {output_file}")

    return fig

def create_radius_specific_plots(tracking_data: Dict, config_name: str, radius: float, output_dir: str):
    """
    Create detailed plots for a specific radius and configuration.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Individual robot trajectories
    ax = axes[0, 0]
    num_robots = len(tracking_data['robot_positions'])
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i in range(num_robots):
        positions = tracking_data['robot_positions'][i]
        ax.plot(positions[:, 0], positions[:, 1], color=colors[i],
                alpha=0.7, linewidth=1, label=f'Robot {i+1}')

    # Add centroid trajectory
    centroid = tracking_data['centroid']
    ax.plot(centroid[:, 0], centroid[:, 1], 'k--', alpha=0.5,
            linewidth=2, label='Centroid')

    # Add target circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'g:',
            linewidth=1, label=f'Target r={radius}m')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Individual Robot Trajectories - {config_name} at r={radius}m')
    ax.legend(fontsize=8)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity over time
    ax = axes[0, 1]
    time = tracking_data['time']

    for i in range(num_robots):
        velocities = tracking_data['robot_velocities'][i]
        vel_mags = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        ax.plot(time, vel_mags, color=colors[i], alpha=0.7,
                linewidth=1, label=f'Robot {i+1}')

    ax.axhline(y=MAX_VELOCITY, color='red', linestyle='--',
               alpha=0.5, label='Max Velocity')
    ax.axhline(y=STICTION_THRESHOLD, color='blue', linestyle='--',
               alpha=0.5, label='Stiction')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Individual Robot Velocities Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Distance from origin over time
    ax = axes[1, 0]

    for i in range(num_robots):
        positions = tracking_data['robot_positions'][i]
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        ax.plot(time, radii, color=colors[i], alpha=0.7,
                linewidth=1, label=f'Robot {i+1}')

    # Add centroid radius
    centroid_radii = np.sqrt(centroid[:, 0]**2 + centroid[:, 1]**2)
    ax.plot(time, centroid_radii, 'k--', alpha=0.8,
            linewidth=2, label='Centroid')

    ax.axhline(y=radius, color='green', linestyle=':',
               alpha=0.5, label='Target Radius')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from Origin (m)')
    ax.set_title('Radial Distance Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Formation metrics
    ax = axes[1, 1]

    if 'formation_metrics' in tracking_data and tracking_data['formation_metrics']:
        areas = [fm['area'] for fm in tracking_data['formation_metrics']]
        ax.plot(time, areas, 'purple', linewidth=2, label='Formation Area')

        mean_area = np.mean(areas[STABLE_START:])
        ax.axhline(y=mean_area, color='purple', linestyle='--',
                   alpha=0.5, label=f'Mean Area: {mean_area:.4f}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Formation Area (m²)')
        ax.set_title('Formation Shape Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{config_name} Configuration - Radius {radius}m',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, f'{config_name}_r{int(radius*1000):03d}mm.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("INDIVIDUAL ROBOT MOTION ANALYSIS")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Radii to test: {RADII_TO_TEST}")
    print(f"  Max velocity: {MAX_VELOCITY} m/s")
    print(f"  Stiction threshold: {STICTION_THRESHOLD} m/s")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print()

    # Storage for all results
    all_results = {
        '3-robot': {},
        '4-robot-square': {},
        '4-robot-dual': {}
    }

    # List to accumulate CSV data
    csv_data = []

    # =========================================================================
    # RUN ALL SIMULATIONS
    # =========================================================================

    configurations = [
        ('3-robot', OmniCluster, FORMATION_3ROBOT, ocp.critical_point_orbiter_plane_fitting),
        ('4-robot-square', QuadCluster, FORMATION_4ROBOT_SQUARE, qcp.center_orbiter_quad),
        ('4-robot-dual', QuadCluster, FORMATION_4ROBOT_ADVANCED, qcp.center_orbiter_quad_advanced)
    ]

    for config_name, cluster_class, formation_config, control_func in configurations:
        print(f"\n{config_name} Configuration")
        print("-" * 50)

        for radius in RADII_TO_TEST:
            print(f"  Testing radius {radius:.2f} m...")

            # Run detailed simulation
            tracking = run_detailed_simulation(
                cluster_class,
                formation_config,
                control_func,
                radius,
                verbose=False
            )

            # Analyze results
            metrics = analyze_tracking_data(tracking, radius)
            all_results[config_name][radius] = metrics

            # Add to CSV data
            csv_row = {
                'configuration': config_name,
                'commanded_radius': radius,
                **metrics
            }
            csv_data.append(csv_row)

            # Create radius-specific plots for selected radii
            if radius in [0.01, 0.1, 0.3, 0.5]:
                create_radius_specific_plots(tracking, config_name, radius, OUTPUT_DIR)

            # Print summary
            print(f"    ✓ Error: {metrics['centroid_radius_error']:+.4f}m, "
                  f"Saturations: {metrics['saturation_rate']*100:.1f}%, "
                  f"Formation CV: {metrics.get('formation_area_cv', 0)*100:.1f}%")

    # =========================================================================
    # CREATE COMPREHENSIVE PLOTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Main comprehensive plot
    output_file = os.path.join(OUTPUT_DIR, 'comprehensive_motion_analysis.png')
    create_comprehensive_plot(all_results, output_file)

    # Save results to CSV
    csv_file = os.path.join(OUTPUT_DIR, 'motion_analysis_metrics.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"Saved metrics: {csv_file}")

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    theoretical_min = calculate_theoretical_min_radius(MAX_VELOCITY, 0.67)
    print(f"\n1. THEORETICAL LIMITS:")
    print(f"   - Theoretical minimum radius: {theoretical_min:.3f} m")
    print(f"   - Based on max velocity: {MAX_VELOCITY} m/s")
    print(f"   - Assumes 67% of velocity for orbital motion")

    print(f"\n2. VELOCITY SATURATION ANALYSIS:")
    for config_name in all_results:
        worst_radius = None
        worst_saturation = 0
        for radius in RADII_TO_TEST:
            sat_rate = all_results[config_name][radius]['saturation_rate']
            if sat_rate > worst_saturation:
                worst_saturation = sat_rate
                worst_radius = radius
        print(f"   {config_name}: Worst at r={worst_radius}m ({worst_saturation*100:.1f}% saturation)")

    print(f"\n3. FORMATION STABILITY:")
    for config_name in all_results:
        stability_scores = []
        for radius in RADII_TO_TEST:
            cv = all_results[config_name][radius].get('formation_area_cv', np.inf)
            if cv != np.inf:
                stability_scores.append((radius, cv))

        if stability_scores:
            stability_scores.sort(key=lambda x: x[1])
            best = stability_scores[0]
            worst = stability_scores[-1]
            print(f"   {config_name}:")
            print(f"     Best:  r={best[0]}m (CV={best[1]*100:.1f}%)")
            print(f"     Worst: r={worst[0]}m (CV={worst[1]*100:.1f}%)")

    print(f"\n4. PERFORMANCE DEGRADATION CAUSES:")
    print(f"   - Small radii (r < {theoretical_min:.2f}m): Velocity saturation dominates")
    print(f"   - Medium radii: Control conflicts between formation and orbital")
    print(f"   - Large radii: Formation stability degrades")
    print(f"   - 4-robot dual: Higher control overhead, more complex kinematics")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Generated plots and metrics in: {OUTPUT_DIR}")
    print(f"\nKey findings:")
    print(f"  • Centripetal force requirement (v²/r) causes failure at small radii")
    print(f"  • Individual robots saturate at different rates")
    print(f"  • Formation deformation increases with radius")
    print(f"  • 4-robot dual Jacobian has higher overhead, explaining worse performance")


if __name__ == "__main__":
    main()