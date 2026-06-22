"""
sim2real_comparison_all.py - Comprehensive Simulation-to-Real Robot Comparison

Compares simulated robot behavior against actual physical robot data across
multiple radius configurations and robot formations.

Configuration naming convention:
- 0XX: 3-robot equilateral formation (radius = XX/100 meters)
- 1XX: 4-robot square formation with d1=d2=0.3 (radius = XX/100 meters)
- 2XX: 4-robot advanced formation with d1=0.433, d2=0.25 (radius = XX/100 meters)
"""
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from glob import glob
from typing import Dict, List, Tuple, Optional

# New architecture imports
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
NUM_TRIALS = 4  # Number of simulation trials per configuration
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds

# Environment
ENVIRONMENT = "vortex1"
ENVIRONMENT_FUNC = vortex1

# Real robot data directory (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(script_dir, "../real_robot_data/real_robot_trajectories")

# Output (relative to this script)
OUTPUT_DIR = os.path.join(script_dir, "sim2real_results")
SAVE_PLOTS = True
MAX_PLOTS_PER_CONFIG = 1  # Save one representative plot per configuration

# Formation configuration files (relative to project root)
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# ============================================================================
# CONFIGURATION MAPPING
# ============================================================================

def get_configuration_details(orbit_name: str) -> Dict:
    """
    Parse orbit configuration from filename.

    Args:
        orbit_name: e.g., "orbit040", "orbit101", "orbit201"

    Returns:
        Dictionary with configuration details
    """
    # Extract numeric value
    orbit_num = int(orbit_name.replace('orbit', ''))

    if orbit_num < 100:
        # 0XX series: 3-robot equilateral
        return {
            'series': '0XX',
            'robot_type': '3-robot',
            'num_robots': 3,
            'radius': orbit_num / 100.0,
            'formation_config': FORMATION_3ROBOT,
            'control_primitive': 'critical_point_orbiter_plane_fitting',
            'control_func': ocp.critical_point_orbiter_plane_fitting,
            'cluster_class': OmniCluster,
            'diagonal_config': None
        }
    elif orbit_num < 200:
        # 1XX series: 4-robot square
        return {
            'series': '1XX',
            'robot_type': '4-robot-square',
            'num_robots': 4,
            'radius': (orbit_num - 100) / 100.0,
            'formation_config': FORMATION_4ROBOT_SQUARE,
            'control_primitive': 'center_orbiter_quad',
            'control_func': qcp.center_orbiter_quad,
            'cluster_class': QuadCluster,
            'diagonal_config': {'d1': 0.3, 'd2': 0.3}
        }
    else:
        # 2XX series: 4-robot advanced
        return {
            'series': '2XX',
            'robot_type': '4-robot-advanced',
            'num_robots': 4,
            'radius': (orbit_num - 200) / 100.0,
            'formation_config': FORMATION_4ROBOT_ADVANCED,
            'control_primitive': 'center_orbiter_quad_advanced',
            'control_func': qcp.center_orbiter_quad_advanced,
            'cluster_class': QuadCluster,
            'diagonal_config': {'d1': 0.433, 'd2': 0.25}
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_real_robot_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real robot trajectory from CSV file.

    Returns:
        trajectory: Nx2 array of (x, y) positions
        time: N-element array of timestamps
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Real robot data not found: {filepath}")

    # Load CSV
    data = pd.read_csv(filepath)

    time = data['time_s'].values
    x = data['x_m'].values
    y = data['y_m'].values

    trajectory = np.column_stack([x, y])

    return trajectory, time


def run_simulation(config: Dict, start_x: float, start_y: float,
                  store_diagnostics: bool = False) -> Tuple[np.ndarray, Optional[List]]:
    """
    Run a single simulation with specified configuration and starting position.

    Args:
        config: Configuration dictionary from get_configuration_details
        start_x: Starting x position
        start_y: Starting y position
        store_diagnostics: Whether to store diagnostic data

    Returns:
        center_points: Nx2 array of (x, y) positions
        diagnostics: List of diagnostic dictionaries (if requested)
    """
    # Create field
    field = AnalyticalField(ENVIRONMENT_FUNC)

    # Create cluster (initially at default position)
    # Suppress misleading initialization print
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cluster = config['cluster_class'](config['formation_config'], field)

    # Reset to actual starting position: (0, radius)
    cluster.reset(x_c=start_x, y_c=start_y)

    # Verify actual starting position
    actual_x, actual_y = cluster.get_centroid()
    if abs(actual_x - start_x) > 0.01 or abs(actual_y - start_y) > 0.01:
        print(f"      WARNING: Reset to ({start_x:.3f}, {start_y:.3f}) but actual is ({actual_x:.3f}, {actual_y:.3f})")

    # Create control primitive wrapper with desired radius
    desired_radius = config['radius']
    if config['num_robots'] == 3:
        def control_primitive(cluster):
            return config['control_func'](cluster, desired_radius=desired_radius)
    else:
        def control_primitive(cluster):
            return config['control_func'](cluster, desired_radius=desired_radius)

    # Storage for trajectory
    center_points = []

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        center_points.append([x_c, y_c])

        # Move cluster using control primitive
        cluster.move(control_primitive)

    center_points = np.array(center_points)

    # Get diagnostics if requested (only OmniCluster has this method)
    diagnostics = None
    if store_diagnostics and hasattr(cluster, 'get_diagnostics'):
        diagnostics = cluster.get_diagnostics()

    return center_points, diagnostics


def estimate_center_from_trajectory(trajectory: np.ndarray,
                                   start_idx: int = 100) -> Tuple[float, float]:
    """
    Estimate the center of rotation from orbital trajectory.

    Uses the latter part of trajectory (after transient) for estimation.
    """
    if len(trajectory) <= start_idx:
        start_idx = len(trajectory) // 4

    # Use data after transient
    stable_trajectory = trajectory[start_idx:]

    # Estimate center as mean position (assumes roughly circular orbit)
    center_x = np.mean(stable_trajectory[:, 0])
    center_y = np.mean(stable_trajectory[:, 1])

    return center_x, center_y


def calculate_statistics(trajectories: List[np.ndarray],
                        real_trajectory: np.ndarray) -> Dict:
    """
    Calculate comprehensive statistics for a set of simulation trajectories.

    Args:
        trajectories: List of simulated trajectory arrays
        real_trajectory: Real robot trajectory array

    Returns:
        Dictionary of statistics
    """
    # Collect metrics for each trajectory
    distances_from_origin = []
    estimated_centers = []
    distances_from_centers = []

    for traj in trajectories:
        # Distance from origin over time
        dist_origin = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        distances_from_origin.append(dist_origin)

        # Estimate center
        center_x, center_y = estimate_center_from_trajectory(traj)
        estimated_centers.append((center_x, center_y))

        # Distance from estimated center
        dist_center = np.sqrt((traj[:, 0] - center_x)**2 +
                            (traj[:, 1] - center_y)**2)
        distances_from_centers.append(dist_center)

    # Convert to arrays for easier calculation
    distances_from_origin = np.array(distances_from_origin)
    estimated_centers = np.array(estimated_centers)
    distances_from_centers = np.array(distances_from_centers)

    # Calculate average distance from origin (use stable part)
    stable_start = 100  # Skip initial transient
    avg_dist_origin = np.mean(distances_from_origin[:, stable_start:])
    std_dist_origin = np.std(distances_from_origin[:, stable_start:])

    # Calculate average distance from estimated centers
    avg_dist_center = np.mean(distances_from_centers[:, stable_start:])
    std_dist_center = np.std(distances_from_centers[:, stable_start:])

    # Average estimated center position
    avg_x_est = np.mean(estimated_centers[:, 0])
    std_x_est = np.std(estimated_centers[:, 0])
    avg_y_est = np.mean(estimated_centers[:, 1])
    std_y_est = np.std(estimated_centers[:, 1])

    # Calculate RMSE from origin (using stable part)
    all_positions = []
    for traj in trajectories:
        all_positions.extend(traj[stable_start:].tolist())
    all_positions = np.array(all_positions)

    rmse = np.sqrt(np.mean(all_positions[:, 0]**2 + all_positions[:, 1]**2))

    # Real robot statistics (for comparison)
    real_dist_origin = np.sqrt(real_trajectory[:, 0]**2 + real_trajectory[:, 1]**2)
    real_center_x, real_center_y = estimate_center_from_trajectory(real_trajectory)
    real_avg_dist = np.mean(real_dist_origin[stable_start:])

    return {
        'avg_dist_origin': avg_dist_origin,
        'std_dist_origin': std_dist_origin,
        'avg_dist_center': avg_dist_center,
        'std_dist_center': std_dist_center,
        'avg_x_est': avg_x_est,
        'std_x_est': std_x_est,
        'avg_y_est': avg_y_est,
        'std_y_est': std_y_est,
        'rmse': rmse,
        'real_avg_dist': real_avg_dist,
        'real_center': (real_center_x, real_center_y)
    }


def plot_comparison(sim_trajectories: List[np.ndarray], real_trajectory: np.ndarray,
                   config: Dict, stats: Dict, output_path: str):
    """
    Create comparison plot for a configuration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left plot: All trajectories overlaid
    ax = axes[0]

    # Plot simulation trajectories
    for i, traj in enumerate(sim_trajectories):
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1.5,
               label=f'Sim {i+1}')

    # Plot real robot trajectory
    ax.plot(real_trajectory[:, 0], real_trajectory[:, 1],
           'r-', linewidth=2, alpha=0.8, label='Real Robot')

    # Add reference circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = config['radius']
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--',
           alpha=0.5, linewidth=1, label=f'Target r={r:.2f}m')

    # Mark origins
    ax.plot(0, 0, 'ko', markersize=8, label='Origin')

    # Mark estimated centers
    ax.plot(stats['avg_x_est'], stats['avg_y_est'], 'b^',
           markersize=10, label=f'Sim Center ({stats["avg_x_est"]:.3f}, {stats["avg_y_est"]:.3f})')
    ax.plot(stats['real_center'][0], stats['real_center'][1], 'r^',
           markersize=10, label=f'Real Center ({stats["real_center"][0]:.3f}, {stats["real_center"][1]:.3f})')

    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title(f'{config["robot_type"]} - Radius {config["radius"]:.2f}m', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=8)

    # Right plot: Statistics summary
    ax = axes[1]
    ax.axis('off')

    # Create statistics text
    stats_text = f"""Configuration: {config['series']} ({config['robot_type']})
Control Primitive: {config['control_primitive']}
Target Radius: {config['radius']:.3f} m

Simulation Statistics ({NUM_TRIALS} trials):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Distance from Origin: {stats['avg_dist_origin']:.4f} ± {stats['std_dist_origin']:.4f} m
Average Distance from Center: {stats['avg_dist_center']:.4f} ± {stats['std_dist_center']:.4f} m
Estimated Center X: {stats['avg_x_est']:.4f} ± {stats['std_x_est']:.4f} m
Estimated Center Y: {stats['avg_y_est']:.4f} ± {stats['std_y_est']:.4f} m
RMSE from Origin: {stats['rmse']:.4f} m

Real Robot Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Distance from Origin: {stats['real_avg_dist']:.4f} m
Estimated Center: ({stats['real_center'][0]:.4f}, {stats['real_center'][1]:.4f}) m

Performance Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distance Error: {abs(stats['avg_dist_origin'] - stats['real_avg_dist']):.4f} m
Center Error: {np.sqrt((stats['avg_x_est'] - stats['real_center'][0])**2 + (stats['avg_y_est'] - stats['real_center'][1])**2):.4f} m"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=10, fontfamily='monospace', verticalalignment='top')

    plt.suptitle(f'Sim-to-Real Comparison: orbit{int(config["radius"]*100):03d}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("COMPREHENSIVE SIM-TO-REAL COMPARISON")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Simulation trials per config: {NUM_TRIALS}")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s ({SIM_TIME} steps)")
    print(f"  Real data directory: {REAL_DATA_DIR}")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all available trajectory files
    trajectory_files = glob(os.path.join(REAL_DATA_DIR, "orbit*_trajectory.csv"))

    if not trajectory_files:
        print(f"ERROR: No trajectory files found in {REAL_DATA_DIR}")
        print("Please run the MATLAB export script first.")
        return

    trajectory_files.sort()
    print(f"Found {len(trajectory_files)} trajectory files to process")
    print()

    # Results storage
    all_results = []

    # Process each configuration
    for file_idx, trajectory_file in enumerate(trajectory_files):
        # Extract orbit name
        basename = os.path.basename(trajectory_file)
        orbit_name = basename.replace('_trajectory.csv', '')

        print(f"\n[{file_idx+1}/{len(trajectory_files)}] Processing {orbit_name}")
        print("-" * 50)

        try:
            # Get configuration details
            config = get_configuration_details(orbit_name)

            print(f"  Type: {config['robot_type']}")
            print(f"  Radius: {config['radius']:.3f} m")
            print(f"  Control: {config['control_primitive']}")

            # Load real robot data
            real_trajectory, real_time = load_real_robot_data(trajectory_file)
            print(f"  Real data: {len(real_trajectory)} points, {real_time[-1]:.1f}s")

            # Run simulations
            print(f"  Running {NUM_TRIALS} simulations...")
            sim_trajectories = []

            for trial in range(NUM_TRIALS):
                # Start at (0, radius)
                start_x = 0.0
                start_y = config['radius']

                # Run simulation
                if trial == 0:
                    print(f"    Starting position: ({start_x:.3f}, {start_y:.3f})")
                trajectory, _ = run_simulation(config, start_x, start_y,
                                             store_diagnostics=(trial == 0))
                sim_trajectories.append(trajectory)

                print(f"    Trial {trial+1}/{NUM_TRIALS} complete")

            # Calculate statistics
            print("  Calculating statistics...")
            stats = calculate_statistics(sim_trajectories, real_trajectory)

            # Store results
            result = {
                'config_name': orbit_name,
                'robot_type': config['robot_type'],
                'series': config['series'],
                'radius_m': config['radius'],
                'control_primitive': config['control_primitive'],
                'avg_dist_origin': stats['avg_dist_origin'],
                'std_dist_origin': stats['std_dist_origin'],
                'avg_dist_center': stats['avg_dist_center'],
                'std_dist_center': stats['std_dist_center'],
                'avg_x_est': stats['avg_x_est'],
                'std_x_est': stats['std_x_est'],
                'avg_y_est': stats['avg_y_est'],
                'std_y_est': stats['std_y_est'],
                'rmse': stats['rmse'],
                'real_avg_dist': stats['real_avg_dist'],
                'real_center_x': stats['real_center'][0],
                'real_center_y': stats['real_center'][1]
            }
            all_results.append(result)

            # Print summary
            print(f"  Results:")
            print(f"    Sim: r={stats['avg_dist_origin']:.4f}±{stats['std_dist_origin']:.4f}m")
            print(f"    Real: r={stats['real_avg_dist']:.4f}m")
            print(f"    RMSE: {stats['rmse']:.4f}m")

            # Create plot
            if SAVE_PLOTS and file_idx < MAX_PLOTS_PER_CONFIG * 3:  # Save some example plots
                plot_path = os.path.join(OUTPUT_DIR, f"{orbit_name}_comparison.png")
                plot_comparison(sim_trajectories, real_trajectory, config, stats, plot_path)
                print(f"  ✓ Saved plot: {plot_path}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue

    # Save results to CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "sim2real_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Print summary statistics by series
    print("\n" + "=" * 70)
    print("SUMMARY BY CONFIGURATION TYPE")
    print("=" * 70)

    for series in ['0XX', '1XX', '2XX']:
        series_results = results_df[results_df['series'] == series]
        if len(series_results) > 0:
            print(f"\n{series} Series ({series_results.iloc[0]['robot_type']}):")
            print(f"  Configurations tested: {len(series_results)}")
            print(f"  Average RMSE: {series_results['rmse'].mean():.4f} ± {series_results['rmse'].std():.4f}")
            print(f"  Average distance error: {(series_results['avg_dist_origin'] - series_results['real_avg_dist']).abs().mean():.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()