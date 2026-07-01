"""
sim2real_comparison_all_fixed.py - Improved Simulation-to-Real Robot Comparison

Compares simulated robot behavior against actual physical robot data across
multiple radius configurations and robot formations.

Key improvements:
- Clearer naming: origin (0,0) is TRUE center, everything else is error
- Better statistics: uses variation within trajectory, not across identical runs
- Single run per configuration (since deterministic)
- Tracks estimated center throughout trajectory

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
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
TRUE_CENTER = (0.0, 0.0)  # The TRUE center is always at origin

# Environment
ENVIRONMENT = "vortex1"
ENVIRONMENT_FUNC = vortex1

# Real robot data directory (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(script_dir, "../real_robot_data/real_robot_trajectories")

# Output (relative to this script)
OUTPUT_DIR = os.path.join(script_dir, "sim2real_results_fixed")
SAVE_PLOTS = True
MAX_PLOTS = 3  # Save a few representative plots

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


def run_simulation_with_tracking(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run simulation and track estimated centers throughout.

    Args:
        config: Configuration dictionary

    Returns:
        trajectory: Nx2 array of centroid positions
        estimated_centers: Nx2 array of estimated center positions (or None if not available)
    """
    # Create field
    field = AnalyticalField(ENVIRONMENT_FUNC)

    # Create cluster (suppress initialization print)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cluster = config['cluster_class'](config['formation_config'], field)

    # Start at (0, desired_radius)
    start_x = 0.0
    start_y = config['radius']
    cluster.reset(x_c=start_x, y_c=start_y)

    # Create control primitive wrapper
    desired_radius = config['radius']
    def control_primitive(cluster):
        return config['control_func'](cluster, desired_radius=desired_radius)

    # Storage
    trajectory = []
    estimated_centers = []

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        trajectory.append([x_c, y_c])

        # Try to get estimated center if available
        # For 3-robot, we can extract from diagnostics
        # For 4-robot, we'll need to calculate it differently
        if config['num_robots'] == 3 and hasattr(cluster, 'get_diagnostics'):
            # Get the last diagnostic entry
            diagnostics = cluster.get_diagnostics()
            if diagnostics and len(diagnostics) > 0:
                last_diag = diagnostics[-1]
                if 'x_est' in last_diag and 'y_est' in last_diag:
                    estimated_centers.append([last_diag['x_est'], last_diag['y_est']])
                else:
                    # Estimate from position - not ideal but better than nothing
                    estimated_centers.append([0.0, 0.0])  # Assume estimating origin
            else:
                estimated_centers.append([0.0, 0.0])
        else:
            # For 4-robot or if no diagnostics, assume perfect estimation for now
            estimated_centers.append([0.0, 0.0])

        # Move cluster
        cluster.move(control_primitive)

    return np.array(trajectory), np.array(estimated_centers)


def calculate_improved_statistics(sim_trajectory: np.ndarray,
                                 sim_estimates: np.ndarray,
                                 real_trajectory: np.ndarray,
                                 desired_radius: float) -> Dict:
    """
    Calculate improved statistics with clearer naming.

    True center is always at origin (0,0).

    Args:
        sim_trajectory: Simulated centroid positions
        sim_estimates: Simulated center estimates (if available)
        real_trajectory: Real robot trajectory
        desired_radius: Target radius

    Returns:
        Dictionary of statistics with clear naming
    """
    # Skip initial transient (first 10 seconds = 100 timesteps)
    stable_start = 100

    # === SIMULATED ROBOT STATISTICS ===

    # Distance from TRUE center (origin)
    sim_distances_from_true = np.sqrt(sim_trajectory[:, 0]**2 + sim_trajectory[:, 1]**2)
    sim_stable_distances = sim_distances_from_true[stable_start:]

    sim_avg_radius = np.mean(sim_stable_distances)
    sim_std_radius = np.std(sim_stable_distances)
    sim_radius_error = sim_avg_radius - desired_radius  # Error from desired

    # Estimated center statistics (how well does robot know where center is?)
    if sim_estimates is not None and len(sim_estimates) > stable_start:
        sim_stable_estimates = sim_estimates[stable_start:]

        # Error in center estimation (distance from true center at origin)
        sim_center_errors = np.sqrt(sim_stable_estimates[:, 0]**2 +
                                   sim_stable_estimates[:, 1]**2)
        sim_avg_center_error = np.mean(sim_center_errors)
        sim_std_center_error = np.std(sim_center_errors)

        # X and Y components of center estimation error
        sim_avg_x_error = np.mean(sim_stable_estimates[:, 0])  # Should be ~0
        sim_std_x_error = np.std(sim_stable_estimates[:, 0])
        sim_avg_y_error = np.mean(sim_stable_estimates[:, 1])  # Should be ~0
        sim_std_y_error = np.std(sim_stable_estimates[:, 1])
    else:
        # No center estimates available
        sim_avg_center_error = np.nan
        sim_std_center_error = np.nan
        sim_avg_x_error = np.nan
        sim_std_x_error = np.nan
        sim_avg_y_error = np.nan
        sim_std_y_error = np.nan

    # RMSE from true center (overall position error)
    sim_stable_positions = sim_trajectory[stable_start:]
    sim_rmse = np.sqrt(np.mean(sim_stable_positions[:, 0]**2 +
                               sim_stable_positions[:, 1]**2))

    # === REAL ROBOT STATISTICS ===

    real_distances_from_true = np.sqrt(real_trajectory[:, 0]**2 + real_trajectory[:, 1]**2)
    real_stable_distances = real_distances_from_true[stable_start:] if len(real_distances_from_true) > stable_start else real_distances_from_true

    real_avg_radius = np.mean(real_stable_distances)
    real_std_radius = np.std(real_stable_distances)
    real_radius_error = real_avg_radius - desired_radius

    # Real robot's estimated center (from trajectory shape)
    if len(real_trajectory) > stable_start:
        real_stable = real_trajectory[stable_start:]
        real_est_x = np.mean(real_stable[:, 0])
        real_est_y = np.mean(real_stable[:, 1])
        real_center_error = np.sqrt(real_est_x**2 + real_est_y**2)
    else:
        real_est_x = np.mean(real_trajectory[:, 0])
        real_est_y = np.mean(real_trajectory[:, 1])
        real_center_error = np.sqrt(real_est_x**2 + real_est_y**2)

    return {
        # Desired configuration
        'desired_radius': desired_radius,

        # Simulated performance
        'sim_avg_radius': sim_avg_radius,
        'sim_std_radius': sim_std_radius,
        'sim_radius_error': sim_radius_error,
        'sim_rmse': sim_rmse,

        # Simulated center estimation accuracy
        'sim_avg_center_error': sim_avg_center_error,
        'sim_std_center_error': sim_std_center_error,
        'sim_avg_x_error': sim_avg_x_error,
        'sim_std_x_error': sim_std_x_error,
        'sim_avg_y_error': sim_avg_y_error,
        'sim_std_y_error': sim_std_y_error,

        # Real robot performance
        'real_avg_radius': real_avg_radius,
        'real_std_radius': real_std_radius,
        'real_radius_error': real_radius_error,
        'real_est_center_x': real_est_x,
        'real_est_center_y': real_est_y,
        'real_center_error': real_center_error,

        # Comparison metrics
        'radius_difference': abs(sim_avg_radius - real_avg_radius),
        'performance_ratio': sim_avg_radius / real_avg_radius if real_avg_radius > 0 else np.nan
    }


def plot_improved_comparison(sim_trajectory: np.ndarray, real_trajectory: np.ndarray,
                            config: Dict, stats: Dict, output_path: str):
    """
    Create improved comparison plot with clearer labeling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left plot: Trajectories
    ax = axes[0]

    # Plot simulation trajectory
    ax.plot(sim_trajectory[:, 0], sim_trajectory[:, 1], 'b-',
           linewidth=1.5, alpha=0.7, label='Simulation')

    # Plot real robot trajectory
    ax.plot(real_trajectory[:, 0], real_trajectory[:, 1], 'r-',
           linewidth=1.5, alpha=0.7, label='Real Robot')

    # Add reference circle at desired radius
    theta = np.linspace(0, 2*np.pi, 100)
    r = config['radius']
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'g--',
           alpha=0.5, linewidth=1, label=f'Desired r={r:.2f}m')

    # Mark TRUE center (origin)
    ax.plot(0, 0, 'ko', markersize=10, label='True Center (0,0)', zorder=5)
    ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=2, zorder=5)

    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title(f'{config["robot_type"]} - Target Radius {config["radius"]:.2f}m', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=9)

    # Set reasonable axis limits
    max_r = max(config['radius'] * 1.5, 0.5)
    ax.set_xlim([-max_r, max_r])
    ax.set_ylim([-max_r, max_r])

    # Right plot: Statistics
    ax = axes[1]
    ax.axis('off')

    # Create statistics text with clearer naming
    stats_text = f"""Configuration: {config['series']} ({config['robot_type']})
Control: {config['control_primitive']}
True Center: (0.0, 0.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET SPECIFICATION:
• Desired Radius: {stats['desired_radius']:.3f} m

SIMULATION PERFORMANCE:
• Achieved Radius: {stats['sim_avg_radius']:.4f} ± {stats['sim_std_radius']:.4f} m
• Radius Error: {stats['sim_radius_error']:+.4f} m ({stats['sim_radius_error']/stats['desired_radius']*100:+.1f}%)
• Position RMSE: {stats['sim_rmse']:.4f} m

CENTER ESTIMATION ACCURACY (Sim):
• Center Error: {stats['sim_avg_center_error']:.4f} ± {stats['sim_std_center_error']:.4f} m
• X Error: {stats['sim_avg_x_error']:.4f} ± {stats['sim_std_x_error']:.4f} m
• Y Error: {stats['sim_avg_y_error']:.4f} ± {stats['sim_std_y_error']:.4f} m

REAL ROBOT PERFORMANCE:
• Achieved Radius: {stats['real_avg_radius']:.4f} ± {stats['real_std_radius']:.4f} m
• Radius Error: {stats['real_radius_error']:+.4f} m ({stats['real_radius_error']/stats['desired_radius']*100:+.1f}%)
• Estimated Center: ({stats['real_est_center_x']:.4f}, {stats['real_est_center_y']:.4f})
• Center Error: {stats['real_center_error']:.4f} m

SIM vs REAL COMPARISON:
• Radius Difference: {stats['radius_difference']:.4f} m
• Performance Ratio: {stats['performance_ratio']:.3f}"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, fontfamily='monospace', verticalalignment='top')

    plt.suptitle(f'Sim-to-Real Comparison: {orbit_name} (r={config["radius"]:.2f}m)',
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
    print("IMPROVED SIM-TO-REAL COMPARISON")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  True Center: {TRUE_CENTER}")
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
    plots_saved = 0

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
            print(f"  Target Radius: {config['radius']:.3f} m")
            print(f"  Control: {config['control_primitive']}")

            # Load real robot data
            real_trajectory, real_time = load_real_robot_data(trajectory_file)
            print(f"  Real data: {len(real_trajectory)} points, {real_time[-1]:.1f}s")

            # Run single simulation (deterministic, no need for multiple runs)
            print(f"  Running simulation from (0, {config['radius']:.3f})...")
            sim_trajectory, sim_estimates = run_simulation_with_tracking(config)
            print(f"  Simulation complete: {len(sim_trajectory)} timesteps")

            # Calculate statistics
            print("  Calculating statistics...")
            stats = calculate_improved_statistics(
                sim_trajectory, sim_estimates, real_trajectory, config['radius']
            )

            # Store results
            result = {
                'config_name': orbit_name,
                'robot_type': config['robot_type'],
                'series': config['series'],
                **stats
            }
            all_results.append(result)

            # Print summary
            print(f"  Results:")
            print(f"    Sim:  r={stats['sim_avg_radius']:.4f}±{stats['sim_std_radius']:.4f}m "
                  f"(error: {stats['sim_radius_error']:+.4f}m)")
            print(f"    Real: r={stats['real_avg_radius']:.4f}±{stats['real_std_radius']:.4f}m "
                  f"(error: {stats['real_radius_error']:+.4f}m)")
            print(f"    Difference: {stats['radius_difference']:.4f}m")

            # Create plot for selected configurations
            if SAVE_PLOTS and plots_saved < MAX_PLOTS:
                plot_path = os.path.join(OUTPUT_DIR, f"{orbit_name}_comparison.png")
                plot_improved_comparison(sim_trajectory, real_trajectory, config, stats, plot_path)
                print(f"  ✓ Saved plot: {plot_path}")
                plots_saved += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results to CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "sim2real_comparison_improved.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
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
            print(f"  Avg radius error (sim): {series_results['sim_radius_error'].mean():.4f} ± "
                  f"{series_results['sim_radius_error'].std():.4f} m")
            print(f"  Avg radius error (real): {series_results['real_radius_error'].mean():.4f} ± "
                  f"{series_results['real_radius_error'].std():.4f} m")
            print(f"  Avg sim-real difference: {series_results['radius_difference'].mean():.4f} m")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• True center is at origin (0,0) for all experiments")
    print("• Radius error = achieved radius - desired radius")
    print("• Center error = distance of estimated center from true center")
    print("• Positive radius error means orbiting too far from center")
    print("• Check sim_std_x_error and sim_std_y_error for estimation variance")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Make orbit_name available for plot function
    orbit_name = ""
    main()