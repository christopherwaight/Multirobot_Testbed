"""
sim2real_comparison_all_multimode.py - Multi-Field-Mode Sim-to-Real Comparison

Compares robot behavior across 4 different field representations:
1. Analytical (ground truth mathematical field)
2. Neural Network (learned approximation)
3. RBF Interpolator (interpolated approximation)
4. Real Robot (actual hardware data from MATLAB)

For each radius configuration, we run simulations with each field type
and compare how they perform relative to the real robot data.

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
from src.fields.field_types import AnalyticalField, NNField, RBFField
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
PREDICTOR_DIR = "vortex_predictors"  # ML models directory

# Real robot data directory (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(script_dir, "../real_robot_data/real_robot_trajectories")

# Output (relative to this script)
OUTPUT_DIR = os.path.join(script_dir, "sim2real_multimode_results")
SAVE_PLOTS = True
PLOTS_TO_SAVE = ['orbit010', 'orbit040', 'orbit101', 'orbit140', 'orbit201', 'orbit240']  # Representative plots

# Formation configuration files (relative to project root)
project_root = os.path.dirname(script_dir)
FORMATION_3ROBOT = os.path.join(project_root, "config/formations/equilateral_default.yaml")
FORMATION_4ROBOT_SQUARE = os.path.join(project_root, "config/formations/quad_square.yaml")
FORMATION_4ROBOT_ADVANCED = os.path.join(project_root, "config/formations/quad_default.yaml")

# Field modes to compare
FIELD_MODES = ['analytical', 'nn', 'rbf']

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

def create_field(mode: str, predictor_dir: str = PREDICTOR_DIR):
    """
    Create field object based on mode.

    Args:
        mode: "analytical", "nn", or "rbf"
        predictor_dir: Directory containing ML models

    Returns:
        VectorField instance
    """
    if mode == "analytical":
        return AnalyticalField(ENVIRONMENT_FUNC)
    elif mode == "nn":
        try:
            return NNField(predictor_dir)
        except Exception as e:
            print(f"  Warning: NN model not found, falling back to analytical. Error: {e}")
            return AnalyticalField(ENVIRONMENT_FUNC)
    elif mode == "rbf":
        try:
            return RBFField(predictor_dir)
        except Exception as e:
            print(f"  Warning: RBF model not found, falling back to analytical. Error: {e}")
            return AnalyticalField(ENVIRONMENT_FUNC)
    else:
        raise ValueError(f"Unknown mode: {mode}")


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


def run_simulation(config: Dict, field_mode: str) -> np.ndarray:
    """
    Run simulation with specified field mode.

    Args:
        config: Configuration dictionary
        field_mode: "analytical", "nn", or "rbf"

    Returns:
        trajectory: Nx2 array of centroid positions
    """
    # Create field based on mode
    field = create_field(field_mode)

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

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        trajectory.append([x_c, y_c])

        # Move cluster
        cluster.move(control_primitive)

    return np.array(trajectory)


def calculate_trajectory_statistics(trajectory: np.ndarray, desired_radius: float,
                                   stable_start: int = 100) -> Dict:
    """
    Calculate statistics for a single trajectory.

    Args:
        trajectory: Nx2 array of positions
        desired_radius: Target radius
        stable_start: Timestep to start stable statistics (skip transient)

    Returns:
        Dictionary of statistics
    """
    # Distance from TRUE center (origin)
    distances_from_true = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
    stable_distances = distances_from_true[stable_start:] if len(distances_from_true) > stable_start else distances_from_true

    avg_radius = np.mean(stable_distances)
    std_radius = np.std(stable_distances)
    radius_error = avg_radius - desired_radius

    # Estimate center from stable trajectory (where robot thinks center is)
    if len(trajectory) > stable_start:
        stable_trajectory = trajectory[stable_start:]
    else:
        stable_trajectory = trajectory

    est_center_x = np.mean(stable_trajectory[:, 0])
    est_center_y = np.mean(stable_trajectory[:, 1])
    center_error = np.sqrt(est_center_x**2 + est_center_y**2)

    # Variation in estimated center (using sliding window)
    window_size = 50  # 5 seconds
    if len(stable_trajectory) > window_size:
        est_centers = []
        for i in range(len(stable_trajectory) - window_size):
            window = stable_trajectory[i:i+window_size]
            wx = np.mean(window[:, 0])
            wy = np.mean(window[:, 1])
            est_centers.append([wx, wy])
        est_centers = np.array(est_centers)
        std_x_est = np.std(est_centers[:, 0])
        std_y_est = np.std(est_centers[:, 1])
    else:
        std_x_est = 0.0
        std_y_est = 0.0

    # RMSE from true center
    rmse = np.sqrt(np.mean(stable_trajectory[:, 0]**2 + stable_trajectory[:, 1]**2))

    return {
        'avg_radius': avg_radius,
        'std_radius': std_radius,
        'radius_error': radius_error,
        'est_center_x': est_center_x,
        'est_center_y': est_center_y,
        'center_error': center_error,
        'std_x_est': std_x_est,
        'std_y_est': std_y_est,
        'rmse': rmse
    }


def calculate_comparison_statistics(trajectories: Dict[str, np.ndarray],
                                   real_trajectory: np.ndarray,
                                   desired_radius: float) -> Dict:
    """
    Calculate comprehensive comparison statistics across all field modes.

    Args:
        trajectories: Dict with keys 'analytical', 'nn', 'rbf' containing trajectories
        real_trajectory: Real robot trajectory
        desired_radius: Target radius

    Returns:
        Dictionary of comprehensive statistics
    """
    results = {}

    # Calculate stats for each simulated field mode
    for mode in ['analytical', 'nn', 'rbf']:
        if mode in trajectories:
            stats = calculate_trajectory_statistics(trajectories[mode], desired_radius)
            for key, value in stats.items():
                results[f'{mode}_{key}'] = value

    # Calculate stats for real robot
    real_stats = calculate_trajectory_statistics(real_trajectory, desired_radius)
    for key, value in real_stats.items():
        results[f'real_{key}'] = value

    # Add comparative metrics
    # How well do approximations match analytical?
    if 'analytical' in trajectories:
        analytical_radius = results.get('analytical_avg_radius', np.nan)

        if 'nn' in trajectories:
            results['nn_vs_analytical_error'] = results.get('nn_avg_radius', np.nan) - analytical_radius

        if 'rbf' in trajectories:
            results['rbf_vs_analytical_error'] = results.get('rbf_avg_radius', np.nan) - analytical_radius

    # How well do simulations match real?
    real_radius = results.get('real_avg_radius', np.nan)
    if 'analytical' in trajectories:
        results['analytical_vs_real_error'] = results.get('analytical_avg_radius', np.nan) - real_radius
    if 'nn' in trajectories:
        results['nn_vs_real_error'] = results.get('nn_avg_radius', np.nan) - real_radius
    if 'rbf' in trajectories:
        results['rbf_vs_real_error'] = results.get('rbf_avg_radius', np.nan) - real_radius

    # Standard deviation across different field estimates at steady state
    steady_state_centers_x = []
    steady_state_centers_y = []
    for mode in ['analytical', 'nn', 'rbf']:
        if f'{mode}_est_center_x' in results:
            steady_state_centers_x.append(results[f'{mode}_est_center_x'])
            steady_state_centers_y.append(results[f'{mode}_est_center_y'])

    if len(steady_state_centers_x) > 1:
        results['std_x_across_modes'] = np.std(steady_state_centers_x)
        results['std_y_across_modes'] = np.std(steady_state_centers_y)
    else:
        results['std_x_across_modes'] = np.nan
        results['std_y_across_modes'] = np.nan

    results['desired_radius'] = desired_radius

    return results


def plot_multimode_comparison(trajectories: Dict[str, np.ndarray],
                             real_trajectory: np.ndarray,
                             config: Dict, stats: Dict, output_path: str):
    """
    Create comparison plot showing all field modes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Colors for each mode
    colors = {
        'analytical': 'blue',
        'nn': 'green',
        'rbf': 'purple',
        'real': 'red'
    }

    # Plot each mode in separate subplot
    plot_configs = [
        ('analytical', 'Analytical Field', axes[0, 0]),
        ('nn', 'Neural Network Field', axes[0, 1]),
        ('rbf', 'RBF Interpolator Field', axes[1, 0]),
        ('real', 'Real Robot', axes[1, 1])
    ]

    for mode, title, ax in plot_configs:
        # Get trajectory
        if mode == 'real':
            traj = real_trajectory
        else:
            traj = trajectories.get(mode)

        if traj is not None:
            # Plot trajectory
            ax.plot(traj[:, 0], traj[:, 1], color=colors[mode],
                   linewidth=1.5, alpha=0.7, label=f'{title} Path')

            # Add reference circle at desired radius
            theta = np.linspace(0, 2*np.pi, 100)
            r = config['radius']
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--',
                   alpha=0.3, linewidth=1, label=f'Target r={r:.2f}m')

            # Mark TRUE center (origin)
            ax.plot(0, 0, 'ko', markersize=8, label='True Center', zorder=5)

            # Mark estimated center
            est_x = stats.get(f'{mode}_est_center_x', 0)
            est_y = stats.get(f'{mode}_est_center_y', 0)
            ax.plot(est_x, est_y, 'r^', markersize=8,
                   label=f'Est Center ({est_x:.3f}, {est_y:.3f})', zorder=4)

            # Add statistics text
            avg_r = stats.get(f'{mode}_avg_radius', np.nan)
            std_r = stats.get(f'{mode}_std_radius', np.nan)
            error = stats.get(f'{mode}_radius_error', np.nan)
            rmse = stats.get(f'{mode}_rmse', np.nan)

            stats_text = (f'r = {avg_r:.4f}±{std_r:.4f}m\n'
                         f'error = {error:+.4f}m\n'
                         f'RMSE = {rmse:.4f}m')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Configure axis
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='upper right', fontsize=8)

        # Set consistent axis limits
        max_r = max(config['radius'] * 1.5, 0.5)
        ax.set_xlim([-max_r, max_r])
        ax.set_ylim([-max_r, max_r])

    plt.suptitle(f'{config["robot_type"]} - {orbit_name} (Target r={config["radius"]:.2f}m)',
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
    print("MULTI-FIELD-MODE SIM-TO-REAL COMPARISON")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Field Modes: Analytical, Neural Network, RBF")
    print(f"  True Center: {TRUE_CENTER}")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s ({SIM_TIME} steps)")
    print(f"  ML Models: {PREDICTOR_DIR}/")
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
            print(f"  Target Radius: {config['radius']:.3f} m")
            print(f"  Control: {config['control_primitive']}")

            # Load real robot data
            real_trajectory, real_time = load_real_robot_data(trajectory_file)
            print(f"  Real data: {len(real_trajectory)} points, {real_time[-1]:.1f}s")

            # Run simulations with different field modes
            trajectories = {}
            print(f"  Running simulations:")

            for mode in FIELD_MODES:
                try:
                    print(f"    {mode.upper()}...", end='', flush=True)
                    trajectories[mode] = run_simulation(config, mode)
                    print(f" ✓ ({len(trajectories[mode])} steps)")
                except Exception as e:
                    print(f" ✗ Error: {e}")

            # Calculate statistics
            print("  Calculating statistics...")
            stats = calculate_comparison_statistics(trajectories, real_trajectory,
                                                   config['radius'])

            # Store results
            result = {
                'config_name': orbit_name,
                'robot_type': config['robot_type'],
                'series': config['series'],
                **stats
            }
            all_results.append(result)

            # Print summary
            print(f"  Results Summary:")
            print(f"    Analytical: r={stats.get('analytical_avg_radius', np.nan):.4f}±"
                  f"{stats.get('analytical_std_radius', np.nan):.4f}m")
            print(f"    NN:         r={stats.get('nn_avg_radius', np.nan):.4f}±"
                  f"{stats.get('nn_std_radius', np.nan):.4f}m")
            print(f"    RBF:        r={stats.get('rbf_avg_radius', np.nan):.4f}±"
                  f"{stats.get('rbf_std_radius', np.nan):.4f}m")
            print(f"    Real:       r={stats.get('real_avg_radius', np.nan):.4f}±"
                  f"{stats.get('real_std_radius', np.nan):.4f}m")
            print(f"    Std across modes: x={stats.get('std_x_across_modes', np.nan):.4f}, "
                  f"y={stats.get('std_y_across_modes', np.nan):.4f}")

            # Create plot for selected configurations
            if SAVE_PLOTS and orbit_name in PLOTS_TO_SAVE:
                plot_path = os.path.join(OUTPUT_DIR, f"{orbit_name}_multimode_comparison.png")
                plot_multimode_comparison(trajectories, real_trajectory, config, stats, plot_path)
                print(f"  ✓ Saved plot: {plot_path}")

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
    csv_path = os.path.join(OUTPUT_DIR, "sim2real_multimode_results.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Results saved to: {csv_path}")

    # Print field mode comparison
    print("\n" + "=" * 70)
    print("FIELD MODE COMPARISON")
    print("=" * 70)

    # Average errors across all configurations
    if len(results_df) > 0:
        print("\nAverage Radius Errors (vs desired):")
        for mode in ['analytical', 'nn', 'rbf', 'real']:
            col = f'{mode}_radius_error'
            if col in results_df.columns:
                mean_err = results_df[col].mean()
                std_err = results_df[col].std()
                print(f"  {mode.upper():12s}: {mean_err:+.4f} ± {std_err:.4f} m")

        print("\nAverage RMSE from True Center:")
        for mode in ['analytical', 'nn', 'rbf', 'real']:
            col = f'{mode}_rmse'
            if col in results_df.columns:
                mean_rmse = results_df[col].mean()
                std_rmse = results_df[col].std()
                print(f"  {mode.upper():12s}: {mean_rmse:.4f} ± {std_rmse:.4f} m")

        print("\nField Approximation vs Analytical:")
        for mode in ['nn', 'rbf']:
            col = f'{mode}_vs_analytical_error'
            if col in results_df.columns:
                mean_err = results_df[col].mean()
                std_err = results_df[col].std()
                print(f"  {mode.upper():12s}: {mean_err:+.4f} ± {std_err:.4f} m")

        print("\nSimulation vs Real Robot:")
        for mode in ['analytical', 'nn', 'rbf']:
            col = f'{mode}_vs_real_error'
            if col in results_df.columns:
                mean_err = results_df[col].mean()
                std_err = results_df[col].std()
                print(f"  {mode.upper():12s}: {mean_err:+.4f} ± {std_err:.4f} m")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• std_x_est and std_y_est show variation in estimated center within each trajectory")
    print("• std_x_across_modes and std_y_across_modes show disagreement between field modes")
    print("• Positive radius error means orbiting too far from center")
    print("• NN and RBF approximations may differ significantly from analytical field")
    print("• Real robot behavior may not match any simulation perfectly")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Make orbit_name available globally for plot function
    orbit_name = ""
    main()