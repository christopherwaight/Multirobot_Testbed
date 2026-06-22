"""
Monte Carlo analysis comparing control primitives across different field types.

Tests each primitive on 8 field types (6 analytical + 2 RBF) with 1000 random starting positions,
calculating average final position and RMSE.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Set random seed for repeatability
np.random.seed(42)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField, RBFField
from src.fields.environments.grid_setup import make_environment_grid
import src.control.primitives as ocp
import src.control.quad_primitives as qcp

# Import environments
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Spewing_Vortex import spewing_vortex1
from src.fields.environments.Saddle import saddle1


# Configuration
NUM_TRIALS = 1000
SIM_TIME = 100  # 10 seconds at 0.1s timestep
TRUE_CENTER = (0.0, 0.0)  # True center of all fields
SAVE_FAILED_TRIALS = True  # Save visualizations of failed trials
MAX_FAILED_PLOTS_PER_CASE = 3  # Maximum failed trials to plot per test case

# Environments to test (8 fields: 6 analytical + 2 RBF)
# Get absolute paths to predictor directories
script_dir = os.path.dirname(os.path.abspath(__file__))
vf_robot_dir = os.path.dirname(script_dir)
vortex_predictor_dir = os.path.join(vf_robot_dir, 'vortex_predictors')
saddle_predictor_dir = os.path.join(vf_robot_dir, 'saddle_predictors')

ENVIRONMENTS = {
    'sink1': sink1,
    'source1': source1,
    'vortex1': vortex1,
    'sinking_vortex1': sinking_vortex1,
    'spewing_vortex1': spewing_vortex1,
    'saddle1': saddle1,
    'vortex1_rbf': vortex_predictor_dir,  # RBF approximation of vortex
    'saddle1_rbf': saddle_predictor_dir   # RBF approximation of saddle
}

# Control primitives to test (3-robot only)
PRIMITIVES = {
    'critical_point_plane_fitting': {
        'num_robots': 3,
        'primitive': ocp.critical_point_plane_fitting,
        'formation_config': 'config/formations/equilateral_default.yaml'
    }
}


def run_single_trial(field_func, primitive_config, start_x, start_y, save_trajectory=False):
    """
    Run a single trial with specified starting position.

    Args:
        field_func: Vector field function (callable) or RBF predictor directory (string)
        primitive_config: Dictionary with primitive configuration
        start_x: Starting x position
        start_y: Starting y position
        save_trajectory: If True, return cluster object with trajectory history

    Returns:
        If save_trajectory=False: Final centroid position (x, y), or (None, None) if trial failed
        If save_trajectory=True: (final_x, final_y, cluster, failure_step) where failure_step=-1 if succeeded
    """
    try:
        # Create field (analytical or RBF)
        if isinstance(field_func, str):
            # RBF field - field_func is a directory path
            field = RBFField(field_func)
        else:
            # Analytical field - field_func is a callable
            field = AnalyticalField(field_func)

        # Create cluster (3-robot or 4-robot based on primitive)
        num_robots = primitive_config['num_robots']
        formation_config = primitive_config['formation_config']

        # Suppress initialization prints
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            if num_robots == 3:
                cluster = OmniCluster(formation_config, field)
            else:
                cluster = QuadCluster(formation_config, field)

        # Reset to starting position (RANDOM for each trial)
        cluster.reset(x_c=start_x, y_c=start_y)

        # Run simulation
        primitive = primitive_config['primitive']
        failure_step = -1
        for step in range(SIM_TIME):
            cluster.move(primitive)

            # Check for divergence
            centroid = cluster.get_centroid()
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                failure_step = step
                if save_trajectory:
                    return None, None, cluster, failure_step
                return None, None
            if np.abs(centroid[0]) > 10.0 or np.abs(centroid[1]) > 10.0:
                # Robot diverged too far
                failure_step = step
                if save_trajectory:
                    return None, None, cluster, failure_step
                return None, None

        # Get final centroid position
        final_position = cluster.get_centroid()

        # Final NaN check
        if np.isnan(final_position[0]) or np.isnan(final_position[1]):
            failure_step = SIM_TIME
            if save_trajectory:
                return None, None, cluster, failure_step
            return None, None

        if save_trajectory:
            return final_position[0], final_position[1], cluster, -1
        return final_position[0], final_position[1]

    except Exception as e:
        # If any error occurs, return None
        if save_trajectory:
            return None, None, None, 0
        return None, None


def save_failed_trial_plot(cluster, field_func, env_name, prim_name, start_x, start_y,
                           failure_step, trial_num, project_root):
    """
    Save visualization of a failed trial.

    Args:
        cluster: Cluster object with trajectory history
        field_func: Vector field function
        env_name: Environment name
        prim_name: Primitive name
        start_x, start_y: Starting position
        failure_step: Step at which failure occurred
        trial_num: Trial number
        project_root: Project root directory
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create environment grid
        X, Y = make_environment_grid()

        # Evaluate field on grid
        shape = X.shape
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        u_flat = np.zeros_like(X_flat)
        v_flat = np.zeros_like(Y_flat)

        for i in range(len(X_flat)):
            u, v = field_func(X_flat[i], Y_flat[i])
            u_flat[i] = u
            v_flat[i] = v

        u_grid = u_flat.reshape(shape)
        v_grid = v_flat.reshape(shape)

        # Plot vector field
        ax.quiver(X, Y, u_grid, v_grid, color='black', alpha=0.3)

        # Plot trajectory
        center_history = cluster.get_center_history()
        if len(center_history) > 0:
            ax.plot(center_history[:, 0], center_history[:, 1],
                   'r-', linewidth=2, label='Centroid Trajectory', alpha=0.8)

            # Mark start
            ax.plot(start_x, start_y, 'go', markersize=15,
                   markeredgecolor='black', markeredgewidth=2,
                   label=f'Start ({start_x:.3f}, {start_y:.3f})', zorder=10)

            # Mark failure point
            ax.plot(center_history[-1, 0], center_history[-1, 1],
                   'rX', markersize=20, markeredgecolor='black', markeredgewidth=2,
                   label=f'Failed at step {failure_step}', zorder=10)

        # Plot robot trajectories
        robot_history = cluster.get_robot_history()
        if len(robot_history) > 0:
            num_robots = robot_history.shape[1]
            colors = ['blue', 'orange', 'green', 'red']
            for robot_idx in range(num_robots):
                trajectory = robot_history[:, robot_idx, :]
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=colors[robot_idx], linewidth=1, alpha=0.4, linestyle='--')

        # Mark true center
        ax.plot(0, 0, 'k*', markersize=20, markeredgecolor='yellow',
               markeredgewidth=2, label='True Center (0, 0)', zorder=5)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'FAILED: {env_name} - {prim_name}\nTrial {trial_num}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Save plot
        output_dir = os.path.join(project_root, "monte_carlo_results", "failed_trials")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{env_name}_{prim_name}_trial{trial_num}_failed.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        # If plotting fails, just skip it
        plt.close('all')
        pass


def calculate_rmse(final_positions, true_center):
    """
    Calculate RMSE from true center.

    Args:
        final_positions: List of (x, y) tuples
        true_center: True center position (x, y)

    Returns:
        RMSE value
    """
    positions_array = np.array(final_positions)
    true_center_array = np.array(true_center)

    # Calculate squared errors
    errors = positions_array - true_center_array
    squared_errors = np.sum(errors**2, axis=1)

    # RMSE
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse


def run_monte_carlo_analysis():
    """
    Run full Monte Carlo analysis across all environments and primitives.
    """
    # Get absolute path to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("=" * 80)
    print("Monte Carlo Analysis: Control Primitive Comparison")
    print("=" * 80)
    print(f"Number of trials per test case: {NUM_TRIALS}")
    print(f"Simulation time: {SIM_TIME} steps (10 seconds)")
    print(f"True center: {TRUE_CENTER}")
    print(f"Starting positions: RANDOM in [-0.5, 0.5] × [-0.5, 0.5]")
    print(f"Save failed trials: {SAVE_FAILED_TRIALS} (up to {MAX_FAILED_PLOTS_PER_CASE} per case)")
    print()

    # Results storage
    results = []

    # Test each environment
    for env_name, env_func in ENVIRONMENTS.items():
        print(f"\n{'='*80}")
        print(f"Environment: {env_name}")
        print(f"{'='*80}")

        # Test each primitive
        for prim_name, prim_config in PRIMITIVES.items():
            print(f"\n  Primitive: {prim_name} ({prim_config['num_robots']} robots)")
            print(f"  Running {NUM_TRIALS} trials...", end='', flush=True)

            # Run trials
            final_positions = []
            failed_trials = 0
            failed_plots_saved = 0
            for trial in range(NUM_TRIALS):
                # Random starting position in [-0.5, 0.5] × [-0.5, 0.5]
                start_x = np.random.uniform(-0.5, 0.5)
                start_y = np.random.uniform(-0.5, 0.5)

                # Run trial (save trajectory if we want to plot failures)
                save_traj = SAVE_FAILED_TRIALS and failed_plots_saved < MAX_FAILED_PLOTS_PER_CASE

                if save_traj:
                    final_x, final_y, cluster, failure_step = run_single_trial(
                        env_func, prim_config, start_x, start_y, save_trajectory=True)
                else:
                    final_x, final_y = run_single_trial(env_func, prim_config, start_x, start_y)
                    cluster = None
                    failure_step = -1

                # Check if trial succeeded
                if final_x is None or final_y is None:
                    failed_trials += 1

                    # Save failed trial plot if enabled
                    if SAVE_FAILED_TRIALS and failed_plots_saved < MAX_FAILED_PLOTS_PER_CASE and cluster is not None:
                        save_failed_trial_plot(cluster, env_func, env_name, prim_name,
                                             start_x, start_y, failure_step, trial + 1, project_root)
                        failed_plots_saved += 1
                else:
                    final_positions.append((final_x, final_y))

                # Progress indicator
                if (trial + 1) % 10 == 0:
                    print(f" {trial + 1}", end='', flush=True)

            print(" Done!")

            # Check if we have any successful trials
            if len(final_positions) == 0:
                print(f"  ⚠️  ALL TRIALS FAILED (0/{NUM_TRIALS} succeeded)")
                if failed_plots_saved > 0:
                    print(f"  📊 Saved {failed_plots_saved} failed trial visualizations")
                avg_x, avg_y = np.nan, np.nan
                std_x, std_y = np.nan, np.nan
                radial_std = np.nan
                rmse = np.nan
                systematic_bias = np.nan
            else:
                # Calculate statistics
                positions_array = np.array(final_positions)
                avg_x = np.mean(positions_array[:, 0])
                avg_y = np.mean(positions_array[:, 1])
                std_x = np.std(positions_array[:, 0])
                std_y = np.std(positions_array[:, 1])
                # Radial standard deviation (combined metric) - this is precision
                radial_std = np.sqrt(std_x**2 + std_y**2)
                rmse = calculate_rmse(final_positions, TRUE_CENTER)
                # Systematic bias: distance from true center (0,0) to mean rest point
                systematic_bias = np.sqrt(avg_x**2 + avg_y**2)

                # Display results
                success_rate = len(final_positions) / NUM_TRIALS * 100
                if failed_trials > 0:
                    print(f"  ⚠️  {failed_trials} trials failed ({success_rate:.1f}% success rate)")
                    if failed_plots_saved > 0:
                        print(f"  📊 Saved {failed_plots_saved} failed trial visualizations")
                print(f"  Average final position: ({avg_x:.6f}, {avg_y:.6f})")
                print(f"  Systematic bias: {systematic_bias:.6f} m")
                print(f"  Precision (radial std): {radial_std:.6f} m")
                print(f"  RMSE from true center: {rmse:.6f} m")

            # Store results
            results.append({
                'environment': env_name,
                'primitive': prim_name,
                'num_robots': prim_config['num_robots'],
                'avg_x': avg_x,
                'avg_y': avg_y,
                'std_x': std_x,
                'std_y': std_y,
                'radial_std': radial_std,
                'systematic_bias': systematic_bias,
                'rmse': rmse,
                'successful_trials': len(final_positions),
                'failed_trials': failed_trials,
                'success_rate': len(final_positions) / NUM_TRIALS * 100
            })

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Environment':<20} {'Primitive':<35} {'Success%':<10} {'RMSE':<12} {'σ_radial':<12}")
    print("-"*100)

    for result in results:
        success_str = f"{result['success_rate']:.1f}%" if result['success_rate'] > 0 else "FAILED"
        rmse_str = f"{result['rmse']:.6f}" if not np.isnan(result['rmse']) else "N/A"
        std_str = f"{result['radial_std']:.6f}" if not np.isnan(result['radial_std']) else "N/A"
        print(f"{result['environment']:<20} "
              f"{result['primitive']:<35} "
              f"{success_str:<10} "
              f"{rmse_str:<12} "
              f"{std_str:<12}")

    # Save results to CSV (absolute path)
    output_dir = os.path.join(project_root, "monte_carlo_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results_8fields_1000runs.csv")

    with open(output_file, 'w') as f:
        # Header
        f.write("Environment,Primitive,NumRobots,SuccessfulTrials,FailedTrials,SuccessRate,"
               f"AvgX,AvgY,StdX,StdY,SystematicBias,Precision,RMSE\n")

        # Data
        for result in results:
            f.write(f"{result['environment']},"
                   f"{result['primitive']},"
                   f"{result['num_robots']},"
                   f"{result['successful_trials']},"
                   f"{result['failed_trials']},"
                   f"{result['success_rate']:.2f},"
                   f"{result['avg_x']:.6f},"
                   f"{result['avg_y']:.6f},"
                   f"{result['std_x']:.6f},"
                   f"{result['std_y']:.6f},"
                   f"{result['systematic_bias']:.6f},"
                   f"{result['radial_std']:.6f},"
                   f"{result['rmse']:.6f}\n")

    print(f"\nResults saved to: {output_file}")

    if SAVE_FAILED_TRIALS:
        failed_trials_dir = os.path.join(project_root, "monte_carlo_results", "failed_trials")
        print(f"Failed trial visualizations saved to: {failed_trials_dir}")

    # Print LaTeX-formatted table for paper
    print("\n" + "="*100)
    print("LATEX TABLE FOR PAPER (Table II)")
    print("="*100)
    print("Copy the following rows into Paper_Draft_1A.tex Table II:")
    print()

    # Define field type names for LaTeX table
    field_names = {
        'sink1': 'Stable Node (Sink)',
        'source1': 'Unstable Node (Source)',
        'vortex1': 'Center (Vortex)',
        'sinking_vortex1': 'Stable Spiral (Sinking Vortex)',
        'spewing_vortex1': 'Unstable Spiral (Spewing Vortex)',
        'saddle1': 'Saddle Point'
    }

    # Group results by environment
    for env_name in ENVIRONMENTS.keys():
        env_results = [r for r in results if r['environment'] == env_name]

        # Print field type header
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{field_names[env_name]}}}}} \\\\")
        print("\\hline")

        # Sort by number of robots (3 first, then 4)
        env_results_sorted = sorted(env_results, key=lambda x: x['num_robots'])

        for result in env_results_sorted:
            cluster_size = f"{result['num_robots']}-Robot"
            avg_pos = f"({result['avg_x']:.4f}, {result['avg_y']:.4f})"
            sys_bias = f"{result['systematic_bias']:.4f}"
            precision = f"{result['radial_std']:.4f}"

            print(f" & {cluster_size} & {avg_pos} & {sys_bias} & {precision} \\\\")

        print("\\hline")

    print()
    print("Table structure:")
    print("Columns: Environment | Cluster Size | Avg. Rest Point (x, y) m | Systematic Bias (m) | Precision σ (m)")
    print()

    # Print best performers
    print("\n" + "="*80)
    print("BEST PERFORMERS (Lowest RMSE)")
    print("="*80)

    for env_name in ENVIRONMENTS.keys():
        env_results = [r for r in results if r['environment'] == env_name]
        # Filter out failed results (NaN RMSE)
        valid_results = [r for r in env_results if not np.isnan(r['rmse'])]

        if len(valid_results) == 0:
            print(f"{env_name:<20} → NO SUCCESSFUL PRIMITIVES")
        else:
            best = min(valid_results, key=lambda x: x['rmse'])
            print(f"{env_name:<20} → {best['primitive']:<35} (RMSE: {best['rmse']:.6f})")

    # Print worst performers
    print("\n" + "="*80)
    print("WORST PERFORMERS (Highest RMSE)")
    print("="*80)

    for env_name in ENVIRONMENTS.keys():
        env_results = [r for r in results if r['environment'] == env_name]
        # Filter out failed results (NaN RMSE)
        valid_results = [r for r in env_results if not np.isnan(r['rmse'])]

        if len(valid_results) == 0:
            print(f"{env_name:<20} → NO SUCCESSFUL PRIMITIVES")
        else:
            worst = max(valid_results, key=lambda x: x['rmse'])
            print(f"{env_name:<20} → {worst['primitive']:<35} (RMSE: {worst['rmse']:.6f})")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility (comment out for true randomness)
    # np.random.seed(42)  # COMMENTED OUT - now truly random!

    # Run analysis
    run_monte_carlo_analysis()
