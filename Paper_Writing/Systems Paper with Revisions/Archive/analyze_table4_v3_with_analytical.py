#!/usr/bin/env python3
"""
analyze_table4_v3_with_analytical.py - Table 4 Analysis with Analytical Simulation

Extends analyze_table4_v2.py to include analytical (no noise) simulation results.
Compares Real Robot vs NN Simulation vs RBF Simulation vs Analytical Simulation.

Shows how the ideal analytical simulation compares to both the real robot
performance and the ML-based approximations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to analytical data CSV (absolute path for reliability)
ANALYTICAL_DATA_PATH = "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/analytical_orbit_results.csv"

# Complete data from Table 4 - All configurations (from analyze_table4_v2.py)
data = {
    # 3-Robot Configuration
    '3robot': {
        'commanded_radius': [0.010, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700],
        'real': {
            'radius_error': [0.264, 0.193, 0.149, 0.091, 0.037, -0.032, -0.065, -0.123],
            'std': [0.020, 0.020, 0.014, 0.013, 0.018, 0.023, 0.038, 0.039]
        },
        'nn': {
            'radius_error': [0.150, 0.118, 0.109, 0.090, 0.067, 0.019, -0.027, -0.067],
            'std': [0.015, 0.013, 0.014, 0.018, 0.026, 0.039, 0.050, 0.060]
        },
        'rbf': {
            'radius_error': [0.084, 0.110, 0.084, 0.048, 0.077, 0.057, 0.032, 0.129],
            'std': [0.026, 0.026, 0.021, 0.025, 0.051, 0.061, 0.077, 0.011]
        }
    },

    # 4-Robot Square Configuration
    '4robot_square': {
        'commanded_radius': [0.010, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600],
        'real': {
            'radius_error': [0.264, 0.213, 0.137, 0.110, 0.054, -0.037, -0.102],
            'std': [0.011, 0.020, 0.026, 0.037, 0.041, 0.044, 0.044]
        },
        'nn': {
            'radius_error': [0.145, 0.118, 0.109, 0.090, 0.061, 0.019, -0.027],
            'std': [0.016, 0.016, 0.015, 0.023, 0.036, 0.051, 0.063]
        },
        'rbf': {
            'radius_error': [0.141, 0.110, 0.084, 0.034, 0.039, 0.017, 0.001],
            'std': [0.025, 0.026, 0.031, 0.033, 0.064, 0.087, 0.089]
        }
    },

    # 4-Robot Dual Jacobian Configuration
    '4robot_dual': {
        'commanded_radius': [0.010, 0.100, 0.200, 0.300, 0.400, 0.500],
        'real': {
            'radius_error': [0.111, 0.026, 0.129, 0.139, 0.110, 0.092],
            'std': [0.007, 0.009, 0.045, 0.039, 0.055, 0.057]
        },
        'nn': {
            'radius_error': [0.037, 0.015, 0.026, 0.087, 0.088, 0.052],
            'std': [0.006, 0.003, 0.012, 0.026, 0.026, 0.030]
        },
        'rbf': {
            'radius_error': [0.099, 0.031, 0.035, 0.070, 0.063, 0.044],
            'std': [0.031, 0.026, 0.035, 0.032, 0.046, 0.075]
        }
    }
}

def load_analytical_data():
    """Load analytical simulation results from CSV."""
    if not os.path.exists(ANALYTICAL_DATA_PATH):
        print(f"Warning: Analytical data file not found at {ANALYTICAL_DATA_PATH}")
        print("Please run generate_analytical_orbit_data.py first.")
        return None

    df = pd.read_csv(ANALYTICAL_DATA_PATH)

    # Organize data by robot type
    analytical_data = {
        '3robot': {'commanded_radius': [], 'radius_error': [], 'std': []},
        '4robot_square': {'commanded_radius': [], 'radius_error': [], 'std': []},
        '4robot_dual': {'commanded_radius': [], 'radius_error': [], 'std': []}
    }

    for _, row in df.iterrows():
        if row['robot_type'] == '3-robot':
            key = '3robot'
        elif row['robot_type'] == '4-robot-square':
            key = '4robot_square'
        elif row['robot_type'] == '4-robot-dual':
            key = '4robot_dual'
        else:
            continue

        analytical_data[key]['commanded_radius'].append(row['commanded_radius'])
        analytical_data[key]['radius_error'].append(row['radius_error'])
        analytical_data[key]['std'].append(row['std_radius'])

    # Convert lists to numpy arrays
    for key in analytical_data:
        for field in analytical_data[key]:
            analytical_data[key][field] = np.array(analytical_data[key][field])

    return analytical_data

def calculate_actual_radius(commanded, error):
    """Calculate actual radius from commanded radius and error"""
    return commanded + error

def plot_configuration(config_name, config_data, analytical_config_data, ax):
    """Plot a single configuration comparison including analytical results"""

    # Extract data
    r_cmd = np.array(config_data['commanded_radius'])

    # Real robot
    real_error = np.array(config_data['real']['radius_error'])
    real_std = np.array(config_data['real']['std'])
    real_actual = calculate_actual_radius(r_cmd, real_error)

    # NN simulation
    nn_error = np.array(config_data['nn']['radius_error'])
    nn_std = np.array(config_data['nn']['std'])
    nn_actual = calculate_actual_radius(r_cmd, nn_error)

    # RBF simulation
    rbf_error = np.array(config_data['rbf']['radius_error'])
    rbf_std = np.array(config_data['rbf']['std'])
    rbf_actual = calculate_actual_radius(r_cmd, rbf_error)

    # Plot real robot (thickest, most prominent)
    ax.errorbar(r_cmd, real_actual, yerr=real_std,
                marker='o', label='Real Robot', color='black',
                linewidth=2.5, markersize=9, capsize=6, capthick=2, zorder=4)

    # Plot simulations with error bars
    ax.errorbar(r_cmd, nn_actual, yerr=nn_std,
                marker='s', label='NN Simulation',
                linewidth=2, markersize=8, alpha=0.9, color='tab:blue',
                capsize=5, capthick=1.5, zorder=3)
    ax.errorbar(r_cmd, rbf_actual, yerr=rbf_std,
                marker='^', label='RBF Simulation',
                linewidth=2, markersize=8, alpha=0.9, color='tab:orange',
                capsize=5, capthick=1.5, zorder=3)

    # Add analytical simulation if data is available
    if analytical_config_data is not None and len(analytical_config_data['commanded_radius']) > 0:
        # Match analytical data to commanded radii
        analytical_actual = []
        analytical_std = []

        for cmd_r in r_cmd:
            # Find matching radius in analytical data
            mask = np.abs(analytical_config_data['commanded_radius'] - cmd_r) < 0.001
            if np.any(mask):
                idx = np.where(mask)[0][0]
                analytical_actual.append(
                    analytical_config_data['commanded_radius'][idx] +
                    analytical_config_data['radius_error'][idx]
                )
                analytical_std.append(analytical_config_data['std'][idx])
            else:
                # No matching radius, skip this point
                analytical_actual.append(np.nan)
                analytical_std.append(0.0)

        analytical_actual = np.array(analytical_actual)
        analytical_std = np.array(analytical_std)

        # Only plot non-NaN values
        valid_mask = ~np.isnan(analytical_actual)
        if np.any(valid_mask):
            ax.errorbar(r_cmd[valid_mask], analytical_actual[valid_mask],
                       yerr=analytical_std[valid_mask],
                       marker='d', label='Analytical Simulation',
                       linewidth=2, markersize=8, alpha=0.9, color='tab:green',
                       capsize=5, capthick=1.5, zorder=2)

    # Add perfect tracking line for reference
    ax.plot([0, max(r_cmd)], [0, max(r_cmd)], 'k--', alpha=0.3,
            label='Perfect tracking', linewidth=1, zorder=1)

    # Formatting
    ax.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax.set_ylabel('Actual Radius (m)', fontsize=12)

    # Set title based on configuration
    if config_name == '3robot':
        title = '3-Robot Configuration'
    elif config_name == '4robot_square':
        title = '4-Robot Square Configuration'
    else:
        title = '4-Robot Dual Jacobian Configuration'

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Use same axis limits for all plots for direct comparison
    ax.set_xlim([-0.05, 0.75])
    ax.set_ylim([-0.2, 0.8])

    # Set consistent tick marks
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Add annotations for which simulation is closer to real (NN vs RBF only)
    add_winner_annotations(ax, r_cmd, real_actual, nn_actual, rbf_actual)

def add_winner_annotations(ax, r_cmd, real_actual, nn_actual, rbf_actual):
    """Add small text annotations showing which simulation (NN or RBF) is closer"""
    for i, r in enumerate(r_cmd):
        nn_diff = abs(nn_actual[i] - real_actual[i])
        rbf_diff = abs(rbf_actual[i] - real_actual[i])

        # Place annotation above the real robot point
        y_pos = real_actual[i] + 0.04

        if nn_diff < rbf_diff:
            ax.text(r, y_pos, 'NN', ha='center', fontsize=8,
                   color='tab:blue', fontweight='bold')
        elif rbf_diff < nn_diff:
            ax.text(r, y_pos, 'RBF', ha='center', fontsize=8,
                   color='tab:orange', fontweight='bold')
        else:
            ax.text(r, y_pos, '=', ha='center', fontsize=8,
                   color='gray')

def calculate_statistics(config_name, config_data, analytical_config_data):
    """Calculate statistics for a configuration including analytical results"""

    r_cmd = np.array(config_data['commanded_radius'])
    real_error = np.array(config_data['real']['radius_error'])
    nn_error = np.array(config_data['nn']['radius_error'])
    rbf_error = np.array(config_data['rbf']['radius_error'])

    real_actual = calculate_actual_radius(r_cmd, real_error)
    nn_actual = calculate_actual_radius(r_cmd, nn_error)
    rbf_actual = calculate_actual_radius(r_cmd, rbf_error)

    # Calculate differences from real robot ACTUAL PERFORMANCE
    nn_diff = nn_actual - real_actual
    rbf_diff = rbf_actual - real_actual

    # RMSE
    nn_rmse = np.sqrt(np.mean(nn_diff**2))
    rbf_rmse = np.sqrt(np.mean(rbf_diff**2))

    # Mean absolute error
    nn_mae = np.mean(np.abs(nn_diff))
    rbf_mae = np.mean(np.abs(rbf_diff))

    # Count winners
    nn_wins = np.sum(np.abs(nn_diff) < np.abs(rbf_diff))
    rbf_wins = np.sum(np.abs(nn_diff) > np.abs(rbf_diff))

    stats = {
        'nn_rmse': nn_rmse,
        'rbf_rmse': rbf_rmse,
        'nn_mae': nn_mae,
        'rbf_mae': rbf_mae,
        'nn_wins': nn_wins,
        'rbf_wins': rbf_wins,
        'total_points': len(r_cmd)
    }

    # Add analytical statistics if available
    if analytical_config_data is not None and len(analytical_config_data['commanded_radius']) > 0:
        # Match analytical data to commanded radii
        analytical_errors = []
        for cmd_r in r_cmd:
            mask = np.abs(analytical_config_data['commanded_radius'] - cmd_r) < 0.001
            if np.any(mask):
                idx = np.where(mask)[0][0]
                analytical_errors.append(analytical_config_data['radius_error'][idx])
            else:
                analytical_errors.append(np.nan)

        analytical_errors = np.array(analytical_errors)
        valid_mask = ~np.isnan(analytical_errors)

        if np.any(valid_mask):
            analytical_actual = calculate_actual_radius(r_cmd[valid_mask], analytical_errors[valid_mask])
            real_actual_valid = real_actual[valid_mask]

            analytical_diff = analytical_actual - real_actual_valid
            stats['analytical_rmse'] = np.sqrt(np.mean(analytical_diff**2))
            stats['analytical_mae'] = np.mean(np.abs(analytical_diff))

    return stats

def print_analysis(analytical_data):
    """Print comprehensive analysis for all configurations"""

    print("\n" + "="*70)
    print("SIMULATION FIDELITY ANALYSIS WITH ANALYTICAL COMPARISON")
    print("="*70)

    configs = ['3robot', '4robot_square', '4robot_dual']
    config_names = ['3-Robot', '4-Robot Square', '4-Robot Dual Jacobian']

    all_stats = {}

    for config, name in zip(configs, config_names):
        analytical_config = analytical_data[config] if analytical_data else None
        stats = calculate_statistics(config, data[config], analytical_config)
        all_stats[config] = stats

        print(f"\n{name} Configuration:")
        print("-" * 50)

        print(f"  RMSE from Real Robot:")
        print(f"    NN:         {stats['nn_rmse']*1000:.1f} mm")
        print(f"    RBF:        {stats['rbf_rmse']*1000:.1f} mm")
        if 'analytical_rmse' in stats:
            print(f"    Analytical: {stats['analytical_rmse']*1000:.1f} mm")

        winner = "NN" if stats['nn_rmse'] < stats['rbf_rmse'] else "RBF"
        diff = abs(stats['nn_rmse'] - stats['rbf_rmse']) * 1000
        print(f"    NN vs RBF Winner: {winner} (by {diff:.1f} mm)")

        print(f"\n  Mean Absolute Error:")
        print(f"    NN:         {stats['nn_mae']*1000:.1f} mm")
        print(f"    RBF:        {stats['rbf_mae']*1000:.1f} mm")
        if 'analytical_mae' in stats:
            print(f"    Analytical: {stats['analytical_mae']*1000:.1f} mm")

        print(f"\n  Point-by-Point Winners (NN vs RBF):")
        print(f"    NN closer:  {stats['nn_wins']}/{stats['total_points']} radii")
        print(f"    RBF closer: {stats['rbf_wins']}/{stats['total_points']} radii")

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY:")
    print("="*70)

    total_nn_wins = sum(s['nn_wins'] for s in all_stats.values())
    total_rbf_wins = sum(s['rbf_wins'] for s in all_stats.values())
    total_points = sum(s['total_points'] for s in all_stats.values())

    print(f"\nAcross all configurations (NN vs RBF):")
    print(f"  NN closer to real:  {total_nn_wins}/{total_points} points ({100*total_nn_wins/total_points:.1f}%)")
    print(f"  RBF closer to real: {total_rbf_wins}/{total_points} points ({100*total_rbf_wins/total_points:.1f}%)")

    avg_nn_rmse = np.mean([s['nn_rmse'] for s in all_stats.values()]) * 1000
    avg_rbf_rmse = np.mean([s['rbf_rmse'] for s in all_stats.values()]) * 1000
    analytical_rmses = [s.get('analytical_rmse', np.nan) for s in all_stats.values()]
    if not all(np.isnan(analytical_rmses)):
        avg_analytical_rmse = np.nanmean(analytical_rmses) * 1000
    else:
        avg_analytical_rmse = None

    print(f"\nAverage RMSE across configurations:")
    print(f"  NN:         {avg_nn_rmse:.1f} mm")
    print(f"  RBF:        {avg_rbf_rmse:.1f} mm")
    if avg_analytical_rmse is not None:
        print(f"  Analytical: {avg_analytical_rmse:.1f} mm")

    if avg_nn_rmse < avg_rbf_rmse:
        print(f"\nConclusion: NN simulation provides superior sim-to-real transfer")
        print(f"            ({(avg_rbf_rmse/avg_nn_rmse - 1)*100:.1f}% better than RBF)")
    else:
        print(f"\nConclusion: RBF simulation provides superior sim-to-real transfer")
        print(f"            ({(avg_nn_rmse/avg_rbf_rmse - 1)*100:.1f}% better than NN)")

    if avg_analytical_rmse is not None:
        print(f"\nAnalytical simulation shows systematic offset from real robot")
        print(f"This is expected due to perfect conditions vs. real-world imperfections")

    print("="*70 + "\n")

def create_comparison_plots(analytical_data):
    """Create three comparison plots, one for each configuration"""

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot each configuration
    analytical_configs = {
        '3robot': analytical_data['3robot'] if analytical_data else None,
        '4robot_square': analytical_data['4robot_square'] if analytical_data else None,
        '4robot_dual': analytical_data['4robot_dual'] if analytical_data else None
    }

    plot_configuration('3robot', data['3robot'], analytical_configs['3robot'], axes[0])
    plot_configuration('4robot_square', data['4robot_square'], analytical_configs['4robot_square'], axes[1])
    plot_configuration('4robot_dual', data['4robot_dual'], analytical_configs['4robot_dual'], axes[2])

    # Overall title
    fig.suptitle('Orbital Control Performance: Real Robot vs Simulations (Including Analytical)\n(Annotations show which ML simulation is closer to real robot)',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    return fig

def plot_configuration_simple(config_name, config_data, analytical_config_data, ax):
    """Plot only Real Robot vs Analytical comparison (no NN/RBF)"""

    # Extract data
    r_cmd = np.array(config_data['commanded_radius'])

    # Real robot
    real_error = np.array(config_data['real']['radius_error'])
    real_std = np.array(config_data['real']['std'])
    real_actual = calculate_actual_radius(r_cmd, real_error)

    # Plot real robot (thick, prominent)
    ax.errorbar(r_cmd, real_actual, yerr=real_std,
                marker='o', label='Real Robot', color='black',
                linewidth=2.5, markersize=9, capsize=6, capthick=2, zorder=3)

    # Add analytical simulation if data is available
    if analytical_config_data is not None and len(analytical_config_data['commanded_radius']) > 0:
        # Match analytical data to commanded radii
        analytical_actual = []
        analytical_std = []

        for cmd_r in r_cmd:
            # Find matching radius in analytical data
            mask = np.abs(analytical_config_data['commanded_radius'] - cmd_r) < 0.001
            if np.any(mask):
                idx = np.where(mask)[0][0]
                analytical_actual.append(
                    analytical_config_data['commanded_radius'][idx] +
                    analytical_config_data['radius_error'][idx]
                )
                analytical_std.append(analytical_config_data['std'][idx])
            else:
                # No matching radius, skip this point
                analytical_actual.append(np.nan)
                analytical_std.append(0.0)

        analytical_actual = np.array(analytical_actual)
        analytical_std = np.array(analytical_std)

        # Only plot non-NaN values
        valid_mask = ~np.isnan(analytical_actual)
        if np.any(valid_mask):
            ax.errorbar(r_cmd[valid_mask], analytical_actual[valid_mask],
                       yerr=analytical_std[valid_mask],
                       marker='d', label='Analytical Simulation',
                       linewidth=2.5, markersize=9, alpha=0.9, color='tab:blue',
                       capsize=6, capthick=2, zorder=2)

    # Add perfect tracking line for reference
    ax.plot([0, max(r_cmd)], [0, max(r_cmd)], 'k--', alpha=0.3,
            label='Perfect tracking', linewidth=1, zorder=1)

    # Formatting
    ax.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax.set_ylabel('Actual Radius (m)', fontsize=12)

    # Set title based on configuration
    if config_name == '3robot':
        title = '3-Robot Configuration'
    elif config_name == '4robot_square':
        title = '4-Robot Square Configuration'
    else:
        title = '4-Robot Dual Jacobian Configuration'

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use same axis limits for all plots for direct comparison
    ax.set_xlim([-0.05, 0.75])
    ax.set_ylim([-0.2, 0.8])

    # Set consistent tick marks
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

def create_simple_comparison_plots(analytical_data):
    """Create comparison plots with only Real Robot vs Analytical"""

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot each configuration
    analytical_configs = {
        '3robot': analytical_data['3robot'] if analytical_data else None,
        '4robot_square': analytical_data['4robot_square'] if analytical_data else None,
        '4robot_dual': analytical_data['4robot_dual'] if analytical_data else None
    }

    plot_configuration_simple('3robot', data['3robot'], analytical_configs['3robot'], axes[0])
    plot_configuration_simple('4robot_square', data['4robot_square'], analytical_configs['4robot_square'], axes[1])
    plot_configuration_simple('4robot_dual', data['4robot_dual'], analytical_configs['4robot_dual'], axes[2])

    # Overall title
    fig.suptitle('Orbital Control Performance: Real Robot vs Analytical Simulation',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("\nAnalyzing Table 4 with Analytical Simulation Data")
    print("Comparing Real Robot vs NN vs RBF vs Analytical (no noise)")

    # Load analytical data
    analytical_data = load_analytical_data()

    if analytical_data is None:
        print("\nWarning: Proceeding without analytical data")
        print("Run generate_analytical_orbit_data.py to include analytical results")

    # Create combined comparison plot (with all simulations)
    fig_combined = create_comparison_plots(analytical_data)
    output_path = os.path.join(os.path.dirname(__file__), 'table4_all_configurations_with_analytical.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close(fig_combined)  # Close figure to free memory

    # Create simple comparison plot (Real vs Analytical only)
    if analytical_data is not None:
        fig_simple = create_simple_comparison_plots(analytical_data)
        output_path_simple = os.path.join(os.path.dirname(__file__), 'table4_real_vs_analytical_only.png')
        plt.savefig(output_path_simple, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path_simple}")
        plt.close(fig_simple)  # Close figure to free memory

    # Print detailed analysis
    print_analysis(analytical_data)

    # Show plot (commented out for batch processing)
    # plt.show()

    print("\nAnalysis complete! Generated visualization with analytical simulation data.")
    print("Key finding: Analytical simulation provides ideal baseline for comparison.")