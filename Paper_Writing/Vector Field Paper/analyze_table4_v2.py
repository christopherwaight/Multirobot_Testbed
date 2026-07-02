#!/usr/bin/env python3
"""
Analyze and visualize orbital control performance data from Table 4
Compares Real Robot vs NN Simulation vs RBF Simulation for all configurations
Focus: Which simulation better matches REAL robot performance (not perfect tracking)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Complete data from Table 4 - All configurations
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

def calculate_actual_radius(commanded, error):
    """Calculate actual radius from commanded radius and error"""
    return commanded + error

def plot_configuration(config_name, config_data, ax):
    """Plot a single configuration comparison"""

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

    # Plot with error bars for all three methods
    ax.errorbar(r_cmd, real_actual, yerr=real_std,
                marker='o', label='Real Robot', color='black',
                linewidth=2.5, markersize=9, capsize=6, capthick=2)

    # Plot simulations with error bars
    ax.errorbar(r_cmd, nn_actual, yerr=nn_std,
                marker='s', label='NN Simulation',
                linewidth=2, markersize=8, alpha=0.9, color='tab:blue',
                capsize=5, capthick=1.5)
    ax.errorbar(r_cmd, rbf_actual, yerr=rbf_std,
                marker='^', label='RBF Simulation',
                linewidth=2, markersize=8, alpha=0.9, color='tab:orange',
                capsize=5, capthick=1.5)

    # Add perfect tracking line for reference
    ax.plot([0, max(r_cmd)], [0, max(r_cmd)], 'k--', alpha=0.3,
            label='Perfect tracking', linewidth=1)

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

    # Add annotations for which simulation is closer to real
    add_winner_annotations(ax, r_cmd, real_actual, nn_actual, rbf_actual)

def add_winner_annotations(ax, r_cmd, real_actual, nn_actual, rbf_actual):
    """Add small text annotations showing which simulation is closer"""
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

def calculate_statistics(config_name, config_data):
    """Calculate statistics for a configuration"""

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

    return {
        'nn_rmse': nn_rmse,
        'rbf_rmse': rbf_rmse,
        'nn_mae': nn_mae,
        'rbf_mae': rbf_mae,
        'nn_wins': nn_wins,
        'rbf_wins': rbf_wins,
        'total_points': len(r_cmd)
    }

def print_analysis():
    """Print comprehensive analysis for all configurations"""

    print("\n" + "="*70)
    print("SIMULATION FIDELITY ANALYSIS: MATCHING REAL ROBOT PERFORMANCE")
    print("="*70)

    configs = ['3robot', '4robot_square', '4robot_dual']
    config_names = ['3-Robot', '4-Robot Square', '4-Robot Dual Jacobian']

    all_stats = {}

    for config, name in zip(configs, config_names):
        stats = calculate_statistics(config, data[config])
        all_stats[config] = stats

        print(f"\n{name} Configuration:")
        print("-" * 50)

        print(f"  RMSE from Real Robot:")
        print(f"    NN:  {stats['nn_rmse']*1000:.1f} mm")
        print(f"    RBF: {stats['rbf_rmse']*1000:.1f} mm")
        winner = "NN" if stats['nn_rmse'] < stats['rbf_rmse'] else "RBF"
        diff = abs(stats['nn_rmse'] - stats['rbf_rmse']) * 1000
        print(f"    Winner: {winner} (by {diff:.1f} mm)")

        print(f"\n  Mean Absolute Error:")
        print(f"    NN:  {stats['nn_mae']*1000:.1f} mm")
        print(f"    RBF: {stats['rbf_mae']*1000:.1f} mm")

        print(f"\n  Point-by-Point Winners:")
        print(f"    NN closer: {stats['nn_wins']}/{stats['total_points']} radii")
        print(f"    RBF closer: {stats['rbf_wins']}/{stats['total_points']} radii")

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY:")
    print("="*70)

    total_nn_wins = sum(s['nn_wins'] for s in all_stats.values())
    total_rbf_wins = sum(s['rbf_wins'] for s in all_stats.values())
    total_points = sum(s['total_points'] for s in all_stats.values())

    print(f"\nAcross all configurations:")
    print(f"  NN closer to real: {total_nn_wins}/{total_points} points ({100*total_nn_wins/total_points:.1f}%)")
    print(f"  RBF closer to real: {total_rbf_wins}/{total_points} points ({100*total_rbf_wins/total_points:.1f}%)")

    avg_nn_rmse = np.mean([s['nn_rmse'] for s in all_stats.values()]) * 1000
    avg_rbf_rmse = np.mean([s['rbf_rmse'] for s in all_stats.values()]) * 1000

    print(f"\nAverage RMSE across configurations:")
    print(f"  NN:  {avg_nn_rmse:.1f} mm")
    print(f"  RBF: {avg_rbf_rmse:.1f} mm")

    if avg_nn_rmse < avg_rbf_rmse:
        print(f"\nConclusion: NN simulation provides superior sim-to-real transfer")
        print(f"            ({(avg_rbf_rmse/avg_nn_rmse - 1)*100:.1f}% better than RBF)")
    else:
        print(f"\nConclusion: RBF simulation provides superior sim-to-real transfer")
        print(f"            ({(avg_nn_rmse/avg_rbf_rmse - 1)*100:.1f}% better than NN)")

    print("="*70 + "\n")

def create_comparison_plots():
    """Create three comparison plots, one for each configuration"""

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot each configuration
    plot_configuration('3robot', data['3robot'], axes[0])
    plot_configuration('4robot_square', data['4robot_square'], axes[1])
    plot_configuration('4robot_dual', data['4robot_dual'], axes[2])

    # Overall title
    fig.suptitle('Orbital Control Performance: Real Robot vs Simulations\n(Annotations show which simulation is closer to real robot)',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    return fig

# Removed individual plots function - not needed

if __name__ == "__main__":
    print("\nAnalyzing Table 4: Orbital Control Performance")
    print("Comparing which simulation (NN or RBF) better matches REAL robot performance")

    # Create combined comparison plot only
    fig_combined = create_comparison_plots()
    plt.savefig('table4_all_configurations.png', dpi=150, bbox_inches='tight')
    print("\nSaved: table4_all_configurations.png")

    # Print detailed analysis
    print_analysis()

    # Show plot
    plt.show()

    print("\nAnalysis complete! Generated visualization and statistics.")
    print("Key finding: Annotations on plots show which simulation is closer at each radius.")