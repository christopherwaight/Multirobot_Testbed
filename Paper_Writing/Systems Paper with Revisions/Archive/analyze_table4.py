#!/usr/bin/env python3
"""
Analyze and visualize orbital control performance data from Table 4
Compares Real Robot, NN Simulation, and RBF Simulation performance
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Data from Table 4 - 3-Robot Configuration
# Format: radius_commanded, radius_error, std_dev, center_x, center_y
data = {
    'commanded_radius': [0.010, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700],

    'real_robot': {
        'radius_error': [0.264, 0.193, 0.149, 0.091, 0.037, -0.032, -0.065, -0.123],
        'std': [0.020, 0.020, 0.014, 0.013, 0.018, 0.023, 0.038, 0.039],
        'center': [(-0.007, -0.008), (0.012, 0.008), (0.018, 0.001), (0.037, -0.003),
                   (0.044, -0.019), (0.053, 0.037), (0.010, 0.003), (0.016, -0.006)]
    },

    'nn_simulation': {
        'radius_error': [0.150, 0.118, 0.109, 0.090, 0.067, 0.019, -0.027, -0.067],
        'std': [0.015, 0.013, 0.014, 0.018, 0.026, 0.039, 0.050, 0.060],
        'center': [(0.011, -0.016), (0.012, -0.008), (-0.000, -0.009), (0.019, -0.002),
                   (0.038, 0.045), (0.030, 0.015), (0.044, 0.074), (0.106, 0.065)]
    },

    'rbf_simulation': {
        'radius_error': [0.084, 0.110, 0.084, 0.048, 0.077, 0.057, 0.032, 0.129],
        'std': [0.026, 0.026, 0.021, 0.025, 0.051, 0.061, 0.077, 0.011],
        'center': [(0.022, -0.028), (0.010, -0.024), (-0.011, -0.018), (-0.012, 0.011),
                   (0.062, 0.085), (0.098, 0.046), (0.123, 0.058), (-0.041, 0.828)]
    }
}

# Also include 4-Robot configurations if needed
data_4robot = {
    'commanded_radius': [0.010, 0.100, 0.200, 0.300, 0.400, 0.500],

    'real_robot_square': {
        'radius_error': [0.264, 0.213, 0.137, 0.110, 0.054, -0.037],
        'std': [0.011, 0.020, 0.026, 0.037, 0.041, 0.044],
    },

    'real_robot_dual': {
        'radius_error': [0.111, 0.026, 0.129, 0.139, 0.110, 0.092],
        'std': [0.007, 0.009, 0.045, 0.039, 0.055, 0.057],
    }
}

def calculate_actual_radius(commanded, error):
    """Calculate actual radius from commanded radius and error"""
    return commanded + error

def plot_orbital_performance():
    """Create comprehensive plot comparing real robot vs simulations"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    r_cmd = np.array(data['commanded_radius'])

    # Real robot data
    real_error = np.array(data['real_robot']['radius_error'])
    real_std = np.array(data['real_robot']['std'])
    real_actual = calculate_actual_radius(r_cmd, real_error)

    # NN simulation
    nn_error = np.array(data['nn_simulation']['radius_error'])
    nn_std = np.array(data['nn_simulation']['std'])
    nn_actual = calculate_actual_radius(r_cmd, nn_error)

    # RBF simulation
    rbf_error = np.array(data['rbf_simulation']['radius_error'])
    rbf_std = np.array(data['rbf_simulation']['std'])
    rbf_actual = calculate_actual_radius(r_cmd, rbf_error)

    # Plot 1: Actual radius vs commanded with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(r_cmd, real_actual, yerr=real_std, marker='o', label='Real Robot',
                 linewidth=2, markersize=8, capsize=5)
    ax1.errorbar(r_cmd, nn_actual, yerr=nn_std, marker='s', label='NN Simulation',
                 linewidth=2, markersize=7, capsize=5, alpha=0.8)
    ax1.errorbar(r_cmd, rbf_actual, yerr=rbf_std, marker='^', label='RBF Simulation',
                 linewidth=2, markersize=7, capsize=5, alpha=0.8)
    ax1.plot(r_cmd, r_cmd, 'k--', alpha=0.5, label='Perfect tracking')
    ax1.set_xlabel('Commanded Radius (m)', fontsize=11)
    ax1.set_ylabel('Actual Radius (m)', fontsize=11)
    ax1.set_title('Orbital Radius Tracking Performance', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.05, 0.75])

    # Plot 2: Radius error comparison
    ax2 = axes[0, 1]
    ax2.plot(r_cmd, real_error, 'o-', label='Real Robot', linewidth=2, markersize=8)
    ax2.plot(r_cmd, nn_error, 's-', label='NN Simulation', linewidth=2, markersize=7)
    ax2.plot(r_cmd, rbf_error, '^-', label='RBF Simulation', linewidth=2, markersize=7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Commanded Radius (m)', fontsize=11)
    ax2.set_ylabel('Radius Error (m)', fontsize=11)
    ax2.set_title('Radius Tracking Error', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Absolute error comparison (which simulation is closer to real?)
    ax3 = axes[1, 0]
    nn_diff_from_real = np.abs(nn_error - real_error)
    rbf_diff_from_real = np.abs(rbf_error - real_error)

    x_pos = np.arange(len(r_cmd))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, nn_diff_from_real, width, label='NN vs Real',
                    color='tab:blue', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, rbf_diff_from_real, width, label='RBF vs Real',
                    color='tab:orange', alpha=0.7)

    ax3.set_xlabel('Commanded Radius (m)', fontsize=11)
    ax3.set_ylabel('|Simulation Error - Real Error| (m)', fontsize=11)
    ax3.set_title('Simulation Fidelity: Which is Closer to Real Robot?', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{r:.2f}' for r in r_cmd])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add winner annotations
    for i, (nn_diff, rbf_diff) in enumerate(zip(nn_diff_from_real, rbf_diff_from_real)):
        if nn_diff < rbf_diff:
            ax3.text(i, max(nn_diff, rbf_diff) + 0.005, 'NN', ha='center', fontsize=9,
                    color='blue', fontweight='bold')
        else:
            ax3.text(i, max(nn_diff, rbf_diff) + 0.005, 'RBF', ha='center', fontsize=9,
                    color='orange', fontweight='bold')

    # Plot 4: Standard deviation comparison
    ax4 = axes[1, 1]
    ax4.plot(r_cmd, real_std * 1000, 'o-', label='Real Robot', linewidth=2, markersize=8)
    ax4.plot(r_cmd, nn_std * 1000, 's-', label='NN Simulation', linewidth=2, markersize=7)
    ax4.plot(r_cmd, rbf_std * 1000, '^-', label='RBF Simulation', linewidth=2, markersize=7)
    ax4.set_xlabel('Commanded Radius (m)', fontsize=11)
    ax4.set_ylabel('Standard Deviation (mm)', fontsize=11)
    ax4.set_title('Tracking Consistency (Lower is Better)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Table 4: Orbital Control Performance Analysis (3-Robot Configuration)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig

def calculate_statistics():
    """Calculate detailed statistics comparing NN vs RBF to real robot"""

    r_cmd = np.array(data['commanded_radius'])
    real_error = np.array(data['real_robot']['radius_error'])
    nn_error = np.array(data['nn_simulation']['radius_error'])
    rbf_error = np.array(data['rbf_simulation']['radius_error'])

    # Calculate differences from real robot
    nn_diff = nn_error - real_error
    rbf_diff = rbf_error - real_error

    # Calculate RMSE
    nn_rmse = np.sqrt(np.mean(nn_diff**2))
    rbf_rmse = np.sqrt(np.mean(rbf_diff**2))

    # Calculate correlation
    nn_corr = np.corrcoef(nn_error, real_error)[0, 1]
    rbf_corr = np.corrcoef(rbf_error, real_error)[0, 1]

    # Count which is closer at each radius
    nn_wins = np.sum(np.abs(nn_diff) < np.abs(rbf_diff))
    rbf_wins = np.sum(np.abs(nn_diff) > np.abs(rbf_diff))
    ties = np.sum(np.abs(nn_diff) == np.abs(rbf_diff))

    print("\n" + "="*60)
    print("SIMULATION FIDELITY ANALYSIS")
    print("="*60)

    print("\n1. RMSE from Real Robot Performance:")
    print(f"   NN Simulation:  {nn_rmse*1000:.2f} mm")
    print(f"   RBF Simulation: {rbf_rmse*1000:.2f} mm")
    print(f"   Winner: {'NN' if nn_rmse < rbf_rmse else 'RBF'} (by {abs(nn_rmse-rbf_rmse)*1000:.2f} mm)")

    print("\n2. Correlation with Real Robot:")
    print(f"   NN Simulation:  {nn_corr:.4f}")
    print(f"   RBF Simulation: {rbf_corr:.4f}")
    print(f"   Winner: {'NN' if nn_corr > rbf_corr else 'RBF'}")

    print("\n3. Point-by-Point Comparison:")
    print(f"   NN closer to real:  {nn_wins} radii")
    print(f"   RBF closer to real: {rbf_wins} radii")
    print(f"   Ties: {ties}")

    print("\n4. Detailed Radius Analysis:")
    print("   Radius | Real Error | NN Error | RBF Error | Closer")
    print("   " + "-"*55)
    for i, r in enumerate(r_cmd):
        nn_d = abs(nn_diff[i])
        rbf_d = abs(rbf_diff[i])
        closer = "NN" if nn_d < rbf_d else ("RBF" if rbf_d < nn_d else "TIE")
        print(f"   {r:.3f}  | {real_error[i]:+.3f}     | {nn_error[i]:+.3f}   | {rbf_error[i]:+.3f}    | {closer}")

    print("\n5. Performance at Key Radii:")
    print(f"   Small radius (0.01m): {'NN' if abs(nn_diff[0]) < abs(rbf_diff[0]) else 'RBF'} closer")
    print(f"   Medium radius (0.3m): {'NN' if abs(nn_diff[3]) < abs(rbf_diff[3]) else 'RBF'} closer")
    print(f"   Large radius (0.7m):  {'NN' if abs(nn_diff[-1]) < abs(rbf_diff[-1]) else 'RBF'} closer")

    # Calculate average absolute difference
    nn_avg_diff = np.mean(np.abs(nn_diff))
    rbf_avg_diff = np.mean(np.abs(rbf_diff))

    print(f"\n6. Average Absolute Difference from Real:")
    print(f"   NN Simulation:  {nn_avg_diff*1000:.2f} mm")
    print(f"   RBF Simulation: {rbf_avg_diff*1000:.2f} mm")
    print(f"   Winner: {'NN' if nn_avg_diff < rbf_avg_diff else 'RBF'} (by {abs(nn_avg_diff-rbf_avg_diff)*1000:.2f} mm)")

    print("\n" + "="*60)
    print("CONCLUSION:")
    if nn_rmse < rbf_rmse and nn_avg_diff < rbf_avg_diff:
        print("NN simulation provides superior sim-to-real transfer.")
    elif rbf_rmse < nn_rmse and rbf_avg_diff < nn_avg_diff:
        print("RBF simulation provides superior sim-to-real transfer.")
    else:
        print("Mixed results - each method has strengths at different radii.")
    print("="*60 + "\n")

    return {
        'nn_rmse': nn_rmse,
        'rbf_rmse': rbf_rmse,
        'nn_corr': nn_corr,
        'rbf_corr': rbf_corr,
        'nn_wins': nn_wins,
        'rbf_wins': rbf_wins
    }

def plot_4robot_comparison():
    """Additional plot comparing 3-robot vs 4-robot configurations"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Data for comparison
    r_cmd_3 = np.array(data['commanded_radius'])
    r_cmd_4 = np.array(data_4robot['commanded_radius'])

    real_3robot = calculate_actual_radius(r_cmd_3, np.array(data['real_robot']['radius_error']))
    real_4square = calculate_actual_radius(r_cmd_4, np.array(data_4robot['real_robot_square']['radius_error']))
    real_4dual = calculate_actual_radius(r_cmd_4, np.array(data_4robot['real_robot_dual']['radius_error']))

    # Plot
    ax.plot(r_cmd_3, real_3robot, 'o-', label='3-Robot', linewidth=2, markersize=8)
    ax.plot(r_cmd_4, real_4square, 's-', label='4-Robot Square', linewidth=2, markersize=7)
    ax.plot(r_cmd_4, real_4dual, '^-', label='4-Robot Dual Jacobian', linewidth=2, markersize=7)
    ax.plot([0, 0.7], [0, 0.7], 'k--', alpha=0.5, label='Perfect tracking')

    ax.set_xlabel('Commanded Radius (m)', fontsize=12)
    ax.set_ylabel('Actual Radius (m)', fontsize=12)
    ax.set_title('Robot Configuration Comparison (Real Hardware)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Analyzing Table 4: Orbital Control Performance")
    print("=" * 60)

    # Create main visualization
    fig1 = plot_orbital_performance()
    plt.savefig('table4_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: table4_analysis.png")

    # Calculate and print statistics
    stats = calculate_statistics()

    # Create 4-robot comparison
    fig2 = plot_4robot_comparison()
    plt.savefig('robot_configuration_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: robot_configuration_comparison.png")

    # Show plots
    plt.show()

    print("\nAnalysis complete! Generated visualizations and statistics.")