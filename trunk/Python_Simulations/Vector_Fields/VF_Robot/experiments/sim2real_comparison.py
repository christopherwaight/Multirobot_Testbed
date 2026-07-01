"""
sim2real_comparison.py - Simulation-to-Real Robot Comparison

Compares simulated robot behavior (Analytical, NN, RBF) against actual
physical robot data from MATLAB experiments.

Creates a 2x2 figure:
  - Subplot 1: Analytical field simulation
  - Subplot 2: Neural Network field simulation
  - Subplot 3: RBF Interpolator field simulation
  - Subplot 4: Real robot trajectory (from orbit040.mat)
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

# New architecture imports
from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField, NNField, RBFField
import src.control.primitives as ocp
from src.fields.environments.Vortex import vortex1

# ============================================================================
# CONFIGURATION
# ============================================================================

# Robot and simulation parameters
NUM_ROBOTS = 3
DESIRED_RADIUS = 0.4  # meters (matches physical robot experiment)
SIM_TIME = 600  # timesteps (60 seconds at 0.1s timestep)
TIMESTEP = 0.1  # seconds
FORMATION_CONFIG = "config/formations/equilateral_default.yaml"
CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"

# Environment
ENVIRONMENT = "vortex1"
PREDICTOR_DIR = "vortex_predictors"  # ML models for vortex1

# Real robot data
REAL_DATA_PATH = "../real_robot_data/orbit040_trajectory.csv"

# Visualization
FIGURE_SIZE = (14, 14)
AXIS_LIMITS = [-0.6, 0.6]  # meters
SAVE_FIGURE = True
OUTPUT_FILENAME = "sim2real_comparison_vortex1.png"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_field(mode):
    """
    Create field object based on mode.

    Args:
        mode: "analytical", "nn", or "rbf"

    Returns:
        VectorField instance
    """
    if mode == "analytical":
        return AnalyticalField(vortex1)
    elif mode == "nn":
        try:
            return NNField(PREDICTOR_DIR)
        except Exception as e:
            print(f"  Warning: NN model not found, falling back to analytical. Error: {e}")
            return AnalyticalField(vortex1)
    elif mode == "rbf":
        try:
            return RBFField(PREDICTOR_DIR)
        except Exception as e:
            print(f"  Warning: RBF model not found, falling back to analytical. Error: {e}")
            return AnalyticalField(vortex1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_simulation(field, mode_name):
    """
    Run a single simulation and return the cluster centroid trajectory.

    Args:
        field: VectorField object
        mode_name: String describing the mode (for progress output)

    Returns:
        center_points: Nx2 array of (x, y) positions
        diagnostics: List of diagnostic dictionaries
    """
    print(f"Running {mode_name} simulation...")

    # Create cluster
    cluster = OmniCluster(FORMATION_CONFIG, field)

    # Create control primitive wrapper with desired radius
    def control_primitive_with_radius(cluster):
        return ocp.critical_point_orbiter_plane_fitting(cluster, desired_radius=DESIRED_RADIUS)

    # Storage for center trajectory
    center_points = []

    # Run simulation
    for step in range(SIM_TIME):
        # Get cluster centroid
        x_c, y_c = cluster.get_centroid()
        center_points.append([x_c, y_c])

        # Move cluster using control primitive
        cluster.move(control_primitive_with_radius)

        # Progress indicator with diagnostics every 100 steps
        if step % 100 == 0 and step > 0:
            diag = cluster.get_diagnostics()[-1]
            print(f"  Step {step:4d}: r={diag['radius']:.3f}m, "
                  f"p={diag['p_current']:.3f}, q={diag['q_current']:.3f}, "
                  f"β={diag['beta_current']:.1f}°, "
                  f"J_cond={diag['jacobian_cond']:.1f}")

    print(f"  ✓ {mode_name} simulation complete")

    # Save diagnostics to CSV
    diagnostics = cluster.get_diagnostics()
    if len(diagnostics) > 0:
        # Save in the experiments directory
        csv_filename = os.path.join(os.path.dirname(__file__),
                                    f"diagnostics_{mode_name.lower().replace(' ', '_')}.csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = diagnostics[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diagnostics)
        print(f"  Saved diagnostics to {csv_filename}")

    return np.array(center_points), diagnostics


def load_real_robot_data():
    """
    Load real robot trajectory from CSV file.

    Returns:
        trajectory: Nx2 array of (x, y) positions
        time: N-element array of timestamps
    """
    print("Loading real robot data...")

    csv_path = os.path.join(os.path.dirname(__file__), REAL_DATA_PATH)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real robot data not found: {csv_path}\n"
                              f"Please run export_orbit040_to_csv.m and copy the CSV to real_robot_data/")

    # Load CSV
    data = pd.read_csv(csv_path)

    time = data['time_s'].values
    x = data['x_m'].values
    y = data['y_m'].values

    trajectory = np.column_stack([x, y])

    print(f"  Loaded {len(trajectory)} data points ({time[-1]:.1f} seconds)")
    print(f"  ✓ Real robot data loaded")

    return trajectory, time


def plot_vortex_field(ax):
    """
    Add vortex field quiver overlay to axis.

    Args:
        ax: Matplotlib axis object
    """
    # Create grid for quiver plot
    x = np.linspace(AXIS_LIMITS[0], AXIS_LIMITS[1], 15)
    y = np.linspace(AXIS_LIMITS[0], AXIS_LIMITS[1], 15)
    X, Y = np.meshgrid(x, y)

    # Calculate vortex field (matching Python vortex1)
    r = np.sqrt(X**2 + Y**2) + 1e-10
    theta = np.arctan2(Y, X)
    U = -r * np.sin(theta)
    V = r * np.cos(theta)

    # Normalize for visualization
    scale_factor = 0.5
    U_norm = U * scale_factor
    V_norm = V * scale_factor

    # Plot quiver
    ax.quiver(X, Y, U_norm, V_norm, color='lightgray', alpha=0.5,
              scale=8, scale_units='xy', width=0.003, zorder=1)


def plot_reference_circle(ax, radius, color='gray', linestyle='--', linewidth=1):
    """
    Add reference circle to axis.

    Args:
        ax: Matplotlib axis object
        radius: Circle radius in meters
        color: Line color
        linestyle: Line style
        linewidth: Line width
    """
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    ax.plot(x_circle, y_circle, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=0.6, zorder=2)


def plot_trajectory(ax, trajectory, title, color='blue', show_markers=True):
    """
    Plot a single trajectory on the given axis.

    Args:
        ax: Matplotlib axis object
        trajectory: Nx2 array of (x, y) positions
        title: Subplot title
        color: Trajectory line color
        show_markers: Whether to show start/end markers
    """
    # Add vortex field background
    plot_vortex_field(ax)

    # Add reference circle at desired radius
    plot_reference_circle(ax, DESIRED_RADIUS, color='black', linestyle='--', linewidth=1.5)

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color,
            linewidth=2, alpha=0.8, zorder=3, label='Trajectory')

    # Add start and end markers
    if show_markers:
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o',
                markersize=10, markerfacecolor='green', markeredgecolor='black',
                markeredgewidth=2, zorder=4, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's',
                markersize=10, markerfacecolor='red', markeredgecolor='black',
                markeredgewidth=2, zorder=4, label='End')

    # Configure axis
    ax.set_xlim(AXIS_LIMITS)
    ax.set_ylim(AXIS_LIMITS)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', fontsize=9)

    # Add text annotation with radius target
    ax.text(0.02, 0.98, f'Target radius: {DESIRED_RADIUS}m',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))


def calculate_trajectory_stats(trajectory):
    """
    Calculate statistics for a trajectory.

    Args:
        trajectory: Nx2 array of (x, y) positions

    Returns:
        dict with mean_radius, std_radius, min_radius, max_radius
    """
    radii = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)

    return {
        'mean_radius': np.mean(radii),
        'std_radius': np.std(radii),
        'min_radius': np.min(radii),
        'max_radius': np.max(radii),
        'radii': radii  # Keep for potential further analysis
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("SIM-TO-REAL COMPARISON: VORTEX ORBITER")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Robots: {NUM_ROBOTS}")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Control: {CONTROL_PRIMITIVE}")
    print(f"  Desired radius: {DESIRED_RADIUS}m")
    print(f"  Simulation time: {SIM_TIME * TIMESTEP:.0f}s ({SIM_TIME} steps)")
    print("=" * 70)
    print()

    # Run simulations
    print("Running simulations...")
    print()

    # 1. Analytical field
    field_analytical = create_field("analytical")
    trajectory_analytical, diag_analytical = run_simulation(field_analytical, "Analytical")
    print()

    # 2. Neural Network field
    field_nn = create_field("nn")
    trajectory_nn, diag_nn = run_simulation(field_nn, "Neural Network")
    print()

    # 3. RBF field
    field_rbf = create_field("rbf")
    trajectory_rbf, diag_rbf = run_simulation(field_rbf, "RBF Interpolator")
    print()

    # 4. Load real robot data
    trajectory_real, time_real = load_real_robot_data()
    print()

    # Calculate statistics
    print("Trajectory Statistics:")
    print("-" * 70)

    stats_analytical = calculate_trajectory_stats(trajectory_analytical)
    stats_nn = calculate_trajectory_stats(trajectory_nn)
    stats_rbf = calculate_trajectory_stats(trajectory_rbf)
    stats_real = calculate_trajectory_stats(trajectory_real)

    modes = ['Analytical', 'Neural Network', 'RBF', 'Real Robot']
    stats_list = [stats_analytical, stats_nn, stats_rbf, stats_real]

    print(f"{'Mode':<20} {'Mean (m)':<12} {'Std (m)':<12} {'Min (m)':<12} {'Max (m)':<12}")
    print("-" * 70)
    for mode, stats in zip(modes, stats_list):
        print(f"{mode:<20} {stats['mean_radius']:<12.4f} {stats['std_radius']:<12.4f} "
              f"{stats['min_radius']:<12.4f} {stats['max_radius']:<12.4f}")
    print("-" * 70)
    print()

    # Print formation stability analysis
    print("Formation Stability Analysis:")
    print("-" * 70)

    # Analyze last 100 timesteps for each simulation
    for mode_name, diagnostics in [("Analytical", diag_analytical),
                                   ("Neural Network", diag_nn),
                                   ("RBF", diag_rbf)]:
        if len(diagnostics) > 100:
            last_100 = diagnostics[-100:]
            avg_q = np.mean([d['q_current'] for d in last_100])
            std_q = np.std([d['q_current'] for d in last_100])
            avg_beta = np.mean([d['beta_current'] for d in last_100])
            std_beta = np.std([d['beta_current'] for d in last_100])
            avg_cond = np.mean([d['jacobian_cond'] for d in last_100])

            print(f"{mode_name:15s}: q={avg_q:.3f}±{std_q:.3f}, "
                  f"β={avg_beta:.1f}°±{std_beta:.1f}°, "
                  f"J_cond={avg_cond:.1f}")

    print("-" * 70)
    print()

    # Create comparison figure
    print("Creating comparison figure...")
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Subplot 1: Analytical
    ax1 = plt.subplot(2, 2, 1)
    plot_trajectory(ax1, trajectory_analytical, 'Analytical Field', color='#1f77b4')

    # Subplot 2: Neural Network
    ax2 = plt.subplot(2, 2, 2)
    plot_trajectory(ax2, trajectory_nn, 'Neural Network Field', color='#ff7f0e')

    # Subplot 3: RBF
    ax3 = plt.subplot(2, 2, 3)
    plot_trajectory(ax3, trajectory_rbf, 'RBF Interpolator Field', color='#2ca02c')

    # Subplot 4: Real Robot
    ax4 = plt.subplot(2, 2, 4)
    plot_trajectory(ax4, trajectory_real, 'Real Robot Data (orbit040)', color='#d62728')

    # Add overall title
    #fig.suptitle(f'Sim-to-Real Comparison: {ENVIRONMENT.upper()} Orbiter '
    #             f'(Target Radius: {DESIRED_RADIUS}m, Sim: {SIM_TIME*TIMESTEP:.0f}s)',
    #             fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    if SAVE_FIGURE:
        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure saved: {output_path}")

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
