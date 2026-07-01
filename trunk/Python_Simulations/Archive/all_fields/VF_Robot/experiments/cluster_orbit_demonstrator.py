"""
Cluster Orbit Demonstrator

Visual demonstration of four different control primitives orbiting around
a critical point in a vortex field. Each primitive is shown in a separate
subplot with the vector field background and robot trajectories.

Primitives tested:
1. vector_sum (3-robot)
2. critical_point_orbiter_plane_fitting (3-robot)
3. center_orbiter_quad_planar (4-robot)
4. center_orbiter_quad_advanced (4-robot)

Environment: Vortex 1
Orbital radius: 0.5
Simulation time: 120 steps
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

# Import robot clusters and field types
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField

# Import kinematics
from src.control.kinematics import inverse_kinematics
from src.control.quad_kinematics import inverse_kinematics_quad

# Import control primitives
import src.control.primitives as ocp
import src.control.quad_primitives as qcp

# Import environment
from src.fields.environments.Vortex import vortex1
from src.fields.environments.grid_setup import make_environment_grid


# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
SIM_TIME = 100  # Number of simulation steps
ORBITAL_RADIUS = 0.4  # Desired orbital radius
INITIAL_X = 0.0  # Starting x position
TIMESTEP = 0.1  # Time step in seconds

# Environment
ENVIRONMENT_FUNC = vortex1
ENVIRONMENT_NAME = "Vortex 1"

# Output directory
OUTPUT_DIR = "experiments/cluster_orbit_plots"

# Figure size
FIGURE_SIZE = (16, 16)

# Primitives to demonstrate
# Format: (name, function, num_robots, formation_config, display_name)
PRIMITIVES = [
    ("vector_sum", ocp.vector_sum, 3,
     "config/formations/equilateral_radius_sweep.yaml",
     "3R Vector Sum"),
    ("orbiter_plane", ocp.critical_point_orbiter_plane_fitting, 3,
     "config/formations/equilateral_radius_sweep.yaml",
     "3R Orbiter (Plane Fitting)"),
    ("orbiter_quad_planar", qcp.center_orbiter_quad_planar, 4,
     "config/formations/quad_square_default.yaml",
     "4R Orbiter (Planar)"),
    ("orbiter_quad_advanced", qcp.center_orbiter_quad_advanced, 4,
     "config/formations/quad_rectangle.yaml",
     "4R Orbiter (Advanced)"),
]


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def evaluate_field_on_grid(field, X, Y):
    """
    Evaluate field on a meshgrid for visualization.

    Args:
        field: VectorField instance
        X, Y: Meshgrid arrays

    Returns:
        (u_grid, v_grid): Field values on grid
    """
    u_grid = np.zeros_like(X)
    v_grid = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u, v = field.get_value(X[i, j], Y[i, j])
            u_grid[i, j] = u
            v_grid[i, j] = v

    return u_grid, v_grid


def run_orbit_simulation(primitive_func, num_robots, formation_config,
                         environment_func, desired_radius, initial_x):
    """
    Run orbital simulation for one primitive.

    Args:
        primitive_func: Control primitive function
        num_robots: 3 or 4
        formation_config: Path to formation YAML
        environment_func: Environment function
        desired_radius: Desired orbital radius
        initial_x: Starting x position

    Returns:
        dict: Contains cluster object and simulation data
    """
    # Create field
    field = AnalyticalField(environment_func)

    # Create cluster based on robot count
    if num_robots == 3:
        cluster = OmniCluster(formation_config, field)

        # Set initial position for 3-robot cluster
        initial_y = desired_radius
        theta_c_init = np.pi  # Initial orientation
        x1, y1, x2, y2, x3, y3 = inverse_kinematics(
            initial_x, initial_y, theta_c_init,
            cluster.desired_p, cluster.desired_beta, cluster.desired_q
        )
        # Update robot positions
        cluster.robots[0].position = np.array([x1, y1])
        cluster.robots[1].position = np.array([x2, y2])
        cluster.robots[2].position = np.array([x3, y3])
    else:
        cluster = QuadCluster(formation_config, field)

        # Set initial position for 4-robot cluster
        initial_y = desired_radius
        theta_c_init = np.pi  # Initial orientation
        x1, y1, x2, y2, x3, y3, x4, y4 = inverse_kinematics_quad(
            initial_x, initial_y, theta_c_init,
            cluster.desired_d1, cluster.desired_d2,
            cluster.desired_r1, cluster.desired_r2, cluster.desired_phi
        )
        # Update robot positions
        cluster.robots[0].position = np.array([x1, y1])
        cluster.robots[1].position = np.array([x2, y2])
        cluster.robots[2].position = np.array([x3, y3])
        cluster.robots[3].position = np.array([x4, y4])

    # Create wrapper function for control primitive with radius
    # Special case for vector_sum which doesn't take desired_radius
    if primitive_func == ocp.vector_sum or primitive_func == qcp.vector_sum_quad:
        control_wrapper = primitive_func
    else:
        def control_wrapper(cluster_arg):
            return primitive_func(cluster_arg, desired_radius=desired_radius)

    # Run simulation
    for step in range(SIM_TIME):
        cluster.move(control_wrapper)

    return {
        'cluster': cluster,
        'num_robots': num_robots,
    }


def plot_simulation_result(ax, result, environment_func, title):
    """
    Plot simulation results on given axis.

    Args:
        ax: Matplotlib axis
        result: Dictionary with cluster and simulation data
        environment_func: Environment function for field visualization
        title: Plot title
    """
    cluster = result['cluster']
    num_robots = result['num_robots']

    # Create environment grid
    X, Y = make_environment_grid()

    # Evaluate field on grid
    field = AnalyticalField(environment_func)
    u_grid, v_grid = evaluate_field_on_grid(field, X, Y)

    # Plot vector field
    ax.quiver(X, Y, u_grid, v_grid, color='black', alpha=0.4, scale=15)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Get robot history
    robot_history = cluster.get_robot_history()

    if len(robot_history) > 0:
        # Define colors for robots
        robot_colors = ['blue', 'orange', 'green', 'red']

        # Plot individual robot trajectories
        for robot_idx in range(num_robots):
            trajectory = robot_history[:, robot_idx, :]
            color = robot_colors[robot_idx]

            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   color=color, linewidth=1.5, alpha=0.7, linestyle='--')

            # Mark start point
            ax.plot(trajectory[0, 0], trajectory[0, 1],
                   marker='o', color=color, markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Robot {robot_idx+1} Start', zorder=10)

            # Mark end point
            ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                   marker='s', color=color, markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5,
                   zorder=10)

    # Plot centroid trajectory
    center_history = cluster.get_center_history()
    if len(center_history) > 0:
        # Trajectory line
        ax.plot(center_history[:, 0], center_history[:, 1],
                color='black', linewidth=3, alpha=0.9, label='Centroid Path', zorder=5)

        # Mark start point
        ax.plot(center_history[0, 0], center_history[0, 1],
               marker='*', color='lime', markersize=20,
               markeredgecolor='black', markeredgewidth=2,
               label='Centroid Start', zorder=15)

        # Mark end point
        ax.plot(center_history[-1, 0], center_history[-1, 1],
               marker='X', color='red', markersize=20,
               markeredgecolor='black', markeredgewidth=2,
               label='Centroid End', zorder=15)

    # Mark origin (critical point)
    ax.plot(0, 0, marker='P', color='magenta', markersize=15,
           markeredgecolor='black', markeredgewidth=2,
           label='Critical Point', zorder=20)

    # Add legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9,
             edgecolor='black', fancybox=False)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the cluster orbit demonstration."""
    print("=" * 80)
    print("CLUSTER ORBIT DEMONSTRATOR")
    print("=" * 80)
    print(f"Environment: {ENVIRONMENT_NAME}")
    print(f"Orbital radius: {ORBITAL_RADIUS} m")
    print(f"Simulation time: {SIM_TIME} steps ({SIM_TIME * TIMESTEP:.1f} seconds)")
    print(f"Primitives: {len(PRIMITIVES)}")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes = axes.flatten()

    # Run simulations for each primitive
    results = []
    for idx, (prim_key, prim_func, num_robots, formation_config, display_name) in enumerate(PRIMITIVES):
        print(f"Running: {display_name}...")

        # Run simulation
        result = run_orbit_simulation(
            prim_func, num_robots, formation_config,
            ENVIRONMENT_FUNC, ORBITAL_RADIUS, INITIAL_X
        )
        results.append(result)

        # Plot on corresponding subplot
        plot_simulation_result(axes[idx], result, ENVIRONMENT_FUNC, display_name)

        print(f"  Completed: {display_name}")

    # Adjust layout
    plt.tight_layout()

    # Save figure with fixed filename (will overwrite previous)
    output_filename = os.path.join(OUTPUT_DIR, "orbit_demonstration.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_filename}")

    # Also save individual plots
    for idx, (prim_key, prim_func, num_robots, formation_config, display_name) in enumerate(PRIMITIVES):
        # Create individual figure
        fig_individual = plt.figure(figsize=(10, 10))
        ax_individual = fig_individual.add_subplot(111)

        # Plot result
        plot_simulation_result(ax_individual, results[idx], ENVIRONMENT_FUNC, display_name)

        # Save individual plot with fixed filename (will overwrite previous)
        individual_filename = os.path.join(OUTPUT_DIR, f"{prim_key}.png")
        plt.savefig(individual_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"Saved: {individual_filename}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"Total plots saved: {len(PRIMITIVES) + 1}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
