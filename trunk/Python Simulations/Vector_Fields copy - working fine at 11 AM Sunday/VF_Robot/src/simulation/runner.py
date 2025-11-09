"""
Simulation execution for robot clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
from ..fields.environments.grid_setup import make_environment_grid


def execute_omni_simulation(cluster, control_primitive, title, sim_time=150, ax=None, skip_legend=False):
    """
    Execute simulation for robot cluster (works with both 3-robot and 4-robot).

    Args:
        cluster: OmniCluster or QuadCluster instance
        control_primitive: Function that takes cluster and returns (vx_c, vy_c) or (vx_c, vy_c, omega_c)
        title: Title for plot
        sim_time: Number of simulation steps
        ax: Matplotlib axis (uses current if None)
        skip_legend: If True, skip adding legend (for multi-plot layouts)
    """
    if ax is None:
        ax = plt.gca()

    # Create environment grid
    X, Y = make_environment_grid()

    # Evaluate field on grid for visualization
    u_grid, v_grid = evaluate_field_on_grid(cluster.field, X, Y)

    # Plot vector field
    ax.quiver(X, Y, u_grid, v_grid, color='black', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot initial robot positions
    cluster.plot(ax)

    # Run simulation
    for i in range(sim_time):
        cluster.move(control_primitive)

    # Plot robot trajectories
    robot_history = cluster.get_robot_history()
    if len(robot_history) > 0:
        num_robots = robot_history.shape[1]
        robot_colors = ['blue', 'orange', 'green', 'red']

        for robot_idx in range(num_robots):
            trajectory = robot_history[:, robot_idx, :]
            color = robot_colors[robot_idx]

            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   color=color, linewidth=1.5, alpha=0.6, linestyle='--')

            # Mark start point
            ax.plot(trajectory[0, 0], trajectory[0, 1],
                   marker='o', color=color, markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Robot {robot_idx+1} Start')

            # Mark end point
            ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                   marker='s', color=color, markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f'Robot {robot_idx+1} End')

    # Plot centroid trajectory
    center_history = cluster.get_center_history()
    if len(center_history) > 0:
        # Trajectory line
        ax.plot(center_history[:, 0], center_history[:, 1],
                color='black', linewidth=2, alpha=0.8, label='Centroid Path')

        # Mark start point
        ax.plot(center_history[0, 0], center_history[0, 1],
               marker='*', color='lime', markersize=15,
               markeredgecolor='black', markeredgewidth=2,
               label='Centroid Start', zorder=10)

        # Mark end point
        ax.plot(center_history[-1, 0], center_history[-1, 1],
               marker='X', color='red', markersize=15,
               markeredgecolor='black', markeredgewidth=2,
               label='Centroid End', zorder=10)

    # Add legend (only if not skipping for multi-plot layouts)
    if not skip_legend:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9,
                 edgecolor='black', fancybox=False)

    # Plot final positions
    cluster.plot(ax)


def evaluate_field_on_grid(field, X, Y):
    """
    Evaluate a field on a grid for visualization.

    Args:
        field: VectorField instance
        X, Y: Meshgrid arrays

    Returns:
        (u, v): Field values on grid
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    u_flat = np.zeros_like(X_flat)
    v_flat = np.zeros_like(Y_flat)

    # Evaluate field at each grid point
    for i in range(len(X_flat)):
        u, v = field.get_value(X_flat[i], Y_flat[i])
        u_flat[i] = u
        v_flat[i] = v

    u = u_flat.reshape(shape)
    v = v_flat.reshape(shape)

    return u, v
