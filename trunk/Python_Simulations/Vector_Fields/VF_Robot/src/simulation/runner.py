"""
Simulation execution for robot clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
from ..fields.environments.grid_setup import make_environment_grid


def execute_omni_simulation(cluster, control_primitive, title, sim_time=2000, ax=None, skip_legend=False):
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

    # Create environment grid (pass field type for appropriate scaling)
    field_type = cluster.field.field_type
    X, Y = make_environment_grid(field_type=field_type)

    # Check if field is scalar or vector
    if cluster.field.field_type == 'scalar':
        # For scalar fields, plot contours instead of vector field
        phi_grid = evaluate_scalar_field_on_grid(cluster.field, X, Y)
        contour = ax.contour(X, Y, phi_grid, levels=15, colors='black', alpha=0.4, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
    else:
        # For vector fields, plot quiver
        u_grid, v_grid = evaluate_field_on_grid(cluster.field, X, Y)
        ax.quiver(X, Y, u_grid, v_grid, color='black', alpha=0.6)

    # Title and labels with larger font sizes
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    # Custom tick marks - scalar fields get larger range
    if cluster.field.field_type == 'scalar':
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    else:
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.65)
    ax.tick_params(labelsize=12)

    # Equal aspect ratio to prevent distortion
    ax.set_aspect('equal')

    # Plot initial robot positions
    cluster.plot(ax)

    # Reset the field clock so the simulation always starts at t=0,
    # even if the same field object is reused across multiple runs.
    if hasattr(cluster.field, "reset_clock"):
        cluster.field.reset_clock()

    # Run simulation
    for i in range(sim_time):
        cluster.move(control_primitive)
        # Advance the field's internal clock after each control step so the
        # field at step k was sampled at t = k * timestep.
        if hasattr(cluster.field, "step"):
            cluster.field.step(cluster.timestep)

    # Plot robot trajectories
    robot_history = cluster.get_robot_history()
    if len(robot_history) > 0:
        num_robots = robot_history.shape[1]
        robot_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

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
        # Original: ax.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='black', fancybox=False)
        # Move legend outside plot to avoid covering data
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8,
                 framealpha=0.9, edgecolor='black', fancybox=False)

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


def evaluate_scalar_field_on_grid(field, X, Y):
    """
    Evaluate scalar field on a grid for visualization.

    Args:
        field: ScalarField object
        X, Y: Meshgrid arrays

    Returns:
        phi: Scalar field values on grid
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    phi_flat = np.zeros_like(X_flat)

    # Evaluate field at each grid point
    for i in range(len(X_flat)):
        phi_flat[i] = field.get_value(X_flat[i], Y_flat[i])

    phi = phi_flat.reshape(shape)

    return phi
