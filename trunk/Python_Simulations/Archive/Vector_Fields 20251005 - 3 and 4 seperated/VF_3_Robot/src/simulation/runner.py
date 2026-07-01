"""
Simulation execution for OmniCluster.
"""
import numpy as np
import matplotlib.pyplot as plt
from ..fields.environments.grid_setup import make_environment_grid


def execute_omni_simulation(cluster, control_primitive, title, sim_time=150, ax=None):
    """
    Execute simulation for OmniCluster.

    Args:
        cluster: OmniCluster instance
        control_primitive: Function that takes cluster and returns (vx_c, vy_c)
        title: Title for plot
        sim_time: Number of simulation steps
        ax: Matplotlib axis (uses current if None)
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

    # Plot trajectory
    center_history = cluster.get_center_history()
    if len(center_history) > 0:
        ax.plot(center_history[:, 0], center_history[:, 1],
                marker='o', color='black', markersize=3, linewidth=1)

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
