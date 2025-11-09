"""
Velocity plotting utilities for robot cluster simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_robot_velocities(velocity_history, timestep=0.1, title="Robot Velocities",
                          save_path=None, num_robots=3):
    """
    Plot velocity magnitude vs time for each robot.

    Args:
        velocity_history: List of velocity arrays for each timestep
                         Each element is shape (num_robots, 2) with [vx, vy] for each robot
        timestep: Time step in seconds (default 0.1s)
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
        num_robots: Number of robots (3 or 4)
    """
    if len(velocity_history) == 0:
        print("No velocity history to plot")
        return

    # Convert to numpy array: shape (timesteps, num_robots, 2)
    vel_array = np.array(velocity_history)
    num_steps = len(vel_array)

    # Create time array in seconds
    time = np.arange(num_steps) * timestep

    # Compute velocity magnitudes for each robot
    vel_magnitudes = np.linalg.norm(vel_array, axis=2)  # shape (timesteps, num_robots)

    # Create figure
    fig, axes = plt.subplots(num_robots + 1, 1, figsize=(10, 3 * (num_robots + 1)))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Robot colors
    colors = ['blue', 'orange', 'green', 'red']
    robot_names = ['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4']

    # Plot individual robot velocities
    for i in range(num_robots):
        ax = axes[i]
        ax.plot(time, vel_magnitudes[:, i], color=colors[i], linewidth=2, label=robot_names[i])
        ax.set_ylabel('Velocity (m/s)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{robot_names[i]} Velocity Magnitude', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # Plot all robots together on last subplot
    ax_all = axes[num_robots]
    for i in range(num_robots):
        ax_all.plot(time, vel_magnitudes[:, i], color=colors[i],
                   linewidth=2, label=robot_names[i], alpha=0.8)
    ax_all.set_ylabel('Velocity (m/s)', fontsize=10)
    ax_all.set_xlabel('Time (s)', fontsize=10)
    ax_all.set_title('All Robots - Velocity Magnitude Comparison', fontsize=11)
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Velocity plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_velocity_components(velocity_history, timestep=0.1, title="Robot Velocity Components",
                            save_path=None, num_robots=3):
    """
    Plot vx and vy components separately for each robot.

    Args:
        velocity_history: List of velocity arrays for each timestep
        timestep: Time step in seconds (default 0.1s)
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
        num_robots: Number of robots (3 or 4)
    """
    if len(velocity_history) == 0:
        print("No velocity history to plot")
        return

    # Convert to numpy array: shape (timesteps, num_robots, 2)
    vel_array = np.array(velocity_history)
    num_steps = len(vel_array)

    # Create time array in seconds
    time = np.arange(num_steps) * timestep

    # Create figure with subplots for each robot
    fig, axes = plt.subplots(num_robots, 2, figsize=(14, 3 * num_robots))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Robot colors
    colors = ['blue', 'orange', 'green', 'red']
    robot_names = ['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4']

    # Handle case of single robot (axes won't be 2D)
    if num_robots == 1:
        axes = axes.reshape(1, -1)

    # Plot vx and vy for each robot
    for i in range(num_robots):
        # vx component
        ax_vx = axes[i, 0]
        ax_vx.plot(time, vel_array[:, i, 0], color=colors[i], linewidth=2)
        ax_vx.set_ylabel('vx (m/s)', fontsize=10)
        ax_vx.set_xlabel('Time (s)', fontsize=10)
        ax_vx.set_title(f'{robot_names[i]} - X Velocity', fontsize=11)
        ax_vx.grid(True, alpha=0.3)
        ax_vx.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # vy component
        ax_vy = axes[i, 1]
        ax_vy.plot(time, vel_array[:, i, 1], color=colors[i], linewidth=2)
        ax_vy.set_ylabel('vy (m/s)', fontsize=10)
        ax_vy.set_xlabel('Time (s)', fontsize=10)
        ax_vy.set_title(f'{robot_names[i]} - Y Velocity', fontsize=11)
        ax_vy.grid(True, alpha=0.3)
        ax_vy.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Velocity components plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
