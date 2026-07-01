"""
Center estimation plotting utilities for robot cluster simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_center_estimation(center_history, estimated_center_history, timestep=0.1,
                           title="Center Estimation Analysis", save_path=None, true_center=(0.0, 0.0)):
    """
    Plot 4 subplots showing center estimation metrics over time.

    Args:
        center_history: Array of centroid positions (timesteps, 2)
        estimated_center_history: Array of estimated center positions (timesteps, 2)
        timestep: Time step in seconds (default 0.1s)
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
        true_center: True center location (default (0, 0))
    """
    if len(center_history) == 0 or len(estimated_center_history) == 0:
        print("No center estimation history to plot")
        return

    # Convert to numpy arrays
    centroid_array = np.array(center_history)
    estimated_array = np.array(estimated_center_history)
    num_steps = len(centroid_array)

    # Create time array in seconds
    time = np.arange(num_steps) * timestep

    # Extract x and y components
    x_estimated = estimated_array[:, 0]
    y_estimated = estimated_array[:, 1]

    # Calculate distance from centroid to estimated center
    distance_to_estimated = np.linalg.norm(estimated_array - centroid_array, axis=1)

    # Calculate distance from centroid to origin (0, 0)
    true_center_array = np.array(true_center)
    distance_to_origin = np.linalg.norm(centroid_array - true_center_array, axis=1)

    # Create figure with 4x1 subplots (single column)
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: X position of estimated center
    ax = axes[0]
    ax.plot(time, x_estimated, 'b-', linewidth=2, label='X estimated')
    ax.axhline(y=true_center[0], color='r', linestyle='--', linewidth=2, label=f'True X ({true_center[0]})')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('X Position (m)', fontsize=11)
    ax.set_title('Estimated Center X Position', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Plot 2: Y position of estimated center
    ax = axes[1]
    ax.plot(time, y_estimated, 'g-', linewidth=2, label='Y estimated')
    ax.axhline(y=true_center[1], color='r', linestyle='--', linewidth=2, label=f'True Y ({true_center[1]})')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('Estimated Center Y Position', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Plot 3: Distance from centroid to estimated center
    ax = axes[2]
    ax.plot(time, distance_to_estimated, 'purple', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.set_title('Distance from Centroid to Estimated Center', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_dist = np.mean(distance_to_estimated)
    final_dist = distance_to_estimated[-1]
    ax.text(0.05, 0.95, f'Mean: {mean_dist:.4f} m\nFinal: {final_dist:.4f} m',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Distance from centroid to true center (0, 0)
    ax = axes[3]
    ax.plot(time, distance_to_origin, 'orange', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.set_title(f'Distance from Centroid to True Center ({true_center[0]}, {true_center[1]})',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_dist_origin = np.mean(distance_to_origin)
    final_dist_origin = distance_to_origin[-1]
    ax.text(0.05, 0.95, f'Mean: {mean_dist_origin:.4f} m\nFinal: {final_dist_origin:.4f} m',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Center estimation plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
