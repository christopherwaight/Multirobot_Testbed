import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def deceptive_vector_field(x, y):
    """
    Creates a vector field with a vortex at (3,3) but with magnitude peaking at (1,1)
    This demonstrates why magnitude-only approaches can be misleading
    """
    # Distance from vortex center (3,3)
    x_vortex = x - 3
    y_vortex = y - 3
    r_vortex = np.sqrt(x_vortex**2 + y_vortex**2)
    r_vortex = np.where(r_vortex < 0.1, 0.1, r_vortex)  # Avoid division by zero
    
    # Create vortex flow (rotational component)
    v_theta = 1.0  # Angular velocity
    u_vortex = -v_theta * y_vortex / r_vortex
    v_vortex = v_theta * x_vortex / r_vortex
    
    # Create magnitude peak at (1,1)
    x_peak = x - 1
    y_peak = y - 1
    r_peak = np.sqrt(x_peak**2 + y_peak**2)
    
    # Create a magnitude multiplier that peaks at (1,1) and falls off with distance
    magnitude_multiplier = 2.0 * np.exp(-0.5 * r_peak**2)
    
    # Apply the magnitude multiplier while preserving the vortex flow direction
    u = u_vortex * magnitude_multiplier
    v = v_vortex * magnitude_multiplier
    
    return u, v

def plot_vector_field(save_filename=None):
    # Create grid
    x = np.linspace(0, 6, 30)
    y = np.linspace(0, 6, 30)
    X, Y = np.meshgrid(x, y)
    
    # Calculate vector field
    U, V = deceptive_vector_field(X, Y)
    
    # Calculate magnitude
    magnitude = np.sqrt(U**2 + V**2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Vector field with arrows
    ax1.set_title('Vector Field (Vortex at (3,3))')
    ax1.quiver(X, Y, U, V, scale=25)
    ax1.plot(3, 3, 'ro', markersize=8, label='Vortex Center')
    ax1.plot(1, 1, 'go', markersize=8, label='Magnitude Peak')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Magnitude as color map
    ax2.set_title('Magnitude Field (Peak at (1,1))')
    im = ax2.pcolormesh(X, Y, magnitude, cmap='viridis', shading='auto')
    ax2.quiver(X, Y, U, V, color='white', scale=25, alpha=0.5)
    ax2.plot(3, 3, 'ro', markersize=8, label='Vortex Center')
    ax2.plot(1, 1, 'go', markersize=8, label='Magnitude Peak')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 6)
    ax2.legend()
    ax2.set_aspect('equal')
    
    fig.colorbar(im, ax=ax2, label='Magnitude')
    
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()

def simulate_robot_behavior():
    """
    Simulates robot behavior with different control strategies to show
    how magnitude-only approach fails to find the vortex center
    """
    # Create grid
    x = np.linspace(0, 6, 30)
    y = np.linspace(0, 6, 30)
    X, Y = np.meshgrid(x, y)
    
    # Calculate vector field
    U, V = deceptive_vector_field(X, Y)
    magnitude = np.sqrt(U**2 + V**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot vector field background
    ax.quiver(X, Y, U, V, scale=25, alpha=0.3)
    ax.contour(X, Y, magnitude, levels=10, cmap='viridis', alpha=0.3)
    
    # Plot vortex center and magnitude peak
    ax.plot(3, 3, 'ro', markersize=10, label='Vortex Center')
    ax.plot(1, 1, 'go', markersize=10, label='Magnitude Peak')
    
    # Simulate magnitude-only control strategy (gradient ascent)
    start_pos_mag = np.array([5.0, 5.0])
    positions_mag = [start_pos_mag.copy()]
    current_pos = start_pos_mag.copy()
    
    for _ in range(50):
        # Sample field at current position
        x, y = current_pos
        u, v = deceptive_vector_field(x, y)
        mag = np.sqrt(u**2 + v**2)
        
        # Sample field at nearby points to estimate gradient
        epsilon = 0.1
        u1, v1 = deceptive_vector_field(x+epsilon, y)
        u2, v2 = deceptive_vector_field(x, y+epsilon)
        mag1 = np.sqrt(u1**2 + v1**2)
        mag2 = np.sqrt(u2**2 + v2**2)
        
        # Compute gradient direction
        grad_x = (mag1 - mag) / epsilon
        grad_y = (mag2 - mag) / epsilon
        
        # Normalize gradient
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        if grad_norm > 1e-10:
            grad_x /= grad_norm
            grad_y /= grad_norm
        
        # Move in gradient direction
        step_size = 0.1
        current_pos[0] += step_size * grad_x
        current_pos[1] += step_size * grad_y
        positions_mag.append(current_pos.copy())
    
    # Simulate direction-only control strategy (following vortex streamlines)
    start_pos_dir = np.array([5.0, 5.0])
    positions_dir = [start_pos_dir.copy()]
    current_pos = start_pos_dir.copy()
    
    for _ in range(200):
        # Sample field at current position
        x, y = current_pos
        u, v = deceptive_vector_field(x, y)
        
        # Normalize direction
        mag = np.sqrt(u**2 + v**2)
        if mag > 1e-10:
            u /= mag
            v /= mag
        
        # Add perpendicular component to spiral inward
        # (This simulates a vortex center-seeking behavior)
        u_perp = -v
        v_perp = u
        u_combined = 0.8 * u_perp + 0.2 * u
        v_combined = 0.8 * v_perp + 0.2 * v
        
        # Normalize
        mag_combined = np.sqrt(u_combined**2 + v_combined**2)
        if mag_combined > 1e-10:
            u_combined /= mag_combined
            v_combined /= mag_combined
        
        # Move in direction
        step_size = 0.05
        current_pos[0] += step_size * u_combined
        current_pos[1] += step_size * v_combined
        positions_dir.append(current_pos.copy())
    
    # Plot trajectories
    positions_mag = np.array(positions_mag)
    positions_dir = np.array(positions_dir)
    
    ax.plot(positions_mag[:, 0], positions_mag[:, 1], 'g-', linewidth=2, 
            label='Magnitude-Only Path')
    ax.plot(positions_dir[:, 0], positions_dir[:, 1], 'r-', linewidth=2,
            label='Direction-Based Path')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.set_title('Comparison of Control Strategies')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot the vector field
    plot_vector_field()
    
    # Simulate robot behavior with different control strategies
    simulate_robot_behavior()