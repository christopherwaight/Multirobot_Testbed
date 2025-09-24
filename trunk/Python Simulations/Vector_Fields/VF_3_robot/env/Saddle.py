import numpy as np
import matplotlib.pyplot as plt

def true_saddle(x, y, plot=False):
    """
    Creates a true mathematical saddle point vector field.
    
    At a saddle point:
    - Flow approaches from two opposite directions
    - Flow moves away in two other opposite directions
    - The eigenvalues of the Jacobian have different signs
    
    Parameters:
    x, y: Grid coordinates or point coordinates
    plot: Boolean flag to plot the vector field (default: False)
    
    Returns:
    u, v: Vector field components
    """
    # Define saddle center
    x_center = 0
    y_center = 0
    
    # Calculate relative position from center
    x_rel = x - x_center
    y_rel = y - y_center
    
    # Create saddle flow:
    # - Flow approaches along x-axis (negative eigenvalue)
    # - Flow diverges along y-axis (positive eigenvalue)
    u = -x_rel  # Flow toward center along x-axis
    v = y_rel   # Flow away from center along y-axis
    
    # Optional: Scale the field for better visualization
    scale_factor = 0.5
    u = scale_factor * u
    v = scale_factor * v
    
    # Clip to prevent extreme values
    u = np.clip(u, -10, 10)
    v = np.clip(v, -10, 10)
    
    # Option to plot the vector field
    if plot and hasattr(x, 'shape') and len(x.shape) == 2:
        plt.figure(figsize=(10, 8))
        plt.quiver(x, y, u, v, scale=25)
        plt.plot(x_center, y_center, 'ro', markersize=8)  # Mark the saddle point
        plt.title('True Mathematical Saddle Point')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    return u, v



def rotated_saddle(x, y, plot=False):
    """
    Creates a rotated saddle point vector field.
    
    This variation rotates the saddle by 45 degrees, creating a flow that:
    - Approaches from northeast and southwest
    - Diverges toward northwest and southeast
    
    Parameters:
    x, y: Grid coordinates or point coordinates
    plot: Boolean flag to plot the vector field (default: False)
    
    Returns:
    u, v: Vector field components
    """
    # Define saddle center
    x_center = 5.0
    y_center = 5.0
    
    # Calculate relative position from center
    x_rel = x - x_center
    y_rel = y - y_center
    
    # Create rotated saddle (45 degrees)
    # This creates attractive flow along one diagonal and repulsive along the other
    u = y_rel - x_rel  # Flow along the -45° and 135° directions
    v = y_rel + x_rel  # Flow along the 45° and -135° directions
    
    # Normalize the strength
    scale_factor = 0.3
    u = scale_factor * u
    v = scale_factor * v
    
    # Clip to prevent extreme values
    u = np.clip(u, -10, 10)
    v = np.clip(v, -10, 10)
    
    # Option to plot the vector field
    if plot and hasattr(x, 'shape') and len(x.shape) == 2:
        plt.figure(figsize=(10, 8))
        plt.quiver(x, y, u, v, scale=25)
        plt.plot(x_center, y_center, 'ro', markersize=8)  # Mark the saddle point
        plt.title('Rotated Saddle Point (45°)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    return u, v


def complex_saddle(x, y, plot=False):
    """
    Creates a more complex saddle with adjustable parameters.
    
    This version allows control of:
    - Saddle center location
    - Saddle strength
    - Saddle orientation
    - Background flow
    
    Parameters:
    x, y: Grid coordinates or point coordinates
    plot: Boolean flag to plot the vector field (default: False)
    
    Returns:
    u, v: Vector field components
    """
    # Parameters
    x_center = 0
    y_center = 0
    saddle_strength = 0.5
    rotation_angle = np.pi/4  # 45 degrees
    background_flow_u = 0.0
    background_flow_v = 0.0
    
    # Calculate relative position from center
    x_rel = x - x_center
    y_rel = y - y_center
    
    # Apply rotation to create a saddle at any angle
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    
    # Rotate the coordinates
    x_rot = x_rel * cos_theta - y_rel * sin_theta
    y_rot = x_rel * sin_theta + y_rel * cos_theta
    
    # Create the saddle in the rotated coordinate system
    u_rot = -x_rot
    v_rot = y_rot
    
    # Rotate the vector field back to original coordinates
    u_saddle = u_rot * cos_theta + v_rot * sin_theta
    v_saddle = -u_rot * sin_theta + v_rot * cos_theta
    
    # Apply saddle strength
    u_saddle *= saddle_strength
    v_saddle *= saddle_strength
    
    # Add background flow
    u = u_saddle + background_flow_u
    v = v_saddle + background_flow_v
    
    # Clip to prevent extreme values
    u = np.clip(u, -10, 10)
    v = np.clip(v, -10, 10)
    
    # Option to plot the vector field
    if plot and hasattr(x, 'shape') and len(x.shape) == 2:
        plt.figure(figsize=(10, 8))
        plt.quiver(x, y, u, v, scale=25)
        plt.plot(x_center, y_center, 'ro', markersize=8)  # Mark the saddle point
        plt.title('Complex Saddle Point with Background Flow')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    return u, v


def hyperbolic_saddle_vector_field(x, y, center_x, center_y, strength=1.0, rotation=np.pi/4):
    """
    Create a hyperbolic saddle with optional rotation.
    
    Parameters:
    -----------
    x, y : array-like
        Grid coordinates or point coordinates
    center_x, center_y : float
        Center position of the saddle point
    strength : float
        Strength of the saddle field (default: 1.0)
    rotation : float
        Rotation angle in radians (default: pi/4)
    
    Returns:
    --------
    u, v : array-like
        Vector field components
    """
    x_centered = x - center_x
    y_centered = y - center_y
    
    # Apply rotation if specified
    if rotation != 0:
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        x_rot = cos_rot * x_centered - sin_rot * y_centered
        y_rot = sin_rot * x_centered + cos_rot * y_centered
        x_centered, y_centered = x_rot, y_rot
    
    # Hyperbolic saddle equations
    u = strength * x_centered
    v = -strength * y_centered
    
    # Rotate back if needed
    if rotation != 0:
        cos_rot = np.cos(-rotation)
        sin_rot = np.sin(-rotation)
        u_rot = cos_rot * u - sin_rot * v
        v_rot = sin_rot * u + cos_rot * v
        u, v = u_rot, v_rot
    
    return u, v