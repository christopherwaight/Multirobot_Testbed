import numpy as np
import matplotlib.pyplot as plt



def saddle1(x, y):
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
    center_x = 0
    center_y = 0
    x_centered = x - center_x
    y_centered = y - center_y
    strength =1
    rotation = np.pi/4
    
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


def saddle2(x, y):
    """
    Create a hyperbolic saddle with optional rotation and normalized vectors.
    All vectors have equal magnitude.
    
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
        Vector field components (normalized)
    """
    center_x = 0
    center_y = 0
    x_centered = x - center_x
    y_centered = y - center_y
    strength = 1
    rotation = np.pi/4
    
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
    
    # Normalize the vectors to have constant magnitude
    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.where(magnitude == 0, 1e-6, magnitude)  # Avoid division by zero
    
    # Normalize to unit vectors (or multiply by desired constant magnitude)
    desired_magnitude = 1.0  # You can adjust this value
    u = desired_magnitude * u / magnitude
    v = desired_magnitude * v / magnitude
    
    return u, v

def saddle3(x, y):
    """
    Create a hyperbolic saddle with singularity at center that tapers off.
    Field strength is inversely proportional to distance from center.
    
    Parameters:
    -----------
    x, y : array-like
        Grid coordinates or point coordinates
    center_x, center_y : float
        Center position of the saddle point
    strength : float
        Strength of the saddle field
    rotation : float
        Rotation angle in radians
    
    Returns:
    --------
    u, v : array-like
        Vector field components
    """
    center_x = 0
    center_y = 0
    x_centered = x - center_x
    y_centered = y - center_y
    strength = 1
    rotation = np.pi/4
    
    # Apply rotation if specified
    if rotation != 0:
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        x_rot = cos_rot * x_centered - sin_rot * y_centered
        y_rot = sin_rot * x_centered + cos_rot * y_centered
        x_centered, y_centered = x_rot, y_rot
    
    # Calculate distance from center
    r = np.sqrt(x_centered**2 + y_centered**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Hyperbolic saddle equations with inverse distance scaling
    # This creates a singularity at the center that tapers off
    u = strength * x_centered / (r**2)  # Divide by r^2 for 1/r scaling
    v = -strength * y_centered / (r**2)
    
    # Optional: clip to prevent extreme values near singularity
    max_magnitude = 10.0  # Adjust this to control maximum vector size
    u = np.clip(u, -max_magnitude, max_magnitude)
    v = np.clip(v, -max_magnitude, max_magnitude)
    
    # Rotate back if needed
    if rotation != 0:
        cos_rot = np.cos(-rotation)
        sin_rot = np.sin(-rotation)
        u_rot = cos_rot * u - sin_rot * v
        v_rot = sin_rot * u + cos_rot * v
        u, v = u_rot, v_rot
    
    return u, v

# x = np.linspace(-5, 5, 20)
# y = np.linspace(-5, 5, 20)
# X, Y = np.meshgrid(x, y)

# # Calculate the vector field
# U, V = saddle3(X, Y)

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 8))

# # Plot the vector field using quiver
# ax.quiver(X, Y, U, V, color='blue', alpha=0.6)

# # Add streamlines for better visualization
# ax.streamplot(X, Y, U, V, color='red', linewidth=1, density=1.5)

# # Mark the sink center
# ax.plot(0, 0, 'ko', markersize=8, label='Sink Center')

# # Set equal aspect ratio and add grid
# ax.set_aspect('equal')
# ax.grid(True, alpha=0.3)

# # Labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Sink Vector Field')
# ax.legend()

# # Set axis limits
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)

# plt.show()