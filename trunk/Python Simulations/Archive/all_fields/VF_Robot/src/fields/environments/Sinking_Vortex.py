import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.



def sinking_vortex1(x, y):
    """
    Sinking vortex field with spinning plate effect.
    Combines rotational vortex with radial sink component.
    Field strength is zero at center and increases with distance.
    """
    # Define center (matching the original parameters)
    center_x = 0
    center_y = 0
    
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Vortex component - scale by r instead of r^2 for zero at center
    u_vortex = - np.sin(theta) * r
    v_vortex =   np.cos(theta) * r
    
    # Calculating sink component
    x_centre = x - center_x
    y_centre = y - center_y
    
    # Sink that also goes to zero at center (scale by r)
    # Using unit radial direction multiplied by r
    r_safe = np.where(r == 0, 1e-10, r)
    u_sink = -x_centre / r_safe * r  # This simplifies to -x_centre
    v_sink = -y_centre / r_safe * r  # This simplifies to -y_centre
    
    # Combine the two fields
    u = 0.4 * u_vortex + 0.15 * u_sink
    v = 0.4 * v_vortex + 0.15 * v_sink
    
    # The field naturally goes to zero at the center now
    # since both components are proportional to r
    
    return u, v


def sinking_vortex2(x, y):
    """
    Sinking vortex field with spinning plate effect - normalized version.
    Combines rotational vortex with radial sink component.
    All vectors have equal magnitude.
    """
    # Define center (matching the original parameters)
    center_x = 0
    center_y = 0
    
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u = - np.sin(theta) * r**2
    v =   np.cos(theta) * r**2

    # Calculating A sink
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    # Sink
    u1 = -x_centre / r2**2
    v1 = -y_centre / r2**2

    # Combine the two fields
    u = 0.4*u + .15*u1
    v = 0.4*v + .15*v1
    
    # Normalize to constant magnitude
    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.where(magnitude == 0, 1e-10, magnitude)  # Avoid division by zero
    
    # Set desired constant magnitude for all vectors
    desired_magnitude = 1.0  # Adjust this value as needed
    u = desired_magnitude * u / magnitude
    v = desired_magnitude * v / magnitude
    
    return u, v

def sinking_vortex3(x, y):
    """
    Sinking vortex field with spinning plate effect.
    Combines rotational vortex with radial sink component.
    """
    # Define center (matching the original parameters)
    center_x = 0
    center_y = 0
    

    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u = - np.sin(theta) * r**2
    v =   np.cos(theta) * r**2

    # Calculating A sink
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    # Sink
    u1 = -x_centre / r2**2
    v1 = -y_centre / r2**2

    u = 0.4*u + .15*u1
    v = 0.4*v + .15*v1
    return u, v
 
# x = np.linspace(-0.5, 0.5, 20)
# y = np.linspace(-0.5, 0.5, 20)
# X, Y = np.meshgrid(x, y)


# # Calculate the vector field
# U, V = sinking_vortex3(X, Y)


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
# ax.set_title('Sinking Vortex Vector Field')
# ax.legend()

# # Set axis limits
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)

# plt.show()