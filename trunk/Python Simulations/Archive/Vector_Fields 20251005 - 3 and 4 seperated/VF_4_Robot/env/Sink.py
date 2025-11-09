import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def sink1(x, y):
    # Define sink center
    x_sink = 0
    y_sink = 0
    
    # Calculate distance from center
    x_centre = x - x_sink
    y_centre = y - y_sink
    
    # Calculate radial distance
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Create inward flow, strength increases as you get closer to center
    strength = -1.0  # Negative for sink (positive would make a source)
    v_r = strength * r  # Radial velocity
    
    # Convert to cartesian coordinates
    u = v_r * x_centre / r
    v = v_r * y_centre / r
    
    # Clip to prevent extreme values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v


def sink2(x, y):
    # Define sink center
    x_sink = 0
    y_sink = 0
    
    # Calculate distance from center
    x_centre = x - x_sink
    y_centre = y - y_sink
    
    # Calculate radial distance
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Create unit vectors pointing toward the center
    # Normalize the direction vector to have constant magnitude
    strength = -1.0  # Negative for sink (positive would make a source)
    
    # Create unit vectors in the radial direction
    u = strength * x_centre / r
    v = strength * y_centre / r
    
    return u, v


# def sink3(x, y):

#     center_x = 0
#     center_y = 0
#     # Calculate radius and angle
#     r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
#     theta = np.arctan2(y - center_y, x - center_x)
    
#     # Adjusting field components to create a "spinning plate" effect
#     u = - np.sin(theta) * r**2
#     v =   np.cos(theta) * r**2

#     # Calculating A sink
#     x_centre = x - center_x
#     y_centre = y - center_y
#     r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

#     # Sink
#     u =  -0.15*x_centre / r2
#     v =  -0.15*y_centre / r2

#     return u, v

def sink3(x, y, eps=0.2):
    """
    Smooth analogue of (sink2 - sink1):
      F(x,y) ≈ (1 - 1/r) * (x, y), but with r smoothed by eps.
    Behavior:
      - For eps < 1: inward near the origin (Jacobian negative), 
        zero ring at r* ≈ sqrt(1 - eps^2), outward beyond that.
    """
    x_c, y_c = x, y
    r = np.sqrt(x_c**2 + y_c**2)
    r_eps = np.sqrt(r**2 + eps**2)          # smooth |r|
    
    coeff = 1.0 - 1.0 / r_eps               # ~ 1 - 1/r
    u = coeff * x_c
    v = coeff * y_c
    return u, v



# def source1(x, y):
#     # Define sink center
#     x_sink = 0
#     y_sink = 0
    
#     # Calculate distance from center
#     x_centre = x - x_sink
#     y_centre = y - y_sink
    
#     # Calculate radial distance
#     r = np.sqrt(x_centre**2 + y_centre**2)
#     r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
#     # Create inward flow, strength increases as you get closer to center
#     strength = -1.0  # Negative for sink (positive would make a source)
#     v_r = strength * r  # Radial velocity
    
#     # Convert to cartesian coordinates
#     u = -v_r * x_centre / r
#     v = -v_r * y_centre / r
    
#     # Clip to prevent extreme values
#     u = np.clip(u, -100, 100)
#     v = np.clip(v, -100, 100)
    
#     return u, v


# def source2(x, y):
#     # Define sink center
#     x_sink = 0
#     y_sink = 0
    
#     # Calculate distance from center
#     x_centre = x - x_sink
#     y_centre = y - y_sink
    
#     # Calculate radial distance
#     r = np.sqrt(x_centre**2 + y_centre**2)
#     r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
#     # Create unit vectors pointing toward the center
#     # Normalize the direction vector to have constant magnitude
#     strength = 1.0  # Negative for sink (positive would make a source)
    
#     # Create unit vectors in the radial direction
#     u = strength * x_centre / r
#     v = strength * y_centre / r
    
#     return u, v



# def source3(x, y):

#     center_x = 0
#     center_y = 0
#     # Calculate radius and angle
#     r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
#     theta = np.arctan2(y - center_y, x - center_x)
    
#     # Adjusting field components to create a "spinning plate" effect
#     u = - np.sin(theta) * r**2
#     v =   np.cos(theta) * r**2

#     # Calculating A sink
#     x_centre = x - center_x
#     y_centre = y - center_y
#     r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

#     # Sink
#     u = 0.15*x_centre / r2**2
#     v = 0.15*y_centre / r2**2

#     return u, v


## Uncomment to pplot
# Create a grid of points
# x = np.linspace(-5, 5, 20)
# y = np.linspace(-5, 5, 20)
# X, Y = np.meshgrid(x, y)

# # Calculate the vector field
# U, V = source3(X, Y)

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