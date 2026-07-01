import numpy as np
import matplotlib.pyplot as plt

def vortex1(x, y,):

    center_x = 0
    center_y = 0
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u = -r * np.sin(theta) #/ r
    v = r * np.cos(theta) #/ r
    return u, v

    
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)



def vortex2(x, y,):

    center_x = 0
    center_y = 0
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u =  np.sin(theta) #/ r
    v =-np.cos(theta) #/ r
    return u, v

    
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)


# Calculate the vector field
U, V = vortex2(X, Y)

def vortex3(x, y,):

    center_x = 0
    center_y = 0
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u =  np.sin(theta) / r
    v =-np.cos(theta) / r
    return u, v

    
# x = np.linspace(-5, 5, 20)
# y = np.linspace(-5, 5, 20)
# X, Y = np.meshgrid(x, y)


# # Calculate the vector field
# U, V = vortex3(X, Y)


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