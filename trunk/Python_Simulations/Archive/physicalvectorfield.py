import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# Define our domain
n = 100  # Grid resolution
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

# Define source points
point1 = np.array([0.5, 0.5])  # First point (x,y) 
point2 = np.array([0.2, 0.25])  # Second point (x,y)

# Initialize flow field arrays
U = np.zeros_like(X)
V = np.zeros_like(Y)

# Calculate distance from each grid point to the source points
for i in range(n):
    for j in range(n):
        grid_point = np.array([X[i, j], Y[i, j]])
        
        # Distance to point 1 (for U field)
        r1 = np.linalg.norm(grid_point - point1)
        # Avoid division by zero by adding a small value
        r1 = max(r1, 1e-10)
        # U values fall off like 1/r²
        U[i, j] = 1 / (r1**2)
        
        # Distance to point 2 (for V field)
        r2 = np.linalg.norm(grid_point - point2)
        r2 = max(r2, 1e-10)
        # V values fall off like 1/r²
        #V[i, j] = 1 / (r2**2)
        V[i, j] = 2- (1 / (r2**2)) - 0.1*U[i, j] 


# Normalize to prevent extremely large vectors
magnitude = np.sqrt(U**2 + V**2)
max_magnitude = np.max(magnitude)
U = U / max_magnitude
V = V / max_magnitude

# Create the plot
plt.figure(figsize=(10, 8))

# Plot the flow field with streamlines
plt.streamplot(X, Y, U, V, density=2, color=magnitude, cmap=plt.cm.viridis,
               linewidth=1, arrowsize=1, arrowstyle='->')

# Mark the source points
plt.plot(point1[0], point1[1], 'ro', markersize=10, label='Point 1 (U source)')
plt.plot(point2[0], point2[1], 'bo', markersize=10, label='Point 2 (V source)')

# Add colorbar and labels
plt.colorbar(label='Flow Magnitude')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Flow Field with 1/r² Falloff from Two Points')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.savefig('flow_field.png', dpi=300, bbox_inches='tight')
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib import cm

# # Define our domain
# n = 20  # Grid resolution - using fewer points for quiver to be readable
# x = np.linspace(0, 1, n)
# y = np.linspace(0, 1, n)
# X, Y = np.meshgrid(x, y)

# # Define source points
# point1 = np.array([0.2, 0.2])  # First point (x,y) 
# point2 = np.array([0.5, 0.5])  # Second point (x,y)

# # Initialize flow field arrays
# U = np.zeros_like(X)
# V = np.zeros_like(Y)

# # Calculate distance from each grid point to the source points
# for i in range(n):
#     for j in range(n):
#         grid_point = np.array([X[i, j], Y[i, j]])
        
#         # Distance to point 1 (for U field)
#         r1 = np.linalg.norm(grid_point - point1)
#         # Avoid division by zero by adding a small value
#         r1 = max(r1, 1e-10)
#         # U values fall off like 1/r²
#         U[i, j] = 1 / (r1**2)
        
#         # Distance to point 2 (for V field)
#         r2 = np.linalg.norm(grid_point - point2)
#         r2 = max(r2, 1e-10)
#         # V values fall off like 1/r²
#         V[i, j] = 1 / (r2**2)

# # Normalize to prevent extremely large vectors
# magnitude = np.sqrt(U**2 + V**2)
# max_magnitude = np.max(magnitude)
# U = U / max_magnitude
# V = V / max_magnitude

# # Create the plot
# plt.figure(figsize=(10, 8))

# # Plot the flow field with quiver plot - ADJUSTED FOR LARGER ARROWS
# q = plt.quiver(X, Y, U, V, magnitude, cmap=plt.cm.viridis, 
#                pivot='mid', scale=8, width=0.015, 
#                headwidth=4, headlength=5, headaxislength=4.5)

# # Mark the source points
# plt.plot(point1[0], point1[1], 'ro', markersize=10, label='Point 1 (U source)')
# plt.plot(point2[0], point2[1], 'bo', markersize=10, label='Point 2 (V source)')

# # Add colorbar and labels
# plt.colorbar(q, label='Flow Magnitude')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Flow Field with 1/r² Falloff from Two Points')
# plt.legend()
# plt.grid(True)
# plt.xlim(0, 1)
# plt.ylim(0, 1)

# plt.savefig('flow_field_quiver.png', dpi=300, bbox_inches='tight')
# plt.show()