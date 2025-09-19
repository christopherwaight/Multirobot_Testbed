import numpy as np
import matplotlib.pyplot as plt

# Define jet parameters
velocity = 100  # m/s, maximum velocity at the nozzle
diameter = 0.5  # m, diameter of the nozzle
length = 2  # m, length of the jet to simulate
spread_rate = 4  # rate at which the jet spreads

# Create a grid
x = np.linspace(0, length, 100)
y = np.linspace(-length*spread_rate, length*spread_rate, 100)
X, Y = np.meshgrid(x, y)

# Calculate the spreading of the jet
spread = diameter/2 + spread_rate * X

# Gaussian velocity profile
U = velocity * np.exp(-((Y**2)/(2*spread**2)))

# Plot the jet flow
plt.figure(figsize=(10, 6))
plt.streamplot(X, Y, U, np.zeros_like(U), density=2)
plt.title('Jet Flow Simulation with Spreading')
plt.xlabel('Distance')
plt.ylabel('Width')
plt.show()
