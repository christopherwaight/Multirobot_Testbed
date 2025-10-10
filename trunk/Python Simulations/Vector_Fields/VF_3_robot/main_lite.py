# main.py

## Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster import RobotCluster
from exec_sim.simulation import execute_simulation

# Import single environment function for testing
from env.Sink import sink3 as enviro 

## Main - Execute single simulation

print("=" * 60)
print("ROBOT CLUSTER SIMULATION - SINGLE ENVIRONMENT TEST")
print("=" * 60)

# Initialize cluster with analytical function
print("Mode: ANALYTICAL FUNCTION")
print("Environment: sink3")
print("-" * 60)
cluster = RobotCluster(environment_function=enviro, use_nn=False, use_rbf=False)

print("=" * 60)
print()

# Set up the figure for visualization
fig = plt.figure(figsize=(10, 10))

# Run simulation with find_center2
plt.title('Analytical - find_center2 Test')
execute_simulation(cluster, cp.find_center2, 'find_center2 Test')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("SIMULATION COMPLETED")
print("=" * 60)