# main.py

## Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster import RobotCluster
from exec_sim.simulation import execute_simulation

# Import all 18 environment functions
from env.Sink import sink1, sink2, sink3
from env.Source import source1, source2, source3
from env.Vortex import vortex1, vortex2, vortex3
from env.Sinking_Vortex import sinking_vortex1, sinking_vortex2, sinking_vortex3
from env.Spewing_Vortex import spewing_vortex1, spewing_vortex2, spewing_vortex3
from env.Saddle import saddle1, saddle2, saddle3

# List of all environment functions with names
environments = [
    (sink1, "Sink 1"),
    (source1, "Source 1"),
    (vortex1, "Vortex 1"),
    (sinking_vortex1, "Sinking Vortex 1"),
    (spewing_vortex1, "Spewing Vortex 1"),
    (saddle1, "Saddle 1"),

    (sink2, "Sink 2"),
    (source2, "Source 2"),
    (vortex2, "Vortex 2"),
    (sinking_vortex2, "Sinking Vortex 2"),
    (spewing_vortex2, "Spewing Vortex 2"),
    (saddle2, "Saddle 2"),

    (sink3, "Sink 3"),
    (source3, "Source 3"),
    (vortex3, "Vortex 3"),
    (sinking_vortex3, "Sinking Vortex 3"),
    (spewing_vortex3, "Spewing Vortex 3"),
    (saddle3, "Saddle 3"),
]

## Main - Execute the simulations
print("=" * 60)
print("ROBOT CLUSTER SIMULATION - 18 ENVIRONMENTS")
print("Mode: ANALYTICAL FUNCTION ONLY")
print("=" * 60)
print()

# Create a single figure with 3x6 subplots
fig = plt.figure(figsize=(24, 12))
fig.suptitle('Robot Cluster Simulations - All 18 Environments (Analytical)', fontsize=16)

# Loop through all 18 environments
for i, (env_func, env_name) in enumerate(environments, 1):
    print(f"[{i}/18] Running simulation with: {env_name}")
    
    # Create subplot in 3x6 grid
    ax = plt.subplot(3, 6, i)
    ax.set_title(env_name, fontsize=10)
    
    # Initialize cluster with analytical function
    cluster = RobotCluster(environment_function=env_func, use_nn=False, use_rbf=False)
    
    # Run simulation
    # execute_simulation(cluster, cp.critical_point_plane_fitting, 
    #                f'{env_name} Critical Point Finder')
    
    # # Run simulation
    execute_simulation(cluster, cp.eigenstep, 
                    f'{env_name} Critical Point Finder')

    # execute_simulation(cluster, cp.center_finder3, 
    #                  f'{env_name} Critical Point Finder')    
   #

    # execute_simulation(cluster, cp.rio_finder3_simple, 
    #                 f'{env_name} Critical Point Finder')
    
    # Adjust subplot to be more compact
    ax.tick_params(labelsize=8)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)

print("\n" + "=" * 60)
print("ALL SIMULATIONS COMPLETED")
print("=" * 60)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Optional: Save the complete figure
# plt.savefig('all_simulations_3x6.png', dpi=150, bbox_inches='tight')

# Display the complete plot
plt.show()