# main.py

## Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster_four import RobotCluster
from exec_sim.simulation import execute_simulation

# Import the environment to test with
#from env.Sinking_Vortex import sinking_vortex as enviro 
from env.Sink import sink as enviro
#from env.Fixed_Vortex import fixed_vortex as enviro
# Set up the plot
plt.figure(figsize=(10, 10))

cluster2 = RobotCluster(enviro)
execute_simulation(cluster2, cp.dual_jacobian_center_finder, 'Center Orbiter')

# Show the plots
plt.tight_layout()
plt.show()