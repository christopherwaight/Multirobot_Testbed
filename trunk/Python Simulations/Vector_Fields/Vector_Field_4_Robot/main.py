# main.py

## Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster_four import RobotCluster
from exec_sim.simulation import execute_simulation

from env.Sinking_Vortex import sinking_vortex3  as enviro 

## Main - Execute the simulations
plt.figure(figsize=(6, 6))
cluster = RobotCluster(enviro)
#execute_simulation(cluster, cp.find_sinking_vortex_center, 'Saddle Point Seeker')
execute_simulation(cluster, cp.dual_jacobian_center_finder, 'Saddle Point Seeker')
plt.show()