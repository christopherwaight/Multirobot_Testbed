# simulation/simulation.py
import numpy as np
import matplotlib.pyplot as plt
from env.grid_setup import make_environment_grid

def execute_simulation(cluster, update_robot_position, title):
    X, Y = make_environment_grid()

    u, v = cluster.environment_function(X, Y)

    plt.quiver(X, Y, u, v, color='black')
    plt.title(title)

    # Simulation
    sim_time = 1000
    cluster.plot()
    
    # Pre-allocate array with zeros
    centre_points = np.zeros((sim_time + 1, 2))  # +1 for initial position
    centre_points[0] = cluster.cluster_centre    # Set initial position

    for i in range(sim_time):
        cluster.move(update_robot_position)
        centre_points[i + 1] = cluster.cluster_centre  # Fill pre-allocated array
            
    plt.plot(centre_points[:, 0], centre_points[:, 1], marker='o', color='black')