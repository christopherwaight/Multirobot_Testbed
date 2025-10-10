# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from env.grid_setup import make_environment_grid
import torch

def execute_simulation(cluster, update_robot_position, title):
    X, Y = make_environment_grid()
    
    # Check which mode the cluster is using and evaluate field accordingly
    if hasattr(cluster, 'use_nn') and cluster.use_nn:
        # Neural network only mode
        u, v = evaluate_nn_on_grid(cluster, X, Y)
    elif hasattr(cluster, 'use_rbf') and cluster.use_rbf:
        # RBF only mode
        u, v = evaluate_rbf_on_grid(cluster, X, Y)
    elif hasattr(cluster, 'use_blended') and cluster.use_blended:
        # Blended mode
        u, v = evaluate_blended_on_grid(cluster, X, Y)
    elif hasattr(cluster, 'environment_function'):
        # Analytical function mode
        u, v = cluster.environment_function(X, Y)
    else:
        # Fallback - try to evaluate using bot_readings at grid points
        print("Warning: No specific mode detected, using bot_readings evaluation")
        u, v = evaluate_generic_on_grid(cluster, X, Y)

    plt.quiver(X, Y, u, v, color='black', alpha=0.6)
    plt.title(title)

    # Simulation
    sim_time = 150
    cluster.plot()
    
    # Pre-allocate array with zeros
    centre_points = np.zeros((sim_time + 1, 2))  # +1 for initial position
    centre_points[0] = cluster.cluster_centre    # Set initial position

    for i in range(sim_time):
        cluster.move(update_robot_position)
        centre_points[i + 1] = cluster.cluster_centre  # Fill pre-allocated array
            
    plt.plot(centre_points[:, 0], centre_points[:, 1], marker='o', color='black')


def evaluate_nn_on_grid(cluster, X, Y):
    """
    Evaluate neural network predictions on a grid for visualization.
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    u_flat = np.zeros_like(X_flat)
    v_flat = np.zeros_like(Y_flat)
    
    batch_size = 100
    n_points = len(X_flat)
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            
            batch_x = X_flat[i:end_idx]
            batch_y = Y_flat[i:end_idx]
            batch_input = torch.FloatTensor(np.column_stack([batch_x, batch_y]))
            
            hue_pred = cluster.hue_model(batch_input).numpy()
            sat_pred = cluster.sat_model(batch_input).numpy()
            
            for j in range(len(batch_x)):
                angle = np.arctan2(hue_pred[j, 0], hue_pred[j, 1])
                sat = (sat_pred[j, 0] + 1) / 2
                magnitude = (sat - 0.3) / 0.7
                
                u_flat[i + j] = magnitude * np.cos(angle)
                v_flat[i + j] = magnitude * np.sin(angle)
    
    u = u_flat.reshape(shape)
    v = v_flat.reshape(shape)
    return u, v


def evaluate_rbf_on_grid(cluster, X, Y):
    """
    Evaluate RBF interpolator predictions on a grid for visualization.
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Prepare points for RBF evaluation
    points = np.column_stack([X_flat, Y_flat])
    
    # Get RBF predictions
    hue_sincos = cluster.rbf_hue(points)
    sat_pred = cluster.rbf_sat(points).flatten()
    
    # Convert to u, v
    u_flat = np.zeros_like(X_flat)
    v_flat = np.zeros_like(Y_flat)
    
    for i in range(len(points)):
        angle = np.arctan2(hue_sincos[i, 0], hue_sincos[i, 1])
        magnitude = (sat_pred[i] - 0.3) / 0.7
        u_flat[i] = magnitude * np.cos(angle)
        v_flat[i] = magnitude * np.sin(angle)
    
    u = u_flat.reshape(shape)
    v = v_flat.reshape(shape)
    return u, v


def evaluate_blended_on_grid(cluster, X, Y):
    """
    Evaluate blended RBF/NN predictions on a grid for visualization.
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    u_flat = np.zeros_like(X_flat)
    v_flat = np.zeros_like(Y_flat)
    
    # Process in batches for NN
    batch_size = 100
    n_points = len(X_flat)
    
    # Prepare all points for RBF (it can handle all at once)
    all_points = np.column_stack([X_flat, Y_flat])
    rbf_hue_all = cluster.rbf_hue(all_points)
    rbf_sat_all = cluster.rbf_sat(all_points).flatten()
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            
            batch_x = X_flat[i:end_idx]
            batch_y = Y_flat[i:end_idx]
            batch_input = torch.FloatTensor(np.column_stack([batch_x, batch_y]))
            
            # Get NN predictions
            nn_hue_pred = cluster.hue_model(batch_input).numpy()
            nn_sat_pred = cluster.sat_model(batch_input).numpy()
            
            for j in range(len(batch_x)):
                idx = i + j
                
                # Blend at hue/sat level
                blended_sin = cluster.rbf_weight * rbf_hue_all[idx, 0] + cluster.nn_weight * nn_hue_pred[j, 0]
                blended_cos = cluster.rbf_weight * rbf_hue_all[idx, 1] + cluster.nn_weight * nn_hue_pred[j, 1]
                
                # NN saturation is in [-1, 1], RBF is in [0, 1]
                nn_sat = (nn_sat_pred[j, 0] + 1) / 2
                blended_sat = cluster.rbf_weight * rbf_sat_all[idx] + cluster.nn_weight * nn_sat
                
                # Convert to angle and magnitude
                angle = np.arctan2(blended_sin, blended_cos)
                magnitude = (blended_sat - 0.3) / 0.7
                
                u_flat[idx] = -magnitude * np.cos(angle)
                v_flat[idx] = -magnitude * np.sin(angle)
    
    u = u_flat.reshape(shape)
    v = v_flat.reshape(shape)
    return u, v


def evaluate_generic_on_grid(cluster, X, Y):
    """
    Generic evaluation using bot_readings at each grid point.
    This is slower but works for any configuration.
    """
    shape = X.shape
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    u_flat = np.zeros_like(X_flat)
    v_flat = np.zeros_like(Y_flat)
    
    # Save original cluster state
    original_centre = cluster.cluster_centre.copy()
    original_offsets = cluster.robot_offsets.copy()
    
    for i in range(len(X_flat)):
        # Move cluster to grid point (all robots at same location)
        cluster.cluster_centre = np.array([X_flat[i], Y_flat[i]])
        cluster.robot_offsets = np.array([[0, 0], [0, 0], [0, 0]])
        
        # Get reading
        readings = cluster.bot_readings()
        u_flat[i] = readings[0][0]
        v_flat[i] = readings[0][1]
    
    # Restore original cluster state
    cluster.cluster_centre = original_centre
    cluster.robot_offsets = original_offsets
    
    u = u_flat.reshape(shape)
    v = v_flat.reshape(shape)
    return u, v