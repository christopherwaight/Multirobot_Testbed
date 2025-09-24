# main.py

## Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster import RobotCluster
from exec_sim.simulation import execute_simulation

# Import environment functions (for comparison if needed)
#from env.Saddle import true_saddle as enviro 
#from env.Sinking_Vortex import sinking_vortex as enviro 
from env.Sink import sink as enviro 

## Main - Execute the simulations

# Configuration options
USE_BLENDED = True       # Use blended RBF/NN approach
USE_NN_ONLY = False      # Use only neural network
USE_RBF_ONLY = False     # Use only RBF
USE_ANALYTICAL = False   # Use analytical function

# Blending weight (only used when USE_BLENDED is True)
RBF_WEIGHT = 0.1  # 90% RBF, 10% NN

print("=" * 60)
print("ROBOT CLUSTER SIMULATION")
print("=" * 60)

# Initialize cluster based on selected mode
if USE_BLENDED:
    print(f"Mode: BLENDED (RBF: {RBF_WEIGHT*100:.0f}%, NN: {(1-RBF_WEIGHT)*100:.0f}%)")
    print("-" * 60)
    cluster = RobotCluster(use_blended=True, rbf_weight=RBF_WEIGHT)
    mode_name = f'Blended ({RBF_WEIGHT*100:.0f}% RBF)'
elif USE_NN_ONLY:
    print("Mode: NEURAL NETWORK ONLY")
    print("-" * 60)
    cluster = RobotCluster(use_nn=True)
    mode_name = 'Neural Network'
elif USE_RBF_ONLY:
    print("Mode: RBF INTERPOLATOR ONLY")
    print("-" * 60)
    cluster = RobotCluster(use_rbf=True)
    mode_name = 'RBF Interpolator'
else:
    print("Mode: ANALYTICAL FUNCTION")
    print("-" * 60)
    cluster = RobotCluster(environment_function=enviro, use_nn=False, use_rbf=False)
    mode_name = 'Analytical'

print("=" * 60)
print()

# Set up the figure for visualization
fig = plt.figure(figsize=(12, 12))

# Main simulation plot
ax1 = plt.subplot(2, 2, 1)
plt.title(f'{mode_name} - Critical Point Orbiter')
execute_simulation(cluster, cp.critical_point_orbiter_plane_fitting, 
                  f'{mode_name} Critical Point Orbiter')

# Optional: Compare with all modes
if USE_BLENDED:
    # Pure NN
    ax2 = plt.subplot(2, 2, 2)
    plt.title('Pure NN (for comparison)')
    cluster_nn = RobotCluster(use_nn=True)
    execute_simulation(cluster_nn, cp.critical_point_orbiter_plane_fitting, 
                      'NN Critical Point Orbiter')
    
    # Pure RBF
    ax3 = plt.subplot(2, 2, 3)
    plt.title('Pure RBF (for comparison)')
    cluster_rbf = RobotCluster(use_rbf=True)
    execute_simulation(cluster_rbf, cp.critical_point_orbiter_plane_fitting, 
                      'RBF Critical Point Orbiter')
    
    # Analytical (Ground Truth)
    ax4 = plt.subplot(2, 2, 4)
    plt.title('Analytical/Ground Truth (for comparison)')
    cluster_analytical = RobotCluster(environment_function=enviro, use_nn=False, use_rbf=False)
    execute_simulation(cluster_analytical, cp.critical_point_orbiter_plane_fitting, 
                      'Analytical Critical Point Orbiter')

plt.tight_layout()
plt.show()

# Test predictions at sample points to verify blending
if USE_BLENDED:
    print("\n" + "=" * 60)
    print("TESTING BLENDED PREDICTIONS AT SAMPLE POINTS")
    print("=" * 60)
    
    # Test at a few sample points
    test_points = [
        [0.0, 0.0],
        [0.3, 0.3],
        [-0.3, 0.3],
    ]
    
    # Create test clusters for each mode
    test_cluster_blend = RobotCluster(use_blended=True, rbf_weight=RBF_WEIGHT)
    test_cluster_nn = RobotCluster(use_nn=True)
    test_cluster_rbf = RobotCluster(use_rbf=True)
    
    for point in test_points:
        print(f"\nPoint ({point[0]:5.2f}, {point[1]:5.2f}):")
        print("-" * 40)
        
        # Set all clusters to same position for comparison
        for cluster in [test_cluster_blend, test_cluster_nn, test_cluster_rbf]:
            cluster.cluster_centre = np.array(point)
            cluster.robot_offsets = np.array([[0, 0], [0, 0], [0, 0]])
        
        # Get readings from each mode
        blend_readings = test_cluster_blend.bot_readings()[0]
        nn_readings = test_cluster_nn.bot_readings()[0]
        rbf_readings = test_cluster_rbf.bot_readings()[0]
        
        print(f"  NN:      u={nn_readings[0]:7.4f}, v={nn_readings[1]:7.4f}")
        print(f"  RBF:     u={rbf_readings[0]:7.4f}, v={rbf_readings[1]:7.4f}")
        print(f"  Blended: u={blend_readings[0]:7.4f}, v={blend_readings[1]:7.4f}")
        
        # Verify blending is approximately correct (accounting for noise)
        expected_u = RBF_WEIGHT * rbf_readings[0] + (1-RBF_WEIGHT) * nn_readings[0]
        expected_v = RBF_WEIGHT * rbf_readings[1] + (1-RBF_WEIGHT) * nn_readings[1]
        print(f"  Expected (no noise): u≈{expected_u:7.4f}, v≈{expected_v:7.4f}")
    
    print("\n" + "=" * 60)
    print("Note: Small differences from expected are due to:")
    print("  1. Random noise added to readings (±0.03)")
    print("  2. Blending happens at hue/sat level, not u/v level")
    print("=" * 60)