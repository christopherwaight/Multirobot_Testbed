# main_new.py - Unified simulation runner
# Combines functionality from main.py, main_lite.py, 3_robot_attract.py, and 3_robot_orbits.py

import numpy as np
import matplotlib.pyplot as plt
from primitives import control_primitives as cp
from clusters.robot_cluster import RobotCluster
from exec_sim.simulation import execute_simulation

# ============================================================================
# CONFIGURATION SECTION - Modify these settings to control behavior
# ============================================================================

# --- Simulation Mode ---
SIMULATION_MODE = "compare"  # Options: "single", "compare", "multi_env"
# "single"    - Run one simulation with one environment
# "compare"   - Compare 4 modes (Blended, NN, RBF, Analytical) side-by-side
# "multi_env" - Run all 18 environments in a grid (3x6)

# --- Field Approximation Mode ---
USE_BLENDED = True       # Use blended RBF/NN approach
USE_NN_ONLY = True      # Use only neural network
USE_RBF_ONLY = True     # Use only RBF
USE_ANALYTICAL = True   # Use analytical function
RBF_WEIGHT = 0.9         # Blending weight (0.9 = 90% RBF, 10% NN)

# --- Environment Selection ---
ENVIRONMENT = "sinking_vortex3"  # Options: see environment_map below
# Single environment options:
#   "sink1", "sink2", "sink3"
#   "source1", "source2", "source3"
#   "vortex1", "vortex2", "vortex3"
#   "sinking_vortex1", "sinking_vortex2", "sinking_vortex3"
#   "spewing_vortex1", "spewing_vortex2", "spewing_vortex3"
#   "saddle1", "saddle2", "saddle3"

# --- Control Primitive Selection ---
CONTROL_PRIMITIVE = "critical_point_orbiter_plane_fitting"
# Available options (from control_primitives.py):
#   "critical_point_plane_fitting"
#   "critical_point_cross_product"
#   "critical_point_orbiter_plane_fitting"
#   "critical_point_orbiter_cross_product"
#   "vector_sum"
#   "find_center"
#   "find_center2"
#   "eigenstep"
#   "direction_follow_with_center_attraction"
#   "center_attraction"
#   "find_sink_center"

# --- Visualization Options ---
FIGURE_SIZE = (12, 12)  # Figure size for compare mode
SAVE_FIGURE = False     # Set to True to save the plot
OUTPUT_FILENAME = "simulation_output.png"

# --- Testing Options ---
RUN_BLENDING_TEST = False  # Run blending verification test (only in compare mode)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Import all environment functions
from env.Sink import sink1, sink2, sink3
from env.Source import source1, source2, source3
from env.Vortex import vortex1, vortex2, vortex3
from env.Sinking_Vortex import sinking_vortex1, sinking_vortex2, sinking_vortex3
from env.Spewing_Vortex import spewing_vortex1, spewing_vortex2, spewing_vortex3
from env.Saddle import saddle1, saddle2, saddle3

# Environment mapping
environment_map = {
    "sink1": (sink1, "Sink 1"),
    "sink2": (sink2, "Sink 2"),
    "sink3": (sink3, "Sink 3"),
    "source1": (source1, "Source 1"),
    "source2": (source2, "Source 2"),
    "source3": (source3, "Source 3"),
    "vortex1": (vortex1, "Vortex 1"),
    "vortex2": (vortex2, "Vortex 2"),
    "vortex3": (vortex3, "Vortex 3"),
    "sinking_vortex1": (sinking_vortex1, "Sinking Vortex 1"),
    "sinking_vortex2": (sinking_vortex2, "Sinking Vortex 2"),
    "sinking_vortex3": (sinking_vortex3, "Sinking Vortex 3"),
    "spewing_vortex1": (spewing_vortex1, "Spewing Vortex 1"),
    "spewing_vortex2": (spewing_vortex2, "Spewing Vortex 2"),
    "spewing_vortex3": (spewing_vortex3, "Spewing Vortex 3"),
    "saddle1": (saddle1, "Saddle 1"),
    "saddle2": (saddle2, "Saddle 2"),
    "saddle3": (saddle3, "Saddle 3"),
}

# All 18 environments for multi_env mode
all_environments = [
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

# Control primitive mapping
control_primitive_map = {
    "critical_point_plane_fitting": cp.critical_point_plane_fitting,
    "critical_point_cross_product": cp.critical_point_cross_product,
    "critical_point_orbiter_plane_fitting": cp.critical_point_orbiter_plane_fitting,
    "critical_point_orbiter_cross_product": cp.critical_point_orbiter_cross_product,
    "vector_sum": cp.vector_sum,
    "find_center": cp.find_center,
    "find_center2": cp.find_center2,
    "eigenstep": cp.eigenstep,
    "direction_follow_with_center_attraction": cp.direction_follow_with_center_attraction,
    "center_attraction": cp.center_attraction,
    "find_sink_center": cp.find_sink_center,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Map environments to their trained model directories
PREDICTOR_DIR_MAP = {
    "saddle1": "saddle_predictors",
    "vortex1": "vortex_predictors",
    "sinking_vortex3": "sinking_vortex_predictors",
    # Add more mappings as you train models for other environments
}

def get_predictor_dir(env_name):
    """Get the predictor directory for a given environment name."""
    # Extract the base name (e.g., "saddle1" -> "saddle1")
    return PREDICTOR_DIR_MAP.get(env_name, 'sinking_vortex_predictors')

def create_cluster(env_func, env_name):
    """Create a robot cluster based on the selected mode and environment"""
    predictor_dir = get_predictor_dir(env_name)

    if USE_BLENDED:
        return RobotCluster(use_blended=True, rbf_weight=RBF_WEIGHT, predictor_dir=predictor_dir), f'Blended ({RBF_WEIGHT*100:.0f}% RBF)'
    elif USE_NN_ONLY:
        return RobotCluster(use_nn=True, predictor_dir=predictor_dir), 'Neural Network'
    elif USE_RBF_ONLY:
        return RobotCluster(use_rbf=True, predictor_dir=predictor_dir), 'RBF Interpolator'
    else:
        return RobotCluster(environment_function=env_func, use_nn=False, use_rbf=False), 'Analytical'

def get_control_primitive():
    """Get the selected control primitive function"""
    if CONTROL_PRIMITIVE not in control_primitive_map:
        print(f"WARNING: Unknown control primitive '{CONTROL_PRIMITIVE}', using 'vector_sum'")
        return cp.vector_sum
    return control_primitive_map[CONTROL_PRIMITIVE]

def run_blending_test(env_func):
    """Run blending verification test"""
    print("\n" + "=" * 60)
    print("TESTING BLENDED PREDICTIONS AT SAMPLE POINTS")
    print("=" * 60)

    test_points = [
        [0.0, 0.0],
        [0.3, 0.3],
        [-0.3, 0.3],
    ]

    # Get the correct predictor directory for this environment
    predictor_dir = get_predictor_dir(ENVIRONMENT)

    # Create test clusters for each mode
    test_cluster_blend = RobotCluster(use_blended=True, rbf_weight=RBF_WEIGHT, predictor_dir=predictor_dir)
    test_cluster_nn = RobotCluster(use_nn=True, predictor_dir=predictor_dir)
    test_cluster_rbf = RobotCluster(use_rbf=True, predictor_dir=predictor_dir)

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

# ============================================================================
# SIMULATION MODES
# ============================================================================

def run_single_simulation():
    """Run a single simulation with one environment"""
    print("=" * 60)
    print("ROBOT CLUSTER SIMULATION - SINGLE ENVIRONMENT")
    print("=" * 60)

    # Get environment
    if ENVIRONMENT not in environment_map:
        print(f"ERROR: Unknown environment '{ENVIRONMENT}'")
        return

    env_func, env_name = environment_map[ENVIRONMENT]

    # Create cluster
    cluster, mode_name = create_cluster(env_func, env_name)

    print(f"Mode: {mode_name}")
    print(f"Environment: {env_name}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print("=" * 60)
    print()

    # Set up figure
    fig = plt.figure(figsize=(10, 10))

    # Get control primitive
    control_func = get_control_primitive()

    # Run simulation
    title = f'{mode_name} - {env_name} - {CONTROL_PRIMITIVE}'
    plt.title(title)
    execute_simulation(cluster, control_func, title)

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)

def run_comparison_simulation():
    """Compare Blended, NN, RBF, and Analytical modes side-by-side"""
    print("=" * 60)
    print("ROBOT CLUSTER SIMULATION - MODE COMPARISON")
    print("=" * 60)

    # Get environment
    if ENVIRONMENT not in environment_map:
        print(f"ERROR: Unknown environment '{ENVIRONMENT}'")
        return

    env_func, env_name = environment_map[ENVIRONMENT]

    print(f"Environment: {env_name}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print(f"RBF Weight: {RBF_WEIGHT*100:.0f}%")
    print("=" * 60)
    print()

    # Set up figure
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Get control primitive
    control_func = get_control_primitive()

    # Get the correct predictor directory for this environment
    predictor_dir = get_predictor_dir(ENVIRONMENT)

    # Main simulation plot (Blended)
    ax1 = plt.subplot(2, 2, 1)
    plt.title(f'Blended ({RBF_WEIGHT*100:.0f}% RBF)')
    cluster = RobotCluster(use_blended=True, rbf_weight=RBF_WEIGHT, predictor_dir=predictor_dir)
    execute_simulation(cluster, control_func, f'Blended - {env_name}')

    # Pure NN
    ax2 = plt.subplot(2, 2, 2)
    plt.title('Pure Neural Network')
    cluster_nn = RobotCluster(use_nn=True, predictor_dir=predictor_dir)
    execute_simulation(cluster_nn, control_func, f'NN - {env_name}')

    # Pure RBF
    ax3 = plt.subplot(2, 2, 3)
    plt.title('Pure RBF Interpolator')
    cluster_rbf = RobotCluster(use_rbf=True, predictor_dir=predictor_dir)
    execute_simulation(cluster_rbf, control_func, f'RBF - {env_name}')

    # Analytical (Ground Truth)
    ax4 = plt.subplot(2, 2, 4)
    plt.title('Analytical (Ground Truth)')
    cluster_analytical = RobotCluster(environment_function=env_func, use_nn=False, use_rbf=False)
    execute_simulation(cluster_analytical, control_func, f'Analytical - {env_name}')

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()

    # Run blending test if requested
    if RUN_BLENDING_TEST:
        run_blending_test(env_func)

def run_multi_environment_simulation():
    """Run all 18 environments in a 3x6 grid"""
    print("=" * 60)
    print("ROBOT CLUSTER SIMULATION - 18 ENVIRONMENTS")
    print("=" * 60)

    # Determine mode
    if USE_BLENDED:
        mode_str = f"Blended ({RBF_WEIGHT*100:.0f}% RBF)"
    elif USE_NN_ONLY:
        mode_str = "Neural Network Only"
    elif USE_RBF_ONLY:
        mode_str = "RBF Interpolator Only"
    else:
        mode_str = "Analytical Function Only"

    print(f"Mode: {mode_str}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print("=" * 60)
    print()

    # Create figure with 3x6 subplots
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'Robot Cluster Simulations - All 18 Environments ({mode_str})', fontsize=16)

    # Get control primitive
    control_func = get_control_primitive()

    # Loop through all 18 environments
    for i, (env_func, env_name) in enumerate(all_environments, 1):
        print(f"[{i}/18] Running simulation with: {env_name}")

        # Create subplot in 3x6 grid
        ax = plt.subplot(3, 6, i)
        ax.set_title(env_name, fontsize=10)

        # Create cluster
        cluster, _ = create_cluster(env_func, env_name)

        # Run simulation
        execute_simulation(cluster, control_func, f'{env_name}')

        # Adjust subplot to be more compact
        ax.tick_params(labelsize=8)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETED")
    print("=" * 60)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - routes to appropriate simulation mode"""
    if SIMULATION_MODE == "single":
        run_single_simulation()
    elif SIMULATION_MODE == "compare":
        run_comparison_simulation()
    elif SIMULATION_MODE == "multi_env":
        run_multi_environment_simulation()
    else:
        print(f"ERROR: Unknown simulation mode '{SIMULATION_MODE}'")
        print("Valid modes: 'single', 'compare', 'multi_env'")

if __name__ == "__main__":
    main()
