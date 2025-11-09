"""
main_omni.py - Simulation runner using new OmniCluster architecture
"""
import sys
import os
# Add project root to path so we can import from src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

# New architecture imports
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField, NNField, RBFField, BlendedField
import src.control.primitives as ocp
import src.control.quad_primitives as qcp
from src.simulation.runner import execute_omni_simulation
from src.simulation.velocity_plotter import plot_robot_velocities

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# --- Robot Configuration ---
NUM_ROBOTS = 4  # Options: 3 or 4
# 3 - Use OmniCluster with triangular formation (SAS parameterization)
# 4 - Use QuadCluster with quadrilateral formation (diagonal parameterization)

# --- Simulation Mode ---
SIMULATION_MODE = "single"  # Options: "single", "compare", "multi_env"
# "single"    - Run one simulation with one environment
# "compare"   - Compare 4 modes (Blended, NN, RBF, Analytical) side-by-side
# "multi_env" - Run all 18 environments in a grid (3x6)

# --- Field Approximation Mode (for single mode) ---
FIELD_MODE = "analytical"  # Options: "analytical", "nn", "rbf", "blended"
RBF_WEIGHT = 0.9  # Blending weight (0.9 = 90% RBF, 10% NN)

# --- Environment Selection ---
ENVIRONMENT = "vortex1"  # See environment_map below

# --- Formation Configuration ---
# For 3 robots:
FORMATION_CONFIG_3 = "config/formations/equilateral_default.yaml"
# For 4 robots:
FORMATION_CONFIG_4 = "config/formations/quad_square_default.yaml"

# Active formation config (set based on NUM_ROBOTS)
FORMATION_CONFIG = FORMATION_CONFIG_3 if NUM_ROBOTS == 3 else FORMATION_CONFIG_4

# --- Control Primitive Selection ---
# For 3 robots:
CONTROL_PRIMITIVE_3 = "critical_point_plane_fitting"
# For 4 robots:
CONTROL_PRIMITIVE_4 = "center_orbiter_quad_planar"

# Active control primitive (set based on NUM_ROBOTS)
CONTROL_PRIMITIVE = CONTROL_PRIMITIVE_3 if NUM_ROBOTS == 3 else CONTROL_PRIMITIVE_4

# Available 3-robot primitives:
#   "vector_sum"
#   "critical_point_plane_fitting"
#   "critical_point_orbiter_plane_fitting"
#   "find_center"                              - Under Development. Do not use 
#   "find_sink_center"                         - Under Development. Do not use
#
# Available 4-robot primitives:
#   "dual_jacobian_center_finder"    - Uses two triangles (top & bottom) to find center
#   "four_planar_center_finder"      - Uses least squares with all 4 robots to find center
#   "center_orbiter_quad"            - Orbits center (estimated via dual Jacobian)
#   "center_orbiter_quad_planar"     - Orbits center (estimated via 4-point planar)
#   "vector_sum_quad"

# --- Predictor Directory Mapping (for ML models) ---
# Maps environment names to their trained model directories
ENVIRONMENT_TO_PREDICTOR = {
    "sinking_vortex1": "sinking_vortex_predictors",
    "sinking_vortex2": "sinking_vortex_predictors",
    "sinking_vortex3": "sinking_vortex_predictors",
    "saddle1": "saddle_predictors",
    "saddle2": "saddle_predictors",
    "saddle3": "saddle_predictors",
    "vortex1": "vortex_predictors",
    "vortex2": "vortex_predictors",
    "vortex3": "vortex_predictors",
    # Environments without ML models will automatically fall back to analytical
}

# --- Visualization Options ---
FIGURE_SIZE = (12, 12)
SAVE_FIGURE = False
OUTPUT_FILENAME = "simulation_output.png"

# --- Velocity Plotting Options ---
SAVE_VELOCITY_PLOTS = True  # Save velocity plots for multi_env mode
VELOCITY_PLOT_DIR = "velocity_plots"  # Directory to save velocity plots

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Import all environment functions
from src.fields.environments.Sink import sink1, sink2, sink3
from src.fields.environments.Source import source1, source2, source3
from src.fields.environments.Vortex import vortex1, vortex2, vortex3
from src.fields.environments.Sinking_Vortex import sinking_vortex1, sinking_vortex2, sinking_vortex3
from src.fields.environments.Spewing_Vortex import spewing_vortex1, spewing_vortex2, spewing_vortex3
from src.fields.environments.Saddle import saddle1, saddle2, saddle3

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

# All 18 environments for multi_env mode (env_func, display_name, env_key)
all_environments = [
    (sink1, "Sink 1", "sink1"), (source1, "Source 1", "source1"), (vortex1, "Vortex 1", "vortex1"),
    (sinking_vortex1, "Sinking Vortex 1", "sinking_vortex1"), (spewing_vortex1, "Spewing Vortex 1", "spewing_vortex1"), (saddle1, "Saddle 1", "saddle1"),
    (sink2, "Sink 2", "sink2"), (source2, "Source 2", "source2"), (vortex2, "Vortex 2", "vortex2"),
    (sinking_vortex2, "Sinking Vortex 2", "sinking_vortex2"), (spewing_vortex2, "Spewing Vortex 2", "spewing_vortex2"), (saddle2, "Saddle 2", "saddle2"),
    (sink3, "Sink 3", "sink3"), (source3, "Source 3", "source3"), (vortex3, "Vortex 3", "vortex3"),
    (sinking_vortex3, "Sinking Vortex 3", "sinking_vortex3"), (spewing_vortex3, "Spewing Vortex 3", "spewing_vortex3"), (saddle3, "Saddle 3", "saddle3"),
]

# Only "1" variants for 1x6 grid display
environments_1 = [
    (sink1, "Sink 1", "sink1"),
    (source1, "Source 1", "source1"),
    (vortex1, "Vortex 1", "vortex1"),
    (sinking_vortex1, "Sinking Vortex 1", "sinking_vortex1"),
    (spewing_vortex1, "Spewing Vortex 1", "spewing_vortex1"),
    (saddle1, "Saddle 1", "saddle1"),
]

# Control primitive mapping - 3 robots
control_primitive_map_3 = {
    "vector_sum": ocp.vector_sum,
    "critical_point_plane_fitting": ocp.critical_point_plane_fitting,
    "critical_point_orbiter_plane_fitting": ocp.critical_point_orbiter_plane_fitting,
    "find_center": ocp.find_center,
    "find_sink_center": ocp.find_sink_center,
}

# Control primitive mapping - 4 robots
control_primitive_map_4 = {
    "dual_jacobian_center_finder": qcp.dual_jacobian_center_finder,
    "four_planar_center_finder": qcp.four_planar_center_finder,
    "center_orbiter_quad": qcp.center_orbiter_quad,
    "center_orbiter_quad_planar": qcp.center_orbiter_quad_planar,
    "vector_sum_quad": qcp.vector_sum_quad,
}

# Active control primitive map (based on NUM_ROBOTS)
control_primitive_map = control_primitive_map_3 if NUM_ROBOTS == 3 else control_primitive_map_4

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_field(env_func, mode, env_name=None, predictor_dir=None, rbf_weight=0.9):
    """
    Create a field object based on the mode.

    Args:
        env_func: Analytical environment function
        mode: "analytical", "nn", "rbf", or "blended"
        env_name: Environment name (to look up predictor directory)
        predictor_dir: Directory for ML models (overrides env_name lookup)
        rbf_weight: Weight for RBF in blended mode

    Returns:
        VectorField instance (falls back to analytical if ML models don't exist)
    """
    if mode == "analytical":
        return AnalyticalField(env_func)

    # Determine predictor directory
    if predictor_dir is None and env_name is not None:
        predictor_dir = ENVIRONMENT_TO_PREDICTOR.get(env_name)

    # If no predictor directory found, fall back to analytical
    if predictor_dir is None:
        print(f"  No ML models available for '{env_name}', using analytical")
        return AnalyticalField(env_func)

    # Try to load ML models, fall back to analytical if they don't exist
    try:
        if mode == "nn":
            return NNField(predictor_dir)
        elif mode == "rbf":
            return RBFField(predictor_dir)
        elif mode == "blended":
            return BlendedField(predictor_dir, rbf_weight=rbf_weight)
        else:
            raise ValueError(f"Unknown field mode: {mode}")
    except (FileNotFoundError, Exception) as e:
        print(f"  ML models not found in '{predictor_dir}', using analytical")
        print(f"  (Error: {e})")
        return AnalyticalField(env_func)


def create_cluster(field):
    """
    Create appropriate cluster type based on NUM_ROBOTS.

    Args:
        field: VectorField object

    Returns:
        OmniCluster (3 robots) or QuadCluster (4 robots)
    """
    if NUM_ROBOTS == 3:
        return OmniCluster(FORMATION_CONFIG, field)
    elif NUM_ROBOTS == 4:
        return QuadCluster(FORMATION_CONFIG, field)
    else:
        raise ValueError(f"NUM_ROBOTS must be 3 or 4, got {NUM_ROBOTS}")


def get_control_primitive():
    """Get the selected control primitive function."""
    if CONTROL_PRIMITIVE not in control_primitive_map:
        fallback = ocp.vector_sum if NUM_ROBOTS == 3 else qcp.vector_sum_quad
        print(f"WARNING: Unknown control primitive '{CONTROL_PRIMITIVE}', using fallback")
        return fallback
    return control_primitive_map[CONTROL_PRIMITIVE]


# ============================================================================
# SIMULATION MODES
# ============================================================================

def run_single_simulation():
    """Run a single simulation with one environment."""
    print("=" * 60)
    print(f"{NUM_ROBOTS}-ROBOT CLUSTER SIMULATION - SINGLE ENVIRONMENT")
    print("=" * 60)

    # Get environment
    if ENVIRONMENT not in environment_map:
        print(f"ERROR: Unknown environment '{ENVIRONMENT}'")
        return

    env_func, env_name = environment_map[ENVIRONMENT]

    # Create field
    field = create_field(env_func, FIELD_MODE, env_name=ENVIRONMENT)

    # Create cluster
    cluster = create_cluster(field)

    print(f"Robots: {NUM_ROBOTS}")
    print(f"Mode: {FIELD_MODE}")
    print(f"Environment: {env_name}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print(f"Formation: {FORMATION_CONFIG}")
    print("=" * 60)
    print()

    # Set up figure
    fig = plt.figure(figsize=(10, 10))

    # Get control primitive
    control_func = get_control_primitive()

    # Run simulation
    title = f'{FIELD_MODE.upper()} - {env_name} - {CONTROL_PRIMITIVE}'
    execute_omni_simulation(cluster, control_func, title)

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()

    # Generate velocity plot if enabled
    if SAVE_VELOCITY_PLOTS:
        velocity_history = cluster.get_velocity_history()
        if len(velocity_history) > 0:
            vel_plot_filename = f"{VELOCITY_PLOT_DIR}/{ENVIRONMENT}_velocity_{FIELD_MODE}.png"
            plot_title = f"{env_name} - Robot Velocities ({FIELD_MODE.upper()})"

            plot_robot_velocities(
                velocity_history,
                timestep=cluster.timestep,
                title=plot_title,
                save_path=vel_plot_filename,
                num_robots=NUM_ROBOTS
            )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)


def run_comparison_simulation():
    """Compare Blended, NN, RBF, and Analytical modes side-by-side."""
    print("=" * 60)
    print(f"{NUM_ROBOTS}-ROBOT CLUSTER SIMULATION - MODE COMPARISON")
    print("=" * 60)

    # Get environment
    if ENVIRONMENT not in environment_map:
        print(f"ERROR: Unknown environment '{ENVIRONMENT}'")
        return

    env_func, env_name = environment_map[ENVIRONMENT]

    print(f"Robots: {NUM_ROBOTS}")
    print(f"Environment: {env_name}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print(f"RBF Weight: {RBF_WEIGHT*100:.0f}%")
    print(f"Formation: {FORMATION_CONFIG}")
    print("=" * 60)
    print()

    # Set up figure
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Get control primitive
    control_func = get_control_primitive()

    # Create fields
    field_blended = create_field(env_func, "blended", env_name=ENVIRONMENT, rbf_weight=RBF_WEIGHT)
    field_nn = create_field(env_func, "nn", env_name=ENVIRONMENT)
    field_rbf = create_field(env_func, "rbf", env_name=ENVIRONMENT)
    field_analytical = create_field(env_func, "analytical", env_name=ENVIRONMENT)

    # Blended mode
    ax1 = plt.subplot(2, 2, 1)
    cluster_blended = create_cluster(field_blended)
    execute_omni_simulation(cluster_blended, control_func,
                           f'Blended ({RBF_WEIGHT*100:.0f}% RBF) - {env_name}', ax=ax1)

    # NN mode
    ax2 = plt.subplot(2, 2, 2)
    cluster_nn = create_cluster(field_nn)
    execute_omni_simulation(cluster_nn, control_func, f'Neural Network - {env_name}', ax=ax2)

    # RBF mode
    ax3 = plt.subplot(2, 2, 3)
    cluster_rbf = create_cluster(field_rbf)
    execute_omni_simulation(cluster_rbf, control_func, f'RBF Interpolator - {env_name}', ax=ax3)

    # Analytical mode
    ax4 = plt.subplot(2, 2, 4)
    cluster_analytical = create_cluster(field_analytical)
    execute_omni_simulation(cluster_analytical, control_func,
                           f'Analytical (Ground Truth) - {env_name}', ax=ax4)

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()


def run_multi_environment_simulation():
    """Run all 6 '1' variant environments in a 1x6 grid."""
    print("=" * 60)
    print(f"{NUM_ROBOTS}-ROBOT CLUSTER SIMULATION - 6 ENVIRONMENTS (VARIANT 1)")
    print("=" * 60)

    mode_str = FIELD_MODE.upper()

    print(f"Robots: {NUM_ROBOTS}")
    print(f"Mode: {mode_str}")
    print(f"Control Primitive: {CONTROL_PRIMITIVE}")
    print(f"Formation: {FORMATION_CONFIG}")
    print("=" * 60)
    print()

    # Create figure with 6x1 subplots (vertical layout) - publication quality
    fig = plt.figure(figsize=(8, 48))

    # Get control primitive
    control_func = get_control_primitive()

    # Loop through 6 environments
    for i, (env_func, env_name, env_key) in enumerate(environments_1, 1):
        print(f"[{i}/6] Running simulation with: {env_name}")

        # Create subplot in 6x1 grid (vertical layout)
        ax = plt.subplot(6, 1, i)

        # Create field
        field = create_field(env_func, FIELD_MODE, env_name=env_key)

        # Create cluster
        cluster = create_cluster(field)

        # Run simulation (skip_legend for all except last)
        skip_legend = (i < 6)
        execute_omni_simulation(cluster, control_func, env_name, ax=ax, skip_legend=skip_legend)

        # Make subplot square with equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Adjust subplot
        ax.tick_params(labelsize=10)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)

        # Generate velocity plot if enabled
        if SAVE_VELOCITY_PLOTS:
            velocity_history = cluster.get_velocity_history()
            if len(velocity_history) > 0:
                # Create filename
                vel_plot_filename = f"{VELOCITY_PLOT_DIR}/{env_key}_velocity_{mode_str.lower()}.png"
                plot_title = f"{env_name} - Robot Velocities ({mode_str})"

                # Generate plot
                plot_robot_velocities(
                    velocity_history,
                    timestep=cluster.timestep,
                    title=plot_title,
                    save_path=vel_plot_filename,
                    num_robots=NUM_ROBOTS
                )

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETED")
    print("=" * 60)

    plt.tight_layout()

    if SAVE_FIGURE:
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {OUTPUT_FILENAME}")

    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - routes to appropriate simulation mode."""
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
