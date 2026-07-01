"""
Standalone script to plot robot velocities from simulation data.

This script demonstrates how to use the velocity plotting utilities
independently from the main simulation runner.

Usage:
    python3 plot_velocities.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.simulation.velocity_plotter import plot_robot_velocities, plot_velocity_components
from src.robot.omni_cluster import OmniCluster
from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sinking_Vortex import sinking_vortex1
import src.control.primitives as ocp


def main():
    """Example: Run a simulation and plot velocities."""

    print("=" * 60)
    print("VELOCITY PLOTTING EXAMPLE")
    print("=" * 60)

    # Configuration
    NUM_ROBOTS = 3
    FORMATION_CONFIG = "config/formations/equilateral_default.yaml"
    TIMESTEP = 0.1  # seconds
    SIM_TIME = 150  # time steps

    # Create field and cluster
    field = AnalyticalField(sinking_vortex1)

    if NUM_ROBOTS == 3:
        cluster = OmniCluster(FORMATION_CONFIG, field, timestep=TIMESTEP)
        control_primitive = ocp.critical_point_orbiter_plane_fitting
    else:
        cluster = QuadCluster(FORMATION_CONFIG, field, timestep=TIMESTEP)
        # Would need to import quad_primitives for 4-robot mode

    print(f"Running {SIM_TIME} simulation steps...")

    # Run simulation
    for i in range(SIM_TIME):
        cluster.move(control_primitive)
        if (i + 1) % 30 == 0:
            print(f"  Step {i+1}/{SIM_TIME}")

    # Get velocity history
    velocity_history = cluster.get_velocity_history()

    print(f"\nCollected {len(velocity_history)} velocity samples")
    print(f"Time span: {len(velocity_history) * TIMESTEP:.1f} seconds")

    # Plot velocity magnitudes
    print("\nGenerating velocity magnitude plot...")
    plot_robot_velocities(
        velocity_history,
        timestep=TIMESTEP,
        title="Example: Sinking Vortex 1 - Robot Velocities",
        save_path="velocity_plots/example_velocity_magnitudes.png",
        num_robots=NUM_ROBOTS
    )

    # Plot velocity components (vx, vy)
    print("Generating velocity components plot...")
    plot_velocity_components(
        velocity_history,
        timestep=TIMESTEP,
        title="Example: Sinking Vortex 1 - Velocity Components",
        save_path="velocity_plots/example_velocity_components.png",
        num_robots=NUM_ROBOTS
    )

    print("\n" + "=" * 60)
    print("PLOTS GENERATED SUCCESSFULLY")
    print("Check the 'velocity_plots/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
