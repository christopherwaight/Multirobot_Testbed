"""
Test single environment with velocity plotting to verify main_omni.py integration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
import src.control.primitives as ocp
from src.simulation.runner import execute_omni_simulation
from src.simulation.velocity_plotter import plot_robot_velocities

print("=" * 60)
print("TESTING SINGLE ENVIRONMENT WITH VELOCITY PLOT")
print("=" * 60)

# Setup (matching main_omni.py structure)
NUM_ROBOTS = 3
FORMATION_CONFIG = "config/formations/equilateral_default.yaml"
FIELD_MODE = "analytical"
CONTROL_PRIMITIVE = "critical_point_plane_fitting"

print(f"\nConfiguration:")
print(f"  Robots: {NUM_ROBOTS}")
print(f"  Environment: Sink 1")
print(f"  Field Mode: {FIELD_MODE}")
print(f"  Control Primitive: {CONTROL_PRIMITIVE}")
print(f"  Simulation steps: 150")

# Create field and cluster
field = AnalyticalField(sink1)
cluster = OmniCluster(FORMATION_CONFIG, field)

# Get control primitive
control_func = ocp.critical_point_plane_fitting

# Run simulation
print("\nRunning simulation...")
for i in range(150):
    cluster.move(control_func)
    if (i+1) % 30 == 0:
        print(f"  Step {i+1}/150")

# Get velocity history
velocity_history = cluster.get_velocity_history()
print(f"\nCollected {len(velocity_history)} velocity samples")

# Generate velocity plot (using same naming as main_omni.py)
env_key = "sink1"
mode_str = FIELD_MODE.upper()
vel_plot_filename = f"velocity_plots/{env_key}_velocity_{FIELD_MODE}.png"
plot_title = f"Sink 1 - Robot Velocities ({mode_str})"

print(f"\nGenerating velocity plot...")
print(f"  Output file: {vel_plot_filename}")

plot_robot_velocities(
    velocity_history,
    timestep=cluster.timestep,
    title=plot_title,
    save_path=vel_plot_filename,
    num_robots=NUM_ROBOTS
)

print("\n" + "=" * 60)
print("SUCCESS! Velocity plot generated.")
print(f"File: {vel_plot_filename}")
print("=" * 60)
print("\nTo view the plot:")
print(f"  open {vel_plot_filename}")
