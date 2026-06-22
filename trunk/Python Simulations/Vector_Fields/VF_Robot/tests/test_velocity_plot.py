"""
Quick test to generate a single velocity plot
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
import src.control.primitives as ocp
from src.simulation.velocity_plotter import plot_robot_velocities

print("Creating cluster and field...")
field = AnalyticalField(sink1)
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

print("Running simulation (50 steps)...")
for i in range(50):
    cluster.move(ocp.critical_point_plane_fitting)
    if (i+1) % 10 == 0:
        print(f"  Step {i+1}/50")

print("Getting velocity history...")
velocity_history = cluster.get_velocity_history()
print(f"  Collected {len(velocity_history)} velocity samples")

print("Generating velocity plot...")
plot_robot_velocities(
    velocity_history,
    timestep=0.1,
    title="Test - Sink 1 Robot Velocities",
    save_path="velocity_plots/test_sink1_velocity.png",
    num_robots=3
)

print("Done! Check velocity_plots/test_sink1_velocity.png")
