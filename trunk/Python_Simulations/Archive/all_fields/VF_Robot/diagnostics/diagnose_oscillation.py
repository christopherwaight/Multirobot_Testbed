"""
Diagnose the velocity oscillation issue
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

print("=" * 60)
print("DIAGNOSING VELOCITY OSCILLATION")
print("=" * 60)

# Setup
field = AnalyticalField(sink1)
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

# Track detailed data
velocities = []
commanded_vels = []
positions = []

print("\nRunning 100 steps with detailed logging...")

for step in range(100):
    # Store position before move
    centroid = cluster.get_centroid()
    positions.append(centroid.copy())

    # Get what the control primitive WANTS to command
    # (We'll need to call it to see)
    vx_c_cmd, vy_c_cmd = ocp.critical_point_plane_fitting(cluster)
    cmd_mag = np.linalg.norm([vx_c_cmd, vy_c_cmd])
    commanded_vels.append(cmd_mag)

    # Execute the move
    cluster.move(ocp.critical_point_plane_fitting)

    # Get actual robot velocities
    robot_vels = [robot.get_velocity() for robot in cluster.robots]
    vel_mags = [np.linalg.norm(v) for v in robot_vels]
    avg_vel = np.mean(vel_mags)
    velocities.append(avg_vel)

    # Print details for some steps
    if step < 10 or step % 20 == 0:
        print(f"\nStep {step}:")
        print(f"  Centroid: ({centroid[0]:.4f}, {centroid[1]:.4f})")
        print(f"  Commanded vel: {cmd_mag:.4f} m/s")
        print(f"  Robot velocities: {[f'{v:.4f}' for v in vel_mags]}")
        print(f"  Avg velocity: {avg_vel:.4f} m/s")

        # Check stiction
        if avg_vel == 0.0:
            print(f"  → STOPPED (stiction)")
        elif avg_vel >= 0.3:
            print(f"  → MAX VELOCITY (clamped)")

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Commanded vs Actual velocity
ax1 = axes[0]
time = np.arange(len(velocities)) * 0.1
ax1.plot(time, commanded_vels, 'b-', label='Commanded Velocity', alpha=0.7, linewidth=2)
ax1.plot(time, velocities, 'r-', label='Actual Avg Velocity', linewidth=2)
ax1.axhline(y=0.3, color='green', linestyle='--', label='Max Velocity (0.3 m/s)', alpha=0.5)
ax1.axhline(y=0.025, color='orange', linestyle='--', label='Stiction Threshold (0.025 m/s)', alpha=0.5)
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Time (s)')
ax1.set_title('Commanded vs Actual Velocity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Position trajectory
ax2 = axes[1]
positions = np.array(positions)
ax2.plot(positions[:, 0], positions[:, 1], 'b-', marker='o', markersize=3)
ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='x', label='End', zorder=5)
ax2.scatter(0, 0, color='black', s=100, marker='*', label='Sink Center', zorder=5)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('Centroid Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Plot 3: Distance to center over time
ax3 = axes[2]
distances = np.linalg.norm(positions, axis=1)
ax3.plot(time, distances, 'g-', linewidth=2)
ax3.set_ylabel('Distance to Sink (m)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Distance to Critical Point')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_plots/oscillation_diagnosis.png', dpi=150)
print("\n" + "=" * 60)
print("Diagnosis plot saved: velocity_plots/oscillation_diagnosis.png")
print("=" * 60)

# Statistical analysis
print("\nSTATISTICAL ANALYSIS:")
print(f"  Commanded velocity - Mean: {np.mean(commanded_vels):.4f} m/s, Std: {np.std(commanded_vels):.4f} m/s")
print(f"  Actual velocity    - Mean: {np.mean(velocities):.4f} m/s, Std: {np.std(velocities):.4f} m/s")
print(f"  Times stopped (stiction): {sum(1 for v in velocities if v == 0)}/{len(velocities)} steps")
print(f"  Times at max velocity: {sum(1 for v in velocities if v >= 0.3)}/{len(velocities)} steps")
print(f"  Final distance to sink: {distances[-1]:.4f} m")

plt.show()
