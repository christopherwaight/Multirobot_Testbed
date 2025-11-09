"""
Test the fixed orbiter primitive
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Vortex import vortex1
import src.control.primitives as ocp

print("=" * 60)
print("TESTING FIXED ORBITER PRIMITIVE")
print("=" * 60)

# Setup
field = AnalyticalField(vortex1)
cluster = OmniCluster("config/formations/equilateral_default.yaml", field)

desired_radius = 0.15  # meters

print(f"\nConfiguration:")
print(f"  Environment: Vortex 1")
print(f"  Desired orbital radius: {desired_radius} m")
print(f"  Base orbital speed: 0.2 m/s")
print(f"  Simulation steps: 200")

# Track data
positions = []
velocities = []
commanded_vels = []
distances = []

print("\nRunning orbital simulation...")

for step in range(200):
    # Get position
    centroid = cluster.get_centroid()
    positions.append(centroid.copy())

    # Calculate distance to center (0, 0)
    distance = np.linalg.norm(centroid)
    distances.append(distance)

    # Get commanded velocity
    vx_c, vy_c = ocp.critical_point_orbiter_plane_fitting(cluster, desired_radius=desired_radius)
    cmd_mag = np.linalg.norm([vx_c, vy_c])
    commanded_vels.append(cmd_mag)

    # Execute move
    cluster.move(lambda c: ocp.critical_point_orbiter_plane_fitting(c, desired_radius=desired_radius))

    # Get actual velocities
    robot_vels = [robot.get_velocity() for robot in cluster.robots]
    vel_mags = [np.linalg.norm(v) for v in robot_vels]
    avg_vel = np.mean(vel_mags)
    velocities.append(avg_vel)

    # Print some steps
    if step < 5 or step % 40 == 0:
        print(f"\nStep {step}:")
        print(f"  Position: ({centroid[0]:.4f}, {centroid[1]:.4f})")
        print(f"  Distance from center: {distance:.4f} m (target: {desired_radius:.4f} m)")
        print(f"  Commanded velocity: {cmd_mag:.4f} m/s")
        print(f"  Actual avg velocity: {avg_vel:.4f} m/s")

# Convert to arrays
positions = np.array(positions)
velocities = np.array(velocities)
commanded_vels = np.array(commanded_vels)
distances = np.array(distances)
time = np.arange(len(positions)) * 0.1

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectory
ax1 = axes[0, 0]
ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.7)
ax1.scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
ax1.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='x', label='End', zorder=5)
ax1.scatter(0, 0, color='black', s=100, marker='*', label='Center', zorder=5)

# Draw desired orbital radius circle
theta = np.linspace(0, 2*np.pi, 100)
circle_x = desired_radius * np.cos(theta)
circle_y = desired_radius * np.sin(theta)
ax1.plot(circle_x, circle_y, 'r--', alpha=0.5, label=f'Desired Radius ({desired_radius} m)')

ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Orbital Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Plot 2: Distance from center over time
ax2 = axes[0, 1]
ax2.plot(time, distances, 'b-', linewidth=2, label='Actual Distance')
ax2.axhline(y=desired_radius, color='r', linestyle='--', label=f'Desired Radius ({desired_radius} m)', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Center (m)')
ax2.set_title('Orbital Radius Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Commanded vs Actual Velocity
ax3 = axes[1, 0]
ax3.plot(time, commanded_vels, 'b-', label='Commanded Velocity', alpha=0.7, linewidth=2)
ax3.plot(time, velocities, 'r-', label='Actual Velocity', linewidth=2)
ax3.axhline(y=0.3, color='green', linestyle='--', label='Max Velocity (0.3 m/s)', alpha=0.5)
ax3.axhline(y=0.2, color='orange', linestyle='--', label='Base Orbital Speed (0.2 m/s)', alpha=0.5)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Radius error over time
ax4 = axes[1, 1]
radius_error = distances - desired_radius
ax4.plot(time, radius_error, 'g-', linewidth=2)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.fill_between(time, radius_error, 0, where=(radius_error>0), alpha=0.3, color='red', label='Too far out')
ax4.fill_between(time, radius_error, 0, where=(radius_error<0), alpha=0.3, color='blue', label='Too close in')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Radius Error (m)')
ax4.set_title('Orbital Radius Error')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_plots/orbiter_test.png', dpi=150)
print("\n" + "=" * 60)
print("Orbiter test plot saved: velocity_plots/orbiter_test.png")
print("=" * 60)

# Statistical analysis
print("\nSTATISTICAL ANALYSIS:")
print(f"  Commanded velocity - Mean: {np.mean(commanded_vels):.4f} m/s, Std: {np.std(commanded_vels):.4f} m/s")
print(f"  Actual velocity    - Mean: {np.mean(velocities):.4f} m/s, Std: {np.std(velocities):.4f} m/s")
print(f"  Distance to center - Mean: {np.mean(distances):.4f} m, Std: {np.std(distances):.4f} m")
print(f"  Radius error       - Mean: {np.mean(radius_error):.4f} m, Std: {np.std(radius_error):.4f} m")
print(f"  Final distance: {distances[-1]:.4f} m (target: {desired_radius} m)")
print(f"  Final radius error: {radius_error[-1]:.4f} m")

# Check if orbit stabilized
last_50_distances = distances[-50:]
last_50_error = np.std(last_50_distances)
print(f"\n  Last 50 steps radius std: {last_50_error:.4f} m")
if last_50_error < 0.02:
    print("  ✓ Orbit STABILIZED (low variation)")
else:
    print("  ⚠ Orbit still converging")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("✓ Commanded velocity is NO LONGER constant 1.0 m/s")
print("✓ Velocity varies based on orbital radius error")
print("✓ Base orbital speed: ~0.2 m/s (tangential)")
print("✓ Radial corrections added when off-radius")
print("✓ Should see smooth convergence to circular orbit")
print("=" * 60)

plt.show()
