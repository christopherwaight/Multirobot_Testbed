"""
Comprehensive orbiter test: Far-to-close and close-to-far
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
print("COMPREHENSIVE ORBITER TEST")
print("=" * 60)

desired_radius = 0.5  # meters (as requested)

# Test 1: Starting FAR away (converge inward)
print("\n[TEST 1] Starting FAR from center, converging INWARD to orbit")
print("-" * 60)

field1 = AnalyticalField(vortex1)
cluster1 = OmniCluster("config/formations/equilateral_default.yaml", field1)

# Reset cluster to start FAR away (1.5m from center)
cluster1.reset(x_c=1.5, y_c=0.0)
print(f"  Initial position: (1.5, 0.0)")
print(f"  Initial distance: 1.5 m")
print(f"  Target radius: {desired_radius} m")
print(f"  Should move INWARD to reach orbit")

positions1 = []
distances1 = []
velocities1 = []

for step in range(150):
    centroid = cluster1.get_centroid()
    positions1.append(centroid.copy())
    distance = np.linalg.norm(centroid)
    distances1.append(distance)

    cluster1.move(lambda c: ocp.critical_point_orbiter_plane_fitting(c, desired_radius=desired_radius))

    robot_vels = [robot.get_velocity() for robot in cluster1.robots]
    avg_vel = np.mean([np.linalg.norm(v) for v in robot_vels])
    velocities1.append(avg_vel)

positions1 = np.array(positions1)
distances1 = np.array(distances1)
print(f"  Final distance: {distances1[-1]:.4f} m")
print(f"  Final error: {abs(distances1[-1] - desired_radius):.4f} m")

# Test 2: Starting CLOSE to center (move outward)
print("\n[TEST 2] Starting CLOSE to center, moving OUTWARD to orbit")
print("-" * 60)

field2 = AnalyticalField(vortex1)
cluster2 = OmniCluster("config/formations/equilateral_default.yaml", field2)

# Reset cluster to start CLOSE (0.1m from center)
cluster2.reset(x_c=0.1, y_c=0.0)
print(f"  Initial position: (0.1, 0.0)")
print(f"  Initial distance: 0.1 m")
print(f"  Target radius: {desired_radius} m")
print(f"  Should move OUTWARD to reach orbit")

positions2 = []
distances2 = []
velocities2 = []

for step in range(150):
    centroid = cluster2.get_centroid()
    positions2.append(centroid.copy())
    distance = np.linalg.norm(centroid)
    distances2.append(distance)

    cluster2.move(lambda c: ocp.critical_point_orbiter_plane_fitting(c, desired_radius=desired_radius))

    robot_vels = [robot.get_velocity() for robot in cluster2.robots]
    avg_vel = np.mean([np.linalg.norm(v) for v in robot_vels])
    velocities2.append(avg_vel)

positions2 = np.array(positions2)
distances2 = np.array(distances2)
print(f"  Final distance: {distances2[-1]:.4f} m")
print(f"  Final error: {abs(distances2[-1] - desired_radius):.4f} m")

# Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Orbiter Tests: Desired Radius = {desired_radius} m', fontsize=16, fontweight='bold')

time = np.arange(150) * 0.1

# Row 1: Test 1 (Far to close)
# Trajectory
ax1 = axes[0, 0]
ax1.plot(positions1[:, 0], positions1[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
ax1.scatter(positions1[0, 0], positions1[0, 1], color='green', s=150, marker='o', label='Start', zorder=5)
ax1.scatter(positions1[-1, 0], positions1[-1, 1], color='red', s=150, marker='x', label='End', zorder=5)
ax1.scatter(0, 0, color='black', s=150, marker='*', label='Center', zorder=5)
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(desired_radius*np.cos(theta), desired_radius*np.sin(theta), 'r--', alpha=0.5, label=f'Target ({desired_radius}m)')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('[TEST 1] Trajectory: FAR → Orbit', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Distance over time
ax2 = axes[0, 1]
ax2.plot(time, distances1, 'b-', linewidth=2, label='Actual Distance')
ax2.axhline(y=desired_radius, color='r', linestyle='--', linewidth=2, label=f'Target ({desired_radius}m)', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance (m)')
ax2.set_title('[TEST 1] Distance Convergence', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Velocity over time
ax3 = axes[0, 2]
ax3.plot(time, velocities1, 'g-', linewidth=2)
ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Max (0.3 m/s)')
ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Base Orbit (0.2 m/s)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('[TEST 1] Velocity', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2: Test 2 (Close to far)
# Trajectory
ax4 = axes[1, 0]
ax4.plot(positions2[:, 0], positions2[:, 1], 'purple', linewidth=2, alpha=0.7, label='Trajectory')
ax4.scatter(positions2[0, 0], positions2[0, 1], color='green', s=150, marker='o', label='Start', zorder=5)
ax4.scatter(positions2[-1, 0], positions2[-1, 1], color='red', s=150, marker='x', label='End', zorder=5)
ax4.scatter(0, 0, color='black', s=150, marker='*', label='Center', zorder=5)
ax4.plot(desired_radius*np.cos(theta), desired_radius*np.sin(theta), 'r--', alpha=0.5, label=f'Target ({desired_radius}m)')
ax4.set_xlabel('x (m)')
ax4.set_ylabel('y (m)')
ax4.set_title('[TEST 2] Trajectory: CLOSE → Orbit', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

# Distance over time
ax5 = axes[1, 1]
ax5.plot(time, distances2, 'purple', linewidth=2, label='Actual Distance')
ax5.axhline(y=desired_radius, color='r', linestyle='--', linewidth=2, label=f'Target ({desired_radius}m)', alpha=0.7)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Distance (m)')
ax5.set_title('[TEST 2] Distance Convergence', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Velocity over time
ax6 = axes[1, 2]
ax6.plot(time, velocities2, 'g-', linewidth=2)
ax6.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Max (0.3 m/s)')
ax6.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Base Orbit (0.2 m/s)')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Velocity (m/s)')
ax6.set_title('[TEST 2] Velocity', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('velocity_plots/orbiter_comprehensive_test.png', dpi=150)
print("\n" + "=" * 60)
print("Plot saved: velocity_plots/orbiter_comprehensive_test.png")
print("=" * 60)

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print("=" * 60)
print(f"[TEST 1] Far → Orbit:")
print(f"  Start distance: 1.500 m")
print(f"  End distance:   {distances1[-1]:.3f} m")
print(f"  Final error:    {abs(distances1[-1] - desired_radius):.3f} m")
print(f"  Last 20 steps std: {np.std(distances1[-20:]):.4f} m")

print(f"\n[TEST 2] Close → Orbit:")
print(f"  Start distance: 0.100 m")
print(f"  End distance:   {distances2[-1]:.3f} m")
print(f"  Final error:    {abs(distances2[-1] - desired_radius):.3f} m")
print(f"  Last 20 steps std: {np.std(distances2[-20:]):.4f} m")

# Success criteria
success1 = abs(distances1[-1] - desired_radius) < 0.1
success2 = abs(distances2[-1] - desired_radius) < 0.1

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"[TEST 1] FAR → Orbit:   {'✓ PASS' if success1 else '✗ FAIL'}")
print(f"[TEST 2] CLOSE → Orbit: {'✓ PASS' if success2 else '✗ FAIL'}")
print("=" * 60)

plt.show()
