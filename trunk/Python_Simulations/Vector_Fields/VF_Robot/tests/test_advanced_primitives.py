"""
Test script for advanced 4-robot primitives with orientation control.

Tests:
1. dual_jacobian_center_finder_advanced - approach with rotation
2. center_orbiter_quad_advanced - orbit with rotation

Both primitives should orient d1 diagonal (robot 1→3) to point toward the critical point.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
import src.control.quad_primitives as qcp


def test_approach_with_rotation():
    """Test dual_jacobian_center_finder_advanced."""
    print("\n" + "="*80)
    print("TEST 1: dual_jacobian_center_finder_advanced")
    print("="*80)
    print("Expected behavior: Formation should move toward sink at (0,0)")
    print("                   and rotate so d1 (robot 1→3) points toward sink")
    print()

    # Create cluster with analytical sink field (center at origin)
    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    # Run simulation for 100 steps
    sim_time = 100

    # Store data for analysis
    theta_c_history = []
    desired_angle_history = []
    angle_error_history = []
    omega_c_history = []

    for step in range(sim_time):
        # Get current state
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        current_theta_c = formation['th_c']

        # Calculate what the desired angle should be (toward origin)
        direction_to_sink = np.array([0.0, 0.0]) - centroid
        desired_angle = np.arctan2(direction_to_sink[1], direction_to_sink[0])

        # Store for analysis
        theta_c_history.append(np.degrees(current_theta_c))
        desired_angle_history.append(np.degrees(desired_angle))

        # Calculate angle error (what the primitive should compute)
        # Unwrap for comparison
        desired_theta = desired_angle
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi
        angle_error = desired_theta - current_theta_c
        angle_error_history.append(np.degrees(angle_error))

        # Move using advanced primitive
        vx_c, vy_c, omega_c = qcp.dual_jacobian_center_finder_advanced(cluster)
        omega_c_history.append(omega_c)
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

        # Print every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: centroid=({centroid[0]:6.3f}, {centroid[1]:6.3f}), "
                  f"theta_c={np.degrees(current_theta_c):7.2f}°, "
                  f"desired={np.degrees(desired_angle):7.2f}°, "
                  f"error={np.degrees(angle_error):7.2f}°, "
                  f"omega_c={omega_c:6.3f}")

    # Final state
    final_formation = cluster.get_current_formation()
    final_centroid = cluster.get_centroid()
    final_distance = np.linalg.norm(final_centroid)

    print(f"\nFinal state:")
    print(f"  Distance to sink: {final_distance:.4f}")
    print(f"  Final theta_c: {np.degrees(final_formation['th_c']):.2f}°")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test 1: dual_jacobian_center_finder_advanced', fontsize=14, fontweight='bold')

    # Plot 1: Trajectory
    ax = axes[0, 0]
    trajectory = cluster.get_center_history()
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Centroid trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=20, label='Sink (target)')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')

    # Plot final formation
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    ax.plot([x1, x3], [y1, y3], 'r-', linewidth=2, label='d1 diagonal (1→3)')
    ax.plot([x2, x4], [y2, y4], 'g-', linewidth=1, label='d2 diagonal (2→4)')
    ax.scatter([x1, x2, x3, x4], [y1, y2, y3, y4], c=['blue', 'yellow', 'green', 'red'], s=100, zorder=5)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Trajectory and Final Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: Angles over time
    ax = axes[0, 1]
    ax.plot(theta_c_history, 'b-', linewidth=2, label='Actual theta_c')
    ax.plot(desired_angle_history, 'r--', linewidth=2, label='Desired angle (toward sink)')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Formation Orientation vs Desired')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Angle error over time
    ax = axes[1, 0]
    ax.plot(angle_error_history, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Angle error (degrees)')
    ax.set_title('Angle Error (desired - actual)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Angular velocity (omega_c) over time
    ax = axes[1, 1]
    ax.plot(omega_c_history, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Max omega_c')
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('omega_c (rad/s)')
    ax.set_title('Angular Velocity Command')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_approach_with_rotation.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: test_approach_with_rotation.png")
    plt.show()

    return cluster


def test_orbit_with_rotation():
    """Test center_orbiter_quad_advanced."""
    print("\n" + "="*80)
    print("TEST 2: center_orbiter_quad_advanced")
    print("="*80)
    print("Expected behavior: Formation should orbit around sink at (0,0)")
    print("                   and rotate so d1 (robot 1→3) points radially toward sink")
    print()

    # Create cluster with analytical sink field (center at origin)
    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    # Run simulation for 150 steps
    sim_time = 150
    desired_radius = 0.15

    # Store data for analysis
    theta_c_history = []
    desired_angle_history = []
    angle_error_history = []
    omega_c_history = []
    radius_history = []

    for step in range(sim_time):
        # Get current state
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        current_theta_c = formation['th_c']

        # Calculate what the desired angle should be (radially toward origin)
        direction_to_sink = np.array([0.0, 0.0]) - centroid
        current_radius = np.linalg.norm(direction_to_sink)
        desired_angle = np.arctan2(direction_to_sink[1], direction_to_sink[0])

        # Store for analysis
        theta_c_history.append(np.degrees(current_theta_c))
        desired_angle_history.append(np.degrees(desired_angle))
        radius_history.append(current_radius)

        # Calculate angle error (what the primitive should compute)
        # Unwrap for comparison
        desired_theta = desired_angle
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi
        angle_error = desired_theta - current_theta_c
        angle_error_history.append(np.degrees(angle_error))

        # Move using advanced primitive
        vx_c, vy_c, omega_c = qcp.center_orbiter_quad_advanced(cluster, desired_radius=desired_radius)
        omega_c_history.append(omega_c)
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

        # Print every 15 steps
        if step % 15 == 0:
            print(f"Step {step:3d}: radius={current_radius:6.3f}, "
                  f"theta_c={np.degrees(current_theta_c):7.2f}°, "
                  f"desired={np.degrees(desired_angle):7.2f}°, "
                  f"error={np.degrees(angle_error):7.2f}°, "
                  f"omega_c={omega_c:6.3f}")

    # Final state
    final_centroid = cluster.get_centroid()
    final_radius = np.linalg.norm(final_centroid)

    print(f"\nFinal state:")
    print(f"  Desired radius: {desired_radius:.4f}")
    print(f"  Actual radius: {final_radius:.4f}")
    print(f"  Radius error: {final_radius - desired_radius:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Test 2: center_orbiter_quad_advanced', fontsize=14, fontweight='bold')

    # Plot 1: Trajectory
    ax = axes[0, 0]
    trajectory = cluster.get_center_history()
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Centroid trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=20, label='Sink (target)')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=8, label='End')

    # Plot desired orbit radius
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(desired_radius * np.cos(theta_circle), desired_radius * np.sin(theta_circle),
            'r--', linewidth=1, alpha=0.5, label=f'Desired orbit (r={desired_radius})')

    # Plot final formation
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    ax.plot([x1, x3], [y1, y3], 'r-', linewidth=2, label='d1 diagonal (1→3)')
    ax.plot([x2, x4], [y2, y4], 'g-', linewidth=1, label='d2 diagonal (2→4)')
    ax.scatter([x1, x2, x3, x4], [y1, y2, y3, y4], c=['blue', 'yellow', 'green', 'red'], s=100, zorder=5)

    # Draw line from centroid to origin (should align with d1)
    final_centroid = cluster.get_centroid()
    ax.plot([final_centroid[0], 0], [final_centroid[1], 0], 'k--', linewidth=1, alpha=0.5, label='Radial direction')

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Orbital Trajectory and Final Formation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: Angles over time
    ax = axes[0, 1]
    ax.plot(theta_c_history, 'b-', linewidth=2, label='Actual theta_c')
    ax.plot(desired_angle_history, 'r--', linewidth=2, label='Desired angle (radial)')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Formation Orientation vs Radial Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Angle error and radius over time
    ax = axes[1, 0]
    ax2 = ax.twinx()

    line1 = ax.plot(angle_error_history, 'g-', linewidth=2, label='Angle error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Angle error (degrees)', color='g')
    ax.tick_params(axis='y', labelcolor='g')

    line2 = ax2.plot(radius_history, 'b-', linewidth=2, label='Actual radius')
    ax2.axhline(y=desired_radius, color='r', linestyle='--', alpha=0.5, label='Desired radius')
    ax2.set_ylabel('Radius', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')

    ax.set_title('Angle Error and Orbital Radius')
    ax.grid(True, alpha=0.3)

    # Plot 4: Angular velocity (omega_c) over time
    ax = axes[1, 1]
    ax.plot(omega_c_history, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Max omega_c')
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('omega_c (rad/s)')
    ax.set_title('Angular Velocity Command')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_orbit_with_rotation.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: test_orbit_with_rotation.png")
    plt.show()

    return cluster


if __name__ == '__main__':
    print("Testing Advanced 4-Robot Primitives with Orientation Control")
    print("=" * 80)
    print("These tests verify that:")
    print("  1. The formation moves toward/orbits the critical point")
    print("  2. The d1 diagonal (robot 1→3) rotates to point toward the critical point")
    print("  3. The angular gain kp=1.0 properly controls omega_c")
    print()

    # Test 1: Approach with rotation
    test_approach_with_rotation()

    # Test 2: Orbit with rotation
    test_orbit_with_rotation()

    print("\n" + "="*80)
    print("Tests complete!")
    print("="*80)
    print("\nLook for:")
    print("  - In Test 1: theta_c should converge to desired angle (toward sink)")
    print("  - In Test 2: theta_c should track the radial direction during orbit")
    print("  - In both: omega_c should be non-zero when there's angle error")
    print("  - In both: angle_error should decrease over time")
