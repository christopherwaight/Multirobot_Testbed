"""
Test angle wrapping, degrees/radians handling in advanced primitives.

This test specifically checks:
1. Angle unwrapping works correctly (e.g., 350° to 10° doesn't cause 340° error)
2. All angles are in radians throughout
3. Error is always the shortest angular distance
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1
import src.control.quad_primitives as qcp


def test_angle_wrapping():
    """Test that angle wrapping works correctly across all quadrants."""

    print("\n" + "="*80)
    print("ANGLE WRAPPING TEST")
    print("="*80)
    print("Testing angle error calculation across 0°/360° boundary")
    print()

    # Create cluster
    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    # Move cluster to specific position to test wrapping
    # Place centroid at (0.2, 0.0) so desired angle is 180° (pointing left to origin)
    cluster.reset(x_c=0.2, y_c=0.0)

    print("Initial setup:")
    formation = cluster.get_current_formation()
    centroid = cluster.get_centroid()
    print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print(f"  Initial theta_c: {np.degrees(formation['th_c']):.2f}° ({formation['th_c']:.4f} rad)")

    # Calculate what the desired angle should be
    direction_to_sink = np.array([0.0, 0.0]) - centroid
    desired_angle_raw = np.arctan2(direction_to_sink[1], direction_to_sink[0])
    print(f"  Desired angle (toward sink): {np.degrees(desired_angle_raw):.2f}° ({desired_angle_raw:.4f} rad)")
    print()

    # Track angles over time
    theta_c_deg_history = []
    theta_c_rad_history = []
    desired_angle_deg_history = []
    angle_error_deg_history = []
    omega_c_history = []
    steps_history = []

    # Run simulation
    sim_time = 150

    print("Running simulation...")
    print(f"{'Step':>5} {'theta_c(°)':>12} {'theta_c(rad)':>14} {'desired(°)':>12} {'error(°)':>10} {'omega_c':>10}")
    print("-" * 80)

    for step in range(sim_time):
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        current_theta_c = formation['th_c']

        # Calculate desired angle
        direction_to_sink = np.array([0.0, 0.0]) - centroid
        desired_angle_raw = np.arctan2(direction_to_sink[1], direction_to_sink[0])

        # Unwrap (same logic as in primitive)
        desired_theta = desired_angle_raw
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi

        angle_error = desired_theta - current_theta_c

        # Call the primitive
        vx_c, vy_c, omega_c = qcp.center_orbiter_quad_advanced(cluster, desired_radius=0.15)

        # Store data
        theta_c_deg_history.append(np.degrees(current_theta_c))
        theta_c_rad_history.append(current_theta_c)
        desired_angle_deg_history.append(np.degrees(desired_angle_raw))
        angle_error_deg_history.append(np.degrees(angle_error))
        omega_c_history.append(omega_c)
        steps_history.append(step)

        # Move cluster
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

        # Print every 10 steps
        if step % 10 == 0:
            print(f"{step:5d} {np.degrees(current_theta_c):12.2f} {current_theta_c:14.4f} "
                  f"{np.degrees(desired_angle_raw):12.2f} {np.degrees(angle_error):10.2f} {omega_c:10.4f}")

    print()
    print("Key observations to check:")
    print("  1. theta_c should continuously increase/decrease (unwrapped)")
    print("  2. Angle error should stay small (not jump to 340° at wrap point)")
    print("  3. omega_c should be smooth (no huge spikes)")
    print()

    # Check for angle wrapping issues
    max_angle_error = np.max(np.abs(angle_error_deg_history))
    max_omega_jump = np.max(np.abs(np.diff(omega_c_history)))

    print(f"Diagnostics:")
    print(f"  Max |angle error|: {max_angle_error:.2f}°")
    print(f"  Max omega_c jump between steps: {max_omega_jump:.4f} rad/s")

    if max_angle_error > 180:
        print("  ⚠️  WARNING: Angle error > 180° detected! Wrapping issue!")
    else:
        print("  ✓ Angle error stayed reasonable")

    if max_omega_jump > 1.0:
        print("  ⚠️  WARNING: Large omega jump detected! Possible wrapping discontinuity!")
    else:
        print("  ✓ Omega_c changed smoothly")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Angle Wrapping Test', fontsize=14, fontweight='bold')

    # Plot 1: theta_c over time (degrees, unwrapped)
    ax = axes[0, 0]
    ax.plot(steps_history, theta_c_deg_history, 'b-', linewidth=2, label='theta_c (degrees)')
    ax.plot(steps_history, desired_angle_deg_history, 'r--', linewidth=2, label='desired angle')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Formation Angle (Unwrapped)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: theta_c in radians
    ax = axes[0, 1]
    ax.plot(steps_history, theta_c_rad_history, 'b-', linewidth=2, label='theta_c (radians)')
    ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.3, label='π')
    ax.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.3, label='-π')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle (radians)')
    ax.set_title('Formation Angle (Radians)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Angle error
    ax = axes[1, 0]
    ax.plot(steps_history, angle_error_deg_history, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=180, color='r', linestyle='--', alpha=0.3, label='±180° (wrap point)')
    ax.axhline(y=-180, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle Error (degrees)')
    ax.set_title('Angle Error (should stay < 180°)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: omega_c
    ax = axes[1, 1]
    ax.plot(steps_history, omega_c_history, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.3, label='Max omega')
    ax.axhline(y=-1.5, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('omega_c (rad/s)')
    ax.set_title('Angular Velocity Command (should be smooth)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_angle_wrapping.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: test_angle_wrapping.png")
    plt.show()


def test_all_quadrants():
    """Test angle calculation in all quadrants."""

    print("\n" + "="*80)
    print("ALL QUADRANTS TEST")
    print("="*80)
    print("Testing angle error calculation from all 4 quadrants")
    print()

    field = AnalyticalField(sink1)

    test_positions = [
        (0.2, 0.0, "Quadrant 1/4 boundary (0°)"),
        (0.0, 0.2, "Quadrant 1 (90°)"),
        (-0.2, 0.0, "Quadrant 2/3 boundary (180°)"),
        (0.0, -0.2, "Quadrant 4 (270°/-90°)"),
        (0.15, 0.15, "Quadrant 1 (45°)"),
        (-0.15, 0.15, "Quadrant 2 (135°)"),
        (-0.15, -0.15, "Quadrant 3 (-135°)"),
        (0.15, -0.15, "Quadrant 4 (-45°)"),
    ]

    print(f"{'Position':>25} {'Expected Angle':>15} {'Initial theta_c':>15} {'Initial Error':>15}")
    print("-" * 80)

    for x_c, y_c, description in test_positions:
        cluster = QuadCluster('config/formations/quad_square_default.yaml', field)
        cluster.reset(x_c=x_c, y_c=y_c)

        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        current_theta_c = formation['th_c']

        # Expected angle toward origin
        direction_to_sink = np.array([0.0, 0.0]) - centroid
        expected_angle = np.arctan2(direction_to_sink[1], direction_to_sink[0])

        # Calculate error with unwrapping
        desired_theta = expected_angle
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi
        angle_error = desired_theta - current_theta_c

        print(f"{description:>25} {np.degrees(expected_angle):>14.2f}° "
              f"{np.degrees(current_theta_c):>14.2f}° {np.degrees(angle_error):>14.2f}°")

    print()
    print("All errors should be reasonable (< 180°)")


if __name__ == '__main__':
    print("Testing Angle Handling in Advanced Primitives")
    print("="*80)

    # Test 1: Wrapping during orbit
    test_angle_wrapping()

    # Test 2: All quadrants
    test_all_quadrants()

    print("\n" + "="*80)
    print("Tests complete!")
    print("="*80)
