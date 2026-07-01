"""
Test script for tuning orbital gains.

Allows easy testing of different parameter values to find optimal settings.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1


# ============================================================================
# TUNING PARAMETERS - ADJUST THESE!
# ============================================================================

# Orbital parameters
DESIRED_RADIUS = 0.15  # Desired orbital radius (m)
BASE_ORBITAL_SPEED = 0.2  # Tangential velocity (m/s)
RADIAL_GAIN = 0.5  # Gain for radius correction
MAX_RADIAL_SPEED = 0.3  # Maximum radial correction velocity (m/s)

# Angular control parameters
OMEGA_GAIN = 1.0  # Feedback gain for angle error
MAX_OMEGA = 1.5  # Maximum angular velocity (rad/s)
USE_FEEDFORWARD = True  # Enable/disable feedforward control

# Simulation parameters
SIM_TIME = 200  # Number of simulation steps
PLOT_INTERVAL = 20  # Print status every N steps

# ============================================================================


def custom_orbiter(cluster, desired_radius=DESIRED_RADIUS):
    """
    Custom orbital primitive with tunable parameters.

    This is a modified version of center_orbiter_quad_advanced that allows
    easy testing of different gain values.
    """
    from src.control.quad_primitives import estimate_center_and_radius

    # Estimate center and current radius using dual Jacobian
    center_estimate, current_radius = estimate_center_and_radius(cluster)

    if center_estimate is None:
        return 0.0, 0.0, 0.0

    # Get current centroid and formation
    current_centroid = cluster.get_centroid()
    formation = cluster.get_current_formation()
    current_theta_c = formation['th_c']

    # Vector from centroid to center
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        # Too close to center - move outward in random direction
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    # Normalize direction to center
    to_center_norm = to_center / center_distance

    # Desired angle: direction from centroid to center (d1 should point radially)
    desired_theta_raw = np.arctan2(to_center[1], to_center[0])

    # Unwrap desired_theta to be continuous with current_theta_c
    desired_theta = desired_theta_raw
    while (desired_theta - current_theta_c) > np.pi:
        desired_theta -= 2 * np.pi
    while (desired_theta - current_theta_c) < -np.pi:
        desired_theta += 2 * np.pi

    # Calculate angle error
    angle_error = desired_theta - current_theta_c

    # Angular velocity control: FEEDFORWARD + FEEDBACK
    if USE_FEEDFORWARD:
        # Feedforward: omega required to keep d1 radial during orbital motion
        omega_feedforward = BASE_ORBITAL_SPEED / center_distance if center_distance > 1e-6 else 0.0
    else:
        omega_feedforward = 0.0

    # Feedback: proportional correction for angle errors
    omega_feedback = OMEGA_GAIN * angle_error

    # Combine feedforward and feedback
    omega_c_raw = omega_feedforward + omega_feedback

    # Clamp to limits
    omega_c = np.clip(omega_c_raw, -MAX_OMEGA, MAX_OMEGA)

    # Perpendicular direction for orbit (90° rotation)
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Compute radius error
    radius_error = current_radius - desired_radius

    # Orbital speed control
    radial_speed = RADIAL_GAIN * radius_error

    # Clamp radial correction
    radial_speed = np.clip(radial_speed, -MAX_RADIAL_SPEED, MAX_RADIAL_SPEED)

    # Compute velocity components
    v_tangential = BASE_ORBITAL_SPEED * orbit_direction
    v_radial = radial_speed * to_center_norm

    # Combine tangential and radial components
    velocity_command = v_tangential + v_radial

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]

    return vx_c, vy_c, omega_c


def run_orbit_test():
    """Run orbital test with current parameter settings."""

    print("\n" + "="*80)
    print("ORBITAL GAIN TUNING TEST")
    print("="*80)
    print(f"\nCurrent settings:")
    print(f"  Desired radius: {DESIRED_RADIUS}")
    print(f"  Base orbital speed: {BASE_ORBITAL_SPEED} m/s")
    print(f"  Radial gain: {RADIAL_GAIN}")
    print(f"  Max radial speed: {MAX_RADIAL_SPEED} m/s")
    print(f"  Omega gain (feedback): {OMEGA_GAIN}")
    print(f"  Max omega: {MAX_OMEGA} rad/s")
    print(f"  Use feedforward: {USE_FEEDFORWARD}")
    print(f"\nExpected omega_orbit at radius {DESIRED_RADIUS}: {BASE_ORBITAL_SPEED/DESIRED_RADIUS:.3f} rad/s")
    print()

    # Create cluster with analytical sink field
    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    # Storage for analysis
    robot_paths = [[], [], [], []]  # Paths for each robot
    centroid_path = []
    theta_c_history = []
    omega_c_history = []
    angle_error_history = []
    radius_history = []

    for step in range(SIM_TIME):
        # Get current state
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

        # Store paths
        robot_paths[0].append([x1, y1])
        robot_paths[1].append([x2, y2])
        robot_paths[2].append([x3, y3])
        robot_paths[3].append([x4, y4])
        centroid_path.append([centroid[0], centroid[1]])

        # Calculate metrics
        current_radius = np.linalg.norm(centroid)
        direction_to_sink = -centroid
        desired_angle = np.arctan2(direction_to_sink[1], direction_to_sink[0])
        current_theta_c = formation['th_c']

        # Unwrap for comparison
        desired_theta = desired_angle
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi
        angle_error = desired_theta - current_theta_c

        # Store history
        theta_c_history.append(np.degrees(current_theta_c))
        radius_history.append(current_radius)
        angle_error_history.append(np.degrees(angle_error))

        # Move using custom orbiter
        vx_c, vy_c, omega_c = custom_orbiter(cluster, desired_radius=DESIRED_RADIUS)
        omega_c_history.append(omega_c)
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

        # Print status
        if step % PLOT_INTERVAL == 0:
            print(f"Step {step:3d}: r={current_radius:6.3f}, "
                  f"θ_err={np.degrees(angle_error):7.2f}°, "
                  f"ω_c={omega_c:6.3f} rad/s")

    # Final metrics
    final_centroid = cluster.get_centroid()
    final_radius = np.linalg.norm(final_centroid)
    avg_radius = np.mean(radius_history[-50:])  # Last 50 steps
    avg_angle_error = np.mean(np.abs(angle_error_history[-50:]))

    print(f"\nFinal state:")
    print(f"  Desired radius: {DESIRED_RADIUS:.4f}")
    print(f"  Final radius: {final_radius:.4f}")
    print(f"  Average radius (last 50 steps): {avg_radius:.4f}")
    print(f"  Average |angle error| (last 50 steps): {avg_angle_error:.2f}°")

    # Plot results
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Individual robot trajectories
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    colors = ['blue', 'yellow', 'green', 'red']
    labels = ['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4']

    for i, (path, color, label) in enumerate(zip(robot_paths, colors, labels)):
        path_array = np.array(path)
        ax1.plot(path_array[:, 0], path_array[:, 1], '-', color=color, linewidth=1.5,
                label=label, alpha=0.7)
        # Mark start and end
        ax1.plot(path_array[0, 0], path_array[0, 1], 'o', color=color, markersize=8)
        ax1.plot(path_array[-1, 0], path_array[-1, 1], 's', color=color, markersize=8)

    # Plot centroid path
    centroid_array = np.array(centroid_path)
    ax1.plot(centroid_array[:, 0], centroid_array[:, 1], 'k--', linewidth=2,
            label='Centroid', alpha=0.5)

    # Plot sink and desired orbit
    ax1.plot(0, 0, 'r*', markersize=20, label='Sink')
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax1.plot(DESIRED_RADIUS * np.cos(theta_circle), DESIRED_RADIUS * np.sin(theta_circle),
            'r--', linewidth=1, alpha=0.3, label=f'Desired orbit (r={DESIRED_RADIUS})')

    ax1.set_xlabel('X position (m)', fontsize=11)
    ax1.set_ylabel('Y position (m)', fontsize=11)
    ax1.set_title('Individual Robot Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Radius over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(radius_history, 'b-', linewidth=2)
    ax2.axhline(y=DESIRED_RADIUS, color='r', linestyle='--', linewidth=2, label='Desired')
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('Radius (m)', fontsize=10)
    ax2.set_title('Orbital Radius', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Angle error over time
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(angle_error_history, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Angle error (°)', fontsize=10)
    ax3.set_title('Angle Error', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: omega_c over time
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(omega_c_history, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=MAX_OMEGA, color='r', linestyle='--', alpha=0.3, label='Max')
    ax4.axhline(y=-MAX_OMEGA, color='r', linestyle='--', alpha=0.3)
    if USE_FEEDFORWARD:
        expected_omega = BASE_ORBITAL_SPEED / DESIRED_RADIUS
        ax4.axhline(y=expected_omega, color='orange', linestyle='--', alpha=0.5,
                   label=f'Expected FF ({expected_omega:.2f})')
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('omega_c (rad/s)', fontsize=10)
    ax4.set_title('Angular Velocity Command', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Plot 5: theta_c over time
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(theta_c_history, 'b-', linewidth=2, label='theta_c')
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('theta_c (°)', fontsize=10)
    ax5.set_title('Formation Orientation', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    summary_text = f"""
PARAMETER SETTINGS:
  Desired radius: {DESIRED_RADIUS}
  Orbital speed: {BASE_ORBITAL_SPEED} m/s
  Radial gain: {RADIAL_GAIN}
  Max radial speed: {MAX_RADIAL_SPEED} m/s
  Omega gain: {OMEGA_GAIN}
  Max omega: {MAX_OMEGA} rad/s
  Feedforward: {USE_FEEDFORWARD}

RESULTS:
  Avg radius: {avg_radius:.4f} m
  Radius error: {avg_radius - DESIRED_RADIUS:.4f} m
  Avg |angle error|: {avg_angle_error:.2f}°

Expected omega: {BASE_ORBITAL_SPEED/DESIRED_RADIUS:.3f} rad/s
Max omega limit: {MAX_OMEGA:.3f} rad/s
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Orbital Controller Tuning Test', fontsize=14, fontweight='bold')
    plt.savefig('test_orbit_gains.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: test_orbit_gains.png")
    plt.show()


if __name__ == '__main__':
    run_orbit_test()

    print("\n" + "="*80)
    print("To test different settings:")
    print("  1. Edit the TUNING PARAMETERS section at the top of this file")
    print("  2. Run the script again: python3 test_orbit_gains.py")
    print("="*80)
