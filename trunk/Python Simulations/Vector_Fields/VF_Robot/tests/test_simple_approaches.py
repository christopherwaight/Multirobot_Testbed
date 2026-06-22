"""
Test different simple approaches WITHOUT feedforward/feedback complexity.

Tests 4 different strategies:
1. No omega control at all (omega_c = 0)
2. Simple proportional only (omega_c = kp * error, no clamping)
3. Fixed rotation rate (omega_c = constant)
4. No velocity clamping (remove translational limits)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1


def approach_1_no_omega(cluster, desired_radius=0.15):
    """Approach 1: No omega control - let formation drift naturally."""
    from src.control.quad_primitives import estimate_center_and_radius

    center_estimate, current_radius = estimate_center_and_radius(cluster)
    if center_estimate is None:
        return 0.0, 0.0, 0.0

    current_centroid = cluster.get_centroid()
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    to_center_norm = to_center / center_distance
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Radius control
    radius_error = current_radius - desired_radius
    radial_speed = 0.5 * radius_error
    radial_speed = np.clip(radial_speed, -0.3, 0.3)

    # Velocity
    base_orbital_speed = 0.2
    v_tangential = base_orbital_speed * orbit_direction
    v_radial = radial_speed * to_center_norm
    velocity_command = v_tangential + v_radial

    # NO OMEGA CONTROL
    omega_c = 0.0

    return velocity_command[0], velocity_command[1], omega_c


def approach_2_simple_proportional(cluster, desired_radius=0.15):
    """Approach 2: Simple proportional omega, NO CLAMPING."""
    from src.control.quad_primitives import estimate_center_and_radius

    center_estimate, current_radius = estimate_center_and_radius(cluster)
    if center_estimate is None:
        return 0.0, 0.0, 0.0

    current_centroid = cluster.get_centroid()
    formation = cluster.get_current_formation()
    current_theta_c = formation['th_c']

    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    to_center_norm = to_center / center_distance
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Desired angle
    desired_theta_raw = np.arctan2(to_center[1], to_center[0])
    desired_theta = desired_theta_raw
    while (desired_theta - current_theta_c) > np.pi:
        desired_theta -= 2 * np.pi
    while (desired_theta - current_theta_c) < -np.pi:
        desired_theta += 2 * np.pi

    angle_error = desired_theta - current_theta_c

    # Simple proportional - NO CLAMPING
    kp = 1.0
    omega_c = kp * angle_error

    # Radius control
    radius_error = current_radius - desired_radius
    radial_speed = 0.5 * radius_error
    radial_speed = np.clip(radial_speed, -0.3, 0.3)

    # Velocity
    base_orbital_speed = 0.2
    v_tangential = base_orbital_speed * orbit_direction
    v_radial = radial_speed * to_center_norm
    velocity_command = v_tangential + v_radial

    return velocity_command[0], velocity_command[1], omega_c


def approach_3_fixed_omega(cluster, desired_radius=0.15):
    """Approach 3: Fixed rotation rate."""
    from src.control.quad_primitives import estimate_center_and_radius

    center_estimate, current_radius = estimate_center_and_radius(cluster)
    if center_estimate is None:
        return 0.0, 0.0, 0.0

    current_centroid = cluster.get_centroid()
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    to_center_norm = to_center / center_distance
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # FIXED omega (calculated for desired radius)
    base_orbital_speed = 0.2
    omega_c = base_orbital_speed / desired_radius  # ~1.33 rad/s

    # Radius control
    radius_error = current_radius - desired_radius
    radial_speed = 0.5 * radius_error
    radial_speed = np.clip(radial_speed, -0.3, 0.3)

    # Velocity
    v_tangential = base_orbital_speed * orbit_direction
    v_radial = radial_speed * to_center_norm
    velocity_command = v_tangential + v_radial

    return velocity_command[0], velocity_command[1], omega_c


def approach_4_no_clamping(cluster, desired_radius=0.15):
    """Approach 4: Remove ALL clamping (velocity and omega)."""
    from src.control.quad_primitives import estimate_center_and_radius

    center_estimate, current_radius = estimate_center_and_radius(cluster)
    if center_estimate is None:
        return 0.0, 0.0, 0.0

    current_centroid = cluster.get_centroid()
    formation = cluster.get_current_formation()
    current_theta_c = formation['th_c']

    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    to_center_norm = to_center / center_distance
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Desired angle
    desired_theta_raw = np.arctan2(to_center[1], to_center[0])
    desired_theta = desired_theta_raw
    while (desired_theta - current_theta_c) > np.pi:
        desired_theta -= 2 * np.pi
    while (desired_theta - current_theta_c) < -np.pi:
        desired_theta += 2 * np.pi

    angle_error = desired_theta - current_theta_c

    # Simple proportional - NO CLAMPING
    kp_omega = 1.0
    omega_c = kp_omega * angle_error  # NO MAX LIMIT

    # Radius control - NO CLAMPING
    kp_radial = 0.5
    radius_error = current_radius - desired_radius
    radial_speed = kp_radial * radius_error  # NO MAX LIMIT

    # Velocity - NO CLAMPING
    base_orbital_speed = 0.2
    v_tangential = base_orbital_speed * orbit_direction
    v_radial = radial_speed * to_center_norm
    velocity_command = v_tangential + v_radial  # NO MAX LIMIT

    return velocity_command[0], velocity_command[1], omega_c


def run_test(approach_func, approach_name, sim_time=200):
    """Run test for a given approach."""
    print(f"\n{'='*80}")
    print(f"Testing: {approach_name}")
    print('='*80)

    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    robot_paths = [[], [], [], []]
    centroid_path = []
    omega_c_history = []
    angle_error_history = []
    radius_history = []

    for step in range(sim_time):
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

        robot_paths[0].append([x1, y1])
        robot_paths[1].append([x2, y2])
        robot_paths[2].append([x3, y3])
        robot_paths[3].append([x4, y4])
        centroid_path.append([centroid[0], centroid[1]])

        current_radius = np.linalg.norm(centroid)
        direction_to_sink = -centroid
        desired_angle = np.arctan2(direction_to_sink[1], direction_to_sink[0])
        current_theta_c = formation['th_c']

        desired_theta = desired_angle
        while (desired_theta - current_theta_c) > np.pi:
            desired_theta -= 2 * np.pi
        while (desired_theta - current_theta_c) < -np.pi:
            desired_theta += 2 * np.pi
        angle_error = desired_theta - current_theta_c

        radius_history.append(current_radius)
        angle_error_history.append(np.degrees(angle_error))

        vx_c, vy_c, omega_c = approach_func(cluster)
        omega_c_history.append(omega_c)
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

        if step % 40 == 0:
            print(f"Step {step:3d}: r={current_radius:6.3f}, "
                  f"θ_err={np.degrees(angle_error):7.2f}°, "
                  f"ω_c={omega_c:7.3f} rad/s")

    avg_radius = np.mean(radius_history[-50:])
    avg_angle_error = np.mean(np.abs(angle_error_history[-50:]))
    max_omega = np.max(np.abs(omega_c_history))
    avg_omega = np.mean(np.abs(omega_c_history[-50:]))

    print(f"\nResults:")
    print(f"  Avg radius (last 50): {avg_radius:.4f} m (desired: 0.15)")
    print(f"  Avg |angle error|: {avg_angle_error:.2f}°")
    print(f"  Max |omega_c|: {max_omega:.3f} rad/s")
    print(f"  Avg |omega_c| (last 50): {avg_omega:.3f} rad/s")

    return robot_paths, centroid_path, omega_c_history, angle_error_history, radius_history


def plot_comparison(results_list, names):
    """Plot comparison of all approaches."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Comparison: Simple Control Approaches (No Feedforward)',
                 fontsize=14, fontweight='bold')

    colors = ['blue', 'yellow', 'green', 'red']

    for idx, (results, name) in enumerate(zip(results_list, names)):
        robot_paths, centroid_path, omega_c_history, angle_error_history, radius_history = results

        # Top row: trajectories
        ax = axes[0, idx]
        for i, (path, color) in enumerate(zip(robot_paths, colors)):
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], '-', color=color,
                   linewidth=1.5, alpha=0.7)

        centroid_array = np.array(centroid_path)
        ax.plot(centroid_array[:, 0], centroid_array[:, 1], 'k--', linewidth=1.5, alpha=0.5)
        ax.plot(0, 0, 'r*', markersize=15)

        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.15 * np.cos(theta_circle), 0.15 * np.sin(theta_circle),
                'r--', linewidth=1, alpha=0.3)

        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])

        # Bottom row: metrics
        ax = axes[1, idx]
        ax2 = ax.twinx()

        line1 = ax.plot(angle_error_history, 'g-', linewidth=1.5, label='Angle error')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Angle error (°)', color='g', fontsize=9)
        ax.tick_params(axis='y', labelcolor='g')

        line2 = ax2.plot(omega_c_history, 'purple', linewidth=1.5, label='omega_c')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('omega_c (rad/s)', color='purple', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='purple')

        ax.grid(True, alpha=0.3)

        # Stats
        avg_radius = np.mean(radius_history[-50:])
        avg_angle_error = np.mean(np.abs(angle_error_history[-50:]))
        avg_omega = np.mean(np.abs(omega_c_history[-50:]))

        ax.text(0.05, 0.95, f'r={avg_radius:.3f}\nθ_err={avg_angle_error:.1f}°\nω={avg_omega:.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('test_simple_approaches.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: test_simple_approaches.png")
    plt.show()


if __name__ == '__main__':
    print("="*80)
    print("Testing Simple Approaches - NO Feedforward/Feedback Complexity")
    print("="*80)

    approaches = [
        (approach_1_no_omega, "1: No Omega Control"),
        (approach_2_simple_proportional, "2: Simple kp*error (no clamp)"),
        (approach_3_fixed_omega, "3: Fixed Omega Rate"),
        (approach_4_no_clamping, "4: No Clamping At All"),
    ]

    results_list = []
    names = []

    for approach_func, approach_name in approaches:
        results = run_test(approach_func, approach_name, sim_time=200)
        results_list.append(results)
        names.append(approach_name)

    plot_comparison(results_list, names)

    print("\n" + "="*80)
    print("SUMMARY:")
    print("  Approach 1: omega_c = 0 (no rotation control)")
    print("  Approach 2: omega_c = kp * error (no max limit)")
    print("  Approach 3: omega_c = v_orbital / r_desired (constant)")
    print("  Approach 4: No clamping on velocity OR omega")
    print("="*80)
