"""
Test different omega_gain values to find optimal setting.

Tests omega_gain = [0.5, 1.0, 2.0, 3.0, 5.0] to see which eliminates lag.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Sink import sink1


def test_orbiter_with_gain(omega_gain, max_omega, desired_radius=0.15, sim_time=200):
    """Test orbiter with specific omega_gain value."""
    from src.control.quad_primitives import estimate_center_and_radius

    def custom_orbiter(cluster):
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

        # Angular control with specified gain
        omega_c = np.clip(omega_gain * angle_error, -max_omega, max_omega)

        # Radius control
        radius_error = current_radius - desired_radius
        base_orbital_speed = 0.2
        radial_gain = 0.5
        radial_speed = radial_gain * radius_error
        radial_speed = np.clip(radial_speed, -0.3, 0.3)

        # Velocity
        v_tangential = base_orbital_speed * orbit_direction
        v_radial = radial_speed * to_center_norm
        velocity_command = v_tangential + v_radial

        return velocity_command[0], velocity_command[1], omega_c

    # Run simulation
    field = AnalyticalField(sink1)
    cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    robot_paths = [[], [], [], []]
    angle_error_history = []
    omega_c_history = []
    radius_history = []

    for step in range(sim_time):
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

        robot_paths[0].append([x1, y1])
        robot_paths[1].append([x2, y2])
        robot_paths[2].append([x3, y3])
        robot_paths[3].append([x4, y4])

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

        vx_c, vy_c, omega_c = custom_orbiter(cluster)
        omega_c_history.append(omega_c)
        cluster.move(lambda c: (vx_c, vy_c, omega_c))

    # Calculate metrics
    avg_radius = np.mean(radius_history[-50:])
    avg_angle_error = np.mean(np.abs(angle_error_history[-50:]))
    std_angle_error = np.std(angle_error_history[-50:])
    avg_omega = np.mean(omega_c_history[-50:])
    max_omega_used = np.max(np.abs(omega_c_history))

    return {
        'robot_paths': robot_paths,
        'angle_error_history': angle_error_history,
        'omega_c_history': omega_c_history,
        'radius_history': radius_history,
        'avg_radius': avg_radius,
        'avg_angle_error': avg_angle_error,
        'std_angle_error': std_angle_error,
        'avg_omega': avg_omega,
        'max_omega_used': max_omega_used,
    }


def main():
    print("="*80)
    print("OMEGA GAIN SWEEP TEST")
    print("="*80)
    print("Testing different omega_gain values to find optimal setting")
    print()

    # Test different gains
    test_gains = [0.5, 1.0, 2.0, 3.0, 5.0]
    max_omega = 3.0  # Increased limit to allow higher gains to work

    print(f"Parameters:")
    print(f"  Desired radius: 0.15 m")
    print(f"  Orbital speed: 0.2 m/s")
    print(f"  Expected omega (v/r): {0.2/0.15:.3f} rad/s")
    print(f"  Max omega limit: {max_omega} rad/s")
    print(f"  Testing gains: {test_gains}")
    print()

    results = {}

    for gain in test_gains:
        print(f"Testing omega_gain = {gain}...")
        result = test_orbiter_with_gain(omega_gain=gain, max_omega=max_omega)
        results[gain] = result

        print(f"  Avg radius: {result['avg_radius']:.4f} m")
        print(f"  Avg |angle error|: {result['avg_angle_error']:.2f}°")
        print(f"  Std angle error: {result['std_angle_error']:.2f}°")
        print(f"  Avg omega_c: {result['avg_omega']:.3f} rad/s")
        print(f"  Max |omega_c| used: {result['max_omega_used']:.3f} rad/s")
        print()

    # Plot comparison
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.35)

    colors = ['blue', 'yellow', 'green', 'red']

    # Top row: Robot trajectories for each gain
    for idx, gain in enumerate(test_gains):
        ax = fig.add_subplot(gs[0, idx])
        result = results[gain]

        for i, (path, color) in enumerate(zip(result['robot_paths'], colors)):
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], '-', color=color,
                   linewidth=1, alpha=0.7)

        # Desired orbit
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.15 * np.cos(theta_circle), 0.15 * np.sin(theta_circle),
                'r--', linewidth=1, alpha=0.3)
        ax.plot(0, 0, 'r*', markersize=10)

        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(f'Gain = {gain}\nAvg r={result["avg_radius"]:.3f}',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])

    # Middle row: Angle error for each gain
    for idx, gain in enumerate(test_gains):
        ax = fig.add_subplot(gs[1, idx])
        result = results[gain]

        ax.plot(result['angle_error_history'], 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Angle Error (°)', fontsize=9)
        ax.set_title(f'Avg |err|={result["avg_angle_error"]:.1f}°', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-100, 100])

    # Bottom row: omega_c for each gain
    for idx, gain in enumerate(test_gains):
        ax = fig.add_subplot(gs[2, idx])
        result = results[gain]

        ax.plot(result['omega_c_history'], 'purple', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=max_omega, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=-max_omega, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=1.33, color='orange', linestyle='--', alpha=0.3, linewidth=1,
                  label='Expected')
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('omega_c (rad/s)', fontsize=9)
        ax.set_title(f'Avg ω={result["avg_omega"]:.2f} rad/s', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_ylim([-3.5, 3.5])

    plt.suptitle('Omega Gain Sweep: Finding Optimal Angular Gain',
                fontsize=14, fontweight='bold')
    plt.savefig('test_gain_sweep.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: test_gain_sweep.png")
    plt.show()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Gain':>6} {'Avg |error|(°)':>15} {'Std error(°)':>15} {'Avg omega':>12} {'Max omega':>12}")
    print("-"*80)
    for gain in test_gains:
        result = results[gain]
        print(f"{gain:>6.1f} {result['avg_angle_error']:>15.2f} {result['std_angle_error']:>15.2f} "
              f"{result['avg_omega']:>12.3f} {result['max_omega_used']:>12.3f}")

    print()
    print(f"Target omega (v/r = 0.2/0.15): 1.333 rad/s")
    print()

    # Find best gain
    best_gain = min(test_gains, key=lambda g: results[g]['avg_angle_error'])
    print(f"Best gain (lowest avg angle error): {best_gain}")
    print(f"  Avg |angle error|: {results[best_gain]['avg_angle_error']:.2f}°")
    print(f"  Avg omega: {results[best_gain]['avg_omega']:.3f} rad/s")
    print("="*80)


if __name__ == '__main__':
    main()
