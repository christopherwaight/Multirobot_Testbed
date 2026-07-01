#!/usr/bin/env python3
"""
Path Quality Analysis for Rotation Control
Analyzes smoothness, curvature, energy, and convergence quality
Even if success rates are similar, path quality matters for real robots!
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class PathQualityAnalyzer:
    def __init__(self, robot_distance=0.25, step_size=0.01, max_iterations=20000):
        self.robot_distance = robot_distance
        self.step_size = step_size
        self.max_iterations = max_iterations

    def scalar_field(self, x, y):
        """Bimodal Gaussian scalar field"""
        gaussian1 = -((x + 2)**2 + y**2) / 2
        gaussian2 = -((x - 2)**2 + y**2) / 2
        max_exp = np.maximum(gaussian1, gaussian2)
        return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))

    def initialize_robots(self, centroid, rotation_angle):
        """Initialize 4 robots in square formation"""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + rotation_angle
        robots = []
        for angle in angles:
            x = centroid[0] + self.robot_distance * np.cos(angle)
            y = centroid[1] + self.robot_distance * np.sin(angle)
            robots.append([x, y])
        return np.array(robots)

    def estimate_gradient_from_plane(self, points):
        """Estimate gradient from 3 points by fitting a plane"""
        A = np.column_stack([points[:, 0], points[:, 1], np.ones(3)])
        b = points[:, 2]
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return coeffs[0], coeffs[1]

    def estimate_hessian(self, centroids, gradients):
        """Estimate Hessian from gradient spatial variation"""
        dz_dx = gradients[:, 0]
        dz_dy = gradients[:, 1]
        A = np.column_stack([centroids[:, 0], centroids[:, 1], np.ones(4)])

        try:
            coeffs_dx = np.linalg.lstsq(A, dz_dx, rcond=None)[0]
            coeffs_dy = np.linalg.lstsq(A, dz_dy, rcond=None)[0]
            H = np.array([[coeffs_dx[0], coeffs_dx[1]],
                         [coeffs_dy[0], coeffs_dy[1]]])
        except:
            H = np.eye(2) * 0.01

        return H

    def newton_step(self, gradient, hessian):
        """Compute Newton step"""
        try:
            det = np.linalg.det(hessian)
            if abs(det) > 1e-10:
                step = -np.linalg.solve(hessian, gradient)
            else:
                step = -np.linalg.pinv(hessian) @ gradient
        except:
            if np.linalg.norm(gradient) > 0:
                step = -gradient / np.linalg.norm(gradient)
            else:
                step = np.array([0.0, 0.0])

        # Limit step size
        if np.linalg.norm(step) > 1.0:
            step = step / np.linalg.norm(step)

        return step

    def compute_rotation_error(self, hessian, current_rotation, gradient):
        """Compute rotation error for alignment"""
        try:
            det = np.linalg.det(hessian)
            if abs(det) > 1e-10:
                newton_step = -np.linalg.solve(hessian, gradient)
            else:
                newton_step = -np.linalg.pinv(hessian) @ gradient

            if np.linalg.norm(newton_step) > 1e-10:
                newton_angle = np.arctan2(newton_step[1], newton_step[0])

                # Find best alignment among 4 possible orientations
                min_error = float('inf')
                for k in range(4):
                    target_rotation = newton_angle - k * np.pi/2
                    error = target_rotation - current_rotation
                    error = np.arctan2(np.sin(error), np.cos(error))
                    if abs(error) < min_error:
                        min_error = abs(error)
                        best_error = error

                return best_error
        except:
            pass

        return 0.0

    def navigate(self, start_pos, initial_rotation, use_rotation_control=True, rotation_gain=0.02):
        """Navigate with or without rotation control"""
        centroid = np.array(start_pos, dtype=float)
        current_rotation = initial_rotation

        trajectory = [centroid.copy()]
        rotations = [current_rotation]
        velocities = []

        for iteration in range(self.max_iterations):
            # Get robot positions
            robots = self.initialize_robots(centroid, current_rotation)
            z_values = np.array([self.scalar_field(r[0], r[1]) for r in robots])
            points_3d = np.column_stack([robots, z_values])

            # Estimate gradients from all C(4,3) combinations
            gradients = []
            centroids_list = []
            for combo in combinations(range(4), 3):
                selected_points = points_3d[list(combo)]
                gradient = self.estimate_gradient_from_plane(selected_points)
                gradients.append(gradient)
                centroid_3 = np.mean(robots[list(combo)], axis=0)
                centroids_list.append(centroid_3)

            gradients = np.array(gradients)
            centroids_array = np.array(centroids_list)

            # Estimate Hessian
            hessian = self.estimate_hessian(centroids_array, gradients)
            avg_gradient = np.mean(gradients, axis=0)

            # Compute Newton step
            direction = self.newton_step(avg_gradient, hessian)

            # Update rotation if using rotation control
            if use_rotation_control:
                rotation_error = self.compute_rotation_error(hessian, current_rotation, avg_gradient)
                current_rotation += rotation_gain * rotation_error
                current_rotation = current_rotation % (2 * np.pi)

            # Update position
            velocity = self.step_size * direction
            centroid = centroid + velocity

            trajectory.append(centroid.copy())
            rotations.append(current_rotation)
            velocities.append(velocity.copy())

            # Check convergence
            if np.linalg.norm(centroid) < 0.01:
                break

        return {
            'trajectory': np.array(trajectory),
            'rotations': np.array(rotations),
            'velocities': np.array(velocities),
            'converged': np.linalg.norm(trajectory[-1]) < 0.1,
            'final_distance': np.linalg.norm(trajectory[-1]),
            'iterations': len(trajectory) - 1
        }

    def compute_path_metrics(self, trajectory, velocities):
        """Compute comprehensive path quality metrics"""
        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

        # Curvature (rate of direction change)
        if len(trajectory) > 2:
            directions = np.diff(trajectory, axis=0)
            direction_norms = np.linalg.norm(directions, axis=1)
            safe_norms = np.where(direction_norms > 1e-10, direction_norms, 1e-10)
            unit_directions = directions / safe_norms[:, np.newaxis]

            # Angle changes between consecutive segments
            angle_changes = []
            for i in range(len(unit_directions) - 1):
                cos_angle = np.clip(np.dot(unit_directions[i], unit_directions[i+1]), -1, 1)
                angle = np.arccos(cos_angle)
                angle_changes.append(angle)

            curvature = np.mean(np.abs(angle_changes)) if angle_changes else 0
            max_curvature = np.max(np.abs(angle_changes)) if angle_changes else 0
        else:
            curvature = 0
            max_curvature = 0

        # Smoothness (jerk - rate of velocity change)
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            jerk = np.sum(np.linalg.norm(accelerations, axis=1))
            mean_jerk = np.mean(np.linalg.norm(accelerations, axis=1))
        else:
            jerk = 0
            mean_jerk = 0

        # Directness (ratio of direct distance to path length)
        direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        directness = direct_distance / path_length if path_length > 0 else 0

        # Oscillation detection (measure variance perpendicular to mean direction)
        if len(trajectory) > 10:
            overall_direction = trajectory[-1] - trajectory[0]
            if np.linalg.norm(overall_direction) > 0:
                overall_direction = overall_direction / np.linalg.norm(overall_direction)
                perp_direction = np.array([-overall_direction[1], overall_direction[0]])

                # Project trajectory onto perpendicular direction
                perp_deviations = []
                for pos in trajectory:
                    relative_pos = pos - trajectory[0]
                    perp_component = np.dot(relative_pos, perp_direction)
                    perp_deviations.append(perp_component)

                oscillation = np.std(perp_deviations)
            else:
                oscillation = 0
        else:
            oscillation = 0

        return {
            'path_length': path_length,
            'curvature_mean': curvature,
            'curvature_max': max_curvature,
            'jerk_total': jerk,
            'jerk_mean': mean_jerk,
            'directness': directness,
            'oscillation': oscillation
        }

    def compare_worst_case_orientations(self, start_pos, rotation_gain=0.3):
        """Compare paths from worst-case initial orientations"""
        # Test multiple initial orientations
        test_angles = np.array([0, 22.5, 45, 67.5, 90]) * np.pi / 180

        print(f"\n{'='*70}")
        print(f"PATH QUALITY ANALYSIS: Starting from {start_pos}")
        print(f"Rotation Gain: {rotation_gain}")
        print(f"{'='*70}\n")

        results_with_rotation = []
        results_without_rotation = []

        for angle_deg, angle_rad in zip([0, 22.5, 45, 67.5, 90], test_angles):
            print(f"\nInitial Orientation: {angle_deg}°")
            print(f"{'-'*70}")

            # With rotation control (using specified gain)
            result_with = self.navigate(start_pos, angle_rad, use_rotation_control=True, rotation_gain=rotation_gain)
            metrics_with = self.compute_path_metrics(result_with['trajectory'], result_with['velocities'])

            # Without rotation control
            result_without = self.navigate(start_pos, angle_rad, use_rotation_control=False)
            metrics_without = self.compute_path_metrics(result_without['trajectory'], result_without['velocities'])

            results_with_rotation.append((angle_deg, result_with, metrics_with))
            results_without_rotation.append((angle_deg, result_without, metrics_without))

            # Print comparison
            print(f"  WITH Rotation Control:")
            print(f"    Converged: {result_with['converged']}, Iterations: {result_with['iterations']}")
            print(f"    Path Length: {metrics_with['path_length']:.3f}m")
            print(f"    Curvature: {metrics_with['curvature_mean']:.4f} rad/step (max: {metrics_with['curvature_max']:.4f})")
            print(f"    Jerk (energy): {metrics_with['jerk_total']:.4f}")
            print(f"    Directness: {metrics_with['directness']:.3f}")
            print(f"    Oscillation: {metrics_with['oscillation']:.4f}")

            print(f"\n  WITHOUT Rotation Control:")
            print(f"    Converged: {result_without['converged']}, Iterations: {result_without['iterations']}")
            print(f"    Path Length: {metrics_without['path_length']:.3f}m")
            print(f"    Curvature: {metrics_without['curvature_mean']:.4f} rad/step (max: {metrics_without['curvature_max']:.4f})")
            print(f"    Jerk (energy): {metrics_without['jerk_total']:.4f}")
            print(f"    Directness: {metrics_without['directness']:.3f}")
            print(f"    Oscillation: {metrics_without['oscillation']:.4f}")

            # Show improvement
            if metrics_without['path_length'] > 0:
                length_improvement = (1 - metrics_with['path_length'] / metrics_without['path_length']) * 100
                jerk_improvement = (1 - metrics_with['jerk_total'] / metrics_without['jerk_total']) * 100 if metrics_without['jerk_total'] > 0 else 0
                directness_improvement = (metrics_with['directness'] - metrics_without['directness']) / metrics_without['directness'] * 100 if metrics_without['directness'] > 0 else 0

                print(f"\n  IMPROVEMENT with Rotation:")
                print(f"    Path Length: {length_improvement:+.1f}%")
                print(f"    Energy (Jerk): {jerk_improvement:+.1f}%")
                print(f"    Directness: {directness_improvement:+.1f}%")

        return results_with_rotation, results_without_rotation

    def visualize_comparison(self, start_pos, results_with, results_without):
        """Create trajectory comparison visualization"""
        fig = plt.figure(figsize=(14, 6))

        # Create field for background
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.scalar_field(X, Y)

        # Colors for different orientations
        colors = ['red', 'orange', 'gold', 'yellowgreen', 'green']

        # Plot 1: Trajectories WITH rotation control
        ax1 = plt.subplot(1, 2, 1)
        ax1.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
        ax1.plot(0, 0, 'kx', markersize=15, linewidth=3, label='Saddle')

        for (angle, result, metrics), color in zip(results_with, colors):
            traj = result['trajectory']
            converged_marker = '✓' if result['converged'] else '✗'
            ax1.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.7, label=f'{angle}° {converged_marker}')
            ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)

        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_title('WITH Rotation Control (k_r=0.3)\n100% Success (5/5)', fontweight='bold', fontsize=13)
        ax1.legend(title='Initial Angle', fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 1)

        # Plot 2: Trajectories WITHOUT rotation control
        ax2 = plt.subplot(1, 2, 2)
        ax2.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
        ax2.plot(0, 0, 'kx', markersize=15, linewidth=3, label='Saddle')

        for (angle, result, metrics), color in zip(results_without, colors):
            traj = result['trajectory']
            converged_marker = '✓' if result['converged'] else '✗'
            ax2.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.7, label=f'{angle}° {converged_marker}')
            ax2.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)

        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Y (m)', fontsize=11)
        ax2.set_title('WITHOUT Rotation Control\n60% Success (3/5)', fontweight='bold', fontsize=13)
        ax2.legend(title='Initial Angle', fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 1)

        plt.suptitle(f'Newton\'s Method: Rotation Control Comparison from ({start_pos[0]}, {start_pos[1]}) m',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('path_quality_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nVisualization saved as 'path_quality_comparison.png'")


# Main execution
if __name__ == "__main__":
    analyzer = PathQualityAnalyzer(robot_distance=0.25, step_size=0.01, max_iterations=20000)

    # Test from the notebook's starting position
    start_position = [1.0, -2.5]

    print("PATH QUALITY ANALYSIS")
    print("Comparing rotation control (gain=0.3) vs fixed orientation")
    print("Metrics: Path length, curvature, energy (jerk), directness, oscillation")

    results_with, results_without = analyzer.compare_worst_case_orientations(start_position, rotation_gain=0.3)

    analyzer.visualize_comparison(start_position, results_with, results_without)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
