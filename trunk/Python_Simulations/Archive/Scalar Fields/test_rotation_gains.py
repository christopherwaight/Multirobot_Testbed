#!/usr/bin/env python3
"""
Test different rotation gains to find optimal value
"""

import numpy as np
from itertools import combinations

class RotationGainTester:
    def __init__(self, robot_distance=0.25, step_size=0.01, max_iterations=5000):
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
        """Estimate gradient from 3 points"""
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

    def test_single_run(self, start_pos, initial_rotation, rotation_gain):
        """Single navigation run with specific rotation gain"""
        centroid = np.array(start_pos, dtype=float)
        current_rotation = initial_rotation

        for iteration in range(self.max_iterations):
            # Get robot positions
            robots = self.initialize_robots(centroid, current_rotation)
            z_values = np.array([self.scalar_field(r[0], r[1]) for r in robots])
            points_3d = np.column_stack([robots, z_values])

            # Estimate gradients
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

            # Update rotation with given gain
            if rotation_gain > 0:
                rotation_error = self.compute_rotation_error(hessian, current_rotation, avg_gradient)
                current_rotation += rotation_gain * rotation_error
                current_rotation = current_rotation % (2 * np.pi)

            # Update position
            centroid = centroid + self.step_size * direction

            # Check convergence
            if np.linalg.norm(centroid) < 0.01:
                return True, iteration + 1, np.linalg.norm(centroid)

        # Did not converge
        return False, self.max_iterations, np.linalg.norm(centroid)

    def test_rotation_gains(self, start_pos, initial_angle_deg):
        """Test multiple rotation gains"""
        initial_rotation = initial_angle_deg * np.pi / 180

        # Test different gains including 0 (no rotation)
        gains_to_test = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

        print(f"\n{'='*80}")
        print(f"Testing Rotation Gains: Start={start_pos}, Initial Angle={initial_angle_deg}°")
        print(f"{'='*80}\n")
        print(f"{'Gain':<10} {'Converged':<12} {'Iterations':<12} {'Final Dist':<15} {'Status'}")
        print(f"{'-'*80}")

        results = []
        for gain in gains_to_test:
            converged, iterations, final_dist = self.test_single_run(start_pos, initial_rotation, gain)
            status = "✓ SUCCESS" if converged else "✗ FAILED"

            print(f"{gain:<10.3f} {str(converged):<12} {iterations:<12} {final_dist:<15.4f} {status}")
            results.append((gain, converged, iterations, final_dist))

        return results

    def comprehensive_test(self, start_pos=[1.0, -2.5]):
        """Test multiple initial angles with different gains"""
        test_angles = [0, 22.5, 45, 67.5, 90]

        print("\n" + "="*80)
        print("COMPREHENSIVE ROTATION GAIN ANALYSIS")
        print("="*80)

        all_results = {}
        for angle in test_angles:
            results = self.test_rotation_gains(start_pos, angle)
            all_results[angle] = results

        # Find optimal gain
        print("\n" + "="*80)
        print("SUMMARY: Success Rate by Rotation Gain")
        print("="*80)

        gains = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        print(f"\n{'Gain':<10} ", end='')
        for angle in test_angles:
            print(f"{angle:>6}° ", end='')
        print("  Avg Success")
        print("-"*80)

        for gain in gains:
            print(f"{gain:<10.3f} ", end='')
            successes = []
            for angle in test_angles:
                result = all_results[angle]
                gain_result = [r for r in result if r[0] == gain][0]
                converged = gain_result[1]
                successes.append(converged)
                symbol = "✓" if converged else "✗"
                print(f"{symbol:>7} ", end='')

            avg_success = sum(successes) / len(successes) * 100
            print(f"  {avg_success:>5.1f}%")

        print("\n" + "="*80)

if __name__ == "__main__":
    tester = RotationGainTester(robot_distance=0.25, step_size=0.01, max_iterations=5000)
    tester.comprehensive_test(start_pos=[1.0, -2.5])
