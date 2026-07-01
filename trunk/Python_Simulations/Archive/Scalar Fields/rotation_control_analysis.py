#!/usr/bin/env python3
"""
Analyze the impact of formation rotation control on Newton's method convergence
Shows why rotation control matters for Hessian estimation quality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

class RotationControlAnalyzer:
    def __init__(self, robot_distance=0.25, step_size=0.01, max_iterations=1000):
        self.robot_distance = robot_distance
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.rotation_gain = 0.5  # Increased for more aggressive rotation

    def scalar_field(self, x, y):
        """Bimodal Gaussian scalar field"""
        gaussian1 = -((x + 2)**2 + y**2) / 2
        gaussian2 = -((x - 2)**2 + y**2) / 2
        max_exp = np.maximum(gaussian1, gaussian2)
        return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))

    def get_robot_positions(self, center, orientation):
        """Get 4 robot positions in square formation with given orientation"""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + orientation
        positions = []
        for angle in angles:
            x = center[0] + self.robot_distance * np.cos(angle)
            y = center[1] + self.robot_distance * np.sin(angle)
            positions.append([x, y])
        return np.array(positions)

    def estimate_hessian_quality(self, center, orientation):
        """
        Estimate how well the formation samples the field for Hessian estimation
        Better sampling = larger gradient variation between robots
        """
        robots = self.get_robot_positions(center, orientation)

        # Get field values at each robot
        values = [self.scalar_field(r[0], r[1]) for r in robots]

        # Measure variation (higher is better for Hessian estimation)
        variation = np.std(values)

        # Also measure gradient differences between opposite robots
        grad_diffs = []
        for i in range(2):
            # Opposite pairs: (0,2) and (1,3)
            j = i + 2
            dx = robots[j][0] - robots[i][0]
            dy = robots[j][1] - robots[i][1]
            dz = values[j] - values[i]
            if np.sqrt(dx**2 + dy**2) > 0:
                grad_mag = abs(dz) / np.sqrt(dx**2 + dy**2)
                grad_diffs.append(grad_mag)

        quality = variation * np.mean(grad_diffs) if len(grad_diffs) > 0 else variation
        return quality

    def compute_newton_step(self, center):
        """Simplified Newton step computation"""
        h = 0.1

        # Estimate gradient
        fx = (self.scalar_field(center[0] + h, center[1]) -
              self.scalar_field(center[0] - h, center[1])) / (2*h)
        fy = (self.scalar_field(center[0], center[1] + h) -
              self.scalar_field(center[0], center[1] - h)) / (2*h)
        gradient = np.array([fx, fy])

        # Estimate Hessian
        fxx = (self.scalar_field(center[0] + h, center[1]) -
               2*self.scalar_field(center[0], center[1]) +
               self.scalar_field(center[0] - h, center[1])) / (h**2)
        fyy = (self.scalar_field(center[0], center[1] + h) -
               2*self.scalar_field(center[0], center[1]) +
               self.scalar_field(center[0], center[1] - h)) / (h**2)
        fxy = (self.scalar_field(center[0] + h, center[1] + h) -
               self.scalar_field(center[0] + h, center[1] - h) -
               self.scalar_field(center[0] - h, center[1] + h) +
               self.scalar_field(center[0] - h, center[1] - h)) / (4*h**2)

        hessian = np.array([[fxx, fxy], [fxy, fyy]])

        # Newton step
        try:
            if abs(np.linalg.det(hessian)) > 1e-10:
                step = -np.linalg.solve(hessian, gradient)
            else:
                step = -np.linalg.pinv(hessian) @ gradient
        except:
            step = -gradient / (np.linalg.norm(gradient) + 1e-10)

        # Limit step size
        if np.linalg.norm(step) > 1.0:
            step = step / np.linalg.norm(step)

        return step, gradient, hessian

    def simulate_with_rotation(self, start_pos, initial_orientation=None):
        """Simulate with rotation control"""
        if initial_orientation is None:
            initial_orientation = np.random.uniform(0, 2*np.pi)

        trajectory = [start_pos.copy()]
        orientations = [initial_orientation]
        qualities = []
        errors = []

        pos = start_pos.copy()
        orientation = initial_orientation

        for i in range(self.max_iterations):
            # Get Newton step
            step, gradient, hessian = self.compute_newton_step(pos)

            # Newton step direction
            step_angle = np.arctan2(step[1], step[0])

            # Compute rotation error to align with Newton step
            # For square formation, we can align any of 4 orientations
            angle_errors = []
            for k in range(4):
                target = step_angle - k * np.pi/2
                error = target - orientation
                # Wrap to [-pi, pi]
                error = np.arctan2(np.sin(error), np.cos(error))
                angle_errors.append(error)

            # Choose minimum error
            min_idx = np.argmin(np.abs(angle_errors))
            rotation_error = angle_errors[min_idx]

            # Apply rotation control
            orientation += self.rotation_gain * rotation_error * self.step_size

            # Record Hessian estimation quality
            quality = self.estimate_hessian_quality(pos, orientation)
            qualities.append(quality)
            errors.append(abs(rotation_error))

            # Move
            pos = pos + self.step_size * step
            trajectory.append(pos.copy())
            orientations.append(orientation)

            # Check convergence
            if np.linalg.norm(pos) < 0.1:
                break

        return np.array(trajectory), orientations, qualities, errors

    def simulate_without_rotation(self, start_pos, fixed_orientation=None):
        """Simulate without rotation control"""
        if fixed_orientation is None:
            fixed_orientation = np.random.uniform(0, 2*np.pi)

        trajectory = [start_pos.copy()]
        qualities = []

        pos = start_pos.copy()

        for i in range(self.max_iterations):
            # Get Newton step
            step, gradient, hessian = self.compute_newton_step(pos)

            # Record Hessian estimation quality
            quality = self.estimate_hessian_quality(pos, fixed_orientation)
            qualities.append(quality)

            # Move
            pos = pos + self.step_size * step
            trajectory.append(pos.copy())

            # Check convergence
            if np.linalg.norm(pos) < 0.1:
                break

        return np.array(trajectory), qualities

    def compare_methods(self, start_positions):
        """Compare rotation vs no rotation for multiple starting positions"""
        results = {
            'with_rotation': {'success': 0, 'path_lengths': [], 'iterations': [], 'avg_quality': []},
            'without_rotation': {'success': 0, 'path_lengths': [], 'iterations': [], 'avg_quality': []}
        }

        # Test each starting position with worst-case initial orientation
        for start_pos in start_positions:
            # Find worst orientation (perpendicular to Newton direction)
            step, _, _ = self.compute_newton_step(start_pos)
            step_angle = np.arctan2(step[1], step[0])
            worst_orientation = step_angle + np.pi/4  # 45 degrees off

            # With rotation control
            traj_rot, orientations, qualities_rot, errors = self.simulate_with_rotation(
                start_pos, initial_orientation=worst_orientation)

            if np.linalg.norm(traj_rot[-1]) < 0.1:
                results['with_rotation']['success'] += 1

            # Path length
            path_length = np.sum(np.linalg.norm(np.diff(traj_rot, axis=0), axis=1))
            results['with_rotation']['path_lengths'].append(path_length)
            results['with_rotation']['iterations'].append(len(traj_rot))
            results['with_rotation']['avg_quality'].append(np.mean(qualities_rot) if qualities_rot else 0)

            # Without rotation control (stuck with bad orientation)
            traj_no_rot, qualities_no_rot = self.simulate_without_rotation(
                start_pos, fixed_orientation=worst_orientation)

            if np.linalg.norm(traj_no_rot[-1]) < 0.1:
                results['without_rotation']['success'] += 1

            path_length = np.sum(np.linalg.norm(np.diff(traj_no_rot, axis=0), axis=1))
            results['without_rotation']['path_lengths'].append(path_length)
            results['without_rotation']['iterations'].append(len(traj_no_rot))
            results['without_rotation']['avg_quality'].append(np.mean(qualities_no_rot) if qualities_no_rot else 0)

        return results

    def visualize_rotation_impact(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))

        # Test from challenging starting position
        start_pos = np.array([1.5, -1.5])

        # Find Newton direction for worst-case orientation
        step, _, _ = self.compute_newton_step(start_pos)
        step_angle = np.arctan2(step[1], step[0])
        worst_orientation = step_angle + np.pi/4  # 45 degrees misaligned
        best_orientation = step_angle  # Aligned with Newton step

        # Generate field for background
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.scalar_field(X, Y)

        # Plot 1: Trajectory comparison
        ax1 = plt.subplot(2, 3, 1)
        ax1.contour(X, Y, Z, levels=15, cmap='gray', alpha=0.3, linewidths=1)

        # With rotation (starting misaligned)
        traj_rot, orientations, qualities_rot, errors = self.simulate_with_rotation(
            start_pos, initial_orientation=worst_orientation)
        ax1.plot(traj_rot[:, 0], traj_rot[:, 1], 'r-', linewidth=2, label='With rotation')

        # Without rotation (stuck misaligned)
        traj_no_rot, _ = self.simulate_without_rotation(
            start_pos, fixed_orientation=worst_orientation)
        ax1.plot(traj_no_rot[:, 0], traj_no_rot[:, 1], 'b--', linewidth=2, label='No rotation (misaligned)')

        # Without rotation (lucky - well aligned)
        traj_lucky, _ = self.simulate_without_rotation(
            start_pos, fixed_orientation=best_orientation)
        ax1.plot(traj_lucky[:, 0], traj_lucky[:, 1], 'g:', linewidth=2, label='No rotation (aligned)')

        ax1.plot(start_pos[0], start_pos[1], 'ko', markersize=8)
        ax1.plot(0, 0, 'rx', markersize=12, linewidth=3)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Trajectory Comparison', fontweight='bold')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Formation orientation evolution
        ax2 = plt.subplot(2, 3, 2)

        # Show formations at key points
        key_points = [0, len(traj_rot)//4, len(traj_rot)//2, 3*len(traj_rot)//4]
        colors = ['blue', 'green', 'orange', 'red']

        for idx, color in zip(key_points, colors):
            if idx < len(traj_rot):
                center = traj_rot[idx]
                orientation = orientations[idx] if idx < len(orientations) else orientations[-1]
                robots = self.get_robot_positions(center, orientation)

                # Draw square formation
                for i in range(4):
                    j = (i + 1) % 4
                    ax2.plot([robots[i, 0], robots[j, 0]],
                           [robots[i, 1], robots[j, 1]],
                           color=color, linewidth=1.5, alpha=0.7)

                # Draw orientation arrow
                arrow_len = 0.15
                ax2.arrow(center[0], center[1],
                        arrow_len * np.cos(orientation),
                        arrow_len * np.sin(orientation),
                        head_width=0.05, head_length=0.05,
                        fc=color, ec=color)

        ax2.plot(traj_rot[:, 0], traj_rot[:, 1], 'k--', alpha=0.3, linewidth=1)
        ax2.plot(0, 0, 'rx', markersize=12, linewidth=3)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Formation Rotation During Navigation', fontweight='bold')
        ax2.set_aspect('equal')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylim(-2.5, 2.5)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Hessian estimation quality
        ax3 = plt.subplot(2, 3, 3)

        # Compare qualities
        iterations = range(min(len(qualities_rot), len(traj_no_rot)-1))
        ax3.plot(iterations, qualities_rot[:len(iterations)], 'r-', linewidth=2,
                label='With rotation')

        # Quality without rotation
        _, qualities_no_rot = self.simulate_without_rotation(
            start_pos, fixed_orientation=worst_orientation)
        ax3.plot(range(len(qualities_no_rot)), qualities_no_rot, 'b--', linewidth=2,
                label='No rotation')

        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Hessian Estimation Quality')
        ax3.set_title('Estimation Quality Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence rate
        ax4 = plt.subplot(2, 3, 4)

        distances_rot = [np.linalg.norm(p) for p in traj_rot]
        distances_no_rot = [np.linalg.norm(p) for p in traj_no_rot]
        distances_lucky = [np.linalg.norm(p) for p in traj_lucky]

        ax4.semilogy(distances_rot, 'r-', linewidth=2, label='With rotation')
        ax4.semilogy(distances_no_rot, 'b--', linewidth=2, label='No rotation (misaligned)')
        ax4.semilogy(distances_lucky, 'g:', linewidth=2, label='No rotation (aligned)')
        ax4.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Convergence threshold')

        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Distance to Saddle (m)')
        ax4.set_title('Convergence Rate Comparison', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Statistical comparison
        ax5 = plt.subplot(2, 3, 5)

        # Test multiple starting points
        n_tests = 20
        angles = np.linspace(0, 2*np.pi, n_tests, endpoint=False)
        start_positions = []
        for angle in angles:
            r = 1.5  # Fixed radius
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            start_positions.append(np.array([x, y]))

        print("Running statistical comparison...")
        results = self.compare_methods(start_positions)

        # Bar plot comparison
        metrics = ['Success Rate', 'Avg Path Length', 'Avg Iterations', 'Avg Quality']
        with_rot = [
            results['with_rotation']['success'] / n_tests * 100,
            np.mean(results['with_rotation']['path_lengths']),
            np.mean(results['with_rotation']['iterations']),
            np.mean(results['with_rotation']['avg_quality']) * 100
        ]
        without_rot = [
            results['without_rotation']['success'] / n_tests * 100,
            np.mean(results['without_rotation']['path_lengths']),
            np.mean(results['without_rotation']['iterations']),
            np.mean(results['without_rotation']['avg_quality']) * 100
        ]

        x_pos = np.arange(len(metrics))
        width = 0.35

        bars1 = ax5.bar(x_pos - width/2, with_rot, width, label='With rotation', color='red', alpha=0.7)
        bars2 = ax5.bar(x_pos + width/2, without_rot, width, label='No rotation', color='blue', alpha=0.7)

        ax5.set_xlabel('Metric')
        ax5.set_ylabel('Value')
        ax5.set_title('Statistical Performance Comparison', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(metrics, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars1, with_rot):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, without_rot):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        # Plot 6: Rotation alignment over time
        ax6 = plt.subplot(2, 3, 6)

        # Show alignment error over time
        ax6.plot(errors, 'r-', linewidth=2, label='Alignment error')
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax6.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='red')

        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Rotation Error (radians)')
        ax6.set_title('Formation Alignment Error', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('Impact of Formation Rotation Control on Newton\'s Method',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('rotation_control_impact.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Print statistics
        print("\n" + "="*50)
        print("ROTATION CONTROL IMPACT ANALYSIS")
        print("="*50)
        print(f"\nTested with {n_tests} starting positions at radius 1.5m")
        print(f"Initial orientation: Worst-case (45° misaligned)")
        print(f"\nWITH ROTATION CONTROL:")
        print(f"  Success rate: {results['with_rotation']['success']}/{n_tests} ({with_rot[0]:.1f}%)")
        print(f"  Avg path length: {with_rot[1]:.3f} m")
        print(f"  Avg iterations: {with_rot[2]:.0f}")
        print(f"  Avg quality: {with_rot[3]:.2f}")
        print(f"\nWITHOUT ROTATION CONTROL:")
        print(f"  Success rate: {results['without_rotation']['success']}/{n_tests} ({without_rot[0]:.1f}%)")
        print(f"  Avg path length: {without_rot[1]:.3f} m")
        print(f"  Avg iterations: {without_rot[2]:.0f}")
        print(f"  Avg quality: {without_rot[3]:.2f}")
        print(f"\nIMPROVEMENT WITH ROTATION:")
        print(f"  Success rate: +{with_rot[0] - without_rot[0]:.1f}%")
        print(f"  Path efficiency: {(without_rot[1] - with_rot[1])/without_rot[1]*100:.1f}% shorter")
        print(f"  Convergence speed: {(without_rot[2] - with_rot[2])/without_rot[2]*100:.1f}% faster")
        print(f"  Estimation quality: {(with_rot[3] - without_rot[3])/without_rot[3]*100:.1f}% better")

# Main execution
if __name__ == "__main__":
    analyzer = RotationControlAnalyzer()
    analyzer.visualize_rotation_impact()