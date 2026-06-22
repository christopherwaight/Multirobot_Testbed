#!/usr/bin/env python3
"""
Basin of Attraction Analysis for Newton's Method at Saddle Points
Creates maps showing where convergence succeeds/fails
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

class NewtonSaddleAnalyzer:
    def __init__(self, step_size=0.01, max_iterations=20000):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.robot_distance = 0.25  # Square formation side length

    def scalar_field(self, x, y):
        """Bimodal Gaussian scalar field"""
        gaussian1 = -((x + 2)**2 + y**2) / 2
        gaussian2 = -((x - 2)**2 + y**2) / 2
        max_exp = np.maximum(gaussian1, gaussian2)
        return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))

    def estimate_gradient_and_hessian(self, xc, yc):
        """Simplified estimation of gradient and Hessian at cluster center"""
        # This is a simplified version - in reality would use 4 robots
        h = 0.1  # Small step for numerical derivatives

        # Estimate gradient
        fx = (self.scalar_field(xc + h, yc) - self.scalar_field(xc - h, yc)) / (2*h)
        fy = (self.scalar_field(xc, yc + h) - self.scalar_field(xc, yc - h)) / (2*h)
        gradient = np.array([fx, fy])

        # Estimate Hessian
        fxx = (self.scalar_field(xc + h, yc) - 2*self.scalar_field(xc, yc) +
               self.scalar_field(xc - h, yc)) / (h**2)
        fyy = (self.scalar_field(xc, yc + h) - 2*self.scalar_field(xc, yc) +
               self.scalar_field(xc, yc - h)) / (h**2)
        fxy = (self.scalar_field(xc + h, yc + h) - self.scalar_field(xc + h, yc - h) -
               self.scalar_field(xc - h, yc + h) + self.scalar_field(xc - h, yc - h)) / (4*h**2)

        hessian = np.array([[fxx, fxy], [fxy, fyy]])

        return gradient, hessian

    def newton_step(self, pos):
        """Compute Newton step at given position"""
        gradient, hessian = self.estimate_gradient_and_hessian(pos[0], pos[1])

        # Handle singular Hessian
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

    def test_convergence(self, start_pos, threshold=0.1):
        """Test if Newton's method converges from given starting position"""
        pos = np.array(start_pos, dtype=float)

        for i in range(self.max_iterations):
            step = self.newton_step(pos)
            pos = pos + self.step_size * step

            # Check if converged to saddle at origin
            if np.linalg.norm(pos) < threshold:
                return True, i

            # Check if diverged too far
            if np.linalg.norm(pos) > 5.0:
                return False, i

        # Check final position
        return np.linalg.norm(pos) < threshold, self.max_iterations

    def test_radius_success_rate(self, radius, n_samples=100):
        """Test success rate for given radius"""
        successes = 0
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            converged, _ = self.test_convergence([x, y])
            if converged:
                successes += 1

        return successes / n_samples

    def find_optimal_radius(self):
        """Find radius that gives high success rate"""
        radii = np.linspace(0.1, 2.0, 20)
        success_rates = []

        for r in radii:
            rate = self.test_radius_success_rate(r, n_samples=50)
            success_rates.append(rate)
            print(f"Radius {r:.2f}m: {rate*100:.1f}% success")

        # Find radii with 100% success
        perfect_radii = radii[np.array(success_rates) == 1.0]
        if len(perfect_radii) > 0:
            optimal_radius = perfect_radii[-1]  # Largest radius with 100% success
            print(f"\nOptimal radius: {optimal_radius:.2f}m (100% success)")
        else:
            # Find radius with highest success
            best_idx = np.argmax(success_rates)
            optimal_radius = radii[best_idx]
            print(f"\nBest radius: {optimal_radius:.2f}m ({success_rates[best_idx]*100:.1f}% success)")

        return optimal_radius, radii, success_rates

    def create_basin_map(self, grid_size=50):
        """Create 2D map of basin of attraction"""
        x_range = np.linspace(-3, 3, grid_size)
        y_range = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x_range, y_range)

        # Test convergence for each starting position
        convergence_map = np.zeros_like(X)
        iteration_map = np.zeros_like(X)

        print("Creating basin of attraction map...")
        for i in range(grid_size):
            for j in range(grid_size):
                if (i * grid_size + j) % 100 == 0:
                    print(f"Progress: {(i * grid_size + j) / (grid_size**2) * 100:.1f}%")

                start_pos = [X[i, j], Y[i, j]]
                converged, iterations = self.test_convergence(start_pos)
                convergence_map[i, j] = 1 if converged else 0
                iteration_map[i, j] = iterations if converged else -1

        return X, Y, convergence_map, iteration_map

    def plot_results(self, optimal_radius, radii, success_rates, X, Y, convergence_map, iteration_map):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Success rate vs radius
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(radii, np.array(success_rates)*100, 'b-', linewidth=2)
        ax1.axvline(optimal_radius, color='r', linestyle='--', label=f'Optimal: {optimal_radius:.2f}m')
        ax1.axhline(100, color='g', linestyle='--', alpha=0.5, label='100% success')
        ax1.set_xlabel('Starting Radius (m)', fontsize=11)
        ax1.set_ylabel('Success Rate (%)', fontsize=11)
        ax1.set_title('Convergence Success vs Starting Radius', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([-5, 105])

        # Plot 2: Basin of attraction map
        ax2 = plt.subplot(2, 3, 2)
        im = ax2.contourf(X, Y, convergence_map, levels=[0, 0.5, 1],
                         colors=['#ffcccc', '#ccffcc'], alpha=0.8)
        ax2.contour(X, Y, convergence_map, levels=[0.5], colors='black', linewidths=2)

        # Add circles for different radii
        for r in [0.5, 1.0, 1.5, 2.0]:
            circle = Circle((0, 0), r, fill=False, edgecolor='gray',
                          linestyle='--', alpha=0.5, linewidth=1)
            ax2.add_patch(circle)

        # Add optimal radius circle
        optimal_circle = Circle((0, 0), optimal_radius, fill=False, edgecolor='red',
                              linestyle='-', linewidth=2, label=f'Optimal radius: {optimal_radius:.2f}m')
        ax2.add_patch(optimal_circle)

        ax2.plot(0, 0, 'rx', markersize=12, linewidth=3, label='Saddle point')
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Y (m)', fontsize=11)
        ax2.set_title('Basin of Attraction Map', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.grid(True, alpha=0.3)

        # Create legend for basin map
        green_patch = mpatches.Patch(color='#ccffcc', label='Converges')
        red_patch = mpatches.Patch(color='#ffcccc', label='Diverges')
        ax2.legend(handles=[green_patch, red_patch], loc='lower right')

        # Plot 3: Convergence time map
        ax3 = plt.subplot(2, 3, 3)
        iteration_plot = iteration_map.copy()
        iteration_plot[iteration_plot < 0] = np.nan  # Don't plot divergent points

        im3 = ax3.contourf(X, Y, iteration_plot, levels=20, cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='Iterations to converge')
        ax3.plot(0, 0, 'rx', markersize=12, linewidth=3)
        ax3.set_xlabel('X (m)', fontsize=11)
        ax3.set_ylabel('Y (m)', fontsize=11)
        ax3.set_title('Convergence Speed Map', fontsize=12, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)

        # Plot 4: Scalar field contours
        ax4 = plt.subplot(2, 3, 4)
        Z = self.scalar_field(X, Y)
        contour = ax4.contour(X, Y, Z, levels=20, cmap='viridis')
        ax4.clabel(contour, inline=True, fontsize=8)
        ax4.plot(0, 0, 'rx', markersize=12, linewidth=3, label='Saddle')
        ax4.plot(-2, 0, 'b*', markersize=10, label='Peak 1')
        ax4.plot(2, 0, 'b*', markersize=10, label='Peak 2')
        ax4.set_xlabel('X (m)', fontsize=11)
        ax4.set_ylabel('Y (m)', fontsize=11)
        ax4.set_title('Bimodal Gaussian Field', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.set_xlim(-3, 3)
        ax4.set_ylim(-3, 3)

        # Plot 5: Basin with overlaid field
        ax5 = plt.subplot(2, 3, 5)
        ax5.contour(X, Y, Z, levels=10, cmap='gray', alpha=0.3, linewidths=1)
        ax5.contourf(X, Y, convergence_map, levels=[0, 0.5, 1],
                    colors=['#ffcccc', '#ccffcc'], alpha=0.6)
        ax5.contour(X, Y, convergence_map, levels=[0.5], colors='black', linewidths=2)

        # Add optimal radius
        optimal_circle2 = Circle((0, 0), optimal_radius, fill=False, edgecolor='red',
                                linestyle='-', linewidth=2)
        ax5.add_patch(optimal_circle2)

        ax5.plot(0, 0, 'rx', markersize=12, linewidth=3)
        ax5.set_xlabel('X (m)', fontsize=11)
        ax5.set_ylabel('Y (m)', fontsize=11)
        ax5.set_title('Basin Overlaid on Field', fontsize=12, fontweight='bold')
        ax5.set_aspect('equal')
        ax5.set_xlim(-3, 3)
        ax5.set_ylim(-3, 3)

        # Plot 6: Sample trajectories
        ax6 = plt.subplot(2, 3, 6)
        ax6.contour(X, Y, Z, levels=15, cmap='gray', alpha=0.3, linewidths=1)

        # Show some sample trajectories
        n_samples = 8
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_samples))

        for angle, color in zip(angles, colors):
            x_start = optimal_radius * np.cos(angle)
            y_start = optimal_radius * np.sin(angle)

            # Simulate trajectory
            trajectory = [[x_start, y_start]]
            pos = np.array([x_start, y_start])

            for _ in range(200):  # Fewer steps for visualization
                step = self.newton_step(pos)
                pos = pos + self.step_size * step
                trajectory.append(pos.copy())
                if np.linalg.norm(pos) < 0.1:
                    break

            trajectory = np.array(trajectory)
            ax6.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5, alpha=0.7)
            ax6.plot(x_start, y_start, 'o', color=color, markersize=6)

        ax6.plot(0, 0, 'rx', markersize=12, linewidth=3)
        ax6.set_xlabel('X (m)', fontsize=11)
        ax6.set_ylabel('Y (m)', fontsize=11)
        ax6.set_title(f'Sample Trajectories (r={optimal_radius:.2f}m)', fontsize=12, fontweight='bold')
        ax6.set_aspect('equal')
        ax6.set_xlim(-3, 3)
        ax6.set_ylim(-3, 3)

        plt.suptitle('Basin of Attraction Analysis for Newton\'s Method at Saddle Point',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('basin_of_attraction_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        return fig

# Main execution
if __name__ == "__main__":
    analyzer = NewtonSaddleAnalyzer(step_size=0.01, max_iterations=20000)

    print("Finding optimal radius for convergence...")
    optimal_radius, radii, success_rates = analyzer.find_optimal_radius()

    print("\nCreating basin of attraction map...")
    X, Y, convergence_map, iteration_map = analyzer.create_basin_map(grid_size=60)

    # Count converged points
    total_points = convergence_map.size
    converged_points = np.sum(convergence_map)
    print(f"\nBasin statistics:")
    print(f"Total grid points: {total_points}")
    print(f"Converged points: {converged_points}")
    print(f"Overall success rate: {converged_points/total_points*100:.1f}%")

    # Test specific radius success
    print(f"\nTesting optimal radius {optimal_radius:.2f}m with 100 samples...")
    success_100 = analyzer.test_radius_success_rate(optimal_radius, n_samples=100)
    print(f"Success rate at r={optimal_radius:.2f}m: {success_100*100:.1f}%")

    print("\nGenerating visualization...")
    fig = analyzer.plot_results(optimal_radius, radii, success_rates, X, Y, convergence_map, iteration_map)

    print("\nAnalysis complete! Saved as 'basin_of_attraction_analysis.png'")