"""
Saddle Point Finding: Newton's Method WITH Rotation Control

This is the ENHANCED version - formation actively rotates to align with
the Newton step direction for improved convergence.

COMPARISON: Run this alongside saddle_experiment_NO_rotation.py to see
the benefit of active rotation control.

USAGE:
    python3 saddle_experiment_WITH_rotation.py
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

from src.robot.quad_cluster import QuadCluster
from src.fields.scalar_field_types import AnalyticalScalarField
from src.fields.scalar_environments.Saddle import bimodal_saddle1
import src.control.scalar_quad_primitives as sqcp


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SIM_TIME = 1000  # Longer to show full rotation behavior
TIMESTEP = 0.1
MOMENTUM_ALPHA = 0.7
FORMATION_CONFIG = "config/formations/quad_square_default.yaml"
START_POSITION = (0.5, -2.0)

# Rotation control parameter (can experiment with 0.02, 0.1, 0.3)
ROTATION_GAIN = 0.3  # Higher = more aggressive rotation

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "saddle_experiments/results"


# ==============================================================================
# SIMULATION
# ==============================================================================

def print_info():
    """Print configuration."""
    print("=" * 70)
    print("SADDLE POINT FINDING: Newton WITH Rotation Control (ENHANCED)")
    print("=" * 70)
    print(f"Field:             Bimodal Gaussian (saddle at origin)")
    print(f"Robots:            4 (square formation)")
    print(f"Method:            Newton's method")
    print(f"Rotation Control:  ENABLED (formation aligns with Newton direction)")
    print(f"Rotation Gain:     {ROTATION_GAIN}")
    print(f"Steps:             {SIM_TIME}")
    print(f"Start:             {START_POSITION}")
    print("=" * 70)
    print()


def run_simulation():
    """Run Newton's method WITH rotation control."""
    print("Creating field and robot cluster...")
    field = AnalyticalScalarField(bimodal_saddle1)
    cluster = QuadCluster(FORMATION_CONFIG, field, TIMESTEP, MOMENTUM_ALPHA)
    cluster.reset(x_c=START_POSITION[0], y_c=START_POSITION[1])

    print("Using: newton_saddle_finder_with_rotation")
    print(f"Running simulation for {SIM_TIME} steps...\n")

    diagnostics = []

    for step in range(SIM_TIME):
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()
        avg_gradient, hessian = cluster.estimate_gradient_and_hessian()
        grad_magnitude = np.linalg.norm(avg_gradient)

        try:
            eigenvals, _ = np.linalg.eigh(hessian)
        except:
            eigenvals = [np.nan, np.nan]

        # Compute Newton step to track rotation alignment
        try:
            det = np.linalg.det(hessian)
            if abs(det) > 1e-10:
                newton_step = -np.linalg.solve(hessian, avg_gradient)
            else:
                newton_step = -np.linalg.pinv(hessian) @ avg_gradient
        except:
            newton_step = -avg_gradient if grad_magnitude > 0 else np.array([0., 0.])

        # Limit step
        if np.linalg.norm(newton_step) > 1.0:
            newton_step = newton_step / np.linalg.norm(newton_step)

        # Calculate Newton step angle
        newton_angle = np.arctan2(newton_step[1], newton_step[0])

        # Calculate rotation error (how misaligned formation is)
        current_theta = formation['th_c']
        best_error = float('inf')
        for k in range(4):  # 4-fold symmetry
            target_angle = newton_angle - k * np.pi/2
            error = target_angle - current_theta
            error = np.arctan2(np.sin(error), np.cos(error))
            if abs(error) < abs(best_error):
                best_error = error

        rotation_error = best_error

        diag_data = {
            'step': step,
            'x_c': centroid[0],
            'y_c': centroid[1],
            'dist_to_origin': np.linalg.norm(centroid),
            'grad_magnitude': grad_magnitude,
            'gradient': avg_gradient.copy(),
            'hessian': hessian.copy(),
            'hessian_eigenvalues': eigenvals,
            'theta_c': current_theta,  # Formation angle
            'newton_angle': newton_angle,  # Newton step direction
            'rotation_error': rotation_error,  # Misalignment
            'newton_step': newton_step.copy(),
        }
        diagnostics.append(diag_data)

        if step % 100 == 0:
            print(f"Step {step:4d}: pos=({centroid[0]:7.4f}, {centroid[1]:7.4f}), "
                  f"dist={diag_data['dist_to_origin']:.4f}, |∇z|={grad_magnitude:.4f}, "
                  f"rot_err={np.degrees(rotation_error):6.2f}°")

        # Move WITH rotation control
        cluster.move(sqcp.newton_saddle_finder_with_rotation)

    final_centroid = cluster.get_centroid()
    total_rotation = abs(diagnostics[-1]['theta_c'] - diagnostics[0]['theta_c'])

    print()
    print("=" * 70)
    print("RESULTS (WITH Rotation Control)")
    print("=" * 70)
    print(f"Final position:       ({final_centroid[0]:.6f}, {final_centroid[1]:.6f})")
    print(f"Distance to saddle:   {np.linalg.norm(final_centroid):.6f}")
    print(f"Final |∇z|:           {diagnostics[-1]['grad_magnitude']:.6f}")
    print(f"Hessian eigenvalues:  {diagnostics[-1]['hessian_eigenvalues']}")
    print(f"Total rotation:       {np.degrees(total_rotation):.1f}°")
    print(f"Final rotation error: {np.degrees(diagnostics[-1]['rotation_error']):.2f}°")
    print("=" * 70)
    print()

    return cluster, diagnostics


def create_field_grid():
    """Create meshgrid for visualization."""
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = bimodal_saddle1(X, Y)
    return X, Y, Z


def visualize(cluster, diagnostics):
    """Create enhanced visualization with rotation diagnostics."""
    print("Generating plots...")

    trajectory = np.array([[d['x_c'], d['y_c']] for d in diagnostics])
    X, Y, Z = create_field_grid()

    fig = plt.figure(figsize=(20, 12))

    # 2D Contour
    ax1 = fig.add_subplot(3, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax1.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    ax1.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Newton WITH Rotation Control - 2D View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 3D Surface
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    z_traj = np.array([bimodal_saddle1(pos[0], pos[1]) for pos in trajectory])
    ax2.plot(trajectory[:, 0], trajectory[:, 1], z_traj, 'r-', linewidth=3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Surface View')

    # Distance to saddle
    ax3 = fig.add_subplot(3, 3, 3)
    distances = [d['dist_to_origin'] for d in diagnostics]
    ax3.plot(distances, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Distance to Saddle')
    ax3.set_title('Convergence (log scale)')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Gradient magnitude
    ax4 = fig.add_subplot(3, 3, 4)
    grad_mags = [d['grad_magnitude'] for d in diagnostics]
    ax4.plot(grad_mags, 'g-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('|∇z|')
    ax4.set_title('Gradient Magnitude (log scale)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Hessian eigenvalues
    ax5 = fig.add_subplot(3, 3, 5)
    eigenvals = np.array([d['hessian_eigenvalues'] for d in diagnostics])
    ax5.plot(eigenvals[:, 0], 'c-', linewidth=2, label='λ₁')
    ax5.plot(eigenvals[:, 1], 'y-', linewidth=2, label='λ₂')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Eigenvalue')
    ax5.set_title('Hessian Eigenvalues')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Formation orientation WITH Newton direction overlay
    ax6 = fig.add_subplot(3, 3, 6)
    theta_history = [np.degrees(d['theta_c']) for d in diagnostics]
    newton_angles = [np.degrees(d['newton_angle']) for d in diagnostics]
    ax6.plot(theta_history, 'b-', linewidth=2, alpha=0.7, label='Formation Angle')
    ax6.plot(newton_angles, 'r--', linewidth=1.5, alpha=0.5, label='Newton Direction')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Angle (degrees)')
    ax6.set_title('Formation Alignment (Rotation Control Active)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.text(0.5, 0.05, 'Blue line should track red dashes\n(formation aligns with Newton step)',
             transform=ax6.transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Rotation error magnitude
    ax7 = fig.add_subplot(3, 3, 7)
    rot_errors = [np.degrees(abs(d['rotation_error'])) for d in diagnostics]
    ax7.plot(rot_errors, 'm-', linewidth=2)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('|Rotation Error| (degrees)')
    ax7.set_title('Formation Misalignment Over Time')
    ax7.grid(True, alpha=0.3)
    ax7.text(0.5, 0.95, 'Should decrease as formation\naligns with approach direction',
             transform=ax7.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # X and Y position
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot([d['x_c'] for d in diagnostics], 'r-', linewidth=2, label='X', alpha=0.7)
    ax8.plot([d['y_c'] for d in diagnostics], 'b-', linewidth=2, label='Y', alpha=0.7)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Position')
    ax8.set_title('X and Y Positions')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Newton step magnitude
    ax9 = fig.add_subplot(3, 3, 9)
    newton_mags = [np.linalg.norm(d['newton_step']) for d in diagnostics]
    ax9.plot(newton_mags, 'orange', linewidth=2)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Newton Step Magnitude')
    ax9.set_title('Newton Step Size Over Time')
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, "saddle_WITH_rotation.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filepath}")

    if SHOW_PLOTS:
        plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print_info()
    cluster, diagnostics = run_simulation()
    visualize(cluster, diagnostics)
    print("Experiment complete!")
    print("\nKey rotation metrics:")
    print(f"  - Formation actively rotated to align with Newton direction")
    print(f"  - Check plot 6: Blue (formation) should track red (Newton direction)")
    print(f"  - Check plot 7: Rotation error should decrease over time")
