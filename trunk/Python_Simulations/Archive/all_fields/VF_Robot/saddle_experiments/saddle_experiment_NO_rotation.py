"""
Saddle Point Finding: Newton's Method WITHOUT Rotation Control

This is the BASELINE version - formation orientation is NOT controlled.
The 4-robot square formation maintains roughly constant orientation while
converging to the saddle point.

COMPARISON: Run this alongside saddle_experiment_WITH_rotation.py to see
the benefit of active formation rotation control.

USAGE:
    python3 saddle_experiment_NO_rotation.py
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

SIM_TIME = 750  # May need more steps without rotation control
TIMESTEP = 0.1
MOMENTUM_ALPHA = 0.7
FORMATION_CONFIG = "config/formations/quad_square_default.yaml"
START_POSITION = (0.5, -2.0)

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "saddle_experiments/results"


# ==============================================================================
# SIMULATION
# ==============================================================================

def print_info():
    """Print configuration."""
    print("=" * 70)
    print("SADDLE POINT FINDING: Newton WITHOUT Rotation Control (BASELINE)")
    print("=" * 70)
    print(f"Field:             Bimodal Gaussian (saddle at origin)")
    print(f"Robots:            4 (square formation)")
    print(f"Method:            Newton's method")
    print(f"Rotation Control:  DISABLED (formation orientation not controlled)")
    print(f"Steps:             {SIM_TIME}")
    print(f"Start:             {START_POSITION}")
    print("=" * 70)
    print()


def run_simulation():
    """Run Newton's method without rotation control."""
    print("Creating field and robot cluster...")
    field = AnalyticalScalarField(bimodal_saddle1)
    cluster = QuadCluster(FORMATION_CONFIG, field, TIMESTEP, MOMENTUM_ALPHA)
    cluster.reset(x_c=START_POSITION[0], y_c=START_POSITION[1])

    print("Using: newton_saddle_finder (NO rotation control)")
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

        diag_data = {
            'step': step,
            'x_c': centroid[0],
            'y_c': centroid[1],
            'dist_to_origin': np.linalg.norm(centroid),
            'grad_magnitude': grad_magnitude,
            'gradient': avg_gradient.copy(),
            'hessian': hessian.copy(),
            'hessian_eigenvalues': eigenvals,
            'theta_c': formation['th_c'],  # Track orientation (but not controlling it)
        }
        diagnostics.append(diag_data)

        if step % 100 == 0:
            print(f"Step {step:4d}: pos=({centroid[0]:7.4f}, {centroid[1]:7.4f}), "
                  f"dist={diag_data['dist_to_origin']:.4f}, |∇z|={grad_magnitude:.4f}")

        # Move WITHOUT rotation control
        cluster.move(sqcp.newton_saddle_finder)

    final_centroid = cluster.get_centroid()
    print()
    print("=" * 70)
    print("RESULTS (NO Rotation Control)")
    print("=" * 70)
    print(f"Final position:       ({final_centroid[0]:.6f}, {final_centroid[1]:.6f})")
    print(f"Distance to saddle:   {np.linalg.norm(final_centroid):.6f}")
    print(f"Final |∇z|:           {diagnostics[-1]['grad_magnitude']:.6f}")
    print(f"Hessian eigenvalues:  {diagnostics[-1]['hessian_eigenvalues']}")
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
    """Create visualization plots."""
    print("Generating plots...")

    trajectory = np.array([[d['x_c'], d['y_c']] for d in diagnostics])
    X, Y, Z = create_field_grid()

    fig = plt.figure(figsize=(18, 10))

    # 2D Contour
    ax1 = fig.add_subplot(2, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax1.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    ax1.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Newton WITHOUT Rotation Control - 2D View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 3D Surface
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    z_traj = np.array([bimodal_saddle1(pos[0], pos[1]) for pos in trajectory])
    ax2.plot(trajectory[:, 0], trajectory[:, 1], z_traj, 'r-', linewidth=3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Surface View')

    # Distance to saddle
    ax3 = fig.add_subplot(2, 3, 3)
    distances = [d['dist_to_origin'] for d in diagnostics]
    ax3.plot(distances, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Distance to Saddle')
    ax3.set_title('Convergence (log scale)')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Gradient magnitude
    ax4 = fig.add_subplot(2, 3, 4)
    grad_mags = [d['grad_magnitude'] for d in diagnostics]
    ax4.plot(grad_mags, 'g-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('|∇z|')
    ax4.set_title('Gradient Magnitude (log scale)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Hessian eigenvalues
    ax5 = fig.add_subplot(2, 3, 5)
    eigenvals = np.array([d['hessian_eigenvalues'] for d in diagnostics])
    ax5.plot(eigenvals[:, 0], 'c-', linewidth=2, label='λ₁')
    ax5.plot(eigenvals[:, 1], 'y-', linewidth=2, label='λ₂')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Eigenvalue')
    ax5.set_title('Hessian Eigenvalues')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Formation orientation (tracked but NOT controlled)
    ax6 = fig.add_subplot(2, 3, 6)
    theta_history = [np.degrees(d['theta_c']) for d in diagnostics]
    ax6.plot(theta_history, 'm-', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Formation Angle (degrees)')
    ax6.set_title('Formation Orientation (NOT Controlled)')
    ax6.grid(True, alpha=0.3)
    ax6.text(0.5, 0.95, 'Should show roughly constant angle\n(no active rotation control)',
             transform=ax6.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, "saddle_NO_rotation.png")
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
