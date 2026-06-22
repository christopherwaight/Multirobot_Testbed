"""
Saddle Point Finding: Comparison of Rotation Methods

Runs both Newton WITHOUT rotation and WITH rotation from the same starting position
and generates side-by-side comparison plots and metrics.

USAGE:
    python3 compare_rotation_methods.py

OUTPUT:
    - Comparison plots showing both methods side-by-side
    - Metrics table comparing convergence performance
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

SIM_TIME = 1000  # Both methods get same number of steps
TIMESTEP = 0.1
MOMENTUM_ALPHA = 0.7
FORMATION_CONFIG = "config/formations/quad_square_default.yaml"
START_POSITION = (0.5, -2.0)  # Same start for fair comparison

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "saddle_experiments/results"


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================

def run_method(method_name, control_primitive, sim_time):
    """Run a single method and collect diagnostics."""
    print(f"\nRunning {method_name}...")

    field = AnalyticalScalarField(bimodal_saddle1)
    cluster = QuadCluster(FORMATION_CONFIG, field, TIMESTEP, MOMENTUM_ALPHA)
    cluster.reset(x_c=START_POSITION[0], y_c=START_POSITION[1])

    diagnostics = []

    for step in range(sim_time):
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
            'theta_c': formation['th_c'],
        }
        diagnostics.append(diag_data)

        if step % 200 == 0:
            print(f"  Step {step:4d}: dist={diag_data['dist_to_origin']:.4f}, |∇z|={grad_magnitude:.4f}")

        cluster.move(control_primitive)

    print(f"{method_name} complete!")
    return cluster, diagnostics


def compute_metrics(diagnostics, method_name):
    """Compute performance metrics."""
    trajectory = np.array([[d['x_c'], d['y_c']] for d in diagnostics])

    # Path length (total distance traveled)
    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += np.linalg.norm(trajectory[i] - trajectory[i-1])

    # Straight line distance from start to end
    straight_line = np.linalg.norm(trajectory[-1] - trajectory[0])

    # Path efficiency (1.0 = straight line, <1.0 = meandering)
    path_efficiency = straight_line / path_length if path_length > 0 else 0.0

    # Final convergence metrics
    final_dist = diagnostics[-1]['dist_to_origin']
    final_grad = diagnostics[-1]['grad_magnitude']
    final_eigenvals = diagnostics[-1]['hessian_eigenvalues']

    # Total rotation (unwrapped angle change)
    total_rotation = abs(diagnostics[-1]['theta_c'] - diagnostics[0]['theta_c'])

    # Steps to convergence (when dist < 0.1)
    steps_to_converge = None
    for i, d in enumerate(diagnostics):
        if d['dist_to_origin'] < 0.1:
            steps_to_converge = i
            break

    metrics = {
        'method': method_name,
        'final_distance': final_dist,
        'final_gradient': final_grad,
        'final_eigenvalues': final_eigenvals,
        'path_length': path_length,
        'straight_line': straight_line,
        'path_efficiency': path_efficiency,
        'total_rotation': np.degrees(total_rotation),
        'steps_to_converge': steps_to_converge if steps_to_converge else SIM_TIME,
    }

    return metrics


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_field_grid():
    """Create meshgrid for visualization."""
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = bimodal_saddle1(X, Y)
    return X, Y, Z


def create_comparison_plots(diagnostics_no_rot, diagnostics_with_rot, metrics_no_rot, metrics_with_rot):
    """Create comprehensive comparison visualization."""
    print("\nGenerating comparison plots...")

    traj_no_rot = np.array([[d['x_c'], d['y_c']] for d in diagnostics_no_rot])
    traj_with_rot = np.array([[d['x_c'], d['y_c']] for d in diagnostics_with_rot])

    X, Y, Z = create_field_grid()

    fig = plt.figure(figsize=(20, 12))

    # ==================================================
    # Row 1: Trajectories
    # ==================================================

    # Plot 1: NO rotation trajectory
    ax1 = fig.add_subplot(3, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.plot(traj_no_rot[:, 0], traj_no_rot[:, 1], 'b-', linewidth=2, alpha=0.8)
    ax1.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax1.plot(traj_no_rot[-1, 0], traj_no_rot[-1, 1], 'ro', markersize=10, label='End')
    ax1.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('NO Rotation Control', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: WITH rotation trajectory
    ax2 = fig.add_subplot(3, 3, 2)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax2.plot(traj_with_rot[:, 0], traj_with_rot[:, 1], 'r-', linewidth=2, alpha=0.8)
    ax2.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax2.plot(traj_with_rot[-1, 0], traj_with_rot[-1, 1], 'ro', markersize=10, label='End')
    ax2.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('WITH Rotation Control', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Plot 3: Overlay comparison
    ax3 = fig.add_subplot(3, 3, 3)
    contour = ax3.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
    ax3.plot(traj_no_rot[:, 0], traj_no_rot[:, 1], 'b-', linewidth=2, alpha=0.7, label='NO rotation')
    ax3.plot(traj_with_rot[:, 0], traj_with_rot[:, 1], 'r-', linewidth=2, alpha=0.7, label='WITH rotation')
    ax3.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax3.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Trajectory Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # ==================================================
    # Row 2: Convergence Metrics
    # ==================================================

    # Plot 4: Distance comparison
    ax4 = fig.add_subplot(3, 3, 4)
    dist_no_rot = [d['dist_to_origin'] for d in diagnostics_no_rot]
    dist_with_rot = [d['dist_to_origin'] for d in diagnostics_with_rot]
    ax4.plot(dist_no_rot, 'b-', linewidth=2, alpha=0.7, label='NO rotation')
    ax4.plot(dist_with_rot, 'r-', linewidth=2, alpha=0.7, label='WITH rotation')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Distance to Saddle')
    ax4.set_title('Convergence Comparison (log scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Plot 5: Gradient magnitude comparison
    ax5 = fig.add_subplot(3, 3, 5)
    grad_no_rot = [d['grad_magnitude'] for d in diagnostics_no_rot]
    grad_with_rot = [d['grad_magnitude'] for d in diagnostics_with_rot]
    ax5.plot(grad_no_rot, 'b-', linewidth=2, alpha=0.7, label='NO rotation')
    ax5.plot(grad_with_rot, 'r-', linewidth=2, alpha=0.7, label='WITH rotation')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('|∇z|')
    ax5.set_title('Gradient Magnitude Comparison (log scale)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')

    # Plot 6: Hessian eigenvalues comparison (WITH rotation only)
    ax6 = fig.add_subplot(3, 3, 6)
    eigenvals_with_rot = np.array([d['hessian_eigenvalues'] for d in diagnostics_with_rot])
    ax6.plot(eigenvals_with_rot[:, 0], 'c-', linewidth=2, label='λ₁ (WITH rot)')
    ax6.plot(eigenvals_with_rot[:, 1], 'y-', linewidth=2, label='λ₂ (WITH rot)')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Eigenvalue')
    ax6.set_title('Hessian Eigenvalues (WITH Rotation)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ==================================================
    # Row 3: Rotation Analysis & Metrics
    # ==================================================

    # Plot 7: Formation orientation comparison
    ax7 = fig.add_subplot(3, 3, 7)
    theta_no_rot = [np.degrees(d['theta_c']) for d in diagnostics_no_rot]
    theta_with_rot = [np.degrees(d['theta_c']) for d in diagnostics_with_rot]
    ax7.plot(theta_no_rot, 'b-', linewidth=2, alpha=0.7, label='NO rotation')
    ax7.plot(theta_with_rot, 'r-', linewidth=2, alpha=0.7, label='WITH rotation')
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Formation Angle (degrees)')
    ax7.set_title('Formation Orientation Over Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: X and Y positions comparison
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot([d['x_c'] for d in diagnostics_no_rot], 'b-', linewidth=2, alpha=0.5, label='X (NO rot)')
    ax8.plot([d['y_c'] for d in diagnostics_no_rot], 'b--', linewidth=2, alpha=0.5, label='Y (NO rot)')
    ax8.plot([d['x_c'] for d in diagnostics_with_rot], 'r-', linewidth=2, alpha=0.5, label='X (WITH rot)')
    ax8.plot([d['y_c'] for d in diagnostics_with_rot], 'r--', linewidth=2, alpha=0.5, label='Y (WITH rot)')
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Position')
    ax8.set_title('X and Y Position Comparison')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # Plot 9: Metrics table
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    # Create metrics table
    table_data = [
        ['Metric', 'NO Rotation', 'WITH Rotation', 'Improvement'],
        ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
        ['Final Distance', f"{metrics_no_rot['final_distance']:.6f}",
         f"{metrics_with_rot['final_distance']:.6f}",
         f"{(1 - metrics_with_rot['final_distance']/metrics_no_rot['final_distance'])*100:.1f}%"],
        ['Final |∇z|', f"{metrics_no_rot['final_gradient']:.6f}",
         f"{metrics_with_rot['final_gradient']:.6f}",
         f"{(1 - metrics_with_rot['final_gradient']/metrics_no_rot['final_gradient'])*100:.1f}%"],
        ['Steps to Converge', f"{metrics_no_rot['steps_to_converge']}",
         f"{metrics_with_rot['steps_to_converge']}",
         f"{(1 - metrics_with_rot['steps_to_converge']/metrics_no_rot['steps_to_converge'])*100:.1f}%"],
        ['Path Length', f"{metrics_no_rot['path_length']:.3f}",
         f"{metrics_with_rot['path_length']:.3f}",
         f"{(1 - metrics_with_rot['path_length']/metrics_no_rot['path_length'])*100:.1f}%"],
        ['Path Efficiency', f"{metrics_no_rot['path_efficiency']:.3f}",
         f"{metrics_with_rot['path_efficiency']:.3f}",
         f"{(metrics_with_rot['path_efficiency']/metrics_no_rot['path_efficiency'] - 1)*100:.1f}%"],
        ['Total Rotation', f"{metrics_no_rot['total_rotation']:.1f}°",
         f"{metrics_with_rot['total_rotation']:.1f}°", '─'],
    ]

    table_text = '\n'.join(['  '.join(row) for row in table_data])
    ax9.text(0.1, 0.5, table_text, fontfamily='monospace', fontsize=9,
             verticalalignment='center', transform=ax9.transAxes)
    ax9.set_title('Performance Comparison', fontweight='bold', pad=20)

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, "rotation_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {filepath}")

    if SHOW_PLOTS:
        plt.show()


def print_metrics_table(metrics_no_rot, metrics_with_rot):
    """Print metrics comparison table to console."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} {'NO Rotation':<20} {'WITH Rotation':<20} {'Change':<15}")
    print("-" * 80)

    print(f"{'Final Distance':<25} {metrics_no_rot['final_distance']:<20.6f} "
          f"{metrics_with_rot['final_distance']:<20.6f} "
          f"{(1 - metrics_with_rot['final_distance']/metrics_no_rot['final_distance'])*100:>+6.1f}%")

    print(f"{'Final |∇z|':<25} {metrics_no_rot['final_gradient']:<20.6f} "
          f"{metrics_with_rot['final_gradient']:<20.6f} "
          f"{(1 - metrics_with_rot['final_gradient']/metrics_no_rot['final_gradient'])*100:>+6.1f}%")

    print(f"{'Steps to Converge':<25} {metrics_no_rot['steps_to_converge']:<20} "
          f"{metrics_with_rot['steps_to_converge']:<20} "
          f"{(1 - metrics_with_rot['steps_to_converge']/metrics_no_rot['steps_to_converge'])*100:>+6.1f}%")

    print(f"{'Path Length':<25} {metrics_no_rot['path_length']:<20.3f} "
          f"{metrics_with_rot['path_length']:<20.3f} "
          f"{(1 - metrics_with_rot['path_length']/metrics_no_rot['path_length'])*100:>+6.1f}%")

    print(f"{'Path Efficiency':<25} {metrics_no_rot['path_efficiency']:<20.3f} "
          f"{metrics_with_rot['path_efficiency']:<20.3f} "
          f"{(metrics_with_rot['path_efficiency']/metrics_no_rot['path_efficiency'] - 1)*100:>+6.1f}%")

    print(f"{'Total Rotation':<25} {metrics_no_rot['total_rotation']:<20.1f}° "
          f"{metrics_with_rot['total_rotation']:<20.1f}° {'─':<15}")

    print("=" * 80)
    print()

    # Saddle point verification
    print("SADDLE POINT VERIFICATION")
    print("-" * 80)
    eigenvals = metrics_with_rot['final_eigenvalues']
    print(f"Final Hessian eigenvalues: [{eigenvals[0]:.6f}, {eigenvals[1]:.6f}]")
    if eigenvals[0] * eigenvals[1] < 0:
        print("✓ SADDLE POINT CONFIRMED (opposite sign eigenvalues)")
    else:
        print("⚠ Warning: Eigenvalues have same sign")
    print("=" * 80)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SADDLE POINT FINDING: ROTATION CONTROL COMPARISON")
    print("=" * 80)
    print(f"Starting position: {START_POSITION}")
    print(f"Simulation steps: {SIM_TIME}")
    print(f"Formation: {FORMATION_CONFIG}")
    print("=" * 80)

    # Run both methods
    cluster_no_rot, diagnostics_no_rot = run_method(
        "Newton WITHOUT Rotation",
        sqcp.newton_saddle_finder,
        SIM_TIME
    )

    cluster_with_rot, diagnostics_with_rot = run_method(
        "Newton WITH Rotation",
        sqcp.newton_saddle_finder_with_rotation,
        SIM_TIME
    )

    # Compute metrics
    print("\nComputing performance metrics...")
    metrics_no_rot = compute_metrics(diagnostics_no_rot, "NO Rotation")
    metrics_with_rot = compute_metrics(diagnostics_with_rot, "WITH Rotation")

    # Print metrics table
    print_metrics_table(metrics_no_rot, metrics_with_rot)

    # Generate comparison plots
    create_comparison_plots(diagnostics_no_rot, diagnostics_with_rot,
                           metrics_no_rot, metrics_with_rot)

    print("\nComparison complete!")
    print("\nKey Findings:")
    print("  - Check trajectory overlay (plot 3) to see path differences")
    print("  - Check convergence plots (plots 4-5) to compare speed")
    print("  - Check orientation plot (plot 7) to see rotation behavior")
    print("  - Check metrics table (plot 9) for quantitative comparison")
