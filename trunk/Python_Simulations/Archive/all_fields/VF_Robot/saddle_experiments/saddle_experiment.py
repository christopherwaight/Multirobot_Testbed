"""
Standalone Saddle Point Finding Experiment

Demonstrates Newton's method for finding saddle points using a 4-robot formation.
Replicates the functionality from Saddle_point.ipynb notebook.

USAGE:
    python3 saddle_experiment.py

This script:
1. Creates a 4-robot cluster in square formation
2. Places it in a bimodal Gaussian scalar field with saddle at origin
3. Uses Newton's method with Hessian estimation to find the saddle
4. Visualizes the convergence with contour plots and diagnostics

CONFIGURATION:
    - NUM_ROBOTS: 4 (required for Hessian estimation)
    - FORMATION: Square (quad_square_default.yaml)
    - FIELD: Bimodal Gaussian saddle (from Saddle_point.ipynb)
    - CONTROL: Newton's method with optional rotation control
    - SIM_TIME: Number of steps (increase for full convergence)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, project_root)

# Import robot and field classes
from src.robot.quad_cluster import QuadCluster
from src.fields.scalar_field_types import AnalyticalScalarField
from src.fields.scalar_environments.Saddle import bimodal_saddle1
import src.control.scalar_quad_primitives as sqcp


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Simulation parameters
SIM_TIME = 500  # Number of simulation steps (increase to 1000-2000 for full convergence)
TIMESTEP = 0.1  # Time step for robot dynamics
MOMENTUM_ALPHA = 0.7  # Momentum coefficient (higher = more inertia)

# Formation configuration
FORMATION_CONFIG = "config/formations/quad_square_default.yaml"

# Control primitive selection
USE_ROTATION_CONTROL = True  # Set to False for Newton without rotation

# Starting position
START_POSITION = (0.5, -2.0)  # Away from saddle at (0, 0)

# Visualization parameters
SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "saddle_experiments/results"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_simulation_info():
    """Print simulation configuration."""
    print("=" * 70)
    print("SADDLE POINT FINDING EXPERIMENT")
    print("=" * 70)
    print(f"Scalar Field:      Bimodal Gaussian (saddle at origin)")
    print(f"Num Robots:        4 (square formation)")
    print(f"Control Method:    Newton's Method" +
          (" with rotation control" if USE_ROTATION_CONTROL else ""))
    print(f"Formation:         {FORMATION_CONFIG}")
    print(f"Simulation Steps:  {SIM_TIME}")
    print(f"Start Position:    {START_POSITION}")
    print(f"Timestep:          {TIMESTEP}s")
    print(f"Momentum Alpha:    {MOMENTUM_ALPHA}")
    print("=" * 70)
    print()


def create_field_grid(x_range=(-3, 3), y_range=(-3, 3), resolution=100):
    """Create meshgrid for field visualization."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = bimodal_saddle1(X, Y)
    return X, Y, Z


def save_diagnostics(diagnostics, filename="diagnostics.txt"):
    """Save diagnostic information to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w') as f:
        f.write("SADDLE POINT FINDING DIAGNOSTICS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total iterations: {len(diagnostics)}\n")
        f.write(f"Initial position: ({diagnostics[0]['x_c']:.4f}, {diagnostics[0]['y_c']:.4f})\n")
        f.write(f"Final position: ({diagnostics[-1]['x_c']:.4f}, {diagnostics[-1]['y_c']:.4f})\n")
        f.write(f"Distance to saddle: {diagnostics[-1]['dist_to_origin']:.6f}\n")
        f.write(f"Final gradient magnitude: {diagnostics[-1]['grad_magnitude']:.6f}\n")

        if 'hessian_eigenvalues' in diagnostics[-1]:
            eigenvals = diagnostics[-1]['hessian_eigenvalues']
            f.write(f"Final Hessian eigenvalues: [{eigenvals[0]:.6f}, {eigenvals[1]:.6f}]\n")
            if eigenvals[0] * eigenvals[1] < 0:
                f.write("STATUS: SADDLE POINT DETECTED (opposite sign eigenvalues)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Iteration History (showing every 50th step):\n\n")

        for i in range(0, len(diagnostics), 50):
            diag = diagnostics[i]
            f.write(f"Step {i:4d}: ")
            f.write(f"pos=({diag['x_c']:7.4f}, {diag['y_c']:7.4f}), ")
            f.write(f"dist={diag['dist_to_origin']:.4f}, ")
            f.write(f"|∇z|={diag['grad_magnitude']:.4f}\n")

    print(f"Diagnostics saved to: {filepath}")


# ==============================================================================
# SIMULATION LOOP
# ==============================================================================

def run_simulation():
    """Run the saddle point finding simulation."""
    # Create scalar field
    print("Creating bimodal Gaussian scalar field...")
    field = AnalyticalScalarField(bimodal_saddle1)

    # Create robot cluster
    print(f"Initializing 4-robot cluster with formation: {FORMATION_CONFIG}...")
    cluster = QuadCluster(FORMATION_CONFIG, field, TIMESTEP, MOMENTUM_ALPHA)

    # Reset to starting position
    cluster.reset(x_c=START_POSITION[0], y_c=START_POSITION[1])

    # Select control primitive
    if USE_ROTATION_CONTROL:
        control_primitive = sqcp.newton_saddle_finder_with_rotation
        print("Using: Newton's method WITH rotation control")
    else:
        control_primitive = sqcp.newton_saddle_finder
        print("Using: Newton's method WITHOUT rotation control")

    print(f"\nRunning simulation for {SIM_TIME} steps...")
    print()

    # Storage for diagnostics
    diagnostics = []

    # Main simulation loop
    for step in range(SIM_TIME):
        # Get current state
        formation = cluster.get_current_formation()
        centroid = cluster.get_centroid()

        # Estimate gradient and Hessian
        avg_gradient, hessian = cluster.estimate_gradient_and_hessian()
        grad_magnitude = np.linalg.norm(avg_gradient)

        # Compute Hessian eigenvalues
        try:
            eigenvals, _ = np.linalg.eigh(hessian)
        except:
            eigenvals = [np.nan, np.nan]

        # Store diagnostics
        diag_data = {
            'step': step,
            'x_c': centroid[0],
            'y_c': centroid[1],
            'dist_to_origin': np.linalg.norm(centroid),
            'grad_magnitude': grad_magnitude,
            'gradient': avg_gradient.copy(),
            'hessian': hessian.copy(),
            'hessian_eigenvalues': eigenvals,
        }

        if USE_ROTATION_CONTROL:
            diag_data['theta_c'] = formation['th_c']

        diagnostics.append(diag_data)

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step:4d}: pos=({centroid[0]:7.4f}, {centroid[1]:7.4f}), "
                  f"dist={diag_data['dist_to_origin']:.4f}, |∇z|={grad_magnitude:.4f}")

        # Move cluster
        cluster.move(control_primitive)

    # Final state
    final_centroid = cluster.get_centroid()
    final_dist = np.linalg.norm(final_centroid)

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Initial position:     {START_POSITION}")
    print(f"Final position:       ({final_centroid[0]:.6f}, {final_centroid[1]:.6f})")
    print(f"Distance to saddle:   {final_dist:.6f}")
    print(f"Final |∇z|:           {diagnostics[-1]['grad_magnitude']:.6f}")
    print(f"Hessian eigenvalues:  {diagnostics[-1]['hessian_eigenvalues']}")
    print("=" * 70)
    print()

    return cluster, diagnostics


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_results(cluster, diagnostics):
    """Create visualization plots."""
    print("Generating visualization...")

    # Extract trajectory
    trajectory = np.array([[d['x_c'], d['y_c']] for d in diagnostics])

    # Create field grid
    X, Y, Z = create_field_grid()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))

    # Subplot 1: 2D Contour with trajectory
    ax1 = fig.add_subplot(2, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Plot trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    ax1.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    ax1.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='True Saddle')

    # Plot final robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    ax1.plot([x1, x2, x3, x4], [y1, y2, y3, y4], 'bs', markersize=8, label='Final Robots')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Saddle Point Navigation - 2D View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Subplot 2: 3D Surface with trajectory
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    # Plot trajectory in 3D
    z_trajectory = np.array([bimodal_saddle1(pos[0], pos[1]) for pos in trajectory])
    ax2.plot(trajectory[:, 0], trajectory[:, 1], z_trajectory, 'r-', linewidth=3, label='Trajectory')
    ax2.scatter([START_POSITION[0]], [START_POSITION[1]],
               [bimodal_saddle1(START_POSITION[0], START_POSITION[1])],
               color='green', s=100, marker='o', label='Start')
    ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [z_trajectory[-1]],
               color='red', s=100, marker='o', label='End')
    ax2.scatter([0], [0], [bimodal_saddle1(0, 0)],
               color='black', s=100, marker='x', label='Saddle')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Surface View')
    ax2.legend()

    # Subplot 3: Distance to saddle vs iteration
    ax3 = fig.add_subplot(2, 3, 3)
    distances = [d['dist_to_origin'] for d in diagnostics]
    ax3.plot(distances, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Distance to Saddle Point')
    ax3.set_title('Convergence to Saddle')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Subplot 4: Gradient magnitude vs iteration
    ax4 = fig.add_subplot(2, 3, 4)
    grad_mags = [d['grad_magnitude'] for d in diagnostics]
    ax4.plot(grad_mags, 'g-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Gradient Magnitude |∇z|')
    ax4.set_title('Gradient Magnitude Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Subplot 5: Hessian eigenvalues vs iteration
    ax5 = fig.add_subplot(2, 3, 5)
    eigenvals_history = np.array([d['hessian_eigenvalues'] for d in diagnostics])
    ax5.plot(eigenvals_history[:, 0], 'c-', linewidth=2, label='λ₁ (smaller)')
    ax5.plot(eigenvals_history[:, 1], 'y-', linewidth=2, label='λ₂ (larger)')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Eigenvalue')
    ax5.set_title('Hessian Eigenvalues Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Subplot 6: X and Y position vs iteration
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot([d['x_c'] for d in diagnostics], 'r-', linewidth=2, label='X position', alpha=0.7)
    ax6.plot([d['y_c'] for d in diagnostics], 'b-', linewidth=2, label='Y position', alpha=0.7)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Position')
    ax6.set_title('X and Y Positions Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"saddle_finding_{'with' if USE_ROTATION_CONTROL else 'without'}_rotation.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    if SHOW_PLOTS:
        plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print_simulation_info()

    # Run simulation
    cluster, diagnostics = run_simulation()

    # Save diagnostics
    save_diagnostics(diagnostics)

    # Visualize results
    if SHOW_PLOTS or SAVE_PLOTS:
        visualize_results(cluster, diagnostics)

    print("\nExperiment complete!")
