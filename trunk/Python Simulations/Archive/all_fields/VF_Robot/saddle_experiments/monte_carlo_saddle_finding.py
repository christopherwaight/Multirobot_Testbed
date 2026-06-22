"""
Monte Carlo Analysis of Newton's Method with Rotation Control

Tests Newton's method with rotation control from 100 random starting positions
uniformly distributed on a circle around the saddle point.

USAGE:
    python3 monte_carlo_saddle_finding.py

OUTPUT:
    - Statistical summary (mean, std, success rate)
    - Trajectory bundle plot
    - Distribution histograms
    - Results saved to results/monte_carlo_results.txt
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

NUM_TRIALS = 100
SIM_TIME = 500  # Max steps (reduced since closer starts)
TIMESTEP = 0.1
MOMENTUM_ALPHA = 0.7
FORMATION_CONFIG = "config/formations/quad_square_default.yaml"
START_RADIUS = 1.0  # meters from origin (reduced for better convergence)

CONVERGENCE_THRESHOLD = 0.1  # meters
SUCCESS_THRESHOLD = 0.01  # meters (final distance for success)

SHOW_PLOTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "saddle_experiments/results"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_path_length(trajectory):
    """Compute total path length from trajectory."""
    if len(trajectory) < 2:
        return 0.0

    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
    return path_length


# ==============================================================================
# MONTE CARLO SIMULATION
# ==============================================================================

def run_monte_carlo():
    """Run 100 trials from random starting positions."""
    print("=" * 80)
    print("MONTE CARLO ANALYSIS: Newton WITH Rotation Control")
    print("=" * 80)
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Starting radius: {START_RADIUS} m")
    print(f"Max simulation time: {SIM_TIME} steps")
    print("=" * 80)
    print()

    # Generate random starting positions uniformly on circle
    angles = np.linspace(0, 2*np.pi, NUM_TRIALS, endpoint=False)
    starts = [(START_RADIUS * np.cos(a), START_RADIUS * np.sin(a)) for a in angles]

    # Run trials
    results = []

    for trial_num, (x_start, y_start) in enumerate(starts):
        # Setup
        field = AnalyticalScalarField(bimodal_saddle1)
        cluster = QuadCluster(FORMATION_CONFIG, field, TIMESTEP, MOMENTUM_ALPHA)
        cluster.reset(x_c=x_start, y_c=y_start)

        # Run simulation
        trajectory = []
        converged_at = None

        for step in range(SIM_TIME):
            centroid = cluster.get_centroid()
            dist = np.linalg.norm(centroid)
            trajectory.append(centroid.copy())

            # Check convergence
            if converged_at is None and dist < CONVERGENCE_THRESHOLD:
                converged_at = step

            # Stop if reached final convergence
            if dist < SUCCESS_THRESHOLD:
                break

            cluster.move(sqcp.newton_saddle_finder_with_rotation)

        # Compute metrics
        trajectory_array = np.array(trajectory)
        final_dist = np.linalg.norm(trajectory_array[-1])
        path_length = compute_path_length(trajectory_array)
        straight_line = np.linalg.norm(np.array([x_start, y_start]))
        efficiency = straight_line / path_length if path_length > 0 else 0.0
        success = final_dist < SUCCESS_THRESHOLD

        results.append({
            'trial': trial_num,
            'start': (x_start, y_start),
            'trajectory': trajectory_array,
            'final_dist': final_dist,
            'steps': len(trajectory),
            'converged_at': converged_at if converged_at is not None else len(trajectory),
            'path_length': path_length,
            'straight_line': straight_line,
            'efficiency': efficiency,
            'success': success
        })

        # Print progress every 10 trials
        if (trial_num + 1) % 10 == 0:
            print(f"Completed {trial_num + 1}/{NUM_TRIALS} trials...")

    print()
    print("All trials complete!")
    print()

    return results


def print_statistics(results):
    """Print statistical summary."""
    # Extract metrics
    final_dists = np.array([r['final_dist'] for r in results]) * 1000  # Convert to mm
    steps = np.array([r['steps'] for r in results])
    converged_at = np.array([r['converged_at'] for r in results])
    path_lengths = np.array([r['path_length'] for r in results])
    efficiencies = np.array([r['efficiency'] for r in results])
    success_rate = np.mean([r['success'] for r in results]) * 100

    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    print(f"{'Final Distance (mm)':<30} {np.mean(final_dists):<12.2f} {np.std(final_dists):<12.2f} "
          f"{np.min(final_dists):<12.2f} {np.max(final_dists):<12.2f}")
    print(f"{'Total Steps':<30} {np.mean(steps):<12.1f} {np.std(steps):<12.1f} "
          f"{np.min(steps):<12.0f} {np.max(steps):<12.0f}")
    print(f"{'Steps to Converge':<30} {np.mean(converged_at):<12.1f} {np.std(converged_at):<12.1f} "
          f"{np.min(converged_at):<12.0f} {np.max(converged_at):<12.0f}")
    print(f"{'Path Length (m)':<30} {np.mean(path_lengths):<12.3f} {np.std(path_lengths):<12.3f} "
          f"{np.min(path_lengths):<12.3f} {np.max(path_lengths):<12.3f}")
    print(f"{'Path Efficiency':<30} {np.mean(efficiencies):<12.3f} {np.std(efficiencies):<12.3f} "
          f"{np.min(efficiencies):<12.3f} {np.max(efficiencies):<12.3f}")
    print("=" * 80)
    print(f"Success Rate (final dist < {SUCCESS_THRESHOLD*1000} mm): {success_rate:.1f}%")
    print("=" * 80)
    print()


def save_results(results):
    """Save results to text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, "monte_carlo_results.txt")

    with open(filepath, 'w') as f:
        f.write("MONTE CARLO SADDLE FINDING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of trials: {NUM_TRIALS}\n")
        f.write(f"Starting radius: {START_RADIUS} m\n")
        f.write(f"Max simulation time: {SIM_TIME} steps\n\n")

        # Statistics
        final_dists = np.array([r['final_dist'] for r in results]) * 1000
        steps = np.array([r['steps'] for r in results])
        converged_at = np.array([r['converged_at'] for r in results])
        path_lengths = np.array([r['path_length'] for r in results])
        efficiencies = np.array([r['efficiency'] for r in results])
        success_rate = np.mean([r['success'] for r in results]) * 100

        f.write("STATISTICAL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Distance (mm):     Mean={np.mean(final_dists):.2f}, "
                f"Std={np.std(final_dists):.2f}, Min={np.min(final_dists):.2f}, "
                f"Max={np.max(final_dists):.2f}\n")
        f.write(f"Steps to Converge:       Mean={np.mean(converged_at):.1f}, "
                f"Std={np.std(converged_at):.1f}, Min={np.min(converged_at):.0f}, "
                f"Max={np.max(converged_at):.0f}\n")
        f.write(f"Path Length (m):         Mean={np.mean(path_lengths):.3f}, "
                f"Std={np.std(path_lengths):.3f}, Min={np.min(path_lengths):.3f}, "
                f"Max={np.max(path_lengths):.3f}\n")
        f.write(f"Path Efficiency:         Mean={np.mean(efficiencies):.3f}, "
                f"Std={np.std(efficiencies):.3f}, Min={np.min(efficiencies):.3f}, "
                f"Max={np.max(efficiencies):.3f}\n")
        f.write(f"Success Rate:            {success_rate:.1f}%\n\n")

        f.write("INDIVIDUAL TRIAL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Trial':<8} {'Start (x,y)':<20} {'Final Dist (mm)':<18} {'Steps':<8} "
                f"{'Converged':<12} {'Efficiency':<12}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r['trial']:<8} ({r['start'][0]:6.3f}, {r['start'][1]:6.3f})   "
                    f"{r['final_dist']*1000:<18.2f} {r['steps']:<8} "
                    f"{r['converged_at']:<12} {r['efficiency']:<12.3f}\n")

    print(f"Results saved to: {filepath}")


def visualize_results(results):
    """Create visualization plots."""
    print("Generating visualization plots...")

    # Create field grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = bimodal_saddle1(X, Y)

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Trajectory bundle
    ax1 = fig.add_subplot(2, 3, 1)
    contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)

    # Color trajectories by convergence time
    converged_times = np.array([r['converged_at'] for r in results])
    vmin, vmax = converged_times.min(), converged_times.max()

    for r in results:
        traj = r['trajectory']
        color_val = (r['converged_at'] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=1,
                color=plt.cm.plasma(color_val))

    ax1.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, label='Saddle')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'{NUM_TRIALS} Random Starting Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Final distance histogram
    ax2 = fig.add_subplot(2, 3, 2)
    final_dists_mm = np.array([r['final_dist'] for r in results]) * 1000
    ax2.hist(final_dists_mm, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(final_dists_mm), color='r', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Final Distance to Saddle (mm)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Final Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence time histogram
    ax3 = fig.add_subplot(2, 3, 3)
    converged_steps = np.array([r['converged_at'] for r in results])
    ax3.hist(converged_steps, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(np.mean(converged_steps), color='r', linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Steps to Convergence')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Convergence Times')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Path efficiency histogram
    ax4 = fig.add_subplot(2, 3, 4)
    efficiencies = np.array([r['efficiency'] for r in results])
    ax4.hist(efficiencies, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax4.axvline(np.mean(efficiencies), color='r', linestyle='--', linewidth=2, label='Mean')
    ax4.set_xlabel('Path Efficiency (straight-line / actual)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Path Efficiencies')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Path length histogram
    ax5 = fig.add_subplot(2, 3, 5)
    path_lengths = np.array([r['path_length'] for r in results])
    ax5.hist(path_lengths, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax5.axvline(np.mean(path_lengths), color='r', linestyle='--', linewidth=2, label='Mean')
    ax5.set_xlabel('Path Length (m)')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Path Lengths')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Statistical summary table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['─' * 20, '─' * 10, '─' * 10, '─' * 10, '─' * 10],
        ['Final Dist (mm)', f"{np.mean(final_dists_mm):.2f}", f"{np.std(final_dists_mm):.2f}",
         f"{np.min(final_dists_mm):.2f}", f"{np.max(final_dists_mm):.2f}"],
        ['Converge Steps', f"{np.mean(converged_steps):.1f}", f"{np.std(converged_steps):.1f}",
         f"{np.min(converged_steps):.0f}", f"{np.max(converged_steps):.0f}"],
        ['Path Length (m)', f"{np.mean(path_lengths):.3f}", f"{np.std(path_lengths):.3f}",
         f"{np.min(path_lengths):.3f}", f"{np.max(path_lengths):.3f}"],
        ['Path Efficiency', f"{np.mean(efficiencies):.3f}", f"{np.std(efficiencies):.3f}",
         f"{np.min(efficiencies):.3f}", f"{np.max(efficiencies):.3f}"],
        ['', '', '', '', ''],
        ['Success Rate', f"{np.mean([r['success'] for r in results])*100:.1f}%", '', '', ''],
    ]

    table_text = '\n'.join(['  '.join(row) for row in table_data])
    ax6.text(0.1, 0.5, table_text, fontfamily='monospace', fontsize=9,
             verticalalignment='center', transform=ax6.transAxes)
    ax6.set_title('Statistical Summary', fontweight='bold', pad=20)

    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, "monte_carlo_analysis.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {filepath}")

    if SHOW_PLOTS:
        plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Run Monte Carlo trials
    results = run_monte_carlo()

    # Print statistics
    print_statistics(results)

    # Save results
    save_results(results)

    # Visualize
    visualize_results(results)

    print("\nMonte Carlo analysis complete!")
