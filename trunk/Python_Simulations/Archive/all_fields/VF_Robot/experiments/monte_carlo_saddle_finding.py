"""
Monte Carlo Analysis: 100 Random Starting Positions

All trials use Newton WITH rotation
Field: Bimodal Gaussian saddle
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, compute_hessian, compute_newton_step
from src.fields.scalar_environments.Saddle import bimodal_saddle1 as bimodal_saddle


def run_trial(start_pos, num_steps=1000, alpha=0.7, k=1.0, dt=0.1, k_r=0.3):
    """Single Newton WITH rotation trial."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])
    theta_c = 0.0
    omega = 0.0

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle, pos[0], pos[1])
        hess = compute_hessian(bimodal_saddle, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        # Rotation control - align with Newton direction
        if np.linalg.norm(delta_p) > 1e-6:
            theta_newton = np.arctan2(delta_p[1], delta_p[0])
            errors = [np.abs(np.arctan2(np.sin(theta_newton - k_sym * np.pi/2 - theta_c),
                                         np.cos(theta_newton - k_sym * np.pi/2 - theta_c)))
                      for k_sym in range(4)]
            theta_error = min(errors)
            omega_cmd = k_r * theta_error
            omega = alpha * omega + (1 - alpha) * omega_cmd
            theta_c += dt * omega

        # Translation control
        v_cmd = k * delta_p
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        new_pos = pos + dt * velocity
        trajectory.append(new_pos)

        # Convergence check
        if np.linalg.norm(grad) < 1e-6 or np.linalg.norm(pos) < 0.01:
            break

    return np.array(trajectory)


# ============================================================================
# RUN MONTE CARLO
# ============================================================================

NUM_TRIALS = 100
START_RADIUS = 2.0

# Generate 100 starting positions on circle
angles = np.linspace(0, 2 * np.pi, NUM_TRIALS, endpoint=False)
starting_positions = START_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])

print(f"Running {NUM_TRIALS} Monte Carlo trials...")
print(f"Starting positions: circle of radius {START_RADIUS} m\n")

results = []
all_trajectories = []

for i, start_pos in enumerate(starting_positions):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{NUM_TRIALS}")

    trajectory = run_trial(start_pos)
    all_trajectories.append(trajectory)

    # Compute metrics
    final_pos = trajectory[-1]
    final_dist = np.linalg.norm(final_pos)
    path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    straight_line = np.linalg.norm(start_pos)
    efficiency = straight_line / path_length if path_length > 0 else 0
    success = final_dist < 0.1  # Within 10 cm

    results.append({
        'start': start_pos,
        'final_dist': final_dist,
        'steps': len(trajectory),
        'path_length': path_length,
        'efficiency': efficiency,
        'success': success
    })

# ============================================================================
# STATISTICS
# ============================================================================

final_dists = np.array([r['final_dist'] for r in results])
steps = np.array([r['steps'] for r in results])
path_lengths = np.array([r['path_length'] for r in results])
efficiencies = np.array([r['efficiency'] for r in results])
success_rate = np.mean([r['success'] for r in results]) * 100

print(f"\n{'='*50}")
print("MONTE CARLO RESULTS")
print(f"{'='*50}")
print(f"Success rate (dist < 0.1 m): {success_rate:.1f}%")
print(f"\nFinal distance to saddle:")
print(f"  Mean:   {np.mean(final_dists)*1000:.2f} mm")
print(f"  StdDev: {np.std(final_dists)*1000:.2f} mm")
print(f"  Min:    {np.min(final_dists)*1000:.2f} mm")
print(f"  Max:    {np.max(final_dists)*1000:.2f} mm")

print(f"\nConvergence time:")
print(f"  Mean:   {np.mean(steps):.0f} steps")
print(f"  StdDev: {np.std(steps):.0f} steps")
print(f"  Min:    {np.min(steps):.0f} steps")
print(f"  Max:    {np.max(steps):.0f} steps")

print(f"\nPath efficiency:")
print(f"  Mean:   {np.mean(efficiencies):.3f}")
print(f"  StdDev: {np.std(efficiencies):.3f}")

# ============================================================================
# PLOT
# ============================================================================

fig = plt.figure(figsize=(15, 5))

# Trajectory bundle (colored by convergence time)
ax1 = plt.subplot(131)
colors = plt.cm.viridis(steps / np.max(steps))
for traj, color in zip(all_trajectories, colors):
    ax1.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=0.5, color=color)
ax1.plot(0, 0, 'r*', markersize=20, label='Saddle')
ax1.scatter(starting_positions[:, 0], starting_positions[:, 1],
            c='green', s=20, alpha=0.5, label='Starts')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title(f'All {NUM_TRIALS} Trajectories (colored by time)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Final distance histogram
ax2 = plt.subplot(132)
ax2.hist(final_dists * 1000, bins=20, edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(final_dists) * 1000, color='r', linestyle='--',
            linewidth=2, label=f'Mean: {np.mean(final_dists)*1000:.1f} mm')
ax2.set_xlabel('Final distance to saddle (mm)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Final Distances')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Convergence time histogram
ax3 = plt.subplot(133)
ax3.hist(steps, bins=20, edgecolor='black', alpha=0.7, color='orange')
ax3.axvline(np.mean(steps), color='r', linestyle='--',
            linewidth=2, label=f'Mean: {np.mean(steps):.0f} steps')
ax3.set_xlabel('Steps to converge')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Convergence Times')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved: monte_carlo_results.png")
plt.show()

# ============================================================================
# SAVE DATA
# ============================================================================

with open('monte_carlo_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Trial', 'Start_X', 'Start_Y', 'Final_Dist_mm', 'Steps',
                     'Path_Length_m', 'Efficiency', 'Success'])
    for i, r in enumerate(results):
        writer.writerow([i+1, r['start'][0], r['start'][1], r['final_dist']*1000,
                        r['steps'], r['path_length'], r['efficiency'], r['success']])
print("Data saved: monte_carlo_data.csv")
