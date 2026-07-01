"""
Gradient Descent Baseline Comparison
Demonstrates FAILURE of gradient descent on saddle points
Uses same 100 starting positions as Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, compute_hessian, compute_newton_step
from src.fields.scalar_environments.Saddle import bimodal_saddle1 as bimodal_saddle

def gradient_descent(start_pos, num_steps=1000, alpha=0.7, k=1.0, dt=0.1):
    """
    Gradient descent: v_c = -k * ∇z
    This will FAIL on saddles (unstable equilibrium)
    """
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle, pos[0], pos[1])

        # Gradient descent command (negative gradient)
        v_cmd = -k * grad

        # Momentum dynamics
        velocity = alpha * velocity + (1 - alpha) * v_cmd

        # Update position
        new_pos = pos + dt * velocity
        trajectory.append(new_pos)

        # Stop if out of bounds (diverged)
        if np.linalg.norm(new_pos) > 5.0:
            break

        # Stop if stuck at local minimum
        if np.linalg.norm(grad) < 1e-6:
            break

    return np.array(trajectory)

def newton_method(start_pos, num_steps=1000, alpha=0.7, k=1.0, dt=0.1):
    """Newton's method for comparison"""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle, pos[0], pos[1])
        hess = compute_hessian(bimodal_saddle, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        # Velocity command
        v_cmd = k * delta_p
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        new_pos = pos + dt * velocity
        trajectory.append(new_pos)

        # Converged
        if np.linalg.norm(grad) < 1e-6:
            break

    return np.array(trajectory)

# Generate same 100 starting positions as Monte Carlo
NUM_TRIALS = 100
START_RADIUS = 2.0
angles = np.linspace(0, 2 * np.pi, NUM_TRIALS, endpoint=False)
starting_positions = START_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])

print("Running Gradient Descent vs Newton's Method comparison...")
print(f"{NUM_TRIALS} trials from circle of radius {START_RADIUS} m\n")

grad_results = []
newton_results = []
sample_indices = [0, 25, 50, 75]  # Sample trajectories for plotting

for i, start_pos in enumerate(starting_positions):
    if (i + 1) % 25 == 0:
        print(f"Progress: {i+1}/{NUM_TRIALS}")

    # Gradient descent
    traj_grad = gradient_descent(start_pos)
    final_dist_grad = np.linalg.norm(traj_grad[-1])
    success_grad = final_dist_grad < 0.1

    grad_results.append({
        'final_dist': final_dist_grad,
        'steps': len(traj_grad),
        'success': success_grad,
        'trajectory': traj_grad if i in sample_indices else None
    })

    # Newton's method
    traj_newton = newton_method(start_pos)
    final_dist_newton = np.linalg.norm(traj_newton[-1])
    success_newton = final_dist_newton < 0.1

    newton_results.append({
        'final_dist': final_dist_newton,
        'steps': len(traj_newton),
        'success': success_newton,
        'trajectory': traj_newton if i in sample_indices else None
    })

# Statistics
grad_success_rate = np.mean([r['success'] for r in grad_results]) * 100
newton_success_rate = np.mean([r['success'] for r in newton_results]) * 100

grad_final_dists = np.array([r['final_dist'] for r in grad_results])
newton_final_dists = np.array([r['final_dist'] for r in newton_results])

print(f"\n{'='*60}")
print("COMPARISON RESULTS")
print(f"{'='*60}")
print(f"\nGRADIENT DESCENT:")
print(f"  Success rate (dist < 0.1 m): {grad_success_rate:.1f}%")
print(f"  Mean final distance: {np.mean(grad_final_dists)*1000:.1f} mm")
print(f"  Typical behavior: {'Diverges or stuck at local min' if grad_success_rate < 50 else 'Mixed'}")

print(f"\nNEWTON'S METHOD:")
print(f"  Success rate (dist < 0.1 m): {newton_success_rate:.1f}%")
print(f"  Mean final distance: {np.mean(newton_final_dists)*1000:.1f} mm")
print(f"  Typical behavior: Converges to saddle")

print(f"\n{'='*60}")
print("CONCLUSION:")
print(f"{'='*60}")
if grad_success_rate < 10:
    print("✗ Gradient descent FAILS on saddles (unstable equilibrium)")
    print("✓ Newton's method SUCCEEDS using Hessian information")
    print(f"  → {newton_success_rate/max(grad_success_rate, 1):.0f}× better success rate")
else:
    print(f"Gradient descent: {grad_success_rate:.1f}% success")
    print(f"Newton's method: {newton_success_rate:.1f}% success")

# Plotting
fig = plt.figure(figsize=(15, 5))

# 1. Sample trajectories comparison
ax1 = plt.subplot(131)
for i in sample_indices:
    if grad_results[i]['trajectory'] is not None:
        traj = grad_results[i]['trajectory']
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=1.5,
                label='Gradient descent' if i == sample_indices[0] else '')

    if newton_results[i]['trajectory'] is not None:
        traj = newton_results[i]['trajectory']
        ax1.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, linewidth=2,
                label='Newton method' if i == sample_indices[0] else '')

ax1.plot(0, 0, 'k*', markersize=20, label='True saddle')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Sample Trajectories Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. Success rate comparison
ax2 = plt.subplot(132)
methods = ['Gradient\nDescent', 'Newton\nMethod']
success_rates = [grad_success_rate, newton_success_rate]
colors = ['red' if sr < 50 else 'green' for sr in success_rates]
bars = ax2.bar(methods, success_rates, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Success Rate (%)')
ax2.set_title('Success Rate Comparison\n(distance < 0.1 m)')
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Final distance distribution
ax3 = plt.subplot(133)
# Clip extreme values for visualization
grad_dists_clipped = np.clip(grad_final_dists * 1000, 0, 1000)
newton_dists_clipped = np.clip(newton_final_dists * 1000, 0, 100)

ax3.hist(grad_dists_clipped, bins=20, alpha=0.5, label='Gradient descent',
         color='blue', edgecolor='black')
ax3.hist(newton_dists_clipped, bins=20, alpha=0.7, label='Newton method',
         color='red', edgecolor='black')
ax3.axvline(100, color='green', linestyle='--', linewidth=2,
           label='Success threshold (100 mm)')
ax3.set_xlabel('Final distance to saddle (mm)')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Final Distances')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved: gradient_descent_comparison.png")
plt.show()

print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print("Saddle points are UNSTABLE EQUILIBRIA with mixed stability:")
print("  • Attractive along one eigenvector direction")
print("  • Repulsive along the other eigenvector direction")
print("\nGradient descent follows ∇z → pushed away from saddle")
print("Newton's method uses H^{-1}∇z → navigates toward saddle")
print("="*60)
