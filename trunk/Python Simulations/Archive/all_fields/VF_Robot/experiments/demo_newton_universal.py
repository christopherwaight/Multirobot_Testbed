"""
Demo: Newton's Method on ALL Critical Point Types

Demonstrates universal applicability:
- Minimum (both λ > 0)
- Maximum (both λ < 0)
- Saddle (opposite sign λ)

Same Δp = -H⁻¹∇z formula works for all three types.
Compares with gradient-based methods to show where they fail.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, compute_hessian, compute_newton_step, analyze_critical_point
from src.fields.scalar_environments.Paraboloid import paraboloid_min1, paraboloid_max1
from src.fields.scalar_environments.Saddle import bimodal_saddle1


def run_newton(field_func, start_pos, num_steps=500, alpha=0.7, k=1.0, dt=0.1):
    """Newton's method - universal for all critical points."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(field_func, pos[0], pos[1])
        hess = compute_hessian(field_func, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        v_cmd = k * delta_p
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        pos = pos + dt * velocity
        trajectory.append(pos)

        if np.linalg.norm(grad) < 1e-6 or np.linalg.norm(pos) < 0.01:
            break

    return np.array(trajectory)


def run_gradient_descent(field_func, start_pos, num_steps=500, alpha=0.7, k=1.0, dt=0.1):
    """Gradient descent - works on minima, fails on saddles."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(field_func, pos[0], pos[1])

        v_cmd = -k * grad  # Negative gradient
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        pos = pos + dt * velocity
        trajectory.append(pos)

        if np.linalg.norm(grad) < 1e-6 or np.linalg.norm(pos) > 5.0:
            break

    return np.array(trajectory)


def run_gradient_ascent(field_func, start_pos, num_steps=500, alpha=0.7, k=1.0, dt=0.1):
    """Gradient ascent - works on maxima."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(field_func, pos[0], pos[1])

        v_cmd = k * grad  # Positive gradient (climb hill)
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        pos = pos + dt * velocity
        trajectory.append(pos)

        if np.linalg.norm(grad) < 1e-6 or np.linalg.norm(pos) > 5.0:
            break

    return np.array(trajectory)


# Test configurations
fields = [
    (paraboloid_min1, "Minimum (Bowl)", (0, 0), run_gradient_descent, "Gradient Descent"),
    (paraboloid_max1, "Maximum (Peak)", (0, 0), run_gradient_ascent, "Gradient Ascent"),
    (bimodal_saddle1, "Saddle (Pass)", (0, 0), run_gradient_descent, "Gradient Descent (FAILS)")
]

# Starting position for all tests
start_pos = np.array([2.0, 1.5])

# Grid for field visualization
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Create figure
fig = plt.figure(figsize=(18, 11))

print("="*70)
print("NEWTON'S METHOD: UNIVERSAL CRITICAL POINT FINDER")
print("="*70)

for idx, (field_func, title, critical_pt, grad_method, grad_name) in enumerate(fields):
    # Evaluate field
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = field_func(X[i, j], Y[i, j])

    # Run both methods
    traj_newton = run_newton(field_func, start_pos)
    traj_grad = grad_method(field_func, start_pos)

    # Analyze critical point
    analysis = analyze_critical_point(field_func, critical_pt[0], critical_pt[1])
    eigenvals = analysis['eigenvalues']
    classification = analysis['classification']

    # Row 1: Trajectories
    ax1 = plt.subplot(3, 3, idx + 1)
    ax1.contourf(X, Y, Z, levels=25, cmap='RdBu_r', alpha=0.4)
    ax1.contour(X, Y, Z, levels=15, colors='black', alpha=0.2, linewidths=0.5)
    ax1.plot(traj_newton[:, 0], traj_newton[:, 1], 'r-', linewidth=3, alpha=0.8, label='Newton')
    ax1.plot(traj_grad[:, 0], traj_grad[:, 1], 'b--', linewidth=2, alpha=0.7, label=grad_name)
    ax1.plot(critical_pt[0], critical_pt[1], 'k*', markersize=20, label='Critical Point')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax1.set_xlabel('x (m)', fontsize=10)
    ax1.set_ylabel('y (m)', fontsize=10)
    ax1.set_title(f'{title}\n{classification.upper()}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)

    # Row 2: Convergence
    ax2 = plt.subplot(3, 3, idx + 4)
    dist_newton = np.linalg.norm(traj_newton - critical_pt, axis=1)
    dist_grad = np.linalg.norm(traj_grad - critical_pt, axis=1)
    ax2.semilogy(dist_newton, 'r-', linewidth=2, label=f'Newton ({len(traj_newton)} steps)')
    ax2.semilogy(dist_grad, 'b--', linewidth=2, label=f'{grad_name} ({len(traj_grad)} steps)')
    ax2.axhline(0.01, color='green', linestyle=':', linewidth=1, label='1 cm threshold')
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('Distance to Critical Point (m)', fontsize=10)
    ax2.set_title('Convergence Comparison', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Row 3: Analysis
    ax3 = plt.subplot(3, 3, idx + 7)
    ax3.axis('off')

    # Success metrics
    newton_final = np.linalg.norm(traj_newton[-1] - critical_pt)
    grad_final = np.linalg.norm(traj_grad[-1] - critical_pt)
    newton_success = newton_final < 0.1
    grad_success = grad_final < 0.1

    info_text = (
        f"Critical Point Type: {classification.upper()}\n"
        f"Eigenvalues: λ₁={eigenvals[0]:.3f}, λ₂={eigenvals[1]:.3f}\n"
        f"Det(H) = {eigenvals[0]*eigenvals[1]:.3f}\n\n"
        f"Newton's Method:\n"
        f"  Steps: {len(traj_newton)}\n"
        f"  Final dist: {newton_final*1000:.2f} mm\n"
        f"  Success: {'✓' if newton_success else '✗'}\n\n"
        f"{grad_name}:\n"
        f"  Steps: {len(traj_grad)}\n"
        f"  Final dist: {grad_final*1000:.2f} mm\n"
        f"  Success: {'✓' if grad_success else '✗'}\n\n"
        f"Speedup: {len(traj_grad)/len(traj_newton):.1f}×"
    )

    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Print console summary
    print(f"\n{title} ({classification.upper()}):")
    print(f"  Eigenvalues: λ₁={eigenvals[0]:.3f}, λ₂={eigenvals[1]:.3f}")
    print(f"  Newton: {len(traj_newton):3d} steps → {newton_final*1000:6.2f} mm ({'SUCCESS' if newton_success else 'FAIL'})")
    print(f"  {grad_name}: {len(traj_grad):3d} steps → {grad_final*1000:6.2f} mm ({'SUCCESS' if grad_success else 'FAIL'})")
    if newton_success and grad_success:
        print(f"  Speedup: {len(traj_grad)/len(traj_newton):.1f}× faster with Newton")

plt.tight_layout()
plt.savefig('demo_newton_universal.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print("Plot saved: demo_newton_universal.png")
print("="*70)
print("\nKEY INSIGHT:")
print("  Same formula (Δp = -H⁻¹∇z) works on ALL critical point types")
print("  Gradient methods work on stable equilibria (min/max)")
print("  Only Newton works on unstable equilibria (saddles)")
print("="*70)

plt.show()
