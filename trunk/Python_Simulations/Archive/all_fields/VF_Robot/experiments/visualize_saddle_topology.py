"""
Paper Figure: Saddle Point Topology and Newton's Method

Generates publication-ready 2×2 figure showing:
(a) Scalar field with saddle at origin
(b) Gradient field showing instability
(c) Eigenstructure (stable/unstable manifolds)
(d) Newton convergence from multiple starts
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, compute_hessian, compute_newton_step, analyze_critical_point
from src.fields.scalar_environments.Saddle import bimodal_saddle1


def run_newton_trajectory(start_pos, num_steps=500, alpha=0.7, k=1.0, dt=0.1):
    """Run Newton's method from starting position."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle1, pos[0], pos[1])
        hess = compute_hessian(bimodal_saddle1, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        v_cmd = k * delta_p
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        pos = pos + dt * velocity
        trajectory.append(pos)

        if np.linalg.norm(grad) < 1e-6 or np.linalg.norm(pos) < 0.01:
            break

    return np.array(trajectory)


# Grid for field evaluation
x = np.linspace(-3, 3, 150)
y = np.linspace(-3, 3, 150)
X, Y = np.meshgrid(x, y)

# Evaluate field
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = bimodal_saddle1(X[i, j], Y[i, j])

# Gradient field (coarser grid)
skip = 8
X_grad = X[::skip, ::skip]
Y_grad = Y[::skip, ::skip]
U_grad = np.zeros_like(X_grad)
V_grad = np.zeros_like(Y_grad)

for i in range(X_grad.shape[0]):
    for j in range(X_grad.shape[1]):
        grad = compute_gradient(bimodal_saddle1, X_grad[i, j], Y_grad[i, j])
        U_grad[i, j] = grad[0]
        V_grad[i, j] = grad[1]

# Analyze saddle
analysis = analyze_critical_point(bimodal_saddle1, 0, 0)
eigenvals = analysis['eigenvalues']
eigenvecs = analysis['eigenvectors']

# Generate Newton trajectories
np.random.seed(42)
num_trajectories = 12
radius = 2.0
angles = np.linspace(0, 2*np.pi, num_trajectories, endpoint=False)
starting_positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
trajectories = [run_newton_trajectory(pos) for pos in starting_positions]

# Create figure
fig = plt.figure(figsize=(14, 12))

# (a) Scalar Field Contours
ax1 = plt.subplot(221)
contourf = ax1.contourf(X, Y, Z, levels=25, cmap='RdBu_r', alpha=0.8)
contours = ax1.contour(X, Y, Z, levels=12, colors='black', alpha=0.3, linewidths=0.8)
ax1.plot(0, 0, 'k*', markersize=20, label='Saddle Point', zorder=5)
ax1.set_xlabel('x (m)', fontsize=11)
ax1.set_ylabel('y (m)', fontsize=11)
ax1.set_title('(a) Bimodal Gaussian Saddle Field', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.legend(fontsize=10)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
cbar1 = plt.colorbar(contourf, ax=ax1, label='z(x,y)')

# (b) Gradient Field
ax2 = plt.subplot(222)
ax2.contourf(X, Y, Z, levels=25, cmap='RdBu_r', alpha=0.2)
quiv = ax2.quiver(X_grad, Y_grad, U_grad, V_grad, alpha=0.7, color='blue', scale=12, width=0.003)
ax2.plot(0, 0, 'k*', markersize=20, label='Saddle Point', zorder=5)
ax2.set_xlabel('x (m)', fontsize=11)
ax2.set_ylabel('y (m)', fontsize=11)
ax2.set_title('(b) Gradient Field ∇z (Unstable)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
ax2.legend(fontsize=10)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)

# (c) Eigenstructure
ax3 = plt.subplot(223)
ax3.contourf(X, Y, Z, levels=25, cmap='RdBu_r', alpha=0.2)
ax3.plot(0, 0, 'k*', markersize=20, label='Saddle Point', zorder=5)

# Draw eigenvector directions (stable/unstable manifolds)
scale = 2.5
# Stable manifold (negative eigenvalue)
ax3.arrow(0, 0, eigenvecs[0, 0]*scale, eigenvecs[1, 0]*scale,
         head_width=0.15, head_length=0.2, fc='green', ec='green',
         alpha=0.8, linewidth=3, zorder=4)
ax3.arrow(0, 0, -eigenvecs[0, 0]*scale, -eigenvecs[1, 0]*scale,
         head_width=0.15, head_length=0.2, fc='green', ec='green',
         alpha=0.8, linewidth=3, zorder=4, label=f'Stable (λ={eigenvals[0]:.2f})')

# Unstable manifold (positive eigenvalue)
ax3.arrow(0, 0, eigenvecs[0, 1]*scale, eigenvecs[1, 1]*scale,
         head_width=0.15, head_length=0.2, fc='red', ec='red',
         alpha=0.8, linewidth=3, zorder=4)
ax3.arrow(0, 0, -eigenvecs[0, 1]*scale, -eigenvecs[1, 1]*scale,
         head_width=0.15, head_length=0.2, fc='red', ec='red',
         alpha=0.8, linewidth=3, zorder=4, label=f'Unstable (λ={eigenvals[1]:.2f})')

ax3.set_xlabel('x (m)', fontsize=11)
ax3.set_ylabel('y (m)', fontsize=11)
ax3.set_title('(c) Hessian Eigenstructure', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')
ax3.legend(fontsize=10, loc='upper right')
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)

# (d) Newton Convergence
ax4 = plt.subplot(224)
ax4.contourf(X, Y, Z, levels=25, cmap='RdBu_r', alpha=0.2)

# Plot all trajectories
colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
for traj, color in zip(trajectories, colors):
    ax4.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.7, linewidth=2)
    ax4.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8, alpha=0.8)

ax4.plot(0, 0, 'k*', markersize=20, label='Saddle Point', zorder=5)
ax4.set_xlabel('x (m)', fontsize=11)
ax4.set_ylabel('y (m)', fontsize=11)
ax4.set_title('(d) Newton Convergence (12 starts)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')
ax4.legend(fontsize=10)
ax4.set_xlim(-3, 3)
ax4.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('saddle_topology_paper_figure.png', dpi=300, bbox_inches='tight')
print("Saved: saddle_topology_paper_figure.png")

# Print analysis
print("\n" + "="*60)
print("SADDLE POINT TOPOLOGY ANALYSIS")
print("="*60)
print(f"Field: Bimodal Gaussian (two peaks at x=±2)")
print(f"Saddle location: (0, 0)")
print(f"Classification: {analysis['classification'].upper()}")
print(f"Hessian eigenvalues: λ₁={eigenvals[0]:.4f} (stable), λ₂={eigenvals[1]:.4f} (unstable)")
print(f"Determinant(H): {eigenvals[0]*eigenvals[1]:.4f} < 0 → saddle")
print(f"\nStable manifold direction: ({eigenvecs[0,0]:.3f}, {eigenvecs[1,0]:.3f})")
print(f"Unstable manifold direction: ({eigenvecs[0,1]:.3f}, {eigenvecs[1,1]:.3f})")
print("="*60)

# Convergence statistics
final_dists = [np.linalg.norm(traj[-1]) for traj in trajectories]
print(f"\nNewton convergence from {num_trajectories} starting positions:")
print(f"  Mean final distance: {np.mean(final_dists)*1000:.2f} mm")
print(f"  Success rate (< 0.1m): {100*np.mean([d < 0.1 for d in final_dists]):.0f}%")
print("="*60)

plt.show()
