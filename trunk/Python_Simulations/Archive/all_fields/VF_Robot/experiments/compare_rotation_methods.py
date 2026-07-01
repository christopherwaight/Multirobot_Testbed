"""
Compare Newton's Method: WITH rotation vs WITHOUT rotation

Starting position: (0.5, -2.0), up to 1000 steps
Field: Bimodal Gaussian saddle (two peaks create saddle at origin)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.fields.scalar_utils import compute_gradient, compute_hessian, compute_newton_step
from src.fields.scalar_environments.Saddle import bimodal_saddle1 as bimodal_saddle


def run_newton_no_rotation(start_pos, num_steps=1000, alpha=0.7, k=1.0, dt=0.1):
    """Newton WITHOUT rotation - standard approach."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle, pos[0], pos[1])
        hess = compute_hessian(bimodal_saddle, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        # Translation only
        v_cmd = k * delta_p
        velocity = alpha * velocity + (1 - alpha) * v_cmd
        new_pos = pos + dt * velocity
        trajectory.append(new_pos)

        if np.linalg.norm(grad) < 1e-6:
            break

    return np.array(trajectory)


def run_newton_with_rotation(start_pos, num_steps=1000, alpha=0.7, k=1.0, dt=0.1, k_r=0.3):
    """Newton WITH rotation - aligns formation with Newton direction."""
    trajectory = [start_pos.copy()]
    velocity = np.array([0.0, 0.0])
    theta_c = 0.0  # Formation orientation
    omega = 0.0    # Angular velocity

    for _ in range(num_steps):
        pos = trajectory[-1]
        grad = compute_gradient(bimodal_saddle, pos[0], pos[1])
        hess = compute_hessian(bimodal_saddle, pos[0], pos[1])
        delta_p = compute_newton_step(grad, hess)

        # Rotation control - align with Newton direction
        if np.linalg.norm(delta_p) > 1e-6:
            theta_newton = np.arctan2(delta_p[1], delta_p[0])

            # Find closest symmetric angle (square has 4-fold symmetry)
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

        if np.linalg.norm(grad) < 1e-6:
            break

    return np.array(trajectory)


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

start_pos = np.array([0.5, -2.0])

print("Running Newton WITHOUT rotation...")
traj_no_rot = run_newton_no_rotation(start_pos)

print("Running Newton WITH rotation...")
traj_with_rot = run_newton_with_rotation(start_pos)

# Compute metrics
final_dist_no_rot = np.linalg.norm(traj_no_rot[-1])
final_dist_with_rot = np.linalg.norm(traj_with_rot[-1])

path_length_no_rot = np.sum(np.linalg.norm(np.diff(traj_no_rot, axis=0), axis=1))
path_length_with_rot = np.sum(np.linalg.norm(np.diff(traj_with_rot, axis=0), axis=1))

straight_line_dist = np.linalg.norm(start_pos)
efficiency_no_rot = straight_line_dist / path_length_no_rot
efficiency_with_rot = straight_line_dist / path_length_with_rot

improvement = (len(traj_no_rot) - len(traj_with_rot)) / len(traj_no_rot) * 100

# Print results
print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"\nNO Rotation:")
print(f"  Final distance: {final_dist_no_rot*1000:.2f} mm")
print(f"  Steps: {len(traj_no_rot)}")
print(f"  Path length: {path_length_no_rot:.3f} m")
print(f"  Efficiency: {efficiency_no_rot:.3f}")

print(f"\nWITH Rotation:")
print(f"  Final distance: {final_dist_with_rot*1000:.2f} mm")
print(f"  Steps: {len(traj_with_rot)}")
print(f"  Path length: {path_length_with_rot:.3f} m")
print(f"  Efficiency: {efficiency_with_rot:.3f}")

print(f"\nImprovement: {improvement:.1f}% faster with rotation")

# ============================================================================
# PLOT
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Trajectory comparison
ax1.plot(traj_no_rot[:, 0], traj_no_rot[:, 1], 'b-', label='NO rotation', linewidth=2)
ax1.plot(traj_with_rot[:, 0], traj_with_rot[:, 1], 'r-', label='WITH rotation', linewidth=2)
ax1.plot(0, 0, 'k*', markersize=15, label='Saddle')
ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Trajectory Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Convergence over time
dist_no_rot = np.linalg.norm(traj_no_rot, axis=1)
dist_with_rot = np.linalg.norm(traj_with_rot, axis=1)
ax2.semilogy(dist_no_rot, 'b-', label='NO rotation')
ax2.semilogy(dist_with_rot, 'r-', label='WITH rotation')
ax2.set_xlabel('Step')
ax2.set_ylabel('Distance to saddle (m)')
ax2.set_title('Convergence Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rotation_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved: rotation_comparison.png")
plt.show()
