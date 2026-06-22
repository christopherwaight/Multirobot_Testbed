import numpy as np
import sys
sys.path.insert(0, 'src')
from control.kinematics import forward_kinematics, inverse_kinematics, compute_inverse_jacobian

print("=== Testing q direction ===\n")

# Start with desired formation
x_c, y_c = 0.0, 0.0
theta_c = 0.0
p_desired = 0.33
q_desired = 0.33
beta_desired = np.radians(60)

# Create initial formation
x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c, p_desired, beta_desired, q_desired)
print(f"Initial positions from inverse kinematics (p={p_desired}, q={q_desired}, beta=60°):")
print(f"  Robot 1: ({x1:.3f}, {y1:.3f})")
print(f"  Robot 2: ({x2:.3f}, {y2:.3f})")
print(f"  Robot 3: ({x3:.3f}, {y3:.3f})")

# Check with forward kinematics
formation = forward_kinematics(x1, y1, x2, y2, x3, y3)
print(f"\nForward kinematics measurement:")
print(f"  p={formation['p']:.3f}, q={formation['q']:.3f}, beta={np.degrees(formation['beta']):.1f}°")

# Now compute Jacobian
J_inv = compute_inverse_jacobian(p_desired, beta_desired, q_desired, theta_c)

# Test: if vq = +0.1 (want to increase q), how do robots move?
shape_vel = np.array([0, 0, 0, 0, 0, 0.1])  # Only vq = 0.1
robot_vel = J_inv @ shape_vel
print(f"\nIf vq = +0.1 (increase q), robot velocities:")
print(f"  Robot 1: ({robot_vel[0]:.4f}, {robot_vel[1]:.4f})")
print(f"  Robot 2: ({robot_vel[2]:.4f}, {robot_vel[3]:.4f})")
print(f"  Robot 3: ({robot_vel[4]:.4f}, {robot_vel[5]:.4f})")

# Robot 3 should move AWAY from Robot 2
dist_23_before = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
x3_new = x3 + robot_vel[4] * 0.1  # move for 0.1s
y3_new = y3 + robot_vel[5] * 0.1
dist_23_after = np.sqrt((x3_new - x2)**2 + (y3_new - y2)**2)
print(f"\nDistance robot 2 to 3:")
print(f"  Before: {dist_23_before:.4f}")
print(f"  After:  {dist_23_after:.4f}")
print(f"  Change: {dist_23_after - dist_23_before:.4f} (should be POSITIVE)")

if dist_23_after > dist_23_before:
    print("\n✓ Jacobian has CORRECT sign for q")
else:
    print("\n✗ Jacobian has WRONG sign for q - THIS IS THE BUG!")
