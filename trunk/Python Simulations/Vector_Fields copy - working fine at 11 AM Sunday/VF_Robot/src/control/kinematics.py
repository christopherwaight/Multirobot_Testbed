"""
Kinematics for 3-robot triangular formation using SAS (Side-Angle-Side) parameterization.

Shape parameters: (p, q, beta)
- p: distance from robot 1 to robot 2
- q: distance from robot 2 to robot 3
- beta: angle at robot 2

Centroid parameters: (x_c, y_c, theta_c)
- x_c, y_c: centroid position
- theta_c: orientation angle
"""
import numpy as np


def forward_kinematics(x1, y1, x2, y2, x3, y3):
    """
    Compute SAS parameters and centroid from robot positions.

    Args:
        x1, y1: Position of robot 1
        x2, y2: Position of robot 2
        x3, y3: Position of robot 3

    Returns:
        dict with keys:
            - 'x_c', 'y_c': Centroid position
            - 'theta_c': Orientation angle
            - 'p': Side length (robot 1 to robot 2)
            - 'q': Side length (robot 2 to robot 3)
            - 'r': Side length (robot 3 to robot 1)
            - 'beta': Angle at robot 2
            - 'alpha': Angle at robot 1
            - 'gamma': Angle at robot 3
    """
    # Compute side lengths
    p = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    q = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    r = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)

    # Compute angles using law of cosines
    # Avoid division by zero
    eps = 1e-10

    # Angle at robot 2 (beta)
    cos_beta = (p**2 + q**2 - r**2) / (2 * p * q + eps)
    cos_beta = np.clip(cos_beta, -1, 1)  # Numerical safety
    beta = np.arccos(cos_beta)

    # Angle at robot 1 (alpha)
    cos_alpha = (p**2 + r**2 - q**2) / (2 * p * r + eps)
    cos_alpha = np.clip(cos_alpha, -1, 1)
    alpha = np.arccos(cos_alpha)

    # Angle at robot 3 (gamma)
    cos_gamma = (q**2 + r**2 - p**2) / (2 * q * r + eps)
    cos_gamma = np.clip(cos_gamma, -1, 1)
    gamma = np.arccos(cos_gamma)

    # Compute centroid
    x_c = (x1 + x2 + x3) / 3.0
    y_c = (y1 + y2 + y3) / 3.0

    # Compute orientation (angle from robot 1 to robot 2)
    theta_c = np.arctan2(y2 - y1, x2 - x1)

    return {
        'x_c': x_c,
        'y_c': y_c,
        'theta_c': theta_c,
        'p': p,
        'q': q,
        'r': r,
        'beta': beta,
        'alpha': alpha,
        'gamma': gamma
    }


def inverse_kinematics(x_c, y_c, theta_c, p, beta, q):
    """
    Compute robot positions from SAS parameters and centroid.

    Args:
        x_c, y_c: Centroid position
        theta_c: Orientation angle
        p: Side length (robot 1 to robot 2)
        beta: Angle at robot 2
        q: Side length (robot 2 to robot 3)

    Returns:
        Robot positions: (x1, y1, x2, y2, x3, y3)
    """
    # Simple and direct approach:
    # Place robot 2 at origin in local frame
    # Place robot 1 at distance p along +x axis
    # Place robot 3 at angle beta from +x axis, distance q from robot 2

    # Local frame positions (robot 2 at origin)
    x2_local = 0.0
    y2_local = 0.0

    x1_local = p
    y1_local = 0.0

    x3_local = q * np.cos(beta)
    y3_local = q * np.sin(beta)

    # Compute local centroid
    x_c_local = (x1_local + x2_local + x3_local) / 3.0
    y_c_local = (y1_local + y2_local + y3_local) / 3.0

    # Shift to center the triangle at local origin
    x1_local -= x_c_local
    y1_local -= y_c_local
    x2_local -= x_c_local
    y2_local -= y_c_local
    x3_local -= x_c_local
    y3_local -= y_c_local

    # Rotation matrix
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)

    # Apply rotation and translation to get global positions
    x1 = cos_theta * x1_local - sin_theta * y1_local + x_c
    y1 = sin_theta * x1_local + cos_theta * y1_local + y_c

    x2 = cos_theta * x2_local - sin_theta * y2_local + x_c
    y2 = sin_theta * x2_local + cos_theta * y2_local + y_c

    x3 = cos_theta * x3_local - sin_theta * y3_local + x_c
    y3 = sin_theta * x3_local + cos_theta * y3_local + y_c

    return x1, y1, x2, y2, x3, y3


def compute_inverse_jacobian(p, beta, q, theta_c, numerical_epsilon=1e-6):
    """
    Compute inverse Jacobian: maps shape parameter velocities to robot velocities.

    J^{-1} maps [delta_x_c, delta_y_c, delta_theta_c, delta_p, delta_beta, delta_q]
    to [delta_x1, delta_y1, delta_x2, delta_y2, delta_x3, delta_y3]

    Args:
        p: Current p value
        beta: Current beta value (in radians)
        q: Current q value
        theta_c: Current orientation angle
        numerical_epsilon: Step size for numerical differentiation

    Returns:
        6x6 inverse Jacobian matrix
    """
    # We'll compute this numerically by perturbing each shape parameter
    # and measuring the change in robot positions

    # Get baseline positions
    x_c, y_c = 0.0, 0.0  # Use origin for simplicity
    x1_0, y1_0, x2_0, y2_0, x3_0, y3_0 = inverse_kinematics(x_c, y_c, theta_c, p, beta, q)
    baseline = np.array([x1_0, y1_0, x2_0, y2_0, x3_0, y3_0])

    J_inv = np.zeros((6, 6))

    # Perturb x_c
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c + numerical_epsilon, y_c, theta_c, p, beta, q)
    J_inv[:, 0] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    # Perturb y_c
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c + numerical_epsilon, theta_c, p, beta, q)
    J_inv[:, 1] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    # Perturb theta_c
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c + numerical_epsilon, p, beta, q)
    J_inv[:, 2] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    # Perturb p
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c, p + numerical_epsilon, beta, q)
    J_inv[:, 3] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    # Perturb beta
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c, p, beta + numerical_epsilon, q)
    J_inv[:, 4] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    # Perturb q
    x1, y1, x2, y2, x3, y3 = inverse_kinematics(x_c, y_c, theta_c, p, beta, q + numerical_epsilon)
    J_inv[:, 5] = (np.array([x1, y1, x2, y2, x3, y3]) - baseline) / numerical_epsilon

    return J_inv


def shape_velocities_to_robot_velocities(J_inv, vx_c, vy_c, omega_c, vp, vbeta, vq):
    """
    Convert shape parameter velocities to individual robot velocities.

    Args:
        J_inv: 6x6 inverse Jacobian matrix
        vx_c, vy_c: Centroid velocities
        omega_c: Angular velocity (rotation rate)
        vp, vbeta, vq: Shape parameter velocities

    Returns:
        (vx1, vy1, vx2, vy2, vx3, vy3): Individual robot velocities
    """
    shape_vel = np.array([vx_c, vy_c, omega_c, vp, vbeta, vq])
    robot_vel = J_inv @ shape_vel
    return robot_vel[0], robot_vel[1], robot_vel[2], robot_vel[3], robot_vel[4], robot_vel[5]
