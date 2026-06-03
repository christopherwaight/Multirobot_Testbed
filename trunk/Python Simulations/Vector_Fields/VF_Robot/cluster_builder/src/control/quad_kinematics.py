"""
Kinematics functions for 4-robot quadrilateral formation.

Uses diagonal parameterization:
- d1, d2: Diagonal lengths (robot 1-3, robot 2-4)
- r1, r2: Intersection ratios (how far along each diagonal they cross)
- psi: Angle between diagonals (cross-diagonal shape variable; phi is reserved for per-robot heading)
- x_c, y_c: Centroid position
- th_c: Overall orientation (angle of diagonal 1-3)
"""

import numpy as np


def forward_kinematics_quad(x1, y1, x2, y2, x3, y3, x4, y4, prev_theta_c=None):
    """
    Convert robot positions to diagonal parameters.

    Based on MATLAB Forward_Kinematics function.

    Args:
        x1, y1: Position of robot 1
        x2, y2: Position of robot 2
        x3, y3: Position of robot 3
        x4, y4: Position of robot 4
        prev_theta_c: Previous theta_c value for angle unwrapping (optional)

    Returns:
        dict with keys: {x_c, y_c, th_c, d1, d2, r1, r2, psi}
    """
    # Centroid
    x_c = (x1 + x2 + x3 + x4) / 4.0
    y_c = (y1 + y2 + y3 + y4) / 4.0

    # Diagonal lengths
    d1 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)  # Diagonal 1-3
    d2 = np.sqrt((x4 - x2)**2 + (y4 - y2)**2)  # Diagonal 2-4

    # Overall rotation angle (angle of diagonal 1-3)
    th_c_raw = np.arctan2(y3 - y1, x3 - x1)

    # Unwrap angle if previous value provided
    if prev_theta_c is not None:
        while (th_c_raw - prev_theta_c) > np.pi:
            th_c_raw -= 2 * np.pi
        while (th_c_raw - prev_theta_c) < -np.pi:
            th_c_raw += 2 * np.pi

    th_c = th_c_raw

    # Angle between diagonals (psi = cross-diagonal shape variable)
    psi = -np.arctan2(y4 - y2, x4 - x2) + np.arctan2(y3 - y1, x3 - x1)
    psi = np.mod(psi + np.pi, 2*np.pi) - np.pi  # Wrap to [-pi, pi]

    # Intersection ratios using line intersection method
    delta = (x3 - x1) * (y2 - y4) + (y3 - y1) * (x4 - x2)

    # Avoid division by zero
    if np.abs(delta) < 1e-10:
        # Diagonals are parallel - degenerate case
        r1 = 0.5
        r2 = 0.5
    else:
        r1 = ((x2 - x1) * (y2 - y4) + (y2 - y1) * (x4 - x2)) / delta
        r2 = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / delta

    return {
        'x_c': x_c,
        'y_c': y_c,
        'th_c': th_c,
        'd1': d1,
        'd2': d2,
        'r1': r1,
        'r2': r2,
        'psi': psi
    }


def inverse_kinematics_quad(x_c, y_c, th_c, d1, d2, r1, r2, psi):
    """
    Convert diagonal parameters to robot positions.

    Args:
        x_c, y_c: Centroid position
        th_c: Overall orientation (angle of diagonal 1-3)
        d1: Length of diagonal 1-3
        d2: Length of diagonal 2-4
        r1: Intersection ratio for diagonal 1 (0 to 1)
        r2: Intersection ratio for diagonal 2 (0 to 1)
        psi: Angle between diagonals (cross-diagonal shape variable)

    Returns:
        x1, y1, x2, y2, x3, y3, x4, y4 (robot positions)
    """
    # Angle of second diagonal
    th_d2 = th_c - psi

    # Step 1: Place intersection point at origin in local frame
    # Robot 1 is at distance r1*d1 backward along diagonal 1
    x1_local = -r1 * d1 * np.cos(th_c)
    y1_local = -r1 * d1 * np.sin(th_c)

    # Robot 3 is at distance (1-r1)*d1 forward along diagonal 1
    x3_local = (1 - r1) * d1 * np.cos(th_c)
    y3_local = (1 - r1) * d1 * np.sin(th_c)

    # Robot 2 is at distance r2*d2 backward along diagonal 2
    x2_local = -r2 * d2 * np.cos(th_d2)
    y2_local = -r2 * d2 * np.sin(th_d2)

    # Robot 4 is at distance (1-r2)*d2 forward along diagonal 2
    x4_local = (1 - r2) * d2 * np.cos(th_d2)
    y4_local = (1 - r2) * d2 * np.sin(th_d2)

    # Step 2: Compute centroid of local frame positions
    x_c_local = (x1_local + x2_local + x3_local + x4_local) / 4.0
    y_c_local = (y1_local + y2_local + y3_local + y4_local) / 4.0

    # Step 3: Shift to global frame (translate so centroid is at x_c, y_c)
    x1 = x1_local - x_c_local + x_c
    y1 = y1_local - y_c_local + y_c

    x2 = x2_local - x_c_local + x_c
    y2 = y2_local - y_c_local + y_c

    x3 = x3_local - x_c_local + x_c
    y3 = y3_local - y_c_local + y_c

    x4 = x4_local - x_c_local + x_c
    y4 = y4_local - y_c_local + y_c

    return x1, y1, x2, y2, x3, y3, x4, y4


def compute_inverse_jacobian_quad(d1, d2, r1, r2, psi, th_c, epsilon=1e-6):
    """
    Compute inverse Jacobian for 4-robot formation using numerical differentiation.

    The Jacobian maps shape velocities to robot velocities:
    [vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4]^T = J_inv * [vx_c, vy_c, omega_c, vd1, vd2, vr1, vr2, vpsi]^T

    Args:
        d1, d2: Diagonal lengths
        r1, r2: Intersection ratios
        psi: Angle between diagonals (cross-diagonal shape variable)
        th_c: Overall orientation
        epsilon: Step size for numerical differentiation

    Returns:
        J_inv: 8x8 numpy array (inverse Jacobian)
    """
    # Nominal configuration (centroid at origin for simplicity)
    x_c_nom, y_c_nom = 0.0, 0.0

    # Get nominal robot positions
    x1_0, y1_0, x2_0, y2_0, x3_0, y3_0, x4_0, y4_0 = inverse_kinematics_quad(
        x_c_nom, y_c_nom, th_c, d1, d2, r1, r2, psi
    )

    robot_pos_0 = np.array([x1_0, y1_0, x2_0, y2_0, x3_0, y3_0, x4_0, y4_0])

    # Jacobian will be 8x8
    J_inv = np.zeros((8, 8))

    # Perturb each shape variable and compute change in robot positions
    shape_vars = [
        (x_c_nom, y_c_nom, th_c, d1, d2, r1, r2, psi),  # Nominal
    ]

    # Index: 0=x_c, 1=y_c, 2=th_c, 3=d1, 4=d2, 5=r1, 6=r2, 7=psi
    for i in range(8):
        # Create perturbed configuration
        x_c_p, y_c_p, th_c_p, d1_p, d2_p, r1_p, r2_p, psi_p = shape_vars[0]

        if i == 0:    # Perturb x_c
            x_c_p += epsilon
        elif i == 1:  # Perturb y_c
            y_c_p += epsilon
        elif i == 2:  # Perturb th_c
            th_c_p += epsilon
        elif i == 3:  # Perturb d1
            d1_p += epsilon
        elif i == 4:  # Perturb d2
            d2_p += epsilon
        elif i == 5:  # Perturb r1
            r1_p += epsilon
        elif i == 6:  # Perturb r2
            r2_p += epsilon
        elif i == 7:  # Perturb psi
            psi_p += epsilon

        # Compute perturbed robot positions
        x1_p, y1_p, x2_p, y2_p, x3_p, y3_p, x4_p, y4_p = inverse_kinematics_quad(
            x_c_p, y_c_p, th_c_p, d1_p, d2_p, r1_p, r2_p, psi_p
        )

        robot_pos_p = np.array([x1_p, y1_p, x2_p, y2_p, x3_p, y3_p, x4_p, y4_p])

        # Numerical derivative
        J_inv[:, i] = (robot_pos_p - robot_pos_0) / epsilon

    return J_inv


def shape_velocities_to_robot_velocities_quad(J_inv, vx_c, vy_c, omega_c=0.0,
                                               vd1=0.0, vd2=0.0, vr1=0.0, vr2=0.0, vpsi=0.0):
    """
    Convert shape velocities to robot velocities using inverse Jacobian.

    For formation maintenance, the shape size/angle velocities are zero.

    Args:
        J_inv: 8x8 inverse Jacobian matrix
        vx_c, vy_c: Centroid velocities
        omega_c: Angular velocity (default 0)
        vd1, vd2: Diagonal length velocities (default 0)
        vr1, vr2: Intersection ratio velocities (default 0)
        vpsi: Cross-diagonal angle velocity (default 0)

    Returns:
        vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4 (robot velocities)
    """
    # Shape velocity vector
    shape_vel = np.array([vx_c, vy_c, omega_c, vd1, vd2, vr1, vr2, vpsi])

    # Robot velocities = J_inv * shape_vel
    robot_vel = J_inv @ shape_vel

    vx1, vy1 = robot_vel[0], robot_vel[1]
    vx2, vy2 = robot_vel[2], robot_vel[3]
    vx3, vy3 = robot_vel[4], robot_vel[5]
    vx4, vy4 = robot_vel[6], robot_vel[7]

    return vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4
