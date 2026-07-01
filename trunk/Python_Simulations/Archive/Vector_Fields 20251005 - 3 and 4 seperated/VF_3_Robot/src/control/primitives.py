"""
Control primitives for OmniCluster.

All primitives have signature: primitive(cluster) -> (vx_c, vy_c)
Returns desired centroid velocity.
"""
import numpy as np


def calculate_jacobian_from_readings(cluster):
    """
    Calculate Jacobian matrix from robot positions and field readings.

    Returns:
        (curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy)
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()

    # Get field readings at robot positions
    readings = cluster.sample_field_at_robots()
    u_readings = np.array([r[0] for r in readings])
    v_readings = np.array([r[1] for r in readings])

    # Calculate normal vector for u field
    pos1_u = np.array([x1, y1, u_readings[0]])
    pos2_u = np.array([x2, y2, u_readings[1]])
    pos3_u = np.array([x3, y3, u_readings[2]])

    R12_u = pos2_u - pos1_u
    R13_u = pos3_u - pos1_u
    N_u = np.cross(-R12_u, R13_u)
    du_dx = N_u[0]
    du_dy = N_u[1]

    # Calculate normal vector for v field
    pos1_v = np.array([x1, y1, v_readings[0]])
    pos2_v = np.array([x2, y2, v_readings[1]])
    pos3_v = np.array([x3, y3, v_readings[2]])

    R12_v = pos2_v - pos1_v
    R13_v = pos3_v - pos1_v
    N_v = np.cross(-R12_v, R13_v)
    dv_dx = N_v[0]
    dv_dy = N_v[1]

    # Calculate curl and divergence
    curl_z = dv_dx - du_dy
    divergence = du_dx + dv_dy

    return curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy


def vector_sum(cluster):
    """
    Simple vector sum primitive - move in direction of summed field vectors.

    Args:
        cluster: OmniCluster instance

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    readings = cluster.sample_field_at_robots()
    readings = np.array(readings)

    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])

    # Scale by step size
    vx_c = cluster.timestep * sum_u / cluster.timestep  # Simplifies to sum_u
    vy_c = cluster.timestep * sum_v / cluster.timestep

    return sum_u, sum_v


def critical_point_plane_fitting(cluster):
    """
    Find critical points using plane fitting method.

    Fits planes to u and v components and solves for where both are zero.

    Returns:
        (vx_c, vy_c): Velocity toward estimated critical point
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]

    # Matrix A for the system of equations
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])

    # RHS for u and v
    b_u = np.array([u1, u2, u3])
    b_v = np.array([v1, v2, v3])

    try:
        # Solve for plane coefficients
        abc = np.linalg.solve(A, b_u)
        def_ = np.linalg.solve(A, b_v)

        a, b, c = abc
        d, e, f = def_

        # Form the Jacobian matrix
        J = np.array([
            [a, b],
            [d, e]
        ])

        # Solve for critical point
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        # Get current centroid
        centroid = cluster.get_centroid()

        # Calculate vector to critical point
        vector_to_critical = critical_point - centroid
        distance = np.linalg.norm(vector_to_critical)

        if distance < 1e-6:
            return 0.0, 0.0

        # Limit maximum step
        max_step = 5.0
        if distance > max_step:
            vector_to_critical = (vector_to_critical / distance) * max_step
        else:
            vector_to_critical = 1.0 * (vector_to_critical / distance)

        return vector_to_critical[0], vector_to_critical[1]

    except np.linalg.LinAlgError:
        # Fall back to vector sum
        return vector_sum(cluster)


def critical_point_orbiter_plane_fitting(cluster, desired_radius=0.4):
    """
    Orbit around critical point estimated using plane fitting.

    Args:
        cluster: OmniCluster instance
        desired_radius: Desired orbital radius

    Returns:
        (vx_c, vy_c): Velocity for orbital motion
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]

    # Matrix A
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])

    b_u = np.array([u1, u2, u3])
    b_v = np.array([v1, v2, v3])

    try:
        # Solve for plane coefficients
        abc = np.linalg.solve(A, b_u)
        def_ = np.linalg.solve(A, b_v)

        a, b, c = abc
        d, e, f = def_

        # Form Jacobian
        J = np.array([[a, b], [d, e]])

        # Solve for critical point
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        # Get current centroid
        centroid = cluster.get_centroid()

        # Vector to center
        to_center = critical_point - centroid
        current_radius = np.linalg.norm(to_center)

        if current_radius < 1e-6:
            # Move in random direction if too close
            angle = np.random.uniform(0, 2*np.pi)
            return np.cos(angle), np.sin(angle)

        # Normalize direction to center
        to_center_norm = to_center / current_radius

        # Perpendicular direction for orbit (90 degrees rotation)
        orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

        # Radius error
        radius_error = 3 * (current_radius - desired_radius)

        # Adjust orbit direction based on radius error
        adjustment_strength = min(1.5, abs(radius_error))

        if radius_error > 0:
            # Too far out - add inward component
            final_direction = orbit_direction + adjustment_strength * to_center_norm
        else:
            # Too close in - add outward component
            final_direction = orbit_direction - adjustment_strength * to_center_norm

        # Normalize
        final_direction = final_direction / np.linalg.norm(final_direction)

        return final_direction[0], final_direction[1]

    except np.linalg.LinAlgError:
        return vector_sum(cluster)


def find_center(cluster):
    """
    Find center using weighted combination of rotational and radial components.

    Returns:
        (vx_c, vy_c): Velocity toward estimated center
    """
    # Get Jacobian information
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian_from_readings(cluster)

    # Get readings
    readings = np.array(cluster.sample_field_at_robots())
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)

    if velocity_magnitude < 1e-6:
        return 0.0, 0.0

    # Normalized velocity vector
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude

    # Calculate weights
    rot_strength = np.abs(curl_z)
    rad_strength = np.abs(divergence)
    total_strength = rot_strength + rad_strength

    if total_strength < 1e-10:
        return 0.0, 0.0

    w_rot = rot_strength / total_strength
    w_rad = rad_strength / total_strength

    # Rotational component
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u

    # Radial component
    if divergence > 0:  # source
        rad_u = -v_hat_u
        rad_v = -v_hat_v
    else:  # sink
        rad_u = v_hat_u
        rad_v = v_hat_v

    # Weighted combination
    vx = w_rot * rot_u + w_rad * rad_u
    vy = w_rot * rot_v + w_rad * rad_v

    return vx, vy


def find_sink_center(cluster):
    """
    Find sink center by moving with/against flow based on divergence.

    Returns:
        (vx_c, vy_c): Velocity toward sink center
    """
    # Get Jacobian
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian_from_readings(cluster)

    # Get readings
    readings = np.array(cluster.sample_field_at_robots())
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])

    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)

    if velocity_magnitude < 1e-6:
        return 0.0, 0.0

    # Unit vectors
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude

    attraction_strength = 0.6

    if divergence > 0:  # source
        move_u = -v_hat_u  # move against flow
        move_v = -v_hat_v
    else:  # sink
        move_u = v_hat_u   # move with flow
        move_v = v_hat_v

    return attraction_strength * move_u, attraction_strength * move_v
