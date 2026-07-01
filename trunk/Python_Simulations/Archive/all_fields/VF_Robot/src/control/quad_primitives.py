"""
Control primitives for 4-robot quadrilateral formation.

All primitives take a QuadCluster object and return desired centroid velocity (vx_c, vy_c)
or (vx_c, vy_c, omega_c) for orientation control.

Ported from VF_4_Robot/primitives/control_primitives.py
"""

import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_angle_error_bidirectional(current_angle, desired_angle):
    """
    Calculate angle error considering bidirectional alignment (either parallel or anti-parallel).

    Since d1 can point in either direction, we want the smallest angular adjustment
    to align with the target direction (or its opposite).

    Args:
        current_angle: Current angle in radians
        desired_angle: Desired angle in radians

    Returns:
        angle_error: Signed angle error in radians, wrapped to [-π, π]
    """
    # Calculate direct error (wrapping to [-π, π])
    error_direct = np.mod(desired_angle - current_angle + np.pi, 2*np.pi) - np.pi

    # Calculate error to opposite direction
    error_opposite = np.mod(desired_angle + np.pi - current_angle + np.pi, 2*np.pi) - np.pi

    # Choose whichever has smaller absolute error
    if np.abs(error_direct) <= np.abs(error_opposite):
        return error_direct
    else:
        return error_opposite


def calculate_jacobian_from_triangle(positions, readings):
    """
    Calculate Jacobian from 3 robot positions and their field readings.

    Uses plane fitting to estimate partial derivatives.

    Args:
        positions: List of 3 (x, y) tuples
        readings: List of 3 (u, v) tuples

    Returns:
        curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy
    """
    (x1, y1), (x2, y2), (x3, y3) = positions
    (u1, v1), (u2, v2), (u3, v3) = readings

    # Separate u and v components
    u_readings = np.array([u1, u2, u3])
    v_readings = np.array([v1, v2, v3])

    # Create position arrays for u field
    pos1_arr = np.array([x1, y1, u_readings[0]])
    pos2_arr = np.array([x2, y2, u_readings[1]])
    pos3_arr = np.array([x3, y3, u_readings[2]])

    # Calculate normal vector for u field using cross product
    R12_u = pos2_arr - pos1_arr
    R13_u = pos3_arr - pos1_arr
    N_u = np.cross(-R12_u, R13_u)

    # Partial derivatives of u
    du_dx = N_u[0]
    du_dy = N_u[1]

    # Repeat for v field
    pos1_arr = np.array([x1, y1, v_readings[0]])
    pos2_arr = np.array([x2, y2, v_readings[1]])
    pos3_arr = np.array([x3, y3, v_readings[2]])

    R12_v = pos2_arr - pos1_arr
    R13_v = pos3_arr - pos1_arr
    N_v = np.cross(-R12_v, R13_v)

    # Partial derivatives of v
    dv_dx = N_v[0]
    dv_dy = N_v[1]

    # Calculate curl and divergence
    curl_z = dv_dx - du_dy  # z-component of curl
    divergence = du_dx + dv_dy

    return curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy


def calculate_jacobian_top(cluster):
    """
    Calculate Jacobian using top triangle (robots 1, 2, 3).

    Args:
        cluster: QuadCluster object

    Returns:
        curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy
    """
    # Get positions and readings
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    readings = cluster.sample_field_at_robots()

    # Top triangle uses robots 1, 2, 3 (indices 0, 1, 2)
    positions = [(x1, y1), (x2, y2), (x3, y3)]
    triangle_readings = [readings[0], readings[1], readings[2]]

    return calculate_jacobian_from_triangle(positions, triangle_readings)


def calculate_jacobian_bottom(cluster):
    """
    Calculate Jacobian using bottom triangle (robots 2, 3, 4).

    Args:
        cluster: QuadCluster object

    Returns:
        curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy
    """
    # Get positions and readings
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    readings = cluster.sample_field_at_robots()

    # Bottom triangle uses robots 2, 3, 4 (indices 1, 2, 3)
    # But we reorder to 4, 2, 3 to match original implementation
    positions = [(x4, y4), (x2, y2), (x3, y3)]
    triangle_readings = [readings[3], readings[1], readings[2]]

    return calculate_jacobian_from_triangle(positions, triangle_readings)


def calculate_center_direction_from_triangle(positions, readings, curl_z, divergence):
    """
    Calculate direction to center using weighted combination of rotational and radial cues.

    Args:
        positions: List of 3 (x, y) tuples
        readings: List of 3 (u, v) tuples
        curl_z: Curl computed from Jacobian
        divergence: Divergence computed from Jacobian

    Returns:
        np.array([r_hat_u, r_hat_v]) direction vector, or None if stagnation point
    """
    # Sum the readings
    u_array = np.array([r[0] for r in readings])
    v_array = np.array([r[1] for r in readings])
    sum_u = np.sum(u_array)
    sum_v = np.sum(v_array)

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)

    # Check if we're at a stagnation point
    if velocity_magnitude < 1e-6:
        return None

    # Unit vector in velocity direction
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude

    # Calculate weights based on relative strengths
    rot_strength = np.abs(curl_z)
    rad_strength = np.abs(divergence)
    total_strength = rot_strength + rad_strength

    if total_strength < 1e-6:
        total_strength = 1e-6  # Prevent division by zero

    w_rot = rot_strength / total_strength
    w_rad = rad_strength / total_strength

    # Rotational component (perpendicular to flow)
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u

    # Radial component (with/against flow based on divergence)
    if divergence > 0:  # source - move against flow
        rad_u = -v_hat_u
        rad_v = -v_hat_v
    else:  # sink - move with flow
        rad_u = v_hat_u
        rad_v = v_hat_v

    # Combine based on relative strengths
    r_hat_u = w_rot * rot_u + w_rad * rad_u
    r_hat_v = w_rot * rot_v + w_rad * rad_v

    # Normalize the final direction vector
    magnitude = np.sqrt(r_hat_u**2 + r_hat_v**2)
    if magnitude > 1e-6:
        r_hat_u /= magnitude
        r_hat_v /= magnitude

    return np.array([r_hat_u, r_hat_v])


def calculate_center_direction_top(cluster):
    """Calculate direction to center using top triangle (robots 1, 2, 3)."""
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    readings = cluster.sample_field_at_robots()

    positions = [(x1, y1), (x2, y2), (x3, y3)]
    triangle_readings = [readings[0], readings[1], readings[2]]

    curl_z, divergence, _, _, _, _ = calculate_jacobian_top(cluster)

    return calculate_center_direction_from_triangle(positions, triangle_readings, curl_z, divergence)


def calculate_center_direction_bottom(cluster):
    """Calculate direction to center using bottom triangle (robots 2, 3, 4)."""
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    readings = cluster.sample_field_at_robots()

    # Match original ordering: robots 4, 2, 3
    positions = [(x4, y4), (x2, y2), (x3, y3)]
    triangle_readings = [readings[3], readings[1], readings[2]]

    curl_z, divergence, _, _, _, _ = calculate_jacobian_bottom(cluster)

    return calculate_center_direction_from_triangle(positions, triangle_readings, curl_z, divergence)


def find_line_intersection(p1, d1, p2, d2):
    """
    Find intersection of two lines defined by point and direction.

    Args:
        p1, p2: Starting points (np.array)
        d1, d2: Direction vectors (np.array)

    Returns:
        Intersection point or None if parallel
    """
    # Check for parallel lines
    det = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(det) < 1e-8:
        return None

    # Solve for intersection parameters
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    try:
        t1, t2 = np.linalg.solve(A, b)

        # Calculate intersection
        intersection1 = p1 + t1 * d1
        intersection2 = p2 + t2 * d2

        # Average for numerical stability
        intersection = (intersection1 + intersection2) / 2

        return intersection

    except np.linalg.LinAlgError:
        return None


def estimate_center_and_radius(cluster):
    """
    Estimate center using dual Jacobian approach - clean, no fallbacks.

    Each triangle estimates center position using plane fitting (just like 3-robot).
    Average the two position estimates.

    Args:
        cluster: QuadCluster object

    Returns:
        center_estimate (np.array), radius (float) or (None, None) if fails
    """
    # Get robot positions for top triangle (1, 2, 3)
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]
    u4, v4 = readings[3]

    # ========== TOP TRIANGLE (1, 2, 3) ==========
    # Matrix A for top triangle
    A_top = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])

    b_u_top = np.array([u1, u2, u3])
    b_v_top = np.array([v1, v2, v3])

    try:
        # Solve for plane coefficients
        abc_top = np.linalg.solve(A_top, b_u_top)
        def_top = np.linalg.solve(A_top, b_v_top)

        a_top, b_top, c_top = abc_top
        d_top, e_top, f_top = def_top

        # Form Jacobian
        J_top = np.array([[a_top, b_top], [d_top, e_top]])

        # Solve for critical point
        rhs_top = np.array([-c_top, -f_top])
        critical_point_top = np.linalg.solve(J_top, rhs_top)

    except np.linalg.LinAlgError:
        # Singular matrix - can't estimate
        return None, None

    # ========== BOTTOM TRIANGLE (2, 3, 4) ==========
    # Matrix A for bottom triangle
    A_bot = np.array([
        [x2, y2, 1],
        [x3, y3, 1],
        [x4, y4, 1]
    ])

    b_u_bot = np.array([u2, u3, u4])
    b_v_bot = np.array([v2, v3, v4])

    try:
        # Solve for plane coefficients
        abc_bot = np.linalg.solve(A_bot, b_u_bot)
        def_bot = np.linalg.solve(A_bot, b_v_bot)

        a_bot, b_bot, c_bot = abc_bot
        d_bot, e_bot, f_bot = def_bot

        # Form Jacobian
        J_bot = np.array([[a_bot, b_bot], [d_bot, e_bot]])

        # Solve for critical point
        rhs_bot = np.array([-c_bot, -f_bot])
        critical_point_bot = np.linalg.solve(J_bot, rhs_bot)

    except np.linalg.LinAlgError:
        # Singular matrix - can't estimate
        return None, None

    # ========== AVERAGE THE TWO POSITION ESTIMATES ==========
    center_estimate = (critical_point_top + critical_point_bot) / 2.0

    # Calculate distance from current centroid
    p0 = cluster.get_centroid()
    estimated_distance = np.linalg.norm(center_estimate - p0)

    return center_estimate, estimated_distance


# ============================================================================
# Control Primitives
# ============================================================================

def dual_jacobian_center_finder(cluster):
    """
    Move toward estimated center using dual Jacobian approach.

    Uses both top triangle (1,2,3) and bottom triangle (2,3,4) to
    estimate the critical point location.

    Args:
        cluster: QuadCluster object

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Estimate center and radius
    center_estimate, radius = estimate_center_and_radius(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0

    # Get current centroid
    current_centroid = cluster.get_centroid()

    # Calculate direction to estimated center
    direction = center_estimate - current_centroid
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        # Already at estimated center
        return 0.0, 0.0

    # Proportional control: velocity proportional to distance
    # This makes the robot slow down as it approaches the center
    gain = 1.0  # Proportional gain (tune for approach speed)
    max_velocity = 0.5  # Maximum commanded velocity (m/s)

    # Calculate proportional velocity
    velocity_magnitude = gain * distance

    # Clamp to maximum
    if velocity_magnitude > max_velocity:
        velocity_magnitude = max_velocity

    # Scale direction vector to desired velocity magnitude
    direction_unit = direction / distance
    velocity_command = direction_unit * velocity_magnitude

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]

    return vx_c, vy_c


def center_orbiter_quad(cluster, desired_radius=0.15):
    """
    Orbit around estimated center at desired radius.

    Args:
        cluster: QuadCluster object
        desired_radius: Desired orbital radius (default 0.15)

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Estimate center and current radius
    center_estimate, current_radius = estimate_center_and_radius(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0

    # Get current centroid
    current_centroid = cluster.get_centroid()

    # Vector from centroid to center
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        # Too close to center - move outward in random direction
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1]

    # Normalize direction to center
    to_center_norm = to_center / center_distance

    # Perpendicular direction for orbit (90° rotation)
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Compute radius error (how far from desired orbital radius)
    radius_error = current_radius - desired_radius

    # Orbital speed control
    # Base tangential speed for circular orbit
    base_orbital_speed = 0.2  # m/s (tune this for orbital velocity)

    # Radial speed for radius correction (proportional to error)
    radial_gain = 0.5  # Tune this to control how aggressively it corrects radius
    radial_speed = radial_gain * radius_error  # Positive = outward, negative = inward

    # Clamp radial correction to avoid extreme velocities
    max_radial_speed = 0.3  # m/s
    radial_speed = np.clip(radial_speed, -max_radial_speed, max_radial_speed)

    # Compute velocity components
    # Tangential component (for orbit)
    v_tangential = base_orbital_speed * orbit_direction

    # Radial component (for radius correction)
    # When too far (radius_error > 0): radial_speed > 0, want to move INWARD (toward center)
    # to_center_norm already points inward, so: positive_speed * inward_vector = move inward ✓
    # When too close (radius_error < 0): radial_speed < 0, negative_speed * inward_vector = move outward ✓
    v_radial = radial_speed * to_center_norm

    # Combine tangential and radial components
    velocity_command = v_tangential + v_radial

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]
    omega_c = 0.0  # No orientation control

    return vx_c, vy_c, omega_c


def vector_sum_quad(cluster):
    """
    Simple control primitive - move in direction of summed field vectors.

    Uses the actual field magnitude (scaled) rather than normalizing to constant speed.

    Args:
        cluster: QuadCluster object

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Get field readings at all robots
    readings = cluster.sample_field_at_robots()

    # Sum all field vectors
    sum_u = sum(r[0] for r in readings)
    sum_v = sum(r[1] for r in readings)

    magnitude = np.sqrt(sum_u**2 + sum_v**2)

    if magnitude < 1e-6:
        return 0.0, 0.0

    # Scale by a gain factor (rather than normalizing to 1.0)
    # This preserves information about field strength
    gain = 0.1  # Tune this to adjust following speed
    max_velocity = 0.5  # Maximum commanded velocity

    velocity_magnitude = gain * magnitude

    # Clamp to maximum
    if velocity_magnitude > max_velocity:
        velocity_magnitude = max_velocity

    # Scale vector to desired magnitude
    direction = np.array([sum_u, sum_v]) / magnitude
    velocity_command = direction * velocity_magnitude

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]
    omega_c = 0.0  # No orientation control

    return vx_c, vy_c, omega_c


def estimate_center_planar(cluster):
    """
    Estimate center location using 4-point least squares plane fitting.

    Fits planes through all 4 robot positions: u(x,y) = ax + by + c, v(x,y) = dx + ey + f
    Then solves for critical point where u=0 and v=0.

    Args:
        cluster: QuadCluster object

    Returns:
        center_estimate: numpy array [x, y] or None if estimation fails
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    # Get field readings
    readings = cluster.sample_field_at_robots()
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]
    u4, v4 = readings[3]

    # Build matrix A for overdetermined system (4 equations, 3 unknowns)
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1],
        [x4, y4, 1]
    ])

    # RHS for u and v components
    b_u = np.array([u1, u2, u3, u4])
    b_v = np.array([v1, v2, v3, v4])

    try:
        # Solve for plane coefficients using least squares
        abc, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
        def_, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)

        a, b, c = abc
        d, e, f = def_

        # Form the Jacobian matrix
        J = np.array([
            [a, b],
            [d, e]
        ])

        # Check if Jacobian is singular
        if np.abs(np.linalg.det(J)) < 1e-8:
            return None

        # Solve for critical point where u=0 and v=0
        rhs = np.array([-c, -f])
        critical_point = np.linalg.solve(J, rhs)

        return critical_point

    except np.linalg.LinAlgError:
        return None


def four_planar_center_finder(cluster):
    """
    Find critical point using least squares plane fitting with all 4 robots.

    Instead of using two triangles separately, this fits a single plane through
    all 4 robot positions simultaneously using least squares. This uses all
    available data at once for potentially more robust estimation.

    Fits planes: u(x,y) = ax + by + c and v(x,y) = dx + ey + f
    Then solves for critical point where u=0 and v=0.

    Args:
        cluster: QuadCluster object

    Returns:
        (vx_c, vy_c): Desired centroid velocity toward critical point
    """
    # Estimate center using 4-point plane fitting
    center_estimate = estimate_center_planar(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0

    # Get current centroid
    centroid = cluster.get_centroid()

    # Calculate vector to critical point
    vector_to_critical = center_estimate - centroid
    distance = np.linalg.norm(vector_to_critical)

    if distance < 1e-6:
        # Already at critical point
        return 0.0, 0.0

    # Proportional control: velocity proportional to distance
    gain = 1.0  # Proportional gain
    max_velocity = 0.5  # Maximum commanded velocity (m/s)

    # Calculate proportional velocity
    velocity_magnitude = gain * distance

    # Clamp to maximum
    if velocity_magnitude > max_velocity:
        velocity_magnitude = max_velocity

    # Scale vector to desired velocity magnitude
    direction = vector_to_critical / distance
    velocity_command = direction * velocity_magnitude

    omega_c = 0.0  # No orientation control
    return velocity_command[0], velocity_command[1], omega_c


def center_orbiter_quad_planar(cluster, desired_radius=0.15):
    """
    Orbit around estimated center at desired radius using 4-point planar estimation.

    Uses least squares plane fitting through all 4 robots (instead of dual Jacobian)
    to estimate the center, then orbits around it at the desired radius.

    Args:
        cluster: QuadCluster object
        desired_radius: Desired orbital radius (default 0.15)

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Estimate center using 4-point plane fitting
    center_estimate = estimate_center_planar(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0

    # Get current centroid
    current_centroid = cluster.get_centroid()

    # Vector from centroid to center
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        # Too close to center - move outward in random direction
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1]

    # Normalize direction to center
    to_center_norm = to_center / center_distance

    # Perpendicular direction for orbit (90° rotation)
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Compute radius error (how far from desired orbital radius)
    # Using center_distance as the current radius estimate
    current_radius = center_distance
    radius_error = current_radius - desired_radius

    # Orbital speed control
    # Base tangential speed for circular orbit
    base_orbital_speed = 0.2  # m/s (tune this for orbital velocity)

    # Radial speed for radius correction (proportional to error)
    radial_gain = 0.5  # Tune this to control how aggressively it corrects radius
    radial_speed = radial_gain * radius_error  # Positive = outward, negative = inward

    # Clamp radial correction to avoid extreme velocities
    max_radial_speed = 0.3  # m/s
    radial_speed = np.clip(radial_speed, -max_radial_speed, max_radial_speed)

    # Compute velocity components
    # Tangential component (for orbit)
    v_tangential = base_orbital_speed * orbit_direction

    # Radial component (for radius correction)
    # When too far (radius_error > 0): radial_speed > 0, want to move INWARD (toward center)
    # to_center_norm already points inward, so: positive_speed * inward_vector = move inward ✓
    # When too close (radius_error < 0): radial_speed < 0, negative_speed * inward_vector = move outward ✓
    v_radial = radial_speed * to_center_norm

    # Combine tangential and radial components
    velocity_command = v_tangential + v_radial

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]
    omega_c = 0.0  # No orientation control

    return vx_c, vy_c, omega_c

def dual_jacobian_center_finder_advanced(cluster, use_angular_control=True):
    """
    Move toward estimated center with optional orientation control.

    Uses dual Jacobian approach to estimate center, then:
    1. Moves toward the center (like dual_jacobian_center_finder)
    2. Optionally orients d1 diagonal to point toward the center

    Designed for use with quad_rectangle formation where d1 is the short diagonal.

    Args:
        cluster: QuadCluster object
        use_angular_control: If False, disables omega_c (for unstable fields like saddles)

    Returns:
        (vx_c, vy_c, omega_c): Desired centroid velocity and angular velocity
    """
    # Estimate center and radius using dual Jacobian
    center_estimate, radius = estimate_center_and_radius(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0, 0.0

    # Get current centroid and formation
    current_centroid = cluster.get_centroid()
    formation = cluster.get_current_formation()
    current_theta_c = formation['th_c']  # Angle of d1 diagonal

    # Calculate direction to estimated center
    direction_to_center = center_estimate - current_centroid
    distance = np.linalg.norm(direction_to_center)

    if distance < 1e-6:
        # Already at estimated center
        return 0.0, 0.0, 0.0

    # Desired angle: direction from centroid to center
    desired_theta_raw = np.arctan2(direction_to_center[1], direction_to_center[0])

    # Unwrap desired_theta to be continuous with current_theta_c
    # This prevents issues when current_theta_c is unwrapped (e.g., 370°)
    # and desired_theta is wrapped (e.g., -10°)
    desired_theta = desired_theta_raw
    while (desired_theta - current_theta_c) > np.pi:
        desired_theta -= 2 * np.pi
    while (desired_theta - current_theta_c) < -np.pi:
        desired_theta += 2 * np.pi

    # Calculate angle error (simple direct difference, no bidirectional)
    # d1 (robot 1→3) always points toward center
    angle_error = desired_theta - current_theta_c

    # Angular velocity control (purely reactive)
    if use_angular_control:
        omega_gain = 2.5  # Angular gain (allows some lag)
        max_omega = 3.0  # Maximum angular velocity (rad/s)
        omega_c = np.clip(omega_gain * angle_error, -max_omega, max_omega)
    else:
        omega_c = 0.0  # No angular control

    # Translational velocity control (proportional to distance)
    gain = 1.0  # Proportional gain
    max_velocity = 0.5  # Maximum commanded velocity (m/s)

    # Calculate proportional velocity
    velocity_magnitude = gain * distance

    # Clamp to maximum
    if velocity_magnitude > max_velocity:
        velocity_magnitude = max_velocity

    # Scale direction vector to desired velocity magnitude
    direction_unit = direction_to_center / distance
    velocity_command = direction_unit * velocity_magnitude

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]

    return vx_c, vy_c, omega_c


def center_orbiter_quad_advanced(cluster, desired_radius=0.15):
    """
    Orbit around estimated center with orientation control (advanced version).

    Uses dual Jacobian approach to estimate center, then:
    1. Orbits around the center at desired radius (like center_orbiter_quad)
    2. Orients d1 diagonal to point toward the center

    Designed for use with quad_rectangle formation where d1 is the short diagonal.

    Args:
        cluster: QuadCluster object
        desired_radius: Desired orbital radius (default 0.15)

    Returns:
        (vx_c, vy_c, omega_c): Desired centroid velocity and angular velocity
    """
    # Estimate center and current radius using dual Jacobian
    center_estimate, current_radius = estimate_center_and_radius(cluster)

    if center_estimate is None:
        # If estimation fails, don't move
        return 0.0, 0.0, 0.0

    # Get current centroid and formation
    current_centroid = cluster.get_centroid()
    formation = cluster.get_current_formation()
    current_theta_c = formation['th_c']  # Angle of d1 diagonal

    # Vector from centroid to center
    to_center = center_estimate - current_centroid
    center_distance = np.linalg.norm(to_center)

    if center_distance < 1e-6:
        # Too close to center - move outward in random direction
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        return orbit_direction[0], orbit_direction[1], 0.0

    # Normalize direction to center
    to_center_norm = to_center / center_distance

    # Desired angle: direction from centroid to center (d1 should point radially)
    desired_theta_raw = np.arctan2(to_center[1], to_center[0])

    # Unwrap desired_theta to be continuous with current_theta_c
    desired_theta = desired_theta_raw
    while (desired_theta - current_theta_c) > np.pi:
        desired_theta -= 2 * np.pi
    while (desired_theta - current_theta_c) < -np.pi:
        desired_theta += 2 * np.pi

    # Calculate angle error (simple direct difference, no bidirectional)
    # d1 (robot 1→3) always points toward center
    angle_error = desired_theta - current_theta_c

    # Angular velocity control (purely reactive)
    omega_gain = 3  # Angular gain (allows some lag)
    max_omega = 5.0  # Maximum angular velocity (rad/s)
    omega_c = np.clip(omega_gain * angle_error, -max_omega, max_omega)

    # Perpendicular direction for orbit (90° rotation)
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Compute radius error (how far from desired orbital radius)
    radius_error = current_radius - desired_radius

    # Orbital speed control
    base_orbital_speed = 0.2  # m/s (tangential velocity)
    radial_gain = 0.5  # Radial correction gain
    radial_speed = radial_gain * radius_error

    # Clamp radial correction
    max_radial_speed = 0.3  # m/s
    radial_speed = np.clip(radial_speed, -max_radial_speed, max_radial_speed)

    # Compute velocity components
    v_tangential = base_orbital_speed * orbit_direction
    v_radial = radial_speed * to_center_norm

    # Combine tangential and radial components
    velocity_command = v_tangential + v_radial

    vx_c = velocity_command[0]
    vy_c = velocity_command[1]

    return vx_c, vy_c, omega_c
