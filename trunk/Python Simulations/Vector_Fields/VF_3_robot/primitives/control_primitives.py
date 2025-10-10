# control_primitives.py
import numpy as np
import matplotlib.pyplot as plt

## Helper Functions

def calculate_jacobian(cluster):
    # Get positions and readings
    pos1, pos2, pos3 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Separate u and v components
    u_readings = np.array([r[0] for r in readings])
    v_readings = np.array([r[1] for r in readings])
    
    # Create position arrays
    pos1_arr = np.array([*pos1, u_readings[0]])  # For u field
    pos2_arr = np.array([*pos2, u_readings[1]])
    pos3_arr = np.array([*pos3, u_readings[2]])
    
    # Calculate normal vector for u field
    R12_u = pos2_arr - pos1_arr
    R13_u = pos3_arr - pos1_arr
    N_u = np.cross(-R12_u, R13_u)
    # du/dx and du/dy are proportional to N_u[0] and N_u[1]
    du_dx = N_u[0]
    du_dy = N_u[1]
    
    # Repeat for v field
    pos1_arr = np.array([*pos1, v_readings[0]])
    pos2_arr = np.array([*pos2, v_readings[1]])
    pos3_arr = np.array([*pos3, v_readings[2]])
    
    R12_v = pos2_arr - pos1_arr
    R13_v = pos3_arr - pos1_arr
    N_v = np.cross(-R12_v, R13_v)
    dv_dx = N_v[0]
    dv_dy = N_v[1]
    
    # Calculate curl and divergence
    curl_z = dv_dx - du_dy  # z-component of curl
    divergence = du_dx + dv_dy
    
    return curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy

def calculate_center_direction(cluster):
    # Get Jacobian information
    curl_z, _, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Get velocity at cluster center
    readings = np.array(cluster.bot_readings())
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)
    
    # Check if we're at a stagnation point
    if velocity_magnitude < 1e-6:
        return None
        
    # Calculate unit vector in velocity direction
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude
    
    # Calculate unit vector pointing to center based on rotation direction
    if curl_z > 0:  # counterclockwise
        r_hat_u = -v_hat_v
        r_hat_v = v_hat_u
    else:  # clockwise
        r_hat_u = v_hat_v
        r_hat_v = -v_hat_u
    
    return np.array([r_hat_u, r_hat_v])




## Full Vector Techniques


def find_center(cluster):
    # Get Jacobian information
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Get readings from all robots
    readings = np.array(cluster.bot_readings())
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)
    
    if velocity_magnitude < 1e-6:
        return cluster.cluster_centre
    
    # Get normalized velocity vector
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude
    
    # Calculate weights based on relative strengths
    rot_strength = np.abs(curl_z)
    rad_strength = np.abs(divergence)
    total_strength = rot_strength + rad_strength
    
    w_rot = rot_strength / total_strength
    w_rad = rad_strength / total_strength
    print()
    print('Some Statistics on what we are seeing')
    print('Rotation strenght ', rot_strength)
    print('w_rot', w_rot)
    print('Radial strenght ', rad_strength)
    print('w_rad', w_rad)
    print('Curl of Z', curl_z)
    print('Divergence', divergence)
    print (du_dx, du_dy)
    print(dv_dx, dv_dy)


    # Get rotational component
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
        print('counterclockwise')
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u
        print('clockwise')
    
    # Get radial component
    if divergence > 0:  # source
        rad_u = -v_hat_u
        rad_v = -v_hat_v
        print('Source detected')
    else:  # sink
        rad_u = v_hat_u
        rad_v = v_hat_v
        print('Sink detected')
    
    # Simple weighted combination
    xₖ = cluster.step_size * (w_rot * rot_u + w_rad * rad_u)
    yₖ = cluster.step_size * (w_rot * rot_v + w_rad * rad_v)
    
    # Update cluster position
    cluster.cluster_centre[0] += xₖ
    cluster.cluster_centre[1] += yₖ
    print()
    return cluster.cluster_centre

import numpy as np
def find_center2(cluster, tiny=1e-12):
    # Local field
    curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)

    readings = np.array(cluster.bot_readings(), float)  # (N,2)
    v = readings.sum(axis=0)
    s = float(np.hypot(v[0], v[1]))
    if s < tiny:
        return cluster.cluster_centre

    vhat = v / (s + tiny)

    # 1) Source vs sink decision (parameter-free)
    ed = -(J.T @ v)                    # energy-descent vector
    go_with_flow = (np.dot(vhat, ed) >= 0.0)
    radial = vhat if go_with_flow else -vhat

    # 2) Case-1 vs case-3 diagnostic (parameter-free)
    sigma = float(v @ (J @ v)) / (s*s + tiny)  # d|v|/ds along +vhat
    # Use radial either way, but ONLY blend in energy-descent when sigma<0 (case-1)
    ed_norm = float(np.hypot(ed[0], ed[1]))
    ehat = ed / (ed_norm + tiny) if ed_norm > tiny else np.zeros(2)

    # 3) Gentle, continuous blend without tunables
    # (i) always keep a strong radial component toward/against flow
    # (ii) add some energy-descent only when it helps (case-1: sigma<0)
    w_rad = 1.0
    w_ene = 1.0 if (sigma < 0.0) else 0.0  # exact sign, no numeric threshold

    step_dir = w_rad*radial + w_ene*ehat
    n = float(np.hypot(step_dir[0], step_dir[1]))
    step_dir = step_dir / (n + tiny)

    h = float(cluster.step_size)
    cluster.cluster_centre[0] += h * step_dir[0]
    cluster.cluster_centre[1] += h * step_dir[1]
    return cluster.cluster_centre






def vector_sum(cluster):
    # Get readings from all robots
    readings = cluster.bot_readings()
    readings = np.array(readings)  # Convert to numpy array for easier manipulation
    
    # Sum u and v components
    sum_u = np.sum(readings[:, 0])  # Sum all u components
    sum_v = np.sum(readings[:, 1])  # Sum all v components
    
    # Take step in the summed direction
    cluster.cluster_centre[0] += cluster.step_size * sum_u
    cluster.cluster_centre[1] += cluster.step_size * sum_v
    
    return cluster.cluster_centre

def direction_follow_with_center_attraction(cluster):
    # Get readings from all robots
    readings = cluster.bot_readings()
    readings = np.array(readings)
    
    # Sum u and v components
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])
    
    # Get direction to center
    center_direction = calculate_center_direction(cluster)
    
    if center_direction is not None:
        # Add attraction force in direction of center
        # Could scale this with radius estimate if needed
        attraction_strength = .6
        sum_u += attraction_strength * center_direction[0]
        sum_v += attraction_strength * center_direction[1]
    
    # Take step in the combined direction
    cluster.cluster_centre[0] += cluster.step_size * sum_u
    cluster.cluster_centre[1] += cluster.step_size * sum_v
    
    return cluster.cluster_centre

def center_attraction(cluster):
    # Get readings from all robots
    readings = cluster.bot_readings()
    readings = np.array(readings)
    
    
    # Get direction to center
    center_direction = calculate_center_direction(cluster)
    
    if center_direction is not None:
        # Add attraction force in direction of center
        # Could scale this with radius estimate if needed
        attraction_strength = .6
        sum_u = attraction_strength * center_direction[0]
        sum_v = attraction_strength * center_direction[1]
    
    # Take step in the combined direction
    cluster.cluster_centre[0] += cluster.step_size * sum_u
    cluster.cluster_centre[1] += cluster.step_size * sum_v
    
    return cluster.cluster_centre


def find_sink_center(cluster):
    # Get Jacobian information
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Get readings from all robots
    readings = np.array(cluster.bot_readings())
    
    # Sum u and v components
    sum_u = np.sum(readings[:, 0])
    sum_v = np.sum(readings[:, 1])
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)
    
    # Check if at stagnation point
    if velocity_magnitude < 1e-6:
        return cluster.cluster_centre
    
    # Get unit vectors
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude
    
    # Set direction based on divergence
    attraction_strength = 0.6
    if divergence > 0:  # source
        move_u = -v_hat_u  # move against flow
        move_v = -v_hat_v
    else:  # sink
        move_u = v_hat_u   # move with flow
        move_v = v_hat_v
        
    # Update cluster center position
    cluster.cluster_centre[0] += cluster.step_size * attraction_strength * move_u
    cluster.cluster_centre[1] += cluster.step_size * attraction_strength * move_v
    
    return cluster.cluster_centre


def critical_point_plane_fitting(cluster):
    """
    Estimates the location of the critical point using plane fitting method (Approach 1).
    
    This function:
    1. Fits planes to the u and v components of the vector field using 3 robot positions
    2. Solves for the critical point where both u and v equal zero
    3. Moves the cluster toward this critical point
    
    The plane fitting approach uses:
    U(x,y) = ax + by + c
    V(x,y) = dx + ey + f
    
    And solves for the critical point where U=V=0
    """
    # Get positions and readings
    pos1, pos2, pos3 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Extract coordinates and field values
    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    
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
    #b_u = 1/b_u
    #b_v = 1/b_v
    
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
        
        # Calculate vector from cluster center to critical point
        vector_to_critical = critical_point - cluster.cluster_centre
        
        # Calculate distance to critical point
        distance = np.linalg.norm(vector_to_critical)
        
        if distance < 1e-6:  # Already at critical point
            return cluster.cluster_centre
            
        # Limit maximum step and normalize
        max_step = 5 * cluster.step_size
        if distance > max_step:
            vector_to_critical = (vector_to_critical / distance) * max_step
        else:
            vector_to_critical = cluster.step_size * (vector_to_critical / distance)
        
        # Move toward critical point
        cluster.cluster_centre += vector_to_critical
        
        return cluster.cluster_centre
    
    except np.linalg.LinAlgError:
        # In case of singular matrix (e.g., collinear points)
        # Fall back to a simpler approach
        return vector_sum(cluster)


def critical_point_cross_product(cluster):
    """
    Estimates the location of the critical point using cross product method (Approach 2).
    
    This function:
    1. Uses cross products to estimate the gradient of the vector field
    2. Computes the Jacobian matrix from these gradients
    3. Solves for the critical point and moves the cluster toward it
    
    The cross product method finds the normal vectors to the planes:
    Ax + By + Cu + D = 0
    Ex + Fy + Gv + H = 0
    
    From which we get the partial derivatives:
    du/dx = -A/C, du/dy = -B/C
    dv/dx = -E/G, dv/dy = -F/G
    """
    # Get positions and readings
    pos1, pos2, pos3 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Extract coordinates and field values
    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]
    
    # Calculate vectors for cross product
    vec1_u = np.array([x2 - x1, y2 - y1, u2 - u1])
    vec2_u = np.array([x3 - x1, y3 - y1, u3 - u1])
    cross_u = np.cross(vec1_u, vec2_u)
    
    vec1_v = np.array([x2 - x1, y2 - y1, v2 - v1])
    vec2_v = np.array([x3 - x1, y3 - y1, v3 - v1])
    cross_v = np.cross(vec1_v, vec2_v)
    
    # Extract coefficients for the planes
    A, B, C = cross_u
    E, F, G = cross_v
    
    # Avoid division by zero
    if abs(C) < 1e-6 or abs(G) < 1e-6:
        return vector_sum(cluster)
    
    # Calculate partial derivatives
    du_dx = -A/C
    du_dy = -B/C
    dv_dx = -E/G
    dv_dy = -F/G
    
    # Form Jacobian matrix
    J = np.array([
        [du_dx, du_dy],
        [dv_dx, dv_dy]
    ])
    
    # Calculate average position and field values
    xc = cluster.cluster_centre[0]
    yc = cluster.cluster_centre[1]
    avg_u = (u1 + u2 + u3) / 3
    avg_v = (v1 + v2 + v3) / 3
    
    try:
        # Solve for the critical point
        # U = a(x-x0) + b(y-y0) where U=0 at critical point
        # So avg_u = du/dx(xc-x0) + du/dy(yc-y0) 
        # Therefore: du/dx*x0 + du/dy*y0 = du/dx*xc + du/dy*yc - avg_u
        rhs = np.array([
            du_dx * xc + du_dy * yc - avg_u,
            dv_dx * xc + dv_dy * yc - avg_v
        ])
        
        critical_point = np.linalg.solve(J, rhs)
        
        # Calculate vector from cluster center to critical point
        vector_to_critical = critical_point - cluster.cluster_centre
        
        # Calculate distance to critical point
        distance = np.linalg.norm(vector_to_critical)
        
        if distance < 1e-6:  # Already at critical point
            return cluster.cluster_centre
            
        # Limit maximum step and normalize
        max_step = 5 * cluster.step_size
        if distance > max_step:
            vector_to_critical = (vector_to_critical / distance) * max_step
        else:
            vector_to_critical = cluster.step_size * (vector_to_critical / distance)
        
        # Move toward critical point
        cluster.cluster_centre += vector_to_critical
        
        return cluster.cluster_centre
    
    except np.linalg.LinAlgError:
        # In case of singular matrix or if determinant is close to zero
        # Fall back to a simpler approach
        return vector_sum(cluster)
    
def critical_point_orbiter_plane_fitting(cluster):
    """
    Makes the robot cluster orbit around a critical point estimated using the plane fitting method.
    
    Features:
    - Uses plane fitting method to estimate critical point
    - Maintains a desired orbital radius from the critical point
    - Adjusts trajectory to move closer or farther to maintain orbital radius
    - Moves perpendicular to the radial direction for orbital motion
    """
    # Desired orbital radius - can be adjusted as needed
    des_rad = 0.4
    
    print("\n=== Starting critical point orbiter (plane fitting) ===")
    print(f"Initial cluster center: {cluster.cluster_centre}")
    print(f"Desired orbit radius: {des_rad}")
    
    # Get positions and readings
    pos1, pos2, pos3 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Extract coordinates and field values
    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    
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
        
        print(f"Estimated critical point: {critical_point}")
        
        # Calculate vector from cluster center to critical point
        to_center = critical_point - cluster.cluster_centre
        current_radius = np.linalg.norm(to_center)
        
        print(f"Current distance to critical point: {current_radius}")
        
        if current_radius < 1e-6:
            print("Too close to critical point - moving outward in random direction")
            # If we're too close to center, move in a random direction
            angle = np.random.uniform(0, 2*np.pi)
            orbit_direction = np.array([np.cos(angle), np.sin(angle)])
            new_center = cluster.cluster_centre + cluster.step_size * orbit_direction
            return new_center
        
        # Normalize direction to center
        to_center_norm = to_center / current_radius
        
        # Calculate perpendicular direction for orbit (90 degrees rotation)
        orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])
        
        # Compute the radius error
        radius_error = 3*(current_radius - des_rad)
        
        # Print initial directions for debugging
        print(f"To center vector: [{to_center_norm[0]:.4f}, {to_center_norm[1]:.4f}]")
        print(f"Orbit direction: [{orbit_direction[0]:.4f}, {orbit_direction[1]:.4f}]")
        
        # Calculate adjustment - SIMPLIFIED APPROACH
        # Start with pure orbit direction
        final_direction = orbit_direction.copy()
        
        # If we're too far out (radius_error > 0), add component toward center
        # If we're too close in (radius_error < 0), add component away from center
        adjustment_strength = min(1.5, abs(radius_error))  # Cap at 0.5 for stability
        
        if radius_error > 0:
            print(f"Too far out by {radius_error:.4f} - adding inward component")
            # Add inward component (toward center)
            final_direction += adjustment_strength * to_center_norm
        else:
            print(f"Too close in by {abs(radius_error):.4f} - adding outward component")
            # Add outward component (away from center)
            final_direction -= adjustment_strength * to_center_norm
        
        # Normalize the final direction
        final_direction = final_direction / np.linalg.norm(final_direction)
        
        # Print debugging information
        print(f"Current radius: {current_radius}, desired radius: {des_rad}")
        print(f"Radius error: {radius_error}")
        print(f"Adjustment strength: {adjustment_strength:.4f}")
        print(f"Final direction: [{final_direction[0]:.4f}, {final_direction[1]:.4f}]")
        
        # Update cluster center with standard step size
        new_center = cluster.cluster_centre + cluster.step_size * final_direction
        
        # Additional debugging - show expected new distance to center
        expected_new_distance = np.linalg.norm(new_center - critical_point)
        print(f"Expected new distance to critical point: {expected_new_distance:.4f}")
        print(f"New cluster center: {new_center}")
        print("=== Finished critical point orbiter (plane fitting) ===\n")
        
        return new_center
        
    except np.linalg.LinAlgError:
        print("Error in critical point estimation - falling back to vector sum")
        # Fall back to a simpler approach
        return vector_sum(cluster)


def critical_point_orbiter_cross_product(cluster):
    """
    Makes the robot cluster orbit around a critical point estimated using the cross product method.
    
    Features:
    - Uses cross product method to estimate critical point
    - Maintains a desired orbital radius from the critical point
    - Adjusts trajectory to move closer or farther to maintain orbital radius
    - Moves perpendicular to the radial direction for orbital motion
    """
    # Desired orbital radius - can be adjusted as needed
    des_rad = 3
    
    print("\n=== Starting critical point orbiter (cross product) ===")
    print(f"Initial cluster center: {cluster.cluster_centre}")
    print(f"Desired orbit radius: {des_rad}")
    
    # Get positions and readings
    pos1, pos2, pos3 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Extract coordinates and field values
    x1, y1 = pos1
    x2, y2 = pos2
    x3, y3 = pos3
    
    u1, v1 = readings[0]
    u2, v2 = readings[1]
    u3, v3 = readings[2]
    
    # Calculate vectors for cross product
    vec1_u = np.array([x2 - x1, y2 - y1, u2 - u1])
    vec2_u = np.array([x3 - x1, y3 - y1, u3 - u1])
    cross_u = np.cross(vec1_u, vec2_u)
    
    vec1_v = np.array([x2 - x1, y2 - y1, v2 - v1])
    vec2_v = np.array([x3 - x1, y3 - y1, v3 - v1])
    cross_v = np.cross(vec1_v, vec2_v)
    
    # Extract coefficients for the planes
    A, B, C = cross_u
    E, F, G = cross_v
    
    # Avoid division by zero
    if abs(C) < 1e-6 or abs(G) < 1e-6:
        print("Division by zero in cross product - falling back to vector sum")
        return vector_sum(cluster)
    
    # Calculate partial derivatives
    du_dx = -A/C
    du_dy = -B/C
    dv_dx = -E/G
    dv_dy = -F/G
    
    # Form Jacobian matrix
    J = np.array([
        [du_dx, du_dy],
        [dv_dx, dv_dy]
    ])
    
    # Calculate average position and field values
    xc = cluster.cluster_centre[0]
    yc = cluster.cluster_centre[1]
    avg_u = (u1 + u2 + u3) / 3
    avg_v = (v1 + v2 + v3) / 3
    
    try:
        # Solve for the critical point
        rhs = np.array([
            du_dx * xc + du_dy * yc - avg_u,
            dv_dx * xc + dv_dy * yc - avg_v
        ])
        
        critical_point = np.linalg.solve(J, rhs)
        print(f"Estimated critical point: {critical_point}")
        
        # Calculate vector from cluster center to critical point
        to_center = critical_point - cluster.cluster_centre
        current_radius = np.linalg.norm(to_center)
        
        print(f"Current distance to critical point: {current_radius}")
        
        if current_radius < 1e-6:
            print("Too close to critical point - moving outward in random direction")
            # If we're too close to center, move in a random direction
            angle = np.random.uniform(0, 2*np.pi)
            orbit_direction = np.array([np.cos(angle), np.sin(angle)])
            new_center = cluster.cluster_centre + cluster.step_size * orbit_direction
            return new_center
        
        # Normalize direction to center
        to_center_norm = to_center / current_radius
        
        # Calculate perpendicular direction for orbit (90 degrees rotation)
        orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])
        
        # Compute the radius error
        radius_error = current_radius - des_rad
        
        # Print initial directions for debugging
        print(f"To center vector: [{to_center_norm[0]:.4f}, {to_center_norm[1]:.4f}]")
        print(f"Orbit direction: [{orbit_direction[0]:.4f}, {orbit_direction[1]:.4f}]")
        
        # Calculate adjustment
        # Start with pure orbit direction
        final_direction = orbit_direction.copy()
        
        # If we're too far out (radius_error > 0), add component toward center
        # If we're too close in (radius_error < 0), add component away from center
        adjustment_strength = min(0.5, abs(radius_error))  # Cap at 0.5 for stability
        
        if radius_error > 0:
            print(f"Too far out by {radius_error:.4f} - adding inward component")
            # Add inward component (toward center)
            final_direction += adjustment_strength * to_center_norm
        else:
            print(f"Too close in by {abs(radius_error):.4f} - adding outward component")
            # Add outward component (away from center)
            final_direction -= adjustment_strength * to_center_norm
        
        # Normalize the final direction
        final_direction = final_direction / np.linalg.norm(final_direction)
        
        # Print debugging information
        print(f"Current radius: {current_radius}, desired radius: {des_rad}")
        print(f"Radius error: {radius_error}")
        print(f"Adjustment strength: {adjustment_strength:.4f}")
        print(f"Final direction: [{final_direction[0]:.4f}, {final_direction[1]:.4f}]")
        
        # Update cluster center with standard step size
        new_center = cluster.cluster_centre + cluster.step_size * final_direction
        
        # Additional debugging - show expected new distance to center
        expected_new_distance = np.linalg.norm(new_center - critical_point)
        print(f"Expected new distance to critical point: {expected_new_distance:.4f}")
        print(f"New cluster center: {new_center}")
        print("=== Finished critical point orbiter (cross product) ===\n")
        
        return new_center
        
    except np.linalg.LinAlgError:
        print("Error in critical point estimation - falling back to vector sum")
        # Fall back to a simpler approach
        return vector_sum(cluster)
    

import numpy as np

def center_finder3(
    cluster,
    eps_cue=0.05, eps_div=0.05, eps_om=0.05,
    tiny=1e-12,
    verbose=True, log_vectors=False, log_every=1,
    log_path="/mnt/data/center_finder3_log.txt"
):
    """
    Center finder (v3) – compact, sign-flip version with None-able thresholds.

    Uses only local (v, J) at the current cluster center and picks direction via:
      - radial family:     center_dir = +ĝ for 1/r-type,  -ĝ for r-type
      - vortex family:     center_dir = -ŝ⊥   (toward curvature center)

    Threshold behavior:
      - If eps_cue is None: pick family by relative size (γ vs σ), no absolute cutoff.
      - If eps_div / eps_om are None: skip k-type shortcuts for that family.
      - Otherwise use the provided thresholds (defaults 0.05).

    Logging preserved (γ, σ, d̄, w̄, divergences, curls, radii, step, etc.).
    """
    import numpy as np

    # -------------------------- helpers --------------------------------------
    def _emit(line: str):
        if not verbose:
            return
        print(line)
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass  # never crash on logging

    def _env_eval(xy):
        """Evaluate field (u,v) at xy via cluster's env hook; fallback: robot readings mean."""
        x, y = float(xy[0]), float(xy[1])
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn):
                return np.array(fn(x, y), dtype=float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m):
                return np.array(m(x, y), dtype=float)
        R = np.array(cluster.bot_readings(), dtype=float)  # (N,2)
        return R.mean(axis=0)

    def _calc_jacobian(center_xy):
        """Use user-provided calculate_jacobian(cluster) if present; else numeric central diffs."""
        try:
            # expected: curl, div, du_dx, du_dy, dv_dx, dv_dy
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            return float(curl), float(div), float(du_dx), float(du_dy), float(dv_dx), float(dv_dy)
        except Exception:
            x, y = float(center_xy[0]), float(center_xy[1])
            base = max(1.0, np.hypot(x, y))
            h = 1e-4 * base

            u0, v0 = _env_eval((x, y))
            u_xp, v_xp = _env_eval((x + h, y))
            u_xm, v_xm = _env_eval((x - h, y))
            u_yp, v_yp = _env_eval((x, y + h))
            u_ym, v_ym = _env_eval((x, y - h))

            du_dx = (u_xp - u_xm) / (2*h)
            du_dy = (u_yp - u_ym) / (2*h)
            dv_dx = (v_xp - v_xm) / (2*h)
            dv_dy = (v_yp - v_ym) / (2*h)

            div  = du_dx + dv_dy
            curl = dv_dx - du_dy
            return float(curl), float(div), float(du_dx), float(du_dy), float(dv_dx), float(dv_dy)

    # --------------------------- prolog logging ------------------------------
    step = int(getattr(cluster, "_cf3_step", 0))
    setattr(cluster, "_cf3_step", step + 1)

    if verbose and step == 0:
        _emit("=" * 76)
        _emit("center_finder3: detailed logging (γ, σ, d̄, w̄, div, ω, cues, subtype, radii)")
        ec = "None(rel)" if eps_cue is None else f"{eps_cue:.3f}"
        ed = "None(off)" if eps_div is None else f"{eps_div:.3f}"
        eo = "None(off)" if eps_om  is None else f"{eps_om:.3f}"
        _emit(f"eps_cue={ec}   eps_div={ed}   eps_om={eo}")
        _emit("=" * 76)

    # --------------------- local probes at current center --------------------
    center_xy = np.asarray(cluster.cluster_centre, dtype=float)

    # Local Jacobian
    curl, div, du_dx, du_dy, dv_dx, dv_dy = _calc_jacobian(center_xy)
    J = np.array([[du_dx, du_dy],
                  [dv_dx, dv_dy]], dtype=float)

    # Aggregate local flow from robots (sum)
    readings = np.array(cluster.bot_readings(), dtype=float)  # (N,2)
    v = readings.sum(axis=0)
    V = float(np.hypot(v[0], v[1]))

    if V < tiny:
        _emit(f"[CF3] step={step:04d}  near-stagnation; |v|={V:.3e}  pos={center_xy.tolist()}")
        return cluster.cluster_centre

    t = v / (V + tiny)          # unit flow direction
    g = J.T @ v                 # = ∇(½||v||²)
    s = J @ v                   # = (v·∇)v
    s_para = (t @ s) * t
    s_perp = s - s_para         # curvature cue (normal acceleration)

    if log_vectors:
        _emit(f"[CF3] step={step:04d}  pos={center_xy.tolist()}  v={v.tolist()}  |v|={V:.4f}")
        _emit(f"      J=[[{J[0,0]:+.4f},{J[0,1]:+.4f}],[{J[1,0]:+.4f},{J[1,1]:+.4f}]]")
        _emit(f"      g={g.tolist()}  s_perp={s_perp.tolist()}")

    # ------------------------ dimensionless cues -----------------------------
    gnorm   = float(np.linalg.norm(g))
    sperpn  = float(np.linalg.norm(s_perp))
    gamma   = gnorm   / (V*V + tiny)   # radial/energy cue
    sigma   = sperpn  / (V*V + tiny)   # curvature cue
    dbar    = abs(float(div))  / (V + tiny)   # radial k-type shortcut
    wbar    = abs(float(curl)) / (V + tiny)   # vortex k-type shortcut

    if (step % max(1, int(log_every))) == 0:
        _emit(f"[CF3] step={step:04d}  γ={gamma:.3f}  σ={sigma:.3f}  d̄={dbar:.3f}  w̄={wbar:.3f}  "
              f"div={div:+.4f}  ω={curl:+.4f}")

    # ---------------------- family (radial vs vortex) ------------------------
    # If eps_cue is None: pick by relative size unless both vanishing
    cue_floor = 1e-6  # only used when eps_cue is None
    if eps_cue is None:
        if max(gamma, sigma) < cue_floor:
            family = 'unobservable'
        else:
            family = 'radial' if gamma >= sigma else 'vortex'
    else:
        if   (gamma >= eps_cue) and (sigma <  eps_cue): family = 'radial'
        elif (sigma >= eps_cue) and (gamma <  eps_cue): family = 'vortex'
        else:                                           family = 'radial' if gamma >= sigma else 'vortex'

    cue_label = "γ" if family == "radial" else ("σ" if family == "vortex" else "-")
    _emit(f"      Linear Class: {family.capitalize():9s}   Cue Label: {cue_label}")

    # ----------------------- subtype (r / k / 1/r) ---------------------------
    is_k_rad = (family == 'radial') and (eps_div is not None) and (gamma < (eps_cue if eps_cue is not None else cue_floor)) and (dbar >= eps_div)
    is_k_vor = (family == 'vortex') and (eps_om  is not None) and (sigma >= (eps_cue if eps_cue is not None else cue_floor)) and (wbar >= eps_om)

    if is_k_rad or is_k_vor:
        subtype = 'k-type'
    elif family == 'radial':
        # Decide r-type vs 1/r-type by divergence presence:
        #   |div| appreciable → r-type (V ~ r); near-zero div → 1/r-type
        if eps_div is None:
            subtype = 'r-type' if dbar >= 1e-3 else '1/r-type'
        else:
            subtype = 'r-type' if dbar >= eps_div else '1/r-type'
    elif family == 'vortex':
        # No hard need to separate p here for direction; keep label via ω presence if desired
        if eps_om is None:
            subtype = 'k-type' if wbar >= 1e-3 and gamma < (eps_cue if eps_cue is not None else cue_floor) else 'r/1r-type'
        else:
            subtype = 'k-type' if is_k_vor else 'r/1r-type'
    else:
        subtype = 'unknown'

    # Optional diagnostic: downstream speed slope
    speed_slope = float(v @ (J @ v)) / (V*V + tiny)  # sign(V dV/ds)
    _emit(f"      Subtype: {subtype:9s}   d|v|/ds ≈ {speed_slope:+.4f}")

    # ----------------------- radius estimators -------------------------------
    r_g   = (V*V) / (gnorm   + tiny)   # r & 1/r families (exact up to |p|)
    r_cur = (V*V) / (sperpn  + tiny)   # curvature/osculating radius
    r_div =  V     / (abs(float(div))  + tiny)   # k-type (radial)
    r_om  =  V     / (abs(float(curl)) + tiny)   # k-type (vortex)

    if subtype == 'k-type':
        r_hat = r_div if (family == 'radial') else r_om
    else:
        r_hat = r_g if (family == 'radial') else r_cur

    _emit(f"      Radii: r_g={r_g:.3f}  r_cur={r_cur:.3f}  r_div={r_div:.3f}  r_om={r_om:.3f}  r̂={r_hat:.3f}")

    # ----------------------- motion direction & step -------------------------
    # Unit helpers
    def _unit(z):
        n = float(np.linalg.norm(z))
        return z / (n + tiny), n

    g_hat,  _ = _unit(g)
    sp_hat, _ = _unit(s_perp)

    # Radial direction by sign-flip rule (no sink/source needed):
    #   r-type   → center_dir = -ĝ  (V increases outward)
    #   1/r-type → center_dir = +ĝ  (V increases inward)
    if family == 'radial':
        if subtype == '1/r-type':
            radial_dir = +g_hat
            radial_tag = "+ĝ"
        else:
            radial_dir = -g_hat
            radial_tag = "-ĝ"
    else:
        # Fallback if someone forces family='vortex' but g≈0:
        radial_dir = -g_hat if gnorm > tiny else -sp_hat
        radial_tag = "-ĝ" if gnorm > tiny else "-ŝ⊥"

    # Curvature tendency: toward center of curvature
    curve_dir = -sp_hat if sigma > 0 else radial_dir

    # Emphasize active family
    w_rad, w_cur = (0.8, 0.2) if (family == 'radial') else (0.2, 0.8)
    step_dir = w_rad * radial_dir + w_cur * curve_dir
    step_dir, _ = _unit(step_dir)

    # Step size: limit to fraction of current radius estimate for stability
    h = float(getattr(cluster, "step_size", 0.1))
    max_scale = 0.5  # at most 50% of radius per step
    h_eff = min(h, max_scale * r_hat) if (np.isfinite(r_hat) and r_hat > tiny) else h

    _emit(f"      Step: h={h:.3f}  h_eff={h_eff:.3f}  dir={[float(step_dir[0]), float(step_dir[1])]}  "
          f"(radial_dir={radial_tag})")

    # Integrate one step
    cluster.cluster_centre[0] += h_eff * step_dir[0]
    cluster.cluster_centre[1] += h_eff * step_dir[1]

    _emit(f"      New pos: {cluster.cluster_centre.tolist()}")
    _emit(f"[CF3] step={step:04d} | fam={family:9s} sub={subtype:9s} | γ={gamma:.3f} σ={sigma:.3f} "
          f"| div={div:+.3f} ω={curl:+.3f} | r̂={r_hat:.3f}")

    return cluster.cluster_centre



import numpy as np

def roi_finder2(
    cluster,
    eps_div=0.05, eps_om=0.05,
    tiny=1e-12,
    verbose=True, log_vectors=False, log_every=1,
    log_path="/mnt/data/roi_finder2_log.txt"
):
    """
    ROI finder specialized for ~constant-magnitude fields (k-type).
    Chooses between:
      • Radial constant-speed (±V r̂):   r = V / |div v|,  dir_to_center = -sign(div)*t
      • Tangential constant-speed (V θ̂): r = V / |ω|,    dir_to_center = -ŝ⊥  (fallback: -sign(ω) * R_cw t)
    """
    def _emit(line: str):
        if not verbose: return
        print(line)
        if log_path:
            try:
                with open(log_path, "a") as f: f.write(line + "\n")
            except Exception:
                pass

    def _env_eval(xy):
        x, y = float(xy[0]), float(xy[1])
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn): return np.array(fn(x, y), dtype=float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m): return np.array(m(x, y), dtype=float)
        R = np.array(cluster.bot_readings(), dtype=float)
        return R.mean(axis=0)

    def _calc_jac(center_xy):
        try:
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            return float(curl), float(div), np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
        except Exception:
            x, y = float(center_xy[0]), float(center_xy[1])
            base = max(1.0, np.hypot(x, y))
            h = 1e-4 * base
            u0, v0 = _env_eval((x, y))
            u_xp, v_xp = _env_eval((x + h, y))
            u_xm, v_xm = _env_eval((x - h, y))
            u_yp, v_yp = _env_eval((x, y + h))
            u_ym, v_ym = _env_eval((x, y - h))
            du_dx = (u_xp - u_xm) / (2*h); du_dy = (u_yp - u_ym) / (2*h)
            dv_dx = (v_xp - v_xm) / (2*h); dv_dy = (v_yp - v_ym) / (2*h)
            div  = du_dx + dv_dy
            curl = dv_dx - du_dy
            return float(curl), float(div), np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)

    def _unit(z):
        n = float(np.linalg.norm(z));  return z/(n+tiny), n

    def _rot_cw(vec):  # rotate by -90°
        x, y = float(vec[0]), float(vec[1])
        return np.array([ y, -x ], float)

    # step counter
    step = int(getattr(cluster, "_roi2_step", 0)); setattr(cluster, "_roi2_step", step + 1)
    if verbose and step == 0:
        _emit("=" * 76)
        _emit("roi_finder2 (constant magnitude): uses div/ω shortcuts; logs d̄, w̄, radii")
        ed = "None(off)" if eps_div is None else f"{eps_div:.3f}"
        eo = "None(off)" if eps_om  is None else f"{eps_om:.3f}"
        _emit(f"eps_div={ed}   eps_om={eo}")
        _emit("=" * 76)

    cxy = np.asarray(cluster.cluster_centre, float)
    readings = np.array(cluster.bot_readings(), float)  # (N,2)
    v = readings.sum(axis=0)
    V = float(np.hypot(v[0], v[1]))
    if V < tiny:
        _emit(f"[ROI2] step={step:04d} near-stagnation |v|={V:.3e} pos={cxy.tolist()}")
        return cluster.cluster_centre

    t = v / (V + tiny)
    curl, div, J = _calc_jac(cxy)
    s = J @ v
    s_para = (t @ s) * t
    s_perp = s - s_para
    _, n_sp = _unit(s_perp)

    dbar = abs(float(div))  / (V + tiny)
    wbar = abs(float(curl)) / (V + tiny)
    r_div = V / (abs(float(div))  + tiny)
    r_om  = V / (abs(float(curl)) + tiny)

    if (step % max(1, int(log_every))) == 0:
        _emit(f"[ROI2] step={step:04d}  d̄={dbar:.3f}  w̄={wbar:.3f}  div={div:+.4f}  ω={curl:+.4f}")

    # Decide which constant-speed model dominates
    choose_radial = (eps_div is not None and dbar >= (eps_div if eps_div is not None else 0)) \
                    and (dbar >= wbar or eps_om is None)

    if not choose_radial and (eps_om is not None and wbar >= eps_om):
        # Tangential constant-speed (vortex-k)
        r_hat = r_om
        if n_sp > tiny:
            dir_c = - s_perp / n_sp
            tag = "-ŝ⊥"
        else:
            dir_c = - np.sign(curl if curl != 0 else 1.0) * _rot_cw(t)
            dir_c, _ = _unit(dir_c)
            tag = f"-sign(ω)·R_cw·t"
        family = "vortex-k"
    elif choose_radial:
        # Radial constant-speed (sink/source-k)
        r_hat = r_div
        dir_c = - np.sign(div if div != 0 else 1.0) * t  # -sign(div)*t
        family = "radial-k"; tag = "-sign(div)·t"
    else:
        # Nothing dominates → treat as unobservable for this model
        _emit(f"[ROI2] step={step:04d} cues weak for k-type; no move.")
        return cluster.cluster_centre

    # Step size clamp
    h = float(getattr(cluster, "step_size", 0.1))
    h_eff = min(h, 0.5 * r_hat) if np.isfinite(r_hat) and r_hat > tiny else h

    if log_vectors:
        _emit(f"      v={v.tolist()} |v|={V:.4f}  J=[[{J[0,0]:+.4f},{J[0,1]:+.4f}],[{J[1,0]:+.4f},{J[1,1]:+.4f}]]")
        _emit(f"      s_perp={s_perp.tolist()}")

    _emit(f"      Family={family:9s}  r_div={r_div:.3f}  r_om={r_om:.3f}  r̂={r_hat:.3f}  "
          f"h={h:.3f} h_eff={h_eff:.3f}  dir={dir_c.tolist()} ({tag})")

    cluster.cluster_centre[0] += h_eff * dir_c[0]
    cluster.cluster_centre[1] += h_eff * dir_c[1]

    _emit(f"[ROI2] step={step:04d} → new_pos={cluster.cluster_centre.tolist()}")
    return cluster.cluster_centre


def rio_finder3(
    cluster,
    tiny=1e-12,
    verbose=True, log_vectors=False, log_every=1,
    log_path="/mnt/data/rio_finder3_log.txt"
):
    """
    RIO finder specialized for near-singularity (1/r) fields.
    Uses exact 1/r identities:
      r_g   = V^2 / ||g||,  with g = J^T v = ∇(½||v||^2)
      r_cur = V^2 / ||s_perp||,  s_perp = Jv - (t^T Jv) t  (osculating radius)
    Direction to center blends inward cues:
      u_c ∝ +ĝ   (inward for 1/r)  and  -ŝ⊥   (inward curvature)
    """
    def _emit(line: str):
        if not verbose: return
        print(line)
        if log_path:
            try:
                with open(log_path, "a") as f: f.write(line + "\n")
            except Exception:
                pass

    def _env_eval(xy):
        x, y = float(xy[0]), float(xy[1])
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn): return np.array(fn(x, y), dtype=float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m): return np.array(m(x, y), dtype=float)
        R = np.array(cluster.bot_readings(), dtype=float)
        return R.mean(axis=0)

    def _calc_jac(center_xy):
        try:
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return float(curl), float(div), J
        except Exception:
            x, y = float(center_xy[0]), float(center_xy[1])
            base = max(1.0, np.hypot(x, y))
            h = 1e-4 * base
            u0, v0 = _env_eval((x, y))
            u_xp, v_xp = _env_eval((x + h, y));  u_xm, v_xm = _env_eval((x - h, y))
            u_yp, v_yp = _env_eval((x, y + h));  u_ym, v_ym = _env_eval((x, y - h))
            du_dx = (u_xp - u_xm)/(2*h); du_dy = (u_yp - u_ym)/(2*h)
            dv_dx = (v_xp - v_xm)/(2*h); dv_dy = (v_yp - v_ym)/(2*h)
            div  = du_dx + dv_dy
            curl = dv_dx - du_dy
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return float(curl), float(div), J

    def _unit(z):
        n = float(np.linalg.norm(z));  return z/(n+tiny), n

    # step counter
    step = int(getattr(cluster, "_rio3_step", 0)); setattr(cluster, "_rio3_step", step + 1)
    if verbose and step == 0:
        _emit("=" * 76)
        _emit("rio_finder3 (near singularity 1/r): uses r_g, r_cur; blends ĝ and -ŝ⊥ for direction")
        _emit("=" * 76)

    cxy = np.asarray(cluster.cluster_centre, float)
    readings = np.array(cluster.bot_readings(), float)
    v = readings.sum(axis=0)
    V = float(np.hypot(v[0], v[1]))
    if V < tiny:
        _emit(f"[RIO3] step={step:04d} near-stagnation |v|={V:.3e} pos={cxy.tolist()}")
        return cluster.cluster_centre

    t = v / (V + tiny)
    curl, div, J = _calc_jac(cxy)
    g = J.T @ v
    s = J @ v
    s_para = (t @ s) * t
    s_perp = s - s_para

    g_hat,  gnorm  = _unit(g)
    sp_hat, spnorm = _unit(s_perp)

    r_g   = (V*V) / (gnorm  + tiny)
    r_cur = (V*V) / (spnorm + tiny)

    # Cue strengths (dimensionless)
    gamma = gnorm  / (V*V + tiny)
    sigma = spnorm / (V*V + tiny)
    w_g = gamma; w_s = sigma
    if (w_g + w_s) < 1e-9:
        # Degenerate (shouldn't happen in true 1/r); do nothing
        _emit(f"[RIO3] step={step:04d} cues ~0; no move.")
        return cluster.cluster_centre
    w_g /= (w_g + w_s); w_s /= (w_g + w_s + 1e-12)  # normalize safely

    # Radius fuse (harmonic mean to be conservative)
    # 1/r models should agree; harmonic mean downweights outliers
    r_hat = 1.0 / ( (w_g/(r_g + tiny)) + (w_s/(r_cur + tiny)) )

    # Direction: inward for 1/r → +ĝ and -ŝ⊥ both point inward; blend for robustness
    dir_c = w_g * g_hat + w_s * (-sp_hat)
    dir_c, _ = _unit(dir_c)

    # Step size clamp
    h = float(getattr(cluster, "step_size", 0.1))
    h_eff = min(h, 0.3 * r_hat) if np.isfinite(r_hat) and r_hat > tiny else h

    if log_vectors:
        _emit(f"      v={v.tolist()} |v|={V:.4f}  g={g.tolist()}  s_perp={s_perp.tolist()}")
        _emit(f"      J=[[{J[0,0]:+.4f},{J[0,1]:+.4f}],[{J[1,0]:+.4f},{J[1,1]:+.4f}]]")

    if (step % max(1, int(log_every))) == 0:
        _emit(f"[RIO3] step={step:04d}  γ={gamma:.3f} σ={sigma:.3f}  r_g={r_g:.3f} r_cur={r_cur:.3f} r̂={r_hat:.3f}  "
              f"div={div:+.4f} ω={curl:+.4f}")

    _emit(f"      Step: h={h:.3f}  h_eff={h_eff:.3f}  dir={dir_c.tolist()}  (blend ĝ & -ŝ⊥)")
    cluster.cluster_centre[0] += h_eff * dir_c[0]
    cluster.cluster_centre[1] += h_eff * dir_c[1]

    _emit(f"[RIO3] step={step:04d} → new_pos={cluster.cluster_centre.tolist()}")
    return cluster.cluster_centre

def roi_finder2_simple(
    cluster,
    mode="auto",          # "auto" | "radial" | "vortex"
    tiny=1e-12, frac=0.5,
    verbose=True, log_vectors=False, log_path="/mnt/data/roi_finder2_simple.log"
):
    """Super-simple k-type finder: pick radial or vortex and use the textbook shortcut."""
    def _emit(msg):
        if verbose: print(msg)
        if log_path:
            try:
                with open(log_path, "a") as f: f.write(msg + "\n")
            except Exception:
                pass

    def _env_eval(xy):
        x, y = map(float, xy)
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn): return np.array(fn(x, y), float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m): return np.array(m(x, y), float)
        R = np.array(cluster.bot_readings(), float)
        return R.mean(axis=0)

    def _calc_J(center_xy):
        try:
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return J, float(div), float(curl)
        except Exception:
            x, y = map(float, center_xy)
            base = max(1.0, np.hypot(x, y)); h = 1e-4 * base
            u_x, v_x = _env_eval((x + h, y)); u_X, v_X = _env_eval((x - h, y))
            u_y, v_y = _env_eval((x, y + h)); u_Y, v_Y = _env_eval((x, y - h))
            du_dx = (u_x - u_X)/(2*h); du_dy = (u_y - u_Y)/(2*h)
            dv_dx = (v_x - v_X)/(2*h); dv_dy = (v_y - v_Y)/(2*h)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return J, float(du_dx + dv_dy), float(dv_dx - du_dy)

    def _rot_cw(z): return np.array([ z[1], -z[0] ], float)

    c = np.asarray(cluster.cluster_centre, float)
    v = np.array(cluster.bot_readings(), float).sum(axis=0)
    V = float(np.hypot(*v))
    if V < tiny:
        _emit(f"[ROI2s] stagnation |v|≈0 at {c.tolist()} — no move"); return cluster.cluster_centre

    J, div, curl = _calc_J(c)
    t = v / (V + tiny)
    s = J @ v; s_perp = s - (t @ s) * t     # optional inward cue for vortex

    # choose simple model
    if mode == "radial":
        choose = "radial"
    elif mode == "vortex":
        choose = "vortex"
    else:
        # auto: whichever invariant is larger in V-units
        dbar = abs(div)/(V + tiny); wbar = abs(curl)/(V + tiny)
        choose = "radial" if dbar >= wbar else "vortex"

    if choose == "radial":
        r_hat = V / (abs(div) + tiny)
        dir_c = -np.sign(div if div != 0 else 1.0) * t
        tag = "radial: r=V/|div|, dir=-sign(div)·t"
    else:
        r_hat = V / (abs(curl) + tiny)
        sp_n = float(np.linalg.norm(s_perp))
        if sp_n > 1e-12:
            dir_c = -s_perp / sp_n
            tag = "vortex: r=V/|ω|, dir=-ŝ⊥"
        else:
            dir_c = -np.sign(curl if curl != 0 else 1.0) * _rot_cw(t)
            tag = "vortex: r=V/|ω|, dir≈-sign(ω)·R_cw·t"

    h = float(getattr(cluster, "step_size", 0.1))
    h_eff = min(h, frac * r_hat) if np.isfinite(r_hat) and r_hat > tiny else h

    if log_vectors:
        _emit(f"[ROI2s] v={v.tolist()} |v|={V:.4f}  J=[[{J[0,0]:+.3f},{J[0,1]:+.3f}],[{J[1,0]:+.3f},{J[1,1]:+.3f}]]")
        _emit(f"         div={div:+.4f}  ω={curl:+.4f}  s⊥={s_perp.tolist()}")
    _emit(f"[ROI2s] {tag}  r̂={r_hat:.3f}  h_eff={h_eff:.3f}")

    cluster.cluster_centre[0] += h_eff * dir_c[0]
    cluster.cluster_centre[1] += h_eff * dir_c[1]
    _emit(f"[ROI2s] → new_pos={cluster.cluster_centre.tolist()}")
    return cluster.cluster_centre


import numpy as np

def rio_finder3_simple(
    cluster,
    tiny=1e-12, frac=0.3,
    verbose=True, log_vectors=False, log_path="/mnt/data/rio_finder3_simple.log"
):
    """Super-simple 1/r finder: dir = +ĝ (fallback -ŝ⊥), r = V^2/||g|| (fallback V^2/||s⊥||)."""
    def _emit(msg):
        if verbose: print(msg)
        if log_path:
            try:
                with open(log_path, "a") as f: f.write(msg + "\n")
            except Exception:
                pass

    def _env_eval(xy):
        x, y = map(float, xy)
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn): return np.array(fn(x, y), float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m): return np.array(m(x, y), float)
        R = np.array(cluster.bot_readings(), float)
        return R.mean(axis=0)

    def _calc_J(center_xy):
        try:
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            return np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float), float(div), float(curl)
        except Exception:
            x, y = map(float, center_xy)
            base = max(1.0, np.hypot(x, y)); h = 1e-4 * base
            u_x, v_x = _env_eval((x + h, y)); u_X, v_X = _env_eval((x - h, y))
            u_y, v_y = _env_eval((x, y + h)); u_Y, v_Y = _env_eval((x, y - h))
            du_dx = (u_x - u_X)/(2*h); du_dy = (u_y - u_Y)/(2*h)
            dv_dx = (v_x - v_X)/(2*h); dv_dy = (v_y - v_Y)/(2*h)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            div = du_dx + dv_dy; curl = dv_dx - du_dy
            return J, float(div), float(curl)

    # values at current centre
    c = np.asarray(cluster.cluster_centre, float)
    v = np.array(cluster.bot_readings(), float).sum(axis=0)
    V = float(np.hypot(*v))
    if V < tiny:
        _emit(f"[RIO3s] stagnation |v|≈0 at {c.tolist()} — no move"); return cluster.cluster_centre

    J, div, curl = _calc_J(c)
    t = v / (V + tiny)
    g = J.T @ v
    s = J @ v
    s_perp = s - (t @ s) * t

    g_n = float(np.linalg.norm(g))
    sp_n = float(np.linalg.norm(s_perp))

    # radius estimate (prefer g; fallback s_perp)
    r_g   = (V*V)/(g_n  + tiny)
    r_cur = (V*V)/(sp_n + tiny)
    use_g = g_n > 1e-12
    r_hat = r_g if use_g else r_cur

    # direction to center
    dir_c = (g / (g_n + tiny)) if use_g else (-(s_perp / (sp_n + tiny)))
    n_dir = float(np.linalg.norm(dir_c)); 
    if n_dir < tiny:
        _emit(f"[RIO3s] cues too small (||g||={g_n:.3e}, ||s⊥||={sp_n:.3e}) — no move")
        return cluster.cluster_centre
    dir_c /= n_dir

    # step
    h = float(getattr(cluster, "step_size", 0.1))
    h_eff = min(h, frac * r_hat) if np.isfinite(r_hat) and r_hat > tiny else h

    if log_vectors:
        _emit(f"[RIO3s] v={v.tolist()} |v|={V:.4f} J=[[{J[0,0]:+.3f},{J[0,1]:+.3f}],[{J[1,0]:+.3f},{J[1,1]:+.3f}]]")
        _emit(f"         g={g.tolist()} ||g||={g_n:.3e}  s⊥={s_perp.tolist()} ||s⊥||={sp_n:.3e}")
    _emit(f"[RIO3s] r_g={r_g:.3f} r_cur={r_cur:.3f} r̂={r_hat:.3f}  div={div:+.3f} ω={curl:+.3f}  h_eff={h_eff:.3f}")

    cluster.cluster_centre[0] += h_eff * dir_c[0]
    cluster.cluster_centre[1] += h_eff * dir_c[1]
    _emit(f"[RIO3s] → new_pos={cluster.cluster_centre.tolist()}")
    return cluster.cluster_centre
