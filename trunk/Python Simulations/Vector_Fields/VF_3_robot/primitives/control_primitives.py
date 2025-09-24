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


def find_sinking_vortex_center(cluster):
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
    
    # Get rotational component
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u
    
    # Get radial component
    if divergence > 0:  # source
        rad_u = -v_hat_u
        rad_v = -v_hat_v
    else:  # sink
        rad_u = v_hat_u
        rad_v = v_hat_v
    
    # Simple weighted combination
    xₖ = cluster.step_size * (w_rot * rot_u + w_rad * rad_u)
    yₖ = cluster.step_size * (w_rot * rot_v + w_rad * rad_v)
    
    # Update cluster position
    cluster.cluster_centre[0] += xₖ
    cluster.cluster_centre[1] += yₖ
    
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