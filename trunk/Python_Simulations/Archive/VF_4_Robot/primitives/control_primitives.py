# control_primitives.py
import numpy as np
import matplotlib.pyplot as plt

## Helper Functions

def calculate_jacobian(cluster):
    """
    From Top 3 Robots
    """
    # Get positions and readings
    pos1, pos2, pos3, pos4 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Print debugging info for robot positions
    print(f"Robot positions - Top triangle:")
    print(f"Robot 1: {pos1}")
    print(f"Robot 2: {pos2}")
    print(f"Robot 3: {pos3}")
    
    # Separate u and v components
    u_readings = np.array([readings[0][0], readings[1][0], readings[2][0]])  # Just robots 1,2,3
    v_readings = np.array([readings[0][1], readings[1][1], readings[2][1]])
    
    print(f"Robot readings - Top triangle:")
    print(f"u_readings: {u_readings}")
    print(f"v_readings: {v_readings}")
    
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
    
    # Print detailed Jacobian information
    print(f"Jacobian components - Top triangle:")
    print(f"du_dx: {du_dx}, du_dy: {du_dy}")
    print(f"dv_dx: {dv_dx}, dv_dy: {dv_dy}")
    print(f"curl_z: {curl_z}, divergence: {divergence}")
    
    return curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy

def calculate_jacobian_bottom(cluster):
    """
    Calculate the Jacobian matrix (gradient of the vector field) using the bottom triangle of robots.
    Returns curl, divergence, and partial derivatives.
    """
    # Get positions and readings
    pos1, pos2, pos3, pos4 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Print debugging info for robot positions
    print(f"Robot positions - Bottom triangle:")
    print(f"Robot 4: {pos4}")
    print(f"Robot 2: {pos2}")
    print(f"Robot 3: {pos3}")
    
    # Separate u and v components - USING ROBOTS 4,2,3 !!!
    u_readings = np.array([readings[3][0], readings[1][0], readings[2][0]])  # Just robots 4,2,3
    v_readings = np.array([readings[3][1], readings[1][1], readings[2][1]])
    
    print(f"Robot readings - Bottom triangle:")
    print(f"u_readings: {u_readings}")
    print(f"v_readings: {v_readings}")
    
    # Create position arrays
    pos4_arr = np.array([*pos4, u_readings[0]])  # For u field - USING POS4 FIRST!
    pos2_arr = np.array([*pos2, u_readings[1]])
    pos3_arr = np.array([*pos3, u_readings[2]])
    
    # Calculate normal vector for u field - DIFFERENT ORDER!
    R42_u = pos2_arr - pos4_arr  # From 4 to 2
    R43_u = pos3_arr - pos4_arr  # From 4 to 3
    N_u = np.cross(R42_u, R43_u)  # Note: order matters for cross product!
    
    # du/dx and du/dy are proportional to N_u[0] and N_u[1]
    du_dx = N_u[0]
    du_dy = N_u[1]
    
    # Repeat for v field
    pos4_arr = np.array([*pos4, v_readings[0]])
    pos2_arr = np.array([*pos2, v_readings[1]])
    pos3_arr = np.array([*pos3, v_readings[2]])
    
    R42_v = pos2_arr - pos4_arr
    R43_v = pos3_arr - pos4_arr
    N_v = np.cross(R42_v, R43_v)
    dv_dx = N_v[0]
    dv_dy = N_v[1]
    
    # Calculate curl and divergence
    curl_z = dv_dx - du_dy  # z-component of curl
    divergence = du_dx + dv_dy
    
    # Print detailed Jacobian information
    print(f"Jacobian components - Bottom triangle:")
    print(f"du_dx: {du_dx}, du_dy: {du_dy}")
    print(f"dv_dx: {dv_dx}, dv_dy: {dv_dy}")
    print(f"curl_z: {curl_z}, divergence: {divergence}")
    
    return curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy

def calculate_center_direction(cluster):
    """
    Calculate the direction vector toward the center using the top triangle measurements.
    Uses both rotational and radial components of the vector field.
    """
    # Get Jacobian information
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Get readings from first three robots
    readings = np.array(cluster.bot_readings())
    sum_u = np.sum(readings[0:3, 0])  # Just first 3 robots
    sum_v = np.sum(readings[0:3, 1])
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)
    
    # Print debugging info
    print(f"Top triangle - total flow: u_sum={sum_u}, v_sum={sum_v}, velocity_magnitude={velocity_magnitude}")
    
    # Check if we're at a stagnation point
    if velocity_magnitude < 1e-6:
        print("Top triangle - stagnation point detected")
        return None
        
    # Calculate unit vector in velocity direction
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude
    print(f"Top triangle - normalized velocity vector: [{v_hat_u}, {v_hat_v}]")
    
    # Calculate weights based on relative strengths
    rot_strength = np.abs(curl_z)
    rad_strength = np.abs(divergence)
    total_strength = rot_strength + rad_strength
    
    if total_strength < 1e-6:
        total_strength = 1e-6  # Prevent division by zero
    
    w_rot = rot_strength / total_strength
    w_rad = rad_strength / total_strength
    
    print(f"Top triangle - component strengths: rotation={rot_strength:.4f}, radial={rad_strength:.4f}")
    print(f"Top triangle - weights: w_rot={w_rot:.4f}, w_rad={w_rad:.4f}")
    
    # Get rotational component (perpendicular to flow)
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u
    
    print(f"Top triangle - rotational direction: [{rot_u}, {rot_v}]")
    
    # Get radial component (against the flow for a sink)
    if divergence > 0:  # source
        rad_u = -v_hat_u
        rad_v = -v_hat_v
    else:  # sink
        rad_u = v_hat_u
        rad_v = v_hat_v
    
    print(f"Top triangle - radial direction: [{rad_u}, {rad_v}]")
    
    # Combine based on relative strengths
    r_hat_u = w_rot * rot_u + w_rad * rad_u
    r_hat_v = w_rot * rot_v + w_rad * rad_v
    
    print(f"Top triangle - combined direction (before normalization): [{r_hat_u}, {r_hat_v}]")
    
    # Normalize the final direction vector
    magnitude = np.sqrt(r_hat_u**2 + r_hat_v**2)
    if magnitude > 1e-6:
        r_hat_u /= magnitude
        r_hat_v /= magnitude
    
    print(f"Top triangle - final direction vector: [{r_hat_u}, {r_hat_v}]")
    

    
    return np.array([r_hat_u, r_hat_v])

def calculate_center_direction_bottom(cluster):
    """
    Calculate the direction vector toward the center using the bottom triangle measurements.
    Uses both rotational and radial components of the vector field.
    """
    # Get Jacobian information from bottom triangle
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian_bottom(cluster)
    
    # Get readings from robots 2, 3, 4
    readings = np.array(cluster.bot_readings())
    indices = [1, 2, 3]  # Robots 2, 3, 4 (0-indexed)
    sum_u = np.sum(readings[indices, 0])
    sum_v = np.sum(readings[indices, 1])
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(sum_u**2 + sum_v**2)
    
    # Print debugging info
    print(f"Bottom triangle - total flow: u_sum={sum_u}, v_sum={sum_v}, velocity_magnitude={velocity_magnitude}")
    
    # Check if we're at a stagnation point
    if velocity_magnitude < 1e-6:
        print("Bottom triangle - stagnation point detected")
        return None
        
    # Calculate unit vector in velocity direction
    v_hat_u = sum_u / velocity_magnitude
    v_hat_v = sum_v / velocity_magnitude
    print(f"Bottom triangle - normalized velocity vector: [{v_hat_u}, {v_hat_v}]")
    
    # Calculate weights based on relative strengths
    rot_strength = np.abs(curl_z)
    rad_strength = np.abs(divergence)
    total_strength = rot_strength + rad_strength
    
    if total_strength < 1e-6:
        total_strength = 1e-6  # Prevent division by zero
    
    w_rot = rot_strength / total_strength
    w_rad = rad_strength / total_strength
    
    print(f"Bottom triangle - component strengths: rotation={rot_strength:.4f}, radial={rad_strength:.4f}")
    print(f"Bottom triangle - weights: w_rot={w_rot:.4f}, w_rad={w_rad:.4f}")
    
    # Get rotational component (perpendicular to flow)
    if curl_z > 0:  # counterclockwise
        rot_u = -v_hat_v
        rot_v = v_hat_u
    else:  # clockwise
        rot_u = v_hat_v
        rot_v = -v_hat_u
    
    print(f"Bottom triangle - rotational direction: [{rot_u}, {rot_v}]")
    
    # Get radial component (against the flow for a sink)
    if divergence > 0:  # source
        rad_u = -v_hat_u
        rad_v = -v_hat_v
    else:  # sink
        rad_u = v_hat_u
        rad_v = v_hat_v
    
    print(f"Bottom triangle - radial direction: [{rad_u}, {rad_v}]")
    
    # Combine based on relative strengths
    r_hat_u = w_rot * rot_u + w_rad * rad_u
    r_hat_v = w_rot * rot_v + w_rad * rad_v
    
    print(f"Bottom triangle - combined direction (before normalization): [{r_hat_u}, {r_hat_v}]")
    
    # Normalize the final direction vector
    magnitude = np.sqrt(r_hat_u**2 + r_hat_v**2)
    if magnitude > 1e-6:
        r_hat_u /= magnitude
        r_hat_v /= magnitude
    
    print(f"Bottom triangle - final direction vector: [{r_hat_u}, {r_hat_v}]")
    
    
    return np.array([r_hat_u, r_hat_v])

def find_line_intersection(p1, d1, p2, d2):
    """
    Finds the intersection point of two lines defined by points and direction vectors.
    
    Parameters:
    p1, p2: Starting points of the lines
    d1, d2: Direction vectors of the lines
    
    Returns:
    Intersection point or None if lines are parallel
    """
    # Check for parallel or coincident lines
    det = d1[0] * d2[1] - d1[1] * d2[0]
    print(f"Line intersection - determinant: {det}")
    
    if abs(det) < 1e-8:  # Determinant close to zero
        print("Lines are parallel or coincident (determinant near zero)")
        return None

    # Line 1: p1 + t1 * d1
    # Line 2: p2 + t2 * d2

    # Solve for t1 and t2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    print(f"Line intersection - matrix A:\n{A}")
    print(f"Line intersection - vector b: {b}")

    try:
        t1, t2 = np.linalg.solve(A, b)
        
        # For debugging, print the parameters
        print(f"Line 1 parameter t1: {t1}")
        print(f"Line 2 parameter t2: {t2}")
        
        # Calculate the intersection point
        intersection1 = p1 + t1 * d1
        intersection2 = p2 + t2 * d2
        
        # These should be very close to each other
        print(f"Intersection from line 1: {intersection1}")
        print(f"Intersection from line 2: {intersection2}")
        
        # Use the average of both calculations for better numerical stability
        intersection = (intersection1 + intersection2) / 2
        
        return intersection
        
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        return None

def estimate_center_and_radius(cluster):
    """
    Estimates the center and radius of a vector field feature by finding the intersection of
    lines defined by two directional vectors from different triangles in the cluster.
    """
    print("\n=== Estimating center and radius ===")
    
    # Get robot positions
    pos1, pos2, pos3, pos4 = cluster.pose()
    
    # Calculate average position for top triangle (robots 1, 2, 3)
    top_triangle_center = np.mean([pos1, pos2, pos3], axis=0)
    print(f"Top triangle center: {top_triangle_center}")
    
    # Calculate average position for bottom triangle (robots 4, 2, 3)
    bottom_triangle_center = np.mean([pos4, pos2, pos3], axis=0)
    print(f"Bottom triangle center: {bottom_triangle_center}")
    
    # Get center directions from both triangles
    print("Calculating top triangle direction...")
    dir1 = calculate_center_direction(cluster)
    
    print("\nCalculating bottom triangle direction...")
    dir2 = calculate_center_direction_bottom(cluster)
    
    # If either direction is None, return None
    if dir1 is None or dir2 is None:
        print("One of the directions is None, cannot estimate center")
        return None, None
    
    # Print angle between directions for debugging
    dot_product = np.dot(dir1, dir2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
    print(f"Angle between directions: {angle} degrees")
    
    # If the vectors are too similar, we'll have numerical instability
    if angle < 5 or angle > 175:
        print(f"Warning: Vectors are nearly parallel or anti-parallel (angle: {angle}°)")
        # Could implement a fallback estimation method here
    
    # Current cluster position - keep for reference
    p0 = cluster.cluster_centre
    print(f"Cluster center: {p0}")
    
    # Find the intersection of the two lines using triangle centers as starting points
    try:
        center_estimate = find_line_intersection(top_triangle_center, dir1, bottom_triangle_center, dir2)
    except Exception as e:
        print(f"Error calculating line intersection: {e}")
        return None, None
    
    if center_estimate is None:
        print("Could not determine intersection, using fallback method")
        # Fallback: use a weighted average of the directions based on robot positions
        combined_dir = (dir1 + dir2) / 2
        combined_dir = combined_dir / np.linalg.norm(combined_dir)
        
        # Use a default distance based on field properties
        default_distance = 3.0
        center_estimate = p0 + default_distance * combined_dir
        estimated_distance = default_distance
        print(f"Fallback center estimate: {center_estimate}")
    else:
        # Calculate the radius (distance from cluster center to estimated center)
        estimated_distance = np.linalg.norm(center_estimate - p0)
        
        # Sanity check - if the distance is unreasonably large, cap it
        max_reasonable_distance = 5.0  # Adjust based on your environment scale
        if estimated_distance > max_reasonable_distance:
            print(f"Warning: Estimated distance ({estimated_distance}) is large, capping at {max_reasonable_distance}")
            unit_vector = (center_estimate - p0) / estimated_distance
            center_estimate = p0 + max_reasonable_distance * unit_vector
            estimated_distance = max_reasonable_distance
    
    # DEBUG: Calculate direct vector to true center (5,5) for comparison
    true_center = np.array([5.0, 5.0])
    true_direction = true_center - p0
    true_distance = np.linalg.norm(true_direction)
    if true_distance > 0:
        true_direction = true_direction / true_distance
    
    print(f"DEBUG - True center: {true_center}")
    print(f"DEBUG - Vector to true center: {true_direction}, distance: {true_distance}")
    print(f"DEBUG - Estimated center: {center_estimate}")
    print(f"DEBUG - Difference from true center: {np.linalg.norm(center_estimate - true_center)}")
    
    print(f"Estimated center: {center_estimate}")
    print(f"Estimated distance to center: {estimated_distance}")
    print("=== Estimation complete ===\n")
    
    # TEMPORARY TEST - Uncomment to use true center instead of estimated
    # print("WARNING: OVERRIDING WITH TRUE CENTER (5,5) FOR TESTING")
    # center_estimate = np.array([5.0, 5.0])
    # estimated_distance = true_distance
    
    return center_estimate, estimated_distance

def dual_jacobian_center_finder(cluster):
    """
    Control primitive that uses both Jacobian calculations to estimate
    the center of the vector field feature and move toward it.
    """

    # Start the Program
    print("\n=== Starting dual Jacobian center finder ===")
    print(f"Initial cluster center: {cluster.cluster_centre}")
    
    # Estimate center and radius
    center_estimate, radius = estimate_center_and_radius(cluster)
    print(f"Center estimate: {center_estimate}, radius: {radius}")
    
    if center_estimate is None:
        # If estimation fails, don't move
        print("Estimation failed - not moving")
        return cluster.cluster_centre
    
    # Calculate direction to estimated center
    direction = center_estimate - cluster.cluster_centre
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-6:
        print("Already at estimated center")
        return cluster.cluster_centre
    
    direction = direction / direction_norm    
    
    # Steps in Direction - Update cluster center
    new_center = cluster.cluster_centre + cluster.step_size* direction
    
    print(f"New cluster center: {new_center}")
    print("=== Finished dual Jacobian center finder ===\n")
    
    return new_center


def center_orbiter(cluster):
    """
    Control primitive that makes the cluster orbit around the estimated center 
    of a vector field feature at a desired radius.
    
    Features:
    - Hard-coded desired radius of 2.0
    - Estimates center position and current radius
    - Calculates perpendicular direction for orbital motion
    - Applies correct inward/outward adjustment to maintain desired radius
    """
    # Hard code desired radius
    des_rad = 1.5
    
    print("\n=== Starting center orbiter ===")
    print(f"Initial cluster center: {cluster.cluster_centre}")
    print(f"Desired orbit radius: {des_rad}")
    
    # Estimate center and current radius
    center_estimate, current_radius = estimate_center_and_radius(cluster)
    print(f"Center estimate: {center_estimate}, current radius: {current_radius}")
    
    if center_estimate is None:
        # If estimation fails, don't move
        print("Center estimation failed - not moving")
        return cluster.cluster_centre
    
    # Calculate vector from cluster to center
    to_center = center_estimate - cluster.cluster_centre
    center_distance = np.linalg.norm(to_center)
    
    if center_distance < 1e-6:
        print("Too close to estimated center - moving outward in random direction")
        # If we're too close to center, move in a random direction
        angle = np.random.uniform(0, 2*np.pi)
        orbit_direction = np.array([np.cos(angle), np.sin(angle)])
        new_center = cluster.cluster_centre + cluster.step_size * orbit_direction
        return new_center
    
    # Normalize direction to center
    to_center_norm = to_center / center_distance
    
    # Calculate perpendicular direction for orbit (90 degrees rotation)
    orbit_direction = np.array([to_center_norm[1], -to_center_norm[0]])

    # Compute the radius error
    radius_error = current_radius - des_rad
    
    # Print initial directions for debugging
    print(f"To center vector: [{to_center_norm[0]:.4f}, {to_center_norm[1]:.4f}]")
    print(f"Orbit direction: [{orbit_direction[0]:.4f}, {orbit_direction[1]:.4f}]")
    
    # Calculate adjustment - SIMPLIFIED APPROACH
    # Start with pure orbit direction
    final_direction = orbit_direction.copy()
    
    # If we're too far out (radius_error > 0), add component toward center
    # If we're too close in (radius_error < 0), add component away from center
    adjustment_strength = min(0.5, abs(radius_error))  # Cap at 0.2 for stability
    
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
    expected_new_distance = np.linalg.norm(new_center - center_estimate)
    print(f"Expected new distance to center: {expected_new_distance:.4f}")
    print(f"New cluster center: {new_center}")
    print("=== Finished center orbiter ===\n")
    
    return new_center