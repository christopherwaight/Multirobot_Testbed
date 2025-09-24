# control_primitives.py
import numpy as np
import matplotlib.pyplot as plt

## Helper Functions

def calculate_jacobian(cluster):
    # Get positions and readings
    pos1, pos2, pos3, pos4 = cluster.pose()
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

def calculate_jacobian_bottom(cluster):
    # Get positions and readings
    pos1, pos2, pos3, pos4 = cluster.pose()
    readings = cluster.bot_readings()
    
    # Separate u and v components
    u_readings = np.array([r[0] for r in readings])
    v_readings = np.array([r[1] for r in readings])
    
    # Create position arrays
    pos1_arr = np.array([*pos4, u_readings[3]])  # For u field
    pos2_arr = np.array([*pos2, u_readings[1]])
    pos3_arr = np.array([*pos3, u_readings[2]])
    
    # Calculate normal vector for u field
    R12_u = pos2_arr - pos1_arr
    R13_u = pos3_arr - pos1_arr
    N_u = np.cross(R12_u, R13_u)
    # du/dx and du/dy are proportional to N_u[0] and N_u[1]
    du_dx = N_u[0]
    du_dy = N_u[1]
    
    # Repeat for v field
    pos1_arr = np.array([*pos4, v_readings[3]])
    pos2_arr = np.array([*pos2, v_readings[1]])
    pos3_arr = np.array([*pos3, v_readings[2]])
    
    R12_v = pos2_arr - pos1_arr
    R13_v = pos3_arr - pos1_arr
    N_v = np.cross(R12_v, R13_v)
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