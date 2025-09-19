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





## Untested
def ride_separatrix_dipole_jacobian(cluster):
    # Calculate Jacobian manually since it's not in cluster yet
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Construct Jacobian matrix
    J = np.array([[du_dx, du_dy],
                  [dv_dx, dv_dy]])
    
    # Get values and vectors
    eigenvals, eigenvecs = np.linalg.eig(J)
    
    # The eigenvector with the largest eigenvalue (in magnitude)
    # points in the direction of strongest flow change
    idx = np.argmax(np.abs(eigenvals))
    separatrix_direction = eigenvecs[:, idx]
    
    # Get current velocity direction to resolve sign ambiguity
    readings = np.array(cluster.bot_readings())
    avg_velocity = np.mean(readings, axis=0)
    velocity_direction = avg_velocity / np.linalg.norm(avg_velocity)
    
    # Make sure we follow separatrix in direction of flow
    if np.dot(separatrix_direction, velocity_direction) < 0:
        separatrix_direction = -separatrix_direction
    
    # Move along separatrix direction
    cluster.cluster_centre[0] += cluster.step_size * separatrix_direction[0]
    cluster.cluster_centre[1] += cluster.step_size * separatrix_direction[1]
    
    return cluster.cluster_centre

def ride_separatrix_dipole_dir2(cluster):
    # Get readings and positions
    readings = np.array(cluster.bot_readings())
    
    # 1. Calculate normalized direction vectors for each robot
    magnitudes = np.sqrt(np.sum(readings**2, axis=1))
    dirs = readings / magnitudes[:, np.newaxis]  # normalize each vector
    angles = np.arctan2(readings[:, 1], readings[:, 0])
    
    # 2. Compute normalized vector sum
    v_sum = np.sum(dirs, axis=0)
    v_sum_magnitude = np.linalg.norm(v_sum)
    if v_sum_magnitude > 1e-6:  # avoid division by zero
        v_sum = v_sum / v_sum_magnitude
    
    # 3. Calculate cosine similarities with vector sum
    similarities = np.array([np.dot(dir_i, v_sum) for dir_i in dirs])
    
    # 4. Identify middle vector (most aligned with sum)
    k = np.argmax(similarities)
    v_mid = dirs[k]
    
    # 5. Calculate cosine similarities of other vectors with middle vector
    other_indices = [i for i in range(3) if i != k]
    sim_mid = np.array([np.dot(dirs[i], v_mid) for i in other_indices])
    
    # 6. Generate perpendicular vectors and determine direction
    v_perp1 = np.array([-v_mid[1], v_mid[0]])  # 90° clockwise
    v_perp2 = np.array([v_mid[1], -v_mid[0]])  # 90° counter-clockwise
    
    # Find vector most different from middle
    i_min = other_indices[np.argmin(sim_mid)]
    
    # Choose perpendicular direction that aligns with the most different vector
    dot1 = np.dot(v_perp1, dirs[i_min])
    dot2 = np.dot(v_perp2, dirs[i_min])
    
    v_command = v_perp1 if dot1 > dot2 else v_perp2
    
    # 7. Move in the command direction
    cluster.cluster_centre[0] += cluster.step_size * v_command[0]
    cluster.cluster_centre[1] += cluster.step_size * v_command[1]
    
    return cluster.cluster_centre



def find_saddle_point(cluster):
    """
    A saddle point finder that uses eigenvalues, eigenvectors, curl, and divergence
    to intelligently move toward the saddle point.
    """
    # Get Jacobian information
    curl_z, divergence, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
    
    # Construct Jacobian matrix
    J = np.array([[du_dx, du_dy],
                  [dv_dx, dv_dy]])
    
    # Calculate determinant - negative for saddle regions
    det_J = du_dx * dv_dy - du_dy * dv_dx
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(J)
    eigenvals = eigenvals.real
    eigenvecs = eigenvecs.real
    
    # Get current velocity
    readings = np.array(cluster.bot_readings())
    avg_vel = np.mean(readings, axis=0)
    vel_mag = np.linalg.norm(avg_vel)
    
    # Check if we're at a stagnation point
    if vel_mag < 1e-4:
        # We might be at the saddle or another critical point
        # Check if it's a saddle by examining eigenvalues
        if eigenvals[0] * eigenvals[1] < 0:
            # It's a saddle - stay here
            return cluster.cluster_centre
    
    # For a saddle in spring+stream field, we're interested in:
    # 1. Regions with negative determinant (where eigenvalues have opposite signs)
    # 2. Direction of contracting eigenvector (negative eigenvalue)
    
    # Find most negative eigenvalue and its eigenvector
    neg_idx = np.argmin(eigenvals)
    contract_vec = eigenvecs[:, neg_idx]
    contract_vec = contract_vec / np.linalg.norm(contract_vec)
    
    # Use additional information to resolve direction ambiguity
    if det_J < 0:
        # We're in a saddle-like region (eigenvalues of opposite signs)
        # Use velocity to determine which way to go
        vel_proj = np.dot(avg_vel, contract_vec)
        
        # For a saddle in a source+stream field, we generally want to move against the flow
        if vel_proj > 0:
            move_dir = -contract_vec
        else:
            move_dir = contract_vec
    else:
        # Not in a saddle region yet - use divergence and curl information
        # For our spring+stream field:
        # - Positive divergence indicates source-like behavior 
        # - We want to move away from high positive divergence (source)
        # - and toward regions of negative determinant
        
        # Get a perpendicular direction to try
        perp_dir1 = np.array([-contract_vec[1], contract_vec[0]])
        perp_dir2 = -perp_dir1
        
        # Make small test steps
        test_dist = 0.1 * cluster.step_size
        
        # Test position in eigenvector direction
        test_pos_eigen = cluster.cluster_centre + test_dist * contract_vec
        u_eigen, v_eigen = cluster.environment_function(test_pos_eigen[0], test_pos_eigen[1])
        
        # Test positions in perpendicular directions
        test_pos_perp1 = cluster.cluster_centre + test_dist * perp_dir1
        u_perp1, v_perp1 = cluster.environment_function(test_pos_perp1[0], test_pos_perp1[1])
        
        test_pos_perp2 = cluster.cluster_centre + test_dist * perp_dir2
        u_perp2, v_perp2 = cluster.environment_function(test_pos_perp2[0], test_pos_perp2[1])
        
        # Check which direction has velocity most opposed to current velocity
        # This often indicates movement toward critical points
        vel_change_eigen = np.dot([u_eigen, v_eigen], avg_vel) / vel_mag
        vel_change_perp1 = np.dot([u_perp1, v_perp1], avg_vel) / vel_mag
        vel_change_perp2 = np.dot([u_perp2, v_perp2], avg_vel) / vel_mag
        
        # Choose direction with most negative change (most opposed to current flow)
        min_change = min(vel_change_eigen, vel_change_perp1, vel_change_perp2)
        
        if min_change == vel_change_eigen:
            move_dir = contract_vec
        elif min_change == vel_change_perp1:
            move_dir = perp_dir1
        else:
            move_dir = perp_dir2
    
    # Move in chosen direction
    xₖ = cluster.step_size * move_dir[0]
    yₖ = cluster.step_size * move_dir[1]
    
    cluster.cluster_centre[0] += xₖ
    cluster.cluster_centre[1] += yₖ
    
    return cluster.cluster_centre





def find_source_spring_fvi(cluster):  
    # This one is a littld different.
    x, y = symbols('x y')
    pos1, pos2, pos3 = cluster.pose()
    _, angles = cluster.bot_compvec()

    # Form equations of lines from angles and positions, angle is with respect to upward direction
    eq1 = Eq((y - pos1[1]) - np.tan(angles[0])*(x - pos1[0]), 0)
    eq2 = Eq((y - pos2[1]) - np.tan(angles[1])*(x - pos2[0]), 0)
    eq3 = Eq((y - pos3[1]) - np.tan(angles[2])*(x - pos3[0]), 0)

    solution12 = solve((eq1,eq2), (x, y))
    solution23 = solve((eq2,eq3), (x, y))
    solution13 = solve((eq1,eq3), (x, y))

    # Average of intersection points
    avg_x = (solution12[x] + solution23[x] + solution13[x]) / 3
    avg_y = (solution12[y] + solution23[y] + solution13[y]) / 3
    avg_x = float(avg_x.evalf())
    avg_y = float(avg_y.evalf())
    
    source_location = [avg_x, avg_y]
    angle_to_source = np.arctan2(cluster.cluster_centre[1] - avg_y,cluster.cluster_centre[0] - avg_x) + np.pi
    
    
    cluster.cluster_centre[0] += cluster.step_size*np.cos( angle_to_source)
    cluster.cluster_centre[1] += cluster.step_size*np.sin( angle_to_source)

    return cluster.cluster_centre






def find_stagnation_sprstr_fvi(cluster): # This entire algorithm needs checking
    _,_,angle = cluster.normal_angle()   

    if (angle >0.2 ):     
        #print()       
        cluster.cluster_centre[0] += cluster.step_size*np.sin(angle)
        cluster.cluster_centre[1] += cluster.step_size*np.cos(angle)
    else:
        angle -= np.pi/2
        cluster.cluster_centre[0] += cluster.step_size*np.sin(angle)
        cluster.cluster_centre[1] += cluster.step_size*np.cos(angle)
    return cluster.cluster_centre

def find_stagnation_sprstr_fvi(cluster):     # Computing Direction of the Bifurcation Line
    
    _, _, angle = cluster.normal_angle() +np.pi
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre

def ride_separatrix_sprstr_fvi(cluster):     # Computing Direction of the Bifurcation Line
    
    _, _, angle = cluster.normal_angle()
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre