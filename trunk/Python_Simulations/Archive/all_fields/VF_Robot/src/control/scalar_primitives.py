"""
Control primitives for 3-robot scalar field navigation.

These algorithms use gradient estimation via plane fitting to navigate
scalar field landscapes toward extrema (maxima, minima) or along features.
"""
import numpy as np


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def estimate_gradient_3robot(cluster):
    """
    Estimate gradient from 3 robot scalar field measurements using plane fitting.

    Fits a plane z = ax + by + c to the three robot measurements.
    The gradient is then ∇z = [a, b].

    Args:
        cluster: OmniCluster instance with 3 robots

    Returns:
        gradient: numpy array [grad_x, grad_y]
        avg_z: Average scalar field value at the three robots
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3 = cluster.get_robot_positions()

    # Sample scalar field at each robot
    z1 = cluster.robots[0].sample_scalar_field(cluster.field)
    z2 = cluster.robots[1].sample_scalar_field(cluster.field)
    z3 = cluster.robots[2].sample_scalar_field(cluster.field)

    # Construct matrix A for plane fitting: z = ax + by + c
    # A * [a, b, c]^T = [z1, z2, z3]^T
    A = np.array([[x1, y1, 1.0],
                  [x2, y2, 1.0],
                  [x3, y3, 1.0]])

    b_vec = np.array([z1, z2, z3])

    try:
        # Solve for plane coefficients [a, b, c]
        coeffs = np.linalg.solve(A, b_vec)
        grad_x = coeffs[0]  # ∂z/∂x
        grad_y = coeffs[1]  # ∂z/∂y
    except np.linalg.LinAlgError:
        # Singular matrix (collinear robots) - gradient undefined
        grad_x = 0.0
        grad_y = 0.0

    gradient = np.array([grad_x, grad_y])
    avg_z = np.mean([z1, z2, z3])

    return gradient, avg_z


# ==============================================================================
# GRADIENT-BASED PRIMITIVES
# ==============================================================================

def gradient_ascent(cluster):
    """
    Move in direction of steepest ascent (hill climbing).

    Estimates gradient from 3-robot plane fit, then commands motion
    in the gradient direction to find local maxima.

    Uses proportional control: velocity scales with gradient magnitude.

    Args:
        cluster: OmniCluster instance

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    gradient, avg_z = estimate_gradient_3robot(cluster)

    grad_magnitude = np.linalg.norm(gradient)

    if grad_magnitude > 1e-6:
        # Proportional control - velocity scales with gradient magnitude
        gain = 1.0
        vx_c = gain * gradient[0]
        vy_c = gain * gradient[1]
    else:
        # At critical point or gradient too small - stop
        vx_c = 0.0
        vy_c = 0.0

    return vx_c, vy_c


def gradient_descent(cluster):
    """
    Move opposite to gradient direction (valley seeking).

    Estimates gradient from 3-robot plane fit, then commands motion
    opposite to the gradient to find local minima.

    Uses proportional control: velocity scales with gradient magnitude.

    Args:
        cluster: OmniCluster instance

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    gradient, avg_z = estimate_gradient_3robot(cluster)

    grad_magnitude = np.linalg.norm(gradient)

    if grad_magnitude > 1e-6:
        # Proportional control - move opposite to gradient
        gain = 1.0
        vx_c = -gain * gradient[0]
        vy_c = -gain * gradient[1]
    else:
        # At critical point or gradient too small - stop
        vx_c = 0.0
        vy_c = 0.0

    return vx_c, vy_c


# ==============================================================================
# REACTIVE SADDLE STATION-KEEPING PRIMITIVES
# ==============================================================================

def field_gradient_balance(cluster):
    """
    Reactive saddle navigation via field gradient balance.

    STRATEGY FROM SADDLE_POINT.IPYNB:
    1. Find steepest direction: from lowest robot to highest robot
    2. Move PERPENDICULAR to this steepest direction
    3. Choose perpendicular that moves toward origin

    RATIONALE:
    - At a saddle point, there are two principal directions:
      * Unstable axis: where field curves like a hill (∩)
      * Stable axis: where field curves like a valley (∪)
    - The steepest direction (min→max) aligns with unstable axis
    - Moving perpendicular follows the stable axis toward saddle center
    - It's like sliding down the "valley" direction of a horse saddle

    Args:
        cluster: OmniCluster instance

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Get robot positions and scalar field values
    positions = np.array([robot.get_position() for robot in cluster.robots])
    z_values = np.array([robot.sample_scalar_field(cluster.field)
                         for robot in cluster.robots])

    # Find direction of steepest change (unstable axis direction)
    max_idx = np.argmax(z_values)
    min_idx = np.argmin(z_values)

    steep_direction = positions[max_idx] - positions[min_idx]
    steep_magnitude = np.linalg.norm(steep_direction)

    if steep_magnitude > 1e-6:
        steep_direction = steep_direction / steep_magnitude

        # Move perpendicular to steep direction (along stable axis)
        perp_direction = np.array([-steep_direction[1], steep_direction[0]])

        # Choose perpendicular direction that moves towards center (origin)
        current_centroid = np.mean(positions, axis=0)
        if np.dot(perp_direction, -current_centroid) < 0:
            perp_direction = -perp_direction

        # Smaller gain for stability
        gain = 0.5
        vx_c = gain * perp_direction[0]
        vy_c = gain * perp_direction[1]
    else:
        # Field is flat - no preferred direction
        vx_c = 0.0
        vy_c = 0.0

    return vx_c, vy_c


def variance_minimizer(cluster):
    """
    Reactive saddle navigation via variance minimization.

    STRATEGY FROM SADDLE_POINT.IPYNB:
    Always move toward robot with median z-value.
    This keeps formation balanced between high/low field regions.

    No threshold - continuous operation.

    Args:
        cluster: OmniCluster instance

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Get robot positions and scalar field values
    positions = np.array([robot.get_position() for robot in cluster.robots])
    z_values = np.array([robot.sample_scalar_field(cluster.field)
                         for robot in cluster.robots])

    current_centroid = np.mean(positions, axis=0)

    # Find robot closest to median value
    median_z = np.median(z_values)
    median_idx = np.argmin(np.abs(z_values - median_z))
    median_robot_pos = positions[median_idx]

    # Move toward median robot
    direction = median_robot_pos - current_centroid
    direction_magnitude = np.linalg.norm(direction)

    if direction_magnitude > 1e-6:
        direction = direction / direction_magnitude
        gain = 0.5
        vx_c = gain * direction[0]
        vy_c = gain * direction[1]
    else:
        vx_c = 0.0
        vy_c = 0.0

    return vx_c, vy_c
