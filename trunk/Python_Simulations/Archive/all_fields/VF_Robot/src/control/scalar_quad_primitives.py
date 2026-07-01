"""
Control primitives for 4-robot scalar field navigation with Hessian estimation.

These algorithms use Newton's method to find critical points including saddle points.
The 4-robot square formation enables Hessian estimation via gradient plane fitting.
"""
import numpy as np
from itertools import combinations


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def estimate_gradient_from_plane(points):
    """
    Estimate gradient from 3 points by fitting a plane.

    Args:
        points: 3x3 array [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]

    Returns:
        (grad_x, grad_y): Gradient components [∂z/∂x, ∂z/∂y]
    """
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(3)])
    b = points[:, 2]

    try:
        # Solve for plane coefficients: z = ax + by + c
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        grad_x = coeffs[0]  # ∂z/∂x
        grad_y = coeffs[1]  # ∂z/∂y
    except np.linalg.LinAlgError:
        grad_x = 0.0
        grad_y = 0.0

    return grad_x, grad_y


def estimate_gradients_4robot(cluster):
    """
    Estimate 4 gradients using all C(4,3)=4 combinations of 3 robots.

    From SADDLE_POINT.IPYNB:
    Uses all possible triangular subsets of the 4-robot square formation.
    Each subset provides a gradient estimate at its centroid.

    Args:
        cluster: QuadCluster instance with 4 robots

    Returns:
        gradients: 4x2 array of gradient estimates [[gx1,gy1], ...]
        centroids: 4x2 array of centroids where gradients are estimated
    """
    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()
    positions = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Sample scalar field at all robots
    z_values = np.array([robot.sample_scalar_field(cluster.field)
                        for robot in cluster.robots])

    # Create 3D points [x, y, z]
    points_3d = np.column_stack([positions, z_values])

    # Estimate gradients using all 4 choose 3 combinations
    gradients = []
    centroids = []

    for combo in combinations(range(4), 3):
        selected_points = points_3d[list(combo)]
        grad_x, grad_y = estimate_gradient_from_plane(selected_points)
        gradients.append([grad_x, grad_y])

        # Centroid of the 3 selected robots
        centroid_3 = np.mean(positions[list(combo)], axis=0)
        centroids.append(centroid_3)

    return np.array(gradients), np.array(centroids)


def estimate_hessian_4robot(centroids, gradients):
    """
    Estimate Hessian matrix from 4 gradient measurements.

    FROM SADDLE_POINT.IPYNB CORRECTED APPROACH:
    Fit planes to gradient components:
    ∂z/∂x = αx·x + βx·y + γx
    ∂z/∂y = αy·x + βy·y + γy

    Hessian = [[αx, βx], [αy, βy]]

    This gives us the second derivatives:
    H = [[∂²z/∂x², ∂²z/∂x∂y], [∂²z/∂y∂x, ∂²z/∂y²]]

    Args:
        centroids: 4x2 array of positions where gradients were estimated
        gradients: 4x2 array of gradient values [∂z/∂x, ∂z/∂y]

    Returns:
        hessian: 2x2 Hessian matrix
    """
    # Extract gradient components
    dz_dx = gradients[:, 0]  # 4 values of ∂z/∂x
    dz_dy = gradients[:, 1]  # 4 values of ∂z/∂y

    # Setup system: ∂z/∂x = αx·x + βx·y + γx
    A = np.column_stack([centroids[:, 0], centroids[:, 1], np.ones(4)])

    try:
        # Solve for ∂z/∂x plane coefficients
        coeffs_dx = np.linalg.lstsq(A, dz_dx, rcond=None)[0]
        d2z_dx2 = coeffs_dx[0]          # ∂²z/∂x²
        d2z_dxdy_from_dx = coeffs_dx[1]  # ∂²z/∂x∂y from ∂z/∂x fit
    except np.linalg.LinAlgError:
        d2z_dx2 = 0.0
        d2z_dxdy_from_dx = 0.0

    try:
        # Solve for ∂z/∂y plane coefficients
        coeffs_dy = np.linalg.lstsq(A, dz_dy, rcond=None)[0]
        d2z_dxdy_from_dy = coeffs_dy[0]  # ∂²z/∂x∂y from ∂z/∂y fit
        d2z_dy2 = coeffs_dy[1]          # ∂²z/∂y²
    except np.linalg.LinAlgError:
        d2z_dxdy_from_dy = 0.0
        d2z_dy2 = 0.0

    # Construct Hessian matrix
    # Note: Could average mixed partials for symmetry, but notebook uses asymmetric form
    hessian = np.array([[d2z_dx2, d2z_dxdy_from_dx],
                       [d2z_dxdy_from_dy, d2z_dy2]])

    return hessian


def compute_newton_step(gradient, hessian):
    """
    Compute Newton's method step direction.

    FROM SADDLE_POINT.IPYNB:
    Step = -H^(-1) * gradient

    Limited to max_step = 1.0 for stability.

    Args:
        gradient: 1D array [gx, gy]
        hessian: 2x2 Hessian matrix

    Returns:
        step: Newton step direction [dx, dy]
    """
    try:
        # Check if Hessian is invertible
        det = np.linalg.det(hessian)
        if abs(det) > 1e-10:
            # Newton's method: step = -H^(-1) * gradient
            step = -np.linalg.solve(hessian, gradient)
        else:
            # Hessian is singular, use pseudoinverse
            step = -np.linalg.pinv(hessian) @ gradient
    except np.linalg.LinAlgError:
        # Fallback to gradient descent if linear algebra fails
        grad_magnitude = np.linalg.norm(gradient)
        if grad_magnitude > 1e-6:
            step = -gradient / grad_magnitude
        else:
            step = np.array([0.0, 0.0])

    # Limit step size for stability
    max_step = 1.0
    step_magnitude = np.linalg.norm(step)
    if step_magnitude > max_step:
        step = step / step_magnitude * max_step

    return step


# ==============================================================================
# NEWTON'S METHOD PRIMITIVES
# ==============================================================================

def newton_saddle_finder(cluster):
    """
    Newton's method for finding saddle points WITHOUT rotation control.

    FROM SADDLE_POINT.IPYNB (Version 1 - no rotation):

    Algorithm:
    1. Estimate 4 gradients using C(4,3) triangle combinations
    2. Estimate Hessian from gradient spatial variation
    3. Compute Newton step: -H^(-1) * gradient
    4. Move in Newton step direction (proportional control)

    Finds critical points where gradient = 0, including saddle points
    where Hessian has eigenvalues of opposite signs.

    Args:
        cluster: QuadCluster instance with 4 robots

    Returns:
        (vx_c, vy_c): Desired centroid velocity
    """
    # Estimate gradients at 4 sub-centroids
    gradients, centroids = estimate_gradients_4robot(cluster)

    # Estimate Hessian from gradient spatial variation
    hessian = estimate_hessian_4robot(centroids, gradients)

    # Average gradient as estimate at cluster center
    avg_gradient = np.mean(gradients, axis=0)

    # Compute Newton step direction
    newton_step = compute_newton_step(avg_gradient, hessian)

    # Proportional control - move in Newton direction
    gain = 1.0
    vx_c = gain * newton_step[0]
    vy_c = gain * newton_step[1]

    return vx_c, vy_c


def newton_saddle_finder_with_rotation(cluster):
    """
    Newton's method for finding saddle points WITH formation rotation control.

    FROM SADDLE_POINT.IPYNB (Version 2 - with rotation control):

    Algorithm:
    1. Estimate gradients and Hessian (same as newton_saddle_finder)
    2. Compute Newton step
    3. ADDITIONALLY: Align square formation with Newton step direction
    4. Return both translation AND rotation velocities

    The rotation control aligns one edge of the square with the Newton
    direction, improving gradient/Hessian estimates along the primary
    approach axis.

    Args:
        cluster: QuadCluster instance with 4 robots

    Returns:
        (vx_c, vy_c, omega_c): Desired centroid velocity and rotation rate
    """
    # Estimate gradients at 4 sub-centroids
    gradients, centroids = estimate_gradients_4robot(cluster)

    # Estimate Hessian from gradient spatial variation
    hessian = estimate_hessian_4robot(centroids, gradients)

    # Average gradient as estimate at cluster center
    avg_gradient = np.mean(gradients, axis=0)

    # Compute Newton step direction
    newton_step = compute_newton_step(avg_gradient, hessian)

    # Translation control - move in Newton direction
    translation_gain = 1.0
    vx_c = translation_gain * newton_step[0]
    vy_c = translation_gain * newton_step[1]

    # Rotation control - align formation with Newton direction
    newton_magnitude = np.linalg.norm(newton_step)
    if newton_magnitude > 1e-6:
        # Calculate angle of Newton step
        newton_angle = np.arctan2(newton_step[1], newton_step[0])

        # Get current formation orientation
        formation = cluster.get_current_formation()
        current_theta = formation['th_c']

        # We want to align one of the square's sides with newton_angle
        # Since square has 4-fold symmetry, align with closest of 0°, 90°, 180°, 270°
        best_error = float('inf')
        for k in range(4):
            target_angle = newton_angle - k * np.pi/2
            error = target_angle - current_theta
            # Normalize to [-π, π]
            error = np.arctan2(np.sin(error), np.cos(error))

            if abs(error) < abs(best_error):
                best_error = error

        # Proportional rotation control
        rotation_gain = 0.3  # From notebook: 0.02 to 0.3 tested
        omega_c = rotation_gain * best_error
    else:
        # Newton step too small - no rotation needed
        omega_c = 0.0

    return vx_c, vy_c, omega_c
