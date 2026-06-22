"""
Scalar field analysis utilities.

Shared functions for gradient/Hessian estimation and Newton's method.
Eliminates code duplication across experiment scripts.
"""
import numpy as np


def compute_gradient(field_func, x, y, h=0.001):
    """
    Numerical gradient via central difference.

    ∇z = [∂z/∂x, ∂z/∂y]

    Args:
        field_func: Scalar field function (x, y) -> z
        x, y: Position
        h: Step size for finite difference

    Returns:
        gradient: [dz_dx, dz_dy]
    """
    dz_dx = (field_func(x + h, y) - field_func(x - h, y)) / (2 * h)
    dz_dy = (field_func(x, y + h) - field_func(x, y - h)) / (2 * h)
    return np.array([dz_dx, dz_dy])


def compute_hessian(field_func, x, y, h=0.001):
    """
    Numerical Hessian matrix via second derivatives.

    H = [[∂²z/∂x², ∂²z/∂x∂y],
         [∂²z/∂y∂x, ∂²z/∂y²]]

    Args:
        field_func: Scalar field function (x, y) -> z
        x, y: Position
        h: Step size for finite difference

    Returns:
        hessian: 2x2 Hessian matrix
    """
    # Second partial derivatives
    d2z_dx2 = (field_func(x + h, y) - 2*field_func(x, y) + field_func(x - h, y)) / h**2
    d2z_dy2 = (field_func(x, y + h) - 2*field_func(x, y) + field_func(x, y - h)) / h**2

    # Mixed partial derivative
    d2z_dxdy = (field_func(x + h, y + h) - field_func(x + h, y - h) -
                field_func(x - h, y + h) + field_func(x - h, y - h)) / (4 * h**2)

    return np.array([[d2z_dx2, d2z_dxdy],
                     [d2z_dxdy, d2z_dy2]])


def compute_newton_step(gradient, hessian, max_step=1.0):
    """
    Newton's method step: Δp = -H⁻¹∇z

    Includes step limiting for stability and singularity handling.

    Args:
        gradient: Gradient vector [gx, gy]
        hessian: 2x2 Hessian matrix
        max_step: Maximum step magnitude (for stability)

    Returns:
        step: Newton step direction [dx, dy]
    """
    # Handle singular/near-singular Hessian
    det = np.linalg.det(hessian)
    if abs(det) < 1e-10:
        # Use pseudoinverse for singular Hessian
        H_inv = np.linalg.pinv(hessian)
    else:
        H_inv = np.linalg.inv(hessian)

    # Newton step: -H⁻¹ * gradient
    step = -H_inv @ gradient

    # Limit step size for stability
    step_magnitude = np.linalg.norm(step)
    if step_magnitude > max_step:
        step = step / step_magnitude * max_step

    return step


def analyze_critical_point(field_func, x, y, h=0.001):
    """
    Full analysis of critical point character.

    Computes gradient, Hessian, eigenvalues, and classifies the point.

    Args:
        field_func: Scalar field function
        x, y: Position to analyze
        h: Finite difference step

    Returns:
        dict with keys:
            - gradient: Gradient vector
            - hessian: Hessian matrix
            - eigenvalues: [λ₁, λ₂]
            - eigenvectors: 2x2 matrix (columns are eigenvectors)
            - classification: 'minimum', 'maximum', 'saddle', or 'degenerate'
            - is_critical: True if |∇z| < 1e-4
    """
    grad = compute_gradient(field_func, x, y, h)
    hess = compute_hessian(field_func, x, y, h)

    # Eigenvalue analysis
    eigenvalues, eigenvectors = np.linalg.eig(hess)

    # Sort eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Classify critical point
    grad_norm = np.linalg.norm(grad)
    is_critical = grad_norm < 1e-4

    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        classification = 'minimum'
    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        classification = 'maximum'
    elif eigenvalues[0] * eigenvalues[1] < 0:
        classification = 'saddle'
    else:
        classification = 'degenerate'

    return {
        'gradient': grad,
        'hessian': hess,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'classification': classification,
        'is_critical': is_critical
    }


class NewtonOptimizer:
    """
    Simple Newton's method optimizer for scalar fields.

    Wraps field + dynamics for cleaner experiment scripts.
    """

    def __init__(self, field_func, alpha=0.7, k=1.0, dt=0.1, h=0.001):
        """
        Args:
            field_func: Scalar field (x,y) -> z
            alpha: Momentum coefficient (0.7 = 70% old velocity)
            k: Velocity gain
            dt: Time step
            h: Finite difference step
        """
        self.field = field_func
        self.alpha = alpha
        self.k = k
        self.dt = dt
        self.h = h

        # State
        self.position = None
        self.velocity = np.array([0.0, 0.0])

    def reset(self, start_pos):
        """Reset to starting position."""
        self.position = np.array(start_pos, dtype=float)
        self.velocity = np.array([0.0, 0.0])

    def step(self):
        """
        Single Newton step with momentum dynamics.

        Returns:
            converged: True if gradient norm < 1e-6
        """
        # Compute Newton step
        grad = compute_gradient(self.field, self.position[0], self.position[1], self.h)
        hess = compute_hessian(self.field, self.position[0], self.position[1], self.h)
        delta_p = compute_newton_step(grad, hess)

        # Velocity command
        v_cmd = self.k * delta_p

        # Momentum dynamics: v_new = α*v_old + (1-α)*v_cmd
        self.velocity = self.alpha * self.velocity + (1 - self.alpha) * v_cmd

        # Update position
        self.position = self.position + self.dt * self.velocity

        # Check convergence
        converged = np.linalg.norm(grad) < 1e-6

        return converged

    def run(self, start_pos, max_steps=1000):
        """
        Run Newton's method until convergence.

        Args:
            start_pos: Starting position [x, y]
            max_steps: Maximum iterations

        Returns:
            trajectory: Nx2 array of positions
        """
        self.reset(start_pos)
        trajectory = [self.position.copy()]

        for _ in range(max_steps):
            converged = self.step()
            trajectory.append(self.position.copy())

            if converged:
                break

        return np.array(trajectory)
