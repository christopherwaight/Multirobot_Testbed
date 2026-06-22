"""
Primitive scalar field functions for composing complex scalar environments.

These are the building blocks for creating various scalar field landscapes
including paraboloids, Gaussians, sinusoids, and other mathematical surfaces.
"""
import numpy as np
from scipy.special import jn, jn_zeros


# ==============================================================================
# PARABOLOID PRIMITIVES
# ==============================================================================

def paraboloid_down(x, y, xc=0.0, yc=0.0, a=1.0):
    """
    Downward-opening paraboloid (maximum at center).

    f(x,y) = -a*((x-xc)^2 + (y-yc)^2)

    Args:
        x, y: Coordinates (scalar or array)
        xc, yc: Center coordinates
        a: Curvature coefficient (larger = steeper)

    Returns:
        z: Scalar field value
    """
    return -a * ((x - xc)**2 + (y - yc)**2)


def paraboloid_up(x, y, xc=0.0, yc=0.0, a=1.0):
    """
    Upward-opening paraboloid (minimum at center).

    f(x,y) = a*((x-xc)^2 + (y-yc)^2)

    Args:
        x, y: Coordinates (scalar or array)
        xc, yc: Center coordinates
        a: Curvature coefficient (larger = steeper)

    Returns:
        z: Scalar field value
    """
    return a * ((x - xc)**2 + (y - yc)**2)


def paraboloid_tilted(x, y, xc=0.0, yc=0.0, ax=1.0, ay=1.0, bx=0.0, by=0.0, c=0.0):
    """
    General quadratic with linear terms.

    f(x,y) = -ax*(x-xc)^2 - ay*(y-yc)^2 + bx*(x-xc) + by*(y-yc) + c

    Args:
        x, y: Coordinates
        xc, yc: Reference point
        ax, ay: Quadratic coefficients
        bx, by: Linear coefficients
        c: Constant offset

    Returns:
        z: Scalar field value
    """
    dx = x - xc
    dy = y - yc
    return -ax * dx**2 - ay * dy**2 + bx * dx + by * dy + c


# ==============================================================================
# GAUSSIAN PRIMITIVES
# ==============================================================================

def gaussian(x, y, xc=0.0, yc=0.0, sigma_x=1.0, sigma_y=1.0, amplitude=1.0):
    """
    2D Gaussian peak.

    f(x,y) = amplitude * exp(-((x-xc)^2/(2*sigma_x^2) + (y-yc)^2/(2*sigma_y^2)))

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        sigma_x, sigma_y: Standard deviations
        amplitude: Peak height

    Returns:
        z: Scalar field value
    """
    exponent = -((x - xc)**2 / (2 * sigma_x**2) + (y - yc)**2 / (2 * sigma_y**2))
    return amplitude * np.exp(exponent)


def log_sum_gaussians(x, y, centers, sigma=1.0):
    """
    Smooth field from log-sum-exp of multiple Gaussians.

    Used for creating saddle points between peaks.
    f(x,y) = log(sum_i exp(G_i))
    where G_i = -((x-x_i)^2 + (y-y_i)^2) / (2*sigma^2)

    Args:
        x, y: Coordinates
        centers: List of (xc, yc) tuples for Gaussian centers
        sigma: Standard deviation for all Gaussians

    Returns:
        z: Scalar field value
    """
    # Compute all Gaussian exponents
    gaussians = []
    for xc, yc in centers:
        g = -((x - xc)**2 + (y - yc)**2) / (2 * sigma**2)
        gaussians.append(g)

    # Use log-sum-exp trick for numerical stability
    gaussians = np.array(gaussians)
    max_g = np.max(gaussians, axis=0)
    return max_g + np.log(np.sum(np.exp(gaussians - max_g), axis=0))


# ==============================================================================
# SINUSOIDAL PRIMITIVES
# ==============================================================================

def sinusoidal_simple(x, y, a=1.0, k=1.0, b=1.0, m=1.0):
    """
    Sum of sinusoids in x and y directions.

    f(x,y) = a*sin(k*x) + b*sin(m*y)

    Args:
        x, y: Coordinates
        a, b: Amplitudes
        k, m: Wavenumbers (frequency parameters)

    Returns:
        z: Scalar field value
    """
    return a * np.sin(k * x) + b * np.sin(m * y)


def sinusoidal_product(x, y, a=1.0, k=1.0, m=1.0):
    """
    Product of sinusoids (creates 2D wave pattern).

    f(x,y) = a*sin(k*x)*sin(m*y)

    Args:
        x, y: Coordinates
        a: Amplitude
        k, m: Wavenumbers

    Returns:
        z: Scalar field value
    """
    return a * np.sin(k * x) * np.sin(m * y)


# ==============================================================================
# RADIAL PRIMITIVES
# ==============================================================================

def radial_linear(x, y, xc=0.0, yc=0.0, a=1.0):
    """
    Linear function of distance from center.

    f(x,y) = a*sqrt((x-xc)^2 + (y-yc)^2)

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        a: Scaling factor

    Returns:
        z: Scalar field value
    """
    return a * np.sqrt((x - xc)**2 + (y - yc)**2)


def radial_inverse(x, y, xc=0.0, yc=0.0, a=1.0, epsilon=1e-6):
    """
    Inverse distance from center (singular at center).

    f(x,y) = a / sqrt((x-xc)^2 + (y-yc)^2 + epsilon)

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        a: Scaling factor
        epsilon: Small value to avoid division by zero

    Returns:
        z: Scalar field value
    """
    r = np.sqrt((x - xc)**2 + (y - yc)**2 + epsilon)
    return a / r


def radial_squared(x, y, xc=0.0, yc=0.0, a=1.0):
    """
    Squared distance from center.

    f(x,y) = a*((x-xc)^2 + (y-yc)^2)

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        a: Scaling factor

    Returns:
        z: Scalar field value
    """
    return a * ((x - xc)**2 + (y - yc)**2)


# ==============================================================================
# ANGULAR PRIMITIVES
# ==============================================================================

def angular_field(x, y, xc=0.0, yc=0.0, a=1.0):
    """
    Phase angle from center (wraps at 2π).

    f(x,y) = a*atan2(y-yc, x-xc)

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        a: Scaling factor

    Returns:
        z: Scalar field value
    """
    return a * np.arctan2(y - yc, x - xc)


# ==============================================================================
# BESSEL FUNCTION PRIMITIVES
# ==============================================================================

def bessel_drumhead(x, y, xc=0.0, yc=0.0, n=1, mode=1, phase=0.0, amplitude=1.0):
    """
    Bessel function field (like vibrating drumhead).

    Models vibration modes of a circular membrane.

    f(r,θ) = amplitude * cos(phase) * cos(n*θ) * J_n(k_nm * r)

    where J_n is the Bessel function of order n,
    and k_nm is the m-th zero of J_n scaled to domain.

    Args:
        x, y: Coordinates
        xc, yc: Center coordinates
        n: Angular mode number (0, 1, 2, ...)
        mode: Radial mode number (1, 2, 3, ...)
        phase: Time phase (0 to 2π)
        amplitude: Scaling factor

    Returns:
        z: Scalar field value
    """
    # Convert to polar coordinates
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    theta = np.arctan2(y - yc, x - xc)

    # Get the zeros of Bessel function
    zeros = jn_zeros(n, mode)
    k_nm = zeros[mode - 1]  # mode-th zero (1-indexed)

    # Compute Bessel drumhead value
    z = amplitude * np.cos(phase) * np.cos(n * theta) * jn(n, k_nm * r)

    return z


# ==============================================================================
# COMPOSITE FIELDS
# ==============================================================================

def gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes):
    """
    Superposition of multiple Gaussians with varying parameters.

    f(x,y) = sum_i amplitude_i * exp(-((x-x_i)^2/(2*sigma_xi^2) + (y-y_i)^2/(2*sigma_yi^2)))

    Args:
        x, y: Coordinates
        centers: List of (xc, yc) tuples
        sigmas: List of (sigma_x, sigma_y) tuples or single (sigma_x, sigma_y) for all
        amplitudes: List of amplitude values or single value for all

    Returns:
        z: Scalar field value
    """
    # Initialize result
    result = np.zeros_like(x, dtype=float)

    # Handle single sigma/amplitude case
    if not isinstance(sigmas, list):
        sigmas = [sigmas] * len(centers)
    if not isinstance(amplitudes, (list, np.ndarray)):
        amplitudes = [amplitudes] * len(centers)

    # Sum contributions from all Gaussians
    for (xc, yc), (sigma_x, sigma_y), amp in zip(centers, sigmas, amplitudes):
        result += gaussian(x, y, xc, yc, sigma_x, sigma_y, amp)

    return result
