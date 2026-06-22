"""
Saddle point scalar field environments.

These fields feature saddle points - critical points where the Hessian has
eigenvalues of opposite signs. Saddle points are transition regions between
local extrema and are crucial for understanding field topology.

The primary implementation uses bimodal Gaussian fields where two peaks create
a saddle point between them using the log-sum-exp formulation.
"""
import numpy as np
from .SF_bases import log_sum_gaussians, gaussian


# ==============================================================================
# BIMODAL GAUSSIAN SADDLE FIELDS
# ==============================================================================

def bimodal_saddle1(x, y):
    """
    Primary bimodal Gaussian saddle field - THE reference implementation.

    Creates a saddle point at the origin between two Gaussian peaks
    located at (-2, 0) and (2, 0).

    Uses log-sum-exp formulation for numerical stability:
    f(x,y) = log(exp(G1) + exp(G2))
    where G1, G2 are Gaussian exponents centered at (-2,0) and (2,0).

    This is the exact field from Saddle_point.ipynb notebooks and is the
    primary test case for Newton's method saddle point finding.

    Critical point: (0, 0) - saddle point
    Hessian eigenvalues at (0,0): one positive, one negative

    Returns:
        z: Scalar field value
    """
    # Two Gaussians with std=1, centered at (-2,0) and (+2,0)
    gaussian1 = -((x + 2)**2 + y**2) / 2  # Exponent of first Gaussian
    gaussian2 = -((x - 2)**2 + y**2) / 2  # Exponent of second Gaussian

    # Use log-sum-exp for numerical stability
    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


def bimodal_saddle2(x, y):
    """
    Bimodal Gaussian saddle with closer peaks.

    Two Gaussian peaks at (-1, 0) and (1, 0) create a sharper saddle.
    Smaller separation makes the saddle region steeper.

    Critical point: (0, 0) - saddle point

    Returns:
        z: Scalar field value
    """
    # Two Gaussians with std=1, centered at (-1,0) and (+1,0)
    gaussian1 = -((x + 1)**2 + y**2) / 2
    gaussian2 = -((x - 1)**2 + y**2) / 2

    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


def bimodal_saddle3(x, y):
    """
    Bimodal Gaussian saddle with vertical arrangement.

    Two Gaussian peaks at (0, -2) and (0, 2) create a saddle along y-axis.
    Tests saddle finding with different orientation than saddle1/saddle2.

    Critical point: (0, 0) - saddle point

    Returns:
        z: Scalar field value
    """
    # Two Gaussians with std=1, centered at (0,-2) and (0,+2)
    gaussian1 = -(x**2 + (y + 2)**2) / 2
    gaussian2 = -(x**2 + (y - 2)**2) / 2

    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


# ==============================================================================
# QUADRIMODAL SADDLE (Advanced)
# ==============================================================================

def quadrimodal_saddle1(x, y):
    """
    Four Gaussian peaks arranged in square pattern.

    Creates multiple saddle points:
    - Center saddle at (0, 0)
    - Edge saddles between adjacent peaks

    Peaks at: (-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)

    This is an advanced test case for saddle finding with multiple critical points.

    Returns:
        z: Scalar field value
    """
    centers = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)]
    return log_sum_gaussians(x, y, centers, sigma=1.0)


# ==============================================================================
# OFFSET SADDLE FIELDS
# ==============================================================================

def offset_saddle1(x, y):
    """
    Saddle point offset from origin.

    Two Gaussian peaks at (-1.5, 0.5) and (1.5, 0.5) create
    a saddle at (0, 0.5).

    Tests saddle finding when critical point is not at origin.

    Critical point: approximately (0, 0.5)

    Returns:
        z: Scalar field value
    """
    gaussian1 = -((x + 1.5)**2 + (y - 0.5)**2) / 2
    gaussian2 = -((x - 1.5)**2 + (y - 0.5)**2) / 2

    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


# ==============================================================================
# ASYMMETRIC SADDLE FIELDS
# ==============================================================================

def asymmetric_saddle1(x, y):
    """
    Asymmetric saddle with different peak heights.

    One peak has amplitude 1.5, the other has amplitude 1.0.
    Creates tilted saddle point landscape.

    Tests robustness of Newton's method to asymmetric gradients.

    Returns:
        z: Scalar field value
    """
    # Different amplitude Gaussians
    g1 = 1.5 * gaussian(x, y, -2.0, 0.0, 1.0, 1.0)
    g2 = 1.0 * gaussian(x, y, 2.0, 0.0, 1.0, 1.0)

    # Combine using log for smoothness
    return np.log(g1 + g2 + 1e-10)  # Small epsilon to avoid log(0)
