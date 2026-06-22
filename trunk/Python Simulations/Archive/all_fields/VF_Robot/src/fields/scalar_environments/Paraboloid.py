"""
Paraboloid scalar field environments.

Simple quadratic bowl-shaped fields for testing gradient descent/ascent.
These are the easiest test cases with single global extremum.
"""
import numpy as np
from .SF_bases import paraboloid_down, paraboloid_up, paraboloid_tilted


# ==============================================================================
# DOWNWARD PARABOLOIDS (Maxima)
# ==============================================================================

def paraboloid_max1(x, y):
    """
    Simple downward paraboloid with maximum at origin.

    f(x,y) = -(x^2 + y^2)

    Critical point: (0, 0) - maximum
    Use gradient ascent to find the peak.

    Returns:
        z: Scalar field value
    """
    return paraboloid_down(x, y, xc=0.0, yc=0.0, a=1.0)


def paraboloid_max2(x, y):
    """
    Shallow downward paraboloid - slower convergence test.

    f(x,y) = 0.05 - 0.001*(x^2 + y^2)

    Critical point: (0, 0) - maximum
    Small curvature tests patience of gradient ascent.

    Returns:
        z: Scalar field value
    """
    return 0.05 - 0.001 * (x**2 + y**2)


def paraboloid_max3(x, y):
    """
    Offset downward paraboloid.

    f(x,y) = -(( x-1)^2 + (y-0.5)^2)

    Critical point: (1.0, 0.5) - maximum
    Tests finding maximum away from origin.

    Returns:
        z: Scalar field value
    """
    return paraboloid_down(x, y, xc=1.0, yc=0.5, a=1.0)


# ==============================================================================
# UPWARD PARABOLOIDS (Minima)
# ==============================================================================

def paraboloid_min1(x, y):
    """
    Simple upward paraboloid with minimum at origin.

    f(x,y) = x^2 + y^2

    Critical point: (0, 0) - minimum
    Use gradient descent to find the valley.

    Returns:
        z: Scalar field value
    """
    return paraboloid_up(x, y, xc=0.0, yc=0.0, a=1.0)


def paraboloid_min2(x, y):
    """
    Steep upward paraboloid - fast convergence test.

    f(x,y) = 2*(x^2 + y^2)

    Critical point: (0, 0) - minimum
    Large curvature leads to rapid convergence.

    Returns:
        z: Scalar field value
    """
    return paraboloid_up(x, y, xc=0.0, yc=0.0, a=2.0)


def paraboloid_min3(x, y):
    """
    Offset upward paraboloid.

    f(x,y) = (x+0.5)^2 + (y-1.0)^2

    Critical point: (-0.5, 1.0) - minimum
    Tests finding minimum away from origin.

    Returns:
        z: Scalar field value
    """
    return paraboloid_up(x, y, xc=-0.5, yc=1.0, a=1.0)


# ==============================================================================
# TILTED PARABOLOIDS (with Linear Terms)
# ==============================================================================

def tilted_paraboloid1(x, y):
    """
    Quadratic bowl with linear tilt.

    f(x,y) = -0.65*x^2 - 3*x - 1.25*y^2 + 0.5*y + 10

    From Scalar Field notebooks - tests gradient descent with
    both quadratic and linear components.

    Critical point: approximately (-2.31, 0.20) - maximum

    Returns:
        z: Scalar field value
    """
    return -0.65 * x**2 - 3 * x - 1.25 * y**2 + 0.5 * y + 10


def tilted_paraboloid2(x, y):
    """
    Asymmetric paraboloid with different curvatures.

    f(x,y) = -x^2 - 2*y^2 + x - 0.5*y

    Different curvatures in x and y directions create elliptical contours.

    Returns:
        z: Scalar field value
    """
    return paraboloid_tilted(x, y, xc=0.0, yc=0.0, ax=1.0, ay=2.0, bx=1.0, by=-0.5, c=0.0)


def tilted_paraboloid3(x, y):
    """
    Complex tilted paraboloid.

    f(x,y) = -0.5*x^2 - 0.8*y^2 + 2*x + y + 5

    Multiple linear and quadratic terms test robustness of gradient estimation.

    Returns:
        z: Scalar field value
    """
    return -0.5 * x**2 - 0.8 * y**2 + 2 * x + y + 5
