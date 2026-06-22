"""
Radial scalar field environments.

Distance-based fields including linear, inverse, and squared radial functions.
Simple test cases for radial navigation.
"""
import numpy as np
from .SF_bases import radial_linear, radial_inverse, radial_squared, angular_field


# ==============================================================================
# LINEAR RADIAL FIELDS
# ==============================================================================

def radial_linear1(x, y):
    """
    Simple linear distance from origin.

    f(x,y) = sqrt(x^2 + y^2)

    Increases linearly with distance from (0,0).
    Gradient descent moves toward origin.

    Returns:
        z: Scalar field value
    """
    return radial_linear(x, y, xc=0.0, yc=0.0, a=1.0)


def radial_linear2(x, y):
    """
    Negative linear distance from origin.

    f(x,y) = -sqrt(x^2 + y^2)

    Decreases with distance from origin.
    Gradient ascent moves away from origin.

    Returns:
        z: Scalar field value
    """
    return radial_linear(x, y, xc=0.0, yc=0.0, a=-1.0)


def radial_linear3(x, y):
    """
    Linear distance from offset center.

    f(x,y) = sqrt((x-1)^2 + (y-0.5)^2)

    Center at (1.0, 0.5).

    Returns:
        z: Scalar field value
    """
    return radial_linear(x, y, xc=1.0, yc=0.5, a=1.0)


# ==============================================================================
# INVERSE RADIAL FIELDS
# ==============================================================================

def radial_inverse1(x, y):
    """
    Inverse distance from origin.

    f(x,y) = 1 / sqrt(x^2 + y^2 + epsilon)

    Singular at origin (very sharp peak).
    From Scalar Field notebooks.

    Returns:
        z: Scalar field value
    """
    return radial_inverse(x, y, xc=0.0, yc=0.0, a=1.0)


def radial_inverse2(x, y):
    """
    Negative inverse distance.

    f(x,y) = -1 / sqrt(x^2 + y^2 + epsilon)

    Sharp well at origin.

    Returns:
        z: Scalar field value
    """
    return radial_inverse(x, y, xc=0.0, yc=0.0, a=-1.0)


def radial_inverse3(x, y):
    """
    Offset inverse radial field.

    Center at (0.5, -0.5).

    Returns:
        z: Scalar field value
    """
    return radial_inverse(x, y, xc=0.5, yc=-0.5, a=1.0)


# ==============================================================================
# SQUARED RADIAL FIELDS
# ==============================================================================

def radial_squared1(x, y):
    """
    Squared distance from origin (same as paraboloid_up).

    f(x,y) = x^2 + y^2

    Quadratic growth with distance.

    Returns:
        z: Scalar field value
    """
    return radial_squared(x, y, xc=0.0, yc=0.0, a=1.0)


def radial_squared2(x, y):
    """
    Negative squared distance (same as paraboloid_down).

    f(x,y) = -(x^2 + y^2)

    Quadratic decay with distance.

    Returns:
        z: Scalar field value
    """
    return radial_squared(x, y, xc=0.0, yc=0.0, a=-1.0)


def radial_squared3(x, y):
    """
    Shallow squared radial field.

    f(x,y) = 0.1*(x^2 + y^2)

    Slow quadratic growth.

    Returns:
        z: Scalar field value
    """
    return radial_squared(x, y, xc=0.0, yc=0.0, a=0.1)


# ==============================================================================
# ANGULAR FIELDS
# ==============================================================================

def angular1(x, y):
    """
    Phase angle field.

    f(x,y) = atan2(y, x)

    Wraps from -π to +π around origin.
    Discontinuity along negative x-axis.

    Returns:
        z: Scalar field value
    """
    return angular_field(x, y, xc=0.0, yc=0.0, a=1.0)


def angular2(x, y):
    """
    Scaled angular field.

    f(x,y) = 2*atan2(y, x)

    Larger angular variation.

    Returns:
        z: Scalar field value
    """
    return angular_field(x, y, xc=0.0, yc=0.0, a=2.0)


# ==============================================================================
# COMPOSITE RADIAL FIELDS
# ==============================================================================

def radial_composite1(x, y):
    """
    Combination of linear and squared radial terms.

    f(x,y) = 2*sqrt(x^2 + y^2) - 0.5*(x^2 + y^2)

    Balances linear growth and quadratic decay.

    Returns:
        z: Scalar field value
    """
    r = np.sqrt(x**2 + y**2)
    return 2*r - 0.5*r**2


def radial_composite2(x, y):
    """
    Radial with angular modulation.

    f(x,y) = sqrt(x^2 + y^2) * (1 + 0.3*cos(3*atan2(y,x)))

    Radial field modulated by 3-fold angular pattern.

    Returns:
        z: Scalar field value
    """
    r = np.sqrt(x**2 + y**2 + 1e-10)
    theta = np.arctan2(y, x)
    return r * (1 + 0.3 * np.cos(3 * theta))
