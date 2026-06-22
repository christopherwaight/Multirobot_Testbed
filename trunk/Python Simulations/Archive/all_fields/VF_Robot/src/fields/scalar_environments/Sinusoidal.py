"""
Sinusoidal scalar field environments.

Periodic landscapes with multiple critical points (maxima and minima).
Tests navigation algorithms in environments with many local extrema.
"""
import numpy as np
from .SF_bases import sinusoidal_simple, sinusoidal_product


# ==============================================================================
# SIMPLE SINUSOIDAL FIELDS
# ==============================================================================

def sinusoidal1(x, y):
    """
    Sum of sinusoids - from Scalar Field notebooks.

    f(x,y) = 2*sin(x) + 4*sin(0.8*y)

    Multiple local maxima and minima create complex landscape.
    Tests ability to find nearest critical point.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_simple(x, y, a=2.0, k=1.0, b=4.0, m=0.8)


def sinusoidal2(x, y):
    """
    High frequency sinusoidal field.

    f(x,y) = sin(3*x) + sin(3*y)

    Higher frequency creates more densely packed critical points.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_simple(x, y, a=1.0, k=3.0, b=1.0, m=3.0)


def sinusoidal3(x, y):
    """
    Asymmetric sinusoidal field.

    f(x,y) = 3*sin(2*x) + sin(y)

    Different amplitudes and frequencies in x and y directions.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_simple(x, y, a=3.0, k=2.0, b=1.0, m=1.0)


# ==============================================================================
# PRODUCT SINUSOIDAL FIELDS (2D Wave Patterns)
# ==============================================================================

def wave_pattern1(x, y):
    """
    Product of sinusoids - creates 2D wave pattern.

    f(x,y) = sin(x)*sin(y)

    Regular grid of maxima, minima, and saddle points.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_product(x, y, a=1.0, k=1.0, m=1.0)


def wave_pattern2(x, y):
    """
    High frequency wave pattern.

    f(x,y) = 2*sin(2*x)*sin(2*y)

    Denser grid of critical points.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_product(x, y, a=2.0, k=2.0, m=2.0)


def wave_pattern3(x, y):
    """
    Asymmetric wave pattern.

    f(x,y) = sin(3*x)*sin(y)

    Different frequencies create rectangular wave pattern.

    Returns:
        z: Scalar field value
    """
    return sinusoidal_product(x, y, a=1.0, k=3.0, m=1.0)


# ==============================================================================
# COMPOSITE SINUSOIDAL FIELDS
# ==============================================================================

def composite_sin1(x, y):
    """
    Combination of sum and product terms.

    f(x,y) = sin(x) + sin(y) + 0.5*sin(x)*sin(y)

    Mixes regular and modulated patterns.

    Returns:
        z: Scalar field value
    """
    return np.sin(x) + np.sin(y) + 0.5 * np.sin(x) * np.sin(y)


def composite_sin2(x, y):
    """
    Multi-frequency sinusoidal field.

    f(x,y) = sin(x) + 0.5*sin(2*x) + sin(y) + 0.3*sin(3*y)

    Multiple harmonics create complex landscape.

    Returns:
        z: Scalar field value
    """
    return np.sin(x) + 0.5 * np.sin(2*x) + np.sin(y) + 0.3 * np.sin(3*y)
