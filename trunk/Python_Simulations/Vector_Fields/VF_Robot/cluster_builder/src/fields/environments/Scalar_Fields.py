"""
Scalar field environments for testing gradient-based navigation.

All functions have signature: (x, y) -> φ (returns single scalar value)
"""
import numpy as np


def bimodal_gaussian(x, y):
    """
    Bimodal Gaussian scalar field from path_quality_analysis.py.
    Has saddle at origin between two peaks at (-2, 0) and (2, 0).

    This is the field from the rotation control paper research.
    Newton's method should find the saddle at (0, 0).
    """
    gaussian1 = -((x + 2)**2 + y**2) / 2
    gaussian2 = -((x - 2)**2 + y**2) / 2
    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


def quadratic_bowl(x, y):
    """
    Simple convex quadratic: φ = x² + y².
    Minimum at origin (0, 0).

    Ideal for testing gradient descent - should converge to (0, 0).
    """
    return x**2 + y**2


def quadratic_peak(x, y):
    """
    Inverted quadratic: φ = -(x² + y²).
    Maximum at origin (0, 0).

    Ideal for testing gradient ascent - should converge to (0, 0).
    """
    return -(x**2 + y**2)


def hyperbolic_saddle(x, y):
    """
    Hyperbolic saddle: φ = x² - y².
    Saddle point at origin (0, 0).

    Classic test case for Newton's method.
    """
    return x**2 - y**2


def rosenbrock(x, y, a=1.0, b=100.0):
    """
    Rosenbrock function (banana valley).
    Challenging optimization test with narrow curved valley.
    Minimum at (a, a²) = (1, 1) for default parameters.

    Tests robustness of optimization algorithms.
    """
    return (a - x)**2 + b * (y - x**2)**2


def himmelblau(x, y):
    """
    Himmelblau's function - multimodal optimization problem.
    Has 4 local minima and 1 saddle point.

    Local minima at:
    - (3, 2) with f = 0
    - (-2.805, 3.131) with f = 0
    - (-3.779, -3.283) with f = 0
    - (3.584, -1.848) with f = 0

    Saddle point at (-0.270845, -0.923039)

    Good test for multi-modal navigation.
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
