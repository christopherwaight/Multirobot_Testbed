"""
Scalar field environments for testing gradient-based navigation.

All functions have signature: (x, y) -> phi (returns single scalar value),
except bimodal_gaussian which optionally accepts (x, y, t, config) for
time-varying peak-center wobble.
"""
import numpy as np


_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958


def bimodal_gaussian(x, y, t=0.0, config=None):
    """
    Bimodal Gaussian scalar field from path_quality_analysis.py.
    Has saddle near the origin between two peaks at nominally (-2, 0) and (2, 0).
    Newton's method should find the saddle near (0, 0).

    When eps > 0 in config/fields/bimodal_gaussian.yaml, the two peak centers
    orbit their nominal locations at radius eps and angular frequency omega,
    with opposite phase so the saddle midpoint stays roughly stationary:
        (x1, y1) = (+2 + eps*cos(omega*t), eps*sin(omega*t))
        (x2, y2) = (-2 - eps*cos(omega*t), -eps*sin(omega*t))
    With eps=0 (default), peaks stay at (+/-2, 0) and the field matches the
    original steady form.
    """
    cfg   = config or {}
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)

    cx = eps * np.cos(omega * t)
    cy = eps * np.sin(omega * t)
    x1, y1 = +2.0 + cx, +0.0 + cy
    x2, y2 = -2.0 - cx, -0.0 - cy

    gaussian1 = -((x - x2)**2 + (y - y2)**2) / 2
    gaussian2 = -((x - x1)**2 + (y - y1)**2) / 2
    max_exp = np.maximum(gaussian1, gaussian2)
    return max_exp + np.log(np.exp(gaussian1 - max_exp) + np.exp(gaussian2 - max_exp))


bimodal_gaussian.config_name = "bimodal_gaussian"


def quadratic_bowl(x, y):
    """
    Simple convex quadratic: phi = x^2 + y^2.
    Minimum at origin (0, 0).

    Ideal for testing gradient descent - should converge to (0, 0).
    """
    return x**2 + y**2


def quadratic_peak(x, y):
    """
    Inverted quadratic: phi = -(x^2 + y^2).
    Maximum at origin (0, 0).

    Ideal for testing gradient ascent - should converge to (0, 0).
    """
    return -(x**2 + y**2)


def hyperbolic_saddle(x, y):
    """
    Hyperbolic saddle: phi = x^2 - y^2.
    Saddle point at origin (0, 0).

    Classic test case for Newton's method.
    """
    return x**2 - y**2


def rosenbrock(x, y, a=1.0, b=100.0):
    """
    Rosenbrock function (banana valley).
    Challenging optimization test with narrow curved valley.
    Minimum at (a, a^2) = (1, 1) for default parameters.

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
