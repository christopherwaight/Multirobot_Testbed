"""
Gaussian RBF (Radial Basis Function) scalar field environments.

Superposition of multiple Gaussians arranged in patterns to create
linear ridges, circular ridges, and other complex landscapes.
"""
import numpy as np
from .SF_bases import gaussian_rbf_superposition, gaussian


# ==============================================================================
# LINEAR RIDGE PATTERNS
# ==============================================================================

def linear_ridge1(x, y):
    """
    Linear ridge of 10 Gaussians arranged vertically.

    Gaussians at y = -1.5 to +1.5, all at x = 0.
    Creates a linear ridge along the y-axis.

    From Scalar Field notebooks - tests following along extended feature.

    Returns:
        z: Scalar field value
    """
    # 10 Gaussians arranged vertically
    y_positions = np.linspace(-1.5, 1.5, 10)
    centers = [(0.0, y_pos) for y_pos in y_positions]
    sigmas = (0.2, 0.2)  # Narrow Gaussians
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


def linear_ridge2(x, y):
    """
    Horizontal linear ridge.

    Gaussians arranged along x-axis.

    Returns:
        z: Scalar field value
    """
    x_positions = np.linspace(-1.5, 1.5, 10)
    centers = [(x_pos, 0.0) for x_pos in x_positions]
    sigmas = (0.2, 0.2)
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


def linear_ridge3(x, y):
    """
    Diagonal linear ridge.

    Gaussians arranged along diagonal line.

    Returns:
        z: Scalar field value
    """
    positions = np.linspace(-1.2, 1.2, 10)
    centers = [(pos, pos) for pos in positions]
    sigmas = (0.2, 0.2)
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


# ==============================================================================
# CIRCULAR RIDGE PATTERNS
# ==============================================================================

def circular_ridge1(x, y):
    """
    Circular ridge of 16 Gaussians arranged radially.

    From Scalar Field notebooks - creates ring-shaped ridge.
    Tests circular orbit behavior.

    Returns:
        z: Scalar field value
    """
    # 16 Gaussians arranged in circle of radius 1.0
    n_gaussians = 16
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, n_gaussians, endpoint=False)

    centers = [(radius * np.cos(angle), radius * np.sin(angle))
               for angle in angles]
    sigmas = (0.15, 0.15)
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


def circular_ridge2(x, y):
    """
    Large circular ridge.

    Wider circle with more Gaussians for smoother ridge.

    Returns:
        z: Scalar field value
    """
    n_gaussians = 24
    radius = 1.5
    angles = np.linspace(0, 2*np.pi, n_gaussians, endpoint=False)

    centers = [(radius * np.cos(angle), radius * np.sin(angle))
               for angle in angles]
    sigmas = (0.2, 0.2)
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


def circular_ridge3(x, y):
    """
    Small, dense circular ridge.

    Tighter circle with narrower Gaussians.

    Returns:
        z: Scalar field value
    """
    n_gaussians = 12
    radius = 0.6
    angles = np.linspace(0, 2*np.pi, n_gaussians, endpoint=False)

    centers = [(radius * np.cos(angle), radius * np.sin(angle))
               for angle in angles]
    sigmas = (0.1, 0.1)
    amplitudes = 1.0

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


# ==============================================================================
# RANDOM GAUSSIAN FIELDS
# ==============================================================================

def random_gaussians1(x, y):
    """
    Random scatter of Gaussians.

    Creates complex landscape with multiple local maxima.

    Returns:
        z: Scalar field value
    """
    np.random.seed(42)  # Reproducible random field
    n_gaussians = 20
    centers = [(np.random.uniform(-2, 2), np.random.uniform(-2, 2))
               for _ in range(n_gaussians)]
    sigmas = [(0.3, 0.3)] * n_gaussians
    amplitudes = np.random.uniform(0.5, 1.5, n_gaussians)

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)


def random_gaussians2(x, y):
    """
    Denser random Gaussian field.

    More Gaussians with varying sizes.

    Returns:
        z: Scalar field value
    """
    np.random.seed(123)
    n_gaussians = 30
    centers = [(np.random.uniform(-2, 2), np.random.uniform(-2, 2))
               for _ in range(n_gaussians)]

    # Varying sigma values
    sigmas = [(np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4))
              for _ in range(n_gaussians)]
    amplitudes = np.random.uniform(0.3, 1.2, n_gaussians)

    return gaussian_rbf_superposition(x, y, centers, sigmas, amplitudes)
