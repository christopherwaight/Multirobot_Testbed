"""
Bessel function scalar field environments.

Models vibration modes of circular membranes (like drumheads).
Features complex radial and angular patterns.
"""
import numpy as np
from .SF_bases import bessel_drumhead


# ==============================================================================
# BESSEL DRUMHEAD MODES
# ==============================================================================

def drumhead_mode_01(x, y):
    """
    Fundamental drumhead mode (0,1).

    Circular symmetric vibration - no angular variation.
    Single maximum at center, zero at boundary.

    Mode (n=0, m=1): Radially symmetric, first radial node.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=0, mode=1,
                           phase=0.0, amplitude=1.0)


def drumhead_mode_11(x, y):
    """
    Drumhead mode (1,1).

    One angular node (splits in two lobes) and first radial node.
    Creates cos(θ) angular pattern.

    Mode (n=1, m=1): One angular variation.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=1, mode=1,
                           phase=0.0, amplitude=1.0)


def drumhead_mode_21(x, y):
    """
    Drumhead mode (2,1).

    Two angular nodes (splits in four lobes) and first radial node.
    Creates cos(2θ) angular pattern.

    Mode (n=2, m=1): Two angular variations.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=2, mode=1,
                           phase=0.0, amplitude=1.0)


def drumhead_mode_02(x, y):
    """
    Drumhead mode (0,2).

    Circular symmetric with two radial nodes.
    Ring pattern with alternating positive/negative regions.

    Mode (n=0, m=2): Radially symmetric, second radial node.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=0, mode=2,
                           phase=0.0, amplitude=1.0)


def drumhead_mode_31(x, y):
    """
    Drumhead mode (3,1).

    Three angular nodes (six lobes) and first radial node.
    Creates cos(3θ) angular pattern.

    Mode (n=3, m=1): Three angular variations.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=3, mode=1,
                           phase=0.0, amplitude=1.0)


def drumhead_mode_12(x, y):
    """
    Drumhead mode (1,2).

    One angular node and two radial nodes.
    Complex pattern with rings and angular splits.

    Mode (n=1, m=2): One angular variation, second radial node.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=1, mode=2,
                           phase=0.0, amplitude=1.0)


# ==============================================================================
# OFFSET BESSEL PATTERNS
# ==============================================================================

def offset_drumhead1(x, y):
    """
    Drumhead mode (1,1) offset from origin.

    Tests Bessel patterns centered away from origin.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.5, yc=0.3, n=1, mode=1,
                           phase=0.0, amplitude=1.0)


# ==============================================================================
# TIME-VARYING BESSEL (Phase variations)
# ==============================================================================

def drumhead_phase1(x, y):
    """
    Drumhead mode (1,1) with phase = π/4.

    Time-snapshot of vibrating membrane.

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=1, mode=1,
                           phase=np.pi/4, amplitude=1.0)


def drumhead_phase2(x, y):
    """
    Drumhead mode (2,1) with phase = π/2.

    Different time-snapshot of mode (2,1).

    Returns:
        z: Scalar field value
    """
    return bessel_drumhead(x, y, xc=0.0, yc=0.0, n=2, mode=1,
                           phase=np.pi/2, amplitude=1.0)
