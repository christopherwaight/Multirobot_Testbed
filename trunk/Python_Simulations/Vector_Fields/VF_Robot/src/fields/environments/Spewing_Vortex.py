"""
Spewing vortex vector fields (sign-reversed sinking vortex).

All three variants (spewing_vortex1, spewing_vortex2, spewing_vortex3) have
a nominal center at (0, 0). When loaded with
config/fields/spewing_vortex.yaml (eps > 0), the center wobbles:
    x_cp(t) = eps * cos(omega * t)
    y_cp(t) = eps * sin(omega * t)
With eps=0 (default) the center is fixed at the origin and the output
matches the original steady form.
"""
import numpy as np


_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958


def _wobble_center(t, config):
    cfg   = config or {}
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)
    return eps * np.cos(omega * t), eps * np.sin(omega * t)


def spewing_vortex1(x, y, t=0.0, config=None):
    """
    Spewing vortex field. Combines rotational vortex with radial source.
    Field strength is zero at center and increases with distance.
    """
    center_x, center_y = _wobble_center(t, config)

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u_vortex = -np.sin(theta) * r
    v_vortex =  np.cos(theta) * r

    x_centre = x - center_x
    y_centre = y - center_y

    r_safe = np.where(r == 0, 1e-10, r)
    u_sink = -x_centre / r_safe * r
    v_sink = -y_centre / r_safe * r

    u = 0.4 * u_vortex + 0.15 * u_sink
    v = 0.4 * v_vortex + 0.15 * v_sink
    return -u, -v


def spewing_vortex2(x, y, t=0.0, config=None):
    """
    Spewing vortex field - normalized version. All vectors have equal magnitude.
    """
    center_x, center_y = _wobble_center(t, config)

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u = -np.sin(theta) * r**2
    v =  np.cos(theta) * r**2

    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    u1 = -x_centre / r2**2
    v1 = -y_centre / r2**2

    u = 0.4*u + 0.15*u1
    v = 0.4*v + 0.15*v1

    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.where(magnitude == 0, 1e-10, magnitude)
    desired_magnitude = 1.0
    u = desired_magnitude * u / magnitude
    v = desired_magnitude * v / magnitude
    return -u, -v


def spewing_vortex3(x, y, t=0.0, config=None):
    """
    Spewing vortex field. Combines rotational vortex with radial source.
    """
    center_x, center_y = _wobble_center(t, config)

    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u = -np.sin(theta) * r**2
    v =  np.cos(theta) * r**2

    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    u1 = -x_centre / r2**2
    v1 = -y_centre / r2**2

    u = 0.4*u + 0.15*u1
    v = 0.4*v + 0.15*v1
    return -u, -v


spewing_vortex1.config_name = "spewing_vortex"
spewing_vortex2.config_name = "spewing_vortex"
spewing_vortex3.config_name = "spewing_vortex"
