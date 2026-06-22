"""
Vortex vector fields.

All three variants (vortex1, vortex2, vortex3) have a nominal center at
(0, 0). When loaded with config/fields/vortex.yaml (eps > 0), the center
wobbles in time:
    x_cp(t) = eps * cos(omega * t)
    y_cp(t) = eps * sin(omega * t)
With eps=0 (default) the center is fixed at the origin and the output
matches the original steady form.
"""
import numpy as np


_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958   # 2*pi/10


def _wobble_center(t, config):
    """Return (x_center, y_center) at time t for the nominal-origin fields."""
    cfg   = config or {}
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)
    return eps * np.cos(omega * t), eps * np.sin(omega * t)


def vortex1(x, y, t=0.0, config=None):
    center_x, center_y = _wobble_center(t, config)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u = -r * np.sin(theta)
    v =  r * np.cos(theta)
    return u, v


def vortex2(x, y, t=0.0, config=None):
    center_x, center_y = _wobble_center(t, config)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u =  np.sin(theta)
    v = -np.cos(theta)
    return u, v


def vortex3(x, y, t=0.0, config=None):
    center_x, center_y = _wobble_center(t, config)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)

    u =  np.sin(theta) / r
    v = -np.cos(theta) / r
    return u, v


vortex1.config_name = "vortex"
vortex2.config_name = "vortex"
vortex3.config_name = "vortex"
