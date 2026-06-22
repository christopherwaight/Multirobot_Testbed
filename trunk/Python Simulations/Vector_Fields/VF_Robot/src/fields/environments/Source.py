"""
Source vector fields.

All three variants (source1, source2, source3) have a nominal center at
(0, 0). When loaded with config/fields/source.yaml (eps > 0), the center
wobbles:
    x_cp(t) = eps * cos(omega * t)
    y_cp(t) = eps * sin(omega * t)
With eps=0 (default) the center is fixed at the origin.
"""
import numpy as np


_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958


def _wobble_center(t, config):
    cfg   = config or {}
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)
    return eps * np.cos(omega * t), eps * np.sin(omega * t)


def source1(x, y, t=0.0, config=None):
    x_src, y_src = _wobble_center(t, config)
    x_centre = x - x_src
    y_centre = y - y_src

    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    strength = -1.0
    v_r = strength * r

    u = -v_r * x_centre / r
    v = -v_r * y_centre / r

    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    return u, v


def source2(x, y, t=0.0, config=None):
    x_src, y_src = _wobble_center(t, config)
    x_centre = x - x_src
    y_centre = y - y_src

    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    strength = 1.0
    u = strength * x_centre / r
    v = strength * y_centre / r
    return u, v


def source3(x, y, t=0.0, config=None):
    center_x, center_y = _wobble_center(t, config)

    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    u = 0.15 * x_centre / r2**2
    v = 0.15 * y_centre / r2**2
    return u, v


source1.config_name = "source"
source2.config_name = "source"
source3.config_name = "source"
