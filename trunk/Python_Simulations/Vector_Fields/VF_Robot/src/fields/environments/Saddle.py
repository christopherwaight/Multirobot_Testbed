"""
Saddle vector fields with optional time-varying center wobble.

All three variants (saddle1, saddle2, saddle3) have a nominal center at
(0, 0) and a fixed rotation of pi/4. When loaded with
config/fields/saddle.yaml (eps > 0), the center wobbles:
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


def saddle1(x, y, t=0.0, config=None):
    """Hyperbolic saddle, rotated pi/4, with optional center wobble."""
    center_x, center_y = _wobble_center(t, config)
    x_centered = x - center_x
    y_centered = y - center_y
    strength = 1
    rotation = np.pi / 4

    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    x_rot = cos_rot * x_centered - sin_rot * y_centered
    y_rot = sin_rot * x_centered + cos_rot * y_centered
    x_centered, y_centered = x_rot, y_rot

    u = strength * x_centered
    v = -strength * y_centered

    cos_rot = np.cos(-rotation)
    sin_rot = np.sin(-rotation)
    u_rot = cos_rot * u - sin_rot * v
    v_rot = sin_rot * u + cos_rot * v
    u, v = u_rot, v_rot

    return u, v


def saddle2(x, y, t=0.0, config=None):
    """Hyperbolic saddle with normalized magnitude and optional center wobble."""
    center_x, center_y = _wobble_center(t, config)
    x_centered = x - center_x
    y_centered = y - center_y
    strength = 1
    rotation = np.pi / 4

    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    x_rot = cos_rot * x_centered - sin_rot * y_centered
    y_rot = sin_rot * x_centered + cos_rot * y_centered
    x_centered, y_centered = x_rot, y_rot

    u = strength * x_centered
    v = -strength * y_centered

    cos_rot = np.cos(-rotation)
    sin_rot = np.sin(-rotation)
    u_rot = cos_rot * u - sin_rot * v
    v_rot = sin_rot * u + cos_rot * v
    u, v = u_rot, v_rot

    magnitude = np.sqrt(u**2 + v**2)
    magnitude = np.where(magnitude == 0, 1e-6, magnitude)
    desired_magnitude = 1.0
    u = desired_magnitude * u / magnitude
    v = desired_magnitude * v / magnitude
    return u, v


def saddle3(x, y, t=0.0, config=None):
    """Hyperbolic saddle with 1/r^2 falloff and optional center wobble."""
    center_x, center_y = _wobble_center(t, config)
    x_centered = x - center_x
    y_centered = y - center_y
    strength = 1
    rotation = np.pi / 4

    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    x_rot = cos_rot * x_centered - sin_rot * y_centered
    y_rot = sin_rot * x_centered + cos_rot * y_centered
    x_centered, y_centered = x_rot, y_rot

    r = np.sqrt(x_centered**2 + y_centered**2)
    r = np.where(r == 0, 1e-6, r)

    u = strength * x_centered / (r**2)
    v = -strength * y_centered / (r**2)

    max_magnitude = 10.0
    u = np.clip(u, -max_magnitude, max_magnitude)
    v = np.clip(v, -max_magnitude, max_magnitude)

    cos_rot = np.cos(-rotation)
    sin_rot = np.sin(-rotation)
    u_rot = cos_rot * u - sin_rot * v
    v_rot = sin_rot * u + cos_rot * v
    u, v = u_rot, v_rot

    return u, v


saddle1.config_name = "saddle"
saddle2.config_name = "saddle"
saddle3.config_name = "saddle"
