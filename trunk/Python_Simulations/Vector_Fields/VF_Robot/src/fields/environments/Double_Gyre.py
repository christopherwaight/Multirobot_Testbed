"""
Double-gyre vector field, Shadden-Lekien-Marsden (2005) form, shifted to origin.

The canonical Shadden double gyre lives on x in [0,2], y in [0,1] with a
separatrix at x=1.  This module applies a coordinate shift so the field
lives on approximately [-1,1] x [-0.5,0.5], placing the separatrix at x=0
and the corner saddles at (0, -0.5) and (0, 0.5).  That fits the standard
runner plot bounds [-0.65,0.65] without any change to runner.py.

Shift convention:
    x_field = x_world + 1.0
    y_field = y_world + 0.5

Time-varying parameters (loaded from config/fields/double_gyre.yaml):
    A      amplitude (default 0.1)
    eps    perturbation amplitude (default 0.0 -- steady field)
    omega  perturbation angular frequency (default 2*pi/10)

Shadden formulation:
    f(x,t)  = eps*sin(omega*t) * x^2 + (1 - 2*eps*sin(omega*t)) * x
    df/dx   = 2*eps*sin(omega*t) * x + (1 - 2*eps*sin(omega*t))
    u       = -pi*A * sin(pi*f) * cos(pi*y)
    v       =  pi*A * cos(pi*f) * sin(pi*y) * df/dx

When eps=0 this reduces to f=x, df=1, and the field is identical to the
steady reduction used by existing experiments.

Critical points (world coords, steady case):
    Separatrix at x_world = 0   (x_field = 1, the gyre boundary).
    Corner saddles at (0, -0.5) and (0, 0.5).
    Gyre centers at (-0.5, 0) and (0.5, 0).
"""
import numpy as np

# Saddle locations in world (shifted) coordinates
SADDLE_BOTTOM = (0.0, -0.5)
SADDLE_TOP    = (0.0,  0.5)
SEPARATRIX_X  = 0.0

_DEFAULT_A     = 0.1
_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958   # 2*pi/10


def double_gyre_static(x, y, t=0.0, config=None):
    """
    Double-gyre vector field at world position (x, y) and time t.

    Args:
        x: x world coordinate (scalar)
        y: y world coordinate (scalar)
        t: simulation time (seconds; default 0.0)
        config: dict from config/fields/double_gyre.yaml, or None for defaults.

    Returns:
        (u, v): vector field components at (x, y, t)
    """
    cfg = config or {}
    A     = cfg.get("A",     _DEFAULT_A)
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)

    xf = x + 1.0
    yf = y + 0.5

    s  = np.sin(omega * t)
    f  = eps * s * xf * xf + (1.0 - 2.0 * eps * s) * xf
    df = 2.0 * eps * s * xf + (1.0 - 2.0 * eps * s)

    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * yf)
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * yf) * df
    return float(u), float(v)


double_gyre_static.config_name = "double_gyre"
