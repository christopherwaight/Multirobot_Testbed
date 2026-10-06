"""Reviewer 3 harness: run the repo's PentagonCluster and primitives on
arbitrary fields, with optional oracle substitution and baseline primitives.

Nothing in the repo is modified. Monkeypatching is local to this process.
"""
import contextlib
import io
import os
import sys

import numpy as np

VFR = "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot"
sys.path.insert(0, VFR)
os.chdir(VFR)

from src.robot.pentagon_cluster import PentagonCluster          # noqa: E402
from src.fields.field_types import AnalyticalField              # noqa: E402
import src.control.pentagon_primitives as pp                    # noqa: E402

FORMATION = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
A_DG = 0.1


# ---------------------------------------------------------------------------
# Fields (world coordinates)
# ---------------------------------------------------------------------------
def dg_static(x, y, A=A_DG):
    X, Y = np.pi*(x + 1.0), np.pi*(y + 0.5)
    return -np.pi*A*np.sin(X)*np.cos(Y), np.pi*A*np.cos(X)*np.sin(Y)


def make_dg_frozen(eps, phase=np.pi/2, A=A_DG):
    """Frozen snapshot of the periodically forced (Shadden) double gyre."""
    s = np.sin(phase)
    def f(x, y):
        xf, yf = x + 1.0, y + 0.5
        fx = eps*s*xf**2 + (1 - 2*eps*s)*xf
        dfx = 2*eps*s*xf + (1 - 2*eps*s)
        return (-np.pi*A*np.sin(np.pi*fx)*np.cos(np.pi*yf),
                np.pi*A*np.cos(np.pi*fx)*np.sin(np.pi*yf)*dfx)
    xs = (-(1 - 2*eps*s) + np.sqrt((1 - 2*eps*s)**2 + 4*eps*s)) / (2*eps*s) if eps > 0 else 1.0
    f.separatrix_x = xs - 1.0
    return f


def make_dg_shear(gamma, A=A_DG):
    """Steady double gyre plus a uniform shear u += gamma*y."""
    def f(x, y):
        u, v = dg_static(x, y, A)
        return u + gamma*y, v
    return f


def make_linear_saddle(lam=1.0):
    """Gradient of the hyperbolic paraboloid phi = lam*(x^2 - y^2)/2."""
    def f(x, y):
        return lam*x, -lam*y
    return f


def make_hp_pass(beta, c, u0):
    """Exactly quadratic, incompressible field whose D is a hyperbolic
    paraboloid: D = beta^2 (y^2 - x^2) - c^2.  Trench along x through the
    origin, crest D_c = -c^2 at the origin, flow +x along the trench."""
    def f(x, y):
        return beta*(x**2 + y**2)/2 + c*y + u0, -beta*x*y + c*x
    return f


class _Field(AnalyticalField):
    def __init__(self, fn):
        super().__init__(fn)
        self._fn = fn

    def get_value(self, x, y):
        return self._fn(x, y)


def new_cluster(fn, x0, y0, heading=0.0, sigma_uv=0.0, sigma_p=0.0, alpha=0.7):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(FORMATION, _Field(fn), momentum_alpha=alpha)
    cl.reset(x0, y0, heading_offset=heading)
    cl.measurement_noise_std = sigma_uv
    cl.position_noise_std = sigma_p
    return cl


# ---------------------------------------------------------------------------
# Primitives (repo code, plus baselines and an oracle hook)
# ---------------------------------------------------------------------------
def prim_D(c):
    vx, vy = pp.separatrix_logic_c_step(c, v_max=V_MAX, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
    return vx*GAIN, vy*GAIN


def prim_s1(c):
    vx, vy = pp.oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP, s_trim=S_TRIM,
                                     r_band=R_BAND, g_capture=G_CAPTURE, s_capture=None)
    return vx*GAIN, vy*GAIN


def prim_flow(c):
    """Baseline: follow the fitted centroid flow (a1, b1), vector-saturated."""
    u_arr, v_arr = pp._sample_vector_at_robots(c)
    tu, tv = pp._fit_vector_quadratic(pp._get_relative_positions(c), u_arr, v_arr)
    f = np.array([tu[0], tv[0]])
    n = np.linalg.norm(f)
    if n < 1e-12:
        return 0.0, 0.0
    sc = V_MAX*np.tanh(n/V_MAX)/n
    return f[0]*sc*GAIN, f[1]*sc*GAIN


_ORACLE = {"H": None, "pos": None}
_orig_det_hessian = pp._det_hessian


def _patched_det_hessian(theta_u, theta_v):
    if _ORACLE["H"] is not None:
        return _ORACLE["H"](*_ORACLE["pos"])
    return _orig_det_hessian(theta_u, theta_v)


pp._det_hessian = _patched_det_hessian


def make_prim_D_oracle(H_true_fn):
    """D tracker with the fitted Hessian replaced by the true one at the centroid."""
    def prim(c):
        _ORACLE["H"] = H_true_fn
        _ORACLE["pos"] = tuple(c.get_centroid())
        try:
            return prim_D(c)
        finally:
            _ORACLE["H"] = None
    return prim


def H_true_dg(x, y, A=A_DG):
    X, Y = np.pi*(x + 1.0), np.pi*(y + 0.5)
    return 2*np.pi**6*A**2*np.diag([np.cos(2*X), np.cos(2*Y)])


def make_prim_s1_resign():
    """Ablation: s1 tracker whose tangent sign is re-taken from the measured
    flow every cycle (the D tracker's rule) instead of carried in state."""
    def prim(c):
        c._oecs_prev_tangent = None          # forget the carried sign
        return prim_s1(c)
    return prim


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(fn, prim, x0, y0, steps=600, heading=0.0, sigma_uv=0.0, sigma_p=0.0,
        stop=None, seed=None, alpha=0.7):
    if seed is not None:
        np.random.seed(seed)
    cl = new_cluster(fn, x0, y0, heading, sigma_uv, sigma_p, alpha)
    for k in range(steps):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if stop is not None and stop(k, cx, cy):
            break
    return cl.get_center_history(), cl.get_robot_history(), cl
