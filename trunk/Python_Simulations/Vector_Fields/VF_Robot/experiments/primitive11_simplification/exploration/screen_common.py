"""Shared setup for the 2026-10-09 D-tracker screen (dnew.py variants)."""
import os, sys, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
VF = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (VF, os.path.join(VF, "experiments"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(VF)
import numpy as np
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.control.pentagon_primitives import separatrix_logic_c_step
import dnew

SAD = [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5)]
VARIANTS = ("repo", "new |l2|", "new signed l2")


def step_fn(variant, log):
    if variant == "repo":
        return lambda c: separatrix_logic_c_step(c, v_max=M.V_MAX)
    if variant == "newton + fallback":
        return lambda c: dnew.d_step(c, M.V_MAX, newton=True, fallback=True, log=log)
    if variant == "descend + |l2| no fallback":
        return lambda c: dnew.d_step(c, M.V_MAX, newton=False, fallback=False, log=log)
    if variant == "merge only":
        return lambda c: dnew.d_step(c, M.V_MAX, merge_only=True, log=log)
    return lambda c: dnew.d_step(c, M.V_MAX, abs_l2=(variant == "new |l2|"), log=log)


def make_cluster(sx, sy, heading=0.0, s_uv=0.0, s_p=0.0):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=heading)
    cl.measurement_noise_std = s_uv
    cl.position_noise_std = s_p
    cl.diagnostics = []
    return cl


def modes_of(cl, log):
    return log if log else [d['mode'] for d in cl.diagnostics if 'mode' in d]


def nearest_saddle(p):
    d = [np.hypot(p[0] - a, p[1] - b) for a, b in SAD]
    k = int(np.argmin(d))
    name = "far" if SAD[k] == (0, -0.5) else ("near" if SAD[k] == (0, 0.5) else "corner")
    return name, d[k]


def run_held(variant, sx, sy, heading=0.0, s_uv=0.0, s_p=0.0, seed=None,
             steps=M.SIM_STEPS, hold=M.HOLD_STEPS, stop_on_hold=True):
    """Paper rule: stop once a saddle has been held `hold` steps within 0.06,
    or at |x| > 1, |y| > Y_EXIT. With stop_on_hold=False, run all steps
    (exit at |x| > 1.3, |y| > 0.8) for the drift check."""
    if seed is not None:
        np.random.seed(seed)
    cl = make_cluster(sx, sy, heading, s_uv, s_p)
    log = []
    fn = step_fn(variant, log)
    prim = lambda c: tuple(M.GAIN * x for x in fn(c))
    held, outcome = 0, "timeout"
    xb, yb = (1.0, M.Y_EXIT) if stop_on_hold else (1.3, 0.8)
    for _ in range(steps):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > xb or abs(cy) > yb:
            outcome = "exited"
            break
        name, d = nearest_saddle((cx, cy))
        held = held + 1 if d < M.SADDLE_CONTACT_D else 0
        if stop_on_hold and held >= hold:
            outcome = name
            break
    return cl, modes_of(cl, log), outcome
