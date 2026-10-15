import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField

def slide_only(c, v_max):
    u, v = P._sample_vector_at_robots(c); rel = P._get_relative_positions(c)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv)
    lam, V = np.linalg.eigh(H)
    c0 = -(V[:, 0] @ g) / max(abs(lam[0]), 1e-12)
    c1 = -(V[:, 1] @ g) / max(abs(lam[1]), 1e-12)
    d = v_max * np.tanh(c0 / v_max) * V[:, 0] + v_max * np.tanh(c1 / v_max) * V[:, 1]
    return float(d[0]), float(d[1])

SAD = {"far (0,-0.5)": (0, -0.5), "near (0,+0.5)": (0, 0.5)}
for name, sx, sy in M.STARTS:
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy)
    prim = lambda c: tuple(M.GAIN * x for x in slide_only(c, M.V_MAX))
    for k in range(M.SIM_STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: break
    h = np.asarray(cl.get_center_history())
    closest = {n: float(np.min(np.hypot(h[:,0]-a, h[:,1]-b))) for n, (a, b) in SAD.items()}
    xs = np.abs(h[:, 0]); k_band = next((i for i in range(len(xs)-10) if (xs[i:i+10] < 0.05).all()), -1)
    last50 = h[-50:]; drift = float(np.linalg.norm(last50[-1] - last50[0]))
    print(f"{name} ({sx:+.2f},{sy:+.2f}) {len(h)} steps, end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}), "
          f"on line (|x|<0.05 for 10 steps) at step {k_band}, max |x| after {np.max(xs[k_band:]) if k_band>=0 else float('nan'):.3f}, "
          f"closest far {closest['far (0,-0.5)']:.3f}, near {closest['near (0,+0.5)']:.3f}, moved {drift:.3f} in last 50 steps")
