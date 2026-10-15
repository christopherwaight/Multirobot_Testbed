import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
from collections import Counter
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField

def slide_attract(c, v_max, log):
    u, v = P._sample_vector_at_robots(c); rel = P._get_relative_positions(c)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv)
    lam, V = np.linalg.eigh(H)
    v_neg = V[:, 0 if lam[0] < lam[1] else 1]
    if float(g @ v_neg) > 0: v_neg = -v_neg
    f = np.array([tu[0], tv[0]])
    L = [max(abs(l), 1e-12) for l in lam]
    if float(f @ v_neg) > 0:
        log.append('SLIDE'); den = L
    else:
        log.append('ATTRACT'); den = [np.sign(l) * a if l != 0 else 1e-12 for l, a in zip(lam, L)]
    d = sum(v_max * np.tanh(-(V[:, i] @ g) / den[i] / v_max) * V[:, i] for i in range(2))
    return float(d[0]), float(d[1])

SAD = {"far": (0, -0.5), "near": (0, 0.5)}
for name, sx, sy in M.STARTS:
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy); log = []
    prim = lambda c: tuple(M.GAIN * x for x in slide_attract(c, M.V_MAX, log))
    for k in range(M.SIM_STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: break
    h = np.asarray(cl.get_center_history())
    cl_far = float(np.min(np.hypot(h[:,0], h[:,1] + 0.5))); cl_near = float(np.min(np.hypot(h[:,0], h[:,1] - 0.5)))
    k_far = int(np.argmin(np.hypot(h[:,0], h[:,1] + 0.5)))
    xs = np.abs(h[:, 0]); k_band = next((i for i in range(len(xs)-10) if (xs[i:i+10] < 0.05).all()), -1)
    upto = k_far if cl_far < 0.06 else len(h)
    maxx = np.max(xs[k_band:upto]) if 0 <= k_band < upto else float('nan')
    cnt = Counter(log); sw = sum(1 for i in range(1, len(log)) if log[i] != log[i-1])
    print(f"{name} ({sx:+.2f},{sy:+.2f}) {len(h)} steps, end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}); SLIDE {cnt['SLIDE']} ATTRACT {cnt['ATTRACT']}, {sw} switches; "
          f"on line at step {k_band}, max |x| before far saddle {maxx:.3f}; closest far {cl_far:.3f} (step {k_far}), near {cl_near:.3f}; "
          f"y range [{h[:,1].min():+.3f}, {h[:,1].max():+.3f}]")
