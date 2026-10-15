import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from collections import Counter, defaultdict
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from variant_q import oecs_separatrix_step_q

N, STEPS = 100, 300
SADDLES = [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5)]
def dsad(p): return min(np.hypot(p[0]-a, p[1]-b) for a, b in SADDLES)

def norelease(c, **kw):
    kw.pop('c_out', None)
    return oecs_separatrix_step_q(c, c_out=np.inf, **kw)
PRIMS = {"original": P.oecs_separatrix_step, "no-release": norelease}

def trial(args):
    label, sx, sy, s_uv, s_p, seed = args
    fn = PRIMS[label]
    np.random.seed(seed); rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=rng.uniform(0, 2*np.pi))
    cl.measurement_noise_std = s_uv; cl.position_noise_std = s_p
    def prim(c):
        vx, vy = fn(c, v_max=mc.V_MAX, g_perp=1.0, s_trim=0.05, r_band=0.05,
                    g_capture=0.15, s_capture=None)
        return vx * mc.GAIN, vy * mc.GAIN
    exited = False
    for _ in range(STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: exited = True; break
    h = np.asarray(cl.get_center_history()); end = h[-1]
    modes = [x['mode'] for x in cl.diagnostics if 'mode' in x]
    k0 = next((k for k, m in enumerate(modes) if m == 'CAPTURE'), None)
    entry_d = dsad(h[min(k0, len(h)-1)]) if k0 is not None else None
    cap_travel = (float(np.sum(np.linalg.norm(np.diff(h[k0:], axis=0), axis=1))) if k0 is not None else 0.0)
    if np.hypot(end[0], end[1] + 0.5) < 0.06: cls = "far"
    elif np.hypot(end[0], end[1] - 0.5) < 0.06: cls = "near"
    elif dsad(end) < 0.06: cls = "corner"
    elif exited: cls = "exited"
    else: cls = "elsewhere"
    return label, cls, entry_d, cap_travel, k0 is not None, ('CROSS' in modes)

if __name__ == "__main__":
    for axis in ("uv", "p"):
        for sig in (0.002, 0.005, 0.01):
            jobs = [(lab, sx, sy, sig if axis == "uv" else 0.0, sig if axis == "p" else 0.0, 1000 + 7919*t)
                    for lab in PRIMS for _, sx, sy in M.STARTS for t in range(N)]
            with Pool(7) as pool:
                res = pool.map(trial, jobs, chunksize=10)
            print(f"\nsigma_{axis} = {sig}  ({7*N} trials per controller)")
            for lab in PRIMS:
                rr = [r for r in res if r[0] == lab]
                c = Counter(r[1] for r in rr)
                cap = [r for r in rr if r[4]]
                far_entry = sum(1 for r in cap if r[2] > 0.1)
                trav = [r[3] for r in cap]
                stuck = sum(1 for r in cap if r[1] in ("exited", "elsewhere"))
                print(f"  {lab:10s} far {c['far']:3d}  near {c['near']:3d}  corner {c['corner']:3d}  exited {c['exited']:3d}  elsewhere {c['elsewhere']:3d}"
                      f" | captured {len(cap)}, entered >0.1 from a saddle {far_entry}, captured but not at a saddle at end {stuck},"
                      f" travel while captured median {np.median(trav) if trav else 0:.3f} max {max(trav) if trav else 0:.3f}")
