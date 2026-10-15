"""Runaway check at sigma 0.002 (copy of open_checks.py part 1). Open checks for the rewritten trackers (repo code, not scratch):
1. long-horizon runaway: 1000 steps, sigma_uv 0.005 and sigma_p 0.005, 30 trials x 7 starts,
   no stop at contact; does a latched CAPTURE stay at its saddle?
2. off-trench starts, noise-free and sigma_uv 0.002 (20 trials): near gyre centres and 0.3 off the line."""
import os, sys, io, contextlib
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
from collections import Counter
from multiprocessing import Pool
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
SAD = [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5)]
dsad = lambda p: min(np.hypot(p[0]-a, p[1]-b) for a, b in SAD)
def prim_for(tr):
    if tr == "s1":
        return lambda c: tuple(mc.GAIN * x for x in oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=1.0))
    return lambda c: tuple(mc.GAIN * x for x in separatrix_logic_c_step(c, v_max=mc.V_MAX))
def trial(args):
    tr, sx, sy, s_uv, s_p, seed, steps, rand_head = args
    np.random.seed(seed); rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=rng.uniform(0, 2*np.pi) if rand_head else 0.0)
    cl.measurement_noise_std = s_uv; cl.position_noise_std = s_p
    prim = prim_for(tr); exited = False
    for _ in range(steps):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.3 or abs(cy) > 0.8: exited = True; break
    h = np.asarray(cl.get_center_history())
    modes = [d['mode'] for d in cl.diagnostics if 'mode' in d]
    k0 = next((k for k, m in enumerate(modes) if m == 'CAPTURE'), None)
    if k0 is None: return tr, "never captured", None, None, (h[-1,0], h[-1,1]), exited
    p0 = h[min(k0, len(h)-1)]; post = h[k0:]
    drift_max = float(np.max(np.linalg.norm(post - post[0], axis=1)))
    end_d = dsad(h[-1])
    cls = "held at a saddle" if end_d < 0.06 else ("ran off (exited)" if exited else "captured, drifted off saddle")
    return tr, cls, drift_max, dsad(p0), (h[-1,0], h[-1,1]), exited
if __name__ == "__main__":
    print("1. RUNAWAY CHECK: 1000 steps, 30 trials x 7 starts, random heading, no stop at contact", flush=True)
    for lab, s_uv, s_p in (("sigma_uv 0.002", 0.002, 0.0), ("sigma_p 0.002", 0.0, 0.002)):
        jobs = [(tr, sx, sy, s_uv, s_p, 5000 + 7919*t, 1000, True) for tr in ("s1", "D") for _, sx, sy in M.STARTS for t in range(30)]
        with Pool(3) as pool:
            res = pool.map(trial, jobs, chunksize=5)
        for tr in ("s1", "D"):
            rr = [r for r in res if r[0] == tr]; c = Counter(r[1] for r in rr)
            dr = np.array([r[2] for r in rr if r[2] is not None])
            print(f"  {lab}, {tr:2s}: " + ", ".join(f"{k} {v}" for k, v in c.most_common())
                  + (f" | drift after capture: median {np.median(dr):.3f}, 95th pct {np.percentile(dr, 95):.3f}, max {dr.max():.3f}" if len(dr) else ""), flush=True)
