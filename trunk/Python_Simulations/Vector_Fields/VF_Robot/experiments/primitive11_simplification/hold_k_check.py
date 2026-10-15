"""Hold-for-K termination rule: a trial succeeds at the first step where the
centroid has stayed within SADDLE_CONTACT_D (0.06) of a saddle for K consecutive
steps, and the run would end there. Scored on the same trials for K = 1 (plain
contact), 10, and 20. 30 trials x 7 starts per tracker, random heading, up to 1000 steps."""
import os, sys, io, contextlib
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
from multiprocessing import Pool
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
R = 0.06
KS = (1, 10, 20)
SAD = {"far": (0.0, -0.5), "near": (0.0, 0.5), "corner": None}
def first_hold(h, target, K):
    if target is None:
        d = np.min(np.stack([np.hypot(h[:,0]-a, h[:,1]-b) for a in (-1, 1) for b in (-0.5, 0.5)]), axis=0)
    else:
        d = np.hypot(h[:,0]-target[0], h[:,1]-target[1])
    run = 0
    for k, inside in enumerate(d < R):
        run = run + 1 if inside else 0
        if run >= K: return k
    return None
def trial(args):
    tr, sx, sy, s_uv, s_p, seed = args
    np.random.seed(seed); rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=rng.uniform(0, 2*np.pi))
    cl.measurement_noise_std = s_uv; cl.position_noise_std = s_p
    if tr == "s1":
        prim = lambda c: tuple(mc.GAIN * x for x in oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=1.0))
    else:
        prim = lambda c: tuple(mc.GAIN * x for x in separatrix_logic_c_step(c, v_max=mc.V_MAX))
    for _ in range(1000):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.3 or abs(cy) > 0.8: break
    h = np.asarray(cl.get_center_history())
    out = {}
    for K in KS:
        hits = {n: first_hold(h, t, K) for n, t in SAD.items()}
        hits = {n: k for n, k in hits.items() if k is not None}
        out[K] = (min(hits, key=hits.get), hits[min(hits, key=hits.get)]) if hits else (None, None)
    return tr, out
if __name__ == "__main__":
    for lab, s_uv, s_p in (("sigma_uv 0.002", 0.002, 0.0), ("sigma_uv 0.005", 0.005, 0.0), ("sigma_p 0.005", 0.0, 0.005)):
        jobs = [(tr, sx, sy, s_uv, s_p, 5000 + 7919*t) for tr in ("s1", "D") for _, sx, sy in M.STARTS for t in range(30)]
        with Pool(3) as pool:
            res = pool.map(trial, jobs, chunksize=5)
        print(f"\n{lab}, 210 trials per tracker: first saddle held for K steps (far / near / corner / none), median step it happened", flush=True)
        for tr in ("s1", "D"):
            rr = [r[1] for r in res if r[0] == tr]
            row = []
            for K in KS:
                c = {n: sum(1 for o in rr if o[K][0] == n) for n in ("far", "near", "corner")}
                none = sum(1 for o in rr if o[K][0] is None)
                steps = [o[K][1] for o in rr if o[K][0] is not None]
                row.append(f"K={K:2d}: {c['far']:3d}/{c['near']:3d}/{c['corner']:3d}/{none:3d} (step {int(np.median(steps)) if steps else '-'})")
            print(f"  {tr:2s}  " + "   ".join(row), flush=True)
