import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from multiprocessing import Pool
import main_separatrix_traverse as M
import experiments._mc_common as mc
import fsad
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
SAD = [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5)]
dsad = lambda p: min(np.hypot(p[0]-a, p[1]-b) for a, b in SAD)
CTRL = {
    "s1, q capture (repo)":       lambda c, log: fsad.s1_step(c, mc.V_MAX, entry='q', log=log),
    "s1, flow-saddle capture":    lambda c, log: fsad.s1_step(c, mc.V_MAX, entry='fs', log=log),
    "D, no capture (repo)":       lambda c, log: fsad.d_step(c, mc.V_MAX, capture=None, log=log),
    "D, flow-saddle capture":     lambda c, log: fsad.d_step(c, mc.V_MAX, capture='fs', log=log),
}
def trial(args):
    label, sx, sy, s_uv, seed, steps, heading_rand = args
    np.random.seed(seed); rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=(rng.uniform(0, 2*np.pi) if heading_rand else 0.0))
    cl.measurement_noise_std = s_uv; cl.position_noise_std = 0.0
    log = []
    prim = lambda c: tuple(mc.GAIN * x for x in CTRL[label](c, log))
    for _ in range(steps):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: break
    h = np.asarray(cl.get_center_history()); end = h[-1]
    k0 = next((k for k, m in enumerate(log) if m == 'CAPTURE'), None)
    entry = dsad(h[min(k0, len(h)-1)]) if k0 is not None else None
    if np.hypot(end[0], end[1] + 0.5) < 0.06: cls = "far"
    elif np.hypot(end[0], end[1] - 0.5) < 0.06: cls = "near"
    elif dsad(end) < 0.06: cls = "corner"
    elif abs(end[0]) > 1.0 or abs(end[1]) > 0.60: cls = "exited"
    else: cls = "elsewhere"
    return label, cls, entry, k0, (float(end[0]), float(end[1]))
if __name__ == "__main__":
    print("CLEAN gyre, 600 steps, exit |y|>0.60")
    for label in CTRL:
        res = [trial((label, sx, sy, 0.0, 0, 600, False)) for _, sx, sy in M.STARTS]
        print(f"  {label:26s} " + "; ".join(f"{n}:{r[1]}" + (f"(cap@{r[3]},{r[2]:.3f})" if r[3] is not None else "") for (n, _, _), r in zip(M.STARTS, res)), flush=True)
    for sig in (0.002, 0.005):
        jobs = [(label, sx, sy, sig, 1000 + 7919*t, 300, True) for label in CTRL for _, sx, sy in M.STARTS for t in range(50)]
        with Pool(4) as pool:
            res = pool.map(trial, jobs, chunksize=10)
        print(f"\nNOISE sigma_uv = {sig}, 350 trials each, 300 steps, no stop at contact")
        for label in CTRL:
            rr = [r for r in res if r[0] == label]; c = Counter(r[1] for r in rr)
            ent = np.array([r[2] for r in rr if r[2] is not None])
            print(f"  {label:26s} far {c['far']:3d} near {c['near']:3d} corner {c['corner']:2d} exited {c['exited']:3d} elsewhere {c['elsewhere']:3d}"
                  + (f" | captured {len(ent)}, entry >0.1 from saddle {(ent > 0.1).sum()}, median entry {np.median(ent):.3f}" if len(ent) else " | never captured"), flush=True)
