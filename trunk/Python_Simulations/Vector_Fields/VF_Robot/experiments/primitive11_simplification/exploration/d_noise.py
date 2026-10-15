import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import experiments._mc_common as mc
import experiments.mc_sweep_noise_both_trackers as sw
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D
from experiments.mc_mechanism_starts import STARTS
from dvar import d_step

def make(label):
    if label == "current":
        return sw._prim_d
    cq = {"qD<1": 1.0, "qD<0.03": 0.03}[label]
    def prim(c):
        vx, vy = d_step(c, mc.V_MAX, 'qD', cq, [])
        return vx * mc.GAIN, vy * mc.GAIN
    return prim

def work(spec):
    spec = dict(spec); spec["primitive"] = make(spec.pop("label"))
    with contextlib.redirect_stdout(io.StringIO()):
        r = mc.run_trial(spec)
    far = np.hypot(r["final_x"] - SADDLE_FAR[0], r["final_y"] - SADDLE_FAR[1]) < SADDLE_CONTACT_D and not r["collapsed"]
    return far, far and bool(r["success_straddle"]), r["final_y"] > 0

if __name__ == "__main__":
    N = 50
    for ax, sig in (("uv", 0.002), ("uv", 0.005), ("p", 0.002)):
        s_uv, s_p = (sig, 0.0) if ax == "uv" else (0.0, sig)
        print(f"\nsigma_{ax} = {sig}, {N} trials per start (same seeds as the published sweep's first {N})")
        print("  start   " + "   ".join(f"{l:>22s}" for l in ("current", "qD<1", "qD<0.03")))
        tot = {l: [0, 0] for l in ("current", "qD<1", "qD<0.03")}
        for name, (x, y) in STARTS.items():
            row = []
            for label in ("current", "qD<1", "qD<0.03"):
                specs = sw.cell_specs("D", s_uv, s_p, N, (x, y))
                for s in specs: s.pop("tracker"); s["label"] = label
                with Pool(4) as pool:
                    res = pool.map(work, specs, chunksize=5)
                succ = sum(r[0] for r in res); ret = sum(r[1] for r in res)
                tot[label][0] += succ; tot[label][1] += ret
                row.append(f"success {succ:2d} retain {ret:2d}")
            print(f"  {name}      " + "   ".join(f"{r:>22s}" for r in row), flush=True)
        print("  TOTAL   " + "   ".join(f"{'success %3d retain %3d' % tuple(tot[l]):>22s}" for l in tot))
