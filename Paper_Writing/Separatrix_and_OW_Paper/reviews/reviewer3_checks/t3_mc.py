"""Reviewer 3, check T3: noise sweep with the paper's own trial runner
(experiments/_mc_common.run_trial, same start, heading draw, stop rules and
success test as mc_sweep_noise_both_trackers.py) for

  D        the D tracker (regression check against noise_both.csv)
  s1       the s1 tracker (regression check)
  flow     a pure flow follower, k * sat(v0_hat), no surrogate at all
  s1rs     s1 tracker whose tangent is re-signed from v0_hat every cycle
  Dor      D tracker with the true H_D substituted for the fitted one
  Dst      off-structure starts (S1, S6) for the D tracker under noise
  s1st     off-structure starts for the s1 tracker under noise
"""
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, HERE)
import harness as H                                           # noqa: E402  (chdir + patch)
import experiments._mc_common as mc                           # noqa: E402

FAR = (0.0, -0.5)
_prim_s1rs = H.make_prim_s1_resign()
_prim_Dor = H.make_prim_D_oracle(H.H_true_dg)
PRIMS = {"D": H.prim_D, "s1": H.prim_s1, "flow": H.prim_flow,
         "s1rs": _prim_s1rs, "Dor": _prim_Dor}


def worker(spec):
    spec = dict(spec)
    spec["primitive"] = PRIMS[spec.pop("key")]
    r = mc.run_trial(spec)
    far = np.hypot(r["final_x"] - FAR[0], r["final_y"] - FAR[1]) < 0.06 and not r["collapsed"]
    return far, bool(r["success_straddle"]) and far, r["final_y"] < 0


def specs(key, s_uv, s_p, n, start=mc.FIXED_START):
    base = int(1e6*(s_uv or s_p)*1000) % (2**31)
    out = []
    for t in range(n):
        s = {"key": key, "sigma_uv": s_uv, "sigma_p": s_p, "start": start,
             "seed": (base + 7919*t) % (2**31)}
        if key in ("s1", "s1rs"):
            s["target"] = [(0.0, 0.5), (0.0, -0.5)]
            s["y_exit"] = 0.60
        out.append(s)
    return out


def cell(pool, key, s_uv, s_p, n, start=mc.FIXED_START):
    res = pool.map(worker, specs(key, s_uv, s_p, n, start), chunksize=8)
    a = np.array(res, dtype=float)
    p = a[:, 0].mean()
    return p, a[:, 1].mean(), a[:, 2].mean(), np.sqrt(p*(1-p)/n)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    plan = []
    for s in (0.002, 0.0077, 0.015, 0.03):
        for key in ("D", "s1", "flow", "s1rs", "Dor"):
            plan.append((key, s, 0.0, mc.FIXED_START))
    for s in (0.0023, 0.0081, 0.015, 0.03):
        for key in ("D", "s1", "flow"):
            plan.append((key, 0.0, s, mc.FIXED_START))
    for st in ((-0.15, 0.30), (-0.20, -0.30), (0.15, 0.25)):
        for s in (0.002, 0.0077):
            for key in ("D", "s1", "flow"):
                plan.append((key, s, 0.0, st))
    t0 = time.time()
    with Pool(7) as pool:
        print("key,start,sigma_uv,sigma_p,success,se,straddle_success,end_lower_half", flush=True)
        for key, s_uv, s_p, st in plan:
            p, ps, sign, se = cell(pool, key, s_uv, s_p, N, st)
            print(f"{key},{st[0]}:{st[1]},{s_uv},{s_p},{p:.4f},{se:.4f},{ps:.4f},{sign:.4f}", flush=True)
    print(f"# N={N} per cell, {time.time()-t0:.0f}s", flush=True)
