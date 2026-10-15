"""Noisy double-gyre screen for the dnew.py variants against the repo D tracker.
A: paper stop rule (first saddle held 20 steps), 50 trials x 7 starts, random
   heading, sigma_uv 0.002, 0.005 and sigma_p 0.005.
B: drift, 1000 steps with no stop, 30 trials x 7 starts, sigma_uv 0.005 and
   sigma_p 0.005 (same seeds as open_checks.py)."""
import time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from screen_common import M, run_held, nearest_saddle, VARIANTS

NOISE = (("sigma_uv 0.002", 0.002, 0.0), ("sigma_uv 0.005", 0.005, 0.0), ("sigma_p 0.005", 0.0, 0.005))


def trial_hold(args):
    var, sx, sy, s_uv, s_p, seed = args
    rng = np.random.default_rng(seed)
    cl, modes, out = run_held(var, sx, sy, rng.uniform(0, 2 * np.pi), s_uv, s_p, seed)
    c = Counter(modes)
    return var, out, len(modes), c.get('TRACK_DEFINITE', 0) + c.get('ATTRACT_FALLBACK', 0)


def trial_drift(args):
    var, sx, sy, s_uv, s_p, seed = args
    rng = np.random.default_rng(seed)
    cl, modes, out = run_held(var, sx, sy, rng.uniform(0, 2 * np.pi), s_uv, s_p, seed,
                              steps=1000, stop_on_hold=False)
    h = np.asarray(cl.get_center_history())
    k0 = next((k for k, m in enumerate(modes) if m == 'CAPTURE'), None)
    if k0 is None:
        return var, "never captured", None
    post = h[min(k0, len(h) - 1):]
    drift = float(np.max(np.linalg.norm(post - post[0], axis=1)))
    if out == "exited":
        cls = "exited"
    elif nearest_saddle(h[-1])[1] < 0.06:
        cls = "at a saddle at step 1000"
    else:
        cls = "drifted off"
    return var, cls, drift


if __name__ == "__main__":
    t0 = time.time()
    print("A. Paper stop rule: first saddle held 20 steps within 0.06, 600 steps max, exit |x|>1 or |y|>0.60")
    print("   350 trials per variant (7 starts x 50, random heading)")
    for lab, s_uv, s_p in NOISE:
        jobs = [(v, sx, sy, s_uv, s_p, 1000 + 7919 * t) for v in VARIANTS for _, sx, sy in M.STARTS for t in range(50)]
        with Pool(7) as pool:
            res = pool.map(trial_hold, jobs, chunksize=10)
        print(f"\n  {lab}")
        for v in VARIANTS:
            rr = [r for r in res if r[0] == v]
            c = Counter(r[1] for r in rr)
            st = [r[2] for r in rr if r[1] in ("far", "near", "corner")]
            dfn = [r[3] for r in rr]
            print(f"    {v:14s} far {c['far']:3d}  near {c['near']:3d}  corner {c['corner']:3d}  exited {c['exited']:3d}  timeout {c['timeout']:3d}"
                  f" | median steps to held {int(np.median(st)) if st else '-'}"
                  f" | trials with definite-H steps {sum(1 for d in dfn if d)}, steps {sum(dfn)}", flush=True)
    print(f"[{time.time()-t0:.0f} s]\n")
    print("B. Drift: 1000 steps, no stop at hold, 210 trials per variant, exit |x|>1.3 or |y|>0.8")
    for lab, s_uv, s_p in NOISE[1:]:
        jobs = [(v, sx, sy, s_uv, s_p, 5000 + 7919 * t) for v in VARIANTS for _, sx, sy in M.STARTS for t in range(30)]
        with Pool(7) as pool:
            res = pool.map(trial_drift, jobs, chunksize=5)
        print(f"\n  {lab}")
        for v in VARIANTS:
            rr = [r for r in res if r[0] == v]
            c = Counter(r[1] for r in rr)
            dr = np.array([r[2] for r in rr if r[2] is not None])
            print(f"    {v:14s} " + ", ".join(f"{k} {n}" for k, n in c.most_common())
                  + (f" | drift after capture: median {np.median(dr):.3f}, 95th pct {np.percentile(dr, 95):.3f}" if len(dr) else ""), flush=True)
    print(f"[{time.time()-t0:.0f} s]")
