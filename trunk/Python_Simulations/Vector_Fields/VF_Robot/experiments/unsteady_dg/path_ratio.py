"""
path_ratio.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (candidate online
          scope indicator; nothing here is in the paper).
  Makes:  the FITTED rotation-to-strain ratio |w_hat| / 2 r_hat that the cluster
          itself measures at its centroid, median over each trial's ride (steps
          before saddle contact), for every test field. This is the quantity
          Section V-B reports for the ocean record (median 0.15 to 0.20).
  Reads:  nothing; reruns trials through the three runners with a recording
          wrapper around both trackers. The trackers' code is not modified.
  Writes: experiments/unsteady_dg/outputs/path_ratio_<tag>.csv

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/path_ratio.py --trials 50 --tag n50
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

import src.control.pentagon_primitives as pp
import experiments.unsteady_dg.mc_unsteady_double_gyre as tpg
import experiments.unsteady_dg.mc_bent_double_gyre as bent
import experiments.unsteady_dg.mc_bickley_jet as bj

OUT = os.path.join(project_root, "experiments", "unsteady_dg", "outputs")
_LOG = []


def _fitted_ratio(cluster):
    rel = pp._get_relative_positions(cluster)
    u, v = pp._sample_vector_at_robots(cluster)
    tu, tv = pp._fit_vector_quadratic(rel, u, v)
    ux, uy, vx, vy = tu[1], tu[2], tv[1], tv[2]
    r = np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx))
    w = abs(vx - uy)
    return w / (2 * r) if r > 1e-12 else np.inf


def _wrap(fn):
    def wrapped(cluster, *a, **k):
        _LOG.append(_fitted_ratio(cluster))
        return fn(cluster, *a, **k)
    return wrapped


def _init():
    for mod in (tpg, bent, bj):
        mod.separatrix_logic_c_step = _wrap(pp.separatrix_logic_c_step)
        mod.oecs_separatrix_step = _wrap(pp.oecs_separatrix_step)


def _work(job):
    field, par, spec = job
    _LOG.clear()
    mod = {"tp": tpg, "bent": bent, "bickley": bj}[field]
    row = mod.run_trial(spec)
    ride = np.array(_LOG[:row["steps"]], float)
    ride = ride[np.isfinite(ride)]
    return {"field": field, "par": par, "tracker": spec["tracker"], "start": spec["start"],
            "trial": spec["trial"], "success": row["success"], "retained": row["retained"],
            "ratio_med": float(np.median(ride)) if len(ride) else np.nan,
            "frac_rot": float(np.mean(ride > 1.0)) if len(ride) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    jobs = []
    for eps in (0.0, 0.1, 0.25):
        for js, s in enumerate(tpg.STARTS):
            for t in range(args.trials):
                seed = (1_000_003 * (1 + [0.0, 0.1, 0.25].index(eps)) + 7919 * js + 104_729 * t) % (2**31)
                for tr in ("D", "s1"):
                    jobs.append(("tp", eps, {"eps": eps, "tracker": tr, "start": s,
                                             "trial": t, "seed": seed}))
    for d in (0.1, 0.2):
        for js, s in enumerate(bent.STARTS):
            for t in range(args.trials):
                seed = (2_000_029 * (1 + [0.0, 0.1, 0.2].index(d)) + 7919 * js + 104_729 * t) % (2**31)
                for tr in ("D", "s1"):
                    jobs.append(("bent", d, {"delta": d, "tracker": tr, "start": s,
                                             "trial": t, "seed": seed}))
    for js, s in enumerate(bj.STARTS):
        for t in range(args.trials):
            seed = (3_000_017 + 7919 * js + 104_729 * t) % (2**31)
            for tr in ("D", "s1"):
                jobs.append(("bickley", bj.EPS, {"tracker": tr, "start": s,
                                                 "trial": t, "seed": seed}))
    with Pool(args.workers, initializer=_init) as pool:
        rows = list(pool.imap_unordered(_work, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"path_ratio_{args.tag}.csv"), index=False)

    print("Fitted |w|/2r at the centroid, median over each ride; then median over trials")
    print(f"{'field':>8} {'par':>5} {'trk':>3} {'ratio':>6} {'IQR':>13} {'succ':>6} {'ret':>6}")
    for (f, p, tr), g in df.groupby(["field", "par", "tracker"], sort=False):
        q1, q3 = g.ratio_med.quantile([0.25, 0.75])
        print(f"{f:>8} {p:5.2f} {tr:>3} {g.ratio_med.median():6.3f} "
              f"[{q1:5.3f},{q3:5.3f}] {100 * g.success.mean():6.1f} {100 * g.retained.mean():6.1f}")


if __name__ == "__main__":
    main()
