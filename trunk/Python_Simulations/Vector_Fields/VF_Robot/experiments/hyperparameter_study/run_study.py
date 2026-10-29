"""
Run the hyperparameter study (design.py) with vecsim, saving per-trial results.

    venv/bin/python3 experiments/hyperparameter_study/run_study.py              # full study, resumable
    venv/bin/python3 experiments/hyperparameter_study/run_study.py --pilot      # 5% of trials, separate folder
    venv/bin/python3 experiments/hyperparameter_study/run_study.py --only E2alpha E3k

Output: results/cells/<hash>.npz (per-trial metrics as float32, plus E7 per-trial
parameters), results/manifest.csv, results/progress.log.
"""
import argparse
import csv
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vecsim as vs
import design

_FIELDS = {}


def field_of(cell):
    key = (cell["field"], cell["eps"], cell["family"])
    if key not in _FIELDS:
        _FIELDS[key] = vs.make_field(cell["field"], cell["eps"], cell["family"])
    return _FIELDS[key]


def random_params(cell, n, F):
    rng = np.random.default_rng(cell["seed"] + 1)

    def logu(a, b):
        return np.exp(rng.uniform(np.log(a), np.log(b), n))
    P = dict(side=logu(0.01, 1.0),
             alpha=np.where(rng.random(n) < 0.2, 0.0, rng.uniform(0, 0.95, n)),
             vs=np.where(rng.random(n) < 0.25, 0.0, rng.uniform(0, 0.1, n)),
             vmax=np.where(rng.random(n) < 0.2, 1e9, 0.3),
             k=logu(0.1, 100.0), kt=rng.uniform(0.05, 0.5, n), kr=logu(0.1, 5.0),
             rd=logu(1e-4, F.max_rd),
             sigma_n=np.where(rng.random(n) < 0.4, 0.0, logu(1e-4, 0.1)))
    return vs.params(n, **P)


def fname(cell):
    return f"{zlib.crc32(cell['id'].encode()) & 0xFFFFFFFF:08x}.npz"


def run_cell(args):
    cell, out_dir, scale = args
    path = os.path.join(out_dir, "cells", fname(cell))
    if os.path.exists(path):
        return cell["id"], 0.0, "cached"
    t0 = time.time()
    F = field_of(cell)
    n = max(10, int(round(cell["N"] * scale)))
    starts = thetas = None
    if cell.get("hw"):
        hs = np.load(os.path.join(HERE, f"hardware_{cell['hw']}_starts_finals.npy"))[:, :2]
        reps = max(1, int(round(cell["N"] * scale)))
        starts = np.repeat(hs, reps, axis=0)
        n = len(starts)
        thetas = np.random.default_rng(cell["seed"] + 2).uniform(0, 2 * np.pi, n)
    P = random_params(cell, n, F) if cell.get("random") else vs.params(n, **cell["params"])
    keep = 6 if cell["exp"].startswith("E1") else 0
    M, traj = vs.simulate(F, cell["task"], cell["method"], P, cell["T"], dt=cell["dt"],
                          seed=cell["seed"], window=cell["window"], keep=keep, starts=starts,
                          thetas=thetas, stiction_model=cell.get("stiction", "legacy"))
    save = {k: np.asarray(v).astype(np.float32) for k, v in M.items()}
    if cell.get("random"):
        save.update({"p_" + k: v.astype(np.float32) for k, v in P.items()})
    if traj is not None:
        save["traj"] = traj.astype(np.float32)
    np.savez_compressed(path + ".tmp.npz", **save)
    os.replace(path + ".tmp.npz", path)
    return cell["id"], time.time() - t0, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="5%% of trials into results_pilot/")
    ap.add_argument("--v2", action="store_true", help="dead-zone stiction, larger N, E8-E10 (results_v2/)")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    out_dir = os.path.join(HERE, ("results_v2" if a.v2 else "results") + ("_pilot" if a.pilot else ""))
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)
    scale = 0.05 if a.pilot else 1.0
    cells = design.build_v2() if a.v2 else design.build() + design.random_batches()
    if a.only:
        cells = [c for c in cells if any(c["exp"].startswith(o) for o in a.only)]
    cells.sort(key=lambda c: -c["N"] * c["T"] / c["dt"])
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "id", "exp", "task", "method", "field", "eps", "family", "N", "dt", "T",
                    "window", "random", "stiction", "hw"] + vs.PARAM_KEYS)
        for c in cells:
            P = vs.params(1, **c["params"])
            w.writerow([fname(c), c["id"], c["exp"], c["task"], c["method"], c["field"], c["eps"],
                        c["family"], max(10, int(round(c["N"] * scale))), c["dt"], c["T"], c["window"],
                        int(bool(c.get("random"))), c.get("stiction", "legacy"), c.get("hw") or ""]
                       + [float(P[k][0]) for k in vs.PARAM_KEYS])
    log = open(os.path.join(out_dir, "progress.log"), "a")
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (cid, el, st) in enumerate(pool.imap_unordered(run_cell, [(c, out_dir, scale) for c in cells]), 1):
            if i % 25 == 0 or i == len(cells):
                msg = f"{time.strftime('%H:%M:%S')} {i}/{len(cells)} cells, {time.time() - t0:.0f} s elapsed"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
    log.write(f"done {len(cells)} cells in {time.time() - t0:.0f} s\n")
    log.close()


if __name__ == "__main__":
    main()
