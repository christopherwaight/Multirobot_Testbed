"""
mc_sweep_radius.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_8a.tex -> Draft_9
  Answers: Reviewer 1 item A4, "closed-loop rho sweep". Section II-C claims
          formation radius is "a specifiable mission attribute selected
          against a stated trade" (truncation bias against noise
          suppression) and no closed-loop experiment varies it. This
          script closes that loop.
  Reads:  nothing. Writes experiments/outputs/mc_radius/
          trials.csv (one row per trial) and summary.csv (per cell).

EXPERIMENT
  Both primitives (D tracker and s1 tracker), far-saddle success against
  sigma_uv, at each of the four validated formation scales. The ring
  radius rho equals the SAS parameter L_2 exactly (verified: five robots
  at radius L_2, one at the centroid), so scale s gives rho = 0.15 * s.

  Expected shape: a minimum in the optimal rho. Truncation bias grows
  with rho (the quadratic model fits a wider, less-quadratic patch),
  while the noise gains fall as rho^-q. The two pull opposite ways, so
  success against a fixed noise level should peak at an interior radius
  rather than at either end.

  Position noise is held at zero. The trade under study is truncation
  against measurement noise; sigma_p enters through the same
  sigma_eff channel (Eq. sigma_eff) and would confound the axis.

  Runs on the double gyre only. The noise model is validated there and
  nowhere else, and all four radii are viable there (on the ocean field
  the 0.7x scale loses the trench at the island gap and 0.25x stalls,
  per config/formations/pentagon_small.yaml).

NOTE ON THE FORMATION CONFIG
  _mc_common pins FORMATION_CONFIG at module scope and caches the
  nominal pair-distance vector in a module global. Both are rebound per
  scale here, inside the worker, so each process builds the right
  geometry. The collapse threshold is scaled with the formation as well:
  COLLAPSE_RMS is 4x the nominal pair length at the 0.5x scale, so
  holding it fixed would make collapse detection far stricter for small
  formations and far looser for large ones, biasing the very quantity
  being swept.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_radius.py --trials 1000 --workers 6
"""
import argparse
import os
import sys
import csv
import tempfile
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.control.pentagon_primitives import (separatrix_logic_c_step,
                                             oecs_separatrix_step)
import experiments._mc_common as mc

# Scale factors are the four blocks recorded in
# config/formations/pentagon_small.yaml. rho = 0.15 * scale.
SCALES = [0.25, 0.5, 0.7, 1.0]
BASE_L2 = 0.150000

# Bracket the published cliffs: the D tracker crosses 50% near
# sigma_uv = 0.0079 and the s1 tracker between 0.0015 and 0.002.
SIGMA_UV_LEVELS = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.02]

# Nominal SAS block at scale 1.0 (the top comment block of the yaml).
BASE_FORMATION = {
    "type": "cluster_of_clusters",
    "num_robots": 6,
    "cluster_config": 2,
    "config_tree": "(2,2,2)",
    "L_2": 0.150000, "theta_2": -1.570796,
    "L_3": 0.176336, "theta_3": -1.884956,
    "L_4": 0.176336, "theta_4": 1.884956,
    "p_1": 0.161172, "beta_1": 0.772617,
    "q_1": 0.230827, "theta_c": -0.772617,
    "x_c": 0.0, "y_c": 0.0,
    "position_gain": 1.0, "angle_gain": 0.1,
}
_SCALED_KEYS = ("L_2", "L_3", "L_4", "p_1", "q_1")

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_radius")
os.makedirs(OUT_DIR, exist_ok=True)
CFG_DIR = os.path.join(tempfile.gettempdir(), "mc_radius_cfg")
os.makedirs(CFG_DIR, exist_ok=True)


def config_path_for(scale):
    """Write (once) and return a formation yaml at this scale."""
    path = os.path.join(CFG_DIR, f"pentagon_scale_{scale:.2f}.yaml")
    if not os.path.exists(path):
        f = dict(BASE_FORMATION)
        for k in _SCALED_KEYS:
            f[k] = BASE_FORMATION[k] * scale
        with open(path, "w") as fh:
            yaml.safe_dump({"formation": f}, fh)
    return path


def _prim_d(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX, eps_raw=mc.EPS_RAW,
                                     eps_dim=mc.EPS_DIM)
    return vx * mc.GAIN, vy * mc.GAIN


def _prim_s1(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX)
    return vx * mc.GAIN, vy * mc.GAIN


_PRIMS = {"D": _prim_d, "s1": _prim_s1}


def _worker(spec):
    # spec is a copy in the worker process, so popping here does not touch
    # the parent's list; labels are read from the parent copy.
    spec = dict(spec)
    scale = spec.pop("scale")
    # Rebind the module-level formation for this process, and clear the
    # cached nominal pair distances so they are rebuilt at this scale.
    mc.FORMATION_CONFIG = config_path_for(scale)
    mc._NOMINAL_PDIST = None
    # Scale the collapse threshold with the formation (see module docstring).
    mc.COLLAPSE_RMS = 0.30 * (scale / 0.5)
    spec["primitive"] = _PRIMS[spec.pop("tracker_name")]
    row = mc.run_trial(spec)
    row["scale"] = scale
    row["rho"] = BASE_L2 * scale
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    specs = []
    for tracker in ("D", "s1"):
        for scale in SCALES:
            for suv in SIGMA_UV_LEVELS:
                for t in range(args.trials):
                    # Seed is a pure function of the cell and trial index, so
                    # every (scale, sigma) cell sees the same heading draws and
                    # the comparison across radii is paired rather than
                    # independent.
                    specs.append({
                        "tracker_name": tracker, "scale": scale,
                        "sigma_uv": suv, "sigma_p": 0.0,
                        "seed": 100000 + t,
                        "start": mc.FIXED_START,
                    })

    print(f"{len(specs)} trials: 2 trackers x {len(SCALES)} scales x "
          f"{len(SIGMA_UV_LEVELS)} noise levels x {args.trials}")
    # imap keeps only one chunk of results in flight per worker. pool.map
    # materializes every row at once, which exhausts 8 GB at 64k trials.
    labels = [(s["tracker_name"], s["scale"]) for s in specs]
    t0 = datetime.now(timezone.utc)
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap(_worker, specs, chunksize=64)):
            row["tracker"] = labels[i][0]
            rows.append(row)
            if (i + 1) % 8000 == 0:
                el = (datetime.now(timezone.utc) - t0).total_seconds()
                print(f"  {i+1}/{len(specs)}  {el/60:.1f} min", flush=True)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"done in {dt/60:.1f} min")

    trials_csv = os.path.join(OUT_DIR, "trials.csv")
    with open(trials_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-cell summary.
    summ = []
    for tracker in ("D", "s1"):
        for scale in SCALES:
            for suv in SIGMA_UV_LEVELS:
                sel = [r for r in rows if r["tracker"] == tracker
                       and r["scale"] == scale and r["sigma_uv"] == suv]
                n = len(sel)
                summ.append({
                    "tracker": tracker, "scale": scale,
                    "rho": BASE_L2 * scale, "sigma_uv": suv, "n": n,
                    "success_traverse": sum(r["success_traverse"]
                                            for r in sel) / n,
                    "success_straddle": sum(r["success_straddle"]
                                            for r in sel) / n,
                    "collapsed": sum(r["collapsed"] for r in sel) / n,
                    "track_mean": float(np.mean([r["track_mean"]
                                                 for r in sel])),
                })
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    print(f"wrote {trials_csv}\nwrote {summary_csv}")
    for tracker in ("D", "s1"):
        print(f"\n{tracker} tracker, success_traverse by rho x sigma_uv")
        hdr = "  rho     " + "".join(f"{s:>8.4g}" for s in SIGMA_UV_LEVELS)
        print(hdr)
        for scale in SCALES:
            cells = [next(r for r in summ if r["tracker"] == tracker
                          and r["scale"] == scale and r["sigma_uv"] == s)
                     for s in SIGMA_UV_LEVELS]
            print(f"  {BASE_L2*scale:.4f}  "
                  + "".join(f"{c['success_traverse']:>8.3f}" for c in cells))


if __name__ == "__main__":
    main()
