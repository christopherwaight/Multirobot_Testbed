"""
mc_sweep_separatrix.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  the separatrix-controller success-rate table (and optional
          heatmap fig:sep_success) and tracking-error statistics of the
          Results section; the straddle-retention column is the
          Michini-comparable metric (their Table I).
  Reads:  nothing. Writes experiments/outputs/mc_separatrix/
          trials.csv (one row per trial) and summary.csv (per cell).

EXPERIMENT (merged test plan, plan.md Phase 4)
  Logic C over a 2D noise grid sigma_uv x sigma_p, trials drawn with
  uniform random starts over [-0.8,0.8]x[-0.4,0.4] and random heading.
  AUTO-EXTEND: after the base grid, if the success rate in the
  worst-noise corner cell exceeds EXTEND_UNTIL, the top sigma_uv level is
  doubled and appended, up to MAX_EXTENSIONS times, so the failure cliff
  is always bracketed.

Run (development, 100 trials/cell):
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_separatrix.py --trials 100
Final run for the paper: --trials 10000 (parallel, ~8 workers).
"""
import argparse
import os
import sys
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.control.pentagon_primitives import separatrix_logic_c_step
import experiments._mc_common as mc

SIGMA_UV_BASE = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_P_BASE = [0.0, 0.005, 0.01, 0.02, 0.05]
EXTEND_UNTIL = 0.10       # extend sigma_uv until corner success <= this
MAX_EXTENSIONS = 3

OUT_DIR = os.path.join(project_root, "experiments", "outputs",
                       "mc_separatrix")
os.makedirs(OUT_DIR, exist_ok=True)


def _prim(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX, eps_raw=mc.EPS_RAW,
                                     eps_dim=mc.EPS_DIM)
    return vx * mc.GAIN, vy * mc.GAIN


def _worker(spec):
    spec["primitive"] = _prim
    return mc.run_trial(spec)


START_MODE = {"mode": "fixed"}     # set from --starts in main()


def cell_specs(sigma_uv, sigma_p, n_trials):
    base = int(1e6 * sigma_uv * 1000 + 1e3 * sigma_p * 1000) % (2**31)
    start = mc.FIXED_START if START_MODE["mode"] == "fixed" else None
    return [{"sigma_uv": sigma_uv, "sigma_p": sigma_p, "start": start,
             "seed": (base + 7919 * t) % (2**31)} for t in range(n_trials)]


def run_cells(pool, cells, n_trials, rows, summary):
    for s_uv, s_p in cells:
        specs = cell_specs(s_uv, s_p, n_trials)
        out = list(pool.imap_unordered(_worker, specs, chunksize=16))
        rows.extend(out)
        st = np.mean([r["success_traverse"] for r in out])
        sb = np.mean([r["success_band"] for r in out])
        ss = np.mean([r["success_straddle"] for r in out])
        tm = np.mean([r["track_mean"] for r in out])
        tp = np.mean([r["track_p95"] for r in out])
        summary[(s_uv, s_p)] = (st, sb, ss, tm, tp)
        print(f"  sigma_uv={s_uv:<6} sigma_p={s_p:<6} "
              f"traverse={st:5.1%} band={sb:5.1%} straddle={ss:5.1%} "
              f"track_mean={tm:.4f} p95={tp:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--starts", choices=["fixed", "random"],
                    default="fixed",
                    help="fixed = paper experiment (straddling start "
                         f"{mc.FIXED_START}, noise dialed up); random = "
                         "basin evidence only")
    args = ap.parse_args()
    START_MODE["mode"] = args.starts

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sigma_uv_levels = list(SIGMA_UV_BASE)
    rows, summary = [], {}

    with Pool(args.workers) as pool:
        print(f"Base grid ({len(sigma_uv_levels)}x{len(SIGMA_P_BASE)} "
              f"cells, {args.trials} trials/cell):")
        run_cells(pool, [(u, p) for u in sigma_uv_levels
                         for p in SIGMA_P_BASE], args.trials, rows, summary)

        for _ in range(MAX_EXTENSIONS):
            corner = summary[(sigma_uv_levels[-1], SIGMA_P_BASE[-1])]
            if corner[0] <= EXTEND_UNTIL:
                break
            new_uv = sigma_uv_levels[-1] * 2
            sigma_uv_levels.append(new_uv)
            print(f"Corner success {corner[0]:.1%} > {EXTEND_UNTIL:.0%}, "
                  f"extending sigma_uv to {new_uv}:")
            run_cells(pool, [(new_uv, p) for p in SIGMA_P_BASE],
                      args.trials, rows, summary)

    header = (f"# generated_by: experiments/mc_sweep_separatrix.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  steps: {mc.SIM_STEPS}  "
              f"controller: separatrix_logic_c_step\n"
              f"# start_mode: {args.starts}  fixed_start: {mc.FIXED_START}"
              f"  start_box: {mc.START_BOX}\n"
              f"# band_x: {mc.BAND_X}  collapse_rms: {mc.COLLAPSE_RMS}\n")

    cols = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
            "t_band", "steps", "success_traverse", "success_band",
            "success_straddle", "first_straddle", "collapsed",
            "track_mean", "track_p95", "shape_rms_max", "effort",
            "final_x", "final_y"]
    with open(os.path.join(OUT_DIR, f"trials_{args.starts}.csv"),
              "w") as f:
        f.write(header)
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                             else str(r[c]) for c in cols) + "\n")

    with open(os.path.join(OUT_DIR, f"summary_{args.starts}.csv"),
              "w") as f:
        f.write(header)
        f.write("sigma_uv,sigma_p,success_traverse,success_band,"
                "success_straddle,track_mean,track_p95\n")
        for (u, p), (st, sb, ss, tm, tp) in sorted(summary.items()):
            f.write(f"{u},{p},{st:.4f},{sb:.4f},{ss:.4f},"
                    f"{tm:.5f},{tp:.5f}\n")

    print(f"Wrote {OUT_DIR}/trials_{args.starts}.csv ({len(rows)} rows) "
          f"and summary_{args.starts}.csv")


if __name__ == "__main__":
    main()
