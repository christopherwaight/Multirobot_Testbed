"""
mc_sweep_oecs.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the OECS-tracker success-rate table of the Results section
          (noise robustness of Controller 2), directly comparable cell by
          cell to the Logic C table from mc_sweep_separatrix.py.
  Reads:  nothing. Writes experiments/outputs/mc_oecs/
          trials_fixed.csv (one row per trial) and summary_fixed.csv.

EXPERIMENT
  Primitive 10 (oecs_trap_step) in CORE-SEEK mode over the same 2D noise
  grid sigma_uv x sigma_p as the Logic C sweep, from the same fixed
  straddling start (0, 0.35), random initial heading per trial.  The
  target differs by necessity: the attracting OECS segment through that
  start terminates at its TRAP core, the TOP saddle (0, 0.5), so success
  = reaching (0, 0.5) within 0.06 without formation collapse (Logic C's
  task runs the full trench to the BOTTOM saddle; path lengths differ,
  so cliff locations are compared qualitatively, both being
  estimator-limited).  AUTO-EXTEND doubles the top sigma_uv until the
  worst-corner success rate falls below 10%.

Run (development, 1000 trials/cell):
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_oecs.py --trials 1000 --workers 8
Final run for the paper: --trials 10000 (see FINAL_SWEEPS_HOWTO.md).
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

from src.control.pentagon_primitives import oecs_trap_step
import experiments._mc_common as mc

SIGMA_UV_BASE = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_P_BASE = [0.0, 0.005, 0.01, 0.02, 0.05]
EXTEND_UNTIL = 0.10
MAX_EXTENSIONS = 3

G_PERP, S_TRIM, S_CAPTURE = 1.0, 0.05, -0.9
TARGET = (0.0, 0.5)          # TRAP core of the attracting segment

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_oecs")
os.makedirs(OUT_DIR, exist_ok=True)


def _prim(c):
    vx, vy = oecs_trap_step(c, v_max=mc.V_MAX, g_perp=G_PERP,
                            s_trim=S_TRIM, s_capture=S_CAPTURE)
    return vx * mc.GAIN, vy * mc.GAIN


def _worker(spec):
    spec["primitive"] = _prim
    return mc.run_trial(spec)


def cell_specs(sigma_uv, sigma_p, n_trials):
    base = int(1e6 * sigma_uv * 1000 + 1e3 * sigma_p * 1000) % (2**31)
    return [{"sigma_uv": sigma_uv, "sigma_p": sigma_p,
             "start": mc.FIXED_START, "target": TARGET,
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
              f"core={st:5.1%} band={sb:5.1%} straddle={ss:5.1%} "
              f"track_mean={tm:.4f} p95={tp:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

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

    header = (f"# generated_by: experiments/mc_sweep_oecs.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  steps: {mc.SIM_STEPS}  "
              f"controller: oecs_trap_step (core-seek)\n"
              f"# g_perp: {G_PERP}  s_trim: {S_TRIM}  "
              f"s_capture: {S_CAPTURE}  target: {TARGET}\n"
              f"# start: {mc.FIXED_START}  band_x: {mc.BAND_X}  "
              f"collapse_rms: {mc.COLLAPSE_RMS}\n")

    cols = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
            "t_band", "steps", "success_traverse", "success_band",
            "success_straddle", "first_straddle", "collapsed",
            "track_mean", "track_p95", "shape_rms_max", "effort",
            "final_x", "final_y"]
    with open(os.path.join(OUT_DIR, "trials_fixed.csv"), "w") as f:
        f.write(header)
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                             else str(r[c]) for c in cols) + "\n")

    with open(os.path.join(OUT_DIR, "summary_fixed.csv"), "w") as f:
        f.write(header)
        f.write("sigma_uv,sigma_p,success_core,success_band,"
                "success_straddle,track_mean,track_p95\n")
        for (u, p), (st, sb, ss, tm, tp) in sorted(summary.items()):
            f.write(f"{u},{p},{st:.4f},{sb:.4f},{ss:.4f},"
                    f"{tm:.5f},{tp:.5f}\n")

    print(f"Wrote {OUT_DIR}/trials_fixed.csv ({len(rows)} rows) "
          f"and summary_fixed.csv")


if __name__ == "__main__":
    main()
