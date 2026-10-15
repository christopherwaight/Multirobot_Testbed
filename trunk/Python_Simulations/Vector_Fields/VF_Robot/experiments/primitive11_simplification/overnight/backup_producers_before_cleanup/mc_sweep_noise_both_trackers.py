"""
mc_sweep_noise_both_trackers.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_11.tex
  Makes:  the data behind Fig. fig:flip_resolution (Behavior Under Noise), which
          after the Draft 10 Part A pass shows BOTH trackers' far-saddle success
          against sigma_uv (panel a) and sigma_p (panel b) on one log axis, plus
          the s1 tracker's straddle retention.
  Reads:  nothing (new simulation runs).
  Writes: experiments/outputs/mc_noise_both/noise_both.csv
          (plus a per-cell checkpoint so an interrupted run resumes).

EXPERIMENT
  Same protocol as the two sweeps it merges, so the s1 cells are a regression
  check on flip_resolution.csv / flip_resolution_sigma_p.csv:
    - fixed straddling start mc.FIXED_START = (0, 0.35), random heading per trial
    - 10000 trials per cell, 500 steps, seeds (base + 7919 t) with base from the
      nonzero sigma, identical to mc_sweep_flip_resolution*.py and, on the axes,
      to mc_sweep_separatrix.py
    - D tracker: separatrix_logic_c_step, stop at far-saddle contact or exit
      (mc_sweep_separatrix.py's rule)
    - s1 tracker: oecs_separatrix_step (paper gains), stop at either saddle or
      y_exit 0.60 (mc_sweep_flip_resolution.py's rule)
    - success: final centroid within SADDLE_CONTACT_D of the far saddle (0, -0.5)
      and not collapsed; straddle retention counted only on successful trials
  One-dimensional axes only (sigma_p = 0 on panel a, sigma_uv = 0 on panel b).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_noise_both_trackers.py --trials 10000 --workers 7

  Off-separatrix start (writes noise_both_<tag>.csv, leaves noise_both.csv alone):
  venv/bin/python3 experiments/mc_sweep_noise_both_trackers.py --trials 10000 \
      --start -0.15 0.30 --tag S1
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

from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import experiments._mc_common as mc
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D

SIGMAS = [0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005,
          0.006, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03]

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
S1_TARGET = [(0.0, 0.5), (0.0, -0.5)]
S1_Y_EXIT = 0.60

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_noise_both")
os.makedirs(OUT_DIR, exist_ok=True)


def _prim_d(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX, eps_raw=mc.EPS_RAW,
                                     eps_dim=mc.EPS_DIM)
    return vx * mc.GAIN, vy * mc.GAIN


def _prim_s1(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP,
                                  s_trim=S_TRIM, r_band=R_BAND,
                                  g_capture=G_CAPTURE, s_capture=None)
    return vx * mc.GAIN, vy * mc.GAIN


def _worker(spec):
    spec["primitive"] = _prim_d if spec.pop("tracker") == "D" else _prim_s1
    return mc.run_trial(spec)


def cell_specs(tracker, sigma_uv, sigma_p, n_trials, start):
    base = int(1e6 * (sigma_uv or sigma_p) * 1000) % (2**31)
    specs = []
    for t in range(n_trials):
        s = {"tracker": tracker, "sigma_uv": sigma_uv, "sigma_p": sigma_p,
             "start": start, "seed": (base + 7919 * t) % (2**31)}
        if tracker == "s1":
            s["target"] = S1_TARGET
            s["y_exit"] = S1_Y_EXIT
        specs.append(s)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--start", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="start point; default mc.FIXED_START (0, 0.35)")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint and output files, so a "
                         "new start does not overwrite or resume noise_both.csv")
    args = ap.parse_args()
    start = tuple(args.start) if args.start else mc.FIXED_START
    if args.start and not args.tag:
        ap.error("--tag is required with --start")
    suffix = f"_{args.tag}" if args.tag else ""

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ckpt_path = os.path.join(OUT_DIR, f"checkpoint_noise_both{suffix}.csv")
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) == 7 and int(p[6]) == args.trials:
                    done[(p[0], p[1], float(p[2]))] = (float(p[3]), float(p[4]), float(p[5]))
        if done:
            print(f"  checkpoint: resuming, {len(done)} cells already done")

    cells = [(tr, ax, s) for ax in ("uv", "p") for tr in ("D", "s1") for s in SIGMAS]
    rows = []
    with Pool(args.workers) as pool:
        for tr, ax, s in cells:
            s_uv, s_p = (s, 0.0) if ax == "uv" else (0.0, s)
            key = (tr, ax, s)
            if key in done:
                succ, strad, sign = done[key]
            else:
                out = list(pool.imap_unordered(
                    _worker, cell_specs(tr, s_uv, s_p, args.trials, start),
                    chunksize=16))
                n = len(out)
                hits = hits_strad = sign_ok = 0
                for r in out:
                    far = (np.hypot(r["final_x"] - SADDLE_FAR[0], r["final_y"] - SADDLE_FAR[1])
                           < SADDLE_CONTACT_D and not r["collapsed"])
                    hits += far
                    hits_strad += far and bool(r["success_straddle"])
                    sign_ok += r["final_y"] < 0
                succ, strad, sign = (round(hits / n, 4), round(hits_strad / n, 4),
                                     round(sign_ok / n, 4))
                with open(ckpt_path, "a") as cf:
                    cf.write(f"{tr},{ax},{s},{succ},{strad},{sign},{args.trials}\n")
                    cf.flush()
                    os.fsync(cf.fileno())
            rows.append((tr, ax, s_uv, s_p, succ, strad, sign))
            print(f"  {tr:>2} sigma_{ax:<2}={s:<8} success={succ:6.1%} "
                  f"straddle={strad:6.1%} sign={sign:6.1%}", flush=True)

    out_path = os.path.join(OUT_DIR, f"noise_both{suffix}.csv")
    with open(out_path, "w") as f:
        f.write(f"# generated_by: experiments/mc_sweep_noise_both_trackers.py\n"
                f"# git_commit: {commit}\n# date: {stamp}\n"
                f"# trials_per_cell: {args.trials}  start: {start}\n"
                f"# target: single far saddle {SADDLE_FAR}, contact_d={SADDLE_CONTACT_D}\n")
        f.write("tracker,axis,sigma_uv,sigma_p,success,success_straddle,far_saddle_sign_rate\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
