"""
mc_sweep_flip_resolution_sigma_p.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5b.tex
  Makes:  the second panel of Fig. 7 (fig:flip_resolution) -- a fine-resolution
          sigma_p sweep at sigma_uv = 0, the referee-report finding M3: Table
          II(b)'s sigma_uv = 0 row collapses from 100% (sigma_p = 0) to 1.3%
          (sigma_p = 0.005) with no points in between to show whether that is
          a sharp step or a smooth decline, mirroring the sigma_uv-axis
          question mc_sweep_flip_resolution.py already answered.
  Reads:  nothing (new simulation runs, not a rescoring of existing data).
  Writes: experiments/outputs/mc_oecs_traverse/flip_resolution_sigma_p.csv

WHY THIS SCRIPT EXISTS
  mc_sweep_flip_resolution.py resolved the sigma_uv-axis transition (whether
  Controller 2's tangent-sign flip is a sharp threshold or a smooth decline)
  by adding intermediate points between the coarse grid's sigma_uv = 0.001 and
  0.005. The coarse grid also shows a second, un-resolved transition along the
  OTHER axis: at sigma_uv = 0 exactly, success falls from 100% at sigma_p = 0
  to 1.3% at sigma_p = 0.005 (summary_single_target.csv). The referee report's
  leading hypothesis (M3): position noise enters as an effective measurement
  error J(p_i) xi_i that lands hardest on the second-order fitted coefficients,
  corrupting grad(s1_hat); since Remark 1 (rem:s1hess) already rules out a
  Newton snap for the s1 tracker, that gradient is the transverse channel's
  ONLY restoring signal and it has no curvature normalization to absorb a
  scale error -- unlike the D tracker's transverse channel, which divides by
  lambda_2 and is scale-insensitive. If the sigma_p transition turns out as
  sharp (or sharper) than the sigma_uv one, that supports the same mechanism
  hitting a second, more fragile channel. This script produces the data to
  check that, not the diagnosis itself.

EXPERIMENT
  Same controller (oecs_separatrix_step, Primitive 11), same fixed straddling
  start (0, 0.35), same 10000 trials/cell, same random-heading-per-trial
  protocol as mc_sweep_flip_resolution.py -- but the axes are swapped: ONLY
  sigma_uv = 0 (matches the row already shown in Table II(b); a full 2D grid
  at this resolution is 25x the compute for a question that lives entirely on
  the sigma_p axis, same reasoning as the sigma_uv-only sweep) and ONLY
  sigma_p values at 0.0005 spacing from 0.0005 through 0.005: {0.0005, 0.001,
  0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005}. Scored with the
  SAME single-far-saddle-target logic as rescore_single_target.py (imported
  directly, not reimplemented): contact within SADDLE_CONTACT_D of (0, -0.5),
  not collapsed. Also reports the far-saddle SIGN match rate (final_y < 0) and
  straddle retention conditioned on far-saddle contact, for the same reasons
  documented in mc_sweep_flip_resolution.py's own docstring (straddle is a
  subset of success, not an independent quantity, once both are scored against
  the far saddle only).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_flip_resolution_sigma_p.py --trials 10000 --workers 8
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

from src.control.pentagon_primitives import oecs_separatrix_step
import experiments._mc_common as mc
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D

SIGMA_UV = 0.0
SIGMA_P_POINTS = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035,
                  0.004, 0.0045, 0.005]

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
TARGET = [(0.0, 0.5), (0.0, -0.5)]   # unchanged: run_trial still needs a stop
                                     # condition; single-target scoring happens
                                     # afterward from final_x, final_y directly,
                                     # exactly as rescore_single_target.py does
Y_EXIT = 0.60

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_oecs_traverse")
os.makedirs(OUT_DIR, exist_ok=True)


def _prim(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP,
                                  s_trim=S_TRIM, r_band=R_BAND,
                                  g_capture=G_CAPTURE, s_capture=None)
    return vx * mc.GAIN, vy * mc.GAIN


def _worker(spec):
    spec["primitive"] = _prim
    return mc.run_trial(spec)


def cell_specs(sigma_p, n_trials):
    base = int(1e6 * sigma_p * 1000) % (2**31)
    return [{"sigma_uv": SIGMA_UV, "sigma_p": sigma_p,
             "start": mc.FIXED_START, "target": TARGET, "y_exit": Y_EXIT,
             "seed": (base + 7919 * t) % (2**31)} for t in range(n_trials)]


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

    rows = []
    print(f"Resolving the sigma_p flip: sigma_uv={SIGMA_UV}, {args.trials} "
          f"trials/cell, {len(SIGMA_P_POINTS)} sigma_p points from 0.0005 "
          f"to 0.005:")
    with Pool(args.workers) as pool:
        for s_p in SIGMA_P_POINTS:
            specs = cell_specs(s_p, args.trials)
            out = list(pool.imap_unordered(_worker, specs, chunksize=16))
            n = len(out)

            far_hits = 0
            far_hits_straddled = 0
            sign_matches = 0
            for r in out:
                reached_far = (np.hypot(r["final_x"] - SADDLE_FAR[0],
                                       r["final_y"] - SADDLE_FAR[1])
                              < SADDLE_CONTACT_D and not r["collapsed"])
                if reached_far:
                    far_hits += 1
                    # Straddle can only be nonzero on trials that actually
                    # reached the FAR saddle (see rescore_single_target.py's
                    # own fix and mc_sweep_flip_resolution.py's docstring for
                    # why: run_trial's own success_straddle was computed
                    # against the OLD "either saddle" stop condition, which
                    # only coincides with the far-saddle stop for trials that
                    # reached the far one).
                    if r["success_straddle"]:
                        far_hits_straddled += 1
                if r["final_y"] < 0:
                    sign_matches += 1

            success_rate = far_hits / n
            sign_rate = sign_matches / n
            straddle_rate = far_hits_straddled / n
            rows.append({
                "sigma_uv": SIGMA_UV, "sigma_p": s_p,
                "success_single_target": round(success_rate, 4),
                "far_saddle_sign_rate": round(sign_rate, 4),
                "success_straddle": round(straddle_rate, 4),
            })
            print(f"  sigma_p={s_p:<7} success={success_rate:6.1%}  "
                  f"sign_rate={sign_rate:6.1%}  straddle={straddle_rate:6.1%}",
                  flush=True)

    header = (f"# generated_by: experiments/mc_sweep_flip_resolution_sigma_p.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  sigma_uv: {SIGMA_UV}  "
              f"controller: oecs_separatrix_step (Primitive 11)\n"
              f"# target: single far saddle {SADDLE_FAR}, contact_d={SADDLE_CONTACT_D}\n")

    out_path = os.path.join(OUT_DIR, "flip_resolution_sigma_p.csv")
    with open(out_path, "w") as f:
        f.write(header)
        f.write("sigma_uv,sigma_p,success_single_target,"
                "far_saddle_sign_rate,success_straddle\n")
        for r in rows:
            f.write(f"{r['sigma_uv']},{r['sigma_p']},"
                    f"{r['success_single_target']},"
                    f"{r['far_saddle_sign_rate']},{r['success_straddle']}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
