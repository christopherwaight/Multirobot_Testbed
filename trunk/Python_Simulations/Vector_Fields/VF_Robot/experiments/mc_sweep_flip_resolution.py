"""
mc_sweep_flip_resolution.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the finer-resolution characterization of the tangent-sign flip found in
          rescore_single_target.py -- resolves whether the flip is a sharp step at
          one specific noise value or a smoother transition the coarse
          {0, 0.001, 0.005, ...} grid of mc_sweep_oecs_traverse.py cannot distinguish.
  Reads:  nothing (new simulation runs, not a rescoring of existing data).
  Writes: experiments/outputs/mc_oecs_traverse/flip_resolution.csv

WHY THIS SCRIPT EXISTS
  rescore_single_target.py showed Controller 2's committed ride direction flips from
  the far saddle (0, -0.5) to the near saddle (0, +0.5) between sigma_uv = 0.001
  (91.8% far-saddle success, 91.9% sign match) and sigma_uv = 0.005 (0.0% success,
  0.1% sign match) -- but with only those two grid points, it is impossible to tell
  whether the flip is a sharp threshold near one specific noise value, or a smoother
  sigmoid whose midpoint happens to fall in that interval. Chris asked for the
  intermediate points to resolve this before it goes in the paper.

EXPERIMENT
  Same controller (oecs_separatrix_step, Primitive 11), same fixed straddling start
  (0, 0.35), same 10000 trials/cell, same random-heading-per-trial protocol as
  mc_sweep_oecs_traverse.py -- but ONLY sigma_p = 0 (matches the column already shown
  in the artifact; a full 2D grid at this resolution is 25x the compute for a
  question that lives entirely on the sigma_uv axis) and ONLY sigma_uv values at
  0.0005 spacing from 0.0015 through the flip and out past it to 0.008: {0.0015,
  0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008} (0.005 is
  rerun here rather than reused from the coarse sweep so the whole tail comes from
  one consistent run; the last three points, added after Chris reviewed the first
  version of this sweep, confirm the curve stays flat well past where it has already
  saturated near zero rather than doing anything unexpected just beyond the original
  stopping point). Scored with the SAME single-far-saddle-target logic as
  rescore_single_target.py (imported directly, not reimplemented): contact within
  SADDLE_CONTACT_D of (0, -0.5), not collapsed. Also reports the far-saddle SIGN
  match rate (final_y < 0) at each point, since success itself is already near-zero
  past the flip and the sign rate is the more informative curve for the transition's
  actual shape. Straddle retention is conditioned on that SAME trial having reached
  the far saddle (found necessary after Chris asked why straddle could read higher
  than success: run_trial's own success_straddle column was computed against the
  OLD "either saddle" stop condition, which only coincides with reaching the far
  saddle for trials that actually did; for a trial that stopped at the near saddle
  instead, there is no straddle-to-far-saddle history to report, so it is scored 0,
  not carried over unconditionally -- this makes straddle retention a subset of
  success, as it should be, so P(straddle | success) is now well-defined).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_flip_resolution.py --trials 10000 --workers 8
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

SIGMA_UV_POINTS = [0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045,
                   0.005, 0.006, 0.007, 0.008]
SIGMA_P = 0.0

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


def cell_specs(sigma_uv, n_trials):
    base = int(1e6 * sigma_uv * 1000) % (2**31)
    return [{"sigma_uv": sigma_uv, "sigma_p": SIGMA_P,
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

    # Per-cell checkpoint: append each cell as it finishes so an interrupted
    # multi-hour run resumes instead of starting over.  Only rows matching the
    # current --trials are reused.  Delete the file to force a clean run.
    ckpt_path = os.path.join(OUT_DIR, "checkpoint_flip_resolution.csv")
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) == 5 and int(float(p[4])) == args.trials:
                    done[float(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        if done:
            print(f"  checkpoint: resuming, {len(done)} cells already done")

    rows = []
    print(f"Resolving the flip: sigma_p={SIGMA_P}, {args.trials} trials/cell, "
          f"{len(SIGMA_UV_POINTS)} sigma_uv points from 0.0015 to 0.008:")
    with Pool(args.workers) as pool:
        for s_uv in SIGMA_UV_POINTS:
            if s_uv in done:
                sr, sg, sd = done[s_uv]
                rows.append({"sigma_uv": s_uv, "sigma_p": SIGMA_P,
                             "success_single_target": sr,
                             "far_saddle_sign_rate": sg,
                             "success_straddle": sd})
                print(f"  sigma_uv={s_uv:<7} success={sr:6.1%}  "
                      f"sign_rate={sg:6.1%}  straddle={sd:6.1%}  "
                      f"(from checkpoint)", flush=True)
                continue
            specs = cell_specs(s_uv, args.trials)
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
                    # own fix and its docstring for why: run_trial's own
                    # success_straddle was computed against the OLD "either
                    # saddle" stop condition, which only coincides with the
                    # far-saddle stop for trials that reached the far one).
                    if r["success_straddle"]:
                        far_hits_straddled += 1
                if r["final_y"] < 0:
                    sign_matches += 1

            success_rate = far_hits / n
            sign_rate = sign_matches / n
            straddle_rate = far_hits_straddled / n
            rows.append({
                "sigma_uv": s_uv, "sigma_p": SIGMA_P,
                "success_single_target": round(success_rate, 4),
                "far_saddle_sign_rate": round(sign_rate, 4),
                "success_straddle": round(straddle_rate, 4),
            })
            with open(ckpt_path, "a") as cf:
                cf.write(f"{s_uv},{round(success_rate, 4)},"
                         f"{round(sign_rate, 4)},{round(straddle_rate, 4)},"
                         f"{args.trials}\n")
                cf.flush()
                os.fsync(cf.fileno())
            print(f"  sigma_uv={s_uv:<7} success={success_rate:6.1%}  "
                  f"sign_rate={sign_rate:6.1%}  straddle={straddle_rate:6.1%}",
                  flush=True)

    header = (f"# generated_by: experiments/mc_sweep_flip_resolution.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  sigma_p: {SIGMA_P}  "
              f"controller: oecs_separatrix_step (Primitive 11)\n"
              f"# target: single far saddle {SADDLE_FAR}, contact_d={SADDLE_CONTACT_D}\n")

    out_path = os.path.join(OUT_DIR, "flip_resolution.csv")
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
