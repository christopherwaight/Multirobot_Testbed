"""
mc_mechanism_starts.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_12.tex, Section disc_noise
  Makes:  the numbers behind the explanation of why noise tolerance changes with
          the start (seven-start set, plot_flip_resolution_rows.py).
  Reads:  nothing (new simulation runs).
  Writes: experiments/outputs/mc_mechanism/d_fallback.csv     D tracker, per start
          experiments/outputs/mc_mechanism/s1_reversal.csv    s1 tracker, per start
          experiments/outputs/mc_mechanism/s1_hazard.csv      s1 tracker, per height bin

EXPERIMENT
  Same harness, trackers, seeds and stop rules as mc_sweep_noise_both_trackers.py,
  10,000 trials per start and tracker, random heading per trial.
  D tracker at sigma_uv = 0.005 (sigma_p = 0), plus a noise-free control.
    Logs the branch separatrix_logic_c_step takes each step. ATTRACT_FALLBACK is the
    branch taken when the fitted Hessian of D is definite (lambda_1 lambda_2 >= 0),
    which drops the trench frame for the signed Newton step. Per start: success,
    acquisition (centroid within 0.05 of x = 0 for 10 consecutive steps before the
    stop), median height at acquisition, fraction of trials with any fallback step
    before acquisition, and success with and without such a step.
  s1 tracker at sigma_uv = 0.002 (sigma_p = 0).
    Reads the tracker's tangent each step. A reversal is the first step at which the
    tangent points up (t_y > 0.5) after the ride has pointed down (t_y < -0.5).
    Per start: success, fraction of trials that reverse, success given a reversal,
    median height of the first reversal, and the share of failures that end at the
    top saddle. Pooled over starts: the reversal hazard per riding step by height,
    counting steps while riding down and up to the first reversal.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_mechanism_starts.py --trials 10000 --workers 7
"""
import argparse
import io
import contextlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import experiments._mc_common as mc
import experiments.mc_sweep_noise_both_trackers as sw
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static

STARTS = {"S1": (-0.10, 0.30), "S2": (0.15, 0.25), "S3": (0.0, 0.35), "S4": (0.0, 0.0),
          "S5": (0.0, -0.25), "S6": (0.15, -0.15), "S7": (0.10, -0.20)}
SIGMA_D, SIGMA_S1 = 0.005, 0.002
BINS = np.round(np.arange(-0.5, 0.5001, 0.1), 2)
OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_mechanism")
os.makedirs(OUT_DIR, exist_ok=True)


def trial(args):
    tr, start, sig, seed = args
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(double_gyre_static))
    cl.reset(start[0], start[1], heading_offset=rng.uniform(0, 2 * np.pi))
    cl.measurement_noise_std = sig
    cl.position_noise_std = 0.0
    cl.diagnostics = []
    prim = sw._prim_d if tr == "D" else sw._prim_s1
    targets = [mc.SADDLE] if tr == "D" else sw.S1_TARGET
    y_exit = 0.52 if tr == "D" else sw.S1_Y_EXIT
    xs, ys, ty, modes = [], [], [], []
    for _ in range(mc.SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        xs.append(cx)
        ys.append(cy)
        modes.append(cl.diagnostics[-1]["mode"] if cl.diagnostics else "")
        t = getattr(cl, "_oecs_prev_tangent", None)
        ty.append(float(t[1]) if t is not None else 0.0)
        if min(np.hypot(cx - a, cy - b) for a, b in targets) < mc.SADDLE_CONTACT_D:
            break
        if abs(cx) > 1.0 or abs(cy) > y_exit:
            break
    xs, ys, ty = np.array(xs), np.array(ys), np.array(ty)
    far = bool(np.hypot(xs[-1], ys[-1] + 0.5) < mc.SADDLE_CONTACT_D)
    top = bool(np.hypot(xs[-1], ys[-1] - 0.5) < mc.SADDLE_CONTACT_D)
    inband = np.abs(xs) < mc.BAND_X
    t_band = next((k for k in range(max(1, len(inband) - mc.BAND_HOLD))
                   if inband[k:k + mc.BAND_HOLD].all()), -1)
    pre = modes[:t_band] if t_band >= 0 else modes
    out = {"far": far, "top": top, "t_band": t_band,
           "y_band": float(ys[t_band]) if t_band >= 0 else None,
           "fb_pre": int(sum(m == "ATTRACT_FALLBACK" for m in pre))}
    if tr == "s1":
        dwell = np.zeros(len(BINS) - 1, dtype=int)
        rev_y, stop = None, len(ys)
        down = np.where(ty < -0.5)[0]
        if len(down):
            up = np.where((ty > 0.5) & (np.arange(len(ty)) > down[0]))[0]
            if len(up):
                rev_y, stop = float(ys[up[0]]), int(up[0])
        for k in range(stop):
            if ty[k] < -0.5:
                dwell[np.clip(np.digitize(ys[k], BINS) - 1, 0, len(dwell) - 1)] += 1
        out["rev_y"] = rev_y
        out["dwell"] = dwell
    return out


def specs(tr, start, sig, n):
    base = int(1e6 * sig * 1000) % (2**31) if sig else 0
    return [(tr, start, sig, (base + 7919 * t) % (2**31)) for t in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    head = (f"# generated_by: experiments/mc_mechanism_starts.py\n# git_commit: {commit}\n"
            f"# date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"# trials_per_start: {args.trials}\n")
    n = args.trials

    with Pool(args.workers) as pool:
        rows = []
        print(f"D tracker, sigma_uv = {SIGMA_D}")
        for name, st in STARTS.items():
            out = pool.map(trial, specs("D", st, SIGMA_D, n), chunksize=16)
            clean = pool.map(trial, specs("D", st, 0.0, n), chunksize=16)
            fb = np.array([o["fb_pre"] > 0 for o in out])
            far = np.array([o["far"] for o in out])
            yb = [o["y_band"] for o in out if o["y_band"] is not None]
            r = dict(start=name, x=st[0], y=st[1], n=n, success=far.mean(),
                     acquired=np.mean([o["t_band"] >= 0 for o in out]),
                     y_acq_median=float(np.median(yb)) if yb else float("nan"),
                     p_fallback_pre=fb.mean(),
                     success_given_fallback=far[fb].mean() if fb.any() else float("nan"),
                     success_given_no_fallback=far[~fb].mean() if (~fb).any() else float("nan"),
                     p_fallback_pre_noise_free=np.mean([o["fb_pre"] > 0 for o in clean]),
                     success_noise_free=np.mean([o["far"] for o in clean]))
            rows.append(r)
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in r.items()), flush=True)
        with open(os.path.join(OUT_DIR, "d_fallback.csv"), "w") as f:
            f.write(head + f"# sigma_uv: {SIGMA_D}  sigma_p: 0\n" + ",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")

        rows, dwell, revs = [], np.zeros(len(BINS) - 1, dtype=int), np.zeros(len(BINS) - 1, dtype=int)
        print(f"\ns1 tracker, sigma_uv = {SIGMA_S1}")
        for name, st in STARTS.items():
            out = pool.map(trial, specs("s1", st, SIGMA_S1, n), chunksize=16)
            far = np.array([o["far"] for o in out])
            rv = np.array([o["rev_y"] is not None for o in out])
            fails = [o for o in out if not o["far"]]
            r = dict(start=name, x=st[0], y=st[1], n=n, success=far.mean(), p_reversal=rv.mean(),
                     success_given_reversal=far[rv].mean() if rv.any() else float("nan"),
                     reversal_y_median=float(np.median([o["rev_y"] for o in out if o["rev_y"] is not None]))
                     if rv.any() else float("nan"),
                     failures_at_top=np.mean([o["top"] for o in fails]) if fails else float("nan"))
            rows.append(r)
            for o in out:
                dwell += o["dwell"]
                if o["rev_y"] is not None:
                    revs[np.clip(np.digitize(o["rev_y"], BINS) - 1, 0, len(revs) - 1)] += 1
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in r.items()), flush=True)
        with open(os.path.join(OUT_DIR, "s1_reversal.csv"), "w") as f:
            f.write(head + f"# sigma_uv: {SIGMA_S1}  sigma_p: 0\n" + ",".join(rows[0]) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
        print("\n  height bin, riding steps, first reversals, hazard per step, |cos(pi y)| at bin centre")
        with open(os.path.join(OUT_DIR, "s1_hazard.csv"), "w") as f:
            f.write(head + f"# sigma_uv: {SIGMA_S1}  pooled over all seven starts\n"
                    "y_lo,y_hi,riding_steps,first_reversals,hazard_per_step,abs_cos_pi_y\n")
            for i in range(len(dwell)):
                yc = 0.5 * (BINS[i] + BINS[i + 1])
                h = revs[i] / dwell[i] if dwell[i] else float("nan")
                f.write(f"{BINS[i]:.1f},{BINS[i + 1]:.1f},{dwell[i]},{revs[i]},{h:.5f},"
                        f"{abs(np.cos(np.pi * yc)):.3f}\n")
                print(f"  [{BINS[i]:+.1f},{BINS[i + 1]:+.1f})  {dwell[i]:8d}  {revs[i]:6d}  {h:.4f}  "
                      f"{abs(np.cos(np.pi * yc)):.2f}")
    print(f"\nSaved: {OUT_DIR}/d_fallback.csv, s1_reversal.csv, s1_hazard.csv")


if __name__ == "__main__":
    main()
