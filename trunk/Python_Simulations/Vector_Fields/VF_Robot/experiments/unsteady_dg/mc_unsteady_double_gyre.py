"""
mc_unsteady_double_gyre.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (proposed new
          subsection, drafted separately in Draft_13_unsteady_dg_proposed.tex;
          nothing here is in the paper yet).
  Makes:  noise-free success, retention, reversal, tracking-error, and hold
          statistics for both trackers on the TIME-PERIODIC double gyre.
  Reads:  nothing (new simulation runs).
  Writes: experiments/unsteady_dg/outputs/trials_<tag>.csv (one row per trial)

WHY
  Every simulated review of Draft_13 flags that the steady double gyre has
  special structure: zero shear strain everywhere, J = S on the separatrix,
  a diagonal H_D, and a stationary separatrix. The Shadden time-periodic
  member (eps != 0) removes all four. Its instantaneous separatrix (u = 0)
  is still a vertical line, x_f = x_s(t), but it oscillates sideways, and
  the shear strain and the vorticity on it no longer vanish.

EXPERIMENT
  Field: Double_Gyre.py with A = 0.1, omega = 2 pi / 10, eps in --eps.
  Instantaneous separatrix (world x):
      a = eps sin(omega t), b = 1 - 2a, x_s = 2 / (b + sqrt(b^2 + 4a)) - 1
  Instantaneous saddles: downstream p1*(t) = (x_s, -0.5), upstream
  p2*(t) = (x_s, +0.5). The flow on the separatrix runs downward for every
  eps tested, so p1* stays the downstream end.

  Per trial: the seven starts S1..S7 of the paper, random heading in
  [0, 2 pi), random initial phase t0 in [0, 10). D and s1 trials with the
  same (eps, start, trial index) share heading and phase (paired).
  Noise-free. Same cluster, gains, and primitives as
  mc_sweep_noise_both_trackers.py. The field clock advances by dt after
  every cluster.move (the runner's convention; _mc_common.run_trial skips
  it because the steady field ignores t).

  One stopping rule for both trackers: contact (centroid within 0.06) with
  either instantaneous saddle, centroid exit (|x| > 1 or |y| > 0.6), or 500
  steps. After contact with p1*, the run continues HOLD_STEPS = 100 steps
  (one forcing period) to score the hold.

  Scored per trial
    success     contact with p1*(t) and formation not collapsed
    reversal    contact with p2*(t)
    retained    success and robots on both sides of x_s(t) at every step
                from the first such step to contact (straddle retention,
                against the MOVING separatrix)
    acq_step    first of 10 consecutive steps with |x_c - x_s(t)| < 0.05
    track_mean  mean |x_c - x_s(t)| from acq_step to contact
    hold_frac   fraction of the HOLD_STEPS steps within 0.06 of p1*(t)
    hold_max    largest distance from p1*(t) during the hold

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/mc_unsteady_double_gyre.py \
      --trials 200 --eps 0 0.1 0.25 --workers 6 --tag n200
"""
import argparse
import csv
import io
import contextlib
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import experiments._mc_common as mc

OUT_DIR = os.path.join(project_root, "experiments", "unsteady_dg", "outputs")

STARTS = {
    "S1": (-0.10, 0.30), "S2": (0.15, 0.25), "S3": (0.0, 0.35), "S4": (0.0, 0.0),
    "S5": (0.0, -0.25), "S6": (0.15, -0.15), "S7": (0.10, -0.20),
}
G_PERP = 1.0
PERIOD = 10.0
MAX_STEPS = 500
HOLD_STEPS = 100
X_EXIT, Y_EXIT = 1.0, 0.60
CONTACT = mc.SADDLE_CONTACT_D          # 0.06
BAND_X, BAND_HOLD = mc.BAND_X, mc.BAND_HOLD   # 0.05, 10


def sep_x(t, eps, omega):
    """World x of the instantaneous separatrix (u = 0 line)."""
    a = eps * np.sin(omega * t)
    b = 1.0 - 2.0 * a
    return 2.0 / (b + np.sqrt(b * b + 4.0 * a)) - 1.0


def _prim_d(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX)
    return vx * mc.GAIN, vy * mc.GAIN


def _prim_s1(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP)
    return vx * mc.GAIN, vy * mc.GAIN


def run_trial(spec):
    rng = np.random.default_rng(spec["seed"])
    np.random.seed(spec["seed"] % (2**32))
    heading = rng.uniform(0.0, 2 * np.pi)
    t0 = rng.uniform(0.0, PERIOD)

    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(double_gyre_static)
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)
    field.config = {**field.config, "eps": spec["eps"]}
    omega = field.config.get("omega", 2 * np.pi / PERIOD)
    field.t = t0
    dt = cluster.timestep

    x0, y0 = STARTS[spec["start"]]
    cluster.reset(x0, y0, heading_offset=heading)
    cluster.measurement_noise_std = 0.0
    cluster.position_noise_std = 0.0
    pts0 = np.array(cluster.get_robot_positions()).reshape(6, 2)
    nominal = mc._pairdists(pts0)

    vmax = spec.get("vmax", mc.V_MAX)
    if spec["tracker"] == "D":
        def prim(c):
            vx, vy = separatrix_logic_c_step(c, v_max=vmax)
            return vx * mc.GAIN, vy * mc.GAIN
    else:
        def prim(c):
            vx, vy = oecs_separatrix_step(c, v_max=vmax, g_perp=G_PERP)
            return vx * mc.GAIN, vy * mc.GAIN

    xs_hist, cx_hist, cy_hist, strad_hist, rms_hist = [], [], [], [], []
    outcome, contact_step = "timeout", -1
    for k in range(MAX_STEPS):
        cluster.move(prim)
        field.step(dt)
        xs = sep_x(field.t, spec["eps"], omega)
        cx, cy = cluster.get_centroid()
        rp = np.array(cluster.get_robot_positions()).reshape(6, 2)
        xs_hist.append(xs)
        cx_hist.append(cx)
        cy_hist.append(cy)
        strad_hist.append(bool(rp[:, 0].min() < xs < rp[:, 0].max()))
        if k % 10 == 0:
            rms_hist.append(float(np.sqrt(np.mean((mc._pairdists(rp) - nominal) ** 2))))
        if np.hypot(cx - xs, cy + 0.5) < CONTACT:
            outcome, contact_step = "p1", k
            break
        if np.hypot(cx - xs, cy - 0.5) < CONTACT:
            outcome, contact_step = "p2", k
            break
        if abs(cx) > X_EXIT or abs(cy) > Y_EXIT:
            outcome = "exit"
            break

    # hold phase, only after downstream contact
    hold_frac, hold_max = np.nan, np.nan
    if outcome == "p1":
        d = []
        for _ in range(HOLD_STEPS):
            cluster.move(prim)
            field.step(dt)
            xs = sep_x(field.t, spec["eps"], omega)
            cx, cy = cluster.get_centroid()
            d.append(np.hypot(cx - xs, cy + 0.5))
            rp = np.array(cluster.get_robot_positions()).reshape(6, 2)
            rms_hist.append(float(np.sqrt(np.mean((mc._pairdists(rp) - nominal) ** 2))))
        d = np.array(d)
        hold_frac = float(np.mean(d < CONTACT))
        hold_max = float(d.max())

    collapsed = max(rms_hist) > mc.COLLAPSE_RMS if rms_hist else False
    err = np.abs(np.array(cx_hist) - np.array(xs_hist))
    inband = err < BAND_X
    acq = -1
    for k in range(max(1, len(inband) - BAND_HOLD + 1)):
        if inband[k:k + BAND_HOLD].all():
            acq = k
            break
    strad = np.array(strad_hist)
    first = int(np.argmax(strad)) if strad.any() else -1
    kept = bool(strad.any() and strad[first:].all())
    lost_y = np.nan
    if strad.any() and not kept:
        lost = first + int(np.argmin(strad[first:]))
        lost_y = float(cy_hist[lost])
    success = outcome == "p1" and not collapsed
    seg = err[acq:] if acq >= 0 else err[-50:]
    return {
        "eps": spec["eps"], "tracker": spec["tracker"], "start": spec["start"],
        "trial": spec["trial"], "seed": spec["seed"], "heading": round(heading, 5),
        "t0": round(t0, 5), "outcome": outcome, "success": int(success),
        "reversal": int(outcome == "p2"), "retained": int(success and kept),
        "collapsed": int(collapsed), "steps": len(cx_hist),
        "contact_step": contact_step, "acq_step": acq,
        "track_mean": round(float(seg.mean()), 6),
        "track_max": round(float(seg.max()), 6),
        "hold_frac": hold_frac, "hold_max": hold_max,
        "final_x": round(float(cx_hist[-1]), 5), "final_y": round(float(cy_hist[-1]), 5),
        "min_y": round(float(min(cy_hist)), 5),
        "lost_y": lost_y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vmax", type=float, default=mc.V_MAX,
                    help="c_max per channel (default 0.04, the paper's). "
                         "Diagnostic only: 0.08 tests the speed-budget explanation.")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"

    specs = []
    for ie, eps in enumerate(args.eps):
        for js, s in enumerate(STARTS):
            for t in range(args.trials):
                seed = (1_000_003 * (ie + 1) + 7919 * js + 104_729 * t) % (2**31)
                for tr in ("D", "s1"):
                    specs.append({"eps": eps, "tracker": tr, "start": s,
                                  "trial": t, "seed": seed, "vmax": args.vmax})

    out_path = os.path.join(OUT_DIR, f"trials_{args.tag}.csv")
    t_start = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_trial, specs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                el = time.time() - t_start
                print(f"  {i + 1}/{len(specs)}  {el:6.0f} s elapsed, "
                      f"~{el / (i + 1) * (len(specs) - i - 1):6.0f} s left", flush=True)

    rows.sort(key=lambda r: (r["eps"], r["start"], r["trial"], r["tracker"]))
    with open(out_path, "w", newline="") as f:
        f.write(f"# generated_by: experiments/unsteady_dg/mc_unsteady_double_gyre.py\n"
                f"# git_commit: {commit}\n"
                f"# date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"# trials_per_cell: {args.trials}  eps: {args.eps}  vmax: {args.vmax}  "
                f"noise-free\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}  ({len(rows)} trials, {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
