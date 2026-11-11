"""
mc_bent_double_gyre.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (proposed text in
          Draft_13_unsteady_dg_proposed.tex; nothing here is in the paper).
  Makes:  noise-free success, retention, reversal, tracking-error, and hold
          statistics for both trackers on a BENT double gyre with a curved
          separatrix; plus (--diag) the offsets of the true D and s1 trenches
          from that separatrix.
  Reads:  nothing (new simulation runs).
  Writes: experiments/unsteady_dg/outputs/bent_<tag>.csv (one row per trial)

FIELD (constructed here, not from the literature)
  Steady, incompressible. Stream function, field coords x_f = x + 1,
  y_f = y + 0.5:
      g   = x_f + delta sin(pi y_f)
      psi = A sin(pi g) sin(pi y_f),   u = -d psi/dy,  v = d psi/dx
  delta = 0 is the paper's steady double gyre. For delta > 0 the separatrix
  g = 1 is the CURVE x = -delta cos(pi y) (world), bowed by delta at y = 0,
  still running downward from p2* = (0, 0.5) to p1* = (0, -0.5), which stay
  put. Shear strain and vorticity on it are nonzero and H_D is not diagonal.

EXPERIMENT
  Same cluster, gains, primitives, starts S1..S7, stopping rule, and scoring
  as mc_unsteady_double_gyre.py, with the separatrix x_s(y) = -delta cos(pi y)
  evaluated at each robot's own y for the straddle and at the centroid's y
  for the tracking error (horizontal offset). Random heading per trial; D
  and s1 share it.

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/mc_bent_double_gyre.py --diag
  venv/bin/python3 experiments/unsteady_dg/mc_bent_double_gyre.py \
      --trials 500 --delta 0 0.1 0.2 --workers 6 --tag n500
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
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import experiments._mc_common as mc

OUT_DIR = os.path.join(project_root, "experiments", "unsteady_dg", "outputs")
STARTS = {
    "S1": (-0.10, 0.30), "S2": (0.15, 0.25), "S3": (0.0, 0.35), "S4": (0.0, 0.0),
    "S5": (0.0, -0.25), "S6": (0.15, -0.15), "S7": (0.10, -0.20),
}
A = 0.1
G_PERP = 1.0
MAX_STEPS, HOLD_STEPS = 500, 100
X_EXIT, Y_EXIT = 1.0, 0.60
CONTACT = mc.SADDLE_CONTACT_D
BAND_X, BAND_HOLD = mc.BAND_X, mc.BAND_HOLD
P1, P2 = (0.0, -0.5), (0.0, 0.5)


def bent_gyre(x, y, t=0.0, config=None):
    d = (config or {}).get("delta", 0.0)
    xf, yf = x + 1.0, y + 0.5
    g = xf + d * np.sin(np.pi * yf)
    gy = d * np.pi * np.cos(np.pi * yf)
    u = -(np.pi * A * np.cos(np.pi * g) * gy * np.sin(np.pi * yf)
          + np.pi * A * np.sin(np.pi * g) * np.cos(np.pi * yf))
    v = np.pi * A * np.cos(np.pi * g) * np.sin(np.pi * yf)
    return float(u), float(v)


def sep_x(y, d):
    return -d * np.cos(np.pi * y)


def run_trial(spec):
    rng = np.random.default_rng(spec["seed"])
    np.random.seed(spec["seed"] % (2**32))
    heading = rng.uniform(0.0, 2 * np.pi)
    d = spec["delta"]
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(bent_gyre)
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)
    field.config = {"delta": d}
    x0, y0 = STARTS[spec["start"]]
    cluster.reset(x0, y0, heading_offset=heading)
    cluster.measurement_noise_std = 0.0
    cluster.position_noise_std = 0.0
    nominal = mc._pairdists(np.array(cluster.get_robot_positions()).reshape(6, 2))

    if spec["tracker"] == "D":
        def prim(c):
            vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX)
            return vx * mc.GAIN, vy * mc.GAIN
    else:
        def prim(c):
            vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP)
            return vx * mc.GAIN, vy * mc.GAIN

    err_h, cy_h, strad_h, rms = [], [], [], []
    outcome, contact_step = "timeout", -1
    for k in range(MAX_STEPS):
        cluster.move(prim)
        cx, cy = cluster.get_centroid()
        rp = np.array(cluster.get_robot_positions()).reshape(6, 2)
        side = rp[:, 0] - sep_x(rp[:, 1], d)
        err_h.append(abs(cx - sep_x(cy, d)))
        cy_h.append(cy)
        strad_h.append(bool(side.min() < 0 < side.max()))
        if k % 10 == 0:
            rms.append(float(np.sqrt(np.mean((mc._pairdists(rp) - nominal) ** 2))))
        if np.hypot(cx - P1[0], cy - P1[1]) < CONTACT:
            outcome, contact_step = "p1", k
            break
        if np.hypot(cx - P2[0], cy - P2[1]) < CONTACT:
            outcome, contact_step = "p2", k
            break
        if abs(cx) > X_EXIT or abs(cy) > Y_EXIT:
            outcome = "exit"
            break

    hold_frac = hold_max = np.nan
    if outcome == "p1":
        dd = []
        for _ in range(HOLD_STEPS):
            cluster.move(prim)
            cx, cy = cluster.get_centroid()
            dd.append(np.hypot(cx - P1[0], cy - P1[1]))
        dd = np.array(dd)
        hold_frac, hold_max = float(np.mean(dd < CONTACT)), float(dd.max())

    collapsed = max(rms) > mc.COLLAPSE_RMS if rms else False
    err = np.array(err_h)
    inband = err < BAND_X
    acq = -1
    for k in range(max(1, len(inband) - BAND_HOLD + 1)):
        if inband[k:k + BAND_HOLD].all():
            acq = k
            break
    strad = np.array(strad_h)
    first = int(np.argmax(strad)) if strad.any() else -1
    kept = bool(strad.any() and strad[first:].all())
    lost_y = np.nan
    if strad.any() and not kept:
        lost_y = float(cy_h[first + int(np.argmin(strad[first:]))])
    success = outcome == "p1" and not collapsed
    seg = err[acq:] if acq >= 0 else err[-50:]
    return {
        "delta": d, "tracker": spec["tracker"], "start": spec["start"],
        "trial": spec["trial"], "seed": spec["seed"], "heading": round(heading, 5),
        "outcome": outcome, "success": int(success), "reversal": int(outcome == "p2"),
        "retained": int(success and kept), "collapsed": int(collapsed),
        "steps": len(err_h), "contact_step": contact_step, "acq_step": acq,
        "track_mean": round(float(seg.mean()), 6), "hold_frac": hold_frac,
        "hold_max": hold_max, "final_x": round(float(cluster.get_centroid()[0]), 5),
        "final_y": round(float(cy_h[-1]), 5), "min_y": round(float(min(cy_h)), 5),
        "lost_y": lost_y,
    }


def diag():
    """Offsets of the true D and s1 trenches from the curved separatrix."""
    h = 1e-5

    def jac(x, y, cfg):
        up, um = bent_gyre(x + h, y, config=cfg), bent_gyre(x - h, y, config=cfg)
        vp, vm = bent_gyre(x, y + h, config=cfg), bent_gyre(x, y - h, config=cfg)
        return ((up[0] - um[0]) / (2 * h), (vp[0] - vm[0]) / (2 * h),
                (up[1] - um[1]) / (2 * h), (vp[1] - vm[1]) / (2 * h))

    for d in (0.1, 0.2):
        cfg = {"delta": d}
        offD, offS, shear, vort = [], [], [], []
        for y in np.round(np.arange(-0.45, 0.46, 0.05), 2):
            xs = sep_x(y, d)
            xx = np.linspace(xs - 0.12, xs + 0.12, 2401)
            Dv, Sv = [], []
            for x in xx:
                ux, uy, vx, vy = jac(x, y, cfg)
                Dv.append(ux * vy - uy * vx)
                Sv.append(0.5 * (ux + vy) - np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx)))
            ux, uy, vx, vy = jac(xs, y, cfg)
            shear.append(abs(0.5 * (uy + vx))); vort.append(abs(vx - uy))
            if abs(y) >= 0.1:
                offD.append(abs(xx[int(np.argmin(Dv))] - xs))
                offS.append(abs(xx[int(np.argmin(Sv))] - xs))
        print(f"delta {d}: bow {d}, max slope dx/dy {d * np.pi:.3f}; "
              f"D trench offset max {max(offD):.4f} mean {np.mean(offD):.4f}; "
              f"s1 trench offset max {max(offS):.4f} mean {np.mean(offS):.4f}; "
              f"on separatrix max |shear| {max(shear):.4f}, max |vorticity| {max(vort):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--delta", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    if args.diag:
        diag()
        return
    if not args.tag:
        ap.error("--tag is required for a run")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    specs = []
    for ie, d in enumerate(args.delta):
        for js, s in enumerate(STARTS):
            for t in range(args.trials):
                seed = (2_000_029 * (ie + 1) + 7919 * js + 104_729 * t) % (2**31)
                for tr in ("D", "s1"):
                    specs.append({"delta": d, "tracker": tr, "start": s,
                                  "trial": t, "seed": seed})
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run_trial, specs, chunksize=4))
    rows.sort(key=lambda r: (r["delta"], r["start"], r["trial"], r["tracker"]))
    out = os.path.join(OUT_DIR, f"bent_{args.tag}.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# generated_by: experiments/unsteady_dg/mc_bent_double_gyre.py\n"
                f"# git_commit: {commit}\n"
                f"# date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"# trials_per_cell: {args.trials}  delta: {args.delta}  noise-free\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} trials, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
