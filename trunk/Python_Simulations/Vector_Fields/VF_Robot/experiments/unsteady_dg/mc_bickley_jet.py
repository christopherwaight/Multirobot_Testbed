"""
mc_bickley_jet.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (proposed text in
          Draft_13_unsteady_dg_proposed.tex; nothing here is in the paper).
  Makes:  noise-free success, retention, reversal, and hold statistics for both
          trackers along a curved cat's-eye separatrix of a Bickley jet; plus
          (--geom) the field geometry and trench offsets.
  Reads:  nothing (new simulation runs).
  Writes: experiments/unsteady_dg/outputs/bickley_<tag>.csv

FIELD
  Single-mode Bickley jet seen in the frame moving with its wave (phase
  speed c), which makes it steady. Nondimensional, scaled to the gyre
  benchmark (speeds ~0.3, separatrix length ~1):
      psi = -U L tanh(y/L) + c y + U L eps sech^2(y/L) cos(k (x + 0.5))
      u = -d psi/dy,  v = d psi/dx
  The northern critical layer (U sech^2 = c) carries a row of cat's-eye
  eddies. The tracked separatrix is the LOWER arc of the eye centred near
  x = 0: the curved heteroclinic orbit from the saddle at x = -0.5 to the
  saddle at x = +0.5. The flow on it runs east, so the downstream saddle is
  the one at x = +0.5 (P1 below) and the upstream one is at x = -0.5 (P2).
  The arc is traced numerically as the level set psi = psi(saddle).

EXPERIMENT
  Same cluster, gains, primitives, stopping rule, and scoring as
  mc_bent_double_gyre.py. Straddle: robots on both sides of the arc
  (sign of y_i - y_arc(x_i)). Error: vertical offset |y_c - y_arc(x_c)|.
  Seven starts B1..B7: three on the arc, four off it by 1.3 rho or 2 rho on
  either side. Random heading per trial; D and s1 share it.

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/mc_bickley_jet.py --geom
  venv/bin/python3 experiments/unsteady_dg/mc_bickley_jet.py --trials 500 --tag n500
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
from scipy.optimize import brentq

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import experiments._mc_common as mc

OUT_DIR = os.path.join(project_root, "experiments", "unsteady_dg", "outputs")
U, L, C_FRAC, EPS = 0.3, 0.2, 0.3, 0.3
C = C_FRAC * U
K = 2 * np.pi / 1.0
RHO = 0.075
G_PERP = 1.0
MAX_STEPS, HOLD_STEPS = 500, 100
CONTACT = mc.SADDLE_CONTACT_D
BAND, BAND_HOLD = mc.BAND_X, mc.BAND_HOLD


def psi(x, y):
    s2 = 1.0 / np.cosh(y / L) ** 2
    return -U * L * np.tanh(y / L) + C * y + U * L * EPS * s2 * np.cos(K * (x + 0.5))


def bickley(x, y, t=0.0, config=None):
    s2 = 1.0 / np.cosh(y / L) ** 2
    th = np.tanh(y / L)
    cs = np.cos(K * (x + 0.5))
    u = U * s2 - C + 2.0 * U * EPS * cs * s2 * th
    v = -U * L * EPS * s2 * K * np.sin(K * (x + 0.5))
    return float(u), float(v)


def _u_at(x, y):
    return bickley(x, y)[0]


def _stagnation_y(x, lo=0.0, hi=0.6):
    return brentq(lambda y: _u_at(x, y), lo, hi)


# geometry, computed once at import
Y_SADDLE = _stagnation_y(-0.5)          # cos = +1 at x = -0.5, +0.5
Y_CENTER = _stagnation_y(0.0)           # cos = -1 at x = 0
P2 = (-0.5, Y_SADDLE)                   # upstream saddle
P1 = (0.5, Y_SADDLE)                    # downstream saddle
PSI_S = psi(-0.5, Y_SADDLE)
_XG = np.linspace(-0.5, 0.5, 401)


def _trace_arc():
    """Lower arc: the eastward branch of the upstream saddle's unstable
    manifold, integrated by RK4 in arc length until it reaches x = 0.5."""
    h = 1e-6
    x0, y0 = P2
    J = np.array([[(bickley(x0 + h, y0)[0] - bickley(x0 - h, y0)[0]) / (2 * h),
                   (bickley(x0, y0 + h)[0] - bickley(x0, y0 - h)[0]) / (2 * h)],
                  [(bickley(x0 + h, y0)[1] - bickley(x0 - h, y0)[1]) / (2 * h),
                   (bickley(x0, y0 + h)[1] - bickley(x0, y0 - h)[1]) / (2 * h)]])
    w, V = np.linalg.eig(J)
    e = np.real(V[:, int(np.argmax(np.real(w)))])
    if e[0] < 0:
        e = -e

    def f(p):
        u, v = bickley(p[0], p[1])
        n = np.hypot(u, v)
        return np.array([u, v]) / n

    p = np.array([x0, y0]) + 1e-4 * e
    pts = [np.array([x0, y0]), p.copy()]
    ds = 5e-4
    for _ in range(20000):
        k1 = f(p); k2 = f(p + 0.5 * ds * k1); k3 = f(p + 0.5 * ds * k2); k4 = f(p + ds * k3)
        p = p + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(p.copy())
        if p[0] >= 0.5 - 1e-3:
            break
    pts = np.array(pts)
    return pts


ARC = _trace_arc()
assert np.all(np.diff(ARC[:, 0]) > 0), "arc x not monotone"
_YG = np.interp(_XG, np.append(ARC[:, 0], P1[0]), np.append(ARC[:, 1], P1[1]))


def arc_y(x):
    """Arc height; outside the saddles, hold the saddle height."""
    return np.interp(x, _XG, _YG)


def _starts():
    s = {}
    for name, x, off in (("B1", -0.30, 0.0), ("B2", 0.0, 0.0), ("B3", 0.30, 0.0),
                         ("B4", -0.25, 2.0 * RHO), ("B5", -0.25, -2.0 * RHO),
                         ("B6", 0.15, 1.3 * RHO), ("B7", 0.15, -1.3 * RHO)):
        s[name] = (x, float(arc_y(x)) + off)
    return s


STARTS = _starts()


def run_trial(spec):
    rng = np.random.default_rng(spec["seed"])
    np.random.seed(spec["seed"] % (2**32))
    heading = rng.uniform(0.0, 2 * np.pi)
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(bickley)
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)
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

    err_h, x_h, strad_h, rms = [], [], [], []
    outcome = "timeout"
    for k in range(MAX_STEPS):
        cluster.move(prim)
        cx, cy = cluster.get_centroid()
        rp = np.array(cluster.get_robot_positions()).reshape(6, 2)
        side = rp[:, 1] - arc_y(rp[:, 0])
        err_h.append(abs(cy - arc_y(cx)))
        x_h.append(cx)
        strad_h.append(bool(side.min() < 0 < side.max()))
        if k % 10 == 0:
            rms.append(float(np.sqrt(np.mean((mc._pairdists(rp) - nominal) ** 2))))
        if np.hypot(cx - P1[0], cy - P1[1]) < CONTACT:
            outcome = "p1"
            break
        if np.hypot(cx - P2[0], cy - P2[1]) < CONTACT:
            outcome = "p2"
            break
        if abs(cx) > 1.0 or abs(cy - Y_SADDLE) > 0.6:
            outcome = "exit"
            break

    hold_frac = np.nan
    if outcome == "p1":
        dd = []
        for _ in range(HOLD_STEPS):
            cluster.move(prim)
            cx, cy = cluster.get_centroid()
            dd.append(np.hypot(cx - P1[0], cy - P1[1]))
        hold_frac = float(np.mean(np.array(dd) < CONTACT))

    collapsed = max(rms) > mc.COLLAPSE_RMS if rms else False
    err = np.array(err_h)
    inband = err < BAND
    acq = -1
    for k in range(max(1, len(inband) - BAND_HOLD + 1)):
        if inband[k:k + BAND_HOLD].all():
            acq = k
            break
    strad = np.array(strad_h)
    first = int(np.argmax(strad)) if strad.any() else -1
    kept = bool(strad.any() and strad[first:].all())
    lost_x = np.nan
    if strad.any() and not kept:
        lost_x = float(x_h[first + int(np.argmin(strad[first:]))])
    success = outcome == "p1" and not collapsed
    seg = err[acq:] if acq >= 0 else err[-50:]
    fc = cluster.get_centroid()
    return {
        "tracker": spec["tracker"], "start": spec["start"], "trial": spec["trial"],
        "seed": spec["seed"], "heading": round(heading, 5), "outcome": outcome,
        "success": int(success), "reversal": int(outcome == "p2"),
        "retained": int(success and kept), "collapsed": int(collapsed),
        "steps": len(err_h), "acq_step": acq, "track_mean": round(float(seg.mean()), 6),
        "hold_frac": hold_frac, "final_x": round(float(fc[0]), 5),
        "final_y": round(float(fc[1]), 5), "lost_x": lost_x,
    }


def geom():
    h = 1e-5

    def jac(x, y):
        up, um = bickley(x + h, y), bickley(x - h, y)
        vp, vm = bickley(x, y + h), bickley(x, y - h)
        return ((up[0] - um[0]) / (2 * h), (vp[0] - vm[0]) / (2 * h),
                (up[1] - um[1]) / (2 * h), (vp[1] - vm[1]) / (2 * h))

    ux, uy, vx, vy = jac(*P1)
    print(f"saddles y = {Y_SADDLE:.4f} at x = -0.5, +0.5; eye centre y = {Y_CENTER:.4f}")
    print(f"det J at saddle {ux * vy - uy * vx:+.4f} (negative = saddle); "
          f"strain rate there {np.sqrt(max(0.0, -(ux * vy - uy * vx))):.4f}")
    ux, uy, vx, vy = jac(0.0, Y_CENTER)
    print(f"det J at centre {ux * vy - uy * vx:+.4f}")
    print(f"lower arc: lowest y {_YG.min():.4f} at x = {_XG[np.argmin(_YG)]:+.3f}; "
          f"dip below saddles {Y_SADDLE - _YG.min():.4f}; eye height at x=0 "
          f"{2 * (Y_CENTER - _YG[200]):.4f} (rho = {RHO})")
    slope = np.gradient(_YG, _XG)
    print(f"arc max |slope| {np.abs(slope).max():.3f}; arc length "
          f"{np.sum(np.hypot(np.diff(_XG), np.diff(_YG))):.3f}")
    for x in (-0.3, 0.0, 0.3):
        u, v = bickley(x, float(arc_y(x)))
        print(f"  flow on arc at x={x:+.2f}: (u, v) = ({u:+.4f}, {v:+.4f}), "
              f"arc slope {np.interp(x, _XG, slope):+.4f}, v/u {v / u:+.4f}")
    vort, offD, offS = [], [], []
    for x in np.linspace(-0.4, 0.4, 17):
        ya = float(arc_y(x))
        ux, uy, vx, vy = jac(x, ya)
        vort.append(abs(vx - uy))
        yy = np.linspace(ya - 0.15, ya + 0.15, 1201)
        Dv, Sv = [], []
        for y in yy:
            a, b, c2, d = jac(x, y)
            Dv.append(a * d - b * c2)
            Sv.append(0.5 * (a + d) - np.hypot(0.5 * (a - d), 0.5 * (b + c2)))
        for F, out in ((np.array(Dv), offD), (np.array(Sv), offS)):
            i = np.where((F[1:-1] < F[:-2]) & (F[1:-1] < F[2:]))[0] + 1
            out.append(np.nan if len(i) == 0 else float(np.min(np.abs(yy[i] - ya))))
    print(f"vorticity on arc: max {max(vort):.4f}, min {min(vort):.4f}")
    print(f"nearest D trench to arc (vertical): max {np.nanmax(offD):.4f}, "
          f"mean {np.nanmean(offD):.4f}, missing at {int(np.sum(np.isnan(offD)))} of 17 x")
    print(f"nearest s1 trench to arc (vertical): max {np.nanmax(offS):.4f}, "
          f"mean {np.nanmean(offS):.4f}, missing at {int(np.sum(np.isnan(offS)))} of 17 x")
    print("starts:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in STARTS.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--geom", action="store_true")
    args = ap.parse_args()
    if args.geom:
        geom()
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
    for js, s in enumerate(STARTS):
        for t in range(args.trials):
            seed = (3_000_017 + 7919 * js + 104_729 * t) % (2**31)
            for tr in ("D", "s1"):
                specs.append({"tracker": tr, "start": s, "trial": t, "seed": seed})
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(run_trial, specs, chunksize=4))
    rows.sort(key=lambda r: (r["start"], r["trial"], r["tracker"]))
    out = os.path.join(OUT_DIR, f"bickley_{args.tag}.csv")
    with open(out, "w", newline="") as f:
        f.write(f"# generated_by: experiments/unsteady_dg/mc_bickley_jet.py\n"
                f"# git_commit: {commit}\n"
                f"# date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"# trials_per_cell: {args.trials}  U={U} L={L} c={C} eps={EPS} k={K:.4f}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} trials, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
