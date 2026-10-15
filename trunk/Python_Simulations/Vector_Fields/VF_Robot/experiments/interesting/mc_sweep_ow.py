"""
mc_sweep_ow.py

PAPER TRACEABILITY
  Paper:  NONE. This script belongs to the Okubo-Weiss boundary-tracker
          thread, which was cut from the paper; the current draft
          (Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex) presents the
          D tracker and the s1 tracker and carries no OW boundary-following
          results. Kept for the record, not regenerated for the paper. It
          formerly fed the OW success table of Paper_Draft_2A.tex.
  Makes:  the Okubo-Weiss boundary tracker success-rate table (and
          optional heatmap fig:ow_success) and |D| tracking statistics of
          that superseded draft's Results section.
  Writes: experiments/outputs/mc_ow/trials_<mode>.csv, summary_<mode>.csv.

EXPERIMENT (merged test plan; straight-through behavior per decision (a))
  logic_g_newton_contour_pentagon from ONE consistent start near the left
  diamond boundary, random heading, noise dialed up over the same 2D grid
  with auto-extension. Success = lock onto the D=0 line (analytic |D| <
  D_ON sustained) and stay on it (p95 |D| after lock below D_STAY) with no
  formation collapse; trials stop at domain exit (riding the line out of
  the domain is the intended behavior) or at the step budget.

Run (development):
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_ow.py --trials 100
Final run for the paper: --trials 10000.
"""
import argparse
import io
import os
import sys
import subprocess
import contextlib
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import logic_g_newton_contour_pentagon
import experiments._mc_common as mc

SIGMA_UV_BASE = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_P_BASE = [0.0, 0.005, 0.01, 0.02, 0.05]
EXTEND_UNTIL = 0.10
MAX_EXTENSIONS = 3

FIXED_START = (-0.5, 0.25)   # O3 of ow_clean_runs.py: clean lock-on
SIM_STEPS = 500
A = 0.1
D_ON, ON_HOLD = 0.02, 10     # lock-on: |D| < D_ON for ON_HOLD steps
D_STAY = 0.10                # stay-on-line: p95 |D| after lock < D_STAY

START_MODE = {"mode": "fixed"}

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_ow")
os.makedirs(OUT_DIR, exist_ok=True)


def D_true(x, y):
    return -(np.pi**4 * A**2 / 2) * (np.cos(2*np.pi*(x + 1))
                                     + np.cos(2*np.pi*(y + 0.5)))


def run_trial_ow(spec):
    np.random.seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])

    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(double_gyre_static)
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)

    if spec.get("start") is not None:
        x0, y0 = spec["start"]
    else:
        x0 = rng.uniform(mc.START_BOX[0], mc.START_BOX[1])
        y0 = rng.uniform(mc.START_BOX[2], mc.START_BOX[3])
    heading = rng.uniform(0.0, 2 * np.pi)
    cluster.reset(x0, y0, heading_offset=heading)
    cluster.measurement_noise_std = spec["sigma_uv"]
    cluster.position_noise_std = spec["sigma_p"]

    effort = 0.0

    def prim(c):
        nonlocal effort
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=mc.V_MAX,
                                                 eps_grad=1e-6)
        vx, vy = vx * mc.GAIN, vy * mc.GAIN
        effort += np.hypot(vx, vy) * c.timestep
        return vx, vy

    exited = False
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        cx, cy = cluster.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.52:
            exited = True
            break

    hist = cluster.get_center_history()
    rhist = cluster.get_robot_history()
    absD = np.abs(np.array([D_true(x, y) for x, y in hist]))

    on = absD < D_ON
    t_lock = -1
    for k in range(max(1, len(on) - ON_HOLD)):
        if on[k:k + ON_HOLD].all():
            t_lock = k
            break

    rms_max = 0.0
    nominal = mc._NOMINAL_PDIST
    if nominal is None:
        with contextlib.redirect_stdout(io.StringIO()):
            f2 = AnalyticalField(double_gyre_static)
            c2 = PentagonCluster(mc.FORMATION_CONFIG, f2)
        c2.reset(0.0, 0.0)
        pts0 = np.array(c2.get_robot_positions()).reshape(6, 2)
        nominal = mc._pairdists(pts0)
        mc._NOMINAL_PDIST = nominal
    for k in range(0, len(rhist), 10):
        err = mc._pairdists(rhist[k]) - nominal
        rms_max = max(rms_max, float(np.sqrt(np.mean(err ** 2))))
    collapsed = rms_max > mc.COLLAPSE_RMS

    if t_lock >= 0 and len(absD) - t_lock > 5:
        seg = absD[t_lock:]
        d_mean, d_p95 = float(seg.mean()), float(np.percentile(seg, 95))
    else:
        d_mean, d_p95 = float("nan"), float("nan")

    success = int(t_lock >= 0 and not collapsed
                  and d_p95 == d_p95 and d_p95 < D_STAY)
    return {
        "sigma_uv": spec["sigma_uv"], "sigma_p": spec["sigma_p"],
        "seed": spec["seed"], "start_x": x0, "start_y": y0,
        "heading": heading, "t_lock": t_lock, "steps": len(hist),
        "success_track": success, "exited_domain": int(exited),
        "collapsed": int(collapsed), "d_mean": d_mean, "d_p95": d_p95,
        "shape_rms_max": rms_max, "effort": effort,
        "final_x": float(hist[-1, 0]), "final_y": float(hist[-1, 1]),
    }


def cell_specs(sigma_uv, sigma_p, n_trials):
    base = int(1e6 * sigma_uv * 1000 + 1e3 * sigma_p * 1000 + 17) % (2**31)
    start = FIXED_START if START_MODE["mode"] == "fixed" else None
    return [{"sigma_uv": sigma_uv, "sigma_p": sigma_p, "start": start,
             "seed": (base + 7919 * t) % (2**31)} for t in range(n_trials)]


def run_cells(pool, cells, n_trials, rows, summary):
    for s_uv, s_p in cells:
        out = list(pool.imap_unordered(run_trial_ow,
                                       cell_specs(s_uv, s_p, n_trials),
                                       chunksize=16))
        rows.extend(out)
        sc = np.mean([r["success_track"] for r in out])
        dm = np.nanmean([r["d_mean"] for r in out])
        dp = np.nanmean([r["d_p95"] for r in out])
        summary[(s_uv, s_p)] = (sc, dm, dp)
        print(f"  sigma_uv={s_uv:<6} sigma_p={s_p:<6} "
              f"track={sc:5.1%} mean|D|={dm:.4f} p95|D|={dp:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--starts", choices=["fixed", "random"],
                    default="fixed")
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

    header = (f"# generated_by: experiments/mc_sweep_ow.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  steps: {SIM_STEPS}  "
              f"controller: logic_g_newton_contour_pentagon\n"
              f"# start_mode: {args.starts}  fixed_start: {FIXED_START}\n"
              f"# D_on: {D_ON}  D_stay: {D_STAY}  "
              f"collapse_rms: {mc.COLLAPSE_RMS}\n")
    cols = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
            "t_lock", "steps", "success_track", "exited_domain",
            "collapsed", "d_mean", "d_p95", "shape_rms_max", "effort",
            "final_x", "final_y"]
    with open(os.path.join(OUT_DIR, f"trials_{args.starts}.csv"), "w") as f:
        f.write(header)
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float)
                             else str(r[c]) for c in cols) + "\n")
    with open(os.path.join(OUT_DIR, f"summary_{args.starts}.csv"),
              "w") as f:
        f.write(header)
        f.write("sigma_uv,sigma_p,success_track,d_mean,d_p95\n")
        for (u, p), (sc, dm, dp) in sorted(summary.items()):
            f.write(f"{u},{p},{sc:.4f},{dm:.5f},{dp:.5f}\n")
    print(f"Wrote {OUT_DIR}/trials_{args.starts}.csv ({len(rows)} rows) "
          f"and summary_{args.starts}.csv")


if __name__ == "__main__":
    main()
