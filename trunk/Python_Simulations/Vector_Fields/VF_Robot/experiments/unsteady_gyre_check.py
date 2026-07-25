"""
unsteady_gyre_check.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the unsteady double-gyre paragraph of Section VII-E (SIM-5 of
          the T-RO hardening plan): does the D tracker hold the
          instantaneous trench of a time-varying field with no
          time-derivative term in the controller?
  Reads:  nothing. Writes experiments/outputs/unsteady_gyre/
          run_eps<eps>.csv and prints the summary.

EXPERIMENT
  Shadden double gyre with eps = 0.1, omega = 2*pi/10 (period 10 s),
  Logic C from the standard straddling start (0, 0.35), zero noise,
  600 steps at dt = 0.1 s.  Ground truth per step is the instantaneous
  trench of D(t): the minimizer of D(x, y_c, t) over x near the
  centroid's y.  The tracker carries no time-derivative correction, so
  any lag it develops is measured here.  The instantaneous separatrix
  swings roughly +/- 0.10 in x over the cycle; peak trench speed is
  comparable to the commanded speed cap, so this is a demanding case.

  PASS (plan SIM-5): mean transverse distance to the instantaneous
  trench <= 0.02 over the central-branch phase (|y_c| <= 0.45), with
  no loss of structure (no excursion beyond 0.1).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/unsteady_gyre_check.py
"""
import io
import contextlib
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import separatrix_logic_c_step
import experiments._mc_common as mc

EPS = 0.1
OMEGA = 2.0 * np.pi / 10.0
A_DG = 0.1
CFG = {"A": A_DG, "eps": EPS, "omega": OMEGA}

START = (0.0, 0.35)
STEPS = 600
DT = 0.1
PHASE_Y = 0.45          # central-branch phase: |y_c| <= PHASE_Y
LOSS_DIST = 0.10
FD_H = 1e-5


def d_field(x, y, t):
    h = FD_H
    def f(a, b):
        u, v = double_gyre_static(a, b, t, CFG)
        return np.array([u, v])
    fx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    fy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return fx[0] * fy[1] - fy[0] * fx[1]


def x_trench(y, t, x_lo=-0.30, x_hi=0.30, n=241):
    """Minimizer of D(x, y, t) over x, grid search + parabolic refine."""
    xs = np.linspace(x_lo, x_hi, n)
    ds = np.array([d_field(x, y, t) for x in xs])
    i = int(np.argmin(ds))
    if 0 < i < n - 1:
        d0, d1, d2 = ds[i - 1], ds[i], ds[i + 1]
        denom = d0 - 2 * d1 + d2
        if denom > 0:
            return xs[i] + 0.5 * (d0 - d2) / denom * (xs[1] - xs[0])
    return xs[i]


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(double_gyre_static)
        field.config = dict(CFG)
        field.reset_clock()
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)
    cluster.reset(START[0], START[1])
    cluster.measurement_noise_std = 0.0
    cluster.position_noise_std = 0.0

    def prim(c):
        vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX,
                                         eps_raw=mc.EPS_RAW,
                                         eps_dim=mc.EPS_DIM)
        return vx * mc.GAIN, vy * mc.GAIN

    records = []
    for k in range(STEPS):
        t = field.t
        cluster.move(prim)
        cx, cy = cluster.get_centroid()
        records.append((t, cx, cy))
        field.step(DT)
        if abs(cx) > 1.1 or abs(cy) > 0.6:
            break

    rows = []
    for (t, cx, cy) in records:
        if abs(cy) <= PHASE_Y:
            xt = x_trench(cy, t)
            rows.append((t, cx, cy, xt, abs(cx - xt)))

    dists = np.array([r[4] for r in rows])
    out_dir = os.path.join(project_root, "experiments", "outputs",
                           "unsteady_gyre")
    os.makedirs(out_dir, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    path = os.path.join(out_dir, f"run_eps{EPS:g}.csv")
    with open(path, "w") as f:
        f.write("# generated_by: experiments/unsteady_gyre_check.py\n")
        f.write(f"# git_commit: {commit}\n")
        f.write(f"# date: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# eps: {EPS}  omega: {OMEGA}  A: {A_DG}  start: {START}"
                f"  steps: {STEPS}  dt: {DT}\n")
        f.write(f"# phase: |y_c| <= {PHASE_Y}  controller: Logic C, "
                f"zero noise, no time-derivative term\n")
        f.write("t,x_c,y_c,x_trench,transverse_dist\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")

    # Trench speed context: max |d x_trench / dt| at the start's y.
    ts = np.linspace(0.0, 10.0, 201)
    xt_series = np.array([x_trench(START[1], t) for t in ts])
    trench_speed = np.max(np.abs(np.gradient(xt_series, ts)))

    print(f"steps total: {len(records)}  in central phase: {len(rows)}",
          flush=True)
    print(f"transverse distance: mean {dists.mean():.4f}  "
          f"p95 {np.percentile(dists, 95):.4f}  max {dists.max():.4f}",
          flush=True)
    print(f"trench swing at y={START[1]}: "
          f"{xt_series.min():.3f} to {xt_series.max():.3f}, "
          f"peak speed {trench_speed:.4f} m/s "
          f"(command cap {mc.V_MAX * mc.GAIN:.3f} m/s)", flush=True)
    lost = bool((dists > LOSS_DIST).any())
    verdict = "PASS" if (dists.mean() <= 0.02 and not lost) else "FAIL"
    print(f"structure lost (> {LOSS_DIST}): {lost}", flush=True)
    print(f"verdict: {verdict}  (gate: mean <= 0.02, no loss)", flush=True)
    print(f"written: {path}", flush=True)


if __name__ == "__main__":
    main()
