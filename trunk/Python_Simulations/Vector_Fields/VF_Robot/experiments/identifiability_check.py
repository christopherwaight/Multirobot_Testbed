"""
identifiability_check.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the incompressibility remark (or corollary) after the
          minimality paragraph of Section III-A: whether five robots
          suffice to identify the quadratic model once exact
          incompressibility is imposed (SIM-4 of the T-RO hardening
          plan).
  Reads:  nothing. Writes experiments/outputs/identifiability/report.txt.

EXPERIMENT
  The unconstrained planar quadratic model has 12 coefficients and needs
  six robots (Proposition prop:minimality).  Imposing exact
  incompressibility (u_x + v_y = 0, u_xx + v_xy = 0, u_xy + v_yy = 0)
  removes three coefficients, leaving
      theta = [u0, ux, uy, uxx, uxy, uyy, v0, vx, vxx]  (9 unknowns),
  and each robot contributes one u-equation and one coupled v-equation,
  so five robots supply a 10 x 9 linear system.  This script decides
  whether that system is generically rank 9.

  Method:
  1. Exact-arithmetic rank at random rational placements (sympy).  Rank
     is maximal on a Zariski-open set, so a single exact placement with
     rank 9 proves the generic rank is 9; rank deficiency at every
     placement of a parameterized family proves the family degenerate.
  2. Numeric rank and conditioning over N random placements (numpy).
  3. Degenerate families, all with exact rational coordinates: five
     collinear points, five cocircular points via the tan-half-angle
     parameterization (covers any five-robot subset of a circular ring),
     four-on-a-circle plus center, and five points on a parabola.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/identifiability_check.py
"""
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import sympy as sp

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

OUT_DIR = os.path.join(project_root, "experiments", "outputs",
                       "identifiability")
os.makedirs(OUT_DIR, exist_ok=True)

N_NUMERIC = 10_000
SEED = 20260717
RANK_TOL = 1e-10


def rows_for_point(x, y):
    """u-row and v-row of the constrained system at sample point (x, y).

    theta = [u0, ux, uy, uxx, uxy, uyy, v0, vx, vxx]
    u(x,y) = u0 + ux x + uy y + uxx x^2/2 + uxy x y + uyy y^2/2
    v(x,y) = v0 + vx x - ux y + vxx x^2/2 - uxx x y - uxy y^2/2
    """
    u_row = [1, x, y, x**2 / 2, x * y, y**2 / 2, 0, 0, 0]
    v_row = [0, -y, 0, -x * y, -(y**2) / 2, 0, 1, x, x**2 / 2]
    return u_row, v_row


def build_matrix(points, mat_ctor):
    rows = []
    for (x, y) in points:
        u_row, v_row = rows_for_point(x, y)
        rows.append(u_row)
        rows.append(v_row)
    return mat_ctor(rows)


def exact_rank(points):
    return build_matrix(points, sp.Matrix).rank()


def numeric_rank_and_cond(points):
    M = build_matrix(points, np.array).astype(float)
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > RANK_TOL * s[0]))
    cond = s[0] / s[8] if s[8] > 0 else np.inf
    return rank, cond


def circle_point(R, t):
    """Exact rational point on the circle of radius R: tan-half-angle."""
    return (R * (1 - t**2) / (1 + t**2), R * 2 * t / (1 + t**2))


def main():
    rng = np.random.default_rng(SEED)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    log("# generated_by: experiments/identifiability_check.py")
    log(f"# git_commit: {commit}")
    log(f"# date: {datetime.now(timezone.utc).isoformat()}")
    log(f"# seed: {SEED}   numeric_placements: {N_NUMERIC}")
    log("")

    # 1. Exact rational placements: any rank-9 hit proves generic rank 9.
    log("1. Exact-arithmetic rank at random rational placements:")
    for trial in range(10):
        pts = [(sp.Rational(int(rng.integers(-1000, 1000)), 997),
                sp.Rational(int(rng.integers(-1000, 1000)), 991))
               for _ in range(5)]
        r = exact_rank(pts)
        log(f"   placement {trial + 1}: rank = {r}")

    # 2. Numeric rank and conditioning over random placements.
    log("")
    log(f"2. Numeric rank over {N_NUMERIC} uniform placements in [-1,1]^2:")
    ranks = np.zeros(N_NUMERIC, dtype=int)
    conds = np.zeros(N_NUMERIC)
    for i in range(N_NUMERIC):
        pts = rng.uniform(-1, 1, size=(5, 2))
        ranks[i], conds[i] = numeric_rank_and_cond([tuple(p) for p in pts])
    for r in np.unique(ranks):
        log(f"   rank {r}: {np.sum(ranks == r)} placements")
    if np.any(ranks == 9):
        log(f"   cond (rank-9 cases): median "
            f"{np.median(conds[ranks == 9]):.3g}, "
            f"p95 {np.percentile(conds[ranks == 9], 95):.3g}, "
            f"max {np.max(conds[ranks == 9]):.3g}")

    # 3. Degenerate families (exact rational arithmetic).
    log("")
    log("3. Degenerate families (exact rational arithmetic):")

    t_vals = [sp.Rational(k, 7) for k in (-3, -1, 0, 2, 5)]
    a, b = sp.Rational(2, 3), sp.Rational(1, 5)
    collinear = [(t, a * t + b) for t in t_vals]
    log(f"   five collinear points:            rank = {exact_rank(collinear)}")

    R = sp.Rational(3, 4)
    ts = [sp.Rational(k, 13) for k in (-9, -2, 0, 3, 7)]
    cocirc = [circle_point(R, t) for t in ts]
    log(f"   five cocircular points:           rank = {exact_rank(cocirc)}")

    ts4 = [sp.Rational(k, 11) for k in (-7, -1, 2, 6)]
    four_plus_center = [circle_point(R, t) for t in ts4]
    four_plus_center.append((sp.Integer(0), sp.Integer(0)))
    log(f"   four on a circle plus center:     rank = "
        f"{exact_rank(four_plus_center)}")

    parab = [(t, t**2) for t in t_vals]
    log(f"   five points on a parabola:        rank = {exact_rank(parab)}")

    out_path = os.path.join(OUT_DIR, "report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log("")
    log(f"written: {out_path}")


if __name__ == "__main__":
    main()
