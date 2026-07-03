"""
park_vs_traverse_demo.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  the park-versus-traverse demonstration figure for the Discussion
          (park_vs_traverse.png; no draft label yet) and the numbers quoted
          with it. Review copy + CSV in experiments/outputs/separatrix_clean/.

EXPERIMENT
  Same start, two runs on the steady double gyre:
    TRAVERSE: plain Logic C (library primitive, untouched). The H_D
      structural bias leaves the estimated eigenvalues indefinite near the
      saddle wells, so the eigenvalue-based capture test never fires and
      the team continues onto the wall trench (clean_runs.csv).
    PARK: a script-level wrapper adds one estimator-aware capture test
      BEFORE Logic C, built only from quantities the estimator delivers
      reliably (Fig. est_accuracy): if D_hat < D_capture (deep in a well
      of D), command saturated gradient descent on D, which converges to
      the well minimum, the flow saddle. One threshold selects the
      behavior: D_capture = -inf reproduces pure traversal.
  The library controller is NOT modified; this is the prototype for the
  selector change to be proposed to the user (plan.md Phase 2.4).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/park_vs_traverse_demo.py
"""
import os
import sys
import subprocess
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X)
from src.control.pentagon_primitives import (
    separatrix_logic_c_step, _sample_vector_at_robots,
    _get_relative_positions, _fit_vector_quadratic, _det_value,
    _det_gradient)

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
SIM_STEPS = 500
# S4 of separatrix_clean_runs.py: D(start) ~ -0.24 > D_capture, so the
# capture test is quiet at the start (unlike S2, where the top-saddle well
# captures immediately) and fires only on arrival at the bottom well.
START = (0.10, -0.20)
D_CAPTURE = -0.5              # ~half the well depth pi^4 A^2 = 0.974
A = 0.1

OUT_DIR = os.path.join(project_root, "experiments", "outputs",
                       "separatrix_clean")
os.makedirs(OUT_DIR, exist_ok=True)


def run(d_capture):
    """d_capture = None -> plain Logic C (traverse)."""
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(*START)
    n_park = [0]

    def prim(c):
        if d_capture is not None:
            # Estimator-aware capture test on reliable quantities only.
            # (Duplicates the fit; noise-free demo, identical values.)
            u_arr, v_arr = _sample_vector_at_robots(c)
            rel = _get_relative_positions(c)
            tu, tv = _fit_vector_quadratic(rel, u_arr, v_arr)
            if _det_value(tu, tv) < d_capture:
                g = _det_gradient(tu, tv)
                n_park[0] += 1
                vx = -V_MAX * np.tanh(g[0] / V_MAX)
                vy = -V_MAX * np.tanh(g[1] / V_MAX)
                return vx * GAIN, vy * GAIN
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX,
                                         eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * GAIN, vy * GAIN

    for _ in range(SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        # y margin 0.56, not 0.52: the well minimum lies ON the domain
        # boundary y = -0.5, and park-convergence transients cross it.
        if abs(cx) > 1.0 or abs(cy) > 0.56:
            break
    return cl, n_park[0]


def main():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"

    results = {}
    for name, dc in (("traverse", None), ("park", D_CAPTURE)):
        cl, n_park = run(dc)
        h = cl.get_center_history()
        d_bot = np.linalg.norm(h - np.array(SADDLE_BOTTOM), axis=1)
        tail = h[-100:]
        results[name] = {
            "hist": h, "final": h[-1], "min_d_bot": d_bot.min(),
            "final_d_bot": d_bot[-1],
            "tail_std": tail.std(axis=0).max(), "steps": len(h),
            "n_park_steps": n_park,
        }
        r = results[name]
        print(f"{name:>8}: steps={r['steps']} "
              f"final=({r['final'][0]:+.3f},{r['final'][1]:+.3f}) "
              f"final_d_saddle={r['final_d_bot']:.4f} "
              f"tail_std={r['tail_std']:.5f} park_steps={r['n_park_steps']}")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    gx = np.linspace(-1, 1, 200); gy = np.linspace(-0.5, 0.5, 100)
    GX, GY = np.meshgrid(gx, gy)
    U = np.empty_like(GX); Vv = np.empty_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            U[i, j], Vv[i, j] = double_gyre_static(GX[i, j], GY[i, j])
    ax.streamplot(GX, GY, U, Vv, color="0.85", density=0.9, linewidth=0.5,
                  arrowsize=0.6)
    ax.axvline(SEPARATRIX_X, color="0.4", ls="--", lw=0.9)
    ax.plot(*SADDLE_BOTTOM, marker="x", color="k", ms=9, mew=2, zorder=6)
    ax.plot(*SADDLE_TOP, marker="x", color="k", ms=9, mew=2, zorder=6)

    for name, col in (("traverse", "#2a78d6"), ("park", "#e34948")):
        h = results[name]["hist"]
        ax.plot(h[:, 0], h[:, 1], color=col, lw=1.7, label=name, zorder=5)
        ax.plot(h[-1, 0], h[-1, 1], marker="s", color=col, ms=6,
                mec="k", mew=0.8, zorder=7)
    ax.plot(*START, marker="o", color="k", ms=6, zorder=7)
    ax.legend(loc="center left", frameon=False, fontsize=8)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.55, 0.53)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "park_vs_traverse.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv = os.path.join(OUT_DIR, "park_vs_traverse.csv")
    with open(csv, "w") as f:
        f.write(f"# generated_by: experiments/park_vs_traverse_demo.py\n"
                f"# git_commit: {commit}\n# date: {stamp}\n"
                f"# start: {START}  D_capture: {D_CAPTURE}\n")
        f.write("mode,steps,final_x,final_y,min_d_bot,final_d_bot,"
                "tail_std,park_steps\n")
        for name, r in results.items():
            f.write(f"{name},{r['steps']},{r['final'][0]:.4f},"
                    f"{r['final'][1]:.4f},{r['min_d_bot']:.4f},"
                    f"{r['final_d_bot']:.4f},{r['tail_std']:.5f},"
                    f"{r['n_park_steps']}\n")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
