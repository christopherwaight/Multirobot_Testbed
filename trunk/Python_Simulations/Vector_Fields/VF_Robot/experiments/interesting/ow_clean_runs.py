"""
ow_clean_runs.py

PAPER TRACEABILITY
  Paper:  NONE. This script belongs to the Okubo-Weiss boundary-tracker
          thread, which was cut from the paper; the current draft
          (Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex) presents the
          D tracker and the s1 tracker and carries no OW boundary-following
          results. Kept for the record, not regenerated for the paper. It
          formerly fed fig:ow_trajectories of Paper_Draft_2A.tex.
  Makes:  Fig. \\ref{fig:ow_trajectories} (figures/ow_trajectories.png) of
          that superseded draft; review copy + per-run CSV in
          experiments/outputs/ow_clean/.
  Also produces the corner-event numbers for the Failure Modes discussion
  (corner hop onto the neighboring tile's zero line).

EXPERIMENT
  Noise-free Okubo-Weiss boundary tracking (logic_g_newton_contour_pentagon,
  perp-gradient tangent) on the steady double gyre from four starts:
  outside and inside the left diamond (approach behavior) and above both
  gyre centers (clean circuits). Loop-closure stopping rule: once the
  centroid first reaches the boundary (analytic |D| < D_ON), an anchor is
  dropped; the run stops when the centroid returns within CLOSE_R of the
  anchor after at least MIN_LOOP_STEPS more steps (circumnavigation), at
  domain exit, or at SIM_STEPS. Reported per run: time-to-boundary,
  circumnavigation time, mean/p95 |D| on-boundary, corner dwell (steps
  with analytic ||grad D|| < G_CORNER while on-boundary).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/ow_clean_runs.py
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
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP)
from src.control.pentagon_primitives import logic_g_newton_contour_pentagon

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_GRAD = 1e-6
SIM_STEPS = 1500
A = 0.1

D_ON = 0.02            # |D| threshold: on the boundary
CLOSE_R = 0.06         # loop-closure radius around the anchor
MIN_LOOP_STEPS = 200   # steps after anchor before closure can trigger
G_CORNER = 0.5         # ||grad D|| threshold for corner dwell (max ~ 4.3)

STARTS = [
    ("O1", -0.70,  0.00, "outside left diamond"),
    ("O2", -0.30,  0.00, "inside left diamond"),
    ("O3", -0.50,  0.25, "left gyre, above center"),
    ("O4",  0.50,  0.25, "right gyre, above center"),
]
COLORS = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "ow_clean")
os.makedirs(OUT_DIR, exist_ok=True)


def D_true(x, y):
    return -(np.pi**4 * A**2 / 2) * (np.cos(2*np.pi*(x + 1))
                                     + np.cos(2*np.pi*(y + 0.5)))


def g_true_norm(x, y):
    return np.pi**5 * A**2 * np.hypot(np.sin(2*np.pi*(x + 1)),
                                      np.sin(2*np.pi*(y + 0.5)))


def run_one(sx, sy):
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(sx, sy)

    def prim(c):
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=V_MAX,
                                                 eps_grad=EPS_GRAD)
        return vx * GAIN, vy * GAIN

    anchor, anchor_step, closed_step = None, -1, -1
    for k in range(SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > 1.05 or abs(cy) > 0.55:
            break
        if anchor is None:
            if abs(D_true(cx, cy)) < D_ON:
                anchor, anchor_step = np.array([cx, cy]), k
        elif (k - anchor_step > MIN_LOOP_STEPS
              and np.hypot(cx - anchor[0], cy - anchor[1]) < CLOSE_R):
            closed_step = k
            break
    return cl, anchor_step, closed_step


def analyze(cl, anchor_step, closed_step):
    h = cl.get_center_history()
    Dv = np.array([D_true(x, y) for x, y in h])
    on = np.abs(Dv) < D_ON
    stats = {"t_boundary": anchor_step, "t_closed": closed_step,
             "steps": len(h), "final": h[-1]}
    if anchor_step >= 0:
        seg = slice(anchor_step, closed_step if closed_step > 0 else len(h))
        absD = np.abs(Dv[seg])
        gnorm = np.array([g_true_norm(x, y) for x, y in h[seg]])
        stats.update({
            "meanD": absD.mean(), "p95D": np.percentile(absD, 95),
            "corner_dwell": int(np.sum(gnorm < G_CORNER)),
            "loop_steps": (closed_step - anchor_step
                           if closed_step > 0 else -1),
        })
    return stats, h


def main():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    gx = np.linspace(-1, 1, 240); gy = np.linspace(-0.5, 0.5, 120)
    GX, GY = np.meshgrid(gx, gy)
    Dfield = D_true(GX, GY)
    ax.contourf(GX, GY, Dfield, levels=14, cmap="RdBu_r", alpha=0.25)
    ax.contour(GX, GY, Dfield, levels=[0.0], colors="0.3", linewidths=1.2)
    for sad in (SADDLE_BOTTOM, SADDLE_TOP):
        ax.plot(*sad, marker="x", color="k", ms=8, mew=2, zorder=6)
    ax.plot([-0.5, 0.5], [0, 0], marker="+", color="k", ms=9, mew=1.5,
            ls="none", zorder=6)

    rows = []
    for (name, sx, sy, desc), col in zip(STARTS, COLORS):
        cl, a_step, c_step = run_one(sx, sy)
        st, h = analyze(cl, a_step, c_step)
        ax.plot(h[:, 0], h[:, 1], color=col, lw=1.6, zorder=5)
        ax.plot(sx, sy, marker="o", color=col, ms=6, mec="k", mew=0.8,
                zorder=7)
        ax.plot(h[-1, 0], h[-1, 1], marker="s", color=col, ms=6, mec="k",
                mew=0.8, zorder=7)
        ax.annotate(name, (sx, sy), textcoords="offset points",
                    xytext=(6, 5), fontsize=8, color=col)
        rows.append((name, sx, sy, desc, st))
        print(f"{name} ({desc}): t_boundary={st['t_boundary']} "
              f"t_closed={st['t_closed']} steps={st['steps']} "
              + (f"loop={st.get('loop_steps')} meanD={st.get('meanD'):.4f} "
                 f"p95D={st.get('p95D'):.4f} corner_dwell="
                 f"{st.get('corner_dwell')}"
                 if st['t_boundary'] >= 0 else "never reached boundary"))

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-1, -0.5), 2, 1, fill=False, ec="0.25",
                           lw=0.8, zorder=3))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.55, 0.53)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "ow_trajectories.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv = os.path.join(OUT_DIR, "ow_clean_runs.csv")
    with open(csv, "w") as f:
        f.write(f"# generated_by: experiments/ow_clean_runs.py\n"
                f"# git_commit: {commit}\n# date: {stamp}\n"
                f"# steps_max: {SIM_STEPS}  v_max: {V_MAX}  gain: {GAIN}  "
                f"D_on: {D_ON}  close_r: {CLOSE_R}\n")
        f.write("run,start_x,start_y,description,t_boundary,t_closed,"
                "loop_steps,mean_absD,p95_absD,corner_dwell,steps,"
                "final_x,final_y\n")
        for name, sx, sy, desc, st in rows:
            f.write(f"{name},{sx},{sy},{desc},{st['t_boundary']},"
                    f"{st['t_closed']},{st.get('loop_steps', -1)},"
                    f"{st.get('meanD', float('nan')):.5f},"
                    f"{st.get('p95D', float('nan')):.5f},"
                    f"{st.get('corner_dwell', -1)},{st['steps']},"
                    f"{st['final'][0]:.4f},{st['final'][1]:.4f}\n")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    main()
