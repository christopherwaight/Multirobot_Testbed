"""
main_separatrix_traverse.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the path-match figure for the objective separatrix traverser
          (figures/traverse_vs_logic_c.png); review copy and per-run CSV in
          experiments/outputs/oecs/.

EXPERIMENT
  6-robot pentagon cluster running Primitive 11 (oecs_separatrix_step, the
  objective / frame-invariant traverser of the full separatrix network) on
  the steady double gyre, overlaid against Primitive 7 (separatrix_logic_c_step,
  the non-objective D-trench tracker) from the SAME six starts used in
  separatrix_clean_runs.py, so the two families are directly comparable.

  Expected result (design target, see plan): the two paths coincide on this
  field, because omega = 0 along the entire double-gyre separatrix x = 0, so
  D = omega^2/4 - s1^2 = -s1^2 there -- the objective s1-trench and the
  non-objective D-trench are the same curve. The traverser should ride
  through BOTH separatrix halves and cross the origin isotropic point via
  its CROSS mode, rather than stopping there.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/main_separatrix_traverse.py
"""
import os
import sys
import subprocess
from collections import Counter
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
    separatrix_logic_c_step, oecs_separatrix_step)

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
G_PERP, S_TRIM, R_BAND = 1.0, 0.05, 0.05
SIM_STEPS = 600
A = 0.1
BAND_X = 0.05          # |x_c| threshold for time-to-band
BAND_HOLD = 10          # consecutive steps required
POST_SADDLE_STEPS = 150
SADDLE_CONTACT_D = 0.06

# Same starts as separatrix_clean_runs.py, so the two controllers are
# compared from identical initial conditions.
STARTS = [
    ("S1", -0.45,  0.30),
    ("S2",  0.05,  0.40),
    ("S3",  0.00,  0.00),
    ("S4",  0.10, -0.20),
    ("S5",  0.25,  0.42),
    ("S6", -0.20, -0.30),
]
COLORS = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e34948", "#eb6834", "#e87ba4"]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)


def run_logic_c(sx, sy):
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(sx, sy)

    def prim(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX,
                                         eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * GAIN, vy * GAIN

    contact = -1
    for k in range(SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.52:
            break
        if contact < 0:
            d = np.hypot(cx - SADDLE_BOTTOM[0], cy - SADDLE_BOTTOM[1])
            if d < SADDLE_CONTACT_D:
                contact = k
        elif k - contact >= POST_SADDLE_STEPS:
            break
    return cl


def run_traverser(sx, sy):
    field = AnalyticalField(double_gyre_static)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(sx, sy)

    def prim(c):
        vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP,
                                      s_trim=S_TRIM, r_band=R_BAND,
                                      s_capture=None)
        return vx * GAIN, vy * GAIN

    contact = -1
    for k in range(SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.52:
            break
        if contact < 0:
            d = np.hypot(cx - SADDLE_BOTTOM[0], cy - SADDLE_BOTTOM[1])
            if d < SADDLE_CONTACT_D:
                contact = k
        elif k - contact >= POST_SADDLE_STEPS:
            break
    return cl


def analyze(cl):
    hist = cl.get_center_history()
    xs, ys = hist[:, 0], hist[:, 1]
    inband = np.abs(xs) < BAND_X
    t_band = -1
    for k in range(len(inband) - BAND_HOLD):
        if inband[k:k + BAND_HOLD].all():
            t_band = k
            break
    d_bot = np.linalg.norm(hist - np.array(SADDLE_BOTTOM), axis=1)
    d_top = np.linalg.norm(hist - np.array(SADDLE_TOP), axis=1)
    cont = False
    for d_s, sad in ((d_bot, SADDLE_BOTTOM), (d_top, SADDLE_TOP)):
        hits = np.where(d_s < 0.06)[0]
        if len(hits):
            after = hist[hits[0]:]
            if np.any(np.abs(after[:, 0] - sad[0]) > 0.15):
                cont = True
    modes = Counter(d['mode'] for d in cl.diagnostics)
    return {
        "t_band": t_band, "final_x": xs[-1], "final_y": ys[-1],
        "min_d_bot": d_bot.min(), "min_d_top": d_top.min(),
        "continued": cont, "modes": modes, "hist": hist,
        "diag": cl.diagnostics,
    }


def path_gap(hist_a, hist_b):
    """
    Mean and max nearest-point distance from path A to path B (and vice
    versa), a simple symmetric measure of how well two trajectories
    coincide as CURVES (not requiring matched time steps, since the two
    controllers move through the band at different rates).
    """
    def one_way(a, b):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        return d.min(axis=1)
    d_ab = one_way(hist_a, hist_b)
    d_ba = one_way(hist_b, hist_a)
    all_d = np.concatenate([d_ab, d_ba])
    return float(all_d.mean()), float(all_d.max())


def main():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharex=True, sharey=True)
    ax_c, ax_t = axes

    gx = np.linspace(-1, 1, 240)
    gy = np.linspace(-0.5, 0.5, 120)
    GX, GY = np.meshgrid(gx, gy)
    U = np.empty_like(GX); Vv = np.empty_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            U[i, j], Vv[i, j] = double_gyre_static(GX[i, j], GY[i, j])
    Dfield = -(np.pi**4 * A**2 / 2) * (np.cos(2*np.pi*(GX+1))
                                       + np.cos(2*np.pi*(GY+0.5)))

    for ax, title in ((ax_c, "Controller 1: D tracker (Logic C)"),
                      (ax_t, "Controller 2: objective s1 traverser")):
        ax.streamplot(GX, GY, U, Vv, color="0.82", density=1.1, linewidth=0.6,
                      arrowsize=0.7)
        ax.contour(GX, GY, Dfield, levels=[0.0], colors="0.45", linewidths=1.0)
        ax.axvline(SEPARATRIX_X, color="0.35", ls="--", lw=1.0)
        for sad in (SADDLE_BOTTOM, SADDLE_TOP):
            ax.plot(*sad, marker="x", color="k", ms=9, mew=2, zorder=6)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.55, 0.53)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    ax_c.set_ylabel("y (m)")

    rows = []
    gaps = []
    for (name, sx, sy), col in zip(STARTS, COLORS):
        cl_c = run_logic_c(sx, sy)
        cl_t = run_traverser(sx, sy)
        r_c = analyze(cl_c)
        r_t = analyze(cl_t)
        mean_gap, max_gap = path_gap(r_c["hist"], r_t["hist"])
        gaps.append((name, mean_gap, max_gap))

        for ax, r in ((ax_c, r_c), (ax_t, r_t)):
            h = r["hist"]
            ax.plot(h[:, 0], h[:, 1], color=col, lw=1.7, zorder=5)
            ax.plot(sx, sy, marker="o", color=col, ms=6, mec="k", mew=0.8,
                    zorder=7)
            ax.plot(h[-1, 0], h[-1, 1], marker="s", color=col, ms=6, mec="k",
                    mew=0.8, zorder=7)
            dx, dy = (8, -11) if name == "S1" else (6, 5)
            ax.annotate(name, (sx, sy), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color=col)

        rows.append((name, sx, sy, r_c, r_t, mean_gap, max_gap))
        m_c, m_t = r_c["modes"], r_t["modes"]
        tot_c, tot_t = sum(m_c.values()), sum(m_t.values())
        print(f"{name} ({sx:+.2f},{sy:+.2f})")
        print(f"  Logic C:    t_band={r_c['t_band']:>3} "
              f"final=({r_c['final_x']:+.3f},{r_c['final_y']:+.3f}) "
              f"continued={r_c['continued']} modes="
              + ",".join(f"{k}:{v/tot_c:.2f}" for k, v in m_c.most_common()))
        print(f"  Traverser:  t_band={r_t['t_band']:>3} "
              f"final=({r_t['final_x']:+.3f},{r_t['final_y']:+.3f}) "
              f"continued={r_t['continued']} modes="
              + ",".join(f"{k}:{v/tot_t:.2f}" for k, v in m_t.most_common()))
        print(f"  path gap:   mean={mean_gap:.4f}  max={max_gap:.4f}")

    fig.suptitle("Objective separatrix traverser (Primitive 11) vs Logic C "
                 "(Primitive 7): path match", fontsize=13, fontweight='bold')
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "traverse_vs_logic_c.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv_path = os.path.join(OUT_DIR, "traverse_vs_logic_c.csv")
    with open(csv_path, "w") as f:
        f.write(f"# generated_by: experiments/main_separatrix_traverse.py\n"
                f"# git_commit: {commit}\n# date: {stamp}\n"
                f"# steps: {SIM_STEPS}  v_max: {V_MAX}  gain: {GAIN}  "
                f"r_band: {R_BAND}  s_trim: {S_TRIM}\n")
        f.write("run,start_x,start_y,"
                "logicc_t_band,logicc_final_x,logicc_final_y,logicc_continued,"
                "trav_t_band,trav_final_x,trav_final_y,trav_continued,"
                "path_gap_mean,path_gap_max\n")
        for name, sx, sy, r_c, r_t, mg, xg in rows:
            f.write(f"{name},{sx},{sy},"
                    f"{r_c['t_band']},{r_c['final_x']:.4f},{r_c['final_y']:.4f},"
                    f"{int(r_c['continued'])},"
                    f"{r_t['t_band']},{r_t['final_x']:.4f},{r_t['final_y']:.4f},"
                    f"{int(r_t['continued'])},"
                    f"{mg:.5f},{xg:.5f}\n")
    print(f"Saved: {csv_path}")

    print("\nPath-gap summary (mean nearest-point distance, both directions):")
    for name, mg, xg in gaps:
        print(f"  {name}: mean={mg:.4f}  max={xg:.4f}")
    print(f"  overall mean of means: {np.mean([g[1] for g in gaps]):.4f}")


if __name__ == "__main__":
    main()
