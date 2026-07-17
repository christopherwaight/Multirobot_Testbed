"""
separatrix_clean_runs.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  Fig. \\ref{fig:sep_trajectories}
          (figures/separatrix_trajectories.png); review copy and per-run
          CSV in experiments/outputs/separatrix_clean/.
  Also answers the continuation-mechanism question (plan.md Phase 2.3):
  which selector branch fires near the saddle wells.

EXPERIMENT
  Six noise-free Logic C runs on the steady double gyre from starts that
  cover: upstream turnaround, straddling near the top saddle, the crest,
  a straddling mid start, a far right-gyre start, and a lower-left start.
  Single-panel figure: streamlines, D=0 diamonds, separatrix, saddles,
  six centroid paths. Per-run CSV: time-to-band, final position, min
  distance to each saddle, selector-mode occupancy, continuation flag.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/separatrix_clean_runs.py
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
from src.control.pentagon_primitives import separatrix_logic_c_step

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
SIM_STEPS = 600
A = 0.1
BAND_X = 0.05          # |x_c| threshold for time-to-band
BAND_HOLD = 10         # consecutive steps required

STARTS = [
    ("S1", -0.45,  0.30),
    ("S2",  0.05,  0.40),
    ("S3",  0.00,  0.00),
    ("S4",  0.10, -0.20),
    ("S5",  0.25,  0.42),
    ("S6", -0.20, -0.30),
]
# Stopping rules (2026-07-02 figure iteration): runs stop at domain exit
# (|x|>1 or |y|>0.52; beyond that the analytic tiling is not physical) or
# POST_SADDLE_STEPS after first saddle contact, whichever comes first, so
# the continuation along the wall trench is visible but bounded.
# Contact is tested against the BOTTOM saddle only: the ambient flow runs
# top saddle -> bottom saddle, so every trench traverse ends there; keying
# on the top saddle truncated runs that entered the band near the top.
POST_SADDLE_STEPS = 150
SADDLE_CONTACT_D = 0.06
# Validated categorical palette (dataviz skill) + 3 more distinct steps
COLORS = ["#2a78d6", "#1baf7a", "#4a3aa7", "#e34948", "#eb6834", "#e87ba4"]

OUT_DIR = os.path.join(project_root, "experiments", "outputs",
                       "separatrix_clean")
os.makedirs(OUT_DIR, exist_ok=True)


def run_one(sx, sy):
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


def analyze(cl):
    hist = cl.get_center_history()
    xs, ys = hist[:, 0], hist[:, 1]
    # time-to-band
    inband = np.abs(xs) < BAND_X
    t_band = -1
    for k in range(len(inband) - BAND_HOLD):
        if inband[k:k + BAND_HOLD].all():
            t_band = k
            break
    d_bot = np.linalg.norm(hist - np.array(SADDLE_BOTTOM), axis=1)
    d_top = np.linalg.norm(hist - np.array(SADDLE_TOP), axis=1)
    # continuation: came within 0.06 of a saddle, then later moved > 0.15
    # away from it along the wall (|x| grows while |y| stays near the wall)
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


def main():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"

    fig = plt.figure(figsize=(7.0, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    # streamlines + D=0 contour
    gx = np.linspace(-1, 1, 240)
    gy = np.linspace(-0.5, 0.5, 120)
    GX, GY = np.meshgrid(gx, gy)
    U = np.empty_like(GX); Vv = np.empty_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            U[i, j], Vv[i, j] = double_gyre_static(GX[i, j], GY[i, j])
    Dfield = -(np.pi**4 * A**2 / 2) * (np.cos(2*np.pi*(GX+1))
                                       + np.cos(2*np.pi*(GY+0.5)))
    ax.streamplot(GX, GY, U, Vv, color="0.82", density=1.1, linewidth=0.6,
                  arrowsize=0.7)
    ax.contour(GX, GY, Dfield, levels=[0.0], colors="0.45",
               linewidths=1.0)
    ax.axvline(SEPARATRIX_X, color="0.35", ls="--", lw=1.0)
    for sad in (SADDLE_BOTTOM, SADDLE_TOP):
        ax.plot(*sad, marker="x", color="k", ms=9, mew=2, zorder=6)

    rows = []
    for (name, sx, sy), col in zip(STARTS, COLORS):
        cl = run_one(sx, sy)
        r = analyze(cl)
        h = r["hist"]
        ax.plot(h[:, 0], h[:, 1], color=col, lw=1.7, zorder=5)
        ax.plot(sx, sy, marker="o", color=col, ms=6, mec="k", mew=0.8,
                zorder=7)
        ax.plot(h[-1, 0], h[-1, 1], marker="s", color=col, ms=6, mec="k",
                mew=0.8, zorder=7)
        dx, dy = (8, -11) if name == "S1" else (6, 5)
        ax.annotate(name, (sx, sy), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=col)
        rows.append((name, sx, sy, r))
        m = r["modes"]
        tot = sum(m.values())
        print(f"{name} ({sx:+.2f},{sy:+.2f}): t_band={r['t_band']:>3} "
              f"final=({r['final_x']:+.3f},{r['final_y']:+.3f}) "
              f"min_d(bot,top)=({r['min_d_bot']:.3f},{r['min_d_top']:.3f}) "
              f"continued={r['continued']} modes="
              + ",".join(f"{k}:{v/tot:.2f}" for k, v in m.most_common()))

    # pentagon footprint at S1 start for scale
    fld = AnalyticalField(double_gyre_static)
    cl0 = PentagonCluster(FORMATION_CONFIG, fld)
    cl0.reset(*STARTS[0][1:])
    pts = np.array(cl0.get_robot_positions()).reshape(6, 2)
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="k", zorder=8)

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-1, -0.5), 2, 1, fill=False, ec="0.25",
                           lw=0.8, zorder=3))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.55, 0.53)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # Mode-occupancy strip (V-3): one thin bar per run, colored by the
    # active selector-mode family versus control step. Families follow
    # the selector of Eq. (27): FLOW includes FLOW_DRIFT, ATTRACT
    # includes ATTRACT_FALLBACK; exact per-mode fractions stay in the
    # CSV. Palette: validated categorical steps (dataviz skill).
    FAMILY = {"FLOW": 0, "FLOW_DRIFT": 0, "SLIDE": 1,
              "ATTRACT": 2, "ATTRACT_FALLBACK": 2}
    FAM_COLORS = ["#2a78d6", "#1baf7a", "#eb6834"]
    FAM_LABELS = ["FLOW", "SLIDE", "ATTRACT"]
    from matplotlib.patches import Patch
    for i, (name, sx, sy, r) in enumerate(rows):
        fams = [FAMILY[d["mode"]] for d in r["diag"]]
        k0 = 0
        for k in range(1, len(fams) + 1):
            if k == len(fams) or fams[k] != fams[k0]:
                ax_m.broken_barh([(k0, k - k0)], (i - 0.38, 0.76),
                                 facecolors=FAM_COLORS[fams[k0]])
                k0 = k
    ax_m.set_yticks(range(len(rows)))
    ax_m.set_yticklabels([row[0] for row in rows], fontsize=8)
    ax_m.set_ylim(len(rows) - 0.5, -0.5)
    ax_m.set_xlabel("control step", fontsize=9)
    ax_m.tick_params(labelsize=8)
    ax_m.legend(handles=[Patch(facecolor=c, label=l)
                         for c, l in zip(FAM_COLORS, FAM_LABELS)],
                loc="lower right", frameon=False, fontsize=7.5, ncol=3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "separatrix_trajectories.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # CSV
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    csv = os.path.join(OUT_DIR, "clean_runs.csv")
    with open(csv, "w") as f:
        f.write(f"# generated_by: experiments/separatrix_clean_runs.py\n"
                f"# git_commit: {commit}\n# date: {stamp}\n"
                f"# steps: {SIM_STEPS}  v_max: {V_MAX}  gain: {GAIN}  "
                f"eps_raw: {EPS_RAW}  eps_dim: {EPS_DIM}\n")
        f.write("run,start_x,start_y,t_band,final_x,final_y,"
                "min_d_bot,min_d_top,continued,"
                "frac_FLOW,frac_FLOW_DRIFT,frac_SLIDE,frac_ATTRACT,"
                "frac_ATTRACT_FALLBACK\n")
        for name, sx, sy, r in rows:
            m, tot = r["modes"], sum(r["modes"].values())
            fr = {k: m.get(k, 0)/tot for k in
                  ("FLOW", "FLOW_DRIFT", "SLIDE", "ATTRACT",
                   "ATTRACT_FALLBACK")}
            f.write(f"{name},{sx},{sy},{r['t_band']},"
                    f"{r['final_x']:.4f},{r['final_y']:.4f},"
                    f"{r['min_d_bot']:.4f},{r['min_d_top']:.4f},"
                    f"{int(r['continued'])},"
                    + ",".join(f"{fr[k]:.3f}" for k in
                               ("FLOW", "FLOW_DRIFT", "SLIDE", "ATTRACT",
                                "ATTRACT_FALLBACK")) + "\n")
    print(f"Saved: {csv}")

    # Mechanism probe: what fires within 0.1 of a saddle?
    print("\nSelector modes within 0.10 of a saddle (per run):")
    for name, sx, sy, r in rows:
        h = r["hist"]
        near = (np.linalg.norm(h - np.array(SADDLE_BOTTOM), axis=1) < 0.10) \
             | (np.linalg.norm(h - np.array(SADDLE_TOP), axis=1) < 0.10)
        idx = np.where(near)[0]
        if len(idx) == 0:
            print(f"  {name}: never near a saddle")
            continue
        sub = Counter(r["diag"][k]['mode'] for k in idx if k < len(r["diag"]))
        lam_near = [(r["diag"][k]['lam1'], r["diag"][k]['lam2'])
                    for k in idx[:3] if k < len(r["diag"])]
        print(f"  {name}: {dict(sub)}  first lam pairs {lam_near}")


if __name__ == "__main__":
    main()
