"""
thesis_ch8_one_flow.py

Thesis Chapter 8, "One Flow at Three Orders". One cell of the double gyre is
the gyre cell of Chapter 6, and the right wall of that cell is the separatrix
that the Chapter 7 trackers ride. This script checks that claim numerically
and draws the three rungs on the one flow:

  background   D = det J of the cell (Chapter 6 coordinates X, Y)
  dotted       the det J = 0 diamond |X| + |Y| = 1/2 that bounds the
               first-order basins (Chapter 6)
  dots         the 1000 starts of the Chapter 6 gyre trials, blue where the
               cluster reached the center and orange where it reached a saddle
               (the 2026-11-16 robot model; analysis/simulation/revision_metrics)
  lines        one noise-free ride of the D tracker and of the s1 tracker from
               start S1 of Chapter 7 (2026-11-16 robot model), shifted to cell
               coordinates by X = x + 0.5

Checks printed: (1) the cell field equals the left gyre of the double gyre of
Chapter 3 after the shift, on a grid; (2) the simulator's double_gyre_static
equals the closed form of Chapter 3; (3) det J = 0 on |X| + |Y| = 1/2.

Run from VF_Robot (the rides need D14=1, set here):
    venv/bin/python3 experiments/thesis_ch8_one_flow.py
Writes experiments/outputs/thesis_ch8/one_flow.png
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VF = HERE.parent
REPO = HERE.parents[4]
sys.path.insert(0, str(VF))
os.environ.setdefault("D14", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiments.unsteady_dg.fig_notes as N
from src.fields.environments.Double_Gyre import double_gyre_static

A = 0.1
STARTS = REPO / "analysis" / "simulation" / "revision_metrics" / "results" / "nonlinear_attraction_per_trial.csv"
OUT = HERE / "outputs" / "thesis_ch8"
BLUE, ORANGE, INK, GRAY, GREEN = "#2468c8", "#e8692c", "#1f1f1e", "#6b6a66", "#0b7a53"


def cell(X, Y):
    return 0.1 * np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y), -0.1 * np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)


def dg(x, y):
    xf, yf = x + 1.0, y + 0.5
    return -np.pi * A * np.sin(np.pi * xf) * np.cos(np.pi * yf), np.pi * A * np.cos(np.pi * xf) * np.sin(np.pi * yf)


def det_cell(X, Y):
    return 0.5 * np.pi ** 4 * A ** 2 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))


def checks():
    g = np.linspace(-0.5, 0.5, 41)
    X, Y = np.meshgrid(g, g)
    uc, vc = cell(X, Y)
    ud, vd = dg(X - 0.5, Y)   # cell (X, Y) is the double-gyre point (x, y) = (X - 0.5, Y)
    print(f"1. cell field vs left gyre after the shift: max difference {max(np.abs(uc - ud).max(), np.abs(vc - vd).max()):.2e}")
    sim = np.array([double_gyre_static(x, y) for x, y in zip((X - 0.5).ravel()[::17], Y.ravel()[::17])])
    ref = np.column_stack(dg((X - 0.5).ravel()[::17], Y.ravel()[::17]))
    print(f"2. simulator double_gyre_static vs closed form: max difference {np.abs(sim - ref).max():.2e}")
    t = np.linspace(-0.5, 0.5, 9)
    h = 1e-5
    def det_num(x, y):
        ux = (cell(x + h, y)[0] - cell(x - h, y)[0]) / (2 * h); uy = (cell(x, y + h)[0] - cell(x, y - h)[0]) / (2 * h)
        vx = (cell(x + h, y)[1] - cell(x - h, y)[1]) / (2 * h); vy = (cell(x, y + h)[1] - cell(x, y - h)[1]) / (2 * h)
        return ux * vy - uy * vx
    dev = max(abs(det_num(x, 0.5 - abs(x))) for x in t[:-1] if abs(x) < 0.5)
    print(f"3. det J on |X| + |Y| = 1/2 (numerical Jacobian): max |D| {dev:.2e}; closed form agrees: {np.abs(det_num(0.2, 0.3) - det_cell(0.2, 0.3)) < 1e-6}")


def main():
    checks()
    import csv
    rows = [r for r in csv.DictReader(open(STARTS)) if r["field"] == "gyre"]
    sx = np.array([float(r["start_x"]) for r in rows]); sy = np.array([float(r["start_y"]) for r in rows])
    center = np.array([r["reached_target"] == "True" for r in rows])
    print(f"gyre starts: {len(rows)}, reached the center {center.sum()}, a saddle {(~center).sum()}")

    rides = {}
    for tr in ("D", "s1"):
        path, outc = N.ride("bent", 0.0, tr, "S1")
        rides[tr] = np.array(path) + np.array([0.5, 0.0])
        print(f"ride {tr} from S1 -> {outc}, {len(path)} points")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    g = np.linspace(-0.5, 0.5, 201)
    X, Y = np.meshgrid(g, g)
    ax.contourf(X, Y, det_cell(X, Y), levels=np.linspace(-1.6, 1.6, 15), cmap="Greys", alpha=0.45, zorder=0)
    d = np.array([[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5], [0.5, 0]])
    ax.plot(d[:, 0], d[:, 1], color=INK, lw=1.1, ls=(0, (1, 2)), zorder=2)
    ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], color=GRAY, lw=0.6, zorder=1)
    ax.scatter(sx[center], sy[center], s=6, color=BLUE, alpha=0.8, lw=0, zorder=3)
    ax.scatter(sx[~center], sy[~center], s=6, color=ORANGE, alpha=0.8, lw=0, zorder=3)
    ax.plot(rides["D"][:, 0], rides["D"][:, 1], color=INK, lw=3.0, zorder=5)
    ax.plot(rides["s1"][:, 0], rides["s1"][:, 1], color=GREEN, lw=1.5, ls=(0, (4, 2)), zorder=6)
    ax.plot(*rides["D"][0], "o", ms=4.5, color="white", mec=INK, mew=1.0, zorder=6)
    ax.plot([0], [0], "o", ms=4, color=INK, zorder=6)
    ax.plot([0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], "x", ms=5, mew=1.2, color=INK, zorder=6, clip_on=False)
    ax.plot([0.5], [0], "D", ms=4.5, color="white", mec=INK, mew=1.0, zorder=7)
    ax.annotate("crest", (0.5, 0), xytext=(0.535, 0.045), fontsize=8, color=INK, ha="left")
    ax.text(0.0, 0.07, "center", ha="center", fontsize=8, color=INK)
    ax.text(0.515, -0.30, "separatrix", rotation=90, va="center", ha="left", fontsize=8, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-0.56, 0.60); ax.set_ylim(-0.56, 0.56)
    ax.set_xlabel("$X$ (m), cell coordinates"); ax.set_ylabel("$Y$ (m)")
    ax.tick_params(width=0.6, length=2.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=4, color=BLUE, label="Chapter 6 start, reached the center"),
         Line2D([], [], marker="o", ls="", ms=4, color=ORANGE, label="Chapter 6 start, reached a saddle"),
         Line2D([], [], color=INK, lw=1.1, ls=(0, (1, 2)), label=r"$\det\,\mathbf{J} = 0$"),
         Line2D([], [], color=INK, lw=3.0, label="$D$ tracker ride from S1"),
         Line2D([], [], color=GREEN, lw=1.5, ls=(0, (4, 2)), label="$s_1$ tracker ride from S1")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "one_flow.png", dpi=220)
    print("wrote", OUT / "one_flow.png")


if __name__ == "__main__":
    main()
