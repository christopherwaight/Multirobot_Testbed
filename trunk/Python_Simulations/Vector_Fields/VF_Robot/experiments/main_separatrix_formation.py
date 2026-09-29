"""
main_separatrix_formation.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_11.tex
  Makes:  fig:formation, the six robots of the pentagon-plus-center cluster
          from start S1 under the D tracker and the s1 tracker, with robot
          paths and formation outlines at a few instants, zoomed to the
          separatrix corridor. Output experiments/outputs/oecs/
          separatrix_formation.png, installed by hand as
          figures/separatrix_formation.png.

The runs are those of main_separatrix_traverse.py (same start, gains, and
primitive settings, imported from it), so this figure shows the robots behind
the S1 centroid paths of fig:traverse_vs_logic_c.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/main_separatrix_formation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main_separatrix_traverse as M

START_NAME = "S1"
SNAPSHOT_FRACTIONS = [0.0, 0.3, 0.6, 1.0]  # of each run's ride to the far saddle
X_VIEW = (-0.32, 0.32)
Y_VIEW = (-0.62, 0.42)
ROBOT_COLOR = "#2a78d6"


def ride_end(center):
    d = np.linalg.norm(center - np.array(M.SADDLE_BOTTOM), axis=1)
    return int(np.argmin(d)) + 1


def outline(pts):
    c = pts.mean(axis=0)
    k_center = int(np.argmin(np.linalg.norm(pts - c, axis=1)))
    ring = np.delete(pts, k_center, axis=0)
    order = np.argsort(np.arctan2(ring[:, 1] - c[1], ring[:, 0] - c[0]))
    ring = ring[order]
    return np.vstack([ring, ring[:1]]), pts[k_center]


def main():
    sx, sy = next((x, y) for n, x, y in M.STARTS if n == START_NAME)
    runs = [("$D$ tracker", M.run_logic_c(sx, sy)),
            ("$s_1$ tracker", M.run_traverser(sx, sy))]

    gx = np.linspace(X_VIEW[0], X_VIEW[1], 80)
    gy = np.linspace(-0.5, 0.5, 120)
    GX, GY = np.meshgrid(gx, gy)
    U = np.empty_like(GX); V = np.empty_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            U[i, j], V[i, j] = M.double_gyre_static(GX[i, j], GY[i, j])

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.95), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.94, wspace=0.08)

    for ax, (title, cl) in zip(axes, runs):
        center = np.asarray(cl.get_center_history())
        robots = np.asarray(cl.get_robot_history())
        end = min(ride_end(center), len(robots))
        ax.streamplot(GX, GY, U, V, color="0.82", density=0.8, linewidth=0.4,
                      arrowsize=0.5)
        ax.axvline(0.0, color="0.35", ls="--", lw=0.6)
        ax.plot(*M.SADDLE_BOTTOM, marker="x", color="k", ms=5, mew=1.2, zorder=6)
        for i in range(robots.shape[1]):
            ax.plot(robots[:end, i, 0], robots[:end, i, 1], color=ROBOT_COLOR,
                    lw=0.5, alpha=0.55, zorder=3)
        ax.plot(center[:end, 0], center[:end, 1], color="black", lw=1.0, zorder=4)
        for f in SNAPSHOT_FRACTIONS:
            k = min(int(round(f * (end - 1))), end - 1)
            ring, mid = outline(robots[k])
            ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.7, zorder=5)
            ax.plot(robots[k, :, 0], robots[k, :, 1], "o", color=ROBOT_COLOR,
                    ms=2.6, mec="black", mew=0.4, zorder=6)
        ax.plot(sx, sy, marker="*", color="lime", ms=6, mec="black", mew=0.5,
                zorder=7)
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xlim(*X_VIEW); ax.set_ylim(*Y_VIEW)
        ax.set_aspect("equal")
        ax.set_xticks([-0.25, 0, 0.25])
        ax.set_xlabel("$x$", labelpad=1)
        ax.tick_params(length=2, pad=1)
    axes[0].set_ylabel("$y$", labelpad=1)

    out = os.path.join(M.OUT_DIR, "separatrix_formation.png")
    fig.savefig(out, dpi=400)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
