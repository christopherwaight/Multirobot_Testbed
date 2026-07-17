"""
plot_basin_map.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the basin-map figure (fig:basin_map),
          file figures/basin_map.png (review copy written to
          experiments/outputs/basin_map/; copied into the paper's
          figures/ folder at integration time).
  Reads:  experiments/outputs/basin_map/trials_grid_strict.csv
          (produced by basin_map.py; rerun that script first if the CSV
          is missing).

  Zero-noise, heading-0 outcome of Logic C from every cell of a
  200 x 100 start grid, four classes:
    traverse, direct          reached p1* without touching p2*
    traverse via p2*          rode the top wall to p2*, took the south
                              branch, reached p1*
    lost at p2* north branch  rode to p2*, took the north branch
    lost on boundary trench   rode a wall trench out of the domain
  Overlays: D = 0 diamonds, the central separatrix, both saddles, and
  the six clean-run starts.

  Palette validated with the dataviz skill (all-pairs CVD and
  normal-vision floors PASS; the green's surface-contrast WARN is
  relieved by the legend).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/plot_basin_map.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "basin_map")

NX, NY = 200, 100
X_MIN, X_MAX, Y_MIN, Y_MAX = -1.0, 1.0, -0.5, 0.5
A = 0.1
TOP_CONTACT = 0.06

CLASSES = ["traverse, direct", "traverse via $\\mathbf{p}^*_2$",
           "lost at $\\mathbf{p}^*_2$ north branch",
           "lost on boundary trench"]
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]

STARTS = [("S1", -0.45, 0.30), ("S2", 0.05, 0.40), ("S3", 0.00, 0.00),
          ("S4", 0.10, -0.20), ("S5", 0.25, 0.42), ("S6", -0.20, -0.30)]

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.0,
})


def classify(row):
    top = row["min_d_top"] < TOP_CONTACT
    if row["success_traverse"] == 1:
        return 1 if top else 0
    return 2 if top else 3


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "trials_grid_strict.csv"),
                     comment="#")
    xs = X_MIN + (np.arange(NX) + 0.5) * (X_MAX - X_MIN) / NX
    ys = Y_MIN + (np.arange(NY) + 0.5) * (Y_MAX - Y_MIN) / NY
    ix = np.rint((df["start_x"].values - X_MIN)
                 / ((X_MAX - X_MIN) / NX) - 0.5).astype(int)
    iy = np.rint((df["start_y"].values - Y_MIN)
                 / ((Y_MAX - Y_MIN) / NY) - 0.5).astype(int)
    cls = df.apply(classify, axis=1).values
    grid = np.full((NY, NX), -1, dtype=int)
    grid[iy, ix] = cls
    assert (grid >= 0).all()

    counts = {c: int((cls == k).sum()) for k, c in enumerate(CLASSES)}
    total = len(cls)
    print("class fractions (for the caption):")
    for c in CLASSES:
        print(f"  {c}: {counts[c]} ({100.0 * counts[c] / total:.1f}%)")

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.pcolormesh(xs, ys, grid, cmap=ListedColormap(COLORS),
                  vmin=-0.5, vmax=3.5, rasterized=True)

    GX, GY = np.meshgrid(np.linspace(X_MIN, X_MAX, 400),
                         np.linspace(Y_MIN, Y_MAX, 200))
    D = -(np.pi**4 * A**2 / 2) * (np.cos(2 * np.pi * (GX + 1))
                                  + np.cos(2 * np.pi * (GY + 0.5)))
    ax.contour(GX, GY, D, levels=[0.0], colors="k", linewidths=0.9,
               linestyles="solid")
    ax.axvline(0.0, color="k", ls="--", lw=1.3)
    for sad in ((0.0, -0.5), (0.0, 0.5)):
        ax.plot(*sad, marker="x", color="k", ms=8, mew=1.8, zorder=6,
                clip_on=False)
    for name, sx, sy in STARTS:
        ax.plot(sx, sy, marker="o", color="w", ms=4.5, mec="k", mew=0.9,
                zorder=7)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    handles = [Patch(facecolor=c, label=l) for c, l in zip(COLORS, CLASSES)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "basin_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
