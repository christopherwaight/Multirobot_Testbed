"""
fig_double_gyre_streamlines.py

Figure: double-gyre streamlines with separatrix, saddle points, gyre centers,
and det(J)=0 contour overlaid.

Canonical output: figures/double_gyre_streamlines.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    PAPER_DIR, FIGURES_DIR, VFR_ROOT,
    write_sidecar, compile_paper, make_parser,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(VFR_ROOT))
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X,
)

FIGURE_NAME = "double_gyre_streamlines"

PARAMS = {
    "A":       0.1,
    "grid_n":  60,
    "domain":  [-1.0, 1.0, -0.55, 0.55],
    "stream_density": 1.2,
    "stream_linewidth": 0.8,
    "detJ_contour_levels": [0.0],
    "dpi": 220,
}


def _det_j(x, y, A=0.1):
    xf = x + 1.0
    yf = y + 0.5
    dudx = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    dudy =  np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdy =  np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    return dudx * dvdy - dudy * dvdx


def main(args):
    p = PARAMS.copy()
    A = p["A"]
    x0, x1, y0, y1 = p["domain"]
    n = p["grid_n"]

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    D = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            u, v = double_gyre_static(X[i, j], Y[i, j], A=A)
            U[i, j] = u
            V[i, j] = v
            D[i, j] = _det_j(X[i, j], Y[i, j], A=A)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(
        X, Y, U, V,
        color=speed,
        cmap="Blues",
        density=p["stream_density"],
        linewidth=p["stream_linewidth"],
        arrowsize=0.9,
    )

    # det(J)=0 contour
    cs = ax.contour(X, Y, D, levels=p["detJ_contour_levels"],
                    colors=["#e07b00"], linewidths=1.8, zorder=5)
    ax.clabel(cs, fmt={0.0: r"$\det J=0$"}, fontsize=8, inline=True)

    # Separatrix
    ax.axvline(x=SEPARATRIX_X, color="crimson", linewidth=1.5,
               linestyle="--", zorder=6, label="Separatrix")

    # Saddle points
    ax.plot(*SADDLE_BOTTOM, "x", color="crimson", markersize=10,
            markeredgewidth=2, zorder=7, label="Saddle")
    ax.plot(*SADDLE_TOP, "x", color="crimson", markersize=10,
            markeredgewidth=2, zorder=7)

    # Gyre centers
    ax.plot(-0.5, 0.0, "o", color="#e07b00", markersize=7,
            markeredgewidth=1.5, markeredgecolor="black", zorder=7,
            label=r"Gyre centre")
    ax.plot(0.5, 0.0, "o", color="#e07b00", markersize=7,
            markeredgewidth=1.5, markeredgecolor="black", zorder=7)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Double-gyre streamlines", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    out: "Path" = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"], bbox_inches="tight")
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params=p,
                  source_script=f"scripts/{FIGURE_NAME}.py")

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
