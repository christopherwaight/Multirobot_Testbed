"""
fig_detJ_contour.py

Figure: filled contour plot of det(J) over the double-gyre domain, with the
det(J)=0 boundary highlighted, saddle points and gyre centres marked.

Canonical output: figures/detJ_contour.png
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
from src.fields.environments.Double_Gyre import SADDLE_BOTTOM, SADDLE_TOP

FIGURE_NAME = "detJ_contour"

PARAMS = {
    "A":      0.1,
    "grid_n": 200,
    "domain": [-1.0, 1.0, -0.55, 0.55],
    "cmap":   "RdBu",
    "n_fill_levels": 40,
    "dpi": 220,
}


def _det_j_grid(X, Y, A):
    xf = X + 1.0
    yf = Y + 0.5
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
    D = _det_j_grid(X, Y, A)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    vmax = np.percentile(np.abs(D), 98)
    cf = ax.contourf(X, Y, D, levels=p["n_fill_levels"],
                     cmap=p["cmap"], vmin=-vmax, vmax=vmax, alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label(r"$\det(\mathbf{J})$", fontsize=9)

    # det(J)=0 boundary
    ax.contour(X, Y, D, levels=[0.0], colors=["black"], linewidths=2.0, zorder=5)

    # Saddles and gyre centres
    ax.plot(*SADDLE_BOTTOM, "x", color="black", markersize=10,
            markeredgewidth=2, zorder=7, label="Saddle")
    ax.plot(*SADDLE_TOP, "x", color="black", markersize=10,
            markeredgewidth=2, zorder=7)
    ax.plot(-0.5, 0.0, "o", color="white", markersize=7,
            markeredgewidth=1.5, markeredgecolor="black", zorder=7,
            label="Gyre centre")
    ax.plot(0.5, 0.0, "o", color="white", markersize=7,
            markeredgewidth=1.5, markeredgecolor="black", zorder=7)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\det(\mathbf{J})$ over the double-gyre domain", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    out = args.out
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
