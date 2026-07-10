"""
fig_detJ_trench_cross_section.py

Figure: signed D = det(J) rendered as a 3D elevation surface over the full
double-gyre domain. The gyre cores rise as ridges (D > 0), the exterior sits
below zero (D < 0, strain-dominated), and the separatrix at x = 0 appears as a
valley (trench) connecting the crest at the origin to the two wells at the
saddle points. This replaces the earlier 2D line-cross-section plot.

Canonical output: figures/detJ_trench_cross_section.png
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
from matplotlib import cm

FIGURE_NAME = "detJ_trench_cross_section"

PARAMS = {
    "A":        0.1,
    "x_range":  [-1.0, 1.0],
    "y_range":  [-0.5, 0.5],
    "nx":       220,
    "ny":       220,
    "elev":     28,      # 3D view elevation angle (deg)
    "azim":     -128,    # 3D view azimuth angle (deg)
    "dpi":      220,
}


def _signed_det_j(x, y, A):
    """Signed D = det(J) of the steady double-gyre field (no abs)."""
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

    xs = np.linspace(*p["x_range"], p["nx"])
    ys = np.linspace(*p["y_range"], p["ny"])
    X, Y = np.meshgrid(xs, ys)
    D = _signed_det_j(X, Y, A)

    fig = plt.figure(figsize=(6.0, 4.4))
    ax = fig.add_subplot(111, projection="3d")

    # Symmetric color scale about D = 0 so cores (D>0) and exterior (D<0)
    # read on opposite sides of a diverging map.
    vmax = np.abs(D).max()
    norm = plt.Normalize(-vmax, vmax)
    surf = ax.plot_surface(
        X, Y, D,
        facecolors=cm.RdBu_r(norm(D)),
        rcount=110, ccount=110,
        linewidth=0, antialiased=True, shade=True,
    )

    # Trace the separatrix valley (x = 0) as a heavy line lifted onto the surface.
    y_line = np.linspace(*p["y_range"], 200)
    d_line = _signed_det_j(0.0, y_line, A)
    ax.plot(np.zeros_like(y_line), y_line, d_line,
            color="black", linewidth=2.2, zorder=10,
            label="Separatrix ($x=0$)")

    # Flat reference contour at D = 0 (boundary between rotation and strain),
    # projected onto the floor of the box.
    ax.contour(X, Y, D, levels=[0.0], colors="dimgray",
               linewidths=1.8, linestyles="dashed",
               offset=D.min(), zdir="z")

    ax.set_xlabel(r"$x$", labelpad=6)
    ax.set_ylabel(r"$y$", labelpad=6)
    ax.set_zlabel(r"$D=\det(\mathbf{J})$", labelpad=10)
    ax.set_title(r"Separatrix as a trench of signed $D=\det(\mathbf{J})$",
                 fontsize=10, pad=0)
    ax.view_init(elev=p["elev"], azim=p["azim"])
    ax.set_box_aspect((2.0, 1.0, 1.1))

    # Colorbar keyed to the same diverging norm.
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)
    mappable.set_array(D)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, aspect=14)
    cbar.set_label(r"$D$", rotation=0, labelpad=8)

    ax.legend(fontsize=9, loc="upper left")
    # Reserve room on the left (z-axis label) and top (title, legend).
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.02)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"])
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
