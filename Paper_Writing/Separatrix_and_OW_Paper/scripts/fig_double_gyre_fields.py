"""
fig_double_gyre_fields.py

Figure: three-panel stack over the steady double gyre, one column wide.
  (a) streamlines with separatrix, saddles, and gyre centers
  (b) signed D = det(J) as an elevation surface
  (c) s1 as an elevation surface
Panels (b) and (c) reuse the field functions and view of
fig_detJ_trench_cross_section.py and fig_s1_trench_surface.py, which it replaces
in the paper.

Canonical output: figures/double_gyre_fields.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _common import PAPER_DIR, write_sidecar, compile_paper, make_parser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from fig_detJ_trench_cross_section import _signed_det_j
from fig_s1_trench_surface import _s1

FIGURE_NAME = "double_gyre_fields"

PARAMS = {
    "A":                0.1,
    "x_range":          [-1.0, 1.0],
    "y_range":          [-0.5, 0.5],
    "stream_grid_n":    80,
    "stream_density":   1.1,
    "stream_linewidth": 0.6,
    "surface_n":        220,
    "elev":             28,
    "azim":             -128,
    "fig_width_in":     3.5,
    "fig_height_in":    6.6,
    "dpi":              300,
}


def _velocity(x, y, A):
    xf = x + 1.0
    yf = y + 0.5
    u = -np.pi * A * np.sin(np.pi * xf) * np.cos(np.pi * yf)
    v = np.pi * A * np.cos(np.pi * xf) * np.sin(np.pi * yf)
    return u, v


def _panel_label(fig, x, y, text):
    fig.text(x, y, text, fontsize=9, fontweight="bold", va="top", ha="left")


def _surface(fig, rect, Z, X, Y, cmap, norm, zlabel, p, zlim=None):
    ax = fig.add_axes(rect, projection="3d")
    ax.patch.set_alpha(0.0)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)), rcount=110, ccount=110,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel(r"$x$", labelpad=-4, fontsize=8)
    ax.set_ylabel(r"$y$", labelpad=-4, fontsize=8)
    ax.set_zlabel(zlabel, labelpad=1, fontsize=8)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(0)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-0.5, 0, 0.5])
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=p["elev"], azim=p["azim"])
    ax.set_box_aspect((2.0, 1.0, 1.1))
    return ax


def _colorbar(fig, rect, cmap, norm, label):
    cax = fig.add_axes(rect)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cb = fig.colorbar(m, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label, rotation=0, labelpad=6, fontsize=8)
    return cb


def main(args):
    p = PARAMS.copy()
    A = p["A"]
    x0, x1 = p["x_range"]
    y0, y1 = p["y_range"]

    fig = plt.figure(figsize=(p["fig_width_in"], p["fig_height_in"]))

    # (a) streamlines
    n = p["stream_grid_n"]
    xs = np.linspace(x0, x1, 2 * n)
    ys = np.linspace(y0, y1, n)
    Xs, Ys = np.meshgrid(xs, ys)
    U, V = _velocity(Xs, Ys, A)

    ax_a = fig.add_axes([0.12, 0.775, 0.84, 0.21])
    ax_a.streamplot(Xs, Ys, U, V, color=np.hypot(U, V), cmap="Blues",
                    density=p["stream_density"],
                    linewidth=p["stream_linewidth"], arrowsize=0.6)
    ax_a.plot([0, 0], [y0, y1], color="crimson", lw=1.2, ls="--", zorder=6)
    ax_a.plot([0, 0], [y0, y1], "x", color="crimson", ms=6, mew=1.5,
              zorder=7, clip_on=False)
    ax_a.plot([-0.5, 0.5], [0, 0], "o", color="#e07b00", ms=4.5,
              mec="black", mew=0.8, zorder=7)
    ax_a.set_xlim(x0, x1)
    ax_a.set_ylim(y0, y1)
    ax_a.set_aspect("equal")
    ax_a.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_a.set_yticks([-0.5, 0, 0.5])
    ax_a.tick_params(labelsize=6, pad=1.5)
    ax_a.set_xlabel(r"$x$", fontsize=8, labelpad=0)
    ax_a.set_ylabel(r"$y$", fontsize=8, labelpad=0)

    # Surfaces share one grid
    m = p["surface_n"]
    X, Y = np.meshgrid(np.linspace(x0, x1, m), np.linspace(y0, y1, m))

    # (b) D surface
    D = _signed_det_j(X, Y, A)
    vmax = np.abs(D).max()
    norm_d = plt.Normalize(-vmax, vmax)
    ax_b = _surface(fig, [0.00, 0.40, 0.84, 0.36], D, X, Y, cm.RdBu_r,
                    norm_d, r"$D$", p)
    yl = np.linspace(y0, y1, 200)
    ax_b.plot(np.zeros_like(yl), yl, _signed_det_j(0.0, yl, A),
              color="black", lw=1.6, zorder=10)
    _colorbar(fig, [0.83, 0.47, 0.022, 0.20], cm.RdBu_r, norm_d, r"$D$")

    # (c) s1 surface
    S1 = _s1(X, Y, A)
    norm_s = plt.Normalize(S1.min(), 0.0)
    ax_c = _surface(fig, [0.00, 0.04, 0.84, 0.36], S1, X, Y, cm.viridis,
                    norm_s, r"$s_1$", p, zlim=(S1.min() * 1.02, 0.06))
    y_up = np.linspace(0.0, y1, 120)
    y_dn = np.linspace(y0, 0.0, 120)
    ax_c.plot(np.zeros_like(y_up), y_up, _s1(0.0, y_up, A), color="black",
              lw=1.6, zorder=10, label=r"Attracting, $\mathbf{e}_2$ tangent")
    ax_c.plot(np.zeros_like(y_dn), y_dn, _s1(0.0, y_dn, A), color="black",
              lw=1.6, ls=(0, (4, 2)), zorder=10,
              label=r"Repelling, $\mathbf{e}_1$ tangent")
    ax_c.scatter([0.0], [0.0], [_s1(0.0, 0.0, A)], color="crimson", s=18,
                 depthshade=False, zorder=12, label="Isotropic point")
    _colorbar(fig, [0.83, 0.11, 0.022, 0.20], cm.viridis, norm_s, r"$s_1$")
    h, lab = ax_c.get_legend_handles_labels()
    fig.legend(h, lab, fontsize=6, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, handlelength=2.0,
               columnspacing=0.8, handletextpad=0.4)

    _panel_label(fig, 0.01, 0.995, "(a)")
    _panel_label(fig, 0.01, 0.735, "(b)")
    _panel_label(fig, 0.01, 0.385, "(c)")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"])
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params=p,
                  source_script=f"scripts/fig_{FIGURE_NAME}.py",
                  extra={
                      "D_min": float(D.min()),
                      "D_max": float(D.max()),
                      "s1_min": float(S1.min()),
                  })

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
