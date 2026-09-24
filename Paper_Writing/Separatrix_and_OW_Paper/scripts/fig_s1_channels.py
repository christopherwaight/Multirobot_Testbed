"""
fig_s1_channels.py

Figure: how the s1 tracker's two channels are built, in the plane.

(a) Tangent selection, (eq:tangent_select). The strain eigenframe drawn
    at points on both halves of the separatrix, with the gradient. The
    argmax picks e2 on the attracting half and e1 on the repelling half,
    swapping through the isotropic point at the origin. Verified against
    eigh(S): |grad.e2| carries the whole gradient norm above the origin
    and |grad.e1| carries it below, the other projection being 0.

(b) The velocity command. The along-trench term is open loop at cruise
    speed c_max tanh(1) along t; the transverse term is gradient descent
    through the projector P = I - t t^T, which deletes the along-trench
    component of grad s1. Travel does not depend on the trench-floor
    slope, which is why the cluster does not slide to a saddle.

Canonical output: figures/s1_channels.png
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

FIGURE_NAME = "s1_channels"

PARAMS = {
    "A":          0.1,
    "x_range":    [-0.30, 0.30],
    "y_range":    [-0.38, 0.38],
    "nx":         340,
    "ny":         340,
    "frame_ys":   [0.24, -0.24],   # eigenframe sample points, one per half
    "demo_y":     0.16,    # point used for the command decomposition
    "dpi":        220,
}


def _s1(x, y, A):
    xf = np.asarray(x) + 1.0
    yf = np.asarray(y) + 0.5
    return -np.pi**2 * A * np.abs(np.cos(np.pi * xf) * np.cos(np.pi * yf))


def _S(x, y, A):
    xf, yf = x + 1.0, y + 0.5
    ux = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    uy = np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    vx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    vy = np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    J = np.array([[ux, uy], [vx, vy]])
    return 0.5 * (J + J.T)


def _grad_s1(x, y, A, h=1e-6):
    return np.array([(_s1(x + h, y, A) - _s1(x - h, y, A)) / (2 * h),
                     (_s1(x, y + h, A) - _s1(x, y - h, A)) / (2 * h)])


def _tangent(x, y, A):
    """Eq. (eq:tangent_select) argmax, returned with which eigenvector won."""
    w, V = np.linalg.eigh(_S(x, y, A))       # w[0] <= w[1]  ->  e1, e2
    e1, e2 = V[:, 0], V[:, 1]
    g = _grad_s1(x, y, A)
    d1, d2 = abs(g @ e1), abs(g @ e2)
    return (e2, "e_2") if d2 > d1 else (e1, "e_1")


def main(args):
    p = PARAMS.copy()
    A = p["A"]

    xs = np.linspace(*p["x_range"], p["nx"])
    ys = np.linspace(*p["y_range"], p["ny"])
    X, Y = np.meshgrid(xs, ys)
    Z = _s1(X, Y, A)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))

    for ax in axes:
        ax.contourf(X, Y, Z, levels=28, cmap="viridis", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=2.0, zorder=4)
        ax.plot([0], [0], "o", color="crimson", ms=7, zorder=6)
        ax.set_xlim(*p["x_range"]); ax.set_ylim(*p["y_range"])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.tick_params(labelsize=8)

    # ---------------- (a) tangent selection ----------------
    ax = axes[0]
    ax.set_ylabel(r"$y$")
    L = 0.058
    xo = 0.050          # sample just off the trench so both axes are visible
    for yv in p["frame_ys"]:
        w, V = np.linalg.eigh(_S(xo, yv, A))
        e1, e2 = V[:, 0], V[:, 1]
        g = _grad_s1(xo, yv, A)
        win, name = _tangent(xo, yv, A)
        lose = e1 if name == "e_2" else e2
        # loser: thin grey; winner: heavy white. Both drawn as axes.
        for vec, col, lw, z in ((lose, "0.75", 1.3, 6), (win, "white", 2.8, 7)):
            ax.plot([xo - L * vec[0], xo + L * vec[0]],
                    [yv - L * vec[1], yv + L * vec[1]],
                    color=col, linewidth=lw, zorder=z, solid_capstyle="round")
        gn = g / np.linalg.norm(g)
        ax.annotate("", xy=(xo + 0.78 * L * gn[0], yv + 0.78 * L * gn[1]),
                    xytext=(xo, yv), zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color="orangered", lw=1.7))
        ax.text(xo + L + 0.022, yv, rf"$\mathbf{{{name}}}$", color="white",
                fontsize=9, va="center", zorder=9)
    ax.text(0.0, 0.335, "attracting half", fontsize=8, color="white",
            ha="center", zorder=9)
    ax.text(0.0, -0.355, "repelling half", fontsize=8, color="white",
            ha="center", zorder=9)
    ax.text(-0.285, 0.325, r"$\nabla \hat{s}_1$", color="orangered",
            fontsize=9, ha="left", va="center", zorder=9)
    ax.set_title(r"(a) tangent from $\arg\max|\nabla \hat{s}_1^\top \mathbf{e}|$",
                 fontsize=9)

    # ---------------- (b) command decomposition ----------------
    ax = axes[1]
    yv = p["demo_y"]
    t, _ = _tangent(0.0, yv, A)
    if t[1] < 0:
        t = -t                      # orient the ride up-trench for the sketch
    P = np.eye(2) - np.outer(t, t)
    # start off-trench so the transverse term is visible
    x0, y0 = 0.075, yv
    g = _grad_s1(x0, y0, A)
    perp = -P @ g
    perp = perp / np.linalg.norm(perp) * 0.115
    ride = t * 0.105

    ax.plot([x0], [y0], "o", color="white", ms=6, mec="black", zorder=8)
    ax.annotate("", xy=(x0 + ride[0], y0 + ride[1]), xytext=(x0, y0),
                zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4))
    ax.annotate("", xy=(x0 + perp[0], y0 + perp[1]), xytext=(x0, y0),
                zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="orangered", lw=2.4))
    ax.text(x0 + ride[0] + 0.022, y0 + ride[1],
            r"$\beta c_{\max}\tanh(1)\,\mathbf{t}$",
            color="white", fontsize=8, ha="center", va="bottom", zorder=9)
    ax.text(x0 + perp[0] - 0.02, y0 - 0.075,
            r"$-\mathrm{sat}(g_\perp \mathbf{P}\nabla \hat{s}_1)$",
            color="orangered", fontsize=8, ha="center", zorder=9)
    ax.set_title(r"(b) ride along $\mathbf{t}$, descend across it",
                 fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.14,
                        wspace=0.10)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"])
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")

    # Review numbers (nothing auto-injected into the .tex).
    for yv in p["frame_ys"]:
        _, name = _tangent(0.0, yv, A)
        print(f"  [extra] y={yv:+.2f} -> tangent {name}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params=p,
                  source_script=f"scripts/{FIGURE_NAME}.py",
                  extra={"tangent_by_y":
                         {str(yv): _tangent(0.0, yv, A)[1]
                          for yv in p["frame_ys"]}})

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
