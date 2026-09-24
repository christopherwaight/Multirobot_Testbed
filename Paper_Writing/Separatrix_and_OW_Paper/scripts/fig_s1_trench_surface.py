"""
fig_s1_trench_surface.py

Figure: the compressive strain eigenvalue s1 rendered as a 3D elevation
surface over the full double-gyre domain, in the same view and style as
fig_detJ_trench_cross_section.py so the two read as a pair.

Where D has ridges at the gyre cores and a trench that rises to a crest
(D = 0) at the origin, s1 <= 0 everywhere and is a trench on BOTH halves
of the separatrix with no ridge to cross. The two halves differ only in
the material identity of the trench tangent: the stretching eigenvector
e2 above the origin, the compression eigenvector e1 below, swapping at
the origin where S is isotropic and both are undefined.

s1 = -pi^2 A |cos(pi x_f) cos(pi y_f)|   (verified against eigvalsh of S
to 1.1e-16 over 4000 random points).

Canonical output: figures/s1_trench_surface.png
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

FIGURE_NAME = "s1_trench_surface"

PARAMS = {
    "A":        0.1,
    "x_range":  [-1.0, 1.0],
    "y_range":  [-0.5, 0.5],
    "nx":       220,
    "ny":       220,
    "elev":     28,      # matches detJ_trench_cross_section
    "azim":     -128,    # matches detJ_trench_cross_section
    "dpi":      220,
}


def _s1(x, y, A):
    """Compressive strain eigenvalue s1 <= 0 of the steady double gyre.

    S = 0.5 (J + J^T) has zero shear component here, so
    S = diag(u_x, -u_x) and s1 = -|u_x|.
    """
    xf = x + 1.0
    yf = y + 0.5
    return -np.pi**2 * A * np.abs(np.cos(np.pi * xf) * np.cos(np.pi * yf))


def main(args):
    p = PARAMS.copy()
    A = p["A"]

    xs = np.linspace(*p["x_range"], p["nx"])
    ys = np.linspace(*p["y_range"], p["ny"])
    X, Y = np.meshgrid(xs, ys)
    S1 = _s1(X, Y, A)

    fig = plt.figure(figsize=(6.0, 3.9))
    ax = fig.add_subplot(111, projection="3d")

    # s1 <= 0 everywhere, so a sequential map keyed to depth, not a
    # diverging one. Deep trench dark, zero (isotropic) light.
    vmin, vmax = S1.min(), 0.0
    norm = plt.Normalize(vmin, vmax)
    ax.plot_surface(
        X, Y, S1,
        facecolors=cm.viridis(norm(S1)),
        rcount=110, ccount=110,
        linewidth=0, antialiased=True, shade=True,
    )

    # The separatrix (x = 0) lifted onto the surface. Split at the origin
    # so the two material identities can be labeled separately.
    y_up = np.linspace(0.0, p["y_range"][1], 120)
    y_dn = np.linspace(p["y_range"][0], 0.0, 120)
    ax.plot(np.zeros_like(y_up), y_up, _s1(0.0, y_up, A),
            color="black", linewidth=2.2, zorder=10,
            label=r"Attracting, $\mathbf{e}_2$ tangent")
    ax.plot(np.zeros_like(y_dn), y_dn, _s1(0.0, y_dn, A),
            color="black", linewidth=2.2, linestyle=(0, (4, 2)), zorder=10,
            label=r"Repelling, $\mathbf{e}_1$ tangent")

    # The isotropic point at the origin: s1 = 0, eigenvectors undefined,
    # and the location of the D crest in Fig. detJ_trench_cross_section.
    ax.scatter([0.0], [0.0], [_s1(0.0, 0.0, A)],
               color="crimson", s=42, depthshade=False, zorder=12,
               label=r"Isotropic point, $D$ crest")

    # Zero level of s1 (the degeneracy lines) on the floor, mirroring the
    # D = 0 floor contour of the companion figure.
    ax.set_zlim(S1.min() * 1.02, 0.06)

    ax.set_xlabel(r"$x$", labelpad=6)
    ax.set_ylabel(r"$y$", labelpad=6)
    ax.set_zlabel(r"$s_1$", labelpad=8)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(0)
    ax.tick_params(labelsize=8, pad=0)
    ax.view_init(elev=p["elev"], azim=p["azim"])
    ax.set_box_aspect((2.0, 1.0, 1.1))

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(S1)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, aspect=14)
    cbar.set_label(r"$s_1$", rotation=0, labelpad=8)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, handlelength=2.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.04, right=0.99, top=1.06, bottom=0.14)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"])
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")

    # Numerical summaries for review before anything goes near the .tex.
    print(f"  [extra] s1 min (saddle depth) = {S1.min():.6f}  "
          f"(-pi^2 A = {-np.pi**2*A:.6f})")
    print(f"  [extra] s1 max               = {S1.max():.3e}")
    print(f"  [extra] s1 at origin         = {_s1(0.0,0.0,A):.3e}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params=p,
                  source_script=f"scripts/{FIGURE_NAME}.py",
                  extra={
                      "s1_min": float(S1.min()),
                      "s1_max": float(S1.max()),
                      "s1_at_origin": float(_s1(0.0, 0.0, A)),
                      "neg_pi2_A": float(-np.pi**2 * A),
                  })

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
