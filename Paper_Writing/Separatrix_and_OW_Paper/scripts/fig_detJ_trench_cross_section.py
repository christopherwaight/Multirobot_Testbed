"""
fig_detJ_trench_cross_section.py

Figure: |det(J)| along horizontal cross-sections (fixed y, sweep x) at
y = 0, 0.2, 0.4. Shows the trench minimum at x=0 (the separatrix).

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

FIGURE_NAME = "detJ_trench_cross_section"

PARAMS = {
    "A":        0.1,
    "x_range":  [-1.0, 1.0],
    "nx":       400,
    "y_values": [0.0, 0.2, 0.4],
    "dpi":      220,
}


def _abs_det_j(x, y, A):
    xf = x + 1.0
    yf = y + 0.5
    dudx = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    dudy =  np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdy =  np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    return abs(dudx * dvdy - dudy * dvdx)


def main(args):
    p = PARAMS.copy()
    A = p["A"]
    xs = np.linspace(*p["x_range"], p["nx"])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    styles = ["-", "--", ":"]

    for y_val, col, sty in zip(p["y_values"], colors, styles):
        vals = np.array([_abs_det_j(x, y_val, A) for x in xs])
        ax.plot(xs, vals, color=col, linestyle=sty, linewidth=1.6,
                label=rf"$y = {y_val}$")

    ax.axvline(x=0.0, color="crimson", linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=3, label="Separatrix ($x=0$)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\det(\mathbf{J})|$")
    ax.set_title("Eulerian trench in $|\\det(\\mathbf{J})|$ along the separatrix",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(*p["x_range"])
    ax.set_ylim(bottom=0)
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
