"""
fig_pentagon_formation.py

Figure: annotated diagram of the 6-robot pentagon-of-pairs formation.
Shows robots, pair midpoints, SAS triangle, centroid, and key dimension labels.

Canonical output: figures/pentagon_formation.png
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
import matplotlib.patches as mpatches

sys.path.insert(0, str(VFR_ROOT))
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_kinematics import inverse_kinematics

FIGURE_NAME = "pentagon_formation"

PARAMS = {
    "formation_config": "config/formations/pentagon_small.yaml",
    "centroid": [0.0, 0.0],
    "dpi": 220,
}

# Formation parameters from pentagon_small.yaml (loaded for documentation).
_FORMATION = {
    "L_2": 0.150000,  "theta_2": -1.570796,
    "L_3": 0.176336,  "theta_3": -1.884956,
    "L_4": 0.176336,  "theta_4":  1.884956,
    "p_1": 0.161172,  "beta_1":   0.772617,
    "q_1": 0.230827,  "theta_c": -0.772617,
}


def _midpoint(r1, r2):
    return ((r1[0]+r2[0])/2, (r1[1]+r2[1])/2)


def main(args):
    p = PARAMS.copy()
    xc, yc = p["centroid"]

    # Use inverse kinematics to get the 6 robot positions at the centroid.
    f = _FORMATION
    coords = inverse_kinematics(
        xc, yc, f["theta_c"],
        f["p_1"], f["beta_1"], f["q_1"],
        f["L_2"], f["theta_2"],
        f["L_3"], f["theta_3"],
        f["L_4"], f["theta_4"],
    )
    robots = [(coords[2*i], coords[2*i+1]) for i in range(6)]

    # Pairs: (0,1), (2,3), (4,5)
    pairs = [(0, 1), (2, 3), (4, 5)]
    mids  = [_midpoint(robots[a], robots[b]) for a, b in pairs]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Draw pair lines (dashed)
    colors_pair = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (a, b), col in zip(pairs, colors_pair):
        ra, rb = robots[a], robots[b]
        ax.plot([ra[0], rb[0]], [ra[1], rb[1]], "-", color=col,
                linewidth=1.8, alpha=0.55, zorder=2)

    # Draw SAS triangle
    m0, m1, m2 = mids
    triangle_xs = [m0[0], m1[0], m2[0], m0[0]]
    triangle_ys = [m0[1], m1[1], m2[1], m0[1]]
    ax.plot(triangle_xs, triangle_ys, "k-", linewidth=1.2, zorder=3,
            label="Midpoint triangle")

    # Plot robots
    robot_colors = ["#1f77b4","#1f77b4", "#ff7f0e","#ff7f0e", "#2ca02c","#2ca02c"]
    for i, (rx, ry) in enumerate(robots):
        ax.scatter(rx, ry, s=130, color=robot_colors[i], zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(f"$R_{i+1}$", (rx, ry), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)

    # Plot midpoints
    for k, (mx, my) in enumerate(mids):
        ax.scatter(mx, my, s=80, marker="^", color="black", zorder=6)
        ax.annotate(f"$m_{k+1}$", (mx, my), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)

    # Plot centroid
    ax.scatter(xc, yc, s=100, marker="D", color="crimson", zorder=7,
               label=r"Centroid $\mathbf{p}_c$")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Pentagon-of-pairs formation (6 robots)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"], bbox_inches="tight")
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params={**p, **_FORMATION},
                  source_script=f"scripts/{FIGURE_NAME}.py")

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
