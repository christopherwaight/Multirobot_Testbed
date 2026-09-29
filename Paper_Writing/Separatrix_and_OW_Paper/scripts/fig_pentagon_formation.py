"""
fig_pentagon_formation.py

Figure: the six-robot pentagon-plus-center formation with its cluster space
variables labeled. Robot positions come from the repository's inverse
kinematics (src/control/pentagon_kinematics.py) applied to the formation file
the double-gyre runs load, so every length and angle drawn is the one the
simulations use.

Variables (pentagon_kinematics.py): pairs A = (R1, R2), B = (R3, R4),
C = (R5, R6), each with separation L_i and orientation theta_i; pair
midpoints form the SAS triangle p = |AB|, beta = angle at B, q = |BC|;
the centroid is the mean of the midpoints. Heading theta_c is drawn as the
direction from midpoint B to midpoint A, the convention inverse_kinematics
builds the formation with and the config value matches. forward_kinematics
instead returns atan2(B - A), which differs by pi (flagged 2026-09-29).

Canonical output: figures/pentagon_formation.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _common import PAPER_DIR, VFR_ROOT, write_sidecar, compile_paper, make_parser

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

sys.path.insert(0, str(VFR_ROOT))
from src.control.pentagon_kinematics import inverse_kinematics

FIGURE_NAME = "pentagon_formation"

PARAMS = {
    "formation_config": "config/formations/pentagon_small.yaml",
    "centroid": [0.0, 0.0],
    "fig_width_in": 2.2,
    "dpi": 400,
}

PAIR_COLORS = ["#2a78d6", "#e07b00", "#1baf7a"]


def _load_formation(rel):
    with open(os.path.join(str(VFR_ROOT), rel)) as f:
        return yaml.safe_load(f)["formation"]


def _deg(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def _arc(ax, center, r, a0, a1, **kw):
    lo, hi = (a0, a1) if (a1 - a0) % 360 <= 180 else (a1, a0)
    ax.add_patch(Arc(center, 2 * r, 2 * r, theta1=lo, theta2=hi, lw=0.6, **kw))
    return np.radians((lo + ((hi - lo) % 360) / 2))


def main(args):
    p = PARAMS.copy()
    f = _load_formation(p["formation_config"])
    xc, yc = p["centroid"]
    c = inverse_kinematics(xc, yc, f["theta_c"], f["p_1"], f["beta_1"], f["q_1"],
                           f["L_2"], f["theta_2"], f["L_3"], f["theta_3"],
                           f["L_4"], f["theta_4"])
    R = np.array(c).reshape(6, 2)
    pairs = [(0, 1), (2, 3), (4, 5)]
    M = np.array([(R[a] + R[b]) / 2 for a, b in pairs])   # midpoints A, B, C
    A, B, C = M
    cen = M.mean(axis=0)

    plt.rcParams.update({"font.size": 7, "mathtext.fontset": "cm"})
    w = p["fig_width_in"]
    fig, ax = plt.subplots(figsize=(w, w * 1.02))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    # Pentagon ring (R2..R6) as a faint outline, to read as pentagon-plus-center.
    ring = R[1:]
    order = np.argsort(np.arctan2(ring[:, 1] - cen[1], ring[:, 0] - cen[0]))
    ring = np.vstack([ring[order], ring[order][:1]])
    ax.plot(ring[:, 0], ring[:, 1], color="0.75", lw=0.5, ls=":", zorder=1)

    # Pairs, with separation labels.
    for k, ((a, b), col, lab) in enumerate(zip(pairs, PAIR_COLORS, ["2", "3", "4"])):
        ax.plot(*zip(R[a], R[b]), color=col, lw=1.2, zorder=2)
        mid = (R[a] + R[b]) / 2
        d = R[b] - R[a]; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        off = 0.013 * (1 if k != 0 else -1)
        ax.text(*(mid + off * nrm + 0.22 * d), f"$L_{lab}$", color=col,
                ha="center", va="center", fontsize=7)

    # SAS triangle of midpoints.
    tri = np.vstack([A, B, C, A])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.8, zorder=3)
    for P0, P1, lab in ((A, B, "p"), (B, C, "q")):
        mid = (P0 + P1) / 2
        out_dir = (mid - cen) / np.linalg.norm(mid - cen)
        ax.text(*(mid + 0.011 * out_dir), f"${lab}$", ha="center", va="center",
                fontsize=8)
    th = _arc(ax, B, 0.017, _deg(A - B), _deg(C - B), color="black")
    ax.text(*(B + 0.029 * np.array([np.cos(th), np.sin(th)])), r"$\beta$",
            ha="center", va="center", fontsize=8)

    # Heading theta_c: direction B->A, drawn at A (continuing past it) so its
    # arc does not coincide with beta at B when side BC is horizontal.
    u = (A - B) / np.linalg.norm(A - B)
    ax.plot([A[0], A[0] + 0.032], [A[1], A[1]], color="0.4", lw=0.5, ls="--")
    ax.add_patch(FancyArrowPatch(A, A + 0.034 * u, arrowstyle="-|>",
                                 mutation_scale=6, color="crimson", lw=0.8,
                                 zorder=4))
    th = _arc(ax, A, 0.022, _deg(u), 0.0, color="crimson")
    ax.text(*(A + 0.032 * np.array([np.cos(th), np.sin(th)])), r"$\theta_c$",
            color="crimson", ha="center", va="center", fontsize=8)

    # One pair orientation shown (pair B), the others are defined alike.
    a, b = pairs[1]
    ax.plot([R[a][0], R[a][0] + 0.03], [R[a][1], R[a][1]], color="0.4", lw=0.5,
            ls="--")
    th = _arc(ax, R[a], 0.02, 0.0, _deg(R[b] - R[a]), color=PAIR_COLORS[1])
    ax.text(*(R[a] + 0.031 * np.array([np.cos(th), np.sin(th)])), r"$\theta_3$",
            color=PAIR_COLORS[1], ha="center", va="center", fontsize=7)

    # Robots, midpoints, centroid.
    for i, (x, y) in enumerate(R):
        ax.plot(x, y, "o", ms=7, color="white", mec="black", mew=0.7, zorder=6)
        ax.text(x, y, f"{i + 1}", ha="center", va="center", fontsize=5.5, zorder=7)
    for P0, lab in zip(M, "ABC"):
        ax.plot(*P0, "s", ms=2.8, color="black", zorder=5)
    ax.plot(*cen, marker="+", ms=7, mew=1.0, color="crimson", zorder=6)
    ax.text(cen[0] + 0.009, cen[1] - 0.004, r"$\mathbf{p}_c$", color="crimson",
            fontsize=7, ha="left", va="top")

    # World axes.
    o = np.array([R[:, 0].min() - 0.012, R[:, 1].min() - 0.008])
    for v, lab in (((0.022, 0), "$x$"), ((0, 0.022), "$y$")):
        ax.add_patch(FancyArrowPatch(o, o + v, arrowstyle="-|>", mutation_scale=5,
                                     color="0.3", lw=0.5))
        ax.text(*(o + 1.35 * np.array(v)), lab, fontsize=6, color="0.3",
                ha="center", va="center")

    pad = 0.02
    ax.set_xlim(R[:, 0].min() - pad, R[:, 0].max() + pad)
    ax.set_ylim(R[:, 1].min() - pad, R[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"])
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(out, figure_name=FIGURE_NAME, params={**p, **f},
                  source_script=f"scripts/{FIGURE_NAME}.py",
                  extra={"robots": R.tolist(), "midpoints": M.tolist()})

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
