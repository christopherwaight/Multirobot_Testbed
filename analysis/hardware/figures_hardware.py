"""Hardware attraction figure for the paper (fig:vortextomiddle), one column wide, in the paper style.

(a) vortex 3R and (b) saddle 3R: every run's centroid path from the eight start groups, measured from
the session's fitted p*, with each group's mean path (sample-by-sample mean of its runs). Motive's
ground-plane axes (x, z) are drawn as the paper's (x, y).
(c) distance to p* against time, mean +- 1 sd across runs: vortex 3R, saddle 3R, vortex 4R.

Run from the repo root after score_hardware.py. Writes analysis/hardware/figures/fig_hw_attraction.{pdf,png}
(and the vortex 4R paths, not in the paper, as fig_hw_trajectories_vortex4R).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import paths  # noqa: E402
import hw2025  # noqa: E402
import figstyle as fs  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from centres import decode, field_zero  # noqa: E402

FIG = os.path.join(paths.OUT_HW, "figures")
COLORS = plt.cm.tab10(np.arange(8))
SETS = (("vortex", 3, "Vortex, 3 robots", fs.BLUE), ("saddle", 3, "Saddle, 3 robots", fs.RED),
        ("vortex", 4, "Vortex, 4 robots", fs.GREEN))


def centroid_paths(field, nr):
    runs, ps, _ = hw2025.load(field, nr)
    return {n: (r["t"], r["pos"].mean(1) - ps, r["source"]) for n, r in sorted(runs.items())}


VC, K = 0.025, 1.0  # hardware dead zone (m/s) and attraction gain (1/s); friction ring v_c/k
BG_ALPHA = 0.30     # opacity of the background field arrows (keeps the paths dominant)
BG_STEP = 0.15       # arrow spacing (m)


# Intended linear fields in the net frame of the reconstruction figure (fit values from fig_reconstruction.py:
# vortex J = [[0.07, -0.81], [0.80, 0.05]], saddle J = [[-0.04, -0.81], [-0.85, 0.04]]), idealised. Only the shape
# matters here (arrows are normalised), so they are drawn as the pure vortex and saddle.
BG_J = {"vortex": np.array([[0.0, -1.0], [1.0, 0.0]]), "saddle": np.array([[0.0, -1.0], [-1.0, 0.0]])}


def background_field(ax, field, nr):
    """Semi-transparent arrows of the field v = J (p - p*), colored by direction (the wheel of the
    reconstruction figure), drawn under the paths."""
    J = BG_J[field]
    g = np.arange(-0.675, 0.676, BG_STEP)
    X, Y = np.meshgrid(g, g)
    Q = np.c_[X.ravel(), Y.ravel()]
    F = Q @ J.T
    F = F / np.hypot(*F.T).max() * 0.55                   # longest arrow ~ 0.55 grid steps of length scale
    hue = (np.arctan2(F[:, 1], F[:, 0]) / (2 * np.pi) + 0.5) % 1
    col = mcolors.hsv_to_rgb(np.c_[hue, np.ones_like(hue), 0.95 * np.ones_like(hue)])
    ax.quiver(Q[:, 0], Q[:, 1], F[:, 0], F[:, 1], color=col, alpha=BG_ALPHA, angles="xy", scale_units="xy",
              scale=1.0, width=0.012, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=0)


def trajectories(ax, field, nr, background=False):
    if background:
        background_field(ax, field, nr)
    P = centroid_paths(field, nr)
    for g in range(8):
        group = [P[n] for n in range(10 * g + 1, 10 * g + 11) if n in P]
        for t, c, src in group:
            ax.plot(c[:, 0], c[:, 1], color=COLORS[g], lw=0.4, alpha=0.35)
        m = min(len(c) for _, c, _ in group)
        mean = np.mean([c[:m] for _, c, _ in group], 0)
        ax.plot(mean[:, 0], mean[:, 1], color=COLORS[g], lw=1.2)
        ax.plot(*mean[0], "o", color=COLORS[g], ms=2.5)
    ax.plot(0, 0, "k+", ms=6, mew=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75)
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel("$x$ (m)")


def distance(ax):
    for field, nr, label, col in SETS:
        P = centroid_paths(field, nr)
        m = min(len(c) for _, c, _ in P.values())
        D = np.array([np.hypot(*c[:m].T) for _, c, _ in P.values()])
        t = np.arange(m) * 0.1
        mu, sd = D.mean(0), D.std(0)
        ax.plot(t, mu, color=col, lw=1.2, label=f"{label} ({len(D)} runs)")
        ax.fill_between(t, np.maximum(mu - sd, 1e-3), mu + sd, color=col, alpha=0.18, lw=0)
    ax.axhline(VC / K, color="0.3", lw=0.6, ls=(0, (4, 2)))
    ax.text(0.15, VC / K * 1.08, "$v_c/k$", fontsize=7, va="bottom")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\mathbf{p}_c - \mathbf{p}^*\|$ (m)")
    ax.set_xlim(0, 10)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1)
    ax.legend(frameon=False, loc="upper right")


def main():
    os.makedirs(FIG, exist_ok=True)
    fs.apply()
    fig = plt.figure(figsize=(fs.COLUMN, 2.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.8], hspace=0.55, wspace=0.12)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a); c = fig.add_subplot(gs[1, :])
    trajectories(a, "vortex", 3, True); trajectories(b, "saddle", 3, True); distance(c)
    a.set_ylabel("$y$ (m)"); plt.setp(b.get_yticklabels(), visible=False)
    for ax, t in ((a, "(a) Vortex"), (b, "(b) Saddle"), (c, "(c) Distance to the critical point")):
        ax.set_title(t, pad=3)
        fs.axes(ax)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.165)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_hw_attraction.{ext}"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    trajectories(ax, "vortex", 4); ax.set_ylabel("$y$ (m)"); fs.axes(ax); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hw_trajectories_vortex4R.pdf"))
    plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
