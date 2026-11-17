"""Hardware attraction figures (replace the paper's placeholder_hw_* images).

(a) vortex 3R and (b) saddle 3R: every run's centroid path from the eight start groups, measured from
the session's fitted p*, with each group's mean path (sample-by-sample mean of its runs).
(c) distance to p* against time, mean +- 1 sd across runs: vortex 3R, saddle 3R, vortex 4R.

Run from the repo root after score_hardware.py. Writes analysis/hardware/figures/.
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

FIG = os.path.join(paths.OUT_HW, "figures")
COLORS = plt.cm.tab10(np.arange(8))
SETS = (("vortex", 3, "Vortex, 3 robots"), ("saddle", 3, "Saddle, 3 robots"), ("vortex", 4, "Vortex, 4 robots"))


def centroid_paths(field, nr):
    runs, ps, _ = hw2025.load(field, nr)
    return {n: (r["t"], r["pos"].mean(1) - ps, r["source"]) for n, r in sorted(runs.items())}


def trajectories(field, nr, name):
    P = centroid_paths(field, nr)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    for g in range(8):
        group = [P[n] for n in range(10 * g + 1, 10 * g + 11) if n in P]
        for t, c, src in group:
            ax.plot(c[:, 0], c[:, 1], color=COLORS[g], lw=0.5, alpha=0.35, ls="--" if src != "2025" else "-")
        m = min(len(c) for _, c, _ in group)
        mean = np.mean([c[:m] for _, c, _ in group], 0)
        ax.plot(mean[:, 0], mean[:, 1], color=COLORS[g], lw=1.8)
        ax.plot(*mean[0], "o", color=COLORS[g], ms=4)
    ax.plot(0, 0, "k+", ms=10, mew=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x from p* (m)")
    ax.set_ylabel("z from p* (m)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_hw_trajectories_{name}.{ext}"), dpi=200)
    plt.close(fig)


def distance():
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for (field, nr, label), col in zip(SETS, ("#1f5fa8", "#d1495b", "#2a9d5c")):
        P = centroid_paths(field, nr)
        m = min(len(c) for _, c, _ in P.values())
        D = np.array([np.hypot(*c[:m].T) for _, c, _ in P.values()])
        t = np.arange(m) * 0.1
        mu, sd = D.mean(0), D.std(0)
        ax.plot(t, mu, color=col, lw=1.6, label=f"{label} ({len(D)} runs)")
        ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance to p* (m)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_hw_distance.{ext}"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    trajectories("vortex", 3, "vortex")
    trajectories("saddle", 3, "saddle")
    trajectories("vortex", 4, "vortex4R")
    distance()
    print("figures in", FIG, sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
