"""Limit-cycle figure: vortex, 3 robots, r_d = 0.10 m (chosen in limit_cycle.py).

(a) Simulation on the 2026 reconstructed vortex: paths from 8 starts inside (0.04 m) and 8 outside
    (0.30 m) the cycle, and the cycle r(theta) they converge to.
(b) Hardware: the 10 runs at r_d = 0.10 m, mapped into the same field frame and overlaid
    (concatenated), with their mean cycle and the simulated cycle for comparison.
(c) Distance to p* against time: simulation (inside and outside starts) and hardware runs.
Run from the repo root after limit_cycle.py.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import figstyle as fs  # noqa: E402

FIELD, NR, RD = "vortex", 3, 0.10
BINS = np.radians(np.arange(-180, 181, 10))
MID = 0.5 * (BINS[:-1] + BINS[1:])


def closed(curve):
    th = np.r_[MID, MID[0] + 2 * np.pi]
    r = np.r_[curve, curve[0]]
    return r * np.cos(th), r * np.sin(th)


def column_figure(sim, hw):
    """One-column version for the paper (panels a and b only): fig_limit_cycle_column.{pdf,png}."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 2, figsize=(fs.COLUMN, 1.68), sharey=True)
    for C, tag in zip(sim["C"], sim["tags"]):
        ax[0].plot(C[:, 0], C[:, 1], lw=0.5, alpha=0.6, color=fs.BLUE if tag == "in" else fs.RED)
    ax[0].plot(*closed(sim["curve"]), "k", lw=1.3)
    ax[0].legend([Line2D([], [], color=c, lw=1) for c in (fs.BLUE, fs.RED)], ["Start inside", "Start outside"],
                 frameon=False, loc="upper left", fontsize=6, handlelength=1.2, borderaxespad=0.1)
    for C in hw["C"]:
        ax[1].plot(C[:, 0], C[:, 1], lw=0.4, alpha=0.7, color=fs.GREY)
    ax[1].plot(*closed(hw["curve"]), "k", lw=1.3, label="Hardware cycle")
    ax[1].plot(*closed(sim["curve"]), "--", color=fs.GREEN, lw=1.1, label="Simulated cycle")
    ax[1].legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.5, borderaxespad=0.1)
    circ = np.linspace(0, 2 * np.pi, 200)
    for a, title in zip(ax, ("(a) Simulation", "(b) Hardware, ten runs")):
        a.plot(RD * np.cos(circ), RD * np.sin(circ), ":", color=fs.ORANGE, lw=1.0)
        a.plot(0, 0, "k+", ms=5)
        a.set_aspect("equal"); a.set_xlim(-0.22, 0.22); a.set_ylim(-0.22, 0.25)
        a.set_xticks([-0.2, 0, 0.2]); a.set_yticks([-0.2, 0, 0.2])
        a.set_title(title, pad=2); a.set_xlabel("$x$ (m)")
        fs.axes(a)
    ax[0].set_ylabel("$y$ (m)")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.2, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_limit_cycle_column.{ext}"))
    plt.close(fig)


def main():
    D = pickle.load(open(os.path.join(HERE, "outputs", f"paths_{FIELD}.pkl"), "rb"))
    sim, hw = D["sim"][RD], D["hw"][(NR, RD)]
    fs.apply()
    fig, ax = plt.subplots(1, 3, figsize=(fs.PAGE, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.45]))
    for C, tag in zip(sim["C"], sim["tags"]):
        ax[0].plot(C[:, 0], C[:, 1], lw=0.5, alpha=0.6, color=fs.BLUE if tag == "in" else fs.RED)
    ax[0].plot(*closed(sim["curve"]), "k", lw=1.4)
    for C in hw["C"]:
        ax[1].plot(C[:, 0], C[:, 1], lw=0.5, alpha=0.7, color=fs.GREY)
    ax[1].plot(*closed(hw["curve"]), "k", lw=1.4, label="Hardware cycle")
    ax[1].plot(*closed(sim["curve"]), "--", color=fs.GREEN, lw=1.2, label="Simulated cycle")
    for a, title in zip(ax[:2], ("(a) Simulation, reconstructed vortex", "(b) Hardware, ten runs")):
        circ = np.linspace(0, 2 * np.pi, 200)
        a.plot(RD * np.cos(circ), RD * np.sin(circ), ":", color=fs.ORANGE, lw=1.1)
        a.plot(0, 0, "k+", ms=6)
        a.set_aspect("equal"); a.set_xlim(-0.22, 0.22); a.set_ylim(-0.22, 0.22)
        a.set_xticks([-0.2, -0.1, 0, 0.1, 0.2]); a.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        a.set_title(title, pad=3)
    ax[1].legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.5, borderaxespad=0.2)
    t = np.arange(sim["C"].shape[1]) * 0.2
    for C, tag in zip(sim["C"], sim["tags"]):
        ax[2].plot(t, np.hypot(*C.T), lw=0.5, alpha=0.6, color=fs.BLUE if tag == "in" else fs.RED)
    for C in hw["C"]:
        ax[2].plot(np.arange(len(C)) * 0.1, np.hypot(*C.T), lw=0.6, color=fs.GREY, alpha=0.8)
    ax[2].axhline(RD, ls=":", color=fs.ORANGE, lw=1.1)
    from matplotlib.lines import Line2D
    ax[2].legend([Line2D([], [], color=c, lw=1) for c in (fs.BLUE, fs.RED, fs.GREY)],
                 ["Sim., start inside", "Sim., start outside", "Hardware"], frameon=False, ncol=1,
                 loc="upper right", fontsize=6.5, handlelength=1.2)
    ax[2].set_xlim(0, 40); ax[2].set_ylim(0.03, 0.34); ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel(r"$\|\mathbf{p}_c - \mathbf{p}^*\|$ (m)")
    ax[2].set_title("(c) Distance to the critical point", pad=3)
    for a in ax[:2]:
        a.set_xlabel("$x$ (m)"); a.set_ylabel("$y$ (m)")
    for a in ax:
        fs.axes(a)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.18, wspace=0.32)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_limit_cycle.{ext}"))
    column_figure(sim, hw)
    ok = np.isfinite(sim["curve"]) & np.isfinite(hw["curve"])
    rho = np.corrcoef(sim["curve"][ok], hw["curve"][ok])[0, 1]
    gap = np.median(np.abs(sim["curve"][ok] - hw["curve"][ok]))
    print(f"r(theta) of the simulated vs hardware cycle: correlation {rho:.2f}, median gap {1000 * gap:.1f} mm; "
          f"mean radius sim {np.nanmean(sim['curve']):.3f} m, hw {np.nanmean(hw['curve']):.3f} m")


if __name__ == "__main__":
    main()
