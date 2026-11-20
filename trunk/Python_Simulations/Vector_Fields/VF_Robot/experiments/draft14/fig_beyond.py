"""
fig_beyond.py  (Draft_14 figure, "Beyond the Steady Gyre")

Full-text-width strip of four fields: the steady double gyre, the bent gyre at
delta 0.1 and 0.2, and the Bickley jet. Each panel shows the separatrix, the
true trench of D and of s1 (from the analytic field, not the fit), shading
where rotation dominates (D > 0), and one noise-free ride of each tracker
from one start on the 2026-11-16 robot model. Built from the helpers of
experiments/unsteady_dg/fig_notes.py (the Nov 6 notes' Figs. N1 and N3).

Run from VF_Robot:  D14=1 venv/bin/python3 experiments/draft14/fig_beyond.py
Writes Paper_Writing/Separatrix_and_OW_Paper/figures/beyond_steady_d14.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("D14", "1")

import matplotlib.pyplot as plt
import experiments.unsteady_dg.fig_notes as N
import experiments.unsteady_dg.mc_bent_double_gyre as bent
import experiments.unsteady_dg.mc_bickley_jet as bj

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))),
                   "Paper_Writing", "Separatrix_and_OW_Paper", "figures", "beyond_steady_d14.png")


def main():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7})
    ys = np.linspace(-0.49, 0.49, 99)
    gyres = [("(a) Steady double gyre", 0.0, "S1"),
             ("(b) Bent gyre, $\\delta = 0.1$", 0.1, "S1"),
             ("(c) Bent gyre, $\\delta = 0.2$", 0.2, "S1")]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.4),
                            gridspec_kw={"width_ratios": [0.8, 0.8, 0.8, 1.45]})
    for ax, (title, d, start) in zip(axs[:3], gyres):
        fun = N.f_bent(d)
        xsep = lambda y, d=d: -d * np.cos(np.pi * y)
        N.background(ax, fun, (-0.5, 0.35), (-0.55, 0.55), n=70)
        xd, xs1 = N.trench_gyre(fun, xsep, ys, n=301)
        ax.plot(xsep(ys), ys, color=N.INK, lw=1.8, label="separatrix", zorder=3)
        ax.plot(xd, ys, color=N.C_D, lw=1.0, alpha=0.75, label="$D$ trench", zorder=3)
        ax.plot(xs1, ys, color=N.C_S, lw=1.0, ls="--", alpha=0.75, label="$s_1$ trench", zorder=3)
        ax.plot([xsep(-0.5), xsep(0.5)], [-0.5, 0.5], "x", color=N.INK, ms=5, mew=1.3,
                clip_on=False, zorder=6)
        for tr, col, ls, mk in (("D", N.C_D, "-", "o"), ("s1", N.C_S, "--", "^")):
            path, outc = N.ride("bent", d, tr, start)
            ax.plot(path[:, 0], path[:, 1], color=col, lw=1.4, ls=ls, zorder=5)
            ax.plot(*path[-1], mk, ms=4, color=col, mfc=col if outc == "p1" else "white",
                    mew=1.0, zorder=7)
            print(f"{title}: {tr} from {start} -> {outc}")
        ax.plot(*bent.STARTS[start], "o", ms=3.5, color=N.MUTED, zorder=7)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("$x$", labelpad=1)
        ax.tick_params(length=2, pad=1)
    axs[0].set_ylabel("$y$", labelpad=1)

    ax = axs[3]
    N.background(ax, bj.bickley, (-0.62, 0.62), (-0.12, 0.46), n=90)
    xs = np.linspace(-0.49, 0.49, 99)
    yd, ys1 = N.trench_bickley(xs, n=301)
    ax.plot(bj.ARC[:, 0], bj.ARC[:, 1], color=N.INK, lw=1.8, zorder=3)
    ax.plot(xs, yd, color=N.C_D, lw=1.0, alpha=0.75, zorder=3)
    ax.plot(xs, ys1, color=N.C_S, lw=1.0, ls="--", alpha=0.75, zorder=3)
    ax.plot([bj.P1[0], bj.P2[0]], [bj.P1[1], bj.P2[1]], "x", color=N.INK, ms=5, mew=1.3, zorder=6)
    start = "B2"
    for tr, col, ls, mk in (("D", N.C_D, "-", "o"), ("s1", N.C_S, "--", "^")):
        path, outc = N.ride("bickley", bj.EPS, tr, start)
        path = path[(path[:, 1] > -0.12) & (path[:, 1] < 0.46) & (np.abs(path[:, 0]) < 0.62)]
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.4, ls=ls, zorder=5)
        if len(path):
            ax.plot(*path[-1], mk, ms=4, color=col, mfc=col if outc == "p1" else "white",
                    mew=1.0, zorder=7)
        print(f"Bickley: {tr} from {start} -> {outc}")
    ax.plot(*bj.STARTS[start], "o", ms=3.5, color=N.MUTED, zorder=7)
    ax.set_title("(d) Bickley jet", loc="left", pad=3)
    ax.set_xlabel("$x$", labelpad=1)
    ax.tick_params(length=2, pad=1)

    h, l = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h += [Line2D([], [], color=N.C_D, lw=1.4, marker="o", ms=4, label="$D$ tracker ride"),
          Line2D([], [], color=N.C_S, lw=1.4, ls="--", marker="^", ms=4, label="$s_1$ tracker ride"),
          Patch(color=N.ROT_FILL, label="$D > 0$")]
    l = [x.get_label() for x in h]
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.4)
    fig.savefig(OUT, dpi=300)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
