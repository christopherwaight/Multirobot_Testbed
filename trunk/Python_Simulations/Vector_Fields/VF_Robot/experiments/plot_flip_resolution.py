"""
plot_flip_resolution.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_10.tex
  Makes:  figures/flip_resolution.png, Fig. fig:flip_resolution (Behavior Under
          Noise). Far-saddle success of BOTH trackers against sigma_uv (panel a,
          sigma_p = 0) and sigma_p (panel b, sigma_uv = 0) on a log axis, the
          s1 tracker's straddle retention dashed, a dotted 50% line, and a tick
          at each tracker's 50% crossing (log-linear interpolation).
  Reads:  experiments/outputs/mc_noise_both/noise_both.csv, written by
          mc_sweep_noise_both_trackers.py (10000 trials per cell).
  Prints: the crossings and the D/s1 ratio on each axis, for manual review
          before any of them enters the .tex.

Straddle retention is counted only on trials that reached the far saddle, so it
is a subset of success (see mc_sweep_flip_resolution.py for why).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/plot_flip_resolution.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_DIR = os.path.join(project_root, "experiments", "outputs", "mc_noise_both")
# project_root is .../Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot;
# the paper lives at .../Multirobot_Testbed/Paper_Writing/Separatrix_and_OW_Paper.
REPO_ROOT = project_root
for _ in range(4):
    REPO_ROOT = os.path.dirname(REPO_ROOT)
FIG_DIR = os.path.join(REPO_ROOT, "Paper_Writing", "Separatrix_and_OW_Paper", "figures")


D_STYLE = dict(color="0.1", marker="o", markersize=4, linewidth=1.5)
S1_STYLE = dict(color="#1f77b4", marker="^", markersize=4, linewidth=1.5)


def crossing_50(xs, ys):
    """sigma where success first falls through 50%, log-linear interpolation."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= 50 > y1:
            f = (y0 - 50) / (y0 - y1)
            return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return None


def _panel(ax, data, axis, xlabel, title):
    ax.axhline(50, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
    crossings = {}
    for tr, style, name in (("D", D_STYLE, "$D$"),
                            ("s1", S1_STYLE, "$s_1$")):
        rows = sorted((r for r in data if r["tracker"] == tr and r["axis"] == axis),
                      key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in rows]
        succ = [100 * r["success"] for r in rows]
        c = crossing_50(xs, succ)
        crossings[tr] = c
        ax.plot(xs, succ, label=f"{name} success", zorder=3, **style)
        ax.plot(xs, [100 * r["straddle"] for r in rows], color=style["color"],
                marker="s", markersize=3, linewidth=1.1, linestyle="--",
                alpha=0.6, label=f"{name} straddle", zorder=2)
        if c is not None:
            ax.plot([c, c], [40, 60], color=style["color"], linewidth=2.0, zorder=4)
            # Value just right of the tick, above the line: clear of both
            # trackers' curves on both panels.
            ax.annotate(f"{c:.4f}", xy=(c, 60), xytext=(2, 1),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=6.5, color=style["color"])
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 3.5e-2)
    ax.set_ylim(-3, 103)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Rate (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)
    return crossings


def main():
    data = []
    with open(os.path.join(NOISE_DIR, "noise_both.csv")) as f:
        rows = [l.strip() for l in f if not l.startswith("#")]
    header = rows[0].split(",")
    for line in rows[1:]:
        r = dict(zip(header, line.split(",")))
        data.append({"tracker": r["tracker"], "axis": r["axis"],
                     "sigma": float(r["sigma_uv"]) if r["axis"] == "uv" else float(r["sigma_p"]),
                     "success": float(r["success"]),
                     "straddle": float(r["success_straddle"])})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.2))
    c_uv = _panel(ax1, data, "uv", r"$\sigma_{uv}$",
                  "(a) vs. measurement noise, $\\sigma_p = 0$")
    c_p = _panel(ax2, data, "p", r"$\sigma_p$",
                 "(b) vs. position noise, $\\sigma_{uv} = 0$")
    # One shared legend under both panels, so no entry sits on a curve.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "flip_resolution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    for name, c in (("sigma_uv", c_uv), ("sigma_p", c_p)):
        ratio = c["D"] / c["s1"] if c["D"] and c["s1"] else None
        print(f"50% crossings along {name}: D {c['D']}, s1 {c['s1']}, D/s1 {ratio}")


if __name__ == "__main__":
    main()
