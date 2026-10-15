"""
plot_flip_resolution_starts.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_12.tex (candidate, not yet in the draft)
  Makes:  figures/flip_resolution_starts.png. One column per start, each column
          drawn like Fig. fig:flip_resolution (plot_flip_resolution.py): row (a)
          against sigma_uv with sigma_p = 0, row (b) against sigma_p with
          sigma_uv = 0, D and s1 far-saddle success solid, straddle retention
          dashed, a dotted 50% line, and a tick at each tracker's 50% crossing.
          Seven columns S1 to S7, ordered from the top of the domain. S3, the
          straddling start of Draft_12, is the Fig. 7 column.
  Reads:  the CSV named for each column in CSV_FILES, under
          experiments/outputs/mc_noise_both/, all written by
          mc_sweep_noise_both_trackers.py (10000 trials per cell, same sigma grid
          and seeds, so trial t has the same heading at every start).
  Prints: the 50% crossings per start, tracker and axis, for manual review
          before any of them enters the .tex.

A crossing tick is drawn only for a curve that starts at or above 50% at the
lowest noise level. A curve that starts below 50% and rises (S6, where the
noise-free outcome depends on heading) has no meaningful crossing.

Does not overwrite figures/flip_resolution.png, which Draft_12 includes.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/plot_flip_resolution_starts.py
  venv/bin/python3 experiments/plot_flip_resolution_starts.py --starts S1,S2,S3,S4
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_flip_resolution import (NOISE_DIR, FIG_DIR, D_STYLE, S1_STYLE,
                                              crossing_50)

# tag -> column title. "orig" is noise_both.csv, the Draft_12 start.
START_TITLES = {
    "S1": "S1  (-0.10, 0.30)\n0.10 off, not prestraddled",
    "S2": "S2  (0.15, 0.25)\n0.15 off, not prestraddled",
    "S3": "S3  (0, 0.35)\non line, prestraddled\n(the start of Fig. 7)",
    "S4": "S4  (0, 0)\norigin, prestraddled",
    "S5": "S5  (0, -0.25)\non line, prestraddled",
    "S6": "S6  (0.15, -0.15)\n0.15 off, not prestraddled",
    "S7": "S7  (0.10, -0.20)\n0.10 off, not prestraddled",
}
# Column name -> CSV in NOISE_DIR. The seven-start numbering of 2026-10-06 does not
# match the tags of the older runs, so four columns read files with another name.
CSV_FILES = {
    "S1": "noise_both_P1.csv",
    "S2": "noise_both_S5.csv",   # run earlier as S5
    "S3": "noise_both.csv",      # the straddling start of Draft_12
    "S4": "noise_both_S3.csv",   # run earlier as S3
    "S5": "noise_both_P5.csv",
    "S6": "noise_both_P6.csv",
    "S7": "noise_both_S4.csv",   # run earlier as S4
}
DEFAULT_STARTS = "S1,S2,S3,S4,S5,S6,S7"


def load(path):
    with open(path) as f:
        rows = [l.strip() for l in f if not l.startswith("#")]
    header = rows[0].split(",")
    data = []
    for line in rows[1:]:
        r = dict(zip(header, line.split(",")))
        data.append({"tracker": r["tracker"], "axis": r["axis"],
                     "sigma": float(r["sigma_uv"]) if r["axis"] == "uv" else float(r["sigma_p"]),
                     "success": float(r["success"]),
                     "straddle": float(r["success_straddle"])})
    return data


def panel(ax, data, axis, xlabel):
    """Same drawing as plot_flip_resolution._panel, minus the tick on a curve
    that starts below 50%."""
    ax.axhline(50, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
    crossings = {}
    for tr, style, name in (("D", D_STYLE, "$D$"), ("s1", S1_STYLE, "$s_1$")):
        rows = sorted((r for r in data if r["tracker"] == tr and r["axis"] == axis),
                      key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in rows]
        succ = [100 * r["success"] for r in rows]
        c = crossing_50(xs, succ) if succ[0] >= 50 else None
        crossings[tr] = c
        ax.plot(xs, succ, zorder=3, **style)
        ax.plot(xs, [100 * r["straddle"] for r in rows], color=style["color"],
                marker="s", markersize=3, linewidth=1.1, linestyle="--",
                alpha=0.6, zorder=2)
        if c is not None:
            ax.plot([c, c], [40, 60], color=style["color"], linewidth=2.0, zorder=4)
            ax.annotate(f"{c:.4f}", xy=(c, 60), xytext=(2, 1), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.5, color=style["color"])
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 3.5e-2)
    ax.set_ylim(-3, 103)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return crossings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", default=DEFAULT_STARTS,
                    help="comma-separated tags, left to right")
    args = ap.parse_args()

    cols = []
    for tag in args.starts.split(","):
        name = CSV_FILES[tag]
        path = os.path.join(NOISE_DIR, name)
        if os.path.exists(path):
            cols.append((tag, load(path)))
        else:
            print(f"  not yet available: {name}")

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 0.6, 5.0), sharey=True, squeeze=False)
    crossings = {}
    for j, (tag, data) in enumerate(cols):
        crossings[tag] = {
            "uv": panel(axes[0][j], data, "uv", r"$\sigma_{uv}$"),
            "p": panel(axes[1][j], data, "p", r"$\sigma_p$"),
        }
        axes[0][j].set_title(START_TITLES[tag], fontsize=8)
        if j == 0:
            axes[0][j].set_ylabel("(a)  vs. $\\sigma_{uv}$, $\\sigma_p = 0$\nRate (%)", fontsize=8)
            axes[1][j].set_ylabel("(b)  vs. $\\sigma_p$, $\\sigma_{uv} = 0$\nRate (%)", fontsize=8)

    handles = [Line2D([], [], **D_STYLE, label="$D$ success"),
               Line2D([], [], color=D_STYLE["color"], marker="s", markersize=3, linewidth=1.1,
                      linestyle="--", alpha=0.6, label="$D$ straddle"),
               Line2D([], [], **S1_STYLE, label="$s_1$ success"),
               Line2D([], [], color=S1_STYLE["color"], marker="s", markersize=3, linewidth=1.1,
                      linestyle="--", alpha=0.6, label="$s_1$ straddle")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "flip_resolution_starts.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print("\n50% crossings, success (None = no tick drawn)")
    print(f"{'start':<6}{'D sigma_uv':>12}{'s1 sigma_uv':>13}{'D sigma_p':>12}{'s1 sigma_p':>13}")
    fmt = lambda v: "None" if v is None else f"{v:.4f}"
    for tag, _ in cols:
        c = crossings[tag]
        print(f"{tag:<6}{fmt(c['uv']['D']):>12}{fmt(c['uv']['s1']):>13}"
              f"{fmt(c['p']['D']):>12}{fmt(c['p']['s1']):>13}")


if __name__ == "__main__":
    main()
