"""
Diagnostic figures for the revision, sized for IEEE columns (PNG at 300 dpi and PDF):
  figures/hw_final_error.{png,pdf}   per-run final error, hardware attraction (single column)
  figures/orbit_radius.{png,pdf}     mean orbit radius against r_d: hardware, simulation, theory (double column)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")

# validated categorical slots 1-3 (light mode) and neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def hw_final_error(att):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rng = np.random.default_rng(0)
    groups = [(f, f"{f.capitalize()}\n({(att.field == f).sum()} runs)", col) for f, col in (("saddle", BLUE), ("vortex", ORANGE))]
    for i, (f, label, col) in enumerate(groups):
        e = att[att.field == f].e_final.values * 1000
        ax.boxplot(e, positions=[i], widths=0.45, showfliers=False, whis=(5, 95),
                   medianprops=dict(color=INK, lw=1.0), boxprops=dict(color=INK2, lw=0.7),
                   whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(e)), e, s=9, color=col, alpha=0.75, lw=0.4,
                   edgecolor="white", zorder=3)
        ax.scatter([i + 0.32], [e.mean()], marker="D", s=16, color="white", edgecolor=INK, lw=0.8, zorder=4)
    for eps in (30, 40, 50):
        ax.axhline(eps, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=1)
        ax.text(1.62, eps, f"$\\varepsilon$ = {eps} mm", va="center", ha="left", color=INK2, fontsize=7)
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(55, 1.08 * att.e_final.max() * 1000))
    ax.set_ylabel("Final centroid error (mm)")
    ax.yaxis.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.scatter([], [], marker="D", s=16, color="white", edgecolor=INK, label="mean")
    ax.legend(loc="lower left", fontsize=7, handletextpad=0.2)
    save(fig, "hw_final_error")


def orbit_radius(S, hw_orb):
    T = S["theory"]
    rds = np.array(T["rd"])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True)
    panels = [("vortex", "vortex", "vortex_nn", "Simulink 3R (k_t 0.15, k_r 0.2)", "Vortex"),
              ("saddle", "saddle", "saddle_nn", "Simulink 3R (k_t 0.15, k_r 0.2)", "Saddle")]
    lin = S["sim"]["orbit"]["vortex"]
    for ax, (field, series, nn, sim_gain, title) in zip(axes, panels):
        ax.plot([0, 0.72], [0, 0.72], color=GRID, lw=0.8, zorder=0, label="$r = r_d$")
        th_lines = [("task_formula_paper_gains", (0, (4, 2)), "small-angle formula, paper gains"),
                    ("friction_scaled paper (k_t 0.2, k_r 0.5)", "-", "friction-scaled, paper gains"),
                    ("friction_scaled " + sim_gain, (0, (1, 1.5)), "friction-scaled, " + sim_gain.split(" (")[0] + " gains")]
        for key, ls, lab in th_lines:
            ax.plot(rds, T[key], color=INK2, lw=0.8, ls=ls, label=lab)
        r_lin = sorted((float(k), v["mean_r_w1"]) for k, v in lin.items())
        ax.plot(*zip(*r_lin), color=AQUA, lw=1.4, marker="o", ms=3.5, label="sim, linear fields")
        r_nn = sorted((float(k), v["mean_r_w1"], v["between_sd_w1"]) for k, v in S["sim"]["orbit"][nn].items())
        x, y, sd = map(np.array, zip(*r_nn))
        ax.errorbar(x, y, yerr=sd, color=ORANGE, lw=1.4, marker="s", ms=3.5, capsize=2, elinewidth=0.8,
                    label="sim, reconstructed field")
        h = hw_orb[hw_orb.series == series].sort_values("rd")
        ax.errorbar(h.rd, h.mean_r_w1, yerr=h.lap_mean_r_sd.fillna(0), color=BLUE, lw=0, marker="D", ms=4.5,
                    capsize=2, elinewidth=0.8, mec="white", mew=0.5, label="hardware (lap SD bars)", zorder=5)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Commanded radius $r_d$ (m)")
        ax.set_xlim(0, 0.72)
        ax.set_ylim(0, 0.72)
        ax.grid(True, color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean radius after 10 s (m)")
    handles, labels = [], []
    for ax in axes:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    save(fig, "orbit_radius")


def main(tag=""):
    S = json.load(open(os.path.join(RES, f"summary{tag}.json")))
    hw_final_error(pd.read_csv(os.path.join(RES, "hw_attraction_per_trial.csv")))
    orbit_radius(S, pd.read_csv(os.path.join(RES, "hw_orbit_per_run.csv")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main("_quick" if "--quick" in sys.argv else "")
