"""
Figures for the V-B to VI-A draft, read from results/ (run analyze.py first).

  fig_attraction_ecdf.pdf   per-trial final error as an empirical CDF, one curve per field and team,
                            with the 3, 4 and 5 cm convergence thresholds. Success at any threshold
                            is read off the curve, which answers R1.6 and R3.4 (per-trial errors).
  fig_orbit_radius.pdf      (a) mean radial error against commanded radius with between-trial
                            spread, and the practical-range tolerance line; (b) within-run against
                            between-trial standard deviation (R1.4, R3.4).

Colour carries the field (vortex blue, saddle orange). Line style and marker fill carry the team
size (three robots solid and filled, four robots dashed and hollow), so neither is colour alone.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.dirname(HERE)
RES = os.path.join(DRAFT, "results")
FIG = os.path.join(DRAFT, "figures")

COL = {"vortex": "#2a78d6", "saddle": "#eb6834"}
LS = {3: "-", 4: "--"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def label(field, n, origin):
    return f"{field.capitalize()}, {n}R" + ("" if origin == "real" else " (synthetic)")


def attraction_ecdf():
    rows = read_csv("attraction_trials.csv")
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for eps in (0.03, 0.04, 0.05):
        ax.axvline(1000 * eps, color=GRID, lw=1, zorder=0)
        ax.text(1000 * eps, 1.02, f"{round(eps * 100)} cm", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    for field in ("vortex", "saddle"):
        for n in (3, 4):
            g = [r for r in rows if r["field"] == field and int(r["n_robots"]) == n]
            if not g:
                continue
            e = np.sort([1000 * float(r["e_final"]) for r in g])
            p = np.arange(1, len(e) + 1) / len(e)
            ax.step(e, p, where="post", color=COL[field], ls=LS[n], label=label(field, n, g[0]["origin"]))
    ax.set_xlabel("Final error $|\\mathbf{p}_c(T)-\\mathbf{p}^*|$ (mm)")
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    out = os.path.join(FIG, "fig_attraction_ecdf.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=200)
    plt.close(fig)


def orbit_radius():
    with open(os.path.join(RES, "summary.json")) as f:
        so = json.load(f)["orbit"]
    frac = 0.25
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True, gridspec_kw=dict(hspace=0.12))
    rd_line = np.linspace(0, 0.52, 50)
    a.fill_between(rd_line, -frac * rd_line, frac * rd_line, color=GRID, lw=0, zorder=0)
    a.text(0.02, -0.125, f"shaded: $|$error$|\\leq${frac:.2f}$\\,r_d$", fontsize=6.5, color=MUTED)
    a.axhline(0, color=MUTED, lw=0.6)
    for field in ("vortex", "saddle"):
        for n in (3, 4):
            rows = so[f"{field}_{n}R"]["rows"]
            rd = np.array([r["rd"] for r in rows])
            off = (-0.004 if n == 3 else 0.004) + (-0.002 if field == "vortex" else 0.002)
            mfc = COL[field] if n == 3 else "white"
            a.errorbar(rd + off, [r["bias"] for r in rows], yerr=[r["bias_between_sd"] for r in rows],
                       color=COL[field], ls=LS[n], marker="o", ms=3.5, mfc=mfc, capsize=1.5, elinewidth=0.8,
                       label=f"{field.capitalize()}, {n}R")
            b.plot(rd, [r["sd_within"] for r in rows], color=COL[field], ls=LS[n], marker="o", ms=3.5, mfc=mfc)
            if n == 3:
                b.plot(rd, [r["bias_between_sd"] for r in rows], color=COL[field], ls=":", marker="s", ms=3,
                       mfc="white", lw=1)
    a.set_ylabel("Radial bias (m)")
    b.legend(*a.get_legend_handles_labels(), frameon=False, ncol=2, loc="upper left", fontsize=6.5)
    a.text(0.97, 0.9, "(a)", transform=a.transAxes, fontsize=8, ha="right")
    b.set_ylabel("Std. dev. (m)")
    b.set_xlabel("Commanded radius $r_d$ (m)")
    b.text(0.97, 0.9, "(b)", transform=b.transAxes, fontsize=8, ha="right")
    b.text(0.02, 0.082, "within run (solid, dashed)", fontsize=6.5, color=MUTED)
    b.text(0.21, 0.004, "between trials (3R, dotted)", fontsize=6.5, color=MUTED)
    b.set_ylim(0, 0.16)
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(-0.01, 0.52)
    out = os.path.join(FIG, "fig_orbit_radius.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    attraction_ecdf()
    orbit_radius()
    print("wrote", ", ".join(sorted(os.listdir(FIG))))


if __name__ == "__main__":
    main()
