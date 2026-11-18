"""Hardware against matched simulation, orbits (paper figure, Sec. VI-A).

Rows: vortex, saddle. Columns: radial bias and within-run spread against r_d.
Hardware (2026 runs): markers, bias with +-1 between-run spread s_b; 3R and 4R.
Matched simulation (paired_runs.py: each run from its own start and held formation): reconstructed field
solid, linear field dashed, group means.
Writes analysis/simulation/figures/fig_sim_vs_hw.{pdf,png} and prints the gaps quoted in the text.
Run from the repo root after paired_runs.py and score_hardware.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402
import figstyle as fs  # noqa: E402

KEYS = ["field", "nr", "rd"]


def load():
    S = pd.read_csv(os.path.join(HERE, "paired", "paired_orbits.csv"))
    S = S[~S.recon_lost].groupby(KEYS)[["linear_bias_w1", "recon_bias_w1", "linear_sd_within_w1",
                                        "recon_sd_within_w1"]].mean().reset_index()
    H = pd.read_csv(os.path.join(paths.OUT_HW, "outputs", "orbit_summary.csv"))[KEYS + ["bias", "s_b", "s_w"]]
    return H.merge(S, on=KEYS)


def gaps(M):
    big = M.rd >= 0.3
    ratio = M.s_w / M.recon_sd_within_w1
    small = M[M.rd == 0.01]
    print(f"orbit bias at r_d 0.01: hardware - recon sim = {1000 * (small.bias - small.recon_bias_w1).min():.0f} "
          f"to {1000 * (small.bias - small.recon_bias_w1).max():.0f} mm")
    print(f"within-run spread, r_d >= 0.3: hardware / recon sim = {ratio[big].min():.1f} to {ratio[big].max():.1f}, "
          f"median {ratio[big].median():.1f}; linear sim spread at most {1000 * M.linear_sd_within_w1.max():.1f} mm")
    print(f"bias, all groups: RMS (hardware - recon sim) {1000 * np.sqrt(((M.bias - M.recon_bias_w1) ** 2).mean()):.0f} mm, "
          f"RMS (hardware - linear sim) {1000 * np.sqrt(((M.bias - M.linear_bias_w1) ** 2).mean()):.0f} mm")
    for (f, nr), g in M.groupby(["field", "nr"]):
        print(f"  {f} {nr}R: recon sim spread / hardware spread at r_d >= 0.3: "
              + ", ".join(f"{r:.2f}" for r in (g.recon_sd_within_w1 / g.s_w)[g.rd >= 0.3]))


def main():
    fs.apply()
    M = load()
    gaps(M)
    fig, ax = plt.subplots(2, 2, figsize=(fs.COLUMN, 2.45), sharex=True)
    for row, field in enumerate(("vortex", "saddle")):
        for nr, col, dx in ((3, fs.BLUE, -0.006), (4, fs.ORANGE, 0.006)):
            g = M[(M.field == field) & (M.nr == nr)].sort_values("rd")
            lab = f"{nr} robots"
            ax[row, 0].errorbar(g.rd + dx, g.bias, yerr=g.s_b, fmt="o", ms=3, color=col, elinewidth=0.8, capsize=1.5,
                                label=f"Hardware, {lab}")
            ax[row, 0].plot(g.rd, g.recon_bias_w1, "-", color=col, lw=1.0, label=f"Sim., reconstructed, {lab}")
            ax[row, 1].plot(g.rd + dx, g.s_w, "o", ms=3, color=col)
            ax[row, 1].plot(g.rd, g.recon_sd_within_w1, "-", color=col, lw=1.0)
        g = M[(M.field == field) & (M.nr == 3)].sort_values("rd")
        ax[row, 0].plot(g.rd, g.linear_bias_w1, "--", color=fs.GREY, lw=1.0, label="Sim., linear")
        ax[row, 1].plot(g.rd, g.linear_sd_within_w1, "--", color=fs.GREY, lw=1.0)
        ax[row, 0].axhline(0, color="0.6", lw=0.6)
        ax[row, 0].set_ylabel(f"{field.capitalize()}, bias (m)")
        ax[row, 1].set_ylabel("Spread (m)")
    ax[0, 0].set_title("(a) Radial bias", pad=3)
    ax[0, 1].set_title("(b) Within-run spread", pad=3)
    for a in ax.ravel():
        fs.axes(a)
        a.set_xlim(-0.03, 0.53)
        a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for a in ax[1]:
        a.set_xlabel("$r_d$ (m)")
    for a in ax[:, 1]:
        a.set_ylim(0, 0.09)
        a.set_yticks([0, 0.04, 0.08])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.2", marker="o", ms=3, ls=""), Line2D([], [], color="0.2", lw=1.0),
         Line2D([], [], color=fs.GREY, lw=1.0, ls="--")]
    ax[1, 1].legend(h, ["Hardware", "Sim., reconstructed", "Sim., analytical"], loc="upper left", frameon=False,
                    fontsize=6, handlelength=1.5, borderaxespad=0.1, labelspacing=0.3)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.14, wspace=0.32, hspace=0.12)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_sim_vs_hw.{ext}"))


if __name__ == "__main__":
    main()
