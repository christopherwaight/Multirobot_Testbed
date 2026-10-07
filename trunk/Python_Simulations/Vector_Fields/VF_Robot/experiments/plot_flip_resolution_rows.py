"""
plot_flip_resolution_rows.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_12.tex (candidate, not yet in the draft)
  Makes:  seven-row, two-column versions of the noise figure, one row per start
          (S1 to S7, from the top of the domain). Left column is the sweep against
          sigma_uv (sigma_p = 0), right column against sigma_p (sigma_uv = 0). Each row
          carries a small locator map of the domain showing where the start sits
          against the separatrix x = 0 and the noise-free convergence zone.
          Three treatments of the same data:
            A  Fig. 7 panels per row (success solid, straddle retention dashed)
            B  the gap between success and straddle retention shaded (trials that
               reached the saddle but lost the line), with the D/s1 crossing ratio
            C  the same on a dark background, with drop lines to the axis
  Reads:  the CSVs in plot_flip_resolution_starts.CSV_FILES under
          experiments/outputs/mc_noise_both/, and outputs/mc_zone/zone_trials.csv for
          the zone shading. A row whose CSV does not exist yet falls back to the
          nearest earlier run and is marked PLACEHOLDER on the figure.
  Prints: the 50% crossings per start, for manual review before any enters the .tex.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/plot_flip_resolution_rows.py --variants A,B,C --out-dir <dir>
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from experiments.plot_flip_resolution import NOISE_DIR, FIG_DIR, crossing_50
from experiments.plot_flip_resolution_starts import load, CSV_FILES
import experiments.mc_zone_of_convergence as zone

ROWS = [
    ("S1", (-0.10, 0.30), "0.10 off, upper"),
    ("S2", (0.15, 0.25), "0.15 off, upper"),
    ("S3", (0.0, 0.35), "on line, upper"),
    ("S4", (0.0, 0.0), "origin"),
    ("S5", (0.0, -0.25), "on line, lower"),
    ("S6", (0.15, -0.15), "0.15 off, lower"),
    ("S7", (0.10, -0.20), "0.10 off, lower"),
]
# Stand-ins until the seven-start sweeps finish: the nearest earlier run.
PLACEHOLDER = {"S1": "noise_both_S1.csv", "S5": "noise_both_S6c.csv", "S6": "noise_both_S4b.csv"}

THEMES = {
    "light": dict(bg="white", fg="0.15", grid="0.88", D="0.1", s1="#1f77b4", zone="#bfe3c8",
                  start="#d62728", line="#4a6fa5", dim="0.45"),
    "dark": dict(bg="#0b0f1a", fg="#e6e6e6", grid="#1f2635", D="#f5f5f5", s1="#4cc9f0",
                 zone="#1d5c4c", start="#ff4d6d", line="#7aa2f7", dim="#8b93a7"),
}


# Font and marker sizes. "full" is the text-width figure, "column" the single-column one
# (3.5 in wide), where everything shrinks but stays near 6 pt at print size.
SIZES = {
    "full": dict(tick=6.5, val=6.3, ratio=7.2, ylab=7, title=8, xlab=8, sup=8, leg=6.8, ph=6.5,
                 mk=3.2, mk_s=2.2, lw=1.4, lw_s=1.0, tick_lw=2.0, ins=[0.015, 0.04, 0.27, 0.40],
                 ins_x=3.2, ins_star=4.5, ins_dot=3.6, tick_len=2.5),
    "column": dict(tick=5.5, val=5.2, ratio=5.5, ylab=5.8, title=6.3, xlab=6.5, sup=6.5, leg=5.6,
                   ph=5.5, mk=2.2, mk_s=1.5, lw=1.0, lw_s=0.75, tick_lw=1.5,
                   ins=[0.015, 0.05, 0.30, 0.46], ins_x=2.4, ins_star=3.4, ins_dot=2.8,
                   tick_len=1.8),
}


def zone_mask():
    """Boolean grid, True where both trackers reach the far saddle after a band hold in
    at least 95% of headings (noise-free map), plus its extent."""
    path = os.path.join(project_root, "experiments/outputs/mc_zone/zone_trials.csv")
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    rows = [{"tracker": r["tracker"], "name": r["name"], "start_x": float(r["start_x"]),
             "start_y": float(r["start_y"]), "outcome": r["outcome"]} for r in csv.DictReader(lines)]
    g = zone.grids(rows)
    ok = (g["D"]["p"][1] >= 0.95) & (g["s1"]["p"][1] >= 0.95)
    x0, x1, y0, y1, *_ = zone.cell_grid()
    return ok, (x0, x1, y0, y1)


def locator(ax, start, mask, theme, sz=SIZES["full"]):
    """Small map of the domain [-1, 1] x [-0.5, 0.5]: convergence zone, separatrix x = 0,
    saddles, and the start."""
    ins = ax.inset_axes(sz["ins"])
    if mask is not None:
        ok, (x0, x1, y0, y1) = mask
        ins.imshow(np.where(ok, 1.0, np.nan), extent=(x0, x1, y0, y1), origin="lower",
                   cmap=matplotlib.colors.ListedColormap([theme["zone"]]), vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest", zorder=1)
    ins.plot([0, 0], [-0.5, 0.5], color=theme["line"], linewidth=1.0, linestyle="--", zorder=2)
    ins.plot([0], [0.5], marker="x", color=theme["fg"], markersize=sz["ins_x"], markeredgewidth=0.9,
             zorder=3, clip_on=False)
    ins.plot([0], [-0.5], marker="*", color="#2ca02c", markersize=sz["ins_star"], zorder=3, clip_on=False)
    ins.plot(*start, marker="o", color=theme["start"], markersize=sz["ins_dot"], zorder=4,
             markeredgecolor=theme["bg"], markeredgewidth=0.4)
    ins.set_xlim(-1, 1)
    ins.set_ylim(-0.5, 0.5)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor(theme["bg"])
    for s in ins.spines.values():
        s.set_linewidth(0.5)
        s.set_color(theme["dim"])


def draw_panel(ax, data, axis, variant, theme, sz=SIZES["full"]):
    ax.set_facecolor(theme["bg"])
    ax.axhline(50, color=theme["dim"], linewidth=0.7, linestyle=":", zorder=1)
    cross = {}
    for tr in ("D", "s1"):
        col = theme[tr]
        rows = sorted((r for r in data if r["tracker"] == tr and r["axis"] == axis),
                      key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in rows]
        succ = np.array([100 * r["success"] for r in rows])
        strad = np.array([100 * r["straddle"] for r in rows])
        c = crossing_50(xs, list(succ)) if succ[0] >= 50 else None
        cross[tr] = c
        cs = crossing_50(xs, list(strad)) if strad[0] >= 50 else None
        cross[tr + "_strad"] = cs
        mk = "o" if tr == "D" else "^"
        if variant == "A":
            ax.plot(xs, succ, color=col, marker=mk, markersize=sz["mk"], linewidth=sz["lw"], zorder=3)
            ax.plot(xs, strad, color=col, marker="s", markersize=sz["mk_s"], linewidth=sz["lw_s"],
                    linestyle="--", alpha=0.6, zorder=2)
        else:
            ax.fill_between(xs, strad, succ, color=col, alpha=0.22 if variant == "B" else 0.28,
                            linewidth=0, zorder=1.5)
            line, = ax.plot(xs, succ, color=col, marker=mk, markersize=2.8, linewidth=1.5, zorder=3)
            ax.plot(xs, strad, color=col, linewidth=0.8, alpha=0.75, zorder=2.5)
            if variant == "C":
                line.set_path_effects([pe.Stroke(linewidth=4.2, foreground=col, alpha=0.18),
                                       pe.Normal()])
        if variant in ("B", "C") and cs is not None:
            ax.plot([cs], [50], marker="v", markersize=4.2, markerfacecolor=theme["bg"],
                    markeredgecolor=col, markeredgewidth=1.0, zorder=4.5, linestyle="none")
        if c is not None:
            if variant == "C":
                ax.plot([c, c], [0, 50], color=col, linewidth=0.8, linestyle=":", alpha=0.8, zorder=2)
            ax.plot([c, c], [41, 59], color=col, linewidth=sz["tick_lw"], zorder=4)
    # Value labels after both ticks exist, so two close crossings can be pulled apart and a
    # label near the right edge stays inside the panel.
    cs_list = [(tr, cross[tr]) for tr in ("D", "s1") if cross[tr] is not None]
    if len(cs_list) == 2:
        ratio = max(c for _, c in cs_list) / min(c for _, c in cs_list)
        close = ratio < 1.6 or (max(c for _, c in cs_list) > 0.017 and ratio < 3.2)
        # At column width a label spans most of a decade, so always stack the two.
        close = close or sz is SIZES["column"]
    else:
        close = False
    larger = max(cs_list, key=lambda t: t[1])[0] if cs_list else None
    for tr, c in cs_list:
        below = close and tr == larger
        right = c > 0.017
        ax.annotate(f"{c:.4f}", xy=(c, 40 if below else 60),
                    xytext=(-2 if right else 2, -1 if below else 1), textcoords="offset points",
                    ha="right" if right else "left", va="top" if below else "bottom",
                    fontsize=sz["val"], color=theme[tr], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc=theme["bg"], ec="none", alpha=0.8),
                    zorder=5)
    if variant in ("B", "C") and cross["D_strad"] and cross["s1_strad"]:
        ax.text(0.985, 0.93, f"retention D / s$_1$ = {cross['D_strad'] / cross['s1_strad']:.1f}$\\times$",
                transform=ax.transAxes, ha="right", va="top", fontsize=sz["ratio"], color=theme["fg"],
                fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 3.5e-2)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=sz["tick"], colors=theme["fg"], length=sz["tick_len"], pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(theme["dim"])
    return cross


def build(variant, out_path, data_for, mask, paper=False, column=False):
    theme = THEMES["dark" if variant == "C" else "light"]
    n = len(ROWS)
    # Paper: full text width of an IEEE two-column page, sized to fit a float page.
    # Column: single column of the same page, 3.5 in wide.
    sz = SIZES["column" if column else "full"]
    if column:
        size = (3.5, 0.74 * n + 0.55)
    else:
        size = (7.16, 1.07 * n + 0.75) if paper else (7.4, 1.32 * n + 1.0)
    fig, axes = plt.subplots(n, 2, figsize=size, sharex=True, sharey=True,
                             facecolor=theme["bg"])
    allc = {}
    for i, (name, start, desc) in enumerate(ROWS):
        data, placeholder = data_for[name]
        cu = draw_panel(axes[i][0], data, "uv", variant, theme, sz)
        cp = draw_panel(axes[i][1], data, "p", variant, theme, sz)
        allc[name] = (cu, cp)
        locator(axes[i][0], start, mask, theme, sz)
        coord = lambda v: "0" if v == 0 else f"{v:.2f}"
        ylab = (f"{name}\n({coord(start[0])}, {coord(start[1])})" if column
                else f"{name}  ({coord(start[0])}, {coord(start[1])})\n{desc}")
        axes[i][0].set_ylabel(ylab, fontsize=sz["ylab"], color=theme["fg"], labelpad=2 if column else 4)
        if placeholder:
            axes[i][1].text(0.985, 0.76, "PLACEHOLDER DATA", transform=axes[i][1].transAxes,
                            ha="right", va="top", fontsize=sz["ph"], color="#e5484d", fontweight="bold")
    if column:
        ta, tb = "(a) $\\sigma_{uv}$,  $\\sigma_p = 0$", "(b) $\\sigma_p$,  $\\sigma_{uv} = 0$"
    else:
        ta = "(a)  vs. measurement noise $\\sigma_{uv}$,  $\\sigma_p = 0$"
        tb = "(b)  vs. position noise $\\sigma_p$,  $\\sigma_{uv} = 0$"
    axes[0][0].set_title(ta, fontsize=sz["title"], color=theme["fg"], pad=3 if column else 6)
    axes[0][1].set_title(tb, fontsize=sz["title"], color=theme["fg"], pad=3 if column else 6)
    axes[-1][0].set_xlabel(r"$\sigma_{uv}$", fontsize=sz["xlab"], color=theme["fg"], labelpad=1 if column else 4)
    axes[-1][1].set_xlabel(r"$\sigma_p$", fontsize=sz["xlab"], color=theme["fg"], labelpad=1 if column else 4)
    fig.supylabel("Rate (%)", fontsize=sz["sup"], color=theme["fg"], x=0.04 if column else 0.005)

    cD, cS = theme["D"], theme["s1"]
    if variant == "A":
        handles = [Line2D([], [], color=cD, marker="o", markersize=3, label="$D$ success"),
                   Line2D([], [], color=cD, marker="s", markersize=2.5, linestyle="--", alpha=0.6,
                          label="$D$ straddle"),
                   Line2D([], [], color=cS, marker="^", markersize=3, label="$s_1$ success"),
                   Line2D([], [], color=cS, marker="s", markersize=2.5, linestyle="--", alpha=0.6,
                          label="$s_1$ straddle")]
    else:
        handles = [Line2D([], [], color=cD, marker="o", markersize=3, label="$D$ far-saddle success"),
                   Patch(color=cD, alpha=0.3, label="$D$ reached the saddle, lost the line"),
                   Line2D([], [], color=cS, marker="^", markersize=3, label="$s_1$ far-saddle success"),
                   Patch(color=cS, alpha=0.3, label="$s_1$ reached the saddle, lost the line"),
                   Line2D([], [], color=theme["fg"], linewidth=2.0, label="50% crossing of success"),
                   Line2D([], [], color=theme["fg"], marker="v", markersize=4.2, linestyle="none",
                          markerfacecolor=theme["bg"], markeredgewidth=1.0,
                          label="50% crossing of retention")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3 if variant != "A" else (2 if column else 4),
                     fontsize=sz["leg"], frameon=False, bbox_to_anchor=(0.5, 0.0),
                     columnspacing=1.0 if column else 2.0, handlelength=1.8 if column else 2.0)
    for t in leg.get_texts():
        t.set_color(theme["fg"])
    bottom = 0.032 if column else (0.045 if variant == "A" else 0.065)
    fig.tight_layout(rect=[0.0, bottom, 1, 1], h_pad=0.25 if column else 0.4, w_pad=0.4 if column else 0.6)
    fig.savefig(out_path, dpi=200, facecolor=theme["bg"], bbox_inches="tight")
    print(f"Saved: {out_path}")
    return allc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--out-dir", default=FIG_DIR)
    ap.add_argument("--paper", action="store_true",
                    help="treatment A at IEEE text width, no zone shading in the locators "
                         "(the zone map is below 10^4 trials per cell), written to "
                         "figures/noise_seven_starts.png")
    ap.add_argument("--column", action="store_true",
                    help="as --paper, but at IEEE single-column width (3.5 in), written to "
                         "figures/noise_seven_starts_col.png")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data_for = {}
    for name, _, _ in ROWS:
        real = os.path.join(NOISE_DIR, CSV_FILES[name])
        if os.path.exists(real):
            data_for[name] = (load(real), False)
        else:
            data_for[name] = (load(os.path.join(NOISE_DIR, PLACEHOLDER[name])), True)
            print(f"  {name}: {CSV_FILES[name]} not available yet, using {PLACEHOLDER[name]} (PLACEHOLDER)")
    if args.paper or args.column:
        if any(ph for _, ph in data_for.values()):
            sys.exit("paper figure refused: a row would use placeholder data")
        name = "noise_seven_starts_col.png" if args.column else "noise_seven_starts.png"
        allc = build("A", os.path.join(FIG_DIR, name), data_for, None, paper=True, column=args.column)
    else:
        mask = zone_mask()
        allc = None
        for v in args.variants.split(","):
            allc = build(v, os.path.join(args.out_dir, f"flip_resolution_rows_{v}.png"), data_for, mask)

    fmt = lambda c: "None" if c is None else f"{c:.4f}"
    print("\n50% crossings of success (None = no tick drawn)")
    print(f"{'start':<6}{'D sigma_uv':>12}{'s1 sigma_uv':>13}{'D sigma_p':>12}{'s1 sigma_p':>13}")
    for name, _, _ in ROWS:
        cu, cp = allc[name]
        tag = "  PLACEHOLDER" if data_for[name][1] else ""
        print(f"{name:<6}{fmt(cu['D']):>12}{fmt(cu['s1']):>13}{fmt(cp['D']):>12}{fmt(cp['s1']):>13}{tag}")


if __name__ == "__main__":
    main()
