"""Paper figure style: Times text, STIX math, 8 pt at print size, outward ticks, grey spines, no grid.
Draw figures at their printed width (3.5 in for one column, 7.16 in for two) so the fonts print at 8 pt."""
import matplotlib.pyplot as plt

COLUMN, PAGE = 3.5, 7.16
BLUE, ORANGE, GREEN, RED, GREY = "#1f5fa8", "#e8692c", "#2a9d5c", "#d1495b", "#555555"


def apply():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.2,
                         "axes.linewidth": 0.8, "savefig.dpi": 600})


def axes(ax):
    ax.tick_params(direction="out", length=2.5, width=0.8, colors="0.25")
    for s in ax.spines.values():
        s.set_color("0.3")
        s.set_linewidth(0.8)
