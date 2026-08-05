"""
plot_flip_resolution.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  figures/flip_resolution.png, the two-panel figure for Fig. 7
          (fig:flip_resolution). Panel (a) is the original sigma_uv-axis
          tangent-sign-flip finding (Section results_oecs); panel (b) is the
          sigma_p-axis companion added in response to the referee-report
          finding M3 (Table II(b)'s sigma_uv=0 row collapses with no
          intermediate sigma_p points to show the transition's shape).
  Reads:  experiments/outputs/mc_oecs_traverse/summary_single_target.csv (for
          the sigma_uv=0.001 anchor of panel (a) and the sigma_p=0.005 anchor
          of panel (b), both run at the coarse 2D grid) and
          experiments/outputs/mc_oecs_traverse/flip_resolution.csv (panel a's
          fine sweep, sigma_uv = 0.0015 through 0.008, sigma_p = 0) and
          experiments/outputs/mc_oecs_traverse/flip_resolution_sigma_p.csv
          (panel b's fine sweep, sigma_p = 0.0005 through 0.005, sigma_uv = 0).

Plots far-saddle success and straddle retention (both single-far-saddle-target
scored, straddle conditioned on that same trial's own far-saddle contact -- see
rescore_single_target.py and mc_sweep_flip_resolution.py docstrings for why this
conditioning matters) against sigma_uv (panel a) and sigma_p (panel b).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/plot_flip_resolution.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OECS_DIR = os.path.join(project_root, "experiments", "outputs", "mc_oecs_traverse")
# project_root is .../Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot;
# the paper lives at .../Multirobot_Testbed/Paper_Writing/Separatrix_and_OW_Paper.
REPO_ROOT = project_root
for _ in range(4):
    REPO_ROOT = os.path.dirname(REPO_ROOT)
FIG_DIR = os.path.join(REPO_ROOT, "Paper_Writing", "Separatrix_and_OW_Paper", "figures")


def _read_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line.strip())
    header = rows[0].split(",")
    return [dict(zip(header, r.split(","))) for r in rows[1:]]


def _panel(ax, xs, success, straddle, xlabel, title, xlim):
    ax.axhline(50, color="0.75", linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(xs, success, marker="o", markersize=4.5, color="0.15",
            linewidth=1.6, label="Far-saddle success", zorder=3)
    ax.plot(xs, straddle, marker="s", markersize=4, color="0.5",
            linewidth=1.3, linestyle="--", label="Straddle retention", zorder=2)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Rate (%)", fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)


def main():
    summary = _read_csv(os.path.join(OECS_DIR, "summary_single_target.csv"))
    flip_uv = _read_csv(os.path.join(OECS_DIR, "flip_resolution.csv"))
    flip_p = _read_csv(os.path.join(OECS_DIR, "flip_resolution_sigma_p.csv"))

    anchor_uv = next(r for r in summary
                     if float(r["sigma_uv"]) == 0.001
                     and float(r["sigma_p"]) == 0.0)
    anchor_p = next(r for r in summary
                    if float(r["sigma_uv"]) == 0.0
                    and float(r["sigma_p"]) == 0.005)

    xs_uv = [0.001] + [float(r["sigma_uv"]) for r in flip_uv]
    success_uv = [float(anchor_uv["success_single_target"]) * 100] + \
        [float(r["success_single_target"]) * 100 for r in flip_uv]
    straddle_uv = [float(anchor_uv["success_straddle"]) * 100] + \
        [float(r["success_straddle"]) * 100 for r in flip_uv]

    xs_p = [float(r["sigma_p"]) for r in flip_p] + [0.005]
    success_p = [float(r["success_single_target"]) * 100 for r in flip_p] + \
        [float(anchor_p["success_single_target"]) * 100]
    straddle_p = [float(r["success_straddle"]) * 100 for r in flip_p] + \
        [float(anchor_p["success_straddle"]) * 100]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.0))

    _panel(ax1, xs_uv, success_uv, straddle_uv, r"$\sigma_{uv}$",
          "(a) vs. measurement noise, $\\sigma_p = 0$", (0.0008, 0.0082))
    ax1.legend(loc="upper right", fontsize=7, frameon=False)

    _panel(ax2, xs_p, success_p, straddle_p, r"$\sigma_p$",
          "(b) vs. position noise, $\\sigma_{uv} = 0$", (0.0002, 0.0052))

    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "flip_resolution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
