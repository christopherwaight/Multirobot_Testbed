"""
plot_flip_resolution.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  figures/flip_resolution.png, the figure for the tangent-sign-flip finding
          (Section results_oecs).
  Reads:  experiments/outputs/mc_oecs_traverse/summary_single_target.csv (for the
          sigma_uv = 0.001 point, run at the full 2D grid) and
          experiments/outputs/mc_oecs_traverse/flip_resolution.csv (the fine-resolution
          sweep, sigma_uv = 0.0015 through 0.008, sigma_p = 0 only).

Plots far-saddle success and straddle retention (both single-far-saddle-target
scored, straddle conditioned on that same trial's own far-saddle contact -- see
rescore_single_target.py and mc_sweep_flip_resolution.py docstrings for why this
conditioning matters) against sigma_uv on a linear axis from 0.001 to 0.008.

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


def main():
    summary = _read_csv(os.path.join(OECS_DIR, "summary_single_target.csv"))
    flip = _read_csv(os.path.join(OECS_DIR, "flip_resolution.csv"))

    point_001 = next(r for r in summary
                     if float(r["sigma_uv"]) == 0.001
                     and float(r["sigma_p"]) == 0.0)

    xs = [0.001] + [float(r["sigma_uv"]) for r in flip]
    success = [float(point_001["success_single_target"]) * 100] + \
        [float(r["success_single_target"]) * 100 for r in flip]
    straddle = [float(point_001["success_straddle"]) * 100] + \
        [float(r["success_straddle"]) * 100 for r in flip]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(50, color="0.75", linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(xs, success, marker="o", markersize=4.5, color="0.15",
            linewidth=1.6, label="Far-saddle success", zorder=3)
    ax.plot(xs, straddle, marker="s", markersize=4, color="0.5",
            linewidth=1.3, linestyle="--", label="Straddle retention", zorder=2)

    ax.set_xlabel(r"$\sigma_{uv}$ (m/s)", fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=10)
    ax.set_xlim(0.0008, 0.0082)
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title(
        "Tangent-sign flip: far-saddle success and straddle retention\n"
        r"vs. $\sigma_{uv}$, $\sigma_p = 0$, 10,000 trials/cell",
        fontsize=10)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "flip_resolution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
