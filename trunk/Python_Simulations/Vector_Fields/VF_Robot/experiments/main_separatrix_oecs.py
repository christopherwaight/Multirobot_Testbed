"""
main_separatrix_oecs.py

6-robot pentagon cluster tracking attracting OECS/TRAP structures of the
static double-gyre field using Primitive 10 (oecs_trap_step), the objective
counterpart of the Logic C separatrix tracker in main_separatrix_v6r.py.

Paper traceability: produces the candidate figure
  Paper_Writing/Separatrix_and_OW_Paper/figures/oecs_trajectories.png
and the per-run metrics CSV
  experiments/outputs/oecs/oecs_clean_runs.csv
(review copies stay in experiments/outputs/oecs/).

The tracker is built from the rate-of-strain tensor S = (J + J^T)/2 only:
s1 (smaller eigenvalue), grad s1, and S's eigenframe, all objective
quantities. Expected behavior on the steady double gyre (verified analytic
geometry, see tests/test_oecs_estimator.py):

  - The s1-trench network is the same grid as det(J)'s, but attracting /
    repelling identity alternates by segment: the UPPER separatrix (top
    saddle to crest) is attracting, the LOWER half's attracting partner is
    the bottom wall, and S is degenerate (s1 = 0) on the lines x = +/-0.5
    and y = 0.
  - CORE-SEEK mode (s_capture finite): starts converge to the nearest TRAP
    core, i.e. the flow saddle terminating their attracting segment
    (upper-half starts to the TOP saddle, lower-half starts to the BOTTOM
    saddle), and park there.
  - RIDE mode (s_capture=None): the team rides its attracting segment by
    tangent continuity and TRIMs (holds) where attraction stops dominating
    (s1 > -s_trim), e.g. approaching the crest degeneracy. It does NOT
    sail through the crest the way Logic C's flow mode does; that is the
    honest objective behavior, since the attracting structure ends there.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X
)
from src.control.pentagon_primitives import oecs_trap_step
from src.simulation.runner import execute_omni_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

V_MAX     = 0.04   # per-direction saturation speed (m/s)
SIM_STEPS = 400    # steps per run (parking tail included)
CONTROL_GAIN = 3.0 # same navigation gain as the Logic C experiments

G_PERP    = 1.0    # gradient gain on the s1-descent channels
S_TRIM    = 0.05   # attraction-dominance threshold (TRIM when s1 > -S_TRIM)
S_CAPTURE = -0.9   # TRAP-core park threshold (cores at s1 = -pi^2 A ~ -0.987)

# TRAP cores of the double gyre = flow saddles on the walls, where
# |cos(pi x_f) cos(pi y_f)| = 1.
TRAP_CORES = [(-1.0, -0.5), (-1.0, 0.5), (0.0, -0.5), (0.0, 0.5),
              (1.0, -0.5), (1.0, 0.5)]

# Starting points (world coords). Expectations refer to CORE-SEEK mode.
START_POINTS = [
    (-0.45,  0.30, "S1 upper left -- expect top saddle"),
    ( 0.05,  0.40, "S2 upper right -- expect top saddle"),
    ( 0.00,  0.00, "S3 crest (degenerate S) -- expect escape then a core"),
    ( 0.10, -0.20, "S4 lower right -- expect bottom saddle"),
    ( 0.25,  0.42, "S5 far upper right -- expect top saddle"),
    (-0.40,  0.10, "S6 near gyre center -- weak strain, basin test"),
]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")


# ============================================================================
# HELPERS
# ============================================================================

def run_single(ax, start_x, start_y, title, s_capture):
    """Run one trial from (start_x, start_y) and plot on ax."""
    field   = AnalyticalField(double_gyre_static)
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    cluster.reset(start_x, start_y)

    def primitive(c):
        vx, vy = oecs_trap_step(c, v_max=V_MAX, g_perp=G_PERP,
                                s_trim=S_TRIM, s_capture=s_capture)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    execute_omni_simulation(cluster, primitive, title,
                            sim_time=SIM_STEPS, ax=ax, skip_legend=True)

    for core in TRAP_CORES:
        if abs(core[0]) <= 0.65:
            ax.plot(core[0], core[1], marker='x', color='cyan', markersize=12,
                    markeredgewidth=2, zorder=11)
    ylim = ax.get_ylim()
    ax.axvline(x=SEPARATRIX_X, color='magenta', linewidth=1.0,
               linestyle='--', alpha=0.6, zorder=4)
    ax.set_ylim(ylim)

    # -- Per-run metrics ----------------------------------------------------
    history = cluster.get_center_history()
    final = history[-1]
    dists = [np.hypot(final[0] - c[0], final[1] - c[1]) for c in TRAP_CORES]
    i_min = int(np.argmin(dists))
    tail = history[-50:]
    tail_std = float(np.mean(np.std(tail, axis=0)))

    diag = cluster.diagnostics
    modes = [d['mode'] for d in diag]
    occupancy = {m: modes.count(m) / max(len(modes), 1)
                 for m in ('ACQUIRE', 'RIDE', 'SEEK', 'PARK', 'TRIM')}
    t_band = next((i for i, d in enumerate(diag) if d['mode'] != 'ACQUIRE'), -1)

    ax.set_xlabel(f"final dist to core {TRAP_CORES[i_min]}: {dists[i_min]:.4f}",
                  fontsize=9)
    return {
        'start_x': start_x, 'start_y': start_y,
        'mode_knob': 'core_seek' if s_capture is not None else 'ride',
        't_band': t_band,
        'final_x': float(final[0]), 'final_y': float(final[1]),
        'nearest_core_x': TRAP_CORES[i_min][0],
        'nearest_core_y': TRAP_CORES[i_min][1],
        'dist_to_core': float(dists[i_min]),
        'tail_pos_std': tail_std,
        **{f'occ_{k.lower()}': round(v, 3) for k, v in occupancy.items()},
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(START_POINTS)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 10))

    rows = []
    for j, (sx, sy, title) in enumerate(START_POINTS):
        print(f"\n[core-seek] {title}")
        rows.append(run_single(axes[0][j], sx, sy, title, S_CAPTURE))
        print(f"[ride]      {title}")
        rows.append(run_single(axes[1][j], sx, sy, title, None))

    axes[0][0].set_ylabel("CORE-SEEK (park at TRAP core)", fontsize=11,
                          fontweight='bold')
    axes[1][0].set_ylabel("RIDE (tangent continuity, trim at end)",
                          fontsize=11, fontweight='bold')

    plt.suptitle(
        "Pentagon Cluster -- Objective OECS/TRAP Tracker (Primitive 10)\n"
        "Static double gyre; s1 trenches of S = (J + J^T)/2; "
        "TRAP cores marked x",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "oecs_trajectories.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure written to {fig_path}")

    csv_path = os.path.join(OUT_DIR, "oecs_clean_runs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics written to {csv_path}\n")

    for r in rows:
        print(f"  ({r['start_x']:+.2f},{r['start_y']:+.2f}) {r['mode_knob']:>9}: "
              f"t_band={r['t_band']:>3}  final=({r['final_x']:+.3f},{r['final_y']:+.3f})  "
              f"core=({r['nearest_core_x']:+.1f},{r['nearest_core_y']:+.1f})  "
              f"dist={r['dist_to_core']:.4f}  tail_std={r['tail_pos_std']:.5f}")


if __name__ == "__main__":
    main()
