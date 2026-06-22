"""
main_separatrix_v6r.py

6-robot pentagon cluster navigating the separatrix of a static double-gyre
vector field using Primitive 7 (separatrix_logic_c_step, Logic C).

Port of the notebook:
  trunk/Python Simulations/separatrix_interactive_v6r.ipynb

The cluster fits two independent 6-parameter quadratics (one for each of the
u and v field components) to estimate the local Jacobian J, then navigates
the det(J) landscape.  det(J) = 0 on the separatrix.  Logic C selects between:
  Logic A (signed lambda): attracts the cluster toward the trench saddle.
  Logic B (|lambda|):      slides the cluster along the trench.
The selection is made by projecting the local flow vector onto the along-trench
eigenvector of the det(J) Hessian.  A FLOW-band override kicks in whenever
the formation straddles the separatrix (det(J) near zero).

Field: static double gyre, canonical Shadden domain [0,2]x[0,1] shifted to
[-1,1]x[-0.5,0.5] so the separatrix (formerly at x=1) falls on x=0 and the
field fits the runner's standard vector-field plot bounds.
  Saddle at bottom wall: (0.0, -0.5)
  Saddle at top wall:    (0.0,  0.5)
  Separatrix:            x = 0
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X
)
from src.control.pentagon_primitives import separatrix_logic_c_step
from src.simulation.runner import execute_omni_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

V_MAX    = 0.04   # Logic C saturation speed (m/s)
SIM_STEPS = 100   # Steps per run

# Same stiction-filter gain as the saddle experiments.  The momentum filter
# (alpha=0.7 in Omnibot) means v_actual = 0.3 * v_cmd.  V_MAX * CONTROL_GAIN
# gives ~0.12 m/s commanded, well above the 0.025 m/s stiction threshold.
CONTROL_GAIN = 3.0

# Logic C FLOW-band thresholds (notebook defaults).
EPS_RAW = 1e-3    # raw det(J) threshold
EPS_DIM = 0.025   # dimensionless det(J) / ||H_det||_F threshold

# Starting points in shifted world coords (separatrix at x=0, gyres in [-1,1]).
# Format: (start_x, start_y, title)
START_POINTS = [
    (-0.45, 0.3, "Left sep, lower gyre -- expect Logic A toward (0,-0.5)"),
    ( 0.05, 0.4, "Right sep, lower gyre -- expect Logic B slide to (0,-0.5)"),
    (0.0,  0.0, "Left sep, upper gyre -- expect Logic A toward (0, 0.5)"),
    ( 0.1,  -0.2, "On separatrix center -- FLOW mode active immediately"),
    ( 0.25,  0.42, "Far right, upper gyre -- stress test from far start"),
]

# ============================================================================
# HELPERS
# ============================================================================

def run_single(ax, start_x, start_y, title):
    """Run one trial from (start_x, start_y) and plot on ax."""
    field   = AnalyticalField(double_gyre_static)
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    cluster.reset(start_x, start_y)

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX,
                                          eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    execute_omni_simulation(cluster, primitive, title,
                            sim_time=SIM_STEPS, ax=ax, skip_legend=True)

    # Mark saddle locations
    ax.plot(SADDLE_BOTTOM[0], SADDLE_BOTTOM[1],
            marker='x', color='cyan', markersize=12,
            markeredgewidth=2, zorder=11, label='Saddle bottom')
    ax.plot(SADDLE_TOP[0], SADDLE_TOP[1],
            marker='x', color='cyan', markersize=12,
            markeredgewidth=2, zorder=11, label='Saddle top')

    # Mark separatrix line
    ylim = ax.get_ylim()
    ax.axvline(x=SEPARATRIX_X, color='magenta', linewidth=1.0,
               linestyle='--', alpha=0.6, zorder=4, label='Separatrix')
    ax.set_ylim(ylim)

    # Report distance to nearest saddle at final step
    history = cluster.get_center_history()
    if len(history) > 0:
        final = history[-1]
        d_bot = np.linalg.norm(final - np.array(SADDLE_BOTTOM))
        d_top = np.linalg.norm(final - np.array(SADDLE_TOP))
        nearest = min(d_bot, d_top)
        ax.set_xlabel(f"Final dist to nearest saddle: {nearest:.4f} m", fontsize=9)

    return cluster


# ============================================================================
# MAIN
# ============================================================================

def main():
    n = len(START_POINTS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (sx, sy, title) in zip(axes, START_POINTS):
        print(f"\nRunning: {title}")
        run_single(ax, sx, sy, title)

    plt.suptitle(
        "Pentagon Cluster -- Logic C Separatrix Navigation\n"
        "Static Double Gyre (shifted), separatrix at x=0, saddles at (0, +/-0.5)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
