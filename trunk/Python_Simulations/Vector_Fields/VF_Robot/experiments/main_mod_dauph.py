"""
main_mod_dauph.py

6-robot pentagon cluster executing Case 6 (|lambda| denominator, Dauphin-style)
on the bimodal Gaussian scalar field.

Port of Case 6 from the notebook:
  trunk/Python_Simulations/Separatrix_Control_testing/saddle_point_6_robot2.ipynb

The perpendicular direction (positive Hessian eigenvalue) snaps the formation
to the trench at x = 0. The along-trench direction (negative Hessian
eigenvalue) repels from the saddle and slides the formation down the trench.

This field's trench has no finite minimum: sigma(0, y) = log 2 - (4 + y^2)/2
decreases monotonically as |y| grows. The formation therefore descends along
x = 0 until the simulation ends; v_max bounds the per-step velocity.

Starts are near-saddle nudges with +/- y offsets so the y-sign breaks the
symmetric tie and selects the descent direction.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalScalarField
from src.fields.environments.Scalar_Fields import bimodal_gaussian
from src.control.pentagon_primitives import adaptive_eigen_step_abs
from src.simulation.runner import execute_omni_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

V_MAX = 0.04        # Adaptive eigenstep saturation velocity (m/s)
SIM_STEPS = 300     # Steps per run

# Momentum dynamics in Omnibot apply alpha=0.7 filtering:
#   v_actual = (1 - alpha) * v_cmd = 0.3 * v_cmd
# Stiction threshold is 0.025 m/s, so v_cmd must exceed 0.025/0.3 = 0.083 m/s.
# The adaptive eigenstep saturates at V_MAX (~0.057 m/s combined), which falls
# below the stiction threshold after filtering. CONTROL_GAIN scales the primitive
# output to overcome this. It is a constant multiplicative factor applied to
# (vx_c, vy_c) before the inverse Jacobian step; it does not change the algorithm
# direction, only magnitude. A value of 3.0 gives ~0.17 m/s commanded, well above
# stiction after filtering. Same saturation shape as adaptive_eigen_step, so the
# same tuning carries over.
CONTROL_GAIN = 3.0

# Near-saddle nudges. Each start is inside the saddle basin (|x| < 0.65). The
# small +/- y offset breaks the symmetric tie along the trench so the descent
# direction is unambiguous.
START_POINTS = [
    ( 1,  0, "Start ( 0.3,  0.05) -- right of trench, +y nudge"),
    ( 0.35, -0.15, "Start ( 0.3, -0.05) -- right of trench, -y nudge"),
    ( 0.5,  0.05, "Start (-0.3,  0.05) -- left of trench,  +y nudge"),
    (  0.75, -0.15, "Start (-0.3, -0.05) -- left of trench,  -y nudge"),
]

# ============================================================================
# HELPERS
# ============================================================================

def run_single(ax, start_x, start_y, title):
    """Run one simulation from (start_x, start_y) and plot on ax."""
    field = AnalyticalScalarField(bimodal_gaussian)
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    cluster.reset(start_x, start_y)

    def primitive(c):
        vx, vy = adaptive_eigen_step_abs(c, v_max=V_MAX)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    execute_omni_simulation(cluster, primitive, title,
                            sim_time=SIM_STEPS, ax=ax, skip_legend=True)

    # Mark the saddle. With the |lambda| denominator it is the unstable
    # equilibrium the formation is descending away from, not a target.
    ax.plot(0.0, 0.0, marker='x', color='cyan', markersize=12,
            markeredgewidth=2, zorder=11, label='Saddle (0,0)')

    # Report final centroid distance to the saddle. With this primitive the
    # number should be larger than the starting distance: it measures how far
    # the formation descended along the trench.
    history = cluster.get_center_history()
    if len(history) > 0:
        final = history[-1]
        dist = np.linalg.norm(final)
        ax.set_xlabel(f"Final dist to saddle: {dist:.4f} m", fontsize=9)

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

    plt.suptitle("Pentagon Cluster -- Dauphin Eigenstep (|lambda| denominator)\n"
                 "Bimodal Gaussian field, descending trench at x = 0",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
