"""
main_mod_scal_newt_step.py

6-robot pentagon cluster navigating to the saddle point of a bimodal Gaussian
scalar field using Primitive 5 (Adaptive Eigenstep).

Port of the notebook:
  trunk/Python Simulations/Separatrix_Control_testing/saddle_point_6_robot2.ipynb

The cluster uses a full quadratic (6-parameter) fit of the scalar field to
recover the gradient and Hessian, then applies the adaptive eigenstep control
law to converge to the saddle at (0, 0).

Convergence zone: starts with |x| < ~0.65 land in the saddle basin.
Starts with |x| > ~0.65 fall into the peak at (+/-2, 0) instead.
Multiple starting points are tested to show the convergence zone boundary.
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
from src.control.pentagon_primitives import adaptive_eigen_step
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
# stiction after filtering. Tuning note: increasing this may cause overshoot near
# the saddle -- the algorithm self-limits via the tanh saturation.
CONTROL_GAIN = 3.0

# Starting points to test convergence zone behavior.
# Expected: (0.6, -2.5), (0.0, -2.5), (-0.4, -2.5), (0.3, -2.5) -> converge
# Expected: (1.5, -2.5) -> diverge into peak at (2, 0) -- known behavior
START_POINTS = [
    (0.6,  -2.5, "Start (0.6, -2.5) -- notebook default"),
    (0.0,  -2.5, "Start (0.0, -2.5) -- on-axis"),
    (-0.4, -2.5, "Start (-0.4, -2.5) -- left of axis"),
    (0.3,  -2.5, "Start (0.3, -2.5) -- near axis"),
    (1.5,  -2.5, "Start (1.5, -2.5) -- outside basin, expect divergence"),
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
        vx, vy = adaptive_eigen_step(c, v_max=V_MAX)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    execute_omni_simulation(cluster, primitive, title,
                            sim_time=SIM_STEPS, ax=ax, skip_legend=True)

    # Mark true saddle
    ax.plot(0.0, 0.0, marker='x', color='cyan', markersize=12,
            markeredgewidth=2, zorder=11, label='True saddle (0,0)')

    # Report final centroid distance to saddle
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

    plt.suptitle("Pentagon Cluster -- Adaptive Eigenstep Saddle Navigation\n"
                 "Bimodal Gaussian field, saddle at (0,0)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
