"""
main_logic_g_quad.py

4-robot square cluster tracking the det(J)=0 Okubo-Weiss contour of a static
double-gyre vector field using logic_g_zero_flow_quad (heuristic Logic G).

Port of the notebook:
  trunk/Python_Simulations/Separatrix_Control_testing/separatrix_interactive_v002.ipynb

The cluster uses 4 robots arranged in a square.  It estimates the Jacobian
via 4-point affine LS and grad(det(J)) via C(4,3) sub-formations, then applies
Logic G: a zero-seeking gradient direction blended with mean flow.  The square
diagonal is rotated toward the motion direction via omega.

The det(J)=0 contour traces a diamond around each gyre center at (-0.5, 0) and
(0.5, 0) in the shifted domain.  Trajectories should trace this diamond.

Field: static double gyre, shifted so the separatrix is at x=0 and the domain
fits the runner's standard plot bounds.
  Gyre centers: (-0.5, 0) and (0.5, 0)
  Corner saddles: (0, -0.5) and (0, 0.5)
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X
)
from src.control.quad_primitives import logic_g_zero_flow_quad
from src.simulation.runner import execute_omni_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/quad_square.yaml"

STEP_SIZE        = 0.04   # Logic G step size (m/s)
SIM_STEPS        = 300
CONTROL_GAIN     = 3.0    # Stiction compensation (momentum filter: v_actual = 0.3 * v_cmd)
CORRECTION_WEIGHT = 0.0   # Pure gradient direction (v002 default)
ROTATION_RATE    = 0.15   # Diagonal rotation gain (rad/s per rad error)

GYRE_LEFT  = (-0.5, 0.0)
GYRE_RIGHT = ( 0.5, 0.0)

# Starting points chosen to span the diamond contour region around each gyre.
# Format: (start_x, start_y, title)
START_POINTS = [
    (-0.5,  0.25, "Left gyre, above center -- expect diamond tracking"),
    (-0.7,  0.0,  "Left gyre, outside diamond -- expect inward pull"),
    (-0.3,  0.0,  "Left gyre, inside diamond -- expect outward push"),
    ( 0.5,  0.25, "Right gyre, above center -- expect diamond tracking"),
    ( 0.3,  0.0,  "Right gyre, inside diamond -- expect outward push"),
]

# ============================================================================
# HELPERS
# ============================================================================

def run_single(ax, start_x, start_y, title):
    """Run one trial from (start_x, start_y) and plot on ax."""
    field   = AnalyticalField(double_gyre_static)
    cluster = QuadCluster(FORMATION_CONFIG, field)
    cluster.reset(start_x, start_y)

    def primitive(c):
        vx, vy, omega = logic_g_zero_flow_quad(
            c, step_size=STEP_SIZE,
            correction_weight=CORRECTION_WEIGHT,
            rotation_rate=ROTATION_RATE)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN, omega

    execute_omni_simulation(cluster, primitive, title,
                            sim_time=SIM_STEPS, ax=ax, skip_legend=True)

    # Mark gyre centers
    ax.plot(GYRE_LEFT[0], GYRE_LEFT[1], marker='+', color='magenta',
            markersize=14, markeredgewidth=2, zorder=11, label='Gyre center L')
    ax.plot(GYRE_RIGHT[0], GYRE_RIGHT[1], marker='+', color='magenta',
            markersize=14, markeredgewidth=2, zorder=11, label='Gyre center R')

    # Mark saddle locations
    ax.plot(SADDLE_BOTTOM[0], SADDLE_BOTTOM[1], marker='x', color='cyan',
            markersize=12, markeredgewidth=2, zorder=11, label='Saddle bottom')
    ax.plot(SADDLE_TOP[0], SADDLE_TOP[1], marker='x', color='cyan',
            markersize=12, markeredgewidth=2, zorder=11, label='Saddle top')

    # Separatrix reference
    ylim = ax.get_ylim()
    ax.axvline(x=SEPARATRIX_X, color='magenta', linewidth=1.0,
               linestyle='--', alpha=0.4, zorder=4, label='Separatrix')
    ax.set_ylim(ylim)

    # Report final centroid distance to nearest gyre center
    history = cluster.get_center_history()
    if len(history) > 0:
        final = history[-1]
        d_l = np.linalg.norm(final - np.array(GYRE_LEFT))
        d_r = np.linalg.norm(final - np.array(GYRE_RIGHT))
        nearest = min(d_l, d_r)
        ax.set_xlabel(f"Final dist to nearest gyre center: {nearest:.4f} m",
                      fontsize=9)

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
        "4-Robot Square Cluster -- Logic G Heuristic, Okubo-Weiss Contour\n"
        "Static Double Gyre (shifted), det(J)=0 diamond around gyre centers",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
