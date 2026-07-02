"""
main_logic_g_newton_pentagon.py

6-robot pentagon cluster tracking the det(J)=0 Okubo-Weiss contour of a static
double-gyre vector field using logic_g_newton_contour_pentagon (Newton-step).

Replaces the heuristic gradient/flow blend with a deterministic two-component step:
  Perpendicular: Newton step toward det(J)=0 along the gradient direction,
                 saturated via tanh to v_max.
  Tangential:    Drift along perp(grad_D), the exact level-set tangent,
                 sign-stabilized to the flow direction, saturated via tanh
                 to v_max.  (An earlier Hessian eigen-tangent was removed
                 2026-07-02 after a head-to-head; see the primitive docstring.)

No heuristic branch flipping.  Fixed orientation (no omega).

The det(J)=0 contour traces a diamond around each gyre center at (-0.5, 0) and
(0.5, 0) in the shifted domain.

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

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X
)
from src.control.pentagon_primitives import logic_g_newton_contour_pentagon
from src.simulation.runner import execute_omni_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

V_MAX        = 0.04   # Per-direction saturation speed (m/s)
SIM_STEPS    = 300
CONTROL_GAIN = 3.0    # Stiction compensation

GYRE_LEFT  = (-0.5, 0.0)
GYRE_RIGHT = ( 0.5, 0.0)

# Same 5 starting points as the other two Logic G experiments for comparison.
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
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    cluster.reset(start_x, start_y)

    def primitive(c):
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=V_MAX, eps_grad=1e-6)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

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

    ylim = ax.get_ylim()
    ax.axvline(x=SEPARATRIX_X, color='magenta', linewidth=1.0,
               linestyle='--', alpha=0.4, zorder=4, label='Separatrix')
    ax.set_ylim(ylim)

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
        "Pentagon Cluster -- Newton-Step Logic G, Okubo-Weiss Contour\n"
        "Static Double Gyre (shifted), det(J)=0 diamond around gyre centers",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
