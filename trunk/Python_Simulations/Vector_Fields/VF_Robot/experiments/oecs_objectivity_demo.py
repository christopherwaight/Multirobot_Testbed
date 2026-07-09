"""
oecs_objectivity_demo.py

Objectivity demonstration: run the det(J)-based separatrix tracker
(Logic C, Primitive 7) and the rate-of-strain OECS/TRAP tracker
(Primitive 10, core-seek mode) on the SAME physical double-gyre flow seen
from two observer frames: the inertial frame and a frame rotating at
omega_rot (src/fields/environments/Rotating_Frame.py).

Prediction (verified open-loop in tests/test_oecs_estimator.py):
  - det(J) is not objective: det(J') = det(J) + omega_rot*vorticity +
    omega_rot^2, and the measured flow gains a solid-body swirl, so
    Logic C in the rotating frame tracks a frame artifact; its pulled-back
    trajectory diverges from its inertial one.
  - S = (J + J^T)/2 is objective: eigenvalues invariant, eigenvectors
    co-rotating, so the OECS tracker's pulled-back trajectory reproduces
    its inertial trajectory (same material structure), up to the tracking
    lag of chasing a structure that moves at omega_rot * r in the rotating
    frame.

Paper traceability: produces the candidate figure
  Paper_Writing/Separatrix_and_OW_Paper/figures/oecs_objectivity.png
and the summary CSV
  experiments/outputs/oecs/oecs_objectivity.csv
(review copies stay in experiments/outputs/oecs/).

Tracking-authority constraint: the structure moves at omega_rot * r in the
rotating frame; with r <= 0.5 (top saddle) and authority GAIN * V_MAX =
0.12, omega_rot = 0.2 gives structure speed 0.10 < 0.12.  The added
solid-body vorticity 2*omega_rot = 0.4 is 20% of the gyre-core vorticity
(2 pi^2 A ~ 1.97) and dominates the near-zero local vorticity at the
separatrix, which is exactly where Logic C's surrogate is corrupted.
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
from src.fields.environments.Double_Gyre import double_gyre_static, SADDLE_TOP
from src.fields.environments.Rotating_Frame import (
    make_rotating_frame, pull_back_trajectory
)
from src.control.pentagon_primitives import (
    separatrix_logic_c_step, oecs_trap_step
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

OMEGA_ROT  = 0.2    # frame rotation rate (rad/s)
START      = (0.05, 0.40)
SIM_STEPS  = 400
DT         = 0.1
V_MAX      = 0.04
CONTROL_GAIN = 3.0

G_PERP    = 1.0
S_TRIM    = 0.05
S_CAPTURE = -0.9    # core-seek: park at the TRAP core (top saddle)

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")


def oecs_primitive(c):
    vx, vy = oecs_trap_step(c, v_max=V_MAX, g_perp=G_PERP,
                            s_trim=S_TRIM, s_capture=S_CAPTURE)
    return vx * CONTROL_GAIN, vy * CONTROL_GAIN


def logic_c_primitive(c):
    vx, vy = separatrix_logic_c_step(c, v_max=V_MAX)
    return vx * CONTROL_GAIN, vy * CONTROL_GAIN


# ============================================================================
# SIMULATION
# ============================================================================

def run(field_fn, primitive, rotating=False):
    """
    Run one trial; return (N, 2) centroid history, stopped at domain exit.

    Domain-exit check is done in INERTIAL coordinates (pulled back for the
    rotating-frame runs), since the physical domain is frame-independent.
    """
    field = AnalyticalField(field_fn)
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    cluster.reset(*START)
    field.reset_clock()
    for k in range(SIM_STEPS):
        cluster.move(primitive)
        field.step(cluster.timestep)
        cx, cy = cluster.get_centroid()
        if rotating:
            th = OMEGA_ROT * (k + 1) * DT
            cx, cy = (np.cos(th) * cx + np.sin(th) * cy,
                      -np.sin(th) * cx + np.cos(th) * cy)
        if abs(cx) > 1.05 or abs(cy) > 0.60:
            break
    return cluster.get_center_history()


def trench_network_distance(traj):
    """
    Distance of each trajectory point to the nearest line of the double
    gyre's trench grid (x in {-1, 0, 1} or y in {-0.5, 0.5}), the
    structure set both controllers claim to track.
    """
    dx = np.min(np.abs(traj[:, 0:1] - np.array([[-1.0, 0.0, 1.0]])), axis=1)
    dy = np.min(np.abs(traj[:, 1:2] - np.array([[-0.5, 0.5]])), axis=1)
    return np.minimum(dx, dy)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rotating_gyre = make_rotating_frame(double_gyre_static, OMEGA_ROT)

    runs = {}
    print("Running 4 trials (2 controllers x 2 frames)...")
    runs['oecs_inertial'] = run(double_gyre_static, oecs_primitive)
    runs['logicc_inertial'] = run(double_gyre_static, logic_c_primitive)
    for key, prim in [('oecs_rotating', oecs_primitive),
                      ('logicc_rotating', logic_c_primitive)]:
        traj = run(rotating_gyre, prim, rotating=True)
        # center_history[k] is the position AFTER move k, whose field samples
        # were taken at t = k*DT; pull back with the post-move time (k+1)*DT.
        times = (np.arange(len(traj)) + 1) * DT
        runs[key] = pull_back_trajectory(traj, times, OMEGA_ROT)

    # -- Metrics -------------------------------------------------------------
    def summarize(name_i, name_r):
        ti, tr = runs[name_i], runs[name_r]
        L = min(len(ti), len(tr))
        gap = float(np.linalg.norm(ti[L - 1] - tr[L - 1]))
        mean_gap = float(np.mean(np.linalg.norm(ti[:L] - tr[:L], axis=1)))
        return gap, mean_gap

    oecs_final_gap, oecs_mean_gap = summarize('oecs_inertial', 'oecs_rotating')
    lc_final_gap, lc_mean_gap = summarize('logicc_inertial', 'logicc_rotating')
    oecs_core_dist_i = float(np.linalg.norm(runs['oecs_inertial'][-1]
                                            - np.array(SADDLE_TOP)))
    oecs_core_dist_r = float(np.linalg.norm(runs['oecs_rotating'][-1]
                                            - np.array(SADDLE_TOP)))
    # Mean distance to the true trench network over the post-acquisition
    # phase (step 100 onward): how far off-structure each run actually is.
    trench = {k: float(np.mean(trench_network_distance(v[100:])))
              for k, v in runs.items()}

    rows = [
        {'controller': 'oecs', 'omega_rot': OMEGA_ROT,
         'final_gap_inertial_vs_pulledback': round(oecs_final_gap, 4),
         'mean_gap_common_steps': round(oecs_mean_gap, 4),
         'trench_dist_inertial': round(trench['oecs_inertial'], 4),
         'trench_dist_rotating': round(trench['oecs_rotating'], 4),
         'final_dist_to_top_saddle_inertial': round(oecs_core_dist_i, 4),
         'final_dist_to_top_saddle_rotating': round(oecs_core_dist_r, 4)},
        {'controller': 'logic_c', 'omega_rot': OMEGA_ROT,
         'final_gap_inertial_vs_pulledback': round(lc_final_gap, 4),
         'mean_gap_common_steps': round(lc_mean_gap, 4),
         'trench_dist_inertial': round(trench['logicc_inertial'], 4),
         'trench_dist_rotating': round(trench['logicc_rotating'], 4),
         'final_dist_to_top_saddle_inertial': '',
         'final_dist_to_top_saddle_rotating': ''},
    ]
    csv_path = os.path.join(OUT_DIR, "oecs_objectivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # -- Figure ----------------------------------------------------------
    nx, ny = 60, 30
    gx = np.linspace(-1.0, 1.0, nx)
    gy = np.linspace(-0.5, 0.5, ny)
    GX, GY = np.meshgrid(gx, gy)
    GU = np.zeros_like(GX)
    GV = np.zeros_like(GY)
    for i in range(ny):
        for j in range(nx):
            GU[i, j], GV[i, j] = double_gyre_static(GX[i, j], GY[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [
        ('OECS tracker (objective, S-based)',
         'oecs_inertial', 'oecs_rotating', 'tab:blue', 'tab:orange',
         f"final gap {oecs_final_gap:.3f}"),
        ('Logic C (det(J)-based, not objective)',
         'logicc_inertial', 'logicc_rotating', 'tab:green', 'tab:red',
         f"final gap {lc_final_gap:.3f}"),
    ]
    for ax, (title, k_i, k_r, c_i, c_r, gap_txt) in zip(axes, panels):
        ax.streamplot(GX, GY, GU, GV, color='0.75', density=1.0,
                      linewidth=0.6, arrowsize=0.7)
        ax.axvline(0.0, color='magenta', linewidth=1.0, linestyle='--',
                   alpha=0.6, label='separatrix')
        ax.plot([0, 0], [-0.5, 0.5], linestyle='none')
        for core in [(0.0, 0.5), (0.0, -0.5)]:
            ax.plot(*core, marker='x', color='k', markersize=10,
                    markeredgewidth=2)
        ax.plot(*START, marker='o', color='lime', markersize=9,
                markeredgecolor='black', zorder=10, label='start')
        ax.plot(runs[k_i][:, 0], runs[k_i][:, 1], color=c_i, linewidth=2.0,
                label='inertial frame')
        ax.plot(runs[k_r][:, 0], runs[k_r][:, 1], color=c_r, linewidth=1.6,
                linestyle='--',
                label=f'rotating frame (pulled back), $\\Omega$={OMEGA_ROT}')
        ax.plot(runs[k_i][-1, 0], runs[k_i][-1, 1], marker='s', color=c_i,
                markersize=8, markeredgecolor='black', zorder=10)
        ax.plot(runs[k_r][-1, 0], runs[k_r][-1, 1], marker='s', color=c_r,
                markersize=8, markeredgecolor='black', zorder=10)
        ax.add_patch(plt.Rectangle((-1, -0.5), 2, 1, fill=False,
                                   edgecolor='0.4', linewidth=0.8))
        ax.set_title(f"{title}\n{gap_txt}", fontsize=12)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.75, 0.75)
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    plt.suptitle(
        "Same physical flow, two observer frames: inertial vs rotating "
        f"($\\Omega$ = {OMEGA_ROT} rad/s, pulled back to inertial coords)",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "oecs_objectivity.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    print(f"\nFigure written to {fig_path}")
    print(f"Summary written to {csv_path}\n")
    print(f"OECS:    final gap {oecs_final_gap:.4f}, mean gap {oecs_mean_gap:.4f}, "
          f"dist to top saddle: inertial {oecs_core_dist_i:.4f}, "
          f"rotating {oecs_core_dist_r:.4f}")
    print(f"Logic C: final gap {lc_final_gap:.4f}, mean gap {lc_mean_gap:.4f}")
    print("Mean dist to true trench network (steps 100+):")
    for k in ('oecs_inertial', 'oecs_rotating',
              'logicc_inertial', 'logicc_rotating'):
        print(f"  {k:>16}: {trench[k]:.4f}  ({len(runs[k])} steps)")
    print(f"Logic C finals: inertial ({runs['logicc_inertial'][-1][0]:+.3f},"
          f"{runs['logicc_inertial'][-1][1]:+.3f}), pulled-back rotating "
          f"({runs['logicc_rotating'][-1][0]:+.3f},"
          f"{runs['logicc_rotating'][-1][1]:+.3f})")


if __name__ == "__main__":
    main()
