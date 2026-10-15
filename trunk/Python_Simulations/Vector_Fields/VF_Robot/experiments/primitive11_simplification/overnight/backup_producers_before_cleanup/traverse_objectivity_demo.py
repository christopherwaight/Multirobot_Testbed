"""
traverse_objectivity_demo.py

Objectivity demonstration for the separatrix TRAVERSER: run the det(J)-based
separatrix tracker (Logic C, Primitive 7) and the new objective separatrix
traverser (Primitive 11, oecs_separatrix_step) on the SAME physical
double-gyre flow seen from two observer frames: the inertial frame and a
frame rotating at omega_rot (src/fields/environments/Rotating_Frame.py).

This is the traverser's counterpart to oecs_objectivity_demo.py, which runs
Primitive 10 (the TRAP-core-seeker) instead; per the paper plan the
traverser replaces the core-seeker as Controller 2, so this script
regenerates fig:oecs_objectivity with the traverser's own path.

What this verifies: Primitive 11 selects TRACK's tangent as whichever of
e1, e2 has the LARGER |grad s1 . e_i|, using no flow at all; continuity with
the previous tangent fixes only the residual sign. The ambient flow enters
the primitive in exactly one place, the CROSS fallback, reachable only on
the first tracking step if the eigenframe is degenerate there with no
tangent yet. So the closed loop is frame-objective past that single instant,
which is what Proposition (frame equivariance) claims and what the gaps
measured here confirm.

An earlier design did select TRACK's tangent by best alignment with the
ambient flow at every step, matching Logic C's own flow-projection test.
That design was rejected: in a rotating frame the added solid-body swirl
term (v' = Q v + omega_rot * perp(x')) does not vanish at a
trench-network crossing the way the true flow does, so it biased branch
selection a full trench-length from the crossing at Omega = 0.2. Do not
reintroduce it. See the objectivity note on Primitive 11 in
src/control/pentagon_primitives.py, which names this script as its
closed-loop verification.

Paper traceability: writes both outputs into experiments/outputs/oecs/,
  traverse_objectivity.png  and  traverse_objectivity.csv

The paper copy is installed by hand as
  Paper_Writing/Separatrix_and_OW_Paper/figures/objectivity_traverser.png
which is fig:oecs_objectivity in Draft_5d.tex. That copy is NOT automatic.
Until 2026-07-30 the paper instead embedded figures/oecs_objectivity.png,
the Primitive 10 core-seeker output, whose path parks at the upper saddle
and reports final gap 0.018, while the prose and caption quoted this
script's 0.025 / 0.014 / 0.000 / 0.005. Check the md5 of the installed
figure against outputs/oecs/traverse_objectivity.png after regenerating.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/traverse_objectivity_demo.py
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
from matplotlib.lines import Line2D

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static, SADDLE_TOP
from src.fields.environments.Rotating_Frame import (
    make_rotating_frame, pull_back_trajectory
)
from src.control.pentagon_primitives import (
    separatrix_logic_c_step, oecs_separatrix_step
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

OMEGA_ROT  = 0.23   # frame rotation rate; Logic C first loses the straddle
                    # at ~0.21 (0.205 keeps it), so 0.23 shows the break while
                    # Omega*|p| stays under k*c_max = 0.12 on the whole ride
START      = (0.05, 0.40)
SIM_STEPS  = 400
DT         = 0.1
V_MAX      = 0.04
CONTROL_GAIN = 3.0

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")


def traverser_primitive(c):
    vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP, s_trim=S_TRIM,
                                  r_band=R_BAND, g_capture=G_CAPTURE,
                                  s_capture=None)
    return vx * CONTROL_GAIN, vy * CONTROL_GAIN


def logic_c_primitive(c):
    vx, vy = separatrix_logic_c_step(c, v_max=V_MAX)
    return vx * CONTROL_GAIN, vy * CONTROL_GAIN


# ============================================================================
# SIMULATION
# ============================================================================

def run(field_fn, primitive, rotating=False, return_robots=False):
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
    if return_robots:
        return cluster.get_center_history(), cluster.get_robot_history()
    return cluster.get_center_history()


def straddle_loss(C, R, saddle_tol=0.06):
    """Steps (before far-saddle contact) where all six robots sit on one side
    of x = 0 after the first straddle, the retention measure of _mc_common.py.
    Returns (lost step indices, one-sided depth at each)."""
    hit = np.where(np.linalg.norm(C - np.array([0.0, -0.5]), axis=1)
                   < saddle_tol)[0]
    end = int(hit[0]) if len(hit) else len(C)
    rx = R[:end, :, 0]
    s = (rx.min(axis=1) < 0.0) & (rx.max(axis=1) > 0.0)
    if not s.any():
        return np.array([], dtype=int), np.array([])
    first = int(np.argmax(s))
    lost = np.where(~s[first:])[0] + first
    depth = np.minimum(np.abs(rx.min(axis=1)), np.abs(rx.max(axis=1)))[lost]
    return lost, depth


def trench_network_distance(traj):
    dx = np.min(np.abs(traj[:, 0:1] - np.array([[-1.0, 0.0, 1.0]])), axis=1)
    dy = np.min(np.abs(traj[:, 1:2] - np.array([[-0.5, 0.5]])), axis=1)
    return np.minimum(dx, dy)


def to_far_saddle(traj):
    """Trajectory up to its closest approach to the far saddle (0, -0.5).
    The figure stops there: the objectivity measure is taken on the ride,
    and wall-trench motion past the saddle is shown in Fig. 4 instead."""
    d = np.linalg.norm(traj - np.array([0.0, -0.5]), axis=1)
    return traj[:int(np.argmin(d)) + 1]


def ride_deviation(traj, acquire_tol=0.01, saddle_tol=0.06):
    """Max |x| between first contact with the separatrix and arrival at
    the far saddle (0, -0.5). The final gap is dominated by wall-trench
    motion after that saddle, so this is the paper's objectivity measure."""
    hit = np.where(np.linalg.norm(traj - np.array([0.0, -0.5]), axis=1)
                   < saddle_tol)[0]
    end = int(hit[0]) if len(hit) else len(traj)
    acq = np.where(np.abs(traj[:end, 0]) < acquire_tol)[0]
    start = int(acq[0]) if len(acq) else 0
    return float(np.max(np.abs(traj[start:end, 0])))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rotating_gyre = make_rotating_frame(double_gyre_static, OMEGA_ROT)

    runs = {}
    print("Running 4 trials (2 controllers x 2 frames)...")
    runs['trav_inertial'] = run(double_gyre_static, traverser_primitive)
    runs['logicc_inertial'] = run(double_gyre_static, logic_c_primitive)
    robots = {}
    for key, prim in [('trav_rotating', traverser_primitive),
                      ('logicc_rotating', logic_c_primitive)]:
        traj, rob = run(rotating_gyre, prim, rotating=True,
                        return_robots=True)
        times = (np.arange(len(traj)) + 1) * DT
        runs[key] = pull_back_trajectory(traj, times, OMEGA_ROT)
        robots[key] = np.stack(
            [pull_back_trajectory(rob[:, i, :], times, OMEGA_ROT)
             for i in range(rob.shape[1])], axis=1)
    loss = {k: straddle_loss(runs[k], robots[k]) for k in robots}

    def summarize(name_i, name_r):
        ti, tr = runs[name_i], runs[name_r]
        L = min(len(ti), len(tr))
        gap = float(np.linalg.norm(ti[L - 1] - tr[L - 1]))
        mean_gap = float(np.mean(np.linalg.norm(ti[:L] - tr[:L], axis=1)))
        return gap, mean_gap

    trav_final_gap, trav_mean_gap = summarize('trav_inertial', 'trav_rotating')
    lc_final_gap, lc_mean_gap = summarize('logicc_inertial', 'logicc_rotating')
    trav_core_dist_i = float(np.linalg.norm(runs['trav_inertial'][-1]
                                            - np.array(SADDLE_TOP)))
    trav_core_dist_r = float(np.linalg.norm(runs['trav_rotating'][-1]
                                            - np.array(SADDLE_TOP)))
    trench = {k: float(np.mean(trench_network_distance(v[100:])))
              for k, v in runs.items()}
    ride_dev = {k: ride_deviation(v) for k, v in runs.items()}

    rows = [
        {'controller': 'traverser', 'omega_rot': OMEGA_ROT,
         'final_gap_inertial_vs_pulledback': round(trav_final_gap, 4),
         'mean_gap_common_steps': round(trav_mean_gap, 4),
         'trench_dist_inertial': round(trench['trav_inertial'], 4),
         'trench_dist_rotating': round(trench['trav_rotating'], 4),
         'final_dist_to_top_saddle_inertial': round(trav_core_dist_i, 4),
         'final_dist_to_top_saddle_rotating': round(trav_core_dist_r, 4),
         'ride_max_abs_x_inertial': round(ride_dev['trav_inertial'], 4),
         'ride_max_abs_x_rotating': round(ride_dev['trav_rotating'], 4)},
        {'controller': 'logic_c', 'omega_rot': OMEGA_ROT,
         'final_gap_inertial_vs_pulledback': round(lc_final_gap, 4),
         'mean_gap_common_steps': round(lc_mean_gap, 4),
         'trench_dist_inertial': round(trench['logicc_inertial'], 4),
         'trench_dist_rotating': round(trench['logicc_rotating'], 4),
         'final_dist_to_top_saddle_inertial': '',
         'final_dist_to_top_saddle_rotating': '',
         'ride_max_abs_x_inertial': round(ride_dev['logicc_inertial'], 4),
         'ride_max_abs_x_rotating': round(ride_dev['logicc_rotating'], 4)},
    ]
    csv_path = os.path.join(OUT_DIR, "traverse_objectivity.csv")
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

    # Drawn at the size it is printed (one IEEE column, ~3.45 in) so nothing is
    # scaled down in the PDF.  Panel titles and a suptitle are deliberately
    # absent: the paper's caption carries the controller names, and duplicating them here wasted the vertical space that made the
    # trajectories unreadable.  Panels are labelled (a)/(b) for the caption.
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'legend.fontsize': 6})
    # Height is chosen so the equal-aspect axes fill the canvas: each panel is
    # ~1.41 in wide and the domain is 2.3 x 1.28, so the axes are ~0.78 in tall,
    # leaving the rest for the x label and the single-row legend.  Getting this
    # wrong leaves a band of dead space that bbox_inches='tight' cannot reclaim,
    # since the gap is interior to the figure.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.34))
    panels = [
        ('(a)', 'trav_inertial', 'trav_rotating', 'tab:blue', 'tab:orange'),
        ('(b)', 'logicc_inertial', 'logicc_rotating', 'tab:green', 'tab:red'),
    ]
    for ax, (tag, k_i, k_r, c_i, c_r) in zip(axes, panels):
        ax.streamplot(GX, GY, GU, GV, color='0.55', density=0.7,
                      linewidth=0.4, arrowsize=0.4)
        ax.axvline(0.0, color='magenta', linewidth=0.7, linestyle='--',
                   alpha=0.6, label='separatrix')
        for core in [(0.0, 0.5), (0.0, -0.5)]:
            ax.plot(*core, marker='x', color='k', markersize=4,
                    markeredgewidth=1.0)
        ax.plot(*START, marker='o', color='lime', markersize=3.5,
                markeredgecolor='black', markeredgewidth=0.5, zorder=10,
                label='start')
        tr_i, tr_r = to_far_saddle(runs[k_i]), to_far_saddle(runs[k_r])
        ax.plot(tr_i[:, 0], tr_i[:, 1], color=c_i, linewidth=1.1,
                label='inertial frame')
        ax.plot(tr_r[:, 0], tr_r[:, 1], color=c_r, linewidth=0.9,
                linestyle='--',
                label='rotating frame (pulled back)')
        ax.add_patch(plt.Rectangle((-1, -0.5), 2, 1, fill=False,
                                   edgecolor='0.4', linewidth=0.5))
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, va='top', ha='left',
                fontsize=7, fontweight='bold')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.68, 0.60)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$', labelpad=1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-0.5, 0, 0.5])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.tick_params(length=2, pad=1)
    axes[0].set_ylabel('$y$', labelpad=1)

    # Inset on (b): the Logic C formation at its deepest straddle loss, all
    # six robots on one side of the separatrix.  The one-sided depth is ~0.01
    # against a 0.075 ring, invisible at full-domain scale.
    lost, depth = loss['logicc_rotating']
    if len(lost):
        k_star = int(lost[np.argmax(depth)])
        P = robots['logicc_rotating'][k_star]
        c = runs['logicc_rotating'][k_star]
        axin = axes[1].inset_axes([0.60, 0.50, 0.38, 0.47])
        tr_r = to_far_saddle(runs['logicc_rotating'])
        axin.axvline(0.0, color='magenta', linewidth=0.7, linestyle='--',
                     alpha=0.6)
        axin.plot(tr_r[:, 0], tr_r[:, 1], color='tab:red', linewidth=0.8,
                  linestyle='--')
        ring = P[1:][np.argsort(np.arctan2(P[1:, 1] - c[1], P[1:, 0] - c[0]))]
        ring = np.vstack([ring, ring[:1]])
        axin.plot(ring[:, 0], ring[:, 1], color='0.3', linewidth=0.5)
        axin.plot(P[:, 0], P[:, 1], 'o', color='tab:red', markersize=1.8,
                  markeredgewidth=0)
        half = 0.10
        axin.set_xlim(c[0] - half, c[0] + half)
        axin.set_ylim(c[1] - half, c[1] + half)
        axin.set_aspect('equal')
        axin.set_xticks([])
        axin.set_yticks([])
        for sp in axin.spines.values():
            sp.set_linewidth(0.5)
        axes[1].indicate_inset_zoom(axin, edgecolor='0.3', linewidth=0.5)

    # One shared legend below both panels.  Per-axes legends sat on top of the
    # Logic C inertial trajectory where it runs along y = -0.5, and the start
    # marker clipped the 'inertial frame' label in both panels.  The frame
    # entries are drawn in neutral grey on purpose: solid-vs-dashed is the
    # encoding that carries meaning, while the colors only separate the two
    # controllers, which differ between panels (blue/orange in (a), green/red
    # in (b)).  A colored legend key would be wrong for one panel or the other.
    proxies = [
        Line2D([], [], color='magenta', linestyle='--', linewidth=0.7,
               alpha=0.6, label='separatrix'),
        Line2D([], [], color='lime', marker='o', linestyle='none',
               markersize=3.5, markeredgecolor='black', markeredgewidth=0.5,
               label='start'),
        Line2D([], [], color='0.35', linewidth=1.1, label='inertial frame'),
        Line2D([], [], color='0.35', linewidth=0.9, linestyle='--',
               label='rotating frame (pulled back)'),
    ]
    fig.legend(handles=proxies, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, columnspacing=0.8, handletextpad=0.35,
               labelspacing=0.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.995, bottom=0.20,
                        wspace=0.06)
    fig_path = os.path.join(OUT_DIR, "traverse_objectivity.png")
    # No bbox_inches='tight': the layout above is already sized to the printed
    # width, and letting savefig recrop would rescale the 7 pt type.
    plt.savefig(fig_path, dpi=400, bbox_inches='tight', pad_inches=0.01)

    print(f"\nFigure written to {fig_path}")
    print(f"Summary written to {csv_path}\n")
    print(f"Traverser: final gap {trav_final_gap:.4f}, mean gap {trav_mean_gap:.4f}, "
         f"dist to top saddle: inertial {trav_core_dist_i:.4f}, "
         f"rotating {trav_core_dist_r:.4f}")
    print(f"Logic C:   final gap {lc_final_gap:.4f}, mean gap {lc_mean_gap:.4f}")
    print("Mean dist to true trench network (steps 100+):")
    for k in ('trav_inertial', 'trav_rotating',
              'logicc_inertial', 'logicc_rotating'):
        print(f"  {k:>16}: {trench[k]:.4f}  ({len(runs[k])} steps)")
    print("Max |x| from acquisition to far saddle (paper measure):")
    for k in ('trav_inertial', 'trav_rotating',
              'logicc_inertial', 'logicc_rotating'):
        print(f"  {k:>16}: {ride_dev[k]:.4f}")
    print("Straddle loss before far-saddle contact (rotating frame):")
    for k, (lost, depth) in loss.items():
        if len(lost):
            print(f"  {k:>16}: {len(lost)} steps, max depth {depth.max():.4f}")
        else:
            print(f"  {k:>16}: none")


if __name__ == "__main__":
    main()
