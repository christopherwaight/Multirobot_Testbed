"""
main_ocean_hfr_v6r.py

6-robot pentagon cluster navigating an ocean HF radar surface current field
(Santa Barbara Channel, 2012-05-16 08:00 UTC) using Primitive 7
(separatrix_logic_c_step, Logic C).

The cluster fits two independent 6-parameter quadratics (one per field
component) to estimate the local Jacobian J. Logic C then navigates the
det(J) landscape using the Hessian of det(J) to locate and track
hyperbolic regions (det(J) < 0, near-separatrix flow).

Figure layout (1 x 2):
  Left:  quiver + 6-robot trajectories rendered in lat/lon coordinates,
         covering the same geographic extent as the FTLE reference.
  Right: precomputed 24-h forward-time FTLE reference image.
         Generated once by: python experiments/_generate_ocean_ftle_reference.py

START_POINTS entries are (lat, lon, label). The label is a descriptive string
used for the plot title. Start can be anywhere within the FTLE extent
(33.8..34.7 N, -120.7..-119.7 W).

The cluster simulates in native world coordinates (m/s velocities, meter-scale
formation, 0.1 s timestep). This experiment converts lat/lon start points to
world coords for cluster.reset() and converts the resulting histories back to
lat/lon for plotting. The shared runner (execute_omni_simulation) is not used
here so its hard-coded [-0.65, 0.65] plot bounds do not constrain this view.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/main_ocean_hfr_v6r.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal
from src.control.pentagon_primitives import separatrix_logic_c_step

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

V_MAX        = 0.04    # Logic C saturation speed (m/s)
SIM_STEPS    = 110     # steps per run
CONTROL_GAIN = 3.0     # compensates momentum filter (alpha=0.7 -> v_actual = 0.3*v_cmd)
EPS_RAW      = 1e-3    # raw det(J) threshold for FLOW-band
EPS_DIM      = 0.025   # dimensionless det(J)/||H_det||_F threshold

# Geographic extent matching ftle_frame0_matched.png.
LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7
QUIVER_STEP_DEG  = 0.06    # ~20 arrows per axis across the extent

# Start points: (lat, lon, label).
# Any lat/lon inside the extent above is valid.
START_POINTS = [
    (34.49, -120.33, "furthest point"),
]

ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# ============================================================================
# COORDINATE HELPERS
# ============================================================================

# Was an inline copy of the world <-> lat/lon map, which silently disagreed
# with the field evaluator once the map became isotropic (2026-08-03).
# Delegates to the shared helper now.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world   # noqa: E402
from _coords_common import world_to_latlon as _world_to_latlon   # noqa: E402


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _draw_quiver(ax, field):
    """Render the field as quiver arrows on a lat/lon grid."""
    lons = np.arange(LON_MIN, LON_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    lats = np.arange(LAT_MIN, LAT_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    LON, LAT = np.meshgrid(lons, lats)
    U = np.zeros_like(LON)
    V = np.zeros_like(LAT)
    for r in range(LON.shape[0]):
        for c in range(LON.shape[1]):
            wx, wy = _latlon_to_world(LAT[r, c], LON[r, c], field.config)
            u, v = field.get_value(wx, wy)
            U[r, c] = u
            V[r, c] = v
    ax.quiver(LON, LAT, U, V, color='black', alpha=0.5)


def _draw_ftle_reference(ax):
    """Display the precomputed FTLE PNG on ax, no robot paths."""
    here      = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    png_path  = os.path.join(
        repo_root,
        "trunk", "Python_Simulations", "Vector_Fields",
        "ocean_data", "det_jacobian_plots", "ftle_frame0_matched.png"
    )
    if not os.path.isfile(png_path):
        ax.text(
            0.5, 0.5,
            "FTLE reference not found.\nRun:\n"
            "  python experiments/_generate_ocean_ftle_reference.py",
            ha="center", va="center", transform=ax.transAxes, fontsize=10
        )
        ax.set_axis_off()
        return
    img = plt.imread(png_path)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("Forward-time FTLE (24 h) reference",
                 fontsize=12, fontweight="bold")


# ============================================================================
# TRIAL
# ============================================================================

def _run_trial(ax, start_lat, start_lon, label, field, cluster):
    """Run one simulation trial and draw results on ax in lat/lon."""
    sx, sy = _latlon_to_world(start_lat, start_lon, field.config)
    cluster.reset(sx, sy)

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX,
                                          eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    if hasattr(field, "reset_clock"):
        field.reset_clock()

    for _ in range(SIM_STEPS):
        cluster.move(primitive)
        if hasattr(field, "step"):
            field.step(cluster.timestep)

    # ── Draw quiver background ────────────────────────────────────────────
    _draw_quiver(ax, field)

    # ── Draw robot trajectories ───────────────────────────────────────────
    robot_history = cluster.get_robot_history()   # shape (steps, 6, 2) in world
    if len(robot_history) > 0:
        for i in range(robot_history.shape[1]):
            traj_world = robot_history[:, i, :]
            traj_lats = []
            traj_lons = []
            for pt in traj_world:
                rlat, rlon = _world_to_latlon(pt[0], pt[1], field.config)
                traj_lats.append(rlat)
                traj_lons.append(rlon)
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.plot(traj_lons, traj_lats,
                    color=color, linewidth=1.5, alpha=0.6, linestyle='--')
            ax.plot(traj_lons[0], traj_lats[0],
                    marker='o', color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=1.2)
            ax.plot(traj_lons[-1], traj_lats[-1],
                    marker='s', color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=1.2)

    # ── Draw centroid path ────────────────────────────────────────────────
    center_history = cluster.get_center_history()  # shape (steps, 2) in world
    if len(center_history) > 0:
        clats, clons = [], []
        for pt in center_history:
            clat, clon = _world_to_latlon(pt[0], pt[1], field.config)
            clats.append(clat)
            clons.append(clon)

        ax.plot(clons, clats, color='black', linewidth=2, alpha=0.85,
                label='Centroid path')
        ax.plot(clons[0], clats[0],
                marker='*', color='lime', markersize=14,
                markeredgecolor='black', markeredgewidth=1.5,
                label='Start', zorder=10)
        ax.plot(clons[-1], clats[-1],
                marker='X', color='red', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label='End', zorder=10)

        end_lat, end_lon = clats[-1], clons[-1]
        ax.set_xlabel(
            f"Start: ({start_lat:.4f}N, {start_lon:.4f}W)  "
            f"End: ({end_lat:.4f}N, {end_lon:.4f}W)",
            fontsize=9
        )

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title(f"Pentagon Cluster -- {label}", fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    field   = AnalyticalField(ocean_hfr_socal)
    cluster = PentagonCluster(FORMATION_CONFIG, field)

    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_ref  = fig.add_subplot(gs[0, 1])

    start_lat, start_lon, label = START_POINTS[0]
    print(f"\nRunning: {label}  from ({start_lat:.4f}N, {start_lon:.4f}W)")
    _run_trial(ax_traj, start_lat, start_lon, label, field, cluster)
    _draw_ftle_reference(ax_ref)

    plt.suptitle(
        "Pentagon Cluster -- Logic C Navigation\n"
        "Ocean HFR field, first frame (2012-05-16 08:00 UTC), Santa Barbara Channel",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
