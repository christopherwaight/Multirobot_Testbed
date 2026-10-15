"""
main_ocean_hfr_2km_ftle_overlay.py

Single-panel figure: 6-robot pentagon cluster trajectories drawn directly on
top of the 24-hour forward-time FTLE field (2km native resolution, Santa
Barbara Channel), instead of side-by-side with a separate static reference
image (as in main_ocean_hfr_2km.py).

Runs the same trial as main_ocean_hfr_2km.py (same field config, formation,
control primitive, TIME_WARP, SIM_STEPS, START_POINTS) so the two outputs are
directly comparable, then computes the FTLE background via
_ftle_common.compute_ftle_field (same function used by
_generate_ocean_ftle_2km_matched.py) and overlays the robot/centroid paths.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/main_ocean_hfr_2km_ftle_overlay.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from datetime import datetime, timezone

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

# ============================================================================
# CONFIGURATION
# ============================================================================

# Stale pre-tuning config (predates the 2026-07-01 separatrix-tracking pass
# in main_ocean_hfr_2km.py -- this file was not updated at the same time,
# which caused a confusing "veers left / wrong branch" result when run
# after that tuning). Kept for reference, not deleted.
# FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
# V_MAX        = 0.06
# SIM_STEPS    = 1600
# CONTROL_GAIN = 3.0
# (PentagonCluster below was also called with default momentum_alpha=0.7,
#  stiction_threshold=0.025 -- Decabot hardware values, not the boat model.)

# Validated config (2026-07-01), ported from main_ocean_hfr_2km.py. See that
# file's CONFIGURATION section and config/formations/pentagon_small_2km.yaml
# for the full history/reasoning (0.7x formation to pick the correct branch
# at the 34.1N,-120.4W bifurcation; alpha=0/stiction~0 boat model since
# TIME_WARP=6000 means each 0.1s tick represents 10 real minutes, too long
# for Decabot-style momentum carryover to be physically meaningful).
FORMATION_CONFIG = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

# V_MAX        = 0.05
# SIM_STEPS    = 168     # 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 3.0

# Landfall-target refinement (2026-07-02): ported from main_ocean_hfr_2km.py.
# User wants the trajectory to continue right at the 34.1N,-120.4W
# bifurcation (as the config above already does) but then descend and make
# landfall on the SECOND/middle island near (34.0N, -120.24W), instead of
# continuing east past it. CONTROL_GAIN dropped from 3.0 to 1.8 (V_MAX also
# re-picked at 0.04) lets Logic C track the lower ridge branch that dips
# along the middle island's north coast; first approaches land at
# (34.04N, -120.26W), close to the target.
V_MAX        = 0.04
SIM_STEPS    = 168     # 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA      = 0.0     # boat model: full command authority each 10-min tick
# MOMENTUM_ALPHA    = 0.7      # Decabot hardware value; tau=0.3s at dt=0.1s
STICTION_THRESHOLD  = 0.002    # boat model: negligible "poor signal" floor
# STICTION_THRESHOLD = 0.025    # Decabot hardware value (Omnibot/PentagonCluster default)
EPS_RAW      = 1e-3
EPS_DIM      = 0.025
TIME_WARP    =  6000.0

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7

# START_POINTS = [
#     (34.422, -120.367, "furthest point"),
# ]
START_POINTS = [
    (34.4, -120.39, "furthest point"),
]

ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# FTLE settings -- must match _generate_ocean_ftle_2km_matched.py for a
# consistent-looking background.
FTLE_HOURS    = 24
SUBSTEPS_HR   = 6
SEED_UPSAMPLE = 4
FTLE_VMAX     = None
LAND_RGB      = (0.30, 0.55, 0.25)

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
DATA_DIR  = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "hfr_uswc_2012may_2km")
FRAME_GLOB = "*_2km_rtv_uwls_SIO.nc"
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
os.makedirs(OUT_DIR, exist_ok=True)


def _ts(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d %Y  %H:%M UTC")


# ============================================================================
# TRIAL
# ============================================================================

def _run_trial(field, cluster, start_lat, start_lon):
    """Run one simulation trial with the time-varying field; return histories."""
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
            field.step(cluster.timestep * TIME_WARP)

    total_field_secs = SIM_STEPS * cluster.timestep * TIME_WARP
    total_hours      = total_field_secs / 3600.0
    print(f"  Field-time traversed: {total_field_secs:.0f} s ({total_hours:.2f} h)")

    return cluster.get_robot_history(), cluster.get_center_history(), total_hours


def _draw_trajectories(ax, field, robot_history, center_history):
    """Draw robot + centroid trajectories in lat/lon, same styling as main_ocean_hfr_2km.py."""
    if len(robot_history) > 0:
        for i in range(robot_history.shape[1]):
            traj_world = robot_history[:, i, :]
            traj_lats, traj_lons = [], []
            for pt in traj_world:
                rlat, rlon = _world_to_latlon(pt[0], pt[1], field.config)
                traj_lats.append(rlat)
                traj_lons.append(rlon)
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.plot(traj_lons, traj_lats,
                    color=color, linewidth=1.5, alpha=0.8, linestyle='--')
            ax.plot(traj_lons[0], traj_lats[0],
                    marker='o', color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=1.2)
            ax.plot(traj_lons[-1], traj_lats[-1],
                    marker='s', color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=1.2)

    end_lat = end_lon = None
    if len(center_history) > 0:
        clats, clons = [], []
        for pt in center_history:
            clat, clon = _world_to_latlon(pt[0], pt[1], field.config)
            clats.append(clat)
            clons.append(clon)

        ax.plot(clons, clats, color='black', linewidth=2, alpha=0.9,
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

    return end_lat, end_lon


# ============================================================================
# MAIN
# ============================================================================

def main():
    field   = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)

    start_lat, start_lon, label = START_POINTS[0]
    print(f"\nRunning: {label}  from ({start_lat:.4f}N, {start_lon:.4f}W)")
    robot_history, center_history, total_hours = _run_trial(field, cluster, start_lat, start_lon)

    print("\nComputing FTLE background (2km native)...")
    lat_fine, lon_fine, f_val, land_fine, coast_polys, t_list = compute_ftle_field(
        DATA_DIR, FRAME_GLOB, COAST_SHP,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR, seed_upsample=SEED_UPSAMPLE,
    )
    f_plot = np.ma.array(f_val, mask=land_fine)
    L, LA = np.meshgrid(lon_fine, lat_fine)

    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    vmax = float(np.nanpercentile(f_val, 99)) if FTLE_VMAX is None else float(FTLE_VMAX)
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    end_lat, end_lon = _draw_trajectories(ax, field, robot_history, center_history)

    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)

    subtitle = (f"Start: ({start_lat:.4f}N, {start_lon:.4f}W)  "
                f"End: ({end_lat:.4f}N, {end_lon:.4f}W)  "
                f"[{total_hours:.2f} h of field time]"
                if end_lat is not None else "")
    ax.set_title(
        f"{_ts(t_list[0])}  (TIME_WARP={TIME_WARP:.0f}x)\n{subtitle}",
        fontsize=10
    )
    fig.suptitle(
        f"Pentagon Cluster on Forward-Time FTLE ({FTLE_HOURS} h) -- 2km, {label}",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(OUT_DIR, "ftle_trajectory_overlay_2km.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
