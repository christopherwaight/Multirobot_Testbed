"""
main_ocean_hfr_v6r_timevarying.py

6-robot pentagon cluster navigating a time-varying ocean HF radar surface
current field (Santa Barbara Channel, 2012-05-16 08:00 UTC through
2012-05-17 12:00 UTC, 29 hourly frames) using Primitive 7
(separatrix_logic_c_step, Logic C).

This is a copy of main_ocean_hfr_v6r.py with two changes:
  1. Uses ocean_hfr_socal_timevarying (reads ocean_hfr_timevarying.yaml,
     enable_time_interp: true).  All 29 frames are loaded on first call.
  2. Adds TIME_WARP: each sim step advances the field clock by
     (cluster.timestep * TIME_WARP) instead of just cluster.timestep.

Time math:
  cluster.timestep = 0.1 s  (fixed by PentagonCluster)
  At TIME_WARP = 360:
    field time per step = 0.1 * 360 = 36 s
    SIM_STEPS=110 => 3960 s of field time => ~1.1 HFR frames traversed
  For full 28-hour dataset in 110 steps:
    TIME_WARP = (28*3600) / (110*0.1) ~= 9164

Figure layout (1 x 2):
  Left:  quiver (sampled at t=0) + 6-robot trajectories in lat/lon.
  Right: precomputed 24-h FTLE reference image (static).
         Generate with: python experiments/_generate_ocean_ftle_reference.py

START_POINTS entries are (lat, lon, label). Any point within the extent
(33.8..34.7 N, -120.7..-119.7 W) is valid.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/main_ocean_hfr_v6r_timevarying.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ftle_common import compute_ftle_field

# Live FTLE background (2026-07-01): computed the same way as
# _generate_ocean_ftle_matched.py, so the right panel can show robot paths
# on top of it instead of a static precomputed PNG with no paths (matches
# the same change made to main_ocean_hfr_2km.py).
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", "..", "..", ".."))
_FTLE_DATA_DIR = os.path.join(_repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                               "ocean_data", "hfr_uswc_2012may")
_FTLE_FRAME_GLOB = "*_6km_rtv_uwls_SIO.nc"
_FTLE_COAST_SHP = os.path.join(_repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                                "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"

# Separatrix-tracking tuning (2026-07-01). Original values commented, not
# deleted. Fixes applied alongside this tuning pass:
#   - src/fields/environments/Ocean_HFR.py: field.t was being clipped against
#     absolute Unix-epoch t_frames while reset_clock()/step() only ever
#     produce small values near 0, so the field was silently frozen on frame
#     0 for the entire run regardless of TIME_WARP. Fixed to treat t as
#     relative-to-start seconds. Without this fix none of the parameters
#     below matter -- the field never varied in time.
# With the time bug fixed and formation scaled to 0.5x (see
# config/formations/pentagon_small.yaml), V_MAX=0.04/CONTROL_GAIN=3.0 tracks
# the separatrix from the start point down to the island gap at ~34.0N but
# then stalls/oscillates right at the island -- a real bifurcation point in
# the flow, not a bug. Raising V_MAX and CONTROL_GAIN gives the formation
# enough momentum to carry through the pinch and continue along the ridge's
# full bend past both islands to the domain edge.
# V_MAX        = 0.04    # Logic C saturation speed (m/s)
# SIM_STEPS    = 150     # steps per run
# CONTROL_GAIN = 3.0     # compensates momentum filter (alpha=0.7 -> v_actual = 0.3*v_cmd)

# First working pass (2026-07-01): gets past the island bottleneck and follows
# the ridge to the domain edge, but takes the wide bend well south/west of the
# western island rather than tracing its coastline ridge.
# V_MAX        = 0.06    # Logic C saturation speed (m/s)
# SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 4.0     # compensates momentum filter; raised to carry through island bottleneck

# Island-hugging refinement (2026-07-01, same day): CONTROL_GAIN dropped from
# 4.0 to 3.5 gives the formation just enough speed to clear the bottleneck
# without overshooting past the island's own boundary ridge -- the path now
# traces directly along the north shore of both islands (the det(J)=0 contour
# that hugs their coastlines) instead of cutting through open water further
# out. Confirmed sensitive to the exact start point (same bifurcation noted
# above); keep start point as-is rather than nudging further.
# V_MAX        = 0.06    # Logic C saturation speed (m/s)
# SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 3.5     # tuned to hug the island coastline ridge rather than overshoot past it

# Momentum-model correction (2026-07-01, same day): MOMENTUM_ALPHA=0.7 and
# STICTION_THRESHOLD=0.025 m/s are Decabot hardware measurements (see
# CLAUDE.md) for a REAL 0.1s control tick. Here, TIME_WARP=6000 means each
# 0.1s tick represents 600s (10 min) of scenario time -- the vehicle has a
# full 10 minutes to respond to a new command, not 0.1s, so a Decabot-style
# momentum carryover from "last tick" is not physically meaningful. Setting
# MOMENTUM_ALPHA=0.0 (command reached instantly, no carryover) and
# STICTION_THRESHOLD near 0 (water has no meaningful static-friction
# analogue) reflects that a 10-minute-old velocity command is stale.
# Retuned V_MAX/CONTROL_GAIN for the new dynamics: without momentum
# smoothing the controller reacts more sharply to local field structure, so
# gentler gains than the alpha=0.7 tuning above are needed to avoid
# overshooting past the islands.
# Values used for Decabot hardware (this repo's other experiments) and for
# the untuned original pass above: MOMENTUM_ALPHA=0.7, STICTION_THRESHOLD
# left at the Omnibot/PentagonCluster default (0.025 m/s).
V_MAX        = 0.05    # Logic C saturation speed (m/s)
SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
CONTROL_GAIN = 3.0     # gentler than alpha=0.7 tuning; no momentum smoothing to lean on
MOMENTUM_ALPHA      = 0.0     # boat model: full command authority each 10-min tick, no carryover
# MOMENTUM_ALPHA    = 0.7      # Decabot hardware value (this repo's other experiments); tau=0.3s at dt=0.1s
STICTION_THRESHOLD  = 0.002    # boat model: negligible "poor signal" floor, not true static friction
# STICTION_THRESHOLD = 0.025    # Decabot hardware value (Omnibot/PentagonCluster default)
EPS_RAW      = 1e-3    # raw det(J) threshold for FLOW-band
EPS_DIM      = 0.025   # dimensionless det(J)/||H_det||_F threshold

# Time-warp multiplier.
# Each sim step advances field.t by (cluster.timestep * TIME_WARP).
# See module docstring for the conversion between TIME_WARP and frames traversed.
# 6000 kept as-is: verified realistic. cluster.timestep=0.1s matches the 10 Hz
# hardware control rate; at TIME_WARP=6000 each step advances the field clock
# by 600s (10 min), so SIM_STEPS=168 spans exactly the 28h of the 29-frame
# dataset without exceeding it (field.step() beyond that just clamps to the
# last frame, which is not a "faster" simulation, just a frozen one).
TIME_WARP = 6000.0

# Geographic extent matching ftle_frame0_matched.png.
LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7
QUIVER_STEP_DEG  = 0.06    # ~20 arrows per axis across the extent

# Start points: (lat, lon, label).
# Any lat/lon inside the extent above is valid.
START_POINTS = [
    (34.41, -120.35, "furthest point"),
]

ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# ============================================================================
# COORDINATE HELPERS
# ============================================================================

def _affine(config):
    """Return (center_lat, center_lon, scale) where world_coord * scale = degrees."""
    clat  = config.get("center_lat",   34.2)
    clon  = config.get("center_lon",  -120.4)
    scale = config.get("roi_half_deg", 0.3) / config.get("world_half", 0.65)
    return clat, clon, scale


def _latlon_to_world(lat, lon, config):
    """Convert geographic (lat, lon) to cluster world (x, y)."""
    clat, clon, scale = _affine(config)
    return (lon - clon) / scale, (lat - clat) / scale


def _world_to_latlon(x, y, config):
    """Convert cluster world (x, y) to geographic (lat, lon)."""
    clat, clon, scale = _affine(config)
    return clat + y * scale, clon + x * scale


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _draw_quiver(ax, field):
    """Render the field as quiver arrows on a lat/lon grid (sampled at t=0)."""
    lons = np.arange(LON_MIN, LON_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    lats = np.arange(LAT_MIN, LAT_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    LON, LAT = np.meshgrid(lons, lats)
    U = np.zeros_like(LON)
    V = np.zeros_like(LAT)
    # Snapshot the clock, evaluate at t=0, restore.
    saved_t = field.t
    field.t = 0.0
    for r in range(LON.shape[0]):
        for c in range(LON.shape[1]):
            wx, wy = _latlon_to_world(LAT[r, c], LON[r, c], field.config)
            u, v = field.get_value(wx, wy)
            U[r, c] = u
            V[r, c] = v
    field.t = saved_t
    ax.quiver(LON, LAT, U, V, color='black', alpha=0.5)


def _draw_ftle_with_trajectories(ax, cluster, field):
    """
    Compute the 24h forward-time FTLE field live (6km resolution) and draw
    the robot + centroid trajectories from `cluster` on top of it.

    Replaces the old static-PNG-only right panel (2026-07-01): previously
    this just showed ftle_frame0_matched.png with no paths. Matches the same
    change made to main_ocean_hfr_2km.py so both experiment files have the
    same layout: left panel unchanged (quiver + trajectory), right panel
    shows the live-computed FTLE field with the same run's paths overlaid.
    """
    lat_fine, lon_fine, f_val, land_fine, coast_polys, t_list = compute_ftle_field(
        _FTLE_DATA_DIR, _FTLE_FRAME_GLOB, _FTLE_COAST_SHP,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        ftle_hours=24, substeps_hr=6, seed_upsample=4,
    )
    f_plot = np.ma.array(f_val, mask=land_fine)
    L, LA = np.meshgrid(lon_fine, lat_fine)

    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=(0.30, 0.55, 0.25))
    vmax = float(np.nanpercentile(f_val, 99))
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

    ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    # ── Draw robot trajectories ────────────────────────────────────────────
    robot_history = cluster.get_robot_history()
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

    # ── Draw centroid path ──────────────────────────────────────────────────
    center_history = cluster.get_center_history()
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

    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (deg)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title("Forward-time FTLE (24 h) -- 6km, with robot paths",
                 fontsize=12, fontweight="bold")


# ============================================================================
# TRIAL
# ============================================================================

def _run_trial(ax, start_lat, start_lon, label, field, cluster):
    """Run one simulation trial with time-varying field and draw on ax in lat/lon."""
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
    # 29 frames span 28 hours (frame 0 at 0 h, frame 28 at 28 h).
    frames_spanned   = total_field_secs / 3600.0
    print(f"  Field-time traversed: {total_field_secs:.0f} s "
          f"({total_hours:.2f} h, ~{frames_spanned:.1f} frames)")

    # ── Draw quiver background at t=0 ────────────────────────────────────
    _draw_quiver(ax, field)

    # ── Draw robot trajectories ───────────────────────────────────────────
    robot_history = cluster.get_robot_history()
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
    center_history = cluster.get_center_history()
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
            f"End: ({end_lat:.4f}N, {end_lon:.4f}W)  "
            f"[{total_hours:.2f} h of field time]",
            fontsize=9
        )

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title(f"Pentagon Cluster -- {label}  (TIME_WARP={TIME_WARP:.0f}x)",
                 fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    field   = AnalyticalField(ocean_hfr_socal_timevarying)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)

    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_ref  = fig.add_subplot(gs[0, 1])

    start_lat, start_lon, label = START_POINTS[0]
    print(f"\nRunning: {label}  from ({start_lat:.4f}N, {start_lon:.4f}W)")
    _run_trial(ax_traj, start_lat, start_lon, label, field, cluster)
    print("\nComputing FTLE background for right panel (6km)...")
    _draw_ftle_with_trajectories(ax_ref, cluster, field)

    plt.suptitle(
        f"Pentagon Cluster -- Logic C Navigation  (TIME_WARP={TIME_WARP:.0f}x)\n"
        "Ocean HFR field, time-varying, Santa Barbara Channel (May 16-17 2012)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
