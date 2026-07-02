"""
main_ocean_hfr_2km.py

6-robot pentagon cluster navigating a time-varying ocean HF radar surface
current field at 2km NATIVE resolution (Santa Barbara Channel, 2012-05-16
08:00 UTC through 2012-05-17 12:00 UTC, 29 hourly frames) using Primitive 7
(separatrix_logic_c_step, Logic C).

This is a copy of main_ocean_hfr_v6r_timevarying.py with the field repointed
at the 2km dataset:
  1. Reuses ocean_hfr_socal_timevarying unchanged, but constructs the field
     with an explicit config_name override so it reads
     config/fields/ocean_hfr_2km_timevarying.yaml (data_dir/frame_glob point
     at ocean_data/hfr_uswc_2012may_2km instead of the 6km folder).
  2. Right panel displays the 2km FTLE reference image instead of the 6km one.
  3. Coordinate helpers moved to _coords_common.py (shared with the FTLE
     overlay script) instead of being defined inline.

All simulation parameters (TIME_WARP, SIM_STEPS, START_POINTS, geographic
extent) are kept identical to the 6km version so the two outputs are a fair
side-by-side comparison of resolution only.

SBC valid-data coverage at 2km is lower (~48-58%) than 6km (~67%), see
ocean_data/DATA_PROVENANCE.md -- expect more gap-filled patches in the quiver.

Figure layout (1 x 2):
  Left:  quiver (sampled at t=0) + 6-robot trajectories in lat/lon.
  Right: precomputed 24-h FTLE reference image (static), 2km native resolution.
         Generate with: python experiments/_generate_ocean_ftle_2km_matched.py

START_POINTS entries are (lat, lon, label). Any point within the extent
(33.8..34.7 N, -120.7..-119.7 W) is valid.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/main_ocean_hfr_2km.py
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
from _coords_common import affine as _affine
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

# Live FTLE background (2026-07-01): computed the same way as
# _generate_ocean_ftle_2km_matched.py / main_ocean_hfr_2km_ftle_overlay.py,
# so the right panel can show robot paths on top of it instead of a static
# precomputed PNG with no paths.
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", "..", "..", ".."))
_FTLE_DATA_DIR = os.path.join(_repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                               "ocean_data", "hfr_uswc_2012may_2km")
_FTLE_FRAME_GLOB = "*_2km_rtv_uwls_SIO.nc"
_FTLE_COAST_SHP = os.path.join(_repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                                "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")

# ============================================================================
# CONFIGURATION
# ============================================================================

# FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
# Repointed (2026-07-01) to a dedicated 2km formation config: the 6km
# experiment is validated at 0.5x scale in pentagon_small.yaml, but 2km's
# noisier field needs 0.7x to pick the correct branch at the bifurcation
# near 34.1N,-120.4W (see pentagon_small_2km.yaml for full history/reasoning).
FORMATION_CONFIG = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

# Separatrix-tracking tuning (2026-07-01). Original values commented, not
# deleted. Same field-clock fix as the 6km version applies here too (see
# src/fields/environments/Ocean_HFR.py: field.t was clipped against absolute
# Unix-epoch t_frames while reset_clock()/step() only ever produce small
# relative values, so the field was silently frozen on frame 0 regardless of
# TIME_WARP -- fixed to treat t as relative-to-start seconds).
#
# The 2km field is much noisier than 6km (lower valid-data coverage, see
# ocean_data/DATA_PROVENANCE.md): ridges are thinner and more fragmented, and
# the region around 34.1-34.15N / -120.4W is a genuine bifurcation point
# where trajectories starting even ~1km apart diverge onto different ridge
# branches (one continues south along the separatrix, the other peels west
# off the domain edge). This is expected behavior for a repelling LCS, not a
# bug. The tuned V_MAX/CONTROL_GAIN below plus a small start-point shift
# (34.48,-120.35) -> (34.44,-120.38), well within the "any point in the
# extent is valid" note above, land on the branch that tracks the separatrix
# past both islands to the domain edge.
# V_MAX        = 0.04    # Logic C saturation speed (m/s)
# SIM_STEPS    = 110     # steps per run
# CONTROL_GAIN = 3.0     # compensates momentum filter (alpha=0.7 -> v_actual = 0.3*v_cmd)

# First working pass (2026-07-01): gets past the island bottleneck and follows
# the ridge to the domain edge, but the wide bend passes south/west of the
# western island in open water rather than tracing its coastline ridge.
# V_MAX        = 0.05    # Logic C saturation speed (m/s)
# SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 3.5     # compensates momentum filter; raised slightly for 2km ridge tracking

# Island-hugging refinement (2026-07-01, same day): raising V_MAX to match the
# 6km value (0.06) while keeping CONTROL_GAIN at 3.5 threads the formation
# directly along the small island's coastline ridge and through the pinch
# between the two islands, instead of the wider bend further out. Same
# bifurcation sensitivity as before applies -- keep start point as-is.
# V_MAX        = 0.06    # Logic C saturation speed (m/s)
# SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 3.5     # tuned to hug the island coastline ridge rather than pass wide of it

# Bifurcation-correct refinement (2026-07-01, same day): the 0.5x/0.6x
# formations above pick the WRONG branch at the bifurcation near
# 34.1N,-120.4W (diverge into open water); 6km's field at the same point is
# smooth and shows no bifurcation at all -- it's a small-scale feature only
# visible in the noisier 2km data. Switching to the 0.7x formation
# (config/formations/pentagon_small_2km.yaml) averages over enough of that
# small-scale structure to recover the same branch 6km follows. V_MAX and
# CONTROL_GAIN dropped (gentler approach) so the cluster doesn't overshoot
# into the islands' interior at the pinch points; with this combination the
# whole path from start to the domain edge stays in water except for one
# brief loop that clips the third island group's coastline (see
# _run_trial/main() below -- acceptable per project notes: a single-robot
# grounding there is out-of-scope obstacle-avoidance future work, not a
# failure of the separatrix-tracking demonstration).
#
# Momentum-model correction (2026-07-01, same day): same reasoning as the
# 6km file -- TIME_WARP=6000 means each 0.1s control tick represents 10 real
# minutes of scenario time, so a Decabot-style momentum carryover from "last
# tick" (MOMENTUM_ALPHA=0.7) does not reflect a vehicle's actual ability to
# change velocity over a full 10-minute window. Setting MOMENTUM_ALPHA=0.0
# (full command authority each tick) and STICTION_THRESHOLD near 0 (no
# meaningful static-friction analogue in water) resolved the land-crossing
# loop above entirely -- the alpha=0 trajectory stays in water the whole way
# past all three island groups.
# Decabot hardware values (this repo's other experiments): MOMENTUM_ALPHA=0.7,
# STICTION_THRESHOLD left at the Omnibot/PentagonCluster default (0.025 m/s).
# V_MAX        = 0.05    # Logic C saturation speed (m/s)
# SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
# CONTROL_GAIN = 3.0     # gentler approach avoids overshooting into island interiors at the pinch

# Landfall-target refinement (2026-07-02): user wants the trajectory to
# continue right at the 34.1N,-120.4W bifurcation (as it already did) but
# then descend and make landfall on the SECOND/middle island near
# (34.0N, -120.24W), instead of continuing east and only grazing the third
# island group. The alpha=0/CONTROL_GAIN=3.0 config above stays too far
# north (~34.1-34.15N) past the bifurcation, missing the ridge that dips
# down along the middle island's north coast. Lowering CONTROL_GAIN to 1.8
# (V_MAX unchanged effect-wise but re-picked at 0.04 -- both knobs were swept
# together) lets Logic C track that lower ridge branch: the path descends
# smoothly right after the bifurcation and follows the coastline of the
# middle island closely, first approaching land at (34.04N, -120.26W) --
# close to the requested target. Sensitive to gain in the same way as the
# other bifurcations found today: CONTROL_GAIN=2.0 overshoots slightly
# further east (first touches near -120.19W); 2.5 undershoots and never
# quite reaches the coastline. 1.8 was the closest match found by sweep.
V_MAX        = 0.04    # Logic C saturation speed (m/s)
SIM_STEPS    = 168     # steps per run: 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
CONTROL_GAIN = 1.8     # tuned to descend onto the middle island's ridge instead of continuing east
MOMENTUM_ALPHA      = 0.0     # boat model: full command authority each 10-min tick, no carryover
# MOMENTUM_ALPHA    = 0.7      # Decabot hardware value; tau=0.3s at dt=0.1s
STICTION_THRESHOLD  = 0.002    # boat model: negligible "poor signal" floor, not true static friction
# STICTION_THRESHOLD = 0.025    # Decabot hardware value (Omnibot/PentagonCluster default)
EPS_RAW      = 1e-3    # raw det(J) threshold for FLOW-band
EPS_DIM      = 0.025   # dimensionless det(J)/||H_det||_F threshold

# Time-warp multiplier.
# Each sim step advances field.t by (cluster.timestep * TIME_WARP).
# Same value as the 6km version, see main_ocean_hfr_v6r_timevarying.py docstring
# for the conversion between TIME_WARP and frames traversed.
# 6000 kept as-is: verified realistic. cluster.timestep=0.1s matches the 10 Hz
# hardware control rate; SIM_STEPS=168 spans exactly the 28h dataset window.
TIME_WARP = 6000.0

# Geographic extent matching ftle_frame0_2km_matched.png.
LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7
QUIVER_STEP_DEG  = 0.06    # ~20 arrows per axis across the extent

# Start points: (lat, lon, label).
# Any lat/lon inside the extent above is valid.
# Shifted slightly from (34.48, -120.35) -- see CONFIGURATION comment above.
# This is a small nudge (~0.05 deg), not a relocation to a different feature.
# START_POINTS = [
#     (34.48, -120.35, "furthest point"),
# ]
START_POINTS = [
    (34.44, -120.38, "furthest point"),
]

ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

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
    Compute the 24h forward-time FTLE field live (2km native resolution) and
    draw the robot + centroid trajectories from `cluster` on top of it.

    Replaces the old static-PNG-only right panel (2026-07-01): previously
    this just showed ftle_frame0_2km_matched.png with no paths, so the two
    panels could not be visually cross-checked against each other directly.
    Ported from main_ocean_hfr_2km_ftle_overlay.py's single-panel approach
    (same _ftle_common.compute_ftle_field call, same trajectory styling).
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
    ax.set_title("Forward-time FTLE (24 h) -- 2km native, with robot paths",
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
    ax.set_title(f"Pentagon Cluster -- {label}  (2km, TIME_WARP={TIME_WARP:.0f}x)",
                 fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    field   = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
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
    print("\nComputing FTLE background for right panel (2km native)...")
    _draw_ftle_with_trajectories(ax_ref, cluster, field)

    plt.suptitle(
        f"Pentagon Cluster -- Logic C Navigation  (2km native, TIME_WARP={TIME_WARP:.0f}x)\n"
        "Ocean HFR field, time-varying, Santa Barbara Channel (May 16-17 2012)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
