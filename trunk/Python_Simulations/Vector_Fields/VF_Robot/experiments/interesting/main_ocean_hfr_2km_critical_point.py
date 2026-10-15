"""
main_ocean_hfr_2km_critical_point.py

3-robot equilateral cluster navigating a time-varying ocean HF radar surface
current field (Santa Barbara Channel, 2012-05-16 08:00 UTC through
2012-05-17 12:00 UTC, 29 hourly frames, 2km native resolution) using
critical_point_plane_fitting: the 3-robot plane-fit estimator (paper Eqs.
2-10) that solves for the local critical point from one linear field
sample per robot and drives the centroid toward it.

This is the 3-robot analogue of main_ocean_hfr_2km.py (6-robot pentagon,
Logic C separatrix tracking). Same field, same time-varying clock handling;
only the cluster type, control primitive, formation config, and figure
(single quiver panel, no FTLE) differ.

Because the field is time-varying, the plane-fit estimate at each step is a
local, instantaneous critical point that itself drifts as the field
evolves -- the cluster does not converge to and hold a single fixed point,
it tracks a moving attractor and stays in its neighborhood. See the start
point note below for how this was chosen.

START POINT
Target: the main persistent stagnation feature near (34.3N, -120.1W). A
time-averaged scan (field speed averaged over 8 snapshots spanning the
full 28h window, not just one instant) found the lowest-speed region
centered at (34.31N, -120.22W), speed ~0.10-0.11 m/s vs. higher ambient
elsewhere in the neighborhood -- checked at t=0/14/28h and confirmed to
show the sign-flip (convergence-line) structure of a real saddle/
stagnation feature, not just a single-frame low reading. Target zone used
for search: within 0.08 deg of this point (a margin, not an exact pixel).

Objective (per user request): maximize straight-line (L2) distance between
start and finish, not cumulative path length traveled.

Search process (see scratchpad/search_max_l2.py, search_max_l2_perimeter.py,
search_radial.py, search_radial_bigsize.py -- not part of this repo):
  1. 150-trial random search over the whole domain x 5 formation sizes
     (0.15/0.30/0.50/0.70/1.00): only 6/150 hits, all at size=0.15, all
     converging to the same endpoint (34.31N,-120.18W). Best L2=0.186.
  2. 40-point domain-perimeter/corner sweep at size=0.15: 0/40 hits -- the
     basin does not reach the domain edges; most perimeter starts drift
     off-domain or into unrelated attractors.
  3. Radial sweep at size=0.15 (7 radii x 16 angles around the target):
     mapped the basin as roughly centered on the target itself, not
     domain-spanning. Best: start (34.56N,-120.22W), due north at
     r=0.25 deg, end (34.31N,-120.18W), L2=0.2502.
  4. Repeated the radial sweep at larger sizes (0.5/0.7/1.0, ~230 trials)
     to test whether a bigger formation widens this basin (it does for a
     DIFFERENT target used in an earlier version of this experiment, see
     formation-config header) -- for THIS target it does not: zero hits
     at any size above 0.15. At 0.5+ the plane-fit estimate is pulled to
     an unrelated attractor near (34.31N,-120.00W) or leaves the domain
     entirely. Bigger is not universally better; it depends on the basin.
(34.56N, -120.22W) was chosen as the best-supported start point: largest
L2 (0.25 deg) found across every size and search strategy tried.

Formation scale: p=q=0.15 world units. See
config/formations/equilateral_ocean_hfr_2km.yaml for the full scale
history (including the larger sizes tried and reverted).

Robot dynamics: MOMENTUM_ALPHA=0 (no velocity carryover, commanded velocity
applied exactly each tick) and STICTION_THRESHOLD=0 (no static-friction
floor) -- only the current plane-fit reading drives motion, no history
dependence from the robot model itself. Re-verified the chosen start point
gives an identical trajectory under the old Decabot-like stiction default
(0.025 m/s) vs. 0, since commanded speeds here are almost always well
above that threshold either way.

Figure: single quiver panel (field sampled at end-of-run field time, not
t=0) with the 3-robot trajectories and centroid path overlaid.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/main_ocean_hfr_2km_critical_point.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
import src.control.primitives as ocp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/equilateral_ocean_hfr_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.06    # centroid velocity saturation (m/s), matches critical_point_plane_fitting's own cap
CONTROL_GAIN = 3.0     # compensates momentum filter; same value as main_ocean_hfr_2km.py's first working pass
MOMENTUM_ALPHA = 0.0   # no momentum carryover: velocity = commanded velocity exactly each tick,
                       # only the current plane-fit reading matters (Decabot hardware value is 0.7)
STICTION_THRESHOLD = 0.0  # no static-friction floor (Decabot hardware default is 0.025 m/s)

TIME_WARP = 6000.0     # field seconds per sim step = cluster.timestep * TIME_WARP
SIM_STEPS = 168        # 168 * 0.1s * 6000 = 28h = full dataset window

# Geographic extent matching the FTLE reference imagery.
LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7
QUIVER_STEP_DEG  = 0.06

START_POINTS = [
    (34.56, -120.22, "max L2 distance to stagnation target"),
]

ROBOT_COLORS = ['blue', 'orange', 'green']

# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _draw_quiver(ax, field, t_snapshot):
    """Render the field as quiver arrows on a lat/lon grid, sampled at field
    time `t_snapshot` (seconds). Pass the end-of-run field time to show the
    field as it stood when the trajectory finished, not its initial state."""
    lons = np.arange(LON_MIN, LON_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    lats = np.arange(LAT_MIN, LAT_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    LON, LAT = np.meshgrid(lons, lats)
    U = np.zeros_like(LON)
    V = np.zeros_like(LAT)
    saved_t = field.t
    field.t = t_snapshot
    for r in range(LON.shape[0]):
        for c in range(LON.shape[1]):
            wx, wy = _latlon_to_world(LAT[r, c], LON[r, c], field.config)
            u, v = field.get_value(wx, wy)
            U[r, c] = u
            V[r, c] = v
    field.t = saved_t
    ax.quiver(LON, LAT, U, V, color='black', alpha=0.5)


# ============================================================================
# TRIAL
# ============================================================================

def _run_trial(ax, start_lat, start_lon, label, field, cluster):
    """Run one simulation trial with time-varying field and draw on ax in lat/lon."""
    sx, sy = _latlon_to_world(start_lat, start_lon, field.config)
    cluster.reset(sx, sy)

    def primitive(c):
        vx, vy = ocp.critical_point_plane_fitting(c)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    if hasattr(field, "reset_clock"):
        field.reset_clock()

    for _ in range(SIM_STEPS):
        cluster.move(primitive)
        if hasattr(field, "step"):
            field.step(cluster.timestep * TIME_WARP)

    total_field_secs = SIM_STEPS * cluster.timestep * TIME_WARP
    total_hours      = total_field_secs / 3600.0
    print(f"  Field-time traversed: {total_field_secs:.0f} s "
          f"({total_hours:.2f} h, ~{total_hours:.1f} frames)")

    _draw_quiver(ax, field, t_snapshot=field.t)

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

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_title(f"3-Robot Cluster -- {label}  (2km, TIME_WARP={TIME_WARP:.0f}x)\n"
                 "Quiver shown at end-of-run field time",
                 fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.tick_params(labelsize=10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    field   = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = OmniCluster(FORMATION_CONFIG, field, momentum_alpha=MOMENTUM_ALPHA,
                          stiction_threshold=STICTION_THRESHOLD)

    fig, ax = plt.subplots(figsize=(8, 7))

    start_lat, start_lon, label = START_POINTS[0]
    print(f"\nRunning: {label}  from ({start_lat:.4f}N, {start_lon:.4f}W)")
    _run_trial(ax, start_lat, start_lon, label, field, cluster)

    plt.suptitle(
        f"3-Robot Cluster -- Critical Point Attraction  (2km native, TIME_WARP={TIME_WARP:.0f}x)\n"
        "Ocean HFR field, time-varying, Santa Barbara Channel (May 16-17 2012)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
