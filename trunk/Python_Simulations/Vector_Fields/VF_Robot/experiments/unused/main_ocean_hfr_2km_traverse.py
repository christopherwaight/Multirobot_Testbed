"""
main_ocean_hfr_2km_traverse.py

Objective separatrix traverser (Primitive 11, oecs_separatrix_step) on the
real 2km Santa Barbara Channel HFR field, at the same validated operating
point as the Logic C run in main_ocean_hfr_2km_ftle_overlay.py and the
Primitive 10 core-seek run in main_ocean_hfr_2km_oecs.py (alpha_mom=0,
stiction 0.002, V_MAX 0.04, GAIN 1.8, TIME_WARP 6000, pentagon_small_2km 0.7x
formation, start 34.40N -120.39W, 168 steps = 28h of field time).

GENERICITY CHECK: unlike the double gyre (where the shear strain vanishes
identically and the strain eigenvectors sit exactly on the coordinate axes),
the ocean HFR field is generic -- isolated singular points of S rather than
degeneracy LINES. This script tests whether the flow-free tangent-selection
rule (pick the eigenvector with the LARGER |grad s1 . e_i| as the tangent;
see pentagon_primitives.py Primitive 11) still resolves sensibly on a real,
noisy, non-axis-aligned field, and whether the CAPTURE test (full |grad s1|
below a threshold) still finds a genuine s1 stationary point rather than
never engaging or engaging spuriously.

Differences from main_ocean_hfr_2km_oecs.py (Primitive 10, core-seek):
  - Primitive: oecs_separatrix_step. No s_capture (ride to a CAPTURE-found
    stationary point, or to the run's step budget, whichever comes first);
    g_capture reused from the double-gyre default (0.15) since the field's
    own s1 gradient scale has not been separately re-probed here -- this is
    itself part of the generality test.
  - Ground-truth overlay: same offline TRAP-core extraction as Primitive
    10's ocean run (3x3 local minima of s1 below the 15th percentile on a
    120x120 world grid at the final field time). The distance from the
    parked (or final) centroid to the nearest offline core is the headline
    metric, directly comparable to Primitive 10's ocean result.

Paper traceability: candidate figure
  Paper_Writing/Separatrix_and_OW_Paper/figures/ocean_traverse_overlay_2km.png
(review copy ocean_data/det_jacobian_plots/traverse_trajectory_overlay_2km.png)
and the summary CSV experiments/outputs/oecs/ocean_traverse_summary.csv.
"""
import sys
import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from datetime import datetime, timezone

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import oecs_separatrix_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

# ============================================================================
# CONFIGURATION (operating point identical to main_ocean_hfr_2km_oecs.py)
# ============================================================================

FORMATION_CONFIG  = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.04
SIM_STEPS    = 168     # 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA     = 0.0
STICTION_THRESHOLD = 0.002
TIME_WARP    = 6000.0

# Traverser thresholds: reused from the double-gyre default, NOT re-probed
# for this field's own s1 scale -- part of the generality test.
G_PERP    = 1.0
S_TRIM    = 0.3     # matches Primitive 10's ocean S_TRIM (probed field scale)
R_BAND    = 0.05
G_CAPTURE = 0.15

START = (34.4, -120.39)     # lat, lon; same as the Logic C and OECS runs
                            # default -- overridable via --start-lat/--start-lon
                            # below, for sweeping candidate starts against the
                            # D tracker's own landfall (start-sweep session,
                            # Draft_5b referee-report response)

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7

FTLE_HOURS    = 24
SUBSTEPS_HR   = 6
SEED_UPSAMPLE = 4
LAND_RGB      = (0.30, 0.55, 0.25)

S1_GRID_N       = 120
S1_FD_H         = 0.01
CORE_PERCENTILE = 15.0

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
DATA_DIR  = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "hfr_uswc_2012may_2km")
FRAME_GLOB = "*_2km_rtv_uwls_SIO.nc"
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
CSV_DIR   = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']


def _ts(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d %Y  %H:%M UTC")


# ============================================================================
# OFFLINE s1 FIELD AND TRAP CORES (identical to main_ocean_hfr_2km_oecs.py)
# ============================================================================

def s1_grid(field, n=S1_GRID_N, h=S1_FD_H, lim=0.62):
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    S1 = np.full((n, n), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            try:
                ux = (field.get_value(x + h, y)[0]
                      - field.get_value(x - h, y)[0]) / (2 * h)
                uy = (field.get_value(x, y + h)[0]
                      - field.get_value(x, y - h)[0]) / (2 * h)
                vx = (field.get_value(x + h, y)[1]
                      - field.get_value(x - h, y)[1]) / (2 * h)
                vy = (field.get_value(x, y + h)[1]
                      - field.get_value(x, y - h)[1]) / (2 * h)
            except Exception:
                continue
            mu, a, b = 0.5 * (ux + vy), 0.5 * (ux - vy), 0.5 * (uy + vx)
            S1[j, i] = mu - np.hypot(a, b)
    return xs, ys, S1


def trap_cores(xs, ys, S1, percentile=CORE_PERCENTILE):
    thresh = np.nanpercentile(S1, percentile)
    cores = []
    n = S1.shape[0]
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            v = S1[j, i]
            if not np.isfinite(v) or v > thresh:
                continue
            neigh = S1[j - 1:j + 2, i - 1:i + 2]
            if v <= np.nanmin(neigh):
                cores.append((xs[i], ys[j], v))
    return cores


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-lat", type=float, default=START[0],
                     help="start latitude, deg N (default: validated start)")
    ap.add_argument("--start-lon", type=float, default=START[1],
                     help="start longitude, deg W as a negative number "
                          "(default: validated start)")
    ap.add_argument("--tag", type=str, default="",
                     help="suffix appended to the output PNG/CSV filenames "
                          "(before the extension) so sweep runs do not "
                          "overwrite each other or the validated baseline; "
                          "default '' reproduces today's filenames exactly")
    args = ap.parse_args()
    start = (args.start_lat, args.start_lon)
    suffix = f"_{args.tag}" if args.tag else ""

    field = AnalyticalField(ocean_hfr_socal_timevarying,
                            config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                              momentum_alpha=MOMENTUM_ALPHA,
                              stiction_threshold=STICTION_THRESHOLD)
    sx, sy = _latlon_to_world(start[0], start[1], field.config)
    cluster.reset(sx, sy)

    def primitive(c):
        vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP,
                                      s_trim=S_TRIM, r_band=R_BAND,
                                      g_capture=G_CAPTURE, s_capture=None)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    field.reset_clock()
    print(f"Running objective traverser from ({start[0]:.3f}N, {start[1]:.3f}W)...")
    for _ in range(SIM_STEPS):
        cluster.move(primitive)
        field.step(cluster.timestep * TIME_WARP)
    hours = SIM_STEPS * cluster.timestep * TIME_WARP / 3600.0
    print(f"  Field time traversed: {hours:.1f} h")

    center = cluster.get_center_history()
    robots = cluster.get_robot_history()
    diag = cluster.diagnostics
    modes = [d['mode'] for d in diag]
    occupancy = {m: round(modes.count(m) / max(len(modes), 1), 3)
                 for m in sorted(set(modes))}
    print(f"  Mode occupancy: {occupancy}")

    print("Extracting offline TRAP cores at final field time...")
    xs, ys, S1 = s1_grid(field)
    cores = trap_cores(xs, ys, S1)
    fx, fy = center[-1]
    dists = [np.hypot(fx - cx, fy - cy) for cx, cy, _ in cores]
    i_min = int(np.argmin(dists)) if dists else -1
    d_core_world = dists[i_min] if dists else float('nan')
    d_core_km = d_core_world * 51.0   # ~51 km per world unit (paper Sec. VI)
    s1_tail = [d['s1'] for d in diag[-20:]]
    print(f"  {len(cores)} offline cores; ended {d_core_world:.4f} world "
          f"units ({d_core_km:.1f} km) from the nearest, s1 tail mean "
          f"{np.mean(s1_tail):.2f}")

    print("Computing FTLE background (2km native)...")
    lat_f, lon_f, f_val, land_f, coast_polys, t_list = compute_ftle_field(
        DATA_DIR, FRAME_GLOB, COAST_SHP,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR,
        seed_upsample=SEED_UPSAMPLE,
    )
    f_plot = np.ma.array(f_val, mask=land_f)
    L, LA = np.meshgrid(lon_f, lat_f)
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    vmax = float(np.nanpercentile(f_val, 99))
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    for i in range(robots.shape[1]):
        tw = robots[:, i, :]
        lats, lons = zip(*[_world_to_latlon(p[0], p[1], field.config)
                           for p in tw])
        ax.plot(lons, lats, color=ROBOT_COLORS[i % 6], linewidth=1.2,
                alpha=0.7, linestyle='--')
    clats, clons = zip(*[_world_to_latlon(p[0], p[1], field.config)
                         for p in center])
    ax.plot(clons, clats, color='black', linewidth=2, alpha=0.9,
            label='Centroid path')
    ax.plot(clons[0], clats[0], marker='*', color='lime', markersize=14,
            markeredgecolor='black', markeredgewidth=1.5, label='Start',
            zorder=10)
    ax.plot(clons[-1], clats[-1], marker='X', color='red', markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, label='End',
            zorder=10)
    core_lats, core_lons = [], []
    for cx, cy, _ in cores:
        clat, clon = _world_to_latlon(cx, cy, field.config)
        core_lats.append(clat)
        core_lons.append(clon)
    ax.plot(core_lons, core_lats, linestyle='none', marker='*',
            color='white', markersize=11, markeredgecolor='black',
            markeredgewidth=0.8, label='Offline TRAP cores (final frame)',
            zorder=9)
    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6,
                alpha=0.85)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_aspect("equal")
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)
    ax.set_title(
        f"{_ts(t_list[0])}  (TIME_WARP={TIME_WARP:.0f}x)\n"
        f"End ({clats[-1]:.3f}N, {clons[-1]:.3f}W), "
        f"{d_core_km:.1f} km from nearest offline core "
        f"[{hours:.1f} h of field time]", fontsize=10)
    fig.suptitle(
        "Objective Separatrix Traverser on Forward-Time FTLE -- 2km SBC",
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(OUT_DIR, f"traverse_trajectory_overlay_2km{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    csv_path = os.path.join(CSV_DIR, f"ocean_traverse_summary{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_lat", "start_lon", "end_lat", "end_lon",
                    "hours", "d_core_world", "d_core_km", "n_cores",
                    "s1_tail_mean"] + [f"occ_{k}" for k in occupancy])
        w.writerow([start[0], start[1], round(clats[-1], 4),
                    round(clons[-1], 4), hours, round(d_core_world, 4),
                    round(d_core_km, 2), len(cores),
                    round(float(np.mean(s1_tail)), 3)]
                   + list(occupancy.values()))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
