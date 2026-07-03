"""
ocean_hfr_2km_branch_sensitivity.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  fig:ocean_branch_sensitivity, ~100 centroid starts jittered
          around the validated start point, color-coded by final landfall
          location, plus a distance-to-nearest-FTLE-ridge metric per path.

Same validated trial config as main_ocean_hfr_2km_ftle_overlay.py, run
from a jittered grid of starts around (34.40N, -120.39W). Each trial's
final position is classified by which side of the known bifurcation
(~34.1N, -120.4W; see main_ocean_hfr_2km.py comments) it lands on: the
separatrix-tracking branch continues south/southeast to the middle
island, the other branch peels west/offshore. Distance from each final
position to the nearest point of the 24-h forward FTLE ridge (defined as
the 95th-percentile contour) is reported as a per-path sensitivity
metric, using the same _ftle_common.compute_ftle_field FTLE field as
main_ocean_hfr_2km_ftle_overlay.py.

Running (takes ~10 minutes for 100 trials, no FTLE background per trial):
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_hfr_2km_branch_sensitivity.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.spatial import cKDTree

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

# ============================================================================
# CONFIGURATION -- identical to main_ocean_hfr_2km_ftle_overlay.py
# ============================================================================
FORMATION_CONFIG = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.04
SIM_STEPS    = 168
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA      = 0.0
STICTION_THRESHOLD  = 0.002
EPS_RAW      = 1e-3
EPS_DIM      = 0.025
TIME_WARP    = 6000.0

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7

CENTER_LAT, CENTER_LON = 34.4, -120.39
# ~100 starts: 10x10 jitter grid over a small box around the validated start.
N_SIDE = 10
JITTER_DEG = 0.05   # +/- half-width of the jitter box, degrees

# Branch split: known bifurcation near 34.1N, -120.4W (main_ocean_hfr_2km.py
# comments). Classify by final longitude relative to the middle-island
# landfall longitude band vs. the western/offshore branch.
BRANCH_LON_SPLIT = -120.30

FTLE_HOURS = 24
SUBSTEPS_HR = 6
SEED_UPSAMPLE = 4
RIDGE_PERCENTILE = 95

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
LAND_RGB = (0.30, 0.55, 0.25)


def _run_one(field, cluster, start_lat, start_lon):
    sx, sy = _latlon_to_world(start_lat, start_lon, field.config)
    cluster.reset(sx, sy)

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    if hasattr(field, "reset_clock"):
        field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(primitive)
        if hasattr(field, "step"):
            field.step(cluster.timestep * TIME_WARP)

    cx, cy = cluster.get_centroid()
    end_lat, end_lon = _world_to_latlon(cx, cy, field.config)
    return end_lat, end_lon


def main():
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)

    lat_offsets = np.linspace(-JITTER_DEG, JITTER_DEG, N_SIDE)
    lon_offsets = np.linspace(-JITTER_DEG, JITTER_DEG, N_SIDE)

    starts, ends, branches = [], [], []
    n_total = N_SIDE * N_SIDE
    k = 0
    for dlat in lat_offsets:
        for dlon in lon_offsets:
            k += 1
            slat, slon = CENTER_LAT + dlat, CENTER_LON + dlon
            elat, elon = _run_one(field, cluster, slat, slon)
            branch = "south/island" if elon >= BRANCH_LON_SPLIT else "west/offshore"
            starts.append((slat, slon))
            ends.append((elat, elon))
            branches.append(branch)
            if k % 10 == 0 or k == n_total:
                print(f"  {k}/{n_total}  start=({slat:.3f},{slon:.3f}) "
                      f"-> end=({elat:.3f},{elon:.3f})  [{branch}]")

    starts = np.array(starts)
    ends = np.array(ends)
    branches = np.array(branches)
    n_south = int((branches == "south/island").sum())
    print(f"\nBranch split: {n_south}/{n_total} south/island, "
          f"{n_total - n_south}/{n_total} west/offshore")

    print("\nComputing 24-h forward FTLE for ridge-distance metric...")
    lat_fine, lon_fine, f_val, land_fine, coast_polys, t_list = compute_ftle_field(
        DATA_DIR, FRAME_GLOB, COAST_SHP,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR, seed_upsample=SEED_UPSAMPLE,
    )
    ridge_thresh = np.nanpercentile(f_val, RIDGE_PERCENTILE)
    L, LA = np.meshgrid(lon_fine, lat_fine)
    ridge_mask = (f_val >= ridge_thresh) & (~land_fine)
    ridge_pts = np.column_stack([L[ridge_mask], LA[ridge_mask]])
    tree = cKDTree(ridge_pts)

    # Rough degrees->km conversion for the distance metric (local latitude).
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(CENTER_LAT))

    d_ridge_km = []
    for (slat, slon) in starts:
        pt = np.array([slon * km_per_deg_lon, slat * km_per_deg_lat])
        ridge_scaled = np.column_stack([ridge_pts[:, 0] * km_per_deg_lon,
                                         ridge_pts[:, 1] * km_per_deg_lat])
        d = np.min(np.linalg.norm(ridge_scaled - pt, axis=1))
        d_ridge_km.append(d)
    d_ridge_km = np.array(d_ridge_km)

    # ---- Figure: branch map + FTLE background --------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    f_plot = np.ma.array(f_val, mask=land_fine)
    vmax = float(np.nanpercentile(f_val, 99))
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)
    ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto", alpha=0.55)

    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

    colors = {"south/island": "#2a78d6", "west/offshore": "#e34948"}
    for branch, col in colors.items():
        sel = branches == branch
        ax.scatter(starts[sel, 1], starts[sel, 0], c=col, s=28,
                   edgecolor='black', linewidth=0.4, label=f"{branch} (n={sel.sum()})",
                   zorder=5)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_title(
        f"Landfall Branch by Start Position ({N_SIDE}x{N_SIDE} jittered starts)\n"
        f"mean dist. to {RIDGE_PERCENTILE}th-pct FTLE ridge: {d_ridge_km.mean():.1f} km",
        fontsize=11
    )
    fig.suptitle("Branch Sensitivity, 2 km Ocean HFR", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(OUT_DIR, "ocean_branch_sensitivity_2km.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    csv_path = os.path.join(OUT_DIR, "ocean_branch_sensitivity_2km.csv")
    with open(csv_path, "w") as f:
        f.write("start_lat,start_lon,end_lat,end_lon,branch,dist_ridge_km\n")
        for (slat, slon), (elat, elon), br, d in zip(starts, ends, branches, d_ridge_km):
            f.write(f"{slat:.4f},{slon:.4f},{elat:.4f},{elon:.4f},{br},{d:.3f}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
