"""
main_ocean_hfr_2km_shared_start.py

Paper figure: D tracker and s1 tracker sharing a single start point on the
2km Santa Barbara Channel HFR field, both run at the same fixed 168-step
(28h) budget. Both paths overlap almost the entire ride down the channel,
diverging only in the final stretch near the middle island.

Supersedes the D-tracker-only overlay in main_ocean_hfr_2km_ftle_overlay.py
as the paper's Figure 8: this start point (34.411906N, -120.392016W) was
found by a same-corridor Monte Carlo search around a manually-selected
starting region north of the channel entrance
(ocean_candidateA_finemc_v3.py and its predecessors), scored on how close
each controller's path comes to the ORIGINAL D-tracker-only figure's
endpoint (34.0412N, -120.2617W) at any point along its own 168-step run, not
just at the final step. At this start: D's closest approach to that
reference point is 3.60 km (at 95% of its run); s1's is 14.01 km (at 99%,
i.e. s1 is still closing the distance when the budget ends, not settling and
resuming). Full search provenance in ocean_candidateA_finemc_v3.py's
docstring and in Draft_5c.tex's paper-adjacent working notes.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/main_ocean_hfr_2km_shared_start.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from datetime import datetime, timezone

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

FORMATION_CONFIG  = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.04
SIM_STEPS    = 168
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA     = 0.0
STICTION_THRESHOLD = 0.002
TIME_WARP    = 6000.0

EPS_RAW, EPS_DIM = 1e-3, 0.025
G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.3, 0.05, 0.15

START = (34.411906, -120.392016)   # candidate A, this session's search

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7
FTLE_HOURS, SUBSTEPS_HR, SEED_UPSAMPLE = 24, 6, 4
LAND_RGB = (0.30, 0.55, 0.25)

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


def run_traj(field, cluster, prim, lat, lon):
    sx, sy = _latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)
    field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        field.step(cluster.timestep * TIME_WARP)
    center = cluster.get_center_history()
    return np.array([_world_to_latlon(p[0], p[1], field.config) for p in center])


def main():
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                              momentum_alpha=MOMENTUM_ALPHA,
                              stiction_threshold=STICTION_THRESHOLD)

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP, s_trim=S_TRIM,
                                      r_band=R_BAND, g_capture=G_CAPTURE, s_capture=None)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    print(f"Running D and s1 trackers from shared start ({START[0]:.4f}N, {-START[1]:.4f}W)...")
    d_path  = run_traj(field, cluster, prim_d,  *START)
    s1_path = run_traj(field, cluster, prim_s1, *START)

    print("Computing 24-h forward FTLE background (2km native)...")
    lat_f, lon_f, f_val, land_f, coast_polys, t_list = compute_ftle_field(
        DATA_DIR, FRAME_GLOB, COAST_SHP,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR, seed_upsample=SEED_UPSAMPLE,
    )
    f_plot = np.ma.array(f_val, mask=land_f)
    L, LA = np.meshgrid(lon_f, lat_f)
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    vmax = float(np.nanpercentile(f_val, 99))
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    ax.plot(d_path[:, 1], d_path[:, 0], color="black", linewidth=2.2,
            alpha=0.9, label="D tracker path")
    ax.plot(s1_path[:, 1], s1_path[:, 0], color="deepskyblue", linewidth=2.2,
            alpha=0.9, linestyle="--", label="$s_1$ tracker path")

    ax.plot(START[1], START[0], marker="*", color="lime", markersize=16,
            markeredgecolor="black", markeredgewidth=1.5,
            label="Shared start", zorder=11)
    ax.plot(d_path[-1, 1], d_path[-1, 0], marker="X", color="black",
            markersize=11, markeredgecolor="white", markeredgewidth=1.2,
            label="D end", zorder=11)
    ax.plot(s1_path[-1, 1], s1_path[-1, 0], marker="X", color="deepskyblue",
            markersize=11, markeredgecolor="black", markeredgewidth=1.2,
            label="$s_1$ end", zorder=11)

    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=0.6, alpha=0.85)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (deg)", fontsize=11)
    ax.set_ylabel("Latitude (deg)", fontsize=11)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)
    ax.set_title(
        f"{_ts(t_list[0])}  (TIME_WARP={TIME_WARP:.0f}x)\n"
        f"Shared start ({START[0]:.4f}N, {-START[1]:.4f}W)  [28.0 h of field time]",
        fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "ocean_shared_start_2km.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
