"""
ocean_shared_start_visualize_top2.py

Visualize the top 2 candidates from the closest-approach shared-start search
(ocean_shared_start_closest_approach.py). For each candidate start, runs BOTH
the D tracker and the s1 tracker (same fixed 168-step / 28h budget as every
other variant) and overlays both trajectories on the 24-h forward FTLE
background, styled like the paper's Fig. 8/9 overlays
(main_ocean_hfr_2km_ftle_overlay.py / main_ocean_hfr_2km_traverse.py).

One PNG per candidate, so the two can be viewed side by side. Also marks E0
(the current Figure 8 endpoint) and a 1km ring around it, since that was the
closest-approach search's target radius.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_shared_start_visualize_top2.py
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

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import compute_ftle_field

# ============================================================================
# Same operating point as the search / the paper's Fig 8/9 scripts
# ============================================================================
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

E0 = (34.0412, -120.2617)          # current Fig 8 endpoint (D tracker)
CLOSE_APPROACH_KM = 1.0

# candidate_A: earmarked shared-start result (session finding).
# candidate_B: current leader of the candidateA_finemc_v3 fine-MC search
# (4 km disk around candidate A, 1.5 km corridor cap vs candidate A's own
# paths, J = max closest-approach-to-E0 for D and s1), superseding an
# earlier, different point that used this label.
CANDIDATES = [
    {"label": "candidate_A", "lat": 34.411906, "lon": -120.392016,
     "note": "J=14.01 km, D_closest=3.60 km @ 95%, s1_closest=14.01 km @ 99%"},
    {"label": "candidate_B", "lat": 34.393070, "lon": -120.371297,
     "note": "J=12.73 km, D_closest=3.59 km @ 95%, s1_closest=12.73 km @ 100%"},
]

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


def run_traj(field, cluster, prim, lat, lon):
    sx, sy = _latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)
    field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        field.step(cluster.timestep * TIME_WARP)
    center = cluster.get_center_history()
    latlon = np.array([_world_to_latlon(p[0], p[1], field.config) for p in center])
    return latlon   # (steps, 2) as (lat, lon)


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

    print("Computing 24-h forward FTLE background (shared across both plots)...")
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

    theta = np.linspace(0, 2 * np.pi, 100)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(34.2))
    ring_lat = E0[0] + (CLOSE_APPROACH_KM / km_per_deg_lat) * np.sin(theta)
    ring_lon = E0[1] + (CLOSE_APPROACH_KM / km_per_deg_lon) * np.cos(theta)

    saved = []
    for cand in CANDIDATES:
        la, lo = cand["lat"], cand["lon"]
        print(f"\nRunning {cand['label']} from ({la:.5f}, {lo:.5f})...")
        d_path  = run_traj(field, cluster, prim_d,  la, lo)
        s1_path = run_traj(field, cluster, prim_s1, la, lo)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

        ax.plot(d_path[:, 1], d_path[:, 0], color="black", linewidth=2.2,
                alpha=0.9, label="D tracker path")
        ax.plot(s1_path[:, 1], s1_path[:, 0], color="deepskyblue", linewidth=2.2,
                alpha=0.9, label="s1 tracker path", linestyle="--")

        ax.plot(lo, la, marker="*", color="lime", markersize=16,
                markeredgecolor="black", markeredgewidth=1.5,
                label="Shared start", zorder=11)
        ax.plot(d_path[-1, 1], d_path[-1, 0], marker="X", color="black",
                markersize=11, markeredgecolor="white", markeredgewidth=1.2,
                label="D end", zorder=11)
        ax.plot(s1_path[-1, 1], s1_path[-1, 0], marker="X", color="deepskyblue",
                markersize=11, markeredgecolor="black", markeredgewidth=1.2,
                label="s1 end", zorder=11)
        ax.plot(E0[1], E0[0], marker="P", color="red", markersize=14,
                markeredgecolor="white", markeredgewidth=1.3,
                label="Fig. 8 endpoint (E0)", zorder=12)
        ax.plot(ring_lon, ring_lat, color="red", linewidth=1.2, linestyle=":",
                alpha=0.85, label=f"{CLOSE_APPROACH_KM:.0f} km ring around E0", zorder=10)

        for poly in coast_polys:
            ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=0.6, alpha=0.85)

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel("Longitude (deg)", fontsize=11)
        ax.set_ylabel("Latitude (deg)", fontsize=11)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)
        ax.set_title(f"Shared start ({la:.4f}N, {-lo:.4f}W)\n{cand['note']}", fontsize=10)
        fig.suptitle(f"Shared-Start Search: {cand['label']}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out_path = os.path.join(OUT_DIR, f"shared_start_{cand['label']}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
        saved.append(out_path)

    print("\nDone. Saved:")
    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
