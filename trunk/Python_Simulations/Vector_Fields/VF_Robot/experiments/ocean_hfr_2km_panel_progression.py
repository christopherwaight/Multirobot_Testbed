"""
ocean_hfr_2km_panel_progression.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  fig:ocean_progression, a 1x4 grid of path-so-far over the
          instantaneous current field at ~7h intervals.

Runs the same validated trial as main_ocean_hfr_2km_ftle_overlay.py (same
field config, formation, control primitive, TIME_WARP, SIM_STEPS,
START_POINTS), recording the field clock at every step, then renders four
panels at approximately t = 0, 7, 14, 21, 28 h (five panels, matching the
28h trial in ~7h slices) showing the instantaneous current field (same
_draw_quiver pattern as main_ocean_hfr_2km.py) with the path traveled so
far overlaid. No FTLE computation here; that lives in
ocean_hfr_2km_ftle_snapshots.py and main_ocean_hfr_2km_ftle_overlay.py.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_hfr_2km_panel_progression.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import load_coastline_polygons

# ============================================================================
# CONFIGURATION -- identical to main_ocean_hfr_2km_ftle_overlay.py
# ============================================================================
FORMATION_CONFIG = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.04
SIM_STEPS    = 168     # 168 * 0.1s * TIME_WARP(6000) = 28h = full dataset window
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA      = 0.0
STICTION_THRESHOLD  = 0.002
EPS_RAW      = 1e-3
EPS_DIM      = 0.025
TIME_WARP    = 6000.0

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7

START_LAT, START_LON, START_LABEL = 34.4, -120.39, "furthest point"

QUIVER_STEP_DEG = 0.06
N_PANELS = 5            # t = 0, 7, 14, 21, 28 h
ROBOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
LAND_RGB = (0.30, 0.55, 0.25)

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
os.makedirs(OUT_DIR, exist_ok=True)


def _draw_quiver(ax, field, t_seconds):
    """Render the field as quiver arrows at a fixed field time t_seconds."""
    lons = np.arange(LON_MIN, LON_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    lats = np.arange(LAT_MIN, LAT_MAX + QUIVER_STEP_DEG, QUIVER_STEP_DEG)
    LON, LAT = np.meshgrid(lons, lats)
    U = np.zeros_like(LON)
    V = np.zeros_like(LAT)
    saved_t = field.t
    field.t = t_seconds
    for r in range(LON.shape[0]):
        for c in range(LON.shape[1]):
            wx, wy = _latlon_to_world(LAT[r, c], LON[r, c], field.config)
            u, v = field.get_value(wx, wy)
            U[r, c] = u
            V[r, c] = v
    field.t = saved_t
    ax.quiver(LON, LAT, U, V, color='black', alpha=0.55, scale=3.0)


def main():
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)

    sx, sy = _latlon_to_world(START_LAT, START_LON, field.config)
    cluster.reset(sx, sy)

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    if hasattr(field, "reset_clock"):
        field.reset_clock()

    t_history = [field.t]
    for _ in range(SIM_STEPS):
        cluster.move(primitive)
        if hasattr(field, "step"):
            field.step(cluster.timestep * TIME_WARP)
        t_history.append(field.t)
    t_history = np.array(t_history)

    center_hist = cluster.get_center_history()  # (SIM_STEPS+1, 2) world coords
    print(f"Trial done: {len(center_hist)} steps, "
          f"{t_history[-1] / 3600.0:.2f} h field time traversed.")

    # lat/lon centroid path, precomputed once
    clats, clons = [], []
    for pt in center_hist:
        clat, clon = _world_to_latlon(pt[0], pt[1], field.config)
        clats.append(clat)
        clons.append(clon)
    clats, clons = np.array(clats), np.array(clons)

    coast_polys = load_coastline_polygons(COAST_SHP, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    total_hours = t_history[-1] / 3600.0
    panel_hours = np.linspace(0.0, total_hours, N_PANELS)

    fig, axes = plt.subplots(1, N_PANELS, figsize=(4.0 * N_PANELS, 4.6), sharey=True)

    for k, (ax, h) in enumerate(zip(axes, panel_hours)):
        t_sec = h * 3600.0
        _draw_quiver(ax, field, t_sec)

        idx = int(np.searchsorted(t_history, t_sec))
        idx = min(idx, len(center_hist) - 1)

        ax.plot(clons[:idx + 1], clats[:idx + 1], color='black', linewidth=2.0,
                alpha=0.9, zorder=5)
        ax.plot(clons[0], clats[0], marker='*', color='lime', markersize=13,
                markeredgecolor='black', markeredgewidth=1.3, zorder=10)
        ax.plot(clons[idx], clats[idx], marker='o', color='red', markersize=8,
                markeredgecolor='black', markeredgewidth=1.2, zorder=10)

        for poly in coast_polys:
            ax.fill(poly[:, 0], poly[:, 1], color=LAND_RGB, zorder=1)
            ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6,
                    alpha=0.85, zorder=2)

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("equal")
        ax.set_title(f"t = {h:.0f} h", fontsize=11)
        ax.set_xlabel("Longitude (deg)", fontsize=9)
        if k == 0:
            ax.set_ylabel("Latitude (deg)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle(
        f"Pentagon Cluster Path over Instantaneous Current Field ({START_LABEL})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(OUT_DIR, "ocean_panel_progression_2km.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
