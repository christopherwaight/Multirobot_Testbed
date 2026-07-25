"""
ocean_hfr_2km_ftle_snapshots.py


  cd "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot"
  MPLBACKEND=Agg ./venv/bin/python3 experiments/ocean_hfr_2km_ftle_snapshots.py


PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  fig:ocean_ftle_snapshots, FTLE computed at four anchor times
          across the 28-h dataset, showing the ridge structure evolving.

The 29-hour record only supports a limited number of independent forward
FTLE windows (compute_ftle_field needs ftle_hours+1 frames). Using a 12-h
forward horizon (13 frames) instead of the 24-h horizon used elsewhere
allows four anchor offsets spread across the record: 0, 5, 10, 16 h,
each showing the 12-h-forward ridge structure from that starting time.
Reuses _ftle_common.compute_ftle_field with the new file_offset argument;
no new FTLE math.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_hfr_2km_ftle_snapshots.py
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ftle_common import compute_ftle_field

LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX =   33.8,   34.7

FTLE_HOURS = 12
SUBSTEPS_HR = 6
SEED_UPSAMPLE = 4
ANCHOR_HOURS = [0, 5, 10, 16]   # offsets in whole hours (= file index)
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


def main():
    fig, axes = plt.subplots(1, len(ANCHOR_HOURS), figsize=(4.4 * len(ANCHOR_HOURS), 4.8),
                              sharey=True)

    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)

    for ax, offset_h in zip(axes, ANCHOR_HOURS):
        print(f"\n--- Anchor t = {offset_h} h (file offset {offset_h}) ---")
        lat_fine, lon_fine, f_val, land_fine, coast_polys, t_list = compute_ftle_field(
            DATA_DIR, FRAME_GLOB, COAST_SHP,
            LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
            ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR,
            seed_upsample=SEED_UPSAMPLE, file_offset=offset_h,
        )
        f_plot = np.ma.array(f_val, mask=land_fine)
        L, LA = np.meshgrid(lon_fine, lat_fine)
        vmax = float(np.nanpercentile(f_val, 99))
        norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

        im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")
        for poly in coast_polys:
            ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("equal")
        ax.set_title(f"anchor t = {offset_h} h\n({FTLE_HOURS}-h forward)", fontsize=11)
        ax.set_xlabel("Longitude (deg)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    axes[0].set_ylabel("Latitude (deg)", fontsize=9)
    fig.colorbar(im, ax=axes, label=r"FTLE [s$^{-1}$]", shrink=0.75, pad=0.01)

    fig.suptitle(
        f"Forward-Time FTLE Evolution ({FTLE_HOURS} h window), 2 km",
        fontsize=13, fontweight='bold'
    )

    out_path = os.path.join(OUT_DIR, "ocean_ftle_snapshots_2km.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
