"""
_generate_ocean_ftle_2km_matched.py

Same pipeline as _generate_ocean_ftle_matched.py (see that file's docstring
for the full rendering-style rationale), but sourced from the 2km NATIVE
resolution HFR dataset instead of the 6km product. No spatial upsampling of
the base velocity grid is applied (unlike the 6km path, which upsamples 6x
before computing anything) -- the 2km grid is already fine enough to use
as-is, following the precedent in ocean_data/plot_det_jacobian.py
(process_frame_2km uses the native grid directly).

SEED_UPSAMPLE is reduced from 12 (used for 6km) to 4 for 2km: the 6km grid
at 12x upsample reaches an effective seed spacing of ~0.5 km (6/12); the 2km
grid at 4x upsample reaches the same ~0.5 km effective seed spacing (2/4),
so the two FTLE fields are rendered at comparable effective resolution
without paying for an unnecessarily huge seed grid.

SBC valid-data coverage at 2km is lower (~48-58%) than 6km (~67%), so expect
more gap-filled patches. See ocean_data/DATA_PROVENANCE.md.

Requires: pyshp  (pip install pyshp -- pure Python, no GEOS/PROJ needed)
Shapefile: ocean_data/coastlines/ne_10m_land/ne_10m_land.shp

Run from the VF_Robot root:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/_generate_ocean_ftle_2km_matched.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ftle_common import compute_ftle_field

# ── Config ────────────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7

FTLE_HOURS    = 24       # forward integration window
SUBSTEPS_HR   = 6        # RK4 substeps per HFR hour (dt = 600 s)
SEED_UPSAMPLE = 4        # 2km / 4 ~= 6km / 12 effective seed spacing (~0.5 km)

# Color cap. None = auto (95th percentile). Set to a float to fix across runs.
FTLE_VMAX     = None

# Land colour (Serra/Haller use a flat green).
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


# ── Compute ───────────────────────────────────────────────────────────────────

lat_fine, lon_fine, f_val, land_fine, coast_polys, t_list = compute_ftle_field(
    DATA_DIR, FRAME_GLOB, COAST_SHP,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR, seed_upsample=SEED_UPSAMPLE,
)

f_plot = np.ma.array(f_val, mask=land_fine)

# ── Plot ──────────────────────────────────────────────────────────────────────

L, LA = np.meshgrid(lon_fine, lat_fine)

from matplotlib.colors import PowerNorm

jet = plt.get_cmap("jet").copy()
jet.set_bad(color=LAND_RGB)   # masked (land) cells render as green

vmax = float(np.nanpercentile(f_val, 99)) if FTLE_VMAX is None else float(FTLE_VMAX)
print(f"  color cap (vmax): {vmax:.2e}")

norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm,
                   shading="auto")
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude (deg)", fontsize=11)
ax.set_ylabel("Latitude (deg)",  fontsize=11)
ax.set_title(
    f"Forward-time FTLE ({FTLE_HOURS} h) -- Santa Barbara Channel\n"
    f"{_ts(t_list[0])}  (2km native, matched pipeline)\n"
    "Ridges = repelling LCS / separatrices",
    fontsize=10
)
plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)

# Coastline outline on top of the green fill (matches Serra/Haller style).
for poly in coast_polys:
    ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ftle_frame0_2km_matched.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
