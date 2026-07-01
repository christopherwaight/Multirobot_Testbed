"""
_ftle_grid_sweep.py

Parameter sweep to identify the best FTLE rendering settings.
Produces a 3x3 grid figure (one PNG) with:
  rows    = SEED_UPSAMPLE in [8, 12, 16]
  columns = vmax percentile in [90, 95, 99]

All other parameters fixed at the matched-pipeline defaults:
  FTLE_HOURS=24, SUBSTEPS_HR=6

Output: ocean_data/det_jacobian_plots/ftle_grid_sweep.png

Run from the VF_Robot root:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/_ftle_grid_sweep.py
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.interpolate import RegularGridInterpolator, griddata
import netCDF4 as nc

# ── Fixed config ──────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7

FTLE_HOURS  = 24
SUBSTEPS_HR = 6
FILL        = -32767.0
LAND_RGB    = (0.30, 0.55, 0.25)

UPSAMPLE_VALS  = [12]        # fixed
VMAX_PCT_VALS  = [95, 99]   # columns
# colormaps to test across rows
CMAP_NAMES     = ["jet", "turbo", "michini", "hot"]

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
DATA_DIR  = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "hfr_uswc_2012may")
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
os.makedirs(OUT_DIR, exist_ok=True)

# ── I/O helpers (identical to matched generator) ──────────────────────────────

def _load_frame(fp):
    with nc.Dataset(fp) as f:
        lat = np.array(f.variables["lat"][:], dtype=float)
        lon = np.array(f.variables["lon"][:], dtype=float)
        u   = np.array(f.variables["u"][0, :, :], dtype=float)
        v   = np.array(f.variables["v"][0, :, :], dtype=float)
        t   = float(f.variables["time"][0])
    u[np.abs(u - FILL) < 1.0] = np.nan
    v[np.abs(v - FILL) < 1.0] = np.nan
    pad = 0.3
    lam = (lat >= LAT_MIN - pad) & (lat <= LAT_MAX + pad)
    lom = (lon >= LON_MIN - pad) & (lon <= LON_MAX + pad)
    return lat[lam], lon[lom], u[np.ix_(lam, lom)], v[np.ix_(lam, lom)], t


def _fill_ocean_only(u, v, land_mask):
    nlat, nlon = u.shape
    LON, LAT = np.meshgrid(np.arange(nlon), np.arange(nlat))
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    uf, vf = u.ravel(), v.ravel()
    lm = land_mask.ravel()
    ok = np.isfinite(uf) & np.isfinite(vf) & (~lm)
    if ok.sum() < 10:
        return np.where(land_mask, np.nan, np.nan_to_num(u, nan=0.0)), \
               np.where(land_mask, np.nan, np.nan_to_num(v, nan=0.0))
    uf2 = griddata(pts[ok], uf[ok], pts, method="linear").reshape(u.shape)
    vf2 = griddata(pts[ok], vf[ok], pts, method="linear").reshape(v.shape)
    u_out = np.where(land_mask, np.nan, np.where(np.isfinite(uf2), uf2, 0.0))
    v_out = np.where(land_mask, np.nan, np.where(np.isfinite(vf2), vf2, 0.0))
    return u_out, v_out


def _crop_stack(lat, lon, stack):
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    return lat[lam], lon[lom], [a[np.ix_(lam, lom)] for a in stack]


def _load_coastline_polygons(shp_path, lon_min, lon_max, lat_min, lat_max):
    import shapefile
    sf = shapefile.Reader(shp_path)
    polys = []
    for shape in sf.shapes():
        pts = np.asarray(shape.points)
        if len(pts) < 3:
            continue
        if (pts[:, 0].max() < lon_min or pts[:, 0].min() > lon_max or
                pts[:, 1].max() < lat_min or pts[:, 1].min() > lat_max):
            continue
        starts = list(shape.parts) + [len(pts)]
        for a, b in zip(starts, starts[1:]):
            ring = pts[a:b]
            if len(ring) >= 3:
                polys.append(ring)
    return polys


def _build_land_mask(lat_grid, lon_grid, polys):
    from matplotlib.path import Path
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(pts)
    return mask.reshape(LON.shape)


# ── FTLE kernel (parameterised) ───────────────────────────────────────────────

def _ftle_linear_time(lat, lon, u_stack, v_stack,
                      dt_hour=3600.0,
                      n_substeps_per_hour=SUBSTEPS_HR,
                      seed_upsample=12):
    R_lat = 111320.0
    R_lon = 111320.0 * np.cos(np.radians(lat.mean()))

    lat_fine = np.linspace(lat.min(), lat.max(), len(lat) * seed_upsample)
    lon_fine = np.linspace(lon.min(), lon.max(), len(lon) * seed_upsample)
    LON0, LAT0 = np.meshgrid(lon_fine, lat_fine)
    x, y = LON0.copy(), LAT0.copy()
    sh = x.shape

    n_frames = len(u_stack)
    u_deg_stack = [u / R_lon for u in u_stack]
    v_deg_stack = [v / R_lat for v in v_stack]

    u_interp = [RegularGridInterpolator(
                    (lat, lon), np.nan_to_num(uk, nan=0.0),
                    method="linear", bounds_error=False, fill_value=0.0)
                for uk in u_deg_stack]
    v_interp = [RegularGridInterpolator(
                    (lat, lon), np.nan_to_num(vk, nan=0.0),
                    method="linear", bounds_error=False, fill_value=0.0)
                for vk in v_deg_stack]

    def sample(t_hr, px, py):
        i = int(np.floor(t_hr))
        i = max(0, min(i, n_frames - 2))
        alpha = float(np.clip(t_hr - i, 0.0, 1.0))
        pts = np.column_stack([
            np.clip(py, lat.min(), lat.max()).ravel(),
            np.clip(px, lon.min(), lon.max()).ravel(),
        ])
        u0 = u_interp[i](pts).reshape(sh)
        v0 = v_interp[i](pts).reshape(sh)
        u1 = u_interp[i + 1](pts).reshape(sh)
        v1 = v_interp[i + 1](pts).reshape(sh)
        return (1.0 - alpha) * u0 + alpha * u1, (1.0 - alpha) * v0 + alpha * v1

    dt_sub = dt_hour / n_substeps_per_hour
    total_substeps = (n_frames - 1) * n_substeps_per_hour
    for s in range(total_substeps):
        t0     = s / n_substeps_per_hour
        t_half = (s + 0.5) / n_substeps_per_hour
        t1     = (s + 1) / n_substeps_per_hour
        k1u, k1v = sample(t0,     x,                       y)
        k2u, k2v = sample(t_half, x + 0.5 * dt_sub * k1u,  y + 0.5 * dt_sub * k1v)
        k3u, k3v = sample(t_half, x + 0.5 * dt_sub * k2u,  y + 0.5 * dt_sub * k2v)
        k4u, k4v = sample(t1,     x + dt_sub * k3u,         y + dt_sub * k3v)
        x += dt_sub / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        y += dt_sub / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)

    F11 = np.gradient(x, lon_fine, axis=1)
    F12 = np.gradient(x, lat_fine, axis=0) * R_lat / R_lon
    F21 = np.gradient(y, lon_fine, axis=1) * R_lon / R_lat
    F22 = np.gradient(y, lat_fine, axis=0)
    C11 = F11 ** 2 + F21 ** 2
    C12 = F11 * F12 + F21 * F22
    C22 = F12 ** 2 + F22 ** 2
    tr   = C11 + C22
    disc = np.maximum((tr / 2.0) ** 2 - (C11 * C22 - C12 ** 2), 0.0)
    lam_max = tr / 2.0 + np.sqrt(disc)
    total_T = total_substeps * dt_sub
    ftle = np.log(np.sqrt(np.maximum(lam_max, 1e-12))) / total_T
    return lat_fine, lon_fine, ftle


# ── Load data once ────────────────────────────────────────────────────────────

print("Loading HFR data...")
files   = sorted(glob.glob(os.path.join(DATA_DIR, "*_6km_rtv_uwls_SIO.nc")))
n_load  = min(FTLE_HOURS + 1, len(files))
raw_u, raw_v, t_list = [], [], []
lat0 = lon0 = None
for i, fp in enumerate(files[:n_load]):
    lat, lon, u, v, t = _load_frame(fp)
    if i == 0:
        lat0, lon0 = lat, lon
    raw_u.append(u)
    raw_v.append(v)
    t_list.append(t)
    print(f"  frame {i+1}/{n_load}")

print("Building land mask (native grid)...")
coast_polys = _load_coastline_polygons(
    COAST_SHP,
    LON_MIN - 0.5, LON_MAX + 0.5, LAT_MIN - 0.5, LAT_MAX + 0.5,
)
land_mask = _build_land_mask(lat0, lon0, coast_polys)

print("Gap-filling ocean cells...")
u_stack, v_stack = [], []
for u, v in zip(raw_u, raw_v):
    uf, vf = _fill_ocean_only(u, v, land_mask)
    u_stack.append(uf)
    v_stack.append(vf)

lat_c, lon_c, u_stack_c = _crop_stack(lat0, lon0, u_stack)
_,     _,     v_stack_c = _crop_stack(lat0, lon0, v_stack)

# ── FTLE per upsample value (compute once per upsample, reuse for all vmax) ──

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap matching Michini Fig 10(b): black -> blue -> cyan -> white -> yellow -> red
_michini_colors = [
    (0.00, (0.00, 0.00, 0.00)),   # black   (zero FTLE)
    (0.15, (0.00, 0.00, 0.70)),   # blue
    (0.35, (0.00, 0.70, 1.00)),   # cyan
    (0.55, (1.00, 1.00, 1.00)),   # white   (ridge halo)
    (0.75, (1.00, 1.00, 0.00)),   # yellow
    (1.00, (1.00, 0.00, 0.00)),   # red     (ridge crest)
]
michini_cmap = LinearSegmentedColormap.from_list(
    "michini",
    [(v, c) for v, c in _michini_colors]
)

def _make_cmap(name):
    if name == "michini":
        cmap = michini_cmap.copy()
    else:
        cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=LAND_RGB)
    return cmap

print("\nComputing FTLE with seed_upsample=12...")
lat_fine, lon_fine, ftle = _ftle_linear_time(
    lat_c, lon_c, u_stack_c, v_stack_c, seed_upsample=12
)
print(f"  FTLE 90/95/99 pct: "
      f"{np.nanpercentile(ftle, 90):.2e}, "
      f"{np.nanpercentile(ftle, 95):.2e}, "
      f"{np.nanpercentile(ftle, 99):.2e}")
print("  Building fine land mask...")
land_fine = _build_land_mask(lat_fine, lon_fine, coast_polys)
f_plot = np.ma.array(ftle, mask=land_fine)
L, LA  = np.meshgrid(lon_fine, lat_fine)

nrows = len(CMAP_NAMES)
ncols = len(VMAX_PCT_VALS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

for row, cname in enumerate(CMAP_NAMES):
    cmap = _make_cmap(cname)
    for col, pct in enumerate(VMAX_PCT_VALS):
        ax   = axes[row, col]
        vmax = float(np.nanpercentile(ftle, pct))
        ax.pcolormesh(L, LA, f_plot, cmap=cmap, vmin=0.0, vmax=vmax, shading="auto")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        for poly in coast_polys:
            ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.5, alpha=0.85)
        label = f"cmap={cname}  vmax={pct}pct  ({vmax:.1e})"
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Lat", fontsize=8)
        if row == nrows - 1:
            ax.set_xlabel("Lon", fontsize=8)

fig.suptitle(
    "FTLE parameter sweep -- rows: colormap, cols: vmax percentile\n"
    "Santa Barbara Channel, May 16 2012 08:00 UTC, 24-h forward  (seed_upsample=12)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ftle_grid_sweep.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
