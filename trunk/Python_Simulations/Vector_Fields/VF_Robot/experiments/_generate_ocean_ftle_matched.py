"""
_generate_ocean_ftle_matched.py

Sibling of _generate_ocean_ftle_reference.py that aims to match the FTLE
rendering in Serra and Haller, "Robotic tracking of coherent structures"
(2014, Fig. 11, Santa Barbara Channel, May 16 2012 08:00 UTC).

Differences from the original generator:
  1. Linear (not cubic-spline) spatial interpolation of u, v.
  2. No Gaussian pre-smoothing of u, v (filaments stay sharp).
  3. Linear time interpolation between hourly frames inside the RK4
     loop (the original held each frame constant for the full hour).
  4. Land mask from Natural Earth 1:10m land polygons (ne_10m_land.shp),
     rendered as smooth green fill with a black coastline outline.
  5. 95th-percentile FTLE color cap so only ridge crests hit deep red.
  6. Finer RK4 substepping (default 6 substeps per hour) and a refined
     particle-seed grid (SEED_UPSAMPLE x native resolution).

Both this script and the original write into the same output directory
but to different filenames:
  Original: ocean_data/det_jacobian_plots/ftle_frame0.png
  Matched : ocean_data/det_jacobian_plots/ftle_frame0_matched.png

Requires: pyshp  (pip install pyshp -- pure Python, no GEOS/PROJ needed)
Shapefile: ocean_data/coastlines/ne_10m_land/ne_10m_land.shp
           (Natural Earth 1:10m, CC0, see ocean_data/coastlines/README.md)

Run from the VF_Robot root:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/_generate_ocean_ftle_matched.py
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.interpolate import RegularGridInterpolator, griddata
import netCDF4 as nc

# ── Config ────────────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7

FTLE_HOURS    = 24       # forward integration window
SUBSTEPS_HR   = 6        # RK4 substeps per HFR hour (dt = 600 s)
SEED_UPSAMPLE = 12       # particle-seed grid refinement vs. native 6 km grid
FILL          = -32767.0

# Color cap. None = auto (95th percentile). Set to a float to fix across runs.
FTLE_VMAX     = None

# Land colour (Serra/Haller use a flat green).
LAND_RGB      = (0.30, 0.55, 0.25)

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
DATA_DIR  = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "hfr_uswc_2012may")
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
os.makedirs(OUT_DIR, exist_ok=True)

# ── I/O ───────────────────────────────────────────────────────────────────────

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


def _ts(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d %Y  %H:%M UTC")


def _fill_ocean_only(u, v, land_mask):
    """
    Linear gap-fill on ocean cells only; leave land_mask cells as NaN.
    """
    nlat, nlon = u.shape
    LON, LAT = np.meshgrid(np.arange(nlon), np.arange(nlat))
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    uf, vf = u.ravel(), v.ravel()
    lm     = land_mask.ravel()
    ok     = np.isfinite(uf) & np.isfinite(vf) & (~lm)
    if ok.sum() < 10:
        return np.where(land_mask, np.nan, np.nan_to_num(u, nan=0.0)), \
               np.where(land_mask, np.nan, np.nan_to_num(v, nan=0.0))
    uf2 = griddata(pts[ok], uf[ok], pts, method="linear").reshape(u.shape)
    vf2 = griddata(pts[ok], vf[ok], pts, method="linear").reshape(v.shape)
    u_out = np.where(land_mask, np.nan, np.where(np.isfinite(uf2), uf2, 0.0))
    v_out = np.where(land_mask, np.nan, np.where(np.isfinite(vf2), vf2, 0.0))
    return u_out, v_out


def _crop(lat, lon, *fields):
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    out = [lat[lam], lon[lom]]
    for f in fields:
        out.append(f[np.ix_(lam, lom)])
    return tuple(out)


# ── Coastline helpers (Natural Earth 1:10m, via pyshp) ───────────────────────

def _load_coastline_polygons(shp_path, lon_min, lon_max, lat_min, lat_max):
    """Read ne_10m_land.shp; return list of (N,2) lon/lat arrays inside the bbox."""
    import shapefile
    sf = shapefile.Reader(shp_path)
    polys = []
    for shape in sf.shapes():
        pts = np.asarray(shape.points)
        if len(pts) < 3:
            continue
        # quick bbox cull — skip polygons entirely outside the ROI
        if (pts[:, 0].max() < lon_min or pts[:, 0].min() > lon_max or
                pts[:, 1].max() < lat_min or pts[:, 1].min() > lat_max):
            continue
        # split multi-ring polygons on shapefile part boundaries
        starts = list(shape.parts) + [len(pts)]
        for a, b in zip(starts, starts[1:]):
            ring = pts[a:b]
            if len(ring) >= 3:
                polys.append(ring)
    return polys


def _build_land_mask(lat_grid, lon_grid, polys):
    """Return boolean array (len(lat_grid), len(lon_grid)) True where land."""
    from matplotlib.path import Path
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(pts)
    return mask.reshape(LON.shape)


# ── FTLE with linear-in-time interpolation ────────────────────────────────────

def _ftle_linear_time(lat, lon, u_stack, v_stack, dt_hour=3600.0,
                       n_substeps_per_hour=SUBSTEPS_HR,
                       seed_upsample=SEED_UPSAMPLE):
    """
    Forward-time FTLE using:
      - Particle seed grid refined by `seed_upsample` versus the native HFR grid
        (gives sub-cell ridge resolution without touching the velocity field).
      - Bilinear spatial interpolation (RegularGridInterpolator, linear)
      - Linear temporal interpolation of (u, v) between bracketing hourly frames
      - RK4 with n_substeps_per_hour steps per HFR hour
    """
    R_lat = 111320.0
    R_lon = 111320.0 * np.cos(np.radians(lat.mean()))

    # Refined seed grid (where we launch trajectories from and compute FTLE on).
    lat_fine = np.linspace(lat.min(), lat.max(), len(lat) * seed_upsample)
    lon_fine = np.linspace(lon.min(), lon.max(), len(lon) * seed_upsample)
    LON0, LAT0 = np.meshgrid(lon_fine, lat_fine)
    x, y = LON0.copy(), LAT0.copy()
    sh = x.shape

    n_frames = len(u_stack)
    # Convert each frame's u, v to deg/s up front (units to match the position step).
    u_deg_stack = [u / R_lon for u in u_stack]
    v_deg_stack = [v / R_lat for v in v_stack]

    # Spatial interpolators per frame, fill_value=0 for off-grid / land.
    u_interp = [RegularGridInterpolator(
                    (lat, lon), np.nan_to_num(uk, nan=0.0),
                    method="linear", bounds_error=False, fill_value=0.0)
                for uk in u_deg_stack]
    v_interp = [RegularGridInterpolator(
                    (lat, lon), np.nan_to_num(vk, nan=0.0),
                    method="linear", bounds_error=False, fill_value=0.0)
                for vk in v_deg_stack]

    def sample(t_hr, px, py):
        """
        Sample (u, v) in deg/s at fractional hour t_hr (in [0, n_frames-1]).
        Linear-in-time blend between bracketing frames.
        """
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
        t0 = s / n_substeps_per_hour
        t_half = (s + 0.5) / n_substeps_per_hour
        t1 = (s + 1) / n_substeps_per_hour
        k1u, k1v = sample(t0,     x,                       y)
        k2u, k2v = sample(t_half, x + 0.5 * dt_sub * k1u,  y + 0.5 * dt_sub * k1v)
        k3u, k3v = sample(t_half, x + 0.5 * dt_sub * k2u,  y + 0.5 * dt_sub * k2v)
        k4u, k4v = sample(t1,     x + dt_sub * k3u,        y + dt_sub * k3v)
        x += dt_sub / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        y += dt_sub / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)

    # Cauchy-Green tensor and largest eigenvalue (on refined seed grid).
    F11 = np.gradient(x, lon_fine, axis=1)
    F12 = np.gradient(x, lat_fine, axis=0) * R_lat / R_lon
    F21 = np.gradient(y, lon_fine, axis=1) * R_lon / R_lat
    F22 = np.gradient(y, lat_fine, axis=0)
    C11 = F11 ** 2 + F21 ** 2
    C12 = F11 * F12 + F21 * F22
    C22 = F12 ** 2 + F22 ** 2
    tr  = C11 + C22
    disc = np.maximum((tr / 2.0) ** 2 - (C11 * C22 - C12 ** 2), 0.0)
    lam_max = tr / 2.0 + np.sqrt(disc)
    total_T = total_substeps * dt_sub
    ftle = np.log(np.sqrt(np.maximum(lam_max, 1e-12))) / total_T
    return lat_fine, lon_fine, ftle


# ── Load ──────────────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_6km_rtv_uwls_SIO.nc")))
if not files:
    raise FileNotFoundError(f"No 6km files found in {DATA_DIR}")

n_load = min(FTLE_HOURS + 1, len(files))   # need N+1 frames for N hours of integration
print(f"Loading {n_load} frames for {FTLE_HOURS}-h forward FTLE (of {len(files)} available)...")

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

# Land mask: derived from Natural Earth 1:10m coastline shapefile.
print("Building land mask from Natural Earth shapefile...")
coast_polys = _load_coastline_polygons(
    COAST_SHP,
    LON_MIN - 0.5, LON_MAX + 0.5,
    LAT_MIN - 0.5, LAT_MAX + 0.5,
)
land_mask = _build_land_mask(lat0, lon0, coast_polys)
print(f"Land cells (native grid): {int(land_mask.sum())} / {land_mask.size} "
      f"({100*land_mask.sum()/land_mask.size:.1f}%)")

# Gap-fill ocean cells only.
u_stack, v_stack = [], []
for u, v in zip(raw_u, raw_v):
    uf, vf = _fill_ocean_only(u, v, land_mask)
    u_stack.append(uf)
    v_stack.append(vf)

# Crop to the display extent BEFORE FTLE so we don't waste integration on padding.
def _crop_stack(lat, lon, stack):
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    return lat[lam], lon[lom], [a[np.ix_(lam, lom)] for a in stack]

lat_c, lon_c, u_stack_c = _crop_stack(lat0, lon0, u_stack)
_,     _,     v_stack_c = _crop_stack(lat0, lon0, v_stack)
land_c = land_mask[np.ix_((lat0 >= LAT_MIN) & (lat0 <= LAT_MAX),
                           (lon0 >= LON_MIN) & (lon0 <= LON_MAX))]

print("Computing FTLE (linear in space, linear in time, RK4 6 substeps/hr, "
      f"seed grid x{SEED_UPSAMPLE})...")
lat_fine, lon_fine, f_val = _ftle_linear_time(lat_c, lon_c, u_stack_c, v_stack_c)
print(f"  FTLE 50/95/99 pct: "
      f"{np.nanpercentile(f_val, 50):.2e}, "
      f"{np.nanpercentile(f_val, 95):.2e}, "
      f"{np.nanpercentile(f_val, 99):.2e}")

# Build land mask directly on the refined seed grid for smooth coastline edges.
print("Building fine-grid land mask (this takes a few seconds)...")
land_fine = _build_land_mask(lat_fine, lon_fine, coast_polys)
print(f"Land cells (fine grid): {int(land_fine.sum())} / {land_fine.size} "
      f"({100*land_fine.sum()/land_fine.size:.1f}%)")

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
    f"{_ts(t_list[0])}  (matched pipeline)\n"
    "Ridges = repelling LCS / separatrices",
    fontsize=10
)
plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)

# Coastline outline on top of the green fill (matches Serra/Haller style).
for poly in coast_polys:
    ax.plot(poly[:, 0], poly[:, 1], color='black', linewidth=0.6, alpha=0.85)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ftle_frame0_matched.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
