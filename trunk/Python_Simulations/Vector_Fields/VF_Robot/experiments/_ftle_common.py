"""
_ftle_common.py

Shared FTLE (Finite Time Lyapunov Exponent) computation, extracted from
_generate_ocean_ftle_matched.py so both the static-reference-image generators
(_generate_ocean_ftle_matched.py, _generate_ocean_ftle_2km_matched.py) and the
trajectory-overlay script (main_ocean_hfr_2km_ftle_overlay.py) share one
implementation of the FTLE math instead of duplicating it.

Pipeline (forward-time FTLE, matching Serra and Haller 2014 Fig. 11 style):
  1. Load N+1 hourly HFR frames (u, v, lat, lon), crop to ROI + padding.
  2. Build a land mask from Natural Earth 1:10m land polygons.
  3. Gap-fill NaN ocean cells via linear griddata; leave land cells NaN.
  4. Crop to the exact display extent.
  5. Advect a refined particle-seed grid forward in time with RK4, using
     linear spatial interpolation and linear time interpolation between
     bracketing hourly frames.
  6. Build the Cauchy-Green deformation tensor from the flow map and take
     its largest eigenvalue; FTLE = log(sqrt(lambda_max)) / T.

Requires: pyshp (pip install pyshp) for coastline polygons.
"""
import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import netCDF4 as nc

FILL = -32767.0


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_frame(fp, lat_min, lat_max, lon_min, lon_max, pad=0.3):
    """Load one NetCDF frame, cropped to [lat_min-pad, lat_max+pad] x [lon_min-pad, lon_max+pad]."""
    with nc.Dataset(fp) as f:
        lat = np.array(f.variables["lat"][:], dtype=float)
        lon = np.array(f.variables["lon"][:], dtype=float)
        u   = np.array(f.variables["u"][0, :, :], dtype=float)
        v   = np.array(f.variables["v"][0, :, :], dtype=float)
        t   = float(f.variables["time"][0])
    u[np.abs(u - FILL) < 1.0] = np.nan
    v[np.abs(v - FILL) < 1.0] = np.nan
    lam = (lat >= lat_min - pad) & (lat <= lat_max + pad)
    lom = (lon >= lon_min - pad) & (lon <= lon_max + pad)
    return lat[lam], lon[lom], u[np.ix_(lam, lom)], v[np.ix_(lam, lom)], t


def fill_ocean_only(u, v, land_mask):
    """Linear gap-fill on ocean cells only; leave land_mask cells as NaN."""
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


def crop_stack(lat, lon, stack, lat_min, lat_max, lon_min, lon_max):
    """Crop a list of 2D arrays (all on the same lat/lon grid) to the display extent."""
    lam = (lat >= lat_min) & (lat <= lat_max)
    lom = (lon >= lon_min) & (lon <= lon_max)
    return lat[lam], lon[lom], [a[np.ix_(lam, lom)] for a in stack]


# ── Coastline helpers (Natural Earth 1:10m, via pyshp) ───────────────────────

def load_coastline_polygons(shp_path, lon_min, lon_max, lat_min, lat_max):
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


def build_land_mask(lat_grid, lon_grid, polys):
    """Return boolean array (len(lat_grid), len(lon_grid)) True where land."""
    from matplotlib.path import Path
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(pts)
    return mask.reshape(LON.shape)


# ── FTLE with linear-in-time interpolation ────────────────────────────────────

def ftle_linear_time(lat, lon, u_stack, v_stack, dt_hour=3600.0,
                      n_substeps_per_hour=6, seed_upsample=12):
    """
    Forward-time FTLE using:
      - Particle seed grid refined by `seed_upsample` versus the native HFR grid
        (gives sub-cell ridge resolution without touching the velocity field).
      - Bilinear spatial interpolation (RegularGridInterpolator, linear)
      - Linear temporal interpolation of (u, v) between bracketing hourly frames
      - RK4 with n_substeps_per_hour steps per HFR hour

    Returns (lat_fine, lon_fine, ftle) where ftle has shape
    (len(lat)*seed_upsample, len(lon)*seed_upsample).
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


# ── Orchestration ─────────────────────────────────────────────────────────────

def compute_ftle_field(data_dir, frame_glob, coast_shp,
                        lat_min, lat_max, lon_min, lon_max,
                        ftle_hours=24, substeps_hr=6, seed_upsample=12,
                        verbose=True, file_offset=0):
    """
    Load frames, build land mask, gap-fill, crop, and compute FTLE.

    file_offset: index of the first frame to load (0 = dataset start). Lets
        callers compute a forward-time FTLE snapshot anchored at any point
        in the record, not just the first ftle_hours+1 frames, e.g. a
        sequence of snapshots marching through the 28-h dataset.

    Returns:
        lat_fine, lon_fine: 1D coordinate arrays of the refined seed grid.
        ftle: 2D array (len(lat_fine), len(lon_fine)), unmasked FTLE values.
        land_fine_mask: boolean array, same shape as ftle, True where land.
        coast_polys: list of (N,2) lon/lat arrays, for drawing coastline outlines.
        t_list: list of frame Unix timestamps actually loaded (length ftle_hours+1).
    """
    files = sorted(glob.glob(f"{data_dir}/{frame_glob}"))
    if not files:
        raise FileNotFoundError(f"No files matching '{frame_glob}' in {data_dir}")

    files = files[file_offset:]
    n_load = min(ftle_hours + 1, len(files))  # need N+1 frames for N hours of integration
    if verbose:
        print(f"Loading {n_load} frames for {ftle_hours}-h forward FTLE "
              f"(offset {file_offset}, of {len(files)} available from there)...")

    raw_u, raw_v, t_list = [], [], []
    lat0 = lon0 = None
    for i, fp in enumerate(files[:n_load]):
        lat, lon, u, v, t = load_frame(fp, lat_min, lat_max, lon_min, lon_max)
        if i == 0:
            lat0, lon0 = lat, lon
        raw_u.append(u)
        raw_v.append(v)
        t_list.append(t)
        if verbose:
            print(f"  frame {i + 1}/{n_load}")

    if verbose:
        print("Building land mask from Natural Earth shapefile...")
    coast_polys = load_coastline_polygons(
        coast_shp,
        lon_min - 0.5, lon_max + 0.5,
        lat_min - 0.5, lat_max + 0.5,
    )
    land_mask = build_land_mask(lat0, lon0, coast_polys)
    if verbose:
        print(f"Land cells (native grid): {int(land_mask.sum())} / {land_mask.size} "
              f"({100 * land_mask.sum() / land_mask.size:.1f}%)")

    # Gap-fill ocean cells only.
    u_stack, v_stack = [], []
    for u, v in zip(raw_u, raw_v):
        uf, vf = fill_ocean_only(u, v, land_mask)
        u_stack.append(uf)
        v_stack.append(vf)

    # Crop to the display extent BEFORE FTLE so we don't waste integration on padding.
    lat_c, lon_c, u_stack_c = crop_stack(lat0, lon0, u_stack, lat_min, lat_max, lon_min, lon_max)
    _,     _,     v_stack_c = crop_stack(lat0, lon0, v_stack, lat_min, lat_max, lon_min, lon_max)

    if verbose:
        native_lat_deg = float(np.mean(np.diff(lat_c))) if len(lat_c) > 1 else 0.0
        seed_lat_deg = native_lat_deg / seed_upsample
        print(f"Computing FTLE (linear in space, linear in time, "
              f"RK4 {substeps_hr} substeps/hr, seed grid x{seed_upsample}, "
              f"effective seed spacing ~{seed_lat_deg:.5f} deg lat)...")

    lat_fine, lon_fine, ftle = ftle_linear_time(
        lat_c, lon_c, u_stack_c, v_stack_c,
        n_substeps_per_hour=substeps_hr, seed_upsample=seed_upsample,
    )
    if verbose:
        print(f"  FTLE 50/95/99 pct: "
              f"{np.nanpercentile(ftle, 50):.2e}, "
              f"{np.nanpercentile(ftle, 95):.2e}, "
              f"{np.nanpercentile(ftle, 99):.2e}")
        print("Building fine-grid land mask (this takes a few seconds)...")

    land_fine = build_land_mask(lat_fine, lon_fine, coast_polys)
    if verbose:
        print(f"Land cells (fine grid): {int(land_fine.sum())} / {land_fine.size} "
              f"({100 * land_fine.sum() / land_fine.size:.1f}%)")

    return lat_fine, lon_fine, ftle, land_fine, coast_polys, t_list
