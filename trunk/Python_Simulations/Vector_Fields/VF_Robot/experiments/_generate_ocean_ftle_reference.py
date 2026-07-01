"""
_generate_ocean_ftle_reference.py

One-shot helper: compute 24-h forward-time FTLE from the HF radar dataset
(Santa Barbara Channel, first frame = 2012-05-16 08:00 UTC) and save a
single-panel PNG for use as a reference image in main_ocean_hfr_v6r.py.

Run once from the VF_Robot root:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    source venv/bin/activate
    python experiments/_generate_ocean_ftle_reference.py

Output: trunk/Python_Simulations/Vector_Fields/ocean_data/det_jacobian_plots/ftle_frame0.png
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
import netCDF4 as nc

# ── Config ────────────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7
UPSAMPLE      = 6
SMOOTH_SIGMA  = 2.0
FTLE_HOURS    = 24
FILL          = -32767.0

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
DATA_DIR  = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "hfr_uswc_2012may")
OUT_DIR   = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "det_jacobian_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── I/O helpers (from plot_det_jacobian.py) ───────────────────────────────────

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


def _gap_fill(lat, lon, u, v):
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    uf, vf = u.ravel(), v.ravel()
    ok = np.isfinite(uf) & np.isfinite(vf)
    if ok.sum() < 10:
        return np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)
    uf2 = griddata(pts[ok], uf[ok], pts, method="linear").reshape(u.shape)
    vf2 = griddata(pts[ok], vf[ok], pts, method="linear").reshape(v.shape)
    return (np.where(np.isfinite(uf2), uf2, 0.0),
            np.where(np.isfinite(vf2), vf2, 0.0))


def _upsample_smooth(lat, lon, u, v):
    lat_f = np.linspace(lat.min(), lat.max(), len(lat) * UPSAMPLE)
    lon_f = np.linspace(lon.min(), lon.max(), len(lon) * UPSAMPLE)
    u_f = gaussian_filter(RectBivariateSpline(lat, lon, u)(lat_f, lon_f), SMOOTH_SIGMA)
    v_f = gaussian_filter(RectBivariateSpline(lat, lon, v)(lat_f, lon_f), SMOOTH_SIGMA)
    return lat_f, lon_f, u_f, v_f


def _crop(lat, lon, u, v):
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    return lat[lam], lon[lom], u[np.ix_(lam, lom)], v[np.ix_(lam, lom)]


def _ftle(lat, lon, u_list, v_list, dt=3600.0):
    R_lat = 111320.0
    R_lon = 111320.0 * np.cos(np.radians(lat.mean()))
    LON0, LAT0 = np.meshgrid(lon, lat)
    x, y = LON0.copy(), LAT0.copy()
    sh = x.shape

    def adv(u, v, px, py):
        fn_u = RegularGridInterpolator((lat, lon), u / R_lon,
                                       method="linear", bounds_error=False, fill_value=0.0)
        fn_v = RegularGridInterpolator((lat, lon), v / R_lat,
                                       method="linear", bounds_error=False, fill_value=0.0)
        pts = np.column_stack([np.clip(py, lat.min(), lat.max()).ravel(),
                               np.clip(px, lon.min(), lon.max()).ravel()])
        return fn_u(pts).reshape(sh), fn_v(pts).reshape(sh)

    for uk, vk in zip(u_list, v_list):
        k1u, k1v = adv(uk, vk, x, y)
        k2u, k2v = adv(uk, vk, x + .5 * dt * k1u, y + .5 * dt * k1v)
        k3u, k3v = adv(uk, vk, x + .5 * dt * k2u, y + .5 * dt * k2v)
        k4u, k4v = adv(uk, vk, x + dt * k3u,      y + dt * k3v)
        x += dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        y += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

    F11 = np.gradient(x, lon, axis=1)
    F12 = np.gradient(x, lat, axis=0) * R_lat / R_lon
    F21 = np.gradient(y, lon, axis=1) * R_lon / R_lat
    F22 = np.gradient(y, lat, axis=0)
    C11, C12, C22 = F11 ** 2 + F21 ** 2, F11 * F12 + F21 * F22, F12 ** 2 + F22 ** 2
    tr = C11 + C22
    lam_max = tr / 2 + np.sqrt(np.maximum((tr / 2) ** 2 - (C11 * C22 - C12 ** 2), 0))
    return np.log(np.sqrt(np.maximum(lam_max, 1e-12))) / (len(u_list) * dt)


# ── Load frames ───────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_6km_rtv_uwls_SIO.nc")))
if not files:
    raise FileNotFoundError(f"No 6km files found in {DATA_DIR}")

n_load = min(FTLE_HOURS, len(files))
print(f"Loading {n_load} frames for 24-h FTLE (of {len(files)} available)...")

u_fine_stack, v_fine_stack = [], []
t0 = None

for i, fp in enumerate(files[:n_load]):
    lat, lon, u, v, t = _load_frame(fp)
    if i == 0:
        t0 = t
    ug, vg = _gap_fill(lat, lon, u, v)
    lf, lof, uf, vf = _upsample_smooth(lat, lon, ug, vg)
    lfc, lofc, ufc, vfc = _crop(lf, lof, uf, vf)
    u_fine_stack.append(ufc)
    v_fine_stack.append(vfc)
    print(f"  frame {i+1}/{n_load}")

print("Computing FTLE...")
f_val = _ftle(lfc, lofc, u_fine_stack, v_fine_stack)
fvmax = float(np.nanpercentile(f_val[np.isfinite(f_val)], 98))
print(f"  FTLE range: 0 .. {fvmax:.3e} s^-1")

# ── Plot ──────────────────────────────────────────────────────────────────────

L, LA = np.meshgrid(lofc, lfc)

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.pcolormesh(L, LA, f_val, cmap="jet", vmin=0, vmax=fvmax, shading="auto")
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude (deg)", fontsize=11)
ax.set_ylabel("Latitude (deg)", fontsize=11)
ax.set_title(
    f"Forward-time FTLE ({FTLE_HOURS} h) -- Santa Barbara Channel\n"
    f"{_ts(t0)}\n"
    "Ridges = repelling LCS / separatrices",
    fontsize=11
)
plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "ftle_frame0.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
