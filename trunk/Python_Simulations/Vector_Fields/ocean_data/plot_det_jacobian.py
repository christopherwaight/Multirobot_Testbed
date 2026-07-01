#!/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3
"""
Plot raw velocity, det(J), and FTLE from HF radar surface current data.
Data: May 16-17 2012, Santa Barbara Channel (Michini et al. [34] window).
Source: NCEI THREDDS-Ocean server
  https://www.ncei.noaa.gov/thredds-ocean/fileServer/ioos/hfradar/rtv/2012/201205/USWC/
Accession: gov.noaa.nodc:IOOS-HFRadarRTVector

Resolutions used:
  6km -- 29/29 hours complete, ~67% valid in SBC. Upsampled 6x to ~1km for det(J)/FTLE.
  2km -- 29/29 hours complete, ~48-58% valid in SBC. Used at native resolution.
  1km -- 29/29 files present but ~0% valid in SBC. Not used.

Each comparison figure has two columns per timestep: left = 6km (upsampled), right = 2km (native).

Outputs (all in ocean_data/det_jacobian_plots/):
  velocity_comparison.png      -- raw quiver, both resolutions, all 29 frames
  det_jacobian_comparison.png  -- det(J) colormap, both resolutions, all 29 frames
  ftle_comparison.png          -- forward-time FTLE (24h), both resolutions, all 29 frames
  det_jacobian_3d_0800.png     -- 3D surface of det(J), first frame, 6km upsampled

Usage:
  python3 trunk/Python_Simulations/Vector_Fields/ocean_data/plot_det_jacobian.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import netCDF4 as nc

# ── Config ────────────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7
UPSAMPLE_6KM = 6      # 6km -> ~1km effective spacing
SMOOTH_SIGMA = 2.0    # Gaussian smooth after upsample (grid cells)
FTLE_HOURS   = 24
FILL         = -32767.0

BASE_DIR   = os.path.dirname(__file__)
DIR_6KM    = os.path.join(BASE_DIR, "hfr_uswc_2012may")
DIR_2KM    = os.path.join(BASE_DIR, "hfr_uswc_2012may_2km")
OUTPUT_DIR = os.path.join(BASE_DIR, "det_jacobian_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_frame(fp):
    with nc.Dataset(fp) as f:
        lat = np.array(f.variables["lat"][:])
        lon = np.array(f.variables["lon"][:])
        u   = np.array(f.variables["u"][0, :, :], dtype=float)
        v   = np.array(f.variables["v"][0, :, :], dtype=float)
        t   = float(f.variables["time"][0])
    u[np.abs(u - FILL) < 1.0] = np.nan
    v[np.abs(v - FILL) < 1.0] = np.nan
    # Pad crop so interpolation has margin at edges
    lam = (lat >= LAT_MIN - 0.3) & (lat <= LAT_MAX + 0.3)
    lom = (lon >= LON_MIN - 0.3) & (lon <= LON_MAX + 0.3)
    return lat[lam], lon[lom], u[np.ix_(lam, lom)], v[np.ix_(lam, lom)], t


def ts(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d %Y  %H:%M UTC")


# ── Processing ────────────────────────────────────────────────────────────────

def gap_fill(lat, lon, u, v):
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    uf, vf = u.ravel(), v.ravel()
    ok = np.isfinite(uf) & np.isfinite(vf)
    if ok.sum() < 10:
        return u, v
    uf2 = griddata(pts[ok], uf[ok], pts, method="linear").reshape(u.shape)
    vf2 = griddata(pts[ok], vf[ok], pts, method="linear").reshape(v.shape)
    return (np.where(np.isfinite(uf2), uf2, 0.0),
            np.where(np.isfinite(vf2), vf2, 0.0))


def upsample_smooth(lat, lon, u, v, factor):
    lat_f = np.linspace(lat.min(), lat.max(), len(lat) * factor)
    lon_f = np.linspace(lon.min(), lon.max(), len(lon) * factor)
    u_f = gaussian_filter(RectBivariateSpline(lat, lon, u)(lat_f, lon_f), SMOOTH_SIGMA)
    v_f = gaussian_filter(RectBivariateSpline(lat, lon, v)(lat_f, lon_f), SMOOTH_SIGMA)
    return lat_f, lon_f, u_f, v_f


def crop(lat, lon, u, v):
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    return lat[lam], lon[lom], u[np.ix_(lam, lom)], v[np.ix_(lam, lom)]


def ftle(lat, lon, u_list, v_list, dt=3600.0):
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
        k2u, k2v = adv(uk, vk, x + .5*dt*k1u, y + .5*dt*k1v)
        k3u, k3v = adv(uk, vk, x + .5*dt*k2u, y + .5*dt*k2v)
        k4u, k4v = adv(uk, vk, x + dt*k3u,    y + dt*k3v)
        x += dt / 6 * (k1u + 2*k2u + 2*k3u + k4u)
        y += dt / 6 * (k1v + 2*k2v + 2*k3v + k4v)

    F11 = np.gradient(x, lon, axis=1)
    F12 = np.gradient(x, lat, axis=0) * R_lat / R_lon
    F21 = np.gradient(y, lon, axis=1) * R_lon / R_lat
    F22 = np.gradient(y, lat, axis=0)
    C11, C12, C22 = F11**2 + F21**2, F11*F12 + F21*F22, F12**2 + F22**2
    tr = C11 + C22
    lam_max = tr/2 + np.sqrt(np.maximum((tr/2)**2 - (C11*C22 - C12**2), 0))
    return np.log(np.sqrt(np.maximum(lam_max, 1e-12))) / (len(u_list) * dt)


def det_jacobian(lat, lon, u, v):
    R_lat = 111320.0
    R_lon = 111320.0 * np.cos(np.radians(lat.mean()))
    dy = np.diff(lat) * R_lat
    dx = np.diff(lon) * R_lon
    du_dy = np.full_like(u, np.nan)
    du_dy[1:-1,:] = (u[2:,:] - u[:-2,:]) / (dy[1:]+dy[:-1])[:,None]
    du_dx = np.full_like(u, np.nan)
    du_dx[:,1:-1] = (u[:,2:] - u[:,:-2]) / (dx[1:]+dx[:-1])[None,:]
    dv_dy = np.full_like(v, np.nan)
    dv_dy[1:-1,:] = (v[2:,:] - v[:-2,:]) / (dy[1:]+dy[:-1])[:,None]
    dv_dx = np.full_like(v, np.nan)
    dv_dx[:,1:-1] = (v[:,2:] - v[:,:-2]) / (dx[1:]+dx[:-1])[None,:]
    return du_dx*dv_dy - du_dy*dv_dx


def process_frame_6km(fp):
    lat, lon, u, v, t = load_frame(fp)
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    u_raw = np.where(np.isfinite(u[np.ix_(lam,lom)]), u[np.ix_(lam,lom)], np.nan)
    v_raw = np.where(np.isfinite(v[np.ix_(lam,lom)]), v[np.ix_(lam,lom)], np.nan)
    ug, vg = gap_fill(lat, lon, u, v)
    lf, lof, uf, vf = upsample_smooth(lat, lon, ug, vg, UPSAMPLE_6KM)
    lfc, lofc, ufc, vfc = crop(lf, lof, uf, vf)
    dJ = det_jacobian(lfc, lofc, ufc, vfc)
    return {
        "lat_raw": lat[lam], "lon_raw": lon[lom], "u_raw": u_raw, "v_raw": v_raw,
        "lat": lfc, "lon": lofc, "u": ufc, "v": vfc, "dJ": dJ, "t": t,
    }


def process_frame_2km(fp):
    lat, lon, u, v, t = load_frame(fp)
    lam = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lom = (lon >= LON_MIN) & (lon <= LON_MAX)
    u_raw = np.where(np.isfinite(u[np.ix_(lam,lom)]), u[np.ix_(lam,lom)], np.nan)
    v_raw = np.where(np.isfinite(v[np.ix_(lam,lom)]), v[np.ix_(lam,lom)], np.nan)
    ug, vg = gap_fill(lat, lon, u, v)
    lat_c, lon_c, uc, vc = crop(lat, lon, ug, vg)
    dJ = det_jacobian(lat_c, lon_c, uc, vc)
    return {
        "lat_raw": lat[lam], "lon_raw": lon[lom], "u_raw": u_raw, "v_raw": v_raw,
        "lat": lat_c, "lon": lon_c, "u": uc, "v": vc, "dJ": dJ, "t": t,
    }


# ── Load all frames ───────────────────────────────────────────────────────────

files_6km = sorted(glob.glob(os.path.join(DIR_6KM, "*_6km_rtv_uwls_SIO.nc")))
files_2km = sorted(glob.glob(os.path.join(DIR_2KM, "*_2km_rtv_uwls_SIO.nc")))
if not files_6km:
    raise FileNotFoundError(f"No 6km files in {DIR_6KM}")
if not files_2km:
    raise FileNotFoundError(f"No 2km files in {DIR_2KM}")

N = len(files_6km)
assert len(files_2km) == N, f"File count mismatch: {len(files_6km)} 6km vs {len(files_2km)} 2km"
print(f"Found {N} frames. Processing...")

frames_6km, frames_2km = [], []
for fp6, fp2 in zip(files_6km, files_2km):
    frames_6km.append(process_frame_6km(fp6))
    frames_2km.append(process_frame_2km(fp2))
    print(f"  {os.path.basename(fp6)[:12]}")

print("Done loading.")

NC_COLS = 4
NR = int(np.ceil(N / NC_COLS))


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def hide_axes(axes, last):
    for j in range(last + 1, len(axes)):
        axes[j].set_visible(False)


# ── Plot 1: Velocity quiver comparison ───────────────────────────────────────

print("Plotting velocity comparison...")
fig, axes = plt.subplots(NR, NC_COLS * 2, figsize=(5.5 * NC_COLS * 2, 4.0 * NR))
axes = axes.flatten()

for i, (f6, f2) in enumerate(zip(frames_6km, frames_2km)):
    for col, fr, label in [(0, f6, "6km upsampled"), (1, f2, "2km native")]:
        ax = axes[i * 2 + col]
        L, LA = np.meshgrid(fr["lon_raw"], fr["lat_raw"])
        spd = np.sqrt(fr["u_raw"]**2 + fr["v_raw"]**2)
        scale = 8 if col == 0 else 5
        ax.quiver(L, LA, fr["u_raw"], fr["v_raw"], spd,
                  cmap="viridis", scale=scale, width=0.003)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f"{ts(fr['t'])}\n{label}", fontsize=7)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

hide_axes(axes, (N - 1) * 2 + 1)
fig.suptitle("HFR Surface Currents -- Santa Barbara Channel, May 16-17 2012",
             fontsize=13, y=1.01)
savefig(os.path.join(OUTPUT_DIR, "velocity_comparison.png"))


# ── Plot 2: det(J) comparison ─────────────────────────────────────────────────

print("Plotting det(J) comparison...")
all_dJ = np.concatenate([
    f["dJ"][np.isfinite(f["dJ"])] for f in frames_6km + frames_2km
])
clim = np.nanpercentile(np.abs(all_dJ), 98) * 1e10

fig, axes = plt.subplots(NR, NC_COLS * 2, figsize=(5.5 * NC_COLS * 2, 4.0 * NR))
axes = axes.flatten()

for i, (f6, f2) in enumerate(zip(frames_6km, frames_2km)):
    for col, fr, label in [(0, f6, "6km upsampled"), (1, f2, "2km native")]:
        ax = axes[i * 2 + col]
        L, LA = np.meshgrid(fr["lon"], fr["lat"])
        Z = fr["dJ"] * 1e10
        im = ax.pcolormesh(L, LA, Z, cmap="RdBu_r",
                           vmin=-clim, vmax=clim, shading="auto")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f"{ts(fr['t'])}\n{label}", fontsize=7)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, label=r"det($J$) $\times10^{10}$", shrink=0.85)

hide_axes(axes, (N - 1) * 2 + 1)
fig.suptitle(r"det($J(\mathbf{v})$) -- Santa Barbara Channel, May 16-17 2012"
             "\nRed: elliptic (eddy core)   Blue: hyperbolic",
             fontsize=13, y=1.01)
savefig(os.path.join(OUTPUT_DIR, "det_jacobian_comparison.png"))


# ── Plot 3: 3D surface, first frame (6km upsampled) ──────────────────────────

print("Plotting 3D det(J)...")
f6 = frames_6km[0]
L3, LA3 = np.meshgrid(f6["lon"], f6["lat"])
Z = f6["dJ"] * 1e10
Z_plot = np.where(np.isfinite(Z), Z, 0.0)

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection="3d")

zn = (Z_plot - Z_plot.min()) / (Z_plot.max() - Z_plot.min() + 1e-30)
fc = plt.cm.RdBu_r(zn)

ax.plot_surface(L3, LA3, Z_plot, facecolors=fc,
                linewidth=0, antialiased=True, alpha=0.92)
ax.contour(L3, LA3, Z_plot, levels=[0.0], colors="yellow",
           linewidths=2.0, offset=Z_plot.min())

ax.set_xlabel("Longitude (deg)", labelpad=8)
ax.set_ylabel("Latitude (deg)",  labelpad=8)
ax.set_zlabel(r"det($J$) $\times10^{10}$ [s$^{-2}$]", labelpad=8)
ax.set_title(
    r"det($J(\mathbf{v})$) -- Santa Barbara Channel (6km upsampled to ~1km)" +
    f"\n{ts(f6['t'])}\n"
    r"Peak: eddy core (det>0)   Valleys: hyperbolic regions (det<0)",
    fontsize=11)
ax.view_init(elev=35, azim=-60)

sm = plt.cm.ScalarMappable(cmap="RdBu_r")
sm.set_array(Z_plot)
plt.colorbar(sm, ax=ax, label=r"det($J$) $\times10^{10}$ [s$^{-2}$]",
             shrink=0.55, pad=0.1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "det_jacobian_3d_0800.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'det_jacobian_3d_0800.png')}")

# ── Plot 4: FTLE comparison ───────────────────────────────────────────────────

print(f"Computing FTLE ({FTLE_HOURS}h integration) for both resolutions...")
ftle_frames_6km, ftle_frames_2km = [], []
for i in range(N):
    end = min(i + FTLE_HOURS, N)

    f6 = frames_6km[i]
    f6_val = ftle(f6["lat"], f6["lon"],
                  [f["u"] for f in frames_6km[i:end]],
                  [f["v"] for f in frames_6km[i:end]])
    ftle_frames_6km.append(f6_val)

    f2 = frames_2km[i]
    f2_val = ftle(f2["lat"], f2["lon"],
                  [f["u"] for f in frames_2km[i:end]],
                  [f["v"] for f in frames_2km[i:end]])
    ftle_frames_2km.append(f2_val)
    print(f"  {i+1}/{N}")

all_ftle = np.concatenate([
    v[np.isfinite(v)] for v in ftle_frames_6km + ftle_frames_2km
])
fvmax = np.nanpercentile(all_ftle, 98)

fig, axes = plt.subplots(NR, NC_COLS * 2, figsize=(5.5 * NC_COLS * 2, 4.0 * NR))
axes = axes.flatten()

for i, (f6, f2, fv6, fv2) in enumerate(zip(frames_6km, frames_2km,
                                             ftle_frames_6km, ftle_frames_2km)):
    for col, fr, fv, label in [(0, f6, fv6, "6km upsampled"), (1, f2, fv2, "2km native")]:
        ax = axes[i * 2 + col]
        L, LA = np.meshgrid(fr["lon"], fr["lat"])
        im = ax.pcolormesh(L, LA, fv, cmap="jet", vmin=0, vmax=fvmax, shading="auto")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(f"{ts(fr['t'])}\n{label}", fontsize=7)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)

hide_axes(axes, (N - 1) * 2 + 1)
fig.suptitle(
    f"Forward-time FTLE ({FTLE_HOURS}h) -- Santa Barbara Channel, May 16-17 2012\n"
    "Ridges = repelling LCS / separatrices",
    fontsize=13, y=1.01)
savefig(os.path.join(OUTPUT_DIR, "ftle_comparison.png"))

print("\nAll outputs in:", OUTPUT_DIR)
