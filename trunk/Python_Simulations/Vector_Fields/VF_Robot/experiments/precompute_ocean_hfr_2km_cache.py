"""
precompute_ocean_hfr_2km_cache.py

One-time cache builder for the ocean HFR 2km Manim visualization
(Manim_Viz/ocean_hfr_2km_viz.py). Loads all 29 hourly HFR frames, gap-fills
ocean cells, builds the land mask and coastline polygons, and saves everything
to a single .npz file that the Manim script (running in a separate venv
without netCDF4/scipy/pyshp) can load with numpy alone.

Reuses the same load -> mask -> fill -> crop pipeline as
_ftle_common.compute_ftle_field, minus the FTLE/RK4 advection step, since this
cache needs the raw evolving (u, v) field rather than a derived FTLE scalar.

Run once (or whenever the source data changes), from this directory, in the
VF_Robot venv:

    ../venv/bin/python3 precompute_ocean_hfr_2km_cache.py

Regenerate the cache with the same command if ocean_data/hfr_uswc_2012may_2km
changes.
"""
import glob
import os
from datetime import datetime, timezone

import numpy as np

from _ftle_common import (
    load_frame, fill_ocean_only, crop_stack,
    load_coastline_polygons, build_land_mask,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_VF_ROBOT_DIR = os.path.dirname(_HERE)
_VECTOR_FIELDS_DIR = os.path.dirname(_VF_ROBOT_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VECTOR_FIELDS_DIR)))

DATA_DIR = os.path.join(_VECTOR_FIELDS_DIR, "ocean_data", "hfr_uswc_2012may_2km")
FRAME_GLOB = "*_2km_rtv_uwls_SIO.nc"
COAST_SHP = os.path.join(_VECTOR_FIELDS_DIR, "ocean_data", "coastlines",
                          "ne_10m_land", "ne_10m_land.shp")

LAT_MIN, LAT_MAX = 33.8, 34.7
LON_MIN, LON_MAX = -120.7, -119.7

OUT_PATH = os.path.join(_REPO_ROOT, "Manim_Viz", "ocean_hfr_2km_cache.npz")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, FRAME_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matching '{FRAME_GLOB}' in {DATA_DIR}")
    print(f"Found {len(files)} frames in {DATA_DIR}")

    lat0 = lon0 = None
    raw_u, raw_v, t_list = [], [], []
    for i, fp in enumerate(files):
        lat, lon, u, v, t = load_frame(fp, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, pad=0.3)
        if i == 0:
            lat0, lon0 = lat, lon
        raw_u.append(u)
        raw_v.append(v)
        t_list.append(t)
        print(f"  loaded frame {i + 1}/{len(files)}: {os.path.basename(fp)}")

    print("Building land mask from Natural Earth shapefile...")
    coast_polys = load_coastline_polygons(
        COAST_SHP, LON_MIN - 0.5, LON_MAX + 0.5, LAT_MIN - 0.5, LAT_MAX + 0.5,
    )
    land_mask_native = build_land_mask(lat0, lon0, coast_polys)
    print(f"  land cells (native grid): {int(land_mask_native.sum())} / "
          f"{land_mask_native.size} "
          f"({100 * land_mask_native.sum() / land_mask_native.size:.1f}%)")

    print("Gap-filling ocean cells (linear griddata, land cells stay NaN)...")
    u_stack, v_stack = [], []
    for u, v in zip(raw_u, raw_v):
        uf, vf = fill_ocean_only(u, v, land_mask_native)
        u_stack.append(uf)
        v_stack.append(vf)

    print("Cropping to display extent...")
    lat_c, lon_c, u_stack_c = crop_stack(lat0, lon0, u_stack,
                                          LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    _, _, v_stack_c = crop_stack(lat0, lon0, v_stack,
                                  LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    _, _, land_mask_c_list = crop_stack(lat0, lon0, [land_mask_native],
                                         LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    land_mask_c = land_mask_c_list[0]

    u_arr = np.stack(u_stack_c, axis=0)  # (n_frames, n_lat, n_lon)
    v_arr = np.stack(v_stack_c, axis=0)
    t_arr = np.array(t_list, dtype=np.float64)

    coast_points = np.concatenate(coast_polys, axis=0) if coast_polys else np.zeros((0, 2))
    poly_lengths = np.array([len(p) for p in coast_polys], dtype=np.int64)

    speed = np.hypot(u_arr, v_arr)
    p50, p95, p99 = np.nanpercentile(speed, [50, 95, 99])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        lat=lat_c,
        lon=lon_c,
        t=t_arr,
        u=u_arr,
        v=v_arr,
        land_mask=land_mask_c,
        coast_points=coast_points,
        coast_poly_lengths=poly_lengths,
    )

    t_start = datetime.fromtimestamp(t_arr[0], tz=timezone.utc)
    t_end = datetime.fromtimestamp(t_arr[-1], tz=timezone.utc)
    file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)

    print("\nSaved cache:", OUT_PATH)
    print(f"  frames: {len(t_arr)}")
    print(f"  grid shape: {u_arr.shape[1]} lat x {u_arr.shape[2]} lon")
    print(f"  time range: {t_start.isoformat()} .. {t_end.isoformat()}")
    print(f"  land fraction (display grid): "
          f"{100 * land_mask_c.sum() / land_mask_c.size:.1f}%")
    print(f"  current speed percentiles (m/s): "
          f"p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")
    print(f"  coastline polygons: {len(coast_polys)}")
    print(f"  file size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
