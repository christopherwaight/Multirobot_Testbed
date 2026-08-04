"""
ocean_map_similarity_check.py

Measures how far the world coordinate map is from a similarity, and therefore
how far the quantities the controllers read are from the physical ones.

Written 2026-08-03 to back the coordinate-map paragraph of the T-RO paper's
ocean setup section with something reproducible. The numbers that paragraph
carried before (10^-12 determinant agreement, 0.087 spurious strain per unit
vorticity, median 1.8 deg and maximum 25.4 deg eigenvector offset) were not
produced by any script in this repo and could not be regenerated on request.

What is compared, at each placement and time:

  world frame     the 6 robot positions in world units, the field sampled at
                  them, and the quadratic fit run on world-relative offsets.
                  This is exactly what src/control/pentagon_primitives.py
                  does every control cycle.

  physical frame  the SAME 6 sample points and the SAME (u, v) readings, but
                  the quadratic fit run on offsets in km east/north on the
                  local tangent plane about (center_lat, center_lon), using
                  111.32 km/deg latitude and that times cos(center_lat) for
                  longitude, the same constants experiments/_ftle_common.py
                  uses for the FTLE reference.

Three things are reported:

  1. field direction error. The robots are not advected, but every control
     law reads the sampled vector's direction. Under the world map the
     commanded frame sees atan2(v, u) where the tangent plane sees
     atan2(v/L_y, u/L_x). Equal only when L_x = L_y.

  2. strain eigenvector offset. Angle between the world e2 (the stretching
     eigenvector the s1 tracker rides) and the physical one. This is the
     quantity the paper's median/maximum pair reports.

  3. determinant scaling. det(J_world)/det(J_phys) should be the constant
     L_x L_y at every placement. Its relative spread is what "the zero set
     of D is preserved exactly" means numerically.

Both maps are measured in one run, so the cost of the old map and the
exactness of the new one come out of the same code path.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_map_similarity_check.py
"""
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C
from src.control.pentagon_primitives import _fit_vector_quadratic, _strain_quantities
from src.fields.environments.Ocean_HFR import world_scales, world_unit_km

GRID_N = 7
ANCHOR_HOURS = (0.0, 5.0, 10.0, 16.0)
OUT_JSON = os.path.join(C.OUT_DIR, "map_similarity_check.json")
OUT_TXT = os.path.join(C.OUT_DIR, "map_similarity_check.txt")


def acute_angle_deg(a, b):
    """Angle between two undirected unit vectors, in [0, 90] degrees."""
    c = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def corridor_placements(legacy_d, n=GRID_N):
    """
    Placements covering the corridor the trackers use: an n x n grid over the
    bounding box of the published D path, plus the path's own points.
    """
    lat_lo, lat_hi = legacy_d[:, 0].min(), legacy_d[:, 0].max()
    lon_lo, lon_hi = legacy_d[:, 1].min(), legacy_d[:, 1].max()
    la = np.linspace(lat_lo, lat_hi, n)
    lo = np.linspace(lon_lo, lon_hi, n)
    grid = [(float(a), float(b)) for a in la for b in lo]
    on_path = [(float(p[0]), float(p[1])) for p in legacy_d[::12]]
    return grid, on_path


def measure(field, cluster, placements, hours, cfg):
    """
    Run the world/physical comparison over placements x times.

    Returns a dict of arrays: field-direction error (deg), e2 offset (deg),
    det ratio, and the count of placements skipped for a degenerate eigenframe.
    """
    clat, clon, scale_lat, scale_lon = world_scales(cfg)
    km_lat = C.KM_PER_DEG_LAT
    km_lon = C.KM_PER_DEG_LON
    # km per world unit on each axis
    Lx = scale_lon * km_lon      # east
    Ly = scale_lat * km_lat      # north

    dir_err, e2_err, det_ratio = [], [], []
    n_degenerate = 0

    for hour in hours:
        field.reset_clock()
        field.step(hour * 3600.0)
        for (lat, lon) in placements:
            wx, wy = C.latlon_to_world(lat, lon, cfg)
            cluster.reset(wx, wy)
            coords = cluster.get_robot_positions()
            pos = np.array([[coords[2 * i], coords[2 * i + 1]] for i in range(6)])

            uv = np.array([field.get_value(p[0], p[1]) for p in pos])
            u_arr, v_arr = uv[:, 0], uv[:, 1]
            if not np.any(np.abs(uv) > 1e-9):
                continue          # off-grid / all-zero fill, nothing to compare

            centroid = pos.mean(axis=0)
            rel_world = pos - centroid
            # Same points, expressed in km east/north instead of world units.
            rel_phys = rel_world * np.array([Lx, Ly])

            tu_w, tv_w = _fit_vector_quadratic(rel_world, u_arr, v_arr)
            tu_p, tv_p = _fit_vector_quadratic(rel_phys, u_arr, v_arr)

            # 1. field direction, at the centroid sample
            u_c, v_c = tu_w[0], tv_w[0]
            if np.hypot(u_c, v_c) > 1e-6:
                ang_world = np.arctan2(v_c, u_c)
                ang_phys = np.arctan2(v_c / Ly, u_c / Lx)
                d = np.degrees(abs(ang_world - ang_phys)) % 360.0
                dir_err.append(float(min(d, 360.0 - d)))

            # 2. strain eigenframe
            _, _, _, e2_w, r_w = _strain_quantities(tu_w, tv_w)
            _, _, _, e2_p, r_p = _strain_quantities(tu_p, tv_p)
            if r_w < 1e-9 or r_p < 1e-9:
                n_degenerate += 1
            else:
                # e2 is a direction in its own frame; carry the world one into
                # the physical frame before comparing, the same way a tangent
                # transforms.
                e2_w_phys = e2_w / np.array([Lx, Ly])
                e2_err.append(acute_angle_deg(e2_w_phys, e2_p))

            # 3. determinant
            det_w = tu_w[1] * tv_w[2] - tu_w[2] * tv_w[1]
            det_p = tu_p[1] * tv_p[2] - tu_p[2] * tv_p[1]
            if abs(det_p) > 1e-12:
                det_ratio.append(float(det_w / det_p))

    return {
        "Lx_km": Lx, "Ly_km": Ly,
        "n_samples": len(e2_err),
        "n_degenerate": n_degenerate,
        "dir_err_deg": np.array(dir_err),
        "e2_err_deg": np.array(e2_err),
        "det_ratio": np.array(det_ratio),
    }


def stats(a):
    if len(a) == 0:
        return {"median": None, "mean": None, "p95": None, "max": None}
    return {"median": float(np.median(a)), "mean": float(np.mean(a)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}


def report(name, m, lines):
    Lx, Ly = m["Lx_km"], m["Ly_km"]
    ds, es = stats(m["dir_err_deg"]), stats(m["e2_err_deg"])
    dr = m["det_ratio"]
    expected = Lx * Ly
    rel_spread = float(np.max(np.abs(dr / expected - 1.0))) if len(dr) else float("nan")
    spurious = (1.0 - min(Lx, Ly) / max(Lx, Ly)) / 2.0

    lines += [
        f"--- {name} ---",
        f"  one world unit: {Ly:.2f} km north, {Lx:.2f} km east   (ratio {Lx/Ly:.4f})",
        f"  samples: {m['n_samples']} placements x times, {m['n_degenerate']} degenerate eigenframes skipped",
        "",
        f"  field direction error vs tangent plane (deg):",
        f"    median {ds['median']:.3f}   mean {ds['mean']:.3f}   p95 {ds['p95']:.3f}   max {ds['max']:.3f}",
        f"  strain eigenvector e2 offset vs tangent plane (deg):",
        f"    median {es['median']:.3f}   mean {es['mean']:.3f}   p95 {es['p95']:.3f}   max {es['max']:.3f}",
        f"  det(J_world)/det(J_phys): expected constant {expected:.4f} km^2",
        f"    max relative deviation {rel_spread:.3e}",
        f"  spurious strain of a purely rotational physical Jacobian:",
        f"    (1 - min/max)/2 = {spurious:.4f} per unit vorticity",
        "",
    ]
    return {
        "world_unit_km_north": Ly, "world_unit_km_east": Lx, "axis_ratio": Lx / Ly,
        "n_samples": m["n_samples"], "n_degenerate": m["n_degenerate"],
        "field_direction_error_deg": ds,
        "e2_offset_deg": es,
        "det_ratio_expected": expected,
        "det_ratio_max_rel_dev": rel_spread,
        "spurious_strain_per_unit_vorticity": spurious,
    }


def main():
    _, legacy_d, _ = C.load_legacy_reference()
    grid, on_path = corridor_placements(legacy_d)
    placements = grid + on_path
    print(f"{len(grid)} grid placements over the corridor bounding box + "
          f"{len(on_path)} on the published D path, at {len(ANCHOR_HOURS)} times "
          f"{ANCHOR_HOURS} h\n")

    lines = ["Ocean world-map similarity check",
             f"generated {datetime.now(timezone.utc).isoformat()}",
             "by experiments/ocean_map_similarity_check.py",
             f"placements: {GRID_N}x{GRID_N} grid over the published D path's bounding box "
             f"plus {len(on_path)} points on the path itself",
             f"times: {', '.join(f'{h:.0f}' for h in ANCHOR_HOURS)} h into the record",
             ""]
    out = {}

    for iso, name in ((False, "legacy map (isotropic_map: false)"),
                      (True, "square map (isotropic_map: true)")):
        field, cluster, _, _ = C.build_trial(isotropic_map=iso)
        print(f"Measuring {name}...")
        m = measure(field, cluster, placements, ANCHOR_HOURS, field.config)
        out["legacy" if not iso else "square"] = report(name, m, lines)

    text = "\n".join(lines)
    print("\n" + text)

    os.makedirs(C.OUT_DIR, exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(text + "\n")
    C.atomic_write_json(OUT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_map_similarity_check.py",
        "grid_n": GRID_N, "anchor_hours": list(ANCHOR_HOURS),
        "n_placements": len(placements),
        "results": out,
    })
    print(f"Saved: {OUT_TXT}\n       {OUT_JSON}")


if __name__ == "__main__":
    main()
