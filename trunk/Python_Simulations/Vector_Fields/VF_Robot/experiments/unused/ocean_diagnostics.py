"""
ocean_diagnostics.py

Per-step instrumentation for the paper's ocean trial (Draft_6a.tex sec:disc_ocean),
at the canonical operating point of _ocean_run_common.py, from the shared start
LEGACY_START used in the ocean figure.

Neither main_ocean_hfr_v6r.py nor main_ocean_hfr_v6r_timevarying.py writes any
per-step log (see revision/items.yaml B2, B7_A4, SHAPE_RMSE): the runners return
only the (N,2) centroid path. This script re-derives the missing diagnostics by
re-running the same 168-step trial and, at each step, independently recomputing
the same six-robot quadratic fit the controller itself computes (deterministic,
zero measurement/position noise on this field, so the recomputed fit is exact,
not an approximation of what the controller saw).

Reported, per tracker:
  - s1_hat range along the path
  - (s1 tracker only) beta latch fraction and first-latch step, read directly
    off cluster._oecs_banded rather than re-derived, since that flag IS the
    controller's own state
  - min ||grad s1_hat|| along the path
  - realized centroid speed (m/s, physical)
  - divergence ratio 2|mu|/r (v2 A4's exact formula) along the path
  - cond(Phi) of the 6x6 fit design matrix
  - formation shape RMSE relative to the t=0 pairwise distances
  - mean/max distance from the path to the 24-h forward FTLE ridge, in km
    (via _ocean_run_common.ridge_tree / dist_to_ridge_km, reusing the cached
    ftle_cache_24h_u4.npz)

Writes outputs/oecs/ocean_diagnostics.json (atomic).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.control.pentagon_primitives import (
    _sample_vector_at_robots, _get_relative_positions,
    _fit_vector_quadratic, _strain_quantities, _quadratic_basis,
)

from _ocean_run_common import (
    build_trial, LEGACY_START, SIM_STEPS, TIME_WARP,
    ridge_tree, dist_to_ridge_km, atomic_write_json, OUT_DIR,
    KM_PER_DEG_LAT, KM_PER_DEG_LON,
)
from _coords_common import latlon_to_world, world_to_latlon

FTLE_CACHE_24H = os.path.join(OUT_DIR, "ftle_cache_24h_u4.npz")


def _pairdists(pts):
    d = []
    for i in range(6):
        for j in range(i + 1, 6):
            d.append(np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
    return np.array(d)


def _cond_phi(rel_pos):
    Phi = np.array([_quadratic_basis(rel_pos[j, 0], rel_pos[j, 1]) for j in range(6)])
    return float(np.linalg.cond(Phi))


def run_instrumented(field, cluster, prim, lat, lon, world_L_km,
                      sim_steps=SIM_STEPS, is_s1_tracker=False):
    sx, sy = latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)
    field.reset_clock()

    pts0 = np.array(cluster.get_robot_positions()).reshape(6, 2)
    nominal_pdist = _pairdists(pts0)

    s1_series, grad_norm_series, mu_over_r_series = [], [], []
    cond_series, shape_rms_series = [], []
    banded_series = []

    for _ in range(sim_steps):
        u_arr, v_arr = _sample_vector_at_robots(cluster)
        rel_pos = _get_relative_positions(cluster)
        theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)
        s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
        mu = 0.5 * (theta_u[1] + theta_v[2])

        s1_series.append(s1)
        grad_norm_series.append(float(np.linalg.norm(grad_s1)))
        mu_over_r_series.append(2.0 * abs(mu) / r if r > 1e-9 else float("nan"))
        cond_series.append(_cond_phi(rel_pos))

        pts = np.array(cluster.get_robot_positions()).reshape(6, 2)
        shape_rms_series.append(float(np.sqrt(np.mean((_pairdists(pts) - nominal_pdist) ** 2))))

        cluster.move(prim)
        field.step(cluster.timestep * TIME_WARP)

        if is_s1_tracker:
            banded_series.append(bool(getattr(cluster, "_oecs_banded", False)))

    center = cluster.get_center_history()
    path_latlon = np.array([world_to_latlon(p[0], p[1], field.config) for p in center])

    xy_km = np.column_stack([path_latlon[:, 1] * KM_PER_DEG_LON,
                              path_latlon[:, 0] * KM_PER_DEG_LAT])
    step_dist_km = np.linalg.norm(np.diff(xy_km, axis=0), axis=1)
    dt_s = cluster.timestep * TIME_WARP
    speed_mps = step_dist_km * 1000.0 / dt_s

    result = {
        "s1_range": [float(np.min(s1_series)), float(np.max(s1_series))],
        "grad_s1_norm_min": float(np.min(grad_norm_series)),
        "divergence_ratio_2mu_over_r": {
            "median": float(np.nanmedian(mu_over_r_series)),
            "p90": float(np.nanpercentile(mu_over_r_series, 90)),
        },
        "cond_phi_max": float(np.max(cond_series)),
        "shape_rmse_max": float(np.max(shape_rms_series)),
        "realized_speed_mps": {
            "mean": float(np.mean(speed_mps)),
            "max": float(np.max(speed_mps)),
        },
        "path_latlon": path_latlon.tolist(),
    }
    if is_s1_tracker:
        latched = np.array(banded_series)
        result["beta_latched_fraction"] = float(latched.mean())
        result["beta_latch_step"] = int(np.argmax(latched)) if latched.any() else None
    return result


def main():
    field, cluster, prim_d, prim_s1 = build_trial()
    lat0, lon0 = LEGACY_START

    print("Running D tracker (168 steps, instrumented)...")
    d_result = run_instrumented(field, cluster, prim_d, lat0, lon0, None, is_s1_tracker=False)

    field, cluster, prim_d, prim_s1 = build_trial()
    print("Running s1 tracker (168 steps, instrumented)...")
    s1_result = run_instrumented(field, cluster, prim_s1, lat0, lon0, None, is_s1_tracker=True)

    tree, n_ridge = ridge_tree(FTLE_CACHE_24H, percentile=95.0)
    d_ridge_mean, d_ridge_max = dist_to_ridge_km(tree, np.array(d_result["path_latlon"]))
    s1_ridge_mean, s1_ridge_max = dist_to_ridge_km(tree, np.array(s1_result["path_latlon"]))
    print(f"FTLE ridge cache: {n_ridge} points >= 95th percentile")

    out = {
        "start": {"lat": lat0, "lon": lon0},
        "sim_steps": SIM_STEPS,
        "d_tracker": {k: v for k, v in d_result.items() if k != "path_latlon"},
        "s1_tracker": {k: v for k, v in s1_result.items() if k != "path_latlon"},
        # Convenience aliases matching revision/items.yaml gate keys.
        "s1": {
            "beta_latched_fraction": s1_result["beta_latched_fraction"],
            "beta_latch_step": s1_result["beta_latch_step"],
        },
        "ridge_distance": {
            "d_mean_km": d_ridge_mean, "d_max_km": d_ridge_max,
            "s1_mean_km": s1_ridge_mean, "s1_max_km": s1_ridge_max,
        },
        "shape_rmse": {
            "max": max(d_result["shape_rmse_max"], s1_result["shape_rmse_max"]),
        },
    }

    out_path = os.path.join(OUT_DIR, "ocean_diagnostics.json")
    atomic_write_json(out_path, out)
    print(f"Wrote {out_path}")

    print("\n--- D tracker ---")
    print(f"  s1_hat range: {d_result['s1_range']}")
    print(f"  min ||grad s1_hat||: {d_result['grad_s1_norm_min']:.5f}")
    print(f"  2|mu|/r: median {d_result['divergence_ratio_2mu_over_r']['median']:.4f}, "
          f"p90 {d_result['divergence_ratio_2mu_over_r']['p90']:.4f}")
    print(f"  cond(Phi) max: {d_result['cond_phi_max']:.2f}")
    print(f"  shape RMSE max: {d_result['shape_rmse_max']:.4f}")
    print(f"  realized speed: mean {d_result['realized_speed_mps']['mean']:.3f} m/s, "
          f"max {d_result['realized_speed_mps']['max']:.3f} m/s")
    print(f"  FTLE ridge distance: mean {d_ridge_mean:.2f} km, max {d_ridge_max:.2f} km")

    print("\n--- s1 tracker ---")
    print(f"  s1_hat range: {s1_result['s1_range']}")
    print(f"  beta latched fraction: {s1_result['beta_latched_fraction']:.3f}"
          f"  (first at step {s1_result['beta_latch_step']})")
    print(f"  min ||grad s1_hat||: {s1_result['grad_s1_norm_min']:.5f}")
    print(f"  2|mu|/r: median {s1_result['divergence_ratio_2mu_over_r']['median']:.4f}, "
          f"p90 {s1_result['divergence_ratio_2mu_over_r']['p90']:.4f}")
    print(f"  cond(Phi) max: {s1_result['cond_phi_max']:.2f}")
    print(f"  shape RMSE max: {s1_result['shape_rmse_max']:.4f}")
    print(f"  realized speed: mean {s1_result['realized_speed_mps']['mean']:.3f} m/s, "
          f"max {s1_result['realized_speed_mps']['max']:.3f} m/s")
    print(f"  FTLE ridge distance: mean {s1_ridge_mean:.2f} km, max {s1_ridge_max:.2f} km")


if __name__ == "__main__":
    main()
