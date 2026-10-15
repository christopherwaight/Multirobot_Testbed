"""
ocean_candidateA_finemc_v3.py

Third version of the fine Monte Carlo search around candidate A
(34.411906N, -120.392016W). Supersedes ocean_candidateA_finemc_v2.py, which
scored J on each controller's FINAL position only. The author clarified
mid-run that a controller passing close to E0 and then continuing through is
exactly what should be credited ("i thought it was okay if they pass close
to it, and then continue thru") -- v2's endpoint-only J silently penalized
that. v3 scores on CLOSEST APPROACH ALONG THE PATH instead, same as
ocean_shared_start_closest_approach.py earlier this session.

v2's own search (before being stopped for this reason) found nothing beating
candidate A's endpoint-only J_A=14.11km within ~5 minutes / ~1800 evaluations
(best found: 15.21km, still worse). Whether that changes under closest-
approach scoring is exactly what this version tests.

CHANGES FROM v2:
  - J = max(D's minimum distance to E0 anywhere along its 168-step path,
    s1's minimum distance to E0 anywhere along its path), NOT the endpoint
    distance. Reported per-candidate: the frac of the run (0=start, 1=final
    step) at which that minimum occurred, so "swings close and continues"
    (low frac) is distinguishable from "still closing at the end" (frac~1,
    functionally identical to the v2 endpoint score).
  - Candidate A's own numbers under this metric, verified fresh this
    session: D closest approach 3.60km at 95% of its run, s1 closest
    approach 14.01km at 99% of its run -- so J_A=14.01km, barely different
    from v2's endpoint-based 14.11km, because candidate A's OWN path does
    not swing close and continue -- it is still closing distance at the very
    end, consistent with the s1 tracker's behavior everywhere else this
    session (see feedback_fixed_budget_comparisons in project memory). The
    metric change matters for OTHER candidates that might behave
    differently, not for A's own baseline.

UNCHANGED FROM v2:
  - Corridor cap 1.5 km (both D and s1 paths must stay within this of
    candidate A's own paths).
  - Smoothness (mean turning angle, radians) reported and used ONLY as a
    tie-breaker within TIE_EPSILON_KM of J, never as the primary objective.
  - Search disk RADIUS_KM around candidate A's own start.
  - Land rejection via Natural Earth coastline point-in-polygon test.
  - NO rotation/heading_offset in this pass.
  - Fixed 168-step/28h budget for both controllers.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_candidateA_finemc_v3.py \
        > experiments/outputs/oecs/candidateA_finemc_v3.log 2>&1 &
"""
import sys
import os
import time
import json
import tempfile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from matplotlib.path import Path as MplPath

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon
from _ftle_common import load_coastline_polygons

FORMATION_CONFIG  = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX, SIM_STEPS, CONTROL_GAIN = 0.04, 168, 1.8
MOMENTUM_ALPHA, STICTION_THRESHOLD, TIME_WARP = 0.0, 0.002, 6000.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.3, 0.05, 0.15

CAND_A_START = (34.411906, -120.392016)
E0           = (34.0412, -120.2617)

RADIUS_KM            = 4.0    # search disk radius around candidate A
CORRIDOR_DEV_MAX_KM  = 1.5    # relaxed from 1.0 km (v1) per author instruction
TIE_EPSILON_KM       = 0.3    # J values within this of each other are "tied";
                              # smoothness only breaks ties this close
TOTAL_BUDGET_SECONDS = 2 * 3600.0
HEARTBEAT_EVERY_SEC  = 60.0
SEED = 20260727

TOP_K, DEDUP_RADIUS_KM = 8, 0.1

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(34.2))

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
OUT_DIR   = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)
SHORTLIST_JSON = os.path.join(OUT_DIR, "candidateA_finemc_v3_shortlist.json")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")


def km(p, q):
    return float(np.hypot((p[0] - q[0]) * KM_PER_DEG_LAT, (p[1] - q[1]) * KM_PER_DEG_LON))


def path_dev_km(path_a_latlon, path_b_latlon):
    a = np.column_stack([path_a_latlon[:, 1] * KM_PER_DEG_LON, path_a_latlon[:, 0] * KM_PER_DEG_LAT])
    b = np.column_stack([path_b_latlon[:, 1] * KM_PER_DEG_LON, path_b_latlon[:, 0] * KM_PER_DEG_LAT])
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def mean_turning_angle(path_latlon):
    """Mean absolute turning angle (radians) between consecutive centroid
    steps. 0 = perfectly straight; pi = reversing direction every step.
    Steps with near-zero displacement (parked/stalled) are skipped to avoid
    numerical noise dominating the average."""
    xy = np.column_stack([path_latlon[:, 1] * KM_PER_DEG_LON, path_latlon[:, 0] * KM_PER_DEG_LAT])
    seg = np.diff(xy, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    keep = lens > 1e-4   # drop near-zero-displacement steps (km scale)
    seg = seg[keep]
    if len(seg) < 3:
        return 0.0
    v1 = seg[:-1]
    v2 = seg[1:]
    cos_a = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.mean(np.arccos(cos_a)))


def closest_approach(path_latlon, target):
    """(min_dist_km, frac_of_run) -- minimum distance from any point on the
    path to `target`, and where along the path (0=start, 1=final step) that
    minimum occurred."""
    d = np.hypot((path_latlon[:, 0] - target[0]) * KM_PER_DEG_LAT,
                (path_latlon[:, 1] - target[1]) * KM_PER_DEG_LON)
    i = int(np.argmin(d))
    return float(d[i]), i / max(len(d) - 1, 1)


def run_traj(field, cluster, prim, lat, lon):
    sx, sy = _latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)     # also clears OECS tangent/mode state (session fix)
    field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        field.step(cluster.timestep * TIME_WARP)
    center = cluster.get_center_history()
    return np.array([_world_to_latlon(p[0], p[1], field.config) for p in center])


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_shortlist():
    if os.path.exists(SHORTLIST_JSON):
        try:
            with open(SHORTLIST_JSON) as f:
                d = json.load(f)
            lst = d.get("shortlist", [])
            print(f"Resuming: found an existing shortlist of {len(lst)} candidates.")
            return lst
        except Exception as e:
            print(f"Could not load {SHORTLIST_JSON} ({e}), starting fresh.")
    return []


def save_shortlist(shortlist, n_evals, n_rejected_land, n_rejected_corridor, J_A, smoothness_A):
    atomic_write_json(SHORTLIST_JSON, {
        "shortlist": shortlist,
        "n_evals": n_evals,
        "n_rejected_land": n_rejected_land,
        "n_rejected_corridor": n_rejected_corridor,
        "candidate_A_reference": {
            "start_lat": CAND_A_START[0], "start_lon": CAND_A_START[1],
            "dD_to_E0_km": 6.38, "dS1_to_E0_km": 14.11, "J_km": J_A,
            "smoothness_rad": smoothness_A,
            "note": "twice-verified fresh this session",
        },
        "note": ("Ranked by J=max(D_closest_approach,s1_closest_approach) "
                 "ascending (lower is better -- closer approach to Fig 8's "
                 f"endpoint E0 anywhere along the path) among candidates "
                 f"that keep BOTH D and s1 paths within {CORRIDOR_DEV_MAX_KM} "
                 "km of candidate A's own paths. *_closest_frac is how far "
                 "into the 168-step run that closest approach occurred "
                 "(0=start, 1=final step); near 1.0 means still closing at "
                 "the end (same as endpoint distance), a lower value means "
                 "the path swung close and continued through. Smoothness "
                 "(mean turning angle, radians) is reported but only breaks "
                 f"ties within {TIE_EPSILON_KM} km of J. No "
                 "rotation/heading_offset used."),
    })


def make_record(la, lo, d_from_A, corridor_dev_d, corridor_dev_s1,
                d_end, s1_end, dD_closest, dD_frac, dS1_closest, dS1_frac,
                J, smooth_d, smooth_s1, n_evals, elapsed_s):
    return {
        "start_lat": la, "start_lon": lo, "d_from_candA_km": d_from_A,
        "corridor_dev_D_km": corridor_dev_d, "corridor_dev_s1_km": corridor_dev_s1,
        "D_end_lat": d_end[0], "D_end_lon": d_end[1],
        "s1_end_lat": s1_end[0], "s1_end_lon": s1_end[1],
        "dD_closest_km": dD_closest, "dD_closest_frac": dD_frac,
        "dS1_closest_km": dS1_closest, "dS1_closest_frac": dS1_frac,
        "J_km": J,
        "smoothness_D_rad": smooth_d, "smoothness_s1_rad": smooth_s1,
        "smoothness_mean_rad": 0.5 * (smooth_d + smooth_s1),
        "n_evals_at_discovery": n_evals, "elapsed_seconds_at_discovery": elapsed_s,
    }


def rank_key(r):
    """Primary: J ascending. Tie-break (only within TIE_EPSILON_KM of the
    current best J in the set being sorted): smoothness ascending. Since
    Python sort is stable and we don't know the global best J in isolation,
    approximate by rounding J to the tie-epsilon grid, then smoothness."""
    j_bucket = round(r["J_km"] / TIE_EPSILON_KM)
    return (j_bucket, r["smoothness_mean_rad"])


def maybe_insert(shortlist, record):
    la, lo = record["start_lat"], record["start_lon"]
    for other in shortlist:
        if km((la, lo), (other["start_lat"], other["start_lon"])) < DEDUP_RADIUS_KM:
            if rank_key(record) < rank_key(other):
                shortlist = [r for r in shortlist if r is not other] + [record]
                shortlist.sort(key=rank_key)
                return shortlist, True
            return shortlist, False
    if len(shortlist) < TOP_K:
        shortlist = shortlist + [record]
        shortlist.sort(key=rank_key)
        return shortlist, True
    if rank_key(record) < rank_key(shortlist[-1]):
        shortlist = shortlist[:-1] + [record]
        shortlist.sort(key=rank_key)
        return shortlist, True
    return shortlist, False


def main():
    print("Loading coastline polygons for land-rejection...")
    polys = load_coastline_polygons(COAST_SHP, -120.7, -119.7, 33.8, 34.7)
    mpl_polys = [MplPath(p) for p in polys]

    def is_land(lat, lon):
        pt = np.array([[lon, lat]])
        return any(mp.contains_points(pt)[0] for mp in mpl_polys)

    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster = PentagonCluster(FORMATION_CONFIG, field,
                              momentum_alpha=MOMENTUM_ALPHA,
                              stiction_threshold=STICTION_THRESHOLD)

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=V_MAX, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=V_MAX, g_perp=G_PERP, s_trim=S_TRIM,
                                      r_band=R_BAND, g_capture=G_CAPTURE, s_capture=None)
        return vx * CONTROL_GAIN, vy * CONTROL_GAIN

    print("Establishing (and re-verifying) candidate A's own reference paths...")
    A_D_path  = run_traj(field, cluster, prim_d,  *CAND_A_START)
    A_S1_path = run_traj(field, cluster, prim_s1, *CAND_A_START)
    A_D_path2 = run_traj(field, cluster, prim_d,  *CAND_A_START)
    assert np.allclose(A_D_path[-1], A_D_path2[-1]), \
        "Candidate A's D endpoint did not reproduce -- aborting, do not trust this run."
    A_D_end, A_S1_end = A_D_path[-1], A_S1_path[-1]
    A_dD_closest, A_dD_frac = closest_approach(A_D_path, E0)
    A_dS1_closest, A_dS1_frac = closest_approach(A_S1_path, E0)
    J_A = max(A_dD_closest, A_dS1_closest)
    smooth_A_D  = mean_turning_angle(A_D_path)
    smooth_A_S1 = mean_turning_angle(A_S1_path)
    smoothness_A = 0.5 * (smooth_A_D + smooth_A_S1)
    print(f"  Candidate A verified: D closest={A_dD_closest:.2f}km@{A_dD_frac:.0%}  "
          f"s1 closest={A_dS1_closest:.2f}km@{A_dS1_frac:.0%}   J_A={J_A:.2f}km")
    print(f"  Candidate A smoothness (mean turning angle): D={smooth_A_D:.4f} rad, "
          f"s1={smooth_A_S1:.4f} rad, mean={smoothness_A:.4f} rad\n")

    shortlist = load_shortlist()
    rng = np.random.default_rng(SEED)
    t_start_wall = time.time()
    deadline = t_start_wall + TOTAL_BUDGET_SECONDS
    n_evals = n_rej_land = n_rej_corridor = 0
    last_heartbeat = time.time()
    best_J = shortlist[0]["J_km"] if shortlist else J_A

    print(f"=== Fine MC search v3 (closest-approach), r <= {RADIUS_KM} km around "
          f"candidate A, budget {TOTAL_BUDGET_SECONDS/3600:.1f} h, corridor cap "
          f"{CORRIDOR_DEV_MAX_KM} km, objective J=max(D_closest,s1_closest) to beat, "
          f"J_A={J_A:.2f}km ===")

    while time.time() < deadline:
        r_km = RADIUS_KM * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        la = CAND_A_START[0] + (r_km * np.cos(theta)) / KM_PER_DEG_LAT
        lo = CAND_A_START[1] + (r_km * np.sin(theta)) / KM_PER_DEG_LON

        if is_land(la, lo):
            n_rej_land += 1
            continue

        d_from_A = km((la, lo), CAND_A_START)
        d_path  = run_traj(field, cluster, prim_d,  la, lo)
        s1_path = run_traj(field, cluster, prim_s1, la, lo)
        n_evals += 1

        cdev_d  = path_dev_km(A_D_path,  d_path)
        cdev_s1 = path_dev_km(A_S1_path, s1_path)

        if cdev_d > CORRIDOR_DEV_MAX_KM or cdev_s1 > CORRIDOR_DEV_MAX_KM:
            n_rej_corridor += 1
            continue

        dD_closest, dD_frac = closest_approach(d_path, E0)
        dS1_closest, dS1_frac = closest_approach(s1_path, E0)
        J = max(dD_closest, dS1_closest)
        smooth_d  = mean_turning_angle(d_path)
        smooth_s1 = mean_turning_angle(s1_path)

        worst_kept = rank_key(shortlist[-1]) if len(shortlist) >= TOP_K else (float("inf"), float("inf"))
        record = make_record(la, lo, d_from_A, cdev_d, cdev_s1,
                             d_path[-1], s1_path[-1], dD_closest, dD_frac,
                             dS1_closest, dS1_frac, J, smooth_d, smooth_s1,
                             n_evals, time.time() - t_start_wall)
        if rank_key(record) < worst_kept:
            shortlist, inserted = maybe_insert(shortlist, record)
            if inserted:
                save_shortlist(shortlist, n_evals, n_rej_land, n_rej_corridor, J_A, smoothness_A)
                if J < best_J:
                    best_J = J
                print(f"  [{record['elapsed_seconds_at_discovery']/60:6.1f} min, eval #{n_evals}] "
                      f"shortlist updated: J={J:.2f}km (A={J_A:.2f})  "
                      f"start=({la:.6f},{lo:.6f}) [{d_from_A:.2f}km from A]  "
                      f"corridor_dev D={cdev_d:.2f} s1={cdev_s1:.2f}  "
                      f"D_closest={dD_closest:.2f}@{dD_frac:.0%} "
                      f"s1_closest={dS1_closest:.2f}@{dS1_frac:.0%}  "
                      f"smooth D={smooth_d:.3f} s1={smooth_s1:.3f}  "
                      f"(shortlist size {len(shortlist)}, best J={shortlist[0]['J_km']:.2f})")

        if time.time() - last_heartbeat > HEARTBEAT_EVERY_SEC:
            elapsed = time.time() - t_start_wall
            remaining = deadline - time.time()
            print(f"  heartbeat: {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min left, "
                  f"{n_evals} valid evals, {n_rej_land} land-rejected, "
                  f"{n_rej_corridor} corridor-rejected, best J={best_J:.2f}km (A={J_A:.2f})")
            last_heartbeat = time.time()

    print(f"\n=== SEARCH COMPLETE: {n_evals} valid evaluations "
          f"({n_rej_land} land-rejected, {n_rej_corridor} corridor-rejected), "
          f"{(time.time()-t_start_wall)/3600:.2f} h elapsed ===")

    if not shortlist:
        print("No candidate satisfied the corridor constraint. Candidate A stands.")
        return

    print(f"Shortlist ({len(shortlist)} candidates), ranked by J then smoothness:")
    for i, r in enumerate(shortlist):
        beats = "BEATS A" if r["J_km"] < J_A else "worse than A"
        print(f"  #{i+1}: J={r['J_km']:.2f}km ({beats})  "
              f"start=({r['start_lat']:.6f},{r['start_lon']:.6f})  "
              f"[{r['d_from_candA_km']:.2f}km from A]  "
              f"corridor_dev D={r['corridor_dev_D_km']:.2f} s1={r['corridor_dev_s1_km']:.2f}  "
              f"D_closest={r['dD_closest_km']:.2f}@{r['dD_closest_frac']:.0%} "
              f"s1_closest={r['dS1_closest_km']:.2f}@{r['dS1_closest_frac']:.0%}  "
              f"smooth={r['smoothness_mean_rad']:.3f}rad (A={smoothness_A:.3f})")

    print("\n=== Full-precision re-verification of the shortlist ===")
    field2 = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster2 = PentagonCluster(FORMATION_CONFIG, field2,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)
    for r in shortlist:
        la, lo = r["start_lat"], r["start_lon"]
        d_path  = run_traj(field2, cluster2, prim_d,  la, lo)
        s1_path = run_traj(field2, cluster2, prim_s1, la, lo)
        dD_c, _ = closest_approach(d_path, E0)
        dS1_c, _ = closest_approach(s1_path, E0)
        match = (abs(dD_c - r["dD_closest_km"]) < 1e-6 and
                abs(dS1_c - r["dS1_closest_km"]) < 1e-6)
        r["reverified"] = bool(match)
        status = "OK" if match else "MISMATCH -- treat as artifact"
        print(f"  J={r['J_km']:.2f}km at ({la:.6f},{lo:.6f}): {status}")

    shortlist = [r for r in shortlist if r.get("reverified", True)]
    shortlist.sort(key=rank_key)
    save_shortlist(shortlist, n_evals, n_rej_land, n_rej_corridor, J_A, smoothness_A)
    print(f"\nFinal shortlist saved to {SHORTLIST_JSON}: {len(shortlist)} re-verified candidates.")

    if shortlist and shortlist[0]["J_km"] < J_A:
        print(f"\nBest: J={shortlist[0]['J_km']:.2f}km beats candidate A's J={J_A:.2f}km "
              f"while keeping the corridor.")
    else:
        print(f"\nNothing found beats candidate A's J={J_A:.2f}km while keeping the "
              f"corridor constraint. Candidate A remains the best answer.")


if __name__ == "__main__":
    main()
