"""
ocean_candidateA_finemc_v2.py

Second version of the fine Monte Carlo search around candidate A
(34.411906N, -120.392016W). Supersedes ocean_candidateA_finemc_search.py,
whose south_gain objective (push the worse endpoint further south while
keeping the corridor) converged after ~5000 evaluations to a hard plateau at
south_gain ~= +0.05 km -- three orders of magnitude below anything useful --
while the candidates that reached even that were mostly WORSE on the metric
that actually matters (dS1_to_E0 ballooned to 20-27 km vs. candidate A's own
14.11 km). That confirmed south_gain was the wrong objective, not that the
search had failed.

CHANGES FROM v1, per author instruction this session:
  1. Corridor cap relaxed: CORRIDOR_DEV_MAX_KM 1.0 -> 1.5 km. v1 rejected
     ~99.5% of draws under the 1.0 km cap; loosening it gives the search more
     room without abandoning "same corridor as candidate A" (the entire
     point of building on candidate A rather than searching fresh).
  2. Objective changed to J = max(dD_to_E0_km, dS1_to_E0_km) -- the SAME
     objective as the very first search this session
     (ocean_shared_start_search.py), which is what found candidate A in the
     first place. south_gain is dropped entirely: minimizing it was
     optimizing "push the endpoint" rather than "land close to E0," and the
     author confirmed closeness to E0, not southward distance, is what
     matters ("I really think it is passing close to those endpoints that
     are important").
  3. Smoothness added as a REPORTED metric and TIE-BREAKER ONLY, not a
     primary objective (author: "low penalty ... only breaks the tie").
     Measured as mean absolute turning angle between consecutive centroid
     steps (radians); lower = smoother/straighter. Computed for D and s1
     separately and combined by mean. Used only to order candidates whose J
     is within TIE_EPSILON_KM of each other.

CONSTRAINTS unchanged from v1:
  - Search disk: RADIUS_KM around candidate A's own start (not the original
    Fig 8 start).
  - Land rejection by resampling against Natural Earth coastline polygons
    (verified this session: island interiors correctly register as land;
    the earlier confusion was an incorrectly-chosen test point sitting on a
    shoreline, not a bug in the polygon logic).
  - NO rotation / heading_offset in this pass -- a separate, sequenced
    follow-up per the author's original request.
  - Fixed 168-step / 28h budget for both controllers (see
    feedback_fixed_budget_comparisons in project memory: unbounded runtime
    makes closest-approach/endpoint comparisons meaningless).

Candidate A's own numbers (twice independently re-verified this session,
NOT taken from any earlier unverified search log): dD_to_E0 = 6.38 km,
dS1_to_E0 = 14.11 km, so J_A = 14.11 km. This is the number a candidate must
beat.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_candidateA_finemc_v2.py \
        > experiments/outputs/oecs/candidateA_finemc_v2.log 2>&1 &
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
SHORTLIST_JSON = os.path.join(OUT_DIR, "candidateA_finemc_v2_shortlist.json")
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
        "note": ("Ranked by J=max(dD_to_E0,dS1_to_E0) ascending (lower is "
                 "better -- closer to Fig 8's endpoint E0) among candidates "
                 f"that keep BOTH D and s1 paths within {CORRIDOR_DEV_MAX_KM} "
                 "km of candidate A's own paths. Smoothness (mean turning "
                 f"angle, radians) is reported but only breaks ties within "
                 f"{TIE_EPSILON_KM} km of J. No rotation/heading_offset used."),
    })


def make_record(la, lo, d_from_A, corridor_dev_d, corridor_dev_s1,
                d_end, s1_end, J, smooth_d, smooth_s1, n_evals, elapsed_s):
    return {
        "start_lat": la, "start_lon": lo, "d_from_candA_km": d_from_A,
        "corridor_dev_D_km": corridor_dev_d, "corridor_dev_s1_km": corridor_dev_s1,
        "D_end_lat": d_end[0], "D_end_lon": d_end[1],
        "s1_end_lat": s1_end[0], "s1_end_lon": s1_end[1],
        "dD_to_E0_km": km(d_end, E0), "dS1_to_E0_km": km(s1_end, E0),
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
    J_A = max(km(A_D_end, E0), km(A_S1_end, E0))
    smooth_A_D  = mean_turning_angle(A_D_path)
    smooth_A_S1 = mean_turning_angle(A_S1_path)
    smoothness_A = 0.5 * (smooth_A_D + smooth_A_S1)
    print(f"  Candidate A verified: D_end=({A_D_end[0]:.4f},{A_D_end[1]:.4f}) "
          f"dD={km(A_D_end,E0):.2f}km   s1_end=({A_S1_end[0]:.4f},{A_S1_end[1]:.4f}) "
          f"dS1={km(A_S1_end,E0):.2f}km   J_A={J_A:.2f}km")
    print(f"  Candidate A smoothness (mean turning angle): D={smooth_A_D:.4f} rad, "
          f"s1={smooth_A_S1:.4f} rad, mean={smoothness_A:.4f} rad\n")

    shortlist = load_shortlist()
    rng = np.random.default_rng(SEED)
    t_start_wall = time.time()
    deadline = t_start_wall + TOTAL_BUDGET_SECONDS
    n_evals = n_rej_land = n_rej_corridor = 0
    last_heartbeat = time.time()
    best_J = shortlist[0]["J_km"] if shortlist else J_A

    print(f"=== Fine MC search v2, r <= {RADIUS_KM} km around candidate A, "
          f"budget {TOTAL_BUDGET_SECONDS/3600:.1f} h, corridor cap "
          f"{CORRIDOR_DEV_MAX_KM} km, objective J=max(dD,dS1) to beat, "
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

        J = max(km(d_path[-1], E0), km(s1_path[-1], E0))
        smooth_d  = mean_turning_angle(d_path)
        smooth_s1 = mean_turning_angle(s1_path)

        worst_kept = rank_key(shortlist[-1]) if len(shortlist) >= TOP_K else (float("inf"), float("inf"))
        record = make_record(la, lo, d_from_A, cdev_d, cdev_s1,
                             d_path[-1], s1_path[-1], J, smooth_d, smooth_s1,
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
              f"dD_E0={r['dD_to_E0_km']:.2f} dS1_E0={r['dS1_to_E0_km']:.2f}  "
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
        match = (abs(km(d_path[-1], E0) - r["dD_to_E0_km"]) < 1e-6 and
                abs(km(s1_path[-1], E0) - r["dS1_to_E0_km"]) < 1e-6)
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
