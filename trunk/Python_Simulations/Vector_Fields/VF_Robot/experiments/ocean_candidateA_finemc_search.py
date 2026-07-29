"""
ocean_candidateA_finemc_search.py

Fine Monte Carlo search around candidate A (34.411906N, -120.392016W), the
best shared-start candidate found so far (see ocean_shared_start_search.py /
ocean_shared_start_closest_approach.py). Candidate A's true, twice-verified
numbers: D_end dist to E0 = 6.38 km, s1_end dist to E0 = 14.11 km (an earlier
in-session report of dS1=12.62 km for this same start was WRONG -- that came
from a search run killed before its re-verification step; every number here
is computed fresh, no unverified logged values are reused).

GOAL (per author, this session): candidate A's D+s1 corridor -- the two paths
overlap almost the entire ride down the channel in the paper's visualization
-- is the one the author wants. This script looks for a nearby start that
keeps that SAME corridor (small deviation from candidate A's own paths) while
extending how far south both controllers get before the budget runs out.
A coarse 0.5 km-step scan up/right from candidate A (done earlier this
session, not reproduced here) found the corridor breaks almost immediately in
that direction -- s1 jumps to a different branch entirely (deviation 3-19 km,
endpoint 38-73 km from E0 instead of 14). This script searches denser
(Monte Carlo, not a grid) and, per instruction, does not restrict to the
up/right quadrant -- any direction within RADIUS_KM of candidate A is fair
game, since the coarse scan showed the up/right assumption wasn't panning out
"as-is" and a finer search might find gaps the grid stepped over.

CONSTRAINTS, as specified:
  - Search radius: RADIUS_KM (default 4, i.e. within the stated 3-5 km range)
    around candidate A's own start, not the original Fig 8 start.
  - Land rejection by resampling, not a precomputed ocean-only mask: draw a
    candidate start, check it against the Natural Earth coastline polygons
    (same ne_10m_land.shp used by the FTLE plots) via point-in-polygon: if
    on land, discard and redraw. Cheaper than building a fine ocean mask
    up front for what is a small search region.
  - NO rotation / heading_offset in this pass -- orientation search is a
    separate follow-up, deliberately sequenced after this one so a change in
    result here isn't confounded with a change in heading.

OBJECTIVE: for each valid (ocean) candidate start, run D and s1 (same fixed
168-step/28h budget as every other variant -- see
feedback_fixed_budget_comparisons in project memory), and score:
  - corridor_dev = mean nearest-neighbor deviation of the candidate's D path
    from candidate A's D path, PLUS the same for s1 vs candidate A's s1 path
    (both must stay close to A's own corridor, not just one of them)
  - south_gain = how much further south (lower latitude) the WORSE of the
    two endpoints reaches, relative to candidate A's own worse endpoint
      south_gain = min(A_D_lat, A_s1_lat) - min(cand_D_lat, cand_s1_lat)
    positive means both controllers collectively got further south than A.
A candidate is only kept if corridor_dev stays under CORRIDOR_DEV_MAX_KM
(same corridor requirement); among those, ranked by south_gain descending.
If nothing beats south_gain=0 while satisfying the corridor constraint,
candidate A itself is the answer (author has said this is an acceptable
outcome).

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_candidateA_finemc_search.py \
        > experiments/outputs/oecs/candidateA_finemc.log 2>&1 &
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

RADIUS_KM              = 4.0     # search disk radius around candidate A
CORRIDOR_DEV_MAX_KM    = 1.0     # candidate's D and s1 paths must both stay
                                  # within this of candidate A's own paths
TOTAL_BUDGET_SECONDS   = 2 * 3600.0
HEARTBEAT_EVERY_SEC    = 60.0
SEED = 20260727

TOP_K, DEDUP_RADIUS_KM = 8, 0.1   # finer dedup than the earlier searches,
                                  # matches the finer search resolution here

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(34.2))

here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
OUT_DIR   = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)
SHORTLIST_JSON = os.path.join(OUT_DIR, "candidateA_finemc_shortlist.json")
COAST_SHP = os.path.join(repo_root, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")


def km(p, q):
    return float(np.hypot((p[0] - q[0]) * KM_PER_DEG_LAT, (p[1] - q[1]) * KM_PER_DEG_LON))


def path_dev_km(path_a_latlon, path_b_latlon):
    a = np.column_stack([path_a_latlon[:, 1] * KM_PER_DEG_LON, path_a_latlon[:, 0] * KM_PER_DEG_LAT])
    b = np.column_stack([path_b_latlon[:, 1] * KM_PER_DEG_LON, path_b_latlon[:, 0] * KM_PER_DEG_LAT])
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


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


def save_shortlist(shortlist, n_evals, n_rejected_land, n_rejected_corridor):
    atomic_write_json(SHORTLIST_JSON, {
        "shortlist": shortlist,
        "n_evals": n_evals,
        "n_rejected_land": n_rejected_land,
        "n_rejected_corridor": n_rejected_corridor,
        "candidate_A_reference": {
            "start_lat": CAND_A_START[0], "start_lon": CAND_A_START[1],
            "note": "dD=6.38km, dS1=14.11km, twice-verified fresh this session",
        },
        "note": ("Ranked by south_gain descending among candidates that keep "
                 "BOTH D and s1 paths within CORRIDOR_DEV_MAX_KM of candidate "
                 "A's own paths. south_gain > 0 means the candidate's worse "
                 "endpoint (min of D_end_lat, s1_end_lat) is further south "
                 "than candidate A's worst. No rotation/heading_offset used "
                 "in this pass."),
    })


def make_record(la, lo, d_from_A, corridor_dev_d, corridor_dev_s1,
                d_end, s1_end, south_gain, n_evals, elapsed_s):
    return {
        "start_lat": la, "start_lon": lo, "d_from_candA_km": d_from_A,
        "corridor_dev_D_km": corridor_dev_d, "corridor_dev_s1_km": corridor_dev_s1,
        "D_end_lat": d_end[0], "D_end_lon": d_end[1],
        "s1_end_lat": s1_end[0], "s1_end_lon": s1_end[1],
        "dD_to_E0_km": km(d_end, E0), "dS1_to_E0_km": km(s1_end, E0),
        "south_gain_km": south_gain,
        "n_evals_at_discovery": n_evals, "elapsed_seconds_at_discovery": elapsed_s,
    }


def maybe_insert(shortlist, record):
    """Rank by south_gain_km descending (bigger = better, more southward
    progress while keeping the corridor)."""
    la, lo, sg = record["start_lat"], record["start_lon"], record["south_gain_km"]
    for other in shortlist:
        if km((la, lo), (other["start_lat"], other["start_lon"])) < DEDUP_RADIUS_KM:
            if sg > other["south_gain_km"]:
                shortlist = [r for r in shortlist if r is not other] + [record]
                shortlist.sort(key=lambda r: -r["south_gain_km"])
                return shortlist, True
            return shortlist, False
    if len(shortlist) < TOP_K:
        shortlist = shortlist + [record]
        shortlist.sort(key=lambda r: -r["south_gain_km"])
        return shortlist, True
    if sg > shortlist[-1]["south_gain_km"]:
        shortlist = shortlist[:-1] + [record]
        shortlist.sort(key=lambda r: -r["south_gain_km"])
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
    A_worst_lat = min(A_D_end[0], A_S1_end[0])
    print(f"  Candidate A verified: D_end=({A_D_end[0]:.4f},{A_D_end[1]:.4f}) "
          f"dD={km(A_D_end,E0):.2f}km   s1_end=({A_S1_end[0]:.4f},{A_S1_end[1]:.4f}) "
          f"dS1={km(A_S1_end,E0):.2f}km")
    print(f"  Candidate A's worst (min) endpoint latitude: {A_worst_lat:.4f}\n")

    shortlist = load_shortlist()
    rng = np.random.default_rng(SEED)
    t_start_wall = time.time()
    deadline = t_start_wall + TOTAL_BUDGET_SECONDS
    n_evals = n_rej_land = n_rej_corridor = 0
    last_heartbeat = time.time()
    best_sg = shortlist[0]["south_gain_km"] if shortlist else -np.inf

    print(f"=== Fine MC search, r <= {RADIUS_KM} km around candidate A, "
          f"budget {TOTAL_BUDGET_SECONDS/3600:.1f} h, corridor cap "
          f"{CORRIDOR_DEV_MAX_KM} km ===")

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

        worst_lat = min(d_path[-1, 0], s1_path[-1, 0])
        south_gain = A_worst_lat - worst_lat   # positive = further south = better

        worst_kept = shortlist[-1]["south_gain_km"] if len(shortlist) >= TOP_K else -np.inf
        if south_gain > worst_kept:
            elapsed = time.time() - t_start_wall
            record = make_record(la, lo, d_from_A, cdev_d, cdev_s1,
                                 d_path[-1], s1_path[-1], south_gain, n_evals, elapsed)
            shortlist, inserted = maybe_insert(shortlist, record)
            if inserted:
                save_shortlist(shortlist, n_evals, n_rej_land, n_rej_corridor)
                if south_gain > best_sg:
                    best_sg = south_gain
                print(f"  [{elapsed/60:6.1f} min, eval #{n_evals}] shortlist updated: "
                      f"south_gain={south_gain:+.3f}km  start=({la:.6f},{lo:.6f}) "
                      f"[{d_from_A:.2f}km from A]  corridor_dev D={cdev_d:.2f} s1={cdev_s1:.2f}  "
                      f"dD_E0={km(d_path[-1],E0):.2f} dS1_E0={km(s1_path[-1],E0):.2f}  "
                      f"(shortlist size {len(shortlist)})")

        if time.time() - last_heartbeat > HEARTBEAT_EVERY_SEC:
            elapsed = time.time() - t_start_wall
            remaining = deadline - time.time()
            print(f"  heartbeat: {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min left, "
                  f"{n_evals} valid evals, {n_rej_land} land-rejected, "
                  f"{n_rej_corridor} corridor-rejected, best south_gain={best_sg:+.3f}km")
            last_heartbeat = time.time()

    print(f"\n=== SEARCH COMPLETE: {n_evals} valid evaluations "
          f"({n_rej_land} land-rejected, {n_rej_corridor} corridor-rejected), "
          f"{(time.time()-t_start_wall)/3600:.2f} h elapsed ===")

    if not shortlist:
        print("No candidate beat candidate A's own corridor/south-gain baseline "
              "(south_gain > -inf trivially true for the first find, but nothing "
              "qualified under the corridor constraint). Candidate A stands.")
        return

    print(f"Shortlist ({len(shortlist)} candidates), best south_gain first:")
    for i, r in enumerate(shortlist):
        print(f"  #{i+1}: south_gain={r['south_gain_km']:+.3f}km  "
              f"start=({r['start_lat']:.6f},{r['start_lon']:.6f})  "
              f"[{r['d_from_candA_km']:.2f}km from A]  "
              f"corridor_dev D={r['corridor_dev_D_km']:.2f} s1={r['corridor_dev_s1_km']:.2f}  "
              f"dD_E0={r['dD_to_E0_km']:.2f} dS1_E0={r['dS1_to_E0_km']:.2f}")

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
        print(f"  south_gain={r['south_gain_km']:+.3f}km at ({la:.6f},{lo:.6f}): {status}")

    shortlist = [r for r in shortlist if r.get("reverified", True)]
    shortlist.sort(key=lambda r: -r["south_gain_km"])
    save_shortlist(shortlist, n_evals, n_rej_land, n_rej_corridor)
    print(f"\nFinal shortlist saved to {SHORTLIST_JSON}: {len(shortlist)} re-verified candidates.")

    if shortlist and shortlist[0]["south_gain_km"] > 0:
        print(f"\nBest: south_gain={shortlist[0]['south_gain_km']:+.3f}km beats candidate A "
              f"while keeping the corridor (dev D={shortlist[0]['corridor_dev_D_km']:.2f}, "
              f"s1={shortlist[0]['corridor_dev_s1_km']:.2f} km).")
    else:
        print("\nNothing found beats candidate A's south extent while keeping the "
              "corridor constraint. Candidate A remains the best answer.")


if __name__ == "__main__":
    main()
