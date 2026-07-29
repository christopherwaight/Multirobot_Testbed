"""
ocean_shared_start_search.py

Overnight search for a single SHARED start point, as close as possible to
the current Figure 8 start (34.400N, -120.390W), from which BOTH the D
tracker (separatrix_logic_c_step, Fig. 8) and the s1 tracker
(oecs_separatrix_step, Fig. 9) end near the current Figure 8 endpoint
(34.0412N, -120.2617W), at the SAME fixed 168-step (28h) budget both figures
already use.

Scoring is endpoint-only, at a fixed budget:

    E0 = (34.0412, -120.2617)                         # current Fig 8 endpoint
    J(start) = max( d_km(D_end(start), E0), d_km(s1_end(start), E0) )

max(), not a sum, so a start cannot win by making one controller excellent and
the other terrible. Path length is explicitly NOT part of the objective -- a
controller that loops can travel far and still end where it started, which is
not evidence of tracking the same structure (see
feedback_fixed_budget_comparisons in project memory). The budget is fixed at
168 steps for both controllers for the same reason: letting a slower
controller run indefinitely makes no sense, since a long enough random walk
eventually passes near anywhere -- past 168 steps the field clock clamps to
its final frame anyway (Ocean_HFR.py), so "more time" is a wander on a frozen
field, not continued tracking.

METHOD: dense independent uniform sampling in expanding hard-radius rings
around the current start: 1, 2, 3 km, and 5 km ONLY if 3 km has not produced
an acceptable J after its full time budget (ACCEPT_J_KM below; author's
stated preference is to avoid needing 5 km at all). This is deliberately NOT
simulated annealing or any other chain-based search -- an earlier version of
this script used annealing and it found a single needle (J=15.9 km at 0.56 km
from the current start) that a 5m perturbation broke by 20-45 km. Chains
drift onto one needle and over-sample it; independent uniform draws probe the
needle field honestly. The known, accepted outcome is that the eventual
winner is a specific point, not a basin, and must be re-verified at full
float precision before being reported (done automatically at the end and on
every checkpoint).

OVERNIGHT / CRASH-SAFE DESIGN:
  - Time-budgeted, not draw-count-budgeted: runs for TOTAL_BUDGET_SECONDS
    (default 6h), split evenly across whichever rings actually run.
  - Keeps a SHORTLIST of the best TOP_K distinct candidates (deduplicated
    within DEDUP_RADIUS_KM of each other), not a single winner -- the author
    wants a few options to choose among in the morning. No per-draw CSV
    logging (that log would grow to millions of rows over 6h and is not the
    deliverable). Progress is a fixed-size heartbeat, overwritten in place.
  - Shortlist is checkpointed to disk (atomic write: temp file + os.replace)
    every time an entry is added or improved, so a crash or kill at any
    point loses at most the evaluations since the last update, never the
    shortlist itself.
  - Resumable: if SHORTLIST_JSON already exists on startup, it is loaded and
    extended, so a restarted run does not throw away a previous session's
    candidates.

Same operating point as main_ocean_hfr_2km_ftle_overlay.py / oecs traverse
script (V_MAX=0.04, GAIN=1.8, SIM_STEPS=168, alpha_mom=0, stiction=0.002,
TIME_WARP=6000, pentagon_small_2km 0.7x formation). Gains for both
controllers are UNCHANGED from the validated scripts; only the shared start
is searched.

Requires the PentagonCluster.reset() OECS-state-clear fix (added this
session): oecs_separatrix_step stashes _oecs_prev_tangent/_oecs_banded/
_oecs_captured on the cluster, and without the fix a reused cluster's second
evaluation from an identical start is contaminated by the first's ending
state. Verified fixed this session; do not remove that clearing block from
PentagonCluster.reset().

Running (long; intended to be left overnight):
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_shared_start_search.py \
        > experiments/outputs/oecs/shared_start_search.log 2>&1 &

Checking progress without disturbing the run:
    tail -20 experiments/outputs/oecs/shared_start_search.log
    cat experiments/outputs/oecs/shared_start_search_shortlist.json
"""
import sys
import os
import time
import json
import tempfile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _coords_common import latlon_to_world as _latlon_to_world
from _coords_common import world_to_latlon as _world_to_latlon

# ============================================================================
# CONFIGURATION -- identical operating point to the validated Fig 8/9 scripts
# ============================================================================
FORMATION_CONFIG  = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX        = 0.04
SIM_STEPS    = 168      # 28h of field time, fixed for BOTH controllers
CONTROL_GAIN = 1.8
MOMENTUM_ALPHA     = 0.0
STICTION_THRESHOLD = 0.002
TIME_WARP    = 6000.0

# D tracker (Logic C, Primitive 7) gains -- unchanged from
# main_ocean_hfr_2km_ftle_overlay.py / ocean_hfr_2km_branch_sensitivity.py
EPS_RAW = 1e-3
EPS_DIM = 0.025

# s1 tracker (Primitive 11) gains -- unchanged from main_ocean_hfr_2km_traverse.py
G_PERP    = 1.0
S_TRIM    = 0.3
R_BAND    = 0.05
G_CAPTURE = 0.15

CURRENT_START = (34.40, -120.39)          # Fig 8/9's current start
E0            = (34.0412, -120.2617)      # Fig 8's current endpoint (D tracker)

# Ring schedule: 1/2/3 km always run their full share of the time budget.
# 5 km is appended ONLY if the best J after 1/2/3 km is still worse than
# ACCEPT_J_KM -- author's explicit preference is to avoid needing it.
CORE_RINGS_KM  = [1.0, 2.0, 3.0]
EXTRA_RING_KM  = 5.0
ACCEPT_J_KM    = 10.0     # "acceptable" cutoff -- below this, do not extend to 5km

TOTAL_BUDGET_SECONDS = 6 * 3600.0   # overnight; ~150ms/eval -> ~48k evals/ring at 3 rings
HEARTBEAT_EVERY_SEC  = 60.0
SEED = 20260726   # today's date; fixed so a given run is reproducible

# Shortlist, not a single winner: keep the best TOP_K DISTINCT candidates so
# there is a real choice to make in the morning rather than one point. Two
# candidates are "the same" (and only the better of the pair is kept) if
# their starts are within DEDUP_RADIUS_KM of each other -- otherwise, given
# how needle-like this landscape is (a 5m shift swung dS1 by 20-45 km in an
# earlier run), the shortlist would fill up with near-duplicates of one
# needle instead of showing genuinely different candidate starts/tradeoffs.
TOP_K           = 8
DEDUP_RADIUS_KM = 0.15

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(34.2))

here      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)
SHORTLIST_JSON = os.path.join(OUT_DIR, "shared_start_search_shortlist.json")


def km(p, q):
    """Local planar distance in km between two (lat, lon) pairs."""
    return float(np.hypot((p[0] - q[0]) * KM_PER_DEG_LAT,
                          (p[1] - q[1]) * KM_PER_DEG_LON))


def _run_endpoint(field, cluster, prim, lat, lon):
    sx, sy = _latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)     # also clears OECS tangent/mode state (fixed this session)
    if hasattr(field, "reset_clock"):
        field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        if hasattr(field, "step"):
            field.step(cluster.timestep * TIME_WARP)
    cx, cy = cluster.get_centroid()
    return _world_to_latlon(cx, cy, field.config)


def evaluate(field, cluster, prim_d, prim_s1, lat, lon):
    d_end  = _run_endpoint(field, cluster, prim_d,  lat, lon)
    s1_end = _run_endpoint(field, cluster, prim_s1, lat, lon)
    dD  = km(d_end, E0)
    dS1 = km(s1_end, E0)
    J   = max(dD, dS1)
    return d_end, s1_end, dD, dS1, J


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)   # atomic on POSIX: readers never see a partial file
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
            print(f"Resuming: found an existing shortlist of {len(lst)} "
                  f"candidates from a previous run, continuing from it.")
            return lst
        except Exception as e:
            print(f"Could not load existing {SHORTLIST_JSON} ({e}), starting fresh.")
    return []


def make_record(ring_km, la, lo, d_start, d_end, s1_end, dD, dS1, J, n_evals, elapsed_s):
    return {
        "ring_km": ring_km,
        "start_lat": la, "start_lon": lo, "d_start_km": d_start,
        "D_end_lat": d_end[0], "D_end_lon": d_end[1],
        "s1_end_lat": s1_end[0], "s1_end_lon": s1_end[1],
        "dD_km": dD, "dS1_km": dS1, "J_km": J,
        "n_evals_at_discovery": n_evals,
        "elapsed_seconds_at_discovery": elapsed_s,
    }


def maybe_insert(shortlist, record):
    """Insert `record` into `shortlist` if it's among the TOP_K best AND not
    a near-duplicate (within DEDUP_RADIUS_KM) of an already-kept candidate
    with equal or better J. Returns (new_shortlist, inserted: bool)."""
    la, lo, J = record["start_lat"], record["start_lon"], record["J_km"]

    for other in shortlist:
        if km((la, lo), (other["start_lat"], other["start_lon"])) < DEDUP_RADIUS_KM:
            if J < other["J_km"]:
                shortlist = [r for r in shortlist if r is not other] + [record]
                shortlist.sort(key=lambda r: r["J_km"])
                return shortlist, True
            return shortlist, False   # duplicate of a better or equal candidate

    if len(shortlist) < TOP_K:
        shortlist = shortlist + [record]
        shortlist.sort(key=lambda r: r["J_km"])
        return shortlist, True

    if J < shortlist[-1]["J_km"]:
        shortlist = shortlist[:-1] + [record]
        shortlist.sort(key=lambda r: r["J_km"])
        return shortlist, True

    return shortlist, False


def save_shortlist(shortlist):
    atomic_write_json(SHORTLIST_JSON, {
        "shortlist": shortlist,
        "note": ("Top candidates found so far, best (lowest J_km) first. "
                 "Each is a single verified point, not a robust region -- "
                 "this landscape showed meter-scale sensitivity in an "
                 "earlier run (a 5m shift swung dS1 by 20-45 km), so "
                 "candidates are deduplicated at "
                 f"{DEDUP_RADIUS_KM} km but are not guaranteed individually "
                 "robust until re-verified (see 'reverified' field, set at "
                 "the end of the run)."),
    })


def run_ring(field, cluster, prim_d, prim_s1, rng, ring_km, time_budget_s,
             shortlist, t_start_wall, total_evals_so_far):
    """Dense independent uniform sampling in the disk of radius ring_km
    around CURRENT_START. Returns (shortlist, n_evals_this_ring, total_evals)."""
    print(f"\n=== Ring r <= {ring_km:.0f} km, budget {time_budget_s/60:.0f} min ===")
    ring_deadline = time.time() + time_budget_s
    n_evals = 0
    last_heartbeat = time.time()
    ring_best_J = shortlist[0]["J_km"] if shortlist else float("inf")

    while time.time() < ring_deadline:
        r_km = ring_km * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        la = CURRENT_START[0] + (r_km * np.cos(theta)) / KM_PER_DEG_LAT
        lo = CURRENT_START[1] + (r_km * np.sin(theta)) / KM_PER_DEG_LON
        d_start = km((la, lo), CURRENT_START)

        d_end, s1_end, dD, dS1, J = evaluate(field, cluster, prim_d, prim_s1, la, lo)
        n_evals += 1
        total_evals_so_far += 1

        worst_kept = shortlist[-1]["J_km"] if len(shortlist) >= TOP_K else float("inf")
        if J < worst_kept:
            elapsed = time.time() - t_start_wall
            record = make_record(ring_km, la, lo, d_start, d_end, s1_end,
                                 dD, dS1, J, total_evals_so_far, elapsed)
            shortlist, inserted = maybe_insert(shortlist, record)
            if inserted:
                save_shortlist(shortlist)
                if J < ring_best_J:
                    ring_best_J = J
                print(f"  [{elapsed/60:6.1f} min, eval #{total_evals_so_far}] "
                      f"shortlist updated: J={J:.2f} km  start=({la:.6f},{lo:.6f}) "
                      f"[{d_start:.2f} km from current start]  "
                      f"dD={dD:.1f} dS1={dS1:.1f}  "
                      f"(shortlist size {len(shortlist)}, best {shortlist[0]['J_km']:.2f})")

        if time.time() - last_heartbeat > HEARTBEAT_EVERY_SEC:
            elapsed = time.time() - t_start_wall
            remaining = ring_deadline - time.time()
            print(f"  heartbeat: {elapsed/60:.1f} min elapsed, ring has "
                  f"{remaining/60:.1f} min left, {n_evals} evals this ring "
                  f"({total_evals_so_far} total), best-so-far J={ring_best_J:.2f} km")
            last_heartbeat = time.time()

    print(f"  Ring {ring_km:.0f} km done: {n_evals} evaluations, "
          f"best J so far = {ring_best_J:.2f} km, shortlist size {len(shortlist)}")
    return shortlist, n_evals, total_evals_so_far


def main():
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

    rng = np.random.default_rng(SEED)

    d0, s10, dD0, dS10, J0 = evaluate(field, cluster, prim_d, prim_s1, *CURRENT_START)
    print(f"Reference (current Fig 8/9 start {CURRENT_START}):")
    print(f"  D_end  = ({d0[0]:.4f}, {d0[1]:.4f})   dD  = {dD0:.1f} km")
    print(f"  s1_end = ({s10[0]:.4f}, {s10[1]:.4f})   dS1 = {dS10:.1f} km")
    print(f"  J0 = {J0:.2f} km")

    shortlist = load_shortlist()
    if not shortlist:
        ref = make_record("reference", CURRENT_START[0], CURRENT_START[1], 0.0,
                          d0, s10, dD0, dS10, J0, 0, 0.0)
        shortlist = [ref]
        save_shortlist(shortlist)

    t_start_wall = time.time()
    total_evals = 0
    per_ring_budget = TOTAL_BUDGET_SECONDS / len(CORE_RINGS_KM)

    for ring_km in CORE_RINGS_KM:
        shortlist, _, total_evals = run_ring(
            field, cluster, prim_d, prim_s1, rng, ring_km, per_ring_budget,
            shortlist, t_start_wall, total_evals)

    best_J = shortlist[0]["J_km"]
    if best_J >= ACCEPT_J_KM:
        print(f"\nBest after core rings (1/2/3 km) is J={best_J:.2f} km, "
              f">= ACCEPT_J_KM={ACCEPT_J_KM} km. Extending to the {EXTRA_RING_KM:.0f} km "
              f"ring as instructed (author's stated fallback).")
        shortlist, _, total_evals = run_ring(
            field, cluster, prim_d, prim_s1, rng, EXTRA_RING_KM, TOTAL_BUDGET_SECONDS * 0.5,
            shortlist, t_start_wall, total_evals)
    else:
        print(f"\nBest after core rings is J={best_J:.2f} km, "
              f"< ACCEPT_J_KM={ACCEPT_J_KM} km. Not extending to {EXTRA_RING_KM:.0f} km.")

    print(f"\n=== SEARCH COMPLETE: {total_evals} total evaluations, "
          f"{(time.time()-t_start_wall)/3600:.2f} h elapsed ===")
    print(f"Shortlist ({len(shortlist)} candidates), best first:")
    for i, r in enumerate(shortlist):
        print(f"  #{i+1}: J={r['J_km']:.2f} km  start=({r['start_lat']:.6f},"
              f"{r['start_lon']:.6f})  d_start={r['d_start_km']:.2f} km  "
              f"dD={r['dD_km']:.1f}  dS1={r['dS1_km']:.1f}  ring={r['ring_km']}")

    # ------------------------------------------------------------------
    # Full-precision re-verification of EVERY shortlisted candidate, not
    # just the top one -- the author picks among these in the morning, so
    # each option needs to be a real, re-confirmed point rather than a
    # rounded coordinate or a "near here" region. This landscape showed
    # meter-scale sensitivity in an earlier run (a 5m shift swung dS1 by
    # 20-45 km), so every candidate is re-run fresh (new field + cluster
    # objects, no shared state with the search loop above) and flagged if
    # it fails to reproduce.
    # ------------------------------------------------------------------
    print("\n=== Full-precision re-verification of the full shortlist ===")
    field2 = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster2 = PentagonCluster(FORMATION_CONFIG, field2,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)
    for r in shortlist:
        la, lo = r["start_lat"], r["start_lon"]
        de2, se2, dD2, dS12, J2 = evaluate(field2, cluster2, prim_d, prim_s1, la, lo)
        match = (abs(dD2 - r["dD_km"]) < 1e-9 and abs(dS12 - r["dS1_km"]) < 1e-9)
        r["reverified"] = bool(match)
        status = "OK" if match else "MISMATCH -- treat as artifact"
        print(f"  J={r['J_km']:.2f} km at ({la:.6f},{lo:.6f}): {status}")
        if not match:
            print(f"    re-run gave dD={dD2:.4f} dS1={dS12:.4f} "
                  f"(originally dD={r['dD_km']:.4f} dS1={r['dS1_km']:.4f})")

    shortlist = [r for r in shortlist if r.get("reverified", True) or r["ring_km"] == "reference"]
    shortlist.sort(key=lambda r: r["J_km"])
    save_shortlist(shortlist)

    print(f"\nFinal shortlist saved to {SHORTLIST_JSON}: "
          f"{len(shortlist)} re-verified candidates.")
    print("\nCAVEAT: this landscape has shown meter-scale sensitivity to the "
          "start point in earlier runs. Each shortlisted candidate is a "
          "single verified point, not a robust region -- reproducing one "
          "requires its exact coordinates, not an approximation of them.")


if __name__ == "__main__":
    main()
