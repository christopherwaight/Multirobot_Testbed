"""
ocean_shared_start_closest_approach.py

Third variant of the shared-start search (see ocean_shared_start_search.py
for the first two: endpoint-matching with expanding 1/2/3/5 km rings). This
version relaxes the objective: instead of requiring both controllers' FINAL
position to land near the current Figure 8 endpoint, it only requires each
controller's PATH to pass within CLOSE_APPROACH_KM (1 km) of that endpoint
at some point during the run. The controllers may continue past it.

Constraints, as specified:
  - D and s1 share a single start point.
  - That start must be within START_RADIUS_KM (3 km) of the current Figure 8
    start (34.400N, -120.390W).
  - Each controller's own trajectory (from PentagonCluster.get_center_history,
    the full centroid path, not just the endpoint) must come within
    CLOSE_APPROACH_KM (1 km) of E0 = (34.0412N, -120.2617W) -- the current
    Figure 8 endpoint -- at some point along its 168-step run. This is a
    minimum-over-the-path distance, evaluated at the same fixed 168-step
    (28h) budget as every other variant (see feedback_fixed_budget_comparisons
    in project memory for why the budget must stay fixed: an unbounded walk
    passes near everywhere eventually, which would make this metric
    meaningless).

Objective: J(start) = max( min_t d(D_path(t), E0), min_t d(s1_path(t), E0) )
i.e. the worse of the two controllers' own closest approaches. max(), not a
sum, for the same reason as the endpoint-matching search: a start should not
win by making one controller's approach excellent and the other's terrible.

This is a genuinely easier target than the endpoint-matching search (which
found dS1 stuck at a 12.6-12.9 km floor across 1/2/3/5 km rings and 130k+
evaluations): a path can swing near E0 and continue elsewhere, whereas the
endpoint search required the RUN TO END there. Reported alongside J for each
candidate: the closest-approach distance for each controller AND the
fraction of the run elapsed when that approach occurred (an approach in the
last few steps is a near-endpoint result; an approach at 20% of the run and
then departure means the path visited and moved on, which is the situation
this variant is designed to find and the endpoint variant could not credit).

Uses the SAME dense-independent-uniform-sampling method as
ocean_shared_start_search.py (not a Markov chain -- see that file's
docstring for why: this landscape has shown meter-scale sensitivity, and a
chain that drifts onto one needle over-samples it instead of exploring), and
the SAME shortlist/checkpoint/re-verification machinery, so results are
comparable and equally crash-safe for an unattended run.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_shared_start_closest_approach.py \
        > experiments/outputs/oecs/shared_start_closest_approach.log 2>&1 &

Checking progress:
    tail -20 experiments/outputs/oecs/shared_start_closest_approach.log
    cat experiments/outputs/oecs/shared_start_closest_approach_shortlist.json
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

EPS_RAW = 1e-3   # D tracker (Logic C, Primitive 7) gains, unchanged
EPS_DIM = 0.025

G_PERP    = 1.0   # s1 tracker (Primitive 11) gains, unchanged
S_TRIM    = 0.3
R_BAND    = 0.05
G_CAPTURE = 0.15

CURRENT_START = (34.40, -120.39)          # Fig 8/9's current start
E0            = (34.0412, -120.2617)      # Fig 8's current endpoint (D tracker)

START_RADIUS_KM    = 3.0    # hard cap: shared start must be within this of CURRENT_START
CLOSE_APPROACH_KM  = 1.0    # each path must come within this of E0 at some point

TOTAL_BUDGET_SECONDS = 4 * 3600.0   # single ring this time, no expansion needed
HEARTBEAT_EVERY_SEC  = 60.0
SEED = 20260727

TOP_K           = 8
DEDUP_RADIUS_KM = 0.15

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(34.2))

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")
os.makedirs(OUT_DIR, exist_ok=True)
SHORTLIST_JSON = os.path.join(OUT_DIR, "shared_start_closest_approach_shortlist.json")


def km(p, q):
    return float(np.hypot((p[0] - q[0]) * KM_PER_DEG_LAT,
                          (p[1] - q[1]) * KM_PER_DEG_LON))


def _run_closest_approach(field, cluster, prim, lat, lon):
    """Run one trajectory; return (min_dist_km_to_E0, frac_of_run_at_min,
    end_lat, end_lon)."""
    sx, sy = _latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy)
    if hasattr(field, "reset_clock"):
        field.reset_clock()
    for _ in range(SIM_STEPS):
        cluster.move(prim)
        if hasattr(field, "step"):
            field.step(cluster.timestep * TIME_WARP)

    center_hist = cluster.get_center_history()   # (steps, 2) world coords
    latlon_hist = np.array([_world_to_latlon(p[0], p[1], field.config)
                            for p in center_hist])
    d = np.hypot((latlon_hist[:, 0] - E0[0]) * KM_PER_DEG_LAT,
                (latlon_hist[:, 1] - E0[1]) * KM_PER_DEG_LON)
    i_min = int(np.argmin(d))
    end_lat, end_lon = latlon_hist[-1]
    return float(d[i_min]), i_min / max(len(d) - 1, 1), float(end_lat), float(end_lon)


def evaluate(field, cluster, prim_d, prim_s1, lat, lon):
    dD_min, dD_frac, dD_elat, dD_elon = _run_closest_approach(field, cluster, prim_d, lat, lon)
    dS1_min, dS1_frac, dS1_elat, dS1_elon = _run_closest_approach(field, cluster, prim_s1, lat, lon)
    J = max(dD_min, dS1_min)
    return {
        "dD_closest_km": dD_min, "dD_closest_frac": dD_frac,
        "D_end_lat": dD_elat, "D_end_lon": dD_elon,
        "dS1_closest_km": dS1_min, "dS1_closest_frac": dS1_frac,
        "s1_end_lat": dS1_elat, "s1_end_lon": dS1_elon,
        "J_km": J,
    }


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
            print(f"Resuming: found an existing shortlist of {len(lst)} "
                  f"candidates, continuing from it.")
            return lst
        except Exception as e:
            print(f"Could not load existing {SHORTLIST_JSON} ({e}), starting fresh.")
    return []


def save_shortlist(shortlist):
    atomic_write_json(SHORTLIST_JSON, {
        "shortlist": shortlist,
        "note": ("Closest-approach variant: J is the worse of the two "
                 "controllers' MINIMUM distance to the Fig 8 endpoint E0 "
                 "along their own path (not their final position). "
                 "*_closest_frac is how far into the 168-step run that "
                 "closest approach occurred (0=start, 1=final step) -- a "
                 "value near 1.0 means the controller was still approaching "
                 "at the end, which is really an endpoint result; a low "
                 "value means it passed near E0 early and then moved on."),
    })


def make_record(la, lo, d_start, ev, n_evals, elapsed_s):
    rec = {"start_lat": la, "start_lon": lo, "d_start_km": d_start,
           "n_evals_at_discovery": n_evals, "elapsed_seconds_at_discovery": elapsed_s}
    rec.update(ev)
    return rec


def maybe_insert(shortlist, record):
    la, lo, J = record["start_lat"], record["start_lon"], record["J_km"]
    for other in shortlist:
        if km((la, lo), (other["start_lat"], other["start_lon"])) < DEDUP_RADIUS_KM:
            if J < other["J_km"]:
                shortlist = [r for r in shortlist if r is not other] + [record]
                shortlist.sort(key=lambda r: r["J_km"])
                return shortlist, True
            return shortlist, False
    if len(shortlist) < TOP_K:
        shortlist = shortlist + [record]
        shortlist.sort(key=lambda r: r["J_km"])
        return shortlist, True
    if J < shortlist[-1]["J_km"]:
        shortlist = shortlist[:-1] + [record]
        shortlist.sort(key=lambda r: r["J_km"])
        return shortlist, True
    return shortlist, False


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

    ev0 = evaluate(field, cluster, prim_d, prim_s1, *CURRENT_START)
    print(f"Reference (current Fig 8/9 start {CURRENT_START}):")
    print(f"  D  closest approach = {ev0['dD_closest_km']:.2f} km "
          f"(at {ev0['dD_closest_frac']:.0%} of run)")
    print(f"  s1 closest approach = {ev0['dS1_closest_km']:.2f} km "
          f"(at {ev0['dS1_closest_frac']:.0%} of run)")
    print(f"  J0 = {ev0['J_km']:.2f} km")
    print(f"\nConstraints: start within {START_RADIUS_KM} km of current start, "
          f"target closest-approach <= {CLOSE_APPROACH_KM} km for both.\n")

    shortlist = load_shortlist()
    if not shortlist:
        ref = make_record(CURRENT_START[0], CURRENT_START[1], 0.0, ev0, 0, 0.0)
        shortlist = [ref]
        save_shortlist(shortlist)

    t_start_wall = time.time()
    deadline = t_start_wall + TOTAL_BUDGET_SECONDS
    n_evals = 0
    last_heartbeat = time.time()
    best_J = shortlist[0]["J_km"]

    print(f"=== Searching r <= {START_RADIUS_KM:.0f} km, "
          f"budget {TOTAL_BUDGET_SECONDS/3600:.1f} h ===")

    while time.time() < deadline:
        r_km = START_RADIUS_KM * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        la = CURRENT_START[0] + (r_km * np.cos(theta)) / KM_PER_DEG_LAT
        lo = CURRENT_START[1] + (r_km * np.sin(theta)) / KM_PER_DEG_LON
        d_start = km((la, lo), CURRENT_START)

        ev = evaluate(field, cluster, prim_d, prim_s1, la, lo)
        n_evals += 1

        worst_kept = shortlist[-1]["J_km"] if len(shortlist) >= TOP_K else float("inf")
        if ev["J_km"] < worst_kept:
            elapsed = time.time() - t_start_wall
            record = make_record(la, lo, d_start, ev, n_evals, elapsed)
            shortlist, inserted = maybe_insert(shortlist, record)
            if inserted:
                save_shortlist(shortlist)
                if ev["J_km"] < best_J:
                    best_J = ev["J_km"]
                print(f"  [{elapsed/60:6.1f} min, eval #{n_evals}] shortlist updated: "
                      f"J={ev['J_km']:.2f} km  start=({la:.6f},{lo:.6f}) "
                      f"[{d_start:.2f} km from current start]  "
                      f"D_closest={ev['dD_closest_km']:.2f}@{ev['dD_closest_frac']:.0%}  "
                      f"s1_closest={ev['dS1_closest_km']:.2f}@{ev['dS1_closest_frac']:.0%}  "
                      f"(shortlist size {len(shortlist)}, best {shortlist[0]['J_km']:.2f})")

        if time.time() - last_heartbeat > HEARTBEAT_EVERY_SEC:
            elapsed = time.time() - t_start_wall
            remaining = deadline - time.time()
            print(f"  heartbeat: {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min "
                  f"left, {n_evals} evals, best-so-far J={best_J:.2f} km")
            last_heartbeat = time.time()

    print(f"\n=== SEARCH COMPLETE: {n_evals} evaluations, "
          f"{(time.time()-t_start_wall)/3600:.2f} h elapsed ===")
    print(f"Shortlist ({len(shortlist)} candidates), best first:")
    for i, r in enumerate(shortlist):
        print(f"  #{i+1}: J={r['J_km']:.2f} km  start=({r['start_lat']:.6f},"
              f"{r['start_lon']:.6f})  d_start={r['d_start_km']:.2f} km  "
              f"D_closest={r['dD_closest_km']:.2f}@{r['dD_closest_frac']:.0%}  "
              f"s1_closest={r['dS1_closest_km']:.2f}@{r['dS1_closest_frac']:.0%}")

    # Full-precision re-verification of the whole shortlist, same rationale
    # as ocean_shared_start_search.py: this landscape has shown meter-scale
    # sensitivity, so every reported candidate must reproduce independently.
    print("\n=== Full-precision re-verification of the full shortlist ===")
    field2 = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    cluster2 = PentagonCluster(FORMATION_CONFIG, field2,
                               momentum_alpha=MOMENTUM_ALPHA,
                               stiction_threshold=STICTION_THRESHOLD)
    for r in shortlist:
        la, lo = r["start_lat"], r["start_lon"]
        ev2 = evaluate(field2, cluster2, prim_d, prim_s1, la, lo)
        match = (abs(ev2["dD_closest_km"] - r["dD_closest_km"]) < 1e-9 and
                abs(ev2["dS1_closest_km"] - r["dS1_closest_km"]) < 1e-9)
        r["reverified"] = bool(match)
        status = "OK" if match else "MISMATCH -- treat as artifact"
        print(f"  J={r['J_km']:.2f} km at ({la:.6f},{lo:.6f}): {status}")

    shortlist = [r for r in shortlist if r.get("reverified", True)]
    shortlist.sort(key=lambda r: r["J_km"])
    save_shortlist(shortlist)
    print(f"\nFinal shortlist saved to {SHORTLIST_JSON}: "
          f"{len(shortlist)} re-verified candidates.")

    n_meeting_target = sum(1 for r in shortlist
                           if r["dD_closest_km"] <= CLOSE_APPROACH_KM
                           and r["dS1_closest_km"] <= CLOSE_APPROACH_KM)
    print(f"\n{n_meeting_target}/{len(shortlist)} shortlisted candidates meet the "
          f"stated target (both closest-approaches <= {CLOSE_APPROACH_KM} km).")


if __name__ == "__main__":
    main()
