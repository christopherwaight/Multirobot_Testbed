"""
ocean_rank_candidates.py

Selects a shortlist of shared-start candidates from the sweep CSV written by
ocean_square_map_start_search.py, and prints a comparison table.

Ranking is deliberately split from searching. The sweep records every
evaluation, so what counts as a good candidate can be changed and re-applied
in seconds without re-running anything.

There is no single best start, because the criteria pull apart: getting both
trackers close to the published landfall is not the same as keeping both on
the published corridor. So this picks one candidate per criterion, deduped by
MIN_SEPARATION_KM so the shortlist spans genuinely different behaviors rather
than five neighbors in one basin:

  closest        lowest J = max(D, s1 closest approach to E0). The pre-fix
                 search's objective, carried over unchanged.
  corridor       lowest corridor cost, the mean of deviation and coverage over
                 both trackers. Both halves are needed: a tracker that stalls
                 on the corridor's start has a perfect deviation and a terrible
                 coverage.
  ridge          lowest mean distance to the 24-h FTLE ridge over both
                 trackers. Measures the paper's claim directly, against the
                 transport skeleton rather than against the published paths.
  balanced       lowest sum of the z-scored J and corridor costs.
  nearest_start  lowest J among starts within NEAR_START_KM of the published
                 one, for the least disruptive change to the paper.

Candidates whose D tracker takes the west/offshore branch are dropped: the
paper's claim is a middle-island landfall, and a candidate that abandons it is
not a candidate for this figure.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_rank_candidates.py
    venv/bin/python3 experiments/ocean_rank_candidates.py --near-start-km 2.0
"""
import os
import sys
import csv
import argparse
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C

CSV_PATH = os.path.join(C.OUT_DIR, "square_map_start_search.csv")
SHORTLIST_JSON = os.path.join(C.OUT_DIR, "square_map_candidates.json")

MIN_SEPARATION_KM = 0.5
NEAR_START_KM = 1.5

# Smoothness penalty, km of equivalent cost per radian of mean turning angle.
# Deliberately small. Path length is a poor smoothness proxy because a path can
# be long by covering ground or long by circling; mean turning angle separates
# those. Typical turning angles here are 0.3 to 0.9 rad, so at this weight the
# term moves a candidate by a few hundred metres of equivalent cost: enough to
# break a near-tie in favour of the cleaner path, not enough to outrank a
# genuinely better corridor. Applies to the corridor and balanced criteria
# only; "closest" stays a pure J ranking.
SMOOTHNESS_KM_PER_RAD = 0.4

# Direction preference: the author wants the start further north and east
# ("up and to the right"). ne_km is the displacement from the published start
# projected onto the northeast unit vector, so it is positive to the northeast
# and negative to the southwest. It enters the balanced score as a z-scored
# bonus at this weight, and drives its own criterion, which takes the
# furthest-northeast start that still clears a corridor-quality bar. It is
# never allowed to buy a bad corridor.
NORTHEAST_WEIGHT = 0.6
NE_CORRIDOR_BAR_KM = 2.5    # max corridor cost a "northeast" pick may carry
NE_MAX_J_KM = 20.0          # and it must still approach the landfall
NUMERIC = {"start_lat", "start_lon", "d_from_legacy_km", "J_km", "dD_closest_km",
           "dD_frac", "dS1_closest_km", "dS1_frac", "corrD_mean_km", "corrD_max_km",
           "corrS1_mean_km", "corrS1_max_km", "coverD_km", "coverS1_km",
           "ridgeD_mean_km", "ridgeD_max_km", "ridgeS1_mean_km", "ridgeS1_max_km",
           "D_end_lat", "D_end_lon", "s1_end_lat", "s1_end_lon", "D_len_km",
           "s1_len_km", "smooth_D_rad", "smooth_s1_rad", "elapsed_s"}


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in list(row):
                if k in NUMERIC:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def zscore(vals):
    a = np.array(vals, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 1e-12 else np.zeros_like(a)


def pick(rows, key, chosen, label, note):
    """Best row by `key`, at least MIN_SEPARATION_KM from everything chosen."""
    for r in sorted(rows, key=key):
        p = (r["start_lat"], r["start_lon"])
        if all(C.km(p, (c["start_lat"], c["start_lon"])) >= MIN_SEPARATION_KM for c in chosen):
            r = dict(r)
            r["criterion"] = label
            r["criterion_note"] = note
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-start-km", type=float, default=NEAR_START_KM)
    ap.add_argument("--min-separation-km", type=float, default=MIN_SEPARATION_KM)
    ap.add_argument("--smoothness-km-per-rad", type=float, default=SMOOTHNESS_KM_PER_RAD,
                    help="smoothness penalty weight; 0 disables it")
    ap.add_argument("--northeast-weight", type=float, default=NORTHEAST_WEIGHT,
                    help="northeast bonus in the balanced score; 0 disables it")
    ap.add_argument("--ne-corridor-bar-km", type=float, default=NE_CORRIDOR_BAR_KM)
    ap.add_argument("--ne-max-j-km", type=float, default=NE_MAX_J_KM)
    args = ap.parse_args()

    rows = load_rows(CSV_PATH)
    print(f"{len(rows)} evaluations on file\n")

    baseline = next((r for r in rows if r["stage"] == "baseline"), None)
    viable = [r for r in rows if r["D_branch"] == "south/island"]
    print(f"{len(viable)} keep the D tracker on the middle-island branch "
          f"({len(rows) - len(viable)} dropped for going west/offshore)\n")
    if not viable:
        print("Nothing viable. Widen the sweep.")
        return

    # Corridor cost uses deviation AND coverage. Deviation alone rewards a
    # tracker that stalls near the corridor's start without traversing it.
    corr = [0.25 * (r["corrD_mean_km"] + r["corrS1_mean_km"]
                    + r["coverD_km"] + r["coverS1_km"]) for r in viable]
    ridge = [0.5 * (r["ridgeD_mean_km"] + r["ridgeS1_mean_km"]) for r in viable]
    root2 = float(np.sqrt(2.0))
    for r in viable:
        dn = (r["start_lat"] - C.LEGACY_START[0]) * C.KM_PER_DEG_LAT
        de = (r["start_lon"] - C.LEGACY_START[1]) * C.KM_PER_DEG_LON
        r["ne_km"] = (dn + de) / root2
    turn = [0.5 * (r["smooth_D_rad"] + r["smooth_s1_rad"]) for r in viable]
    smooth_pen = [args.smoothness_km_per_rad * t for t in turn]
    zJ = zscore([r["J_km"] for r in viable])
    zC = zscore([c + s for c, s in zip(corr, smooth_pen)])
    zNE = zscore([r["ne_km"] for r in viable])
    for r, c, g, t, s, a, b, ne in zip(viable, corr, ridge, turn, smooth_pen, zJ, zC, zNE):
        r["corr_cost_km"] = c
        r["ridge_cost_km"] = g
        r["turn_mean_rad"] = t
        r["smoothness_penalty_km"] = s
        r["corr_cost_smoothed_km"] = c + s
        r["balanced_score"] = float(a + b - args.northeast_weight * ne)

    chosen = []
    for key, label, note in [
        (lambda r: r["J_km"], "closest",
         "lowest J = max(D, s1 closest approach to the published landfall)"),
        (lambda r: r["corr_cost_smoothed_km"], "corridor",
         "both paths traverse the published corridor most closely, "
         "deviation and coverage together, with a small smoothness penalty"),
        (lambda r: r["ridge_cost_km"], "ridge",
         "both paths stay closest to the 24-h FTLE ridge"),
        (lambda r: r["balanced_score"], "balanced",
         "lowest sum of z-scored J and corridor cost, with a northeast bonus"),
    ]:
        c = pick(viable, key, chosen, label, note)
        if c:
            chosen.append(c)

    ne_ok = [r for r in viable
             if r["corr_cost_smoothed_km"] <= args.ne_corridor_bar_km
             and r["J_km"] <= args.ne_max_j_km]
    if ne_ok:
        c = pick(ne_ok, lambda r: -r["ne_km"], chosen, "northeast",
                 f"furthest northeast among starts with corridor cost <= "
                 f"{args.ne_corridor_bar_km} km and J <= {args.ne_max_j_km} km")
        if c:
            chosen.append(c)
    else:
        print(f"note: nothing clears the northeast quality bar "
              f"(corridor <= {args.ne_corridor_bar_km} km, J <= {args.ne_max_j_km} km)\n")

    near = [r for r in viable if r["d_from_legacy_km"] <= args.near_start_km]
    if near:
        c = pick(near, lambda r: r["J_km"], chosen, "nearest_start",
                 f"lowest J among starts within {args.near_start_km} km of the published start")
        if c:
            chosen.append(c)
    else:
        print(f"note: no viable candidate within {args.near_start_km} km of the published start\n")

    def line(tag, r):
        return (f"{tag:<14} {r['start_lat']:.5f},{r['start_lon']:.5f}  "
                f"{r.get('d_from_legacy_km', 0.0):>6.2f} {r['J_km']:>6.2f} "
                f"{r['dD_closest_km']:>6.2f} {r['dD_frac']:>4.0%} "
                f"{r['dS1_closest_km']:>7.2f} {r['dS1_frac']:>4.0%} "
                f"{r['corrD_mean_km']:>6.2f} {r['corrS1_mean_km']:>7.2f} "
                f"{r['coverD_km']:>6.2f} {r['coverS1_km']:>7.2f} "
                f"{r['ridgeD_mean_km']:>6.2f} {r['ridgeS1_mean_km']:>7.2f} "
                f"{r['smooth_D_rad']:>6.2f} {r['smooth_s1_rad']:>7.2f} "
                f"{r.get('ne_km', float('nan')):>7.2f}")

    hdr = (f"{'criterion':<14} {'start':<22} {'d_leg':>6} {'J':>6} "
           f"{'D_cls':>6} {'@':>5} {'s1_cls':>7} {'@':>5} "
           f"{'devD':>6} {'devS1':>7} {'covD':>6} {'covS1':>7} "
           f"{'rdgD':>6} {'rdgS1':>7} {'trnD':>6} {'trnS1':>7} {'NE_km':>7}")
    print(hdr)
    print("-" * len(hdr))
    if baseline:
        print(line("(published)", baseline))
    for r in chosen:
        print(line(r["criterion"], r))

    print("\nd_leg   km from the published start")
    print("J       max(D, s1) closest approach to E0, km")
    print("D_cls   closest approach to E0, km @ fraction of the run where it happened")
    print("dev*    mean deviation of the new path off the published corridor, km")
    print("cov*    mean distance from the published corridor to the new path, km")
    print("        (high cov with low dev means the tracker stalled instead of traversing)")
    print("rdg*    mean distance to the 24-h FTLE ridge, km")
    print(f"trn*    mean turning angle, rad. Lower is smoother. Penalised at "
          f"{args.smoothness_km_per_rad} km/rad in the corridor and balanced criteria only.")
    print("NE_km   displacement from the published start along the northeast direction,")
    print("        km. Positive is up and to the right. Higher is preferred.")

    C.atomic_write_json(SHORTLIST_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_rank_candidates.py",
        "source_csv": CSV_PATH,
        "n_evaluations": len(rows),
        "n_viable": len(viable),
        "min_separation_km": args.min_separation_km,
        "near_start_km": args.near_start_km,
        "E0_target": list(C.E0),
        "published_start": list(C.LEGACY_START),
        "baseline_published_start_under_square_map": baseline,
        "candidates": chosen,
    })
    print(f"\nSaved: {SHORTLIST_JSON}")
    print("Render them with: venv/bin/python3 experiments/ocean_candidate_overlays.py")


if __name__ == "__main__":
    main()
