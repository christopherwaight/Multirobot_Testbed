"""
ocean_candidate_jitter.py

Start-sensitivity of the D tracker's landfall, for every shortlisted candidate
from ocean_rank_candidates.py. This is the square-map replacement for the
"84 of 100 jittered starts landing on the same middle-island branch" claim in
the paper's Discussion and Conclusion, which was measured under the old map
and around the old start.

Protocol follows ocean_hfr_2km_branch_sensitivity.py: a 10x10 grid of starts
around the candidate, each run 168 steps with the D tracker at the paper's
operating point, each final position classified by which side of the
bifurcation it lands on.

One deliberate change. The original jitter was +/-0.05 deg on both axes, which
under the old map meant +/-5.55 km north but +/-4.61 km east, an ellipse. The
grid here is +/-5.5 km on both axes, so the jitter is square like the map now
is. That makes the count not directly comparable to the published 84, and the
protocol change should be stated wherever the new number is used.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_candidate_jitter.py
"""
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C

SHORTLIST_JSON = os.path.join(C.OUT_DIR, "square_map_candidates.json")
OUT_JSON = os.path.join(C.OUT_DIR, "square_map_candidate_jitter.json")

JITTER_KM = 5.5
N_SIDE = 10


def jitter_grid(center):
    offs = np.linspace(-JITTER_KM, JITTER_KM, N_SIDE)
    return [(center[0] + dy / C.KM_PER_DEG_LAT, center[1] + dx / C.KM_PER_DEG_LON)
            for dy in offs for dx in offs]


def main():
    with open(SHORTLIST_JSON) as f:
        short = json.load(f)

    field, cluster, prim_d, _ = C.build_trial()

    targets = []
    base = short.get("baseline_published_start_under_square_map")
    if base:
        targets.append(("published start", (base["start_lat"], base["start_lon"])))
    for c in short["candidates"]:
        targets.append((c["criterion"], (c["start_lat"], c["start_lon"])))

    results = []
    for name, center in targets:
        starts = jitter_grid(center)
        branches, ends = [], []
        for (la, lo) in starts:
            path = C.run_traj(field, cluster, prim_d, la, lo)
            ends.append([float(path[-1][0]), float(path[-1][1])])
            branches.append(C.branch_of(path[-1]))
        n_island = sum(b == "south/island" for b in branches)
        # Spread of the landing points that stayed on the island branch, as a
        # second read on how tight the outcome is.
        isl = np.array([e for e, b in zip(ends, branches) if b == "south/island"])
        spread = (float(np.median([C.km(tuple(p), tuple(np.median(isl, axis=0)))
                                   for p in isl])) if len(isl) > 2 else float("nan"))
        print(f"{name:<22} {n_island:>3}/{len(starts)} on the middle-island branch   "
              f"median landing spread {spread:.2f} km")
        results.append({
            "name": name, "center": list(center),
            "n_total": len(starts), "n_island_branch": n_island,
            "fraction": n_island / len(starts),
            "median_landing_spread_km": spread,
            "ends": ends, "branches": branches,
        })

    C.atomic_write_json(OUT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_candidate_jitter.py",
        "protocol": {
            "jitter_km": JITTER_KM, "n_side": N_SIDE,
            "sim_steps": C.SIM_STEPS, "tracker": "D",
            "note": ("Square +/-5.5 km jitter on both axes. The published 84/100 used "
                     "+/-0.05 deg on both axes, an ellipse under the old map, so the "
                     "counts are not directly comparable."),
        },
        "results": results,
    })
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
