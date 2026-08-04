"""
ocean_square_map_start_search.py

Finds a shared start for the D and s1 trackers under the isotropic ("square")
world map introduced 2026-08-03, replacing candidate A
(34.411906N, -120.392016W), which was tuned against the old anisotropic map.

Goal, in the author's words: more or less the same starting point, running
along the existing corridor as much as possible. So:

  objective    J = max(D's closest approach to E0 anywhere along its path,
                       s1's closest approach to E0 anywhere along its path),
                 E0 = (34.0412N, -120.2617W), the middle-island landfall of
                 the published figure. Lower is better. Closest approach
                 rather than endpoint distance, because a controller that
                 passes close and continues through should be credited.

  corridor     measured against the FROZEN legacy paths
               (experiments/outputs/oecs/legacy_corridor_reference.json,
               written by ocean_legacy_corridor_reference.py), in BOTH
               directions. corr* is how far the new path strays off the
               published one; cover* is how far the published one is from the
               new path. Deviation alone is not enough: a tracker that stalls
               near the corridor's start never strays and scores a perfect
               deviation, which is exactly what the first version of this
               sweep rewarded. Low corr AND low cover together mean the
               corridor was traversed.

  ridge        mean and max distance from each path to the 24-h forward FTLE
               ridge, the 95th-percentile set, from the cached field. This is
               the paper's actual claim ("both follow the dominant ridge")
               measured against the transport skeleton itself rather than
               against the published paths, so it does not inherit their
               choices.

  All of these are recorded for every candidate. Only J and a loose corridor
  reject drive the search; the rest exist so ranking can change afterward
  without re-running anything.

  operating point is FIXED at the paper's Ocean column. This search moves the
  start and nothing else.

Every evaluation is appended to a CSV as it happens, not just the shortlist.
The v3 search discarded 15k evaluations and kept 8, so re-ranking under a
different notion of "good" meant re-running the sweep. Ranking now lives in
ocean_rank_candidates.py and reads the CSV, so it costs seconds to change
what counts as a good candidate.

Search plan:
  Stage 1  deterministic hex grid over a disk around the legacy start, so the
           result is a map of the basins rather than a needle. Auto-widens
           once if the grid finds nothing decent.
  Stage 2  Monte Carlo refinement around the best stage-1 basins for the
           remaining time budget.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_square_map_start_search.py --hours 3 \
        > experiments/outputs/oecs/square_map_start_search.log 2>&1 &
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C
from _ftle_common import load_coastline_polygons

CSV_PATH = os.path.join(C.OUT_DIR, "square_map_start_search.csv")

STAGE1_RADIUS_KM = 6.0      # first grid; widened once if nothing decent turns up
STAGE1_WIDE_KM = 10.0
STAGE1_SPACING_KM = 0.3
WIDEN_IF_J_ABOVE_KM = 8.0

CORRIDOR_REJECT_MEAN_KM = 6.0   # loose: recorded for all, rejected only past this
STAGE2_TOP_BASINS = 12
STAGE2_SIGMA_KM = 0.6

HEARTBEAT_SEC = 120.0
SEED = 20260803

CSV_FIELDS = [
    "stage", "start_lat", "start_lon", "d_from_legacy_km",
    "J_km", "dD_closest_km", "dD_frac", "dS1_closest_km", "dS1_frac",
    "corrD_mean_km", "corrD_max_km", "corrS1_mean_km", "corrS1_max_km",
    "coverD_km", "coverS1_km", "ridgeD_mean_km", "ridgeD_max_km",
    "ridgeS1_mean_km", "ridgeS1_max_km",
    "D_end_lat", "D_end_lon", "s1_end_lat", "s1_end_lon",
    "D_branch", "s1_branch", "D_len_km", "s1_len_km",
    "smooth_D_rad", "smooth_s1_rad", "elapsed_s",
]


def hex_grid(center, radius_km, spacing_km):
    """Hex-packed lat/lon starts covering a disk. Hex, not square, for even coverage."""
    pts = []
    dy = spacing_km * np.sqrt(3) / 2.0
    n = int(np.ceil(radius_km / dy))
    for j in range(-n, n + 1):
        y = j * dy
        offset = 0.5 * spacing_km if (j % 2) else 0.0
        m = int(np.ceil(radius_km / spacing_km)) + 1
        for i in range(-m, m + 1):
            x = i * spacing_km + offset
            if np.hypot(x, y) > radius_km:
                continue
            pts.append((center[0] + y / C.KM_PER_DEG_LAT,
                        center[1] + x / C.KM_PER_DEG_LON))
    return pts


def evaluate(field, cluster, prim_d, prim_s1, lat, lon, legacy_d, legacy_s1, ridge):
    """Run both trackers from (lat, lon) and score them. Returns a CSV row dict."""
    d_path = C.run_traj(field, cluster, prim_d, lat, lon)
    s1_path = C.run_traj(field, cluster, prim_s1, lat, lon)

    dD, fD = C.closest_approach(d_path, C.E0)
    dS, fS = C.closest_approach(s1_path, C.E0)
    rD_mean, rD_max = C.dist_to_ridge_km(ridge, d_path)
    rS_mean, rS_max = C.dist_to_ridge_km(ridge, s1_path)

    return {
        "start_lat": lat, "start_lon": lon,
        "d_from_legacy_km": C.km((lat, lon), C.LEGACY_START),
        "J_km": max(dD, dS),
        "dD_closest_km": dD, "dD_frac": fD,
        "dS1_closest_km": dS, "dS1_frac": fS,
        "corrD_mean_km": C.path_dev_km(legacy_d, d_path),
        "corrD_max_km": C.path_dev_max_km(legacy_d, d_path),
        "corrS1_mean_km": C.path_dev_km(legacy_s1, s1_path),
        "corrS1_max_km": C.path_dev_max_km(legacy_s1, s1_path),
        "coverD_km": C.path_cover_km(legacy_d, d_path),
        "coverS1_km": C.path_cover_km(legacy_s1, s1_path),
        "ridgeD_mean_km": rD_mean, "ridgeD_max_km": rD_max,
        "ridgeS1_mean_km": rS_mean, "ridgeS1_max_km": rS_max,
        "D_end_lat": d_path[-1][0], "D_end_lon": d_path[-1][1],
        "s1_end_lat": s1_path[-1][0], "s1_end_lon": s1_path[-1][1],
        "D_branch": C.branch_of(d_path[-1]), "s1_branch": C.branch_of(s1_path[-1]),
        "D_len_km": C.path_length_km(d_path), "s1_len_km": C.path_length_km(s1_path),
        "smooth_D_rad": C.mean_turning_angle(d_path),
        "smooth_s1_rad": C.mean_turning_angle(s1_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0,
                    help="total wall-clock budget, both stages")
    ap.add_argument("--focus", type=str, default=None,
                    help="comma-separated criterion names from square_map_candidates.json "
                         "(e.g. 'balanced,corridor'). Skips the wide grid and refines around "
                         "those starts only: a dense local grid, then MC.")
    ap.add_argument("--focus-radius-km", type=float, default=1.2,
                    help="radius of the dense local grid around each focus start")
    ap.add_argument("--focus-spacing-km", type=float, default=0.1,
                    help="spacing of the dense local grid")
    ap.add_argument("--sigma-km", type=float, default=None,
                    help="MC jitter; defaults to STAGE2_SIGMA_KM, or 0.35 km in focus mode")
    ap.add_argument("--region", type=str, default=None,
                    help="lat_min,lat_max,lon_min,lon_max: sweep a rectangle instead of a "
                         "disk around the published start. Used to explore a preferred "
                         "direction rather than a neighbourhood.")
    ap.add_argument("--region-spacing-km", type=float, default=0.15)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + args.hours * 3600.0

    print("Loading coastline polygons for land-rejection...")
    polys = load_coastline_polygons(C.COAST_SHP, C.LON_MIN, C.LON_MAX, C.LAT_MIN, C.LAT_MAX)
    mpl_polys = [MplPath(p) for p in polys]

    def is_land(lat, lon):
        pt = np.array([[lon, lat]])
        return any(mp.contains_points(pt)[0] for mp in mpl_polys)

    legacy_start, legacy_d, legacy_s1 = C.load_legacy_reference()
    print(f"Legacy corridor reference loaded, start ({legacy_start[0]:.6f}, {legacy_start[1]:.6f})")

    ftle_cache = os.path.join(C.OUT_DIR, "ftle_cache_24h_u4.npz")
    ridge, n_ridge = C.ridge_tree(ftle_cache)
    if ridge is None:
        print(f"WARNING: no FTLE cache at {ftle_cache}, ridge distances will be NaN. "
              f"Generate it with experiments/ocean_candidate_overlays.py")
    else:
        print(f"FTLE ridge loaded: {n_ridge} points at or above the 95th percentile")

    field, cluster, prim_d, prim_s1 = C.build_trial()
    print(f"Map: isotropic_map={field.config.get('isotropic_map')}")

    fresh = not os.path.exists(CSV_PATH)
    os.makedirs(C.OUT_DIR, exist_ok=True)
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if fresh:
        writer.writeheader()
        fh.flush()

    seen = set()
    results = []
    if not fresh:
        # Prior rows count toward the stage-2 basin pool as well as the dedup
        # set, so a resumed run refines what the earlier run found.
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                seen.add((round(float(row["start_lat"]), 6), round(float(row["start_lon"]), 6)))
                for k in row:
                    if k not in ("stage", "D_branch", "s1_branch"):
                        row[k] = float(row[k])
                results.append(row)
        print(f"Resuming: {len(results)} evaluations already on file, "
              f"best J={min(r['J_km'] for r in results):.2f} km")
    n_land = n_done = 0
    last_hb = time.time()

    def record(row, stage):
        nonlocal last_hb
        row["stage"] = stage
        row["elapsed_s"] = round(time.time() - t_start, 1)
        writer.writerow(row)
        fh.flush()
        results.append(row)
        if time.time() - last_hb > HEARTBEAT_SEC:
            best = min(results, key=lambda r: r["J_km"])
            print(f"  heartbeat: {(time.time()-t_start)/60:.1f} min, "
                  f"{(deadline-time.time())/60:.1f} min left, {len(results)} evals, "
                  f"{n_land} land-rejected, best J={best['J_km']:.2f} km at "
                  f"({best['start_lat']:.6f},{best['start_lon']:.6f})", flush=True)
            last_hb = time.time()

    # ---- Region mode: sweep a rectangle, then MC in its best basins ---------
    if args.region:
        la0, la1, lo0, lo1 = [float(v) for v in args.region.split(",")]
        sp = args.region_spacing_km
        dlat = sp / C.KM_PER_DEG_LAT
        dlon = sp / C.KM_PER_DEG_LON
        lats = np.arange(la0, la1 + 0.5 * dlat, dlat)
        lons = np.arange(lo0, lo1 + 0.5 * dlon, dlon)
        pts = [(float(a), float(b)) for a in lats for b in lons
               if (round(a, 6), round(b, 6)) not in seen]
        print(f"\n=== Region sweep: lat [{la0}, {la1}], lon [{lo0}, {lo1}], "
              f"{sp} km spacing, {len(lats)}x{len(lons)} = {len(pts)} new points ===",
              flush=True)

        for (la, lo) in pts:
            if time.time() > deadline:
                print("  budget exhausted during the region sweep")
                break
            if is_land(la, lo):
                n_land += 1
                continue
            row = evaluate(field, cluster, prim_d, prim_s1, la, lo,
                           legacy_d, legacy_s1, ridge)
            record(row, "region")

        # Refine around the best of this region, by corridor quality rather
        # than by J, since the region was chosen for direction not for J.
        sigma = args.sigma_km if args.sigma_km is not None else 0.3
        reg = [r for r in results if r.get("stage") == "region"] or results
        pool = sorted(reg, key=lambda r: 0.25 * (r["corrD_mean_km"] + r["corrS1_mean_km"]
                                                 + r["coverD_km"] + r["coverS1_km"])
                      )[:STAGE2_TOP_BASINS]
        print(f"\n  MC around {len(pool)} best-corridor basins in the region, "
              f"sigma={sigma} km, {(deadline-time.time())/60:.0f} min left", flush=True)
        rng = np.random.default_rng(SEED)
        centers = [(r["start_lat"], r["start_lon"]) for r in pool]
        while time.time() < deadline and centers:
            c = centers[rng.integers(len(centers))]
            la = c[0] + rng.normal(0.0, sigma) / C.KM_PER_DEG_LAT
            lo = c[1] + rng.normal(0.0, sigma) / C.KM_PER_DEG_LON
            if is_land(la, lo):
                n_land += 1
                continue
            row = evaluate(field, cluster, prim_d, prim_s1, la, lo,
                           legacy_d, legacy_s1, ridge)
            record(row, "region_mc")

        fh.close()
        print(f"\n=== Region run done: {len(results)} evaluations in "
              f"{(time.time()-t_start)/60:.1f} min ===")
        print(f"CSV: {CSV_PATH}")
        return

    # ---- Focus mode: refine around named candidates only --------------------
    if args.focus:
        import json
        with open(os.path.join(C.OUT_DIR, "square_map_candidates.json")) as f:
            short = json.load(f)
        by_name = {c["criterion"]: c for c in short["candidates"]}
        wanted = [w.strip() for w in args.focus.split(",")]
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(f"--focus: no such criterion {missing}. "
                             f"Have: {sorted(by_name)}")
        centers = [(by_name[w]["start_lat"], by_name[w]["start_lon"]) for w in wanted]
        sigma = args.sigma_km if args.sigma_km is not None else 0.35

        print(f"\n=== Focus mode: refining around {', '.join(wanted)} ===")
        for w, c in zip(wanted, centers):
            print(f"    {w:<14} ({c[0]:.6f},{c[1]:.6f})  "
                  f"J={by_name[w]['J_km']:.2f} km")

        for w, c in zip(wanted, centers):
            pts = [p for p in hex_grid(c, args.focus_radius_km, args.focus_spacing_km)
                   if (round(p[0], 6), round(p[1], 6)) not in seen]
            print(f"\n  dense grid around {w}: r <= {args.focus_radius_km} km, "
                  f"{args.focus_spacing_km} km spacing, {len(pts)} points", flush=True)
            for (la, lo) in pts:
                if time.time() > deadline:
                    print("  budget exhausted during the dense grid")
                    break
                if is_land(la, lo):
                    n_land += 1
                    continue
                row = evaluate(field, cluster, prim_d, prim_s1, la, lo,
                               legacy_d, legacy_s1, ridge)
                record(row, f"focus_grid_{w}")

        print(f"\n  MC around the {len(centers)} focus starts, sigma={sigma} km, "
              f"{(deadline-time.time())/60:.0f} min left", flush=True)
        rng = np.random.default_rng(SEED)
        while time.time() < deadline:
            c = centers[rng.integers(len(centers))]
            la = c[0] + rng.normal(0.0, sigma) / C.KM_PER_DEG_LAT
            lo = c[1] + rng.normal(0.0, sigma) / C.KM_PER_DEG_LON
            if is_land(la, lo):
                n_land += 1
                continue
            row = evaluate(field, cluster, prim_d, prim_s1, la, lo,
                           legacy_d, legacy_s1, ridge)
            record(row, "focus_mc")

        fh.close()
        best = min(results, key=lambda r: r["J_km"])
        print(f"\n=== Focus run done: {len(results)} evaluations in "
              f"{(time.time()-t_start)/60:.1f} min ===")
        print(f"CSV: {CSV_PATH}")
        return

    # ---- Baseline: the legacy start under the new map -----------------------
    print("\nBaseline: legacy start under the square map...")
    base = evaluate(field, cluster, prim_d, prim_s1, *C.LEGACY_START,
                    legacy_d=legacy_d, legacy_s1=legacy_s1, ridge=ridge)
    record(base, "baseline")
    print(f"  J={base['J_km']:.2f} km  (D {base['dD_closest_km']:.2f}@{base['dD_frac']:.0%}, "
          f"s1 {base['dS1_closest_km']:.2f}@{base['dS1_frac']:.0%})  "
          f"corridor dev D={base['corrD_mean_km']:.2f} s1={base['corrS1_mean_km']:.2f} km")

    # ---- Stage 1: coarse grid ----------------------------------------------
    for radius in (STAGE1_RADIUS_KM, STAGE1_WIDE_KM):
        pts = [p for p in hex_grid(C.LEGACY_START, radius, STAGE1_SPACING_KM)
               if (round(p[0], 6), round(p[1], 6)) not in seen]
        if radius > STAGE1_RADIUS_KM:
            pts = [p for p in pts if C.km(p, C.LEGACY_START) > STAGE1_RADIUS_KM]
        print(f"\n=== Stage 1: hex grid, r <= {radius:.0f} km, {STAGE1_SPACING_KM} km "
              f"spacing, {len(pts)} points ===", flush=True)

        for (la, lo) in pts:
            if time.time() > deadline:
                print("  budget exhausted during stage 1")
                break
            if is_land(la, lo):
                n_land += 1
                continue
            row = evaluate(field, cluster, prim_d, prim_s1, la, lo, legacy_d, legacy_s1, ridge)
            record(row, f"grid{radius:.0f}")
            n_done += 1

        best_J = min(r["J_km"] for r in results)
        print(f"  stage 1 (r<={radius:.0f} km) done: {n_done} evals, best J={best_J:.2f} km")
        if best_J <= WIDEN_IF_J_ABOVE_KM or time.time() > deadline:
            break
        print(f"  best J still above {WIDEN_IF_J_ABOVE_KM} km, widening the grid")

    # ---- Stage 2: MC refinement around the best basins ----------------------
    ok = [r for r in results if r["corrD_mean_km"] <= CORRIDOR_REJECT_MEAN_KM]
    pool = sorted(ok or results, key=lambda r: r["J_km"])[:STAGE2_TOP_BASINS]
    print(f"\n=== Stage 2: MC refinement around {len(pool)} basins, "
          f"sigma={STAGE2_SIGMA_KM} km, {(deadline-time.time())/60:.0f} min left ===",
          flush=True)
    for r in pool:
        print(f"    basin J={r['J_km']:.2f} km at ({r['start_lat']:.6f},{r['start_lon']:.6f}) "
              f"[{r['d_from_legacy_km']:.2f} km from legacy]  "
              f"corr D={r['corrD_mean_km']:.2f} s1={r['corrS1_mean_km']:.2f}")

    rng = np.random.default_rng(SEED)
    centers = [(r["start_lat"], r["start_lon"]) for r in pool]
    while time.time() < deadline and centers:
        c = centers[rng.integers(len(centers))]
        la = c[0] + rng.normal(0.0, STAGE2_SIGMA_KM) / C.KM_PER_DEG_LAT
        lo = c[1] + rng.normal(0.0, STAGE2_SIGMA_KM) / C.KM_PER_DEG_LON
        if is_land(la, lo):
            n_land += 1
            continue
        row = evaluate(field, cluster, prim_d, prim_s1, la, lo, legacy_d, legacy_s1, ridge)
        record(row, "mc")

    fh.close()
    best = min(results, key=lambda r: r["J_km"])
    print(f"\n=== Done: {len(results)} evaluations in {(time.time()-t_start)/60:.1f} min ===")
    print(f"Best J={best['J_km']:.2f} km at ({best['start_lat']:.6f},{best['start_lon']:.6f}), "
          f"{best['d_from_legacy_km']:.2f} km from the legacy start")
    print(f"CSV: {CSV_PATH}")
    print("Rank and pick candidates with: venv/bin/python3 experiments/ocean_rank_candidates.py")


if __name__ == "__main__":
    main()
