"""
ocean_param_mc.py

Monte Carlo sweep over every knob at once, with the corridor gate restored at
4 km, keeping a running top-40 so the survivors can be inspected for a pattern.

Why this exists, given ocean_param_search.py already searched the same five
knobs: that sweep dropped the corridor gate entirely, on the grounds that an
earlier version of it rejected 99% of samples. Re-gating its CSV afterward
showed the gate was not the problem. The best-scoring run in the whole file has
a corridor cost of 3.87 km and passes a 4 km gate untouched, and the best
achievable score is flat from a 4 km gate all the way out to no gate at all.
What the gate does remove is the degenerate kind of run, s1 abandoning the
ridge and driving a straight line at the landfall, which is what the ungated
ranking started surfacing.

The real defect was coverage, not the threshold. That sweep drew scale, gain,
c_max and heading from discrete grids, 7 x 6 x 5 x 12 = 2520 combinations, and
took 10507 samples, so roughly one sample per combination. Only 18 of its 3593
ungated rows landed on the paper's Ocean operating point at any heading, and
exactly one at heading 0. The corridor-tight basin that produced the earlier
start-only candidates (corridor cost 2.2 to 2.8 km) was never visited.

So: continuous sampling instead of grids, starts drawn from a disk centered on
the published start rather than a box, and the gate back in.

  knobs        start lat/lon, formation scale, initial heading, control gain,
               speed cap. All continuous, all independent, drawn fresh every
               sample. No annealing, no basin refinement, no hill climbing.
               A flat Monte Carlo, so the survivors are an unbiased sample of
               what works and can be read for structure.

  score        mean ridge distance over both trackers
                 + SMOOTH_W * mean turning angle over both trackers
               Identical to ocean_param_search.py, so the two CSVs are directly
               comparable and can be pooled.

  gates        branch     the D tracker must still make the middle-island landfall
               stalled    neither track may sit still, by net displacement
               corridor   max(corrD, corrS1, coverD, coverS1) <= CORRIDOR_MAX_KM
                          the worst of the four, so a run passes only if both
                          trackers stay near the published corridor AND both
                          cover it. Deviation alone would pass a tracker that
                          never left the corridor's start.

Gated rows are still written to the CSV, tagged with which gate caught them.
Ranking stays separate and re-readable, as before, so the gate width can be
changed afterward without re-running anything.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_param_mc.py --hours 3 \
        > experiments/outputs/oecs/param_mc.log 2>&1 &
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

CSV_PATH = os.path.join(C.OUT_DIR, "ocean_param_mc.csv")
TOP_PATH = os.path.join(C.OUT_DIR, "ocean_param_mc_top.json")

# Starts are drawn uniformly from a disk centered on the published start. The
# radius covers the whole box the grid sweep used, including the northeast band,
# while putting far more density near the legacy start than a box would.
START_RADIUS_KM = 6.0

SCALE_RANGE = (0.60, 1.50)      # uniform rescale of the pentagon
GAIN_RANGE = (1.00, 3.00)       # control gain k
VMAX_RANGE = (0.025, 0.060)     # speed cap c_max
# heading offset is uniform on [0, 2 pi)

CORRIDOR_MAX_KM = 4.0           # gate on max(corrD, corrS1, coverD, coverS1)
SMOOTH_W = 2.0                  # km of equivalent cost per radian of mean turning
NET_DISPLACEMENT_MIN_KM = 12.0  # a track that goes nowhere is not a candidate

TOP_N = 40
HEARTBEAT_SEC = 120.0
SEED = 20260805

CSV_FIELDS = [
    "start_lat", "start_lon", "scale", "rho_km", "heading_deg", "gain", "v_max",
    "score", "ridge_mean_km", "ridgeD_mean_km", "ridgeD_max_km",
    "ridgeS1_mean_km", "ridgeS1_max_km", "turn_mean_rad",
    "smooth_D_rad", "smooth_s1_rad",
    "J_km", "dD_closest_km", "dD_frac", "dS1_closest_km", "dS1_frac",
    "corrD_mean_km", "corrS1_mean_km", "coverD_km", "coverS1_km", "corr_cost_km",
    "D_end_lat", "D_end_lon", "s1_end_lat", "s1_end_lon",
    "D_branch", "s1_branch", "D_len_km", "s1_len_km", "netD_km", "netS1_km",
    "gated", "elapsed_s",
]


def sample_start(rng):
    """Uniform over a disk of START_RADIUS_KM around the published start."""
    r = START_RADIUS_KM * np.sqrt(rng.uniform())
    th = rng.uniform(0.0, 2.0 * np.pi)
    dlat = (r * np.sin(th)) / C.KM_PER_DEG_LAT
    dlon = (r * np.cos(th)) / C.KM_PER_DEG_LON
    return C.LEGACY_START[0] + dlat, C.LEGACY_START[1] + dlon


def evaluate(field, cluster, lat, lon, scale, heading, gain, v_max,
             legacy_d, legacy_s1, ridge):
    """One trial. Same measurements and same score as ocean_param_search.py."""
    from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

    C.set_formation_scale(cluster, scale)

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=v_max, eps_raw=C.EPS_RAW, eps_dim=C.EPS_DIM)
        return vx * gain, vy * gain

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=v_max, g_perp=C.G_PERP, s_trim=C.S_TRIM,
                                      r_band=C.R_BAND, g_capture=C.G_CAPTURE, s_capture=None)
        return vx * gain, vy * gain

    d_path = C.run_traj(field, cluster, prim_d, lat, lon, heading_offset=heading)
    s1_path = C.run_traj(field, cluster, prim_s1, lat, lon, heading_offset=heading)

    rD_mean, rD_max = C.dist_to_ridge_km(ridge, d_path)
    rS_mean, rS_max = C.dist_to_ridge_km(ridge, s1_path)
    tD, tS = C.mean_turning_angle(d_path), C.mean_turning_angle(s1_path)
    dD, fD = C.closest_approach(d_path, C.E0)
    dS, fS = C.closest_approach(s1_path, C.E0)

    corrD = C.path_dev_km(legacy_d, d_path)
    corrS = C.path_dev_km(legacy_s1, s1_path)
    covD = C.path_cover_km(legacy_d, d_path)
    covS = C.path_cover_km(legacy_s1, s1_path)
    corr_cost = max(corrD, corrS, covD, covS)

    ridge_mean = 0.5 * (rD_mean + rS_mean)
    turn_mean = 0.5 * (tD + tS)
    score = ridge_mean + SMOOTH_W * turn_mean

    netD = C.km((lat, lon), tuple(d_path[-1]))
    netS = C.km((lat, lon), tuple(s1_path[-1]))

    gated = ""
    if C.branch_of(d_path[-1]) != "south/island":
        gated = "branch"
    elif min(netD, netS) < NET_DISPLACEMENT_MIN_KM:
        gated = "stalled"
    elif corr_cost > CORRIDOR_MAX_KM:
        gated = "corridor"

    return {
        "start_lat": lat, "start_lon": lon, "scale": scale,
        "rho_km": C.rho_km(scale), "heading_deg": np.degrees(heading),
        "gain": gain, "v_max": v_max,
        "score": score, "ridge_mean_km": ridge_mean,
        "ridgeD_mean_km": rD_mean, "ridgeD_max_km": rD_max,
        "ridgeS1_mean_km": rS_mean, "ridgeS1_max_km": rS_max,
        "turn_mean_rad": turn_mean, "smooth_D_rad": tD, "smooth_s1_rad": tS,
        "J_km": max(dD, dS), "dD_closest_km": dD, "dD_frac": fD,
        "dS1_closest_km": dS, "dS1_frac": fS,
        "corrD_mean_km": corrD, "corrS1_mean_km": corrS,
        "coverD_km": covD, "coverS1_km": covS, "corr_cost_km": corr_cost,
        "D_end_lat": d_path[-1][0], "D_end_lon": d_path[-1][1],
        "s1_end_lat": s1_path[-1][0], "s1_end_lon": s1_path[-1][1],
        "D_branch": C.branch_of(d_path[-1]), "s1_branch": C.branch_of(s1_path[-1]),
        "D_len_km": C.path_length_km(d_path), "s1_len_km": C.path_length_km(s1_path),
        "netD_km": netD, "netS1_km": netS, "gated": gated,
    }


def write_top(top, n, n_gated, counts):
    C.atomic_write_json(TOP_PATH, {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": "experiments/ocean_param_mc.py",
        "source_csv": CSV_PATH,
        "corridor_max_km": CORRIDOR_MAX_KM,
        "smooth_w": SMOOTH_W,
        "start_radius_km": START_RADIUS_KM,
        "scale_range": SCALE_RANGE, "gain_range": GAIN_RANGE, "vmax_range": VMAX_RANGE,
        "n_evaluations": n,
        "n_gated": n_gated,
        "gated_breakdown": counts,
        "n_kept": len(top),
        "candidates": top,
    })


def main():
    global CORRIDOR_MAX_KM, START_RADIUS_KM

    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--corridor-km", type=float, default=CORRIDOR_MAX_KM)
    ap.add_argument("--radius-km", type=float, default=START_RADIUS_KM)
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args()

    CORRIDOR_MAX_KM = args.corridor_km
    START_RADIUS_KM = args.radius_km

    t_start = time.time()
    deadline = t_start + args.hours * 3600.0

    polys = load_coastline_polygons(C.COAST_SHP, C.LON_MIN, C.LON_MAX, C.LAT_MIN, C.LAT_MAX)
    mpl_polys = [MplPath(p) for p in polys]

    def is_land(lat, lon):
        pt = np.array([[lon, lat]])
        return any(mp.contains_points(pt)[0] for mp in mpl_polys)

    _, legacy_d, legacy_s1 = C.load_legacy_reference()
    ridge, n_ridge = C.ridge_tree(os.path.join(C.OUT_DIR, "ftle_cache_24h_u4.npz"))
    if ridge is None:
        raise SystemExit("No FTLE cache. Run ocean_candidate_overlays.py first.")
    print(f"FTLE ridge: {n_ridge} points at or above the 95th percentile")

    field, cluster, _, _ = C.build_trial()

    fresh = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if fresh:
        writer.writeheader()
        fh.flush()

    rng = np.random.default_rng(SEED)
    n = n_land = n_gated = 0
    counts = {"branch": 0, "stalled": 0, "corridor": 0}
    top = []
    last_hb = time.time()

    print(f"=== MC sweep: start(disk r<={START_RADIUS_KM} km) x scale{SCALE_RANGE} x "
          f"heading[0,360) x gain{GAIN_RANGE} x c_max{VMAX_RANGE}", flush=True)
    print(f"    corridor gate <= {CORRIDOR_MAX_KM} km, score = ridge + {SMOOTH_W} * turn, "
          f"keeping top {args.top}, {args.hours} h budget ===", flush=True)

    while time.time() < deadline:
        lat, lon = sample_start(rng)
        if is_land(lat, lon):
            n_land += 1
            continue
        scale = float(rng.uniform(*SCALE_RANGE))
        gain = float(rng.uniform(*GAIN_RANGE))
        v_max = float(rng.uniform(*VMAX_RANGE))
        heading = float(rng.uniform(0.0, 2.0 * np.pi))

        row = evaluate(field, cluster, lat, lon, scale, heading, gain, v_max,
                       legacy_d, legacy_s1, ridge)
        row["elapsed_s"] = round(time.time() - t_start, 1)
        writer.writerow(row)
        fh.flush()
        n += 1

        if row["gated"]:
            n_gated += 1
            counts[row["gated"]] += 1
        else:
            top.append(row)
            top.sort(key=lambda r: r["score"])
            del top[args.top:]
            if row is top[0]:
                print(f"  [{(time.time()-t_start)/60:5.1f} min, #{n}] NEW BEST "
                      f"score={row['score']:.3f} ridge={row['ridge_mean_km']:.2f} "
                      f"turn={row['turn_mean_rad']:.3f} corr={row['corr_cost_km']:.2f}  "
                      f"({row['start_lat']:.5f},{row['start_lon']:.5f}) "
                      f"scale={scale:.3f} hdg={row['heading_deg']:.1f} k={gain:.2f} "
                      f"c={v_max:.4f}  J={row['J_km']:.1f} s1={row['s1_branch']}",
                      flush=True)
            write_top(top, n, n_gated, counts)

        if time.time() - last_hb > HEARTBEAT_SEC:
            b = f"{top[0]['score']:.3f}" if top else "none"
            worst = f"{top[-1]['score']:.3f}" if len(top) == args.top else "n/a"
            print(f"  heartbeat: {(time.time()-t_start)/60:.1f} min, {n} evals, "
                  f"{len(top)} kept, best {b}, cutoff {worst}, "
                  f"gated {n_gated} (branch {counts['branch']}, stalled {counts['stalled']}, "
                  f"corridor {counts['corridor']}), {n_land} land", flush=True)
            last_hb = time.time()

    fh.close()
    write_top(top, n, n_gated, counts)
    print(f"\n=== Done: {n} evaluations, {n_gated} gated "
          f"(branch {counts['branch']}, stalled {counts['stalled']}, "
          f"corridor {counts['corridor']}), {len(top)} kept, in "
          f"{(time.time()-t_start)/60:.1f} min ===")
    print(f"CSV: {CSV_PATH}")
    print(f"Top {len(top)}: {TOP_PATH}")


if __name__ == "__main__":
    main()
