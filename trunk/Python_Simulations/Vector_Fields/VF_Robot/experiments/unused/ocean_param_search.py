"""
ocean_param_search.py

Searches the operating point as well as the start, looking for a run where both
trackers hug the FTLE ridge smoothly.

The start-only sweep (ocean_square_map_start_search.py) held the paper's Ocean
column fixed and moved only the start. That found starts that reach the
landfall, but none whose s1 track follows the ridge cleanly, because how
tightly a tracker holds a structure is set by the formation size, the initial
heading and the gains, not by where it began. So this varies all five:

  start            lat, lon over a region
  formation scale  uniform rescale of the pentagon, rho = 0.105 * scale
  heading offset   rigid rotation of the formation at t = 0
  control gain     k
  speed cap        c_max

Objective, "hugs the FTLE smoothly":

    score = mean ridge distance over both trackers
          + SMOOTH_W * mean turning angle over both trackers

with hard gates rather than penalties for the things that are not tradeable:
the D tracker must still make the middle-island landfall, and neither track may
stall, measured as net displacement from start to end.

Traversal is deliberately NOT measured against the published paths here. Those
were produced under one particular formation and gain, and once those are being
varied, "how far is this from the old track" stops being evidence that the run
went anywhere. An early version gated on distance to the published corridor and
rejected 99% of samples, including good ridge-hugging runs whose only fault was
taking a different line. Net displacement asks the question directly.

Ridge distance is to the 95th-percentile set of the 24-h forward FTLE field,
the same definition the branch-sensitivity script uses.

Every evaluation is appended to its own CSV. Ranking is separate, as before, so
the objective can be reweighted without re-running.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 -u experiments/ocean_param_search.py --hours 1.5 \
        > experiments/outputs/oecs/param_search.log 2>&1 &
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

CSV_PATH = os.path.join(C.OUT_DIR, "ocean_param_search.csv")

# Region to draw starts from. Covers the channel entrance and the northeast
# band the author asked for, and contains every start the earlier sweep liked.
LAT_RANGE = (34.375, 34.445)
LON_RANGE = (-120.410, -120.300)

SCALE_CHOICES = (0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
GAIN_CHOICES = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0)
VMAX_CHOICES = (0.025, 0.03, 0.04, 0.05, 0.06)
N_HEADINGS = 12          # heading offsets sampled on a 2pi/12 grid

SMOOTH_W = 2.0           # km of equivalent cost per radian of mean turning angle
NET_DISPLACEMENT_MIN_KM = 12.0   # a track that goes nowhere is not a candidate
HEARTBEAT_SEC = 120.0
SEED = 20260803

CSV_FIELDS = [
    "start_lat", "start_lon", "scale", "rho_km", "heading_deg", "gain", "v_max",
    "score", "ridge_mean_km", "ridgeD_mean_km", "ridgeD_max_km",
    "ridgeS1_mean_km", "ridgeS1_max_km", "turn_mean_rad",
    "smooth_D_rad", "smooth_s1_rad",
    "J_km", "dD_closest_km", "dD_frac", "dS1_closest_km", "dS1_frac",
    "corrD_mean_km", "corrS1_mean_km", "coverD_km", "coverS1_km",
    "D_end_lat", "D_end_lon", "s1_end_lat", "s1_end_lon",
    "D_branch", "s1_branch", "D_len_km", "s1_len_km", "netD_km", "netS1_km",
    "gated", "elapsed_s",
]


def evaluate(field, cluster, lat, lon, scale, heading, gain, v_max,
             legacy_d, legacy_s1, ridge):
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
    covD = C.path_cover_km(legacy_d, d_path)
    covS = C.path_cover_km(legacy_s1, s1_path)

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
        "corrD_mean_km": C.path_dev_km(legacy_d, d_path),
        "corrS1_mean_km": C.path_dev_km(legacy_s1, s1_path),
        "coverD_km": covD, "coverS1_km": covS,
        "D_end_lat": d_path[-1][0], "D_end_lon": d_path[-1][1],
        "s1_end_lat": s1_path[-1][0], "s1_end_lon": s1_path[-1][1],
        "D_branch": C.branch_of(d_path[-1]), "s1_branch": C.branch_of(s1_path[-1]),
        "D_len_km": C.path_length_km(d_path), "s1_len_km": C.path_length_km(s1_path),
        "netD_km": netD, "netS1_km": netS, "gated": gated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.5)
    args = ap.parse_args()

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
    best = None
    last_hb = time.time()

    print(f"=== Parameter search: start x scale x heading x gain x c_max, "
          f"score = ridge + {SMOOTH_W} * turn, {args.hours} h budget ===", flush=True)

    while time.time() < deadline:
        lat = rng.uniform(*LAT_RANGE)
        lon = rng.uniform(*LON_RANGE)
        if is_land(lat, lon):
            n_land += 1
            continue
        scale = float(rng.choice(SCALE_CHOICES))
        gain = float(rng.choice(GAIN_CHOICES))
        v_max = float(rng.choice(VMAX_CHOICES))
        heading = float(rng.integers(N_HEADINGS)) * 2.0 * np.pi / N_HEADINGS

        row = evaluate(field, cluster, lat, lon, scale, heading, gain, v_max,
                       legacy_d, legacy_s1, ridge)
        row["elapsed_s"] = round(time.time() - t_start, 1)
        writer.writerow(row)
        fh.flush()
        n += 1
        if row["gated"]:
            n_gated += 1
        elif best is None or row["score"] < best["score"]:
            best = row
            print(f"  [{(time.time()-t_start)/60:5.1f} min, #{n}] score={row['score']:.3f} "
                  f"ridge={row['ridge_mean_km']:.2f} turn={row['turn_mean_rad']:.3f}  "
                  f"({row['start_lat']:.5f},{row['start_lon']:.5f}) "
                  f"scale={scale:.2f} hdg={row['heading_deg']:.0f} k={gain:.1f} "
                  f"c={v_max:.3f}  J={row['J_km']:.1f}", flush=True)

        if time.time() - last_hb > HEARTBEAT_SEC:
            b = f"{best['score']:.3f}" if best else "none"
            print(f"  heartbeat: {(time.time()-t_start)/60:.1f} min, {n} evals, "
                  f"{n_gated} gated, {n_land} land, best score {b}", flush=True)
            last_hb = time.time()

    fh.close()
    print(f"\n=== Done: {n} evaluations, {n_gated} gated, in "
          f"{(time.time()-t_start)/60:.1f} min ===")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
