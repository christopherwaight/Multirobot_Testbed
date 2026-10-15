"""
ocean_ridge_mc.py

Flat Monte Carlo over the Santa Barbara operating point (start, formation
scale, heading, gain, speed cap), scored on how closely BOTH trackers follow
the main FTLE ridge drawn in Fig. 9 (main_ocean_hfr_2km_progression.py), and
how far down it they get within the 28-h record.

Differences from unused/ocean_param_mc.py (which chose the current start):
  - formation scale sampled down to 0.3 (rho 1.6 km); that search stopped at 0.6
  - the target is the main ridge itself, a polyline (main_ridge.npy) extracted
    from the 24-h FTLE cache by a least-cost path through the crest, from the
    channel entrance (34.425 N) through the bifurcation to the island
    (34.0 N, 120.25 W). No legacy-corridor gate.
  - starts drawn near the upper part of the ridge, not around the legacy start
  - a run stops at landfall (any robot over land), as the figure does

Per tracker:
  dev      mean distance (km) from the centroid path to the ridge
  cov      fraction of the ridge's length that the path passes within
           COV_TOL_KM of (1 km). Penalizes riding a parallel filament,
           skipping the upper ridge, and stalling.
  ride     the longest unbroken downstream ride along the ridge: over each
           stretch of the run in which the centroid stays within RIDE_TOL_KM
           (1.5 km) of the ridge, allowing gaps of up to RIDE_GAP steps,
           the furthest ridge arc reached minus the arc where the stretch
           began. Km of ridge followed in one go, downstream only.
  cost     ridge length - ride (km of ridge the tracker did not ride)
score = max(cost_D, cost_s1), so a run is only as good as its worse tracker.
Also logged, for re-ranking without rerunning: p90 distance, coverage at
1 and 2 km, and reach (furthest ridge arc within 1 km before a >3 km
excursion).

Two earlier scores were tried and dropped after rendering their leaders.
Reach at 2 km (v0_loose_score/) let D ride a parallel filament 2 km east.
Mean distance plus coverage (v1_cov_score/) was won by runs that hit land in
the first hours, and by an s1 path that wandered back north over the ridge.

A first version scored reach at 2 km tolerance and let D ride the parallel
filament 2 km east of the ridge. Its output is in v0_loose_score/.

Running (from VF_Robot):
    venv/bin/python3 -u experiments/ocean_small_rho/ocean_ridge_mc.py --hours 8 --workers 8
"""
import os
import sys
import csv
import time
import argparse
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.dirname(EXP))

RIDGE_PATH = os.path.join(HERE, "main_ridge.npy")
CACHE = os.path.join(EXP, "outputs", "oecs", "ftle_cache_24h_u4.npz")
CSV_PATH = os.path.join(HERE, "ridge_mc.csv")

KLA = 111.32
KLO = 111.32 * float(np.cos(np.radians(34.2)))

SCALE_RANGE = (0.30, 1.10)     # 0.934 is the current paper value (rho 5.04 km)
GAIN_RANGE = (1.0, 3.0)        # paper 1.94
VMAX_RANGE = (0.020, 0.060)    # paper 0.0333
START_ARC_KM = [0.0, 14.0]     # start projected onto this arc of the ridge (--arc)
START_OFF_KM = 2.5             # +- perpendicular offset from the ridge
COV_TOL_KM = 1.0
REACH_TOL_KM = 1.0
GAP_KM = 3.0
COV_W = 10.0
RIDE_TOL_KM = 1.5
RIDE_GAP = 3

CSV_FIELDS = ["start_lat", "start_lon", "arc0_km", "off_km", "scale", "rho_km",
              "heading_deg", "gain", "v_max",
              "score", "cost_D", "cost_s1", "ride_D", "ride_s1", "ride_h0_D", "ride_h1_D",
              "ride_h0_s1", "ride_h1_s1", "dev_D", "dev_s1", "cov_D", "cov_s1",
              "reach_D", "reach_s1", "p90_D", "p90_s1", "cov2_D", "cov2_s1",
              "maxdev_D", "maxdev_s1", "land_D_h", "land_s1_h",
              "end_D_lat", "end_D_lon", "end_s1_lat", "end_s1_lon", "worker", "elapsed_s"]


def ridge_geometry(path=None):
    r = np.load(path or RIDGE_PATH)
    xy = np.column_stack([r[:, 1] * KLO, r[:, 0] * KLA])
    # densify to ~0.1 km so nearest-point distance is close to segment distance
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    sd = np.arange(0, s[-1], 0.1)
    dense = np.column_stack([np.interp(sd, s, xy[:, 0]), np.interp(sd, s, xy[:, 1])])
    return dense, sd


def score_path(path_ll, tree, sd, dense):
    """dict of dev, p90, maxdev, cov, cov2, reach for one centroid path."""
    from scipy.spatial import cKDTree
    xy = np.column_stack([path_ll[:, 1] * KLO, path_ll[:, 0] * KLA])
    d, i = tree.query(xy)
    s = sd[i]
    far = np.nonzero(d > GAP_KM)[0]
    upto = far[0] if len(far) else len(d)
    ok = d[:upto] <= REACH_TOL_KM
    reach = float(s[:upto][ok].max()) if ok.any() else 0.0
    # path densified 4x so coverage does not depend on step length
    t = np.linspace(0, len(xy) - 1, 4 * len(xy) - 3)
    dxy = np.column_stack([np.interp(t, np.arange(len(xy)), xy[:, 0]),
                           np.interp(t, np.arange(len(xy)), xy[:, 1])])
    dr, _ = cKDTree(dxy).query(dense)
    on = d <= RIDE_TOL_KM
    ride, ride_k0, ride_k1, k = 0.0, 0, 0, 0
    n = len(d)
    while k < n:
        if not on[k]:
            k += 1
            continue
        k0, last, smax, gap = k, k, s[k], 0
        while k < n and gap <= RIDE_GAP:
            if on[k]:
                gap, last = 0, k
                smax = max(smax, s[k])
            else:
                gap += 1
            k += 1
        if smax - s[k0] > ride:
            ride, ride_k0, ride_k1 = float(smax - s[k0]), k0, last
        k = last + 1
    return {"ride": ride, "ride_h0": ride_k0 / 6, "ride_h1": ride_k1 / 6,
            "dev": float(d.mean()), "p90": float(np.percentile(d, 90)),
            "maxdev": float(d.max()), "cov": float((dr <= COV_TOL_KM).mean()),
            "cov2": float((dr <= 2.0).mean()), "reach": reach}


class Trial:
    """Field, cluster and ridge for one process; .evaluate runs both trackers."""

    def __init__(self, ridge_path=None):
        from scipy.spatial import cKDTree
        import _ocean_run_common as C
        from _coords_common import latlon_to_world, world_to_latlon
        from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
        self.C, self.l2w, self.w2l = C, latlon_to_world, world_to_latlon
        self.prims = (separatrix_logic_c_step, oecs_separatrix_step)
        self.dense, self.sd = ridge_geometry(ridge_path)
        self.tree = cKDTree(self.dense)
        tang = np.gradient(self.dense, axis=0)
        self.tang = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        z = np.load(CACHE, allow_pickle=True)
        self.lat_f, self.lon_f, self.land_f = z["lat"], z["lon"], z["land"]
        self.field, self.cluster, _, _ = C.build_trial(isotropic_map=True)

    def is_land(self, lat, lon):
        ii = np.clip(np.searchsorted(self.lat_f, lat), 0, len(self.lat_f) - 1)
        jj = np.clip(np.searchsorted(self.lon_f, lon), 0, len(self.lon_f) - 1)
        return self.land_f[ii, jj]

    def run(self, prim, lat, lon, heading):
        """Centroid path (lat, lon) and robot paths, cut at the first landfall step."""
        C, cfg, cl = self.C, self.field.config, self.cluster
        sx, sy = self.l2w(lat, lon, cfg)
        cl.reset(sx, sy, heading_offset=heading)
        # PentagonCluster.reset clears the _oecs_* latches but not the D
        # tracker's _logicc_captured (2026-10-09 rewrite), so a reused cluster
        # would start every D run after the first capture already captured.
        for attr in [k for k in vars(cl) if k.startswith(("_logicc", "_oecs"))]:
            delattr(cl, attr)
        self.field.reset_clock()
        for _ in range(C.SIM_STEPS):
            cl.move(prim)
            self.field.step(cl.timestep * C.TIME_WARP)
        rob = cl.get_robot_history()
        rla, rlo = self.w2l(rob[..., 0], rob[..., 1], cfg)
        hit = self.is_land(rla, rlo).any(axis=1)
        cen = cl.get_center_history()
        cla, clo = self.w2l(cen[:, 0], cen[:, 1], cfg)
        path = np.column_stack([cla, clo])
        robots = np.stack([rla, rlo], axis=-1)
        k_land = int(np.argmax(hit)) if hit.any() else None
        if k_land is not None:
            path, robots = path[:k_land + 1], robots[:k_land + 1]
        return path, robots, k_land

    def evaluate(self, lat, lon, scale, heading, gain, vmax, keep_paths=False):
        C = self.C
        C.set_formation_scale(self.cluster, scale)
        fd, fs = self.prims

        def prim_d(c):
            vx, vy = fd(c, v_max=vmax)
            return vx * gain, vy * gain

        def prim_s1(c):
            vx, vy = fs(c, v_max=vmax, g_perp=C.G_PERP)
            return vx * gain, vy * gain

        pD, rbD, kD = self.run(prim_d, lat, lon, heading)
        pS, rbS, kS = self.run(prim_s1, lat, lon, heading)
        D = score_path(pD, self.tree, self.sd, self.dense)
        S1 = score_path(pS, self.tree, self.sd, self.dense)
        L = float(self.sd[-1])
        cD = L - D["ride"]
        cS = L - S1["ride"]
        row = {
            "start_lat": lat, "start_lon": lon,
            "scale": scale, "rho_km": C.rho_km(scale), "heading_deg": np.degrees(heading),
            "gain": gain, "v_max": vmax,
            "score": max(cD, cS), "cost_D": cD, "cost_s1": cS,
            **{f"{k}_D": v for k, v in D.items()},
            **{f"{k}_s1": v for k, v in S1.items()},
            "land_D_h": "" if kD is None else kD / 6, "land_s1_h": "" if kS is None else kS / 6,
            "end_D_lat": pD[-1, 0], "end_D_lon": pD[-1, 1],
            "end_s1_lat": pS[-1, 0], "end_s1_lon": pS[-1, 1]}
        if keep_paths:
            return row, (pD, rbD, pS, rbS)
        return row


def worker(wid, deadline, seed, q, arc_range, ridge_path, scale_range, off_km):
    T = Trial(ridge_path)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    while time.time() < deadline:
        arc0 = rng.uniform(*arc_range)
        off = rng.uniform(-off_km, off_km)
        j = int(np.searchsorted(T.sd, arc0))
        nrm = np.array([-T.tang[j, 1], T.tang[j, 0]])
        x, y = T.dense[j] + off * nrm
        lat, lon = y / KLA, x / KLO
        if T.is_land(lat, lon):
            continue
        scale = float(rng.uniform(*scale_range))
        gain = float(rng.uniform(*GAIN_RANGE))
        vmax = float(rng.uniform(*VMAX_RANGE))
        heading = float(rng.uniform(0, 2 * np.pi))
        try:
            row = T.evaluate(lat, lon, scale, heading, gain, vmax)
        except Exception as e:  # singular fits etc.
            q.put(("err", wid, repr(e)))
            continue
        row.update(arc0_km=arc0, off_km=off, worker=wid, elapsed_s=round(time.time() - t0, 1))
        q.put(("row", row, None))
    q.put(("done", wid, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--arc", type=float, nargs=2, default=START_ARC_KM)
    ap.add_argument("--ridge", default=None, help="ridge .npy in this folder (default main_ridge.npy)")
    ap.add_argument("--scale", type=float, nargs=2, default=SCALE_RANGE)
    ap.add_argument("--off", type=float, default=START_OFF_KM, help="+- start offset from the ridge, km")
    ap.add_argument("--csv", default="ridge_mc.csv")
    args = ap.parse_args()
    START_ARC_KM[:] = args.arc
    csv_path = os.path.join(HERE, args.csv)
    deadline = time.time() + args.hours * 3600
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(w, deadline, args.seed + w, q, tuple(START_ARC_KM),
                                                 args.ridge and os.path.join(HERE, args.ridge),
                                                 tuple(args.scale), args.off))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    fresh = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if fresh:
        wr.writeheader()
    n, best, done, t0, hb = 0, None, 0, time.time(), time.time()
    while done < len(procs):
        kind, a, b = q.get()
        if kind == "done":
            done += 1
            continue
        if kind == "err":
            print(f"  worker {a} error {b}", flush=True)
            continue
        wr.writerow(a)
        fh.flush()
        n += 1
        if best is None or a["score"] < best["score"]:
            best = a
            print(f"  [{(time.time()-t0)/60:6.1f} min #{n}] BEST {a['score']:.2f} "
                  f"D ride {a['ride_D']:.1f} km dev {a['dev_D']:.2f} | s1 ride {a['ride_s1']:.1f} km "
                  f"dev {a['dev_s1']:.2f} | rho {a['rho_km']:.2f} km k {a['gain']:.2f} "
                  f"c {a['v_max']:.4f} hdg {a['heading_deg']:.0f} "
                  f"start ({a['start_lat']:.4f},{a['start_lon']:.4f})", flush=True)
        if time.time() - hb > 600:
            print(f"  heartbeat {(time.time()-t0)/60:.0f} min, {n} trials", flush=True)
            hb = time.time()
    fh.close()
    print(f"done, {n} trials, {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
