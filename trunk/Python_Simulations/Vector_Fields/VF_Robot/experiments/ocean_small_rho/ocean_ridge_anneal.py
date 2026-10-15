"""
ocean_ridge_anneal.py

Simulated annealing with random restarts over the Santa Barbara operating
point, starts confined to the upper ridge (Chris's region: the ridge from
120.28 W along the mainland coast down to about 34.39 N), scored on the
extended ridge main_ridge_ne.npy (69.1 km).

    x = (a, o, log rho, h, k, c)     start arc 0..15 km, offset +-1.5 km,
                                     rho 2.0..4.0 km, heading, gain 1..3,
                                     speed cap 0.020..0.060
    cost(x) = max over trackers of [(L - ride) + 5 dev_eff]
              dev_eff = dev, or if that tracker lands north of 34.12 N at step n,
              (dev (n + 1) + 10 km (168 - n)) / 169, so a run lost to the mainland
              early costs more than one that drifts. (A flat +20 penalty let
              chains settle on formations that hit land in the first minutes.)
              A tracker that reaches the island (south of 34.12 N) before 24 h
              adds 2 km per hour early: Chris wants rides of 20 h or more,
              ideally over 24 h (added 2026-10-13 11:30, run restarted).

    each worker, until the deadline:
        x <- random point (prob 0.75) or jittered global best (prob 0.25)
        T <- 5 km;  repeat 400 steps:
            x' <- x + Gaussian step * T/T0, reflect at bounds, heading wraps
            accept if better, else with prob exp(-(cost' - cost)/T)
            T <- T * (0.2/5)^(1/250)

rho >= 2 km so the footprint (4 km across) spans two cells of the 2 km HFR
grid; below that the fitted curvature mostly comes from the bilinear
interpolation. Every trial is written to ridge_anneal.csv (same columns as
ridge_mc.csv plus chain, step, T, accepted, cost).

Running (from VF_Robot):
    venv/bin/python3 -u experiments/ocean_small_rho/ocean_ridge_anneal.py --hours 4 --workers 8
"""
import os
import sys
import csv
import time
import argparse
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocean_ridge_mc as M

RIDGE = os.path.join(HERE, "main_ridge_ne.npy")
CSV_PATH = os.path.join(HERE, "ridge_anneal.csv")

RHO_KM_PER_SCALE = 0.105 * 111.32 * (0.3 / 0.65)   # _ocean_run_common.rho_km(1.0)
#         name       lo                         hi                         step at T0
BOUNDS = [("arc",    0.0,                       15.0,                      1.5),
          ("off",    -1.5,                      1.5,                       0.5),
          ("lnrho",  np.log(2.0),               np.log(4.0),               0.15),
          ("hdg",    0.0,                       360.0,                     20.0),
          ("gain",   1.0,                       3.0,                       0.15),
          ("vmax",   0.020,                     0.060,                     0.004)]
LO = np.array([b[1] for b in BOUNDS])
HI = np.array([b[2] for b in BOUNDS])
STEP = np.array([b[3] for b in BOUNDS])

T0, T1, N_STEPS = 5.0, 0.2, 400
P_FROM_BEST = 0.25
LOST_KM = 10.0
T_GOAL_H, DUR_W = 24.0, 2.0
LAND_LAT_MAX = 34.12
DEV_W = 5.0
START_LAND_COST = 200.0

EXTRA = ["chain", "step", "T", "accepted", "cost"]


def reflect(x):
    x = x.copy()
    x[3] %= 360.0
    for i in (0, 1, 2, 4, 5):
        span = HI[i] - LO[i]
        y = (x[i] - LO[i]) % (2 * span)
        x[i] = LO[i] + (y if y <= span else 2 * span - y)
    return x


def cost_of(row, L, n_steps=168):
    out = []
    for k in ("D", "s1"):
        dev = row[f"dev_{k}"]
        if row[f"land_{k}_h"] != "" and row[f"end_{k}_lat"] > LAND_LAT_MAX:
            n = int(round(row[f"land_{k}_h"] * 6))
            dev = (dev * (n + 1) + LOST_KM * (n_steps - n)) / (n_steps + 1)
        early = 0.0
        if row[f"land_{k}_h"] != "" and row[f"end_{k}_lat"] <= LAND_LAT_MAX:
            early = DUR_W * max(0.0, T_GOAL_H - row[f"land_{k}_h"])
        out.append(L - row[f"ride_{k}"] + DEV_W * dev + early)
    return max(out)


def worker(wid, deadline, seed, q, best_x, best_c, lock):
    T = M.Trial(RIDGE)
    L = float(T.sd[-1])
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def evaluate(x):
        j = int(np.searchsorted(T.sd, x[0]))
        nrm = np.array([-T.tang[j, 1], T.tang[j, 0]])
        px, py = T.dense[j] + x[1] * nrm
        lat, lon = py / M.KLA, px / M.KLO
        scale = float(np.exp(x[2]) / RHO_KM_PER_SCALE)
        if T.is_land(lat, lon):
            return None, START_LAND_COST
        try:
            row = T.evaluate(lat, lon, scale, np.radians(x[3]), float(x[4]), float(x[5]))
        except Exception:
            return None, START_LAND_COST
        row.update(arc0_km=x[0], off_km=x[1])
        return row, cost_of(row, L)

    chain = 0
    while time.time() < deadline:
        with lock:
            have_best = best_c.value < 1e8
            xb = np.array(best_x[:])
        if have_best and rng.uniform() < P_FROM_BEST:
            x = reflect(xb + rng.normal(size=6) * STEP)
        else:
            x = LO + rng.uniform(size=6) * (HI - LO)
        row, c = evaluate(x)
        Temp = T0
        for step in range(N_STEPS):
            if time.time() > deadline:
                break
            x2 = reflect(x + rng.normal(size=6) * STEP * (Temp / T0))
            row2, c2 = evaluate(x2)
            acc = c2 < c or rng.uniform() < np.exp(-(c2 - c) / Temp)
            if row2 is not None:
                row2.update(chain=f"{wid}-{chain}", step=step, T=round(Temp, 4),
                            accepted=int(acc), cost=c2, worker=wid,
                            elapsed_s=round(time.time() - t0, 1))
                q.put(("row", row2, None))
            if acc:
                x, c, row = x2, c2, row2
                with lock:
                    if c < best_c.value:
                        best_c.value = c
                        best_x[:] = list(x)
            Temp *= (T1 / T0) ** (1.0 / N_STEPS)
        q.put(("chain", wid, (chain, c)))
        chain += 1
    q.put(("done", wid, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--seed-csv", default=None, help="earlier run to rescore and seed the global best")
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    q = mp.Queue()
    lock = mp.Lock()
    best_x = mp.Array("d", 6, lock=False)
    best_c = mp.Value("d", 1e9, lock=False)
    if args.seed_csv:
        import pandas as pd
        S = pd.read_csv(os.path.join(HERE, args.seed_csv))
        for k in ("land_D_h", "land_s1_h"):        # "" means at sea, as in the worker rows
            S[k] = S[k].astype(object).where(S[k].notna(), "")
        L = float(M.ridge_geometry(RIDGE)[1][-1])
        S["c"] = [cost_of(r, L) for r in S.to_dict("records")]
        r = S.loc[S.c.idxmin()]
        best_x[:] = [r.arc0_km, r.off_km, np.log(r.rho_km), r.heading_deg, r.gain, r.v_max]
        best_c.value = float(r.c)
        print(f"seeded from {args.seed_csv}: best rescored cost {r.c:.2f} "
              f"(land D {r.land_D_h} h, s1 {r.land_s1_h} h)", flush=True)
    procs = [mp.Process(target=worker, args=(w, deadline, args.seed + w, q, best_x, best_c, lock))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    fields = M.CSV_FIELDS + EXTRA
    fresh = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if fresh:
        wr.writeheader()
    n, n_chain, done, best, t0 = 0, 0, 0, None, time.time()
    while done < len(procs):
        kind, a, b = q.get()
        if kind == "done":
            done += 1
        elif kind == "chain":
            n_chain += 1
            print(f"  [{(time.time()-t0)/60:6.1f} min] chain {a}-{b[0]} ended at cost {b[1]:.2f} "
                  f"({n_chain} chains, {n} trials)", flush=True)
        else:
            wr.writerow(a)
            n += 1
            if n % 200 == 0:
                fh.flush()
            if best is None or a["cost"] < best["cost"]:
                best = a
                print(f"  [{(time.time()-t0)/60:6.1f} min #{n}] BEST {a['cost']:.2f} "
                      f"D ride {a['ride_D']:.1f} dev {a['dev_D']:.2f} | s1 ride {a['ride_s1']:.1f} "
                      f"dev {a['dev_s1']:.2f} | rho {a['rho_km']:.2f} k {a['gain']:.2f} "
                      f"c {a['v_max']:.4f} hdg {a['heading_deg']:.0f} "
                      f"start ({a['start_lat']:.4f},{a['start_lon']:.4f}) chain {a['chain']}",
                      flush=True)
    fh.close()
    print(f"done, {n} trials, {n_chain} chains, {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
