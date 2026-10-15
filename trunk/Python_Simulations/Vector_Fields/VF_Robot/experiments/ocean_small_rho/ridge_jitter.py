"""
ridge_jitter.py

Robustness check of the best ocean_ridge_mc.py candidates. Picks the top
distinct candidates from ridge_mc.csv and reruns each under jitter: start
within 1 km, heading +-15 deg, scale, gain and speed cap +-5% (the band the
candidate B note in main_ocean_hfr_2km_shared_start.py used).

Ranking uses rank = max over trackers of (cost + DEV_W * dev): km of ridge
not ridden in one go, plus a penalty on mean distance from the ridge over the
whole run, so a tracker that rides early and then drifts away loses.

Each candidate's best jittered run is also reported (jit_best_*), so this
doubles as a local refinement.

Writes ridge_jitter.csv (every jittered run) and ridge_jitter_summary.csv.

Running (from VF_Robot):
    venv/bin/python3 -u experiments/ocean_small_rho/ridge_jitter.py --top 40 --n 60
"""
import os
import sys
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocean_ridge_mc as M

DEV_W = 5.0
PAPER_SCORE = 70.3  # current Fig. 9 operating point under rank (s1: 56.5 + 5 * 2.77)
KEYS = ["start_lat", "start_lon", "scale", "heading_deg", "gain", "v_max"]
_T = None


LAND_LAT_MAX = 34.10   # landfall counts only at the island end of the ridge


def add_rank(df):
    df["rank"] = np.maximum(df.cost_D + DEV_W * df.dev_D, df.cost_s1 + DEV_W * df.dev_s1)
    bad = ((df.land_D_h.notna() & (df.end_D_lat > LAND_LAT_MAX))
           | (df.land_s1_h.notna() & (df.end_s1_lat > LAND_LAT_MAX)))
    df.loc[bad, "rank"] = np.inf
    return df


def distinct_top(df, n):
    """Greedy pick of the n best rows that differ from every row already picked."""
    df = df.sort_values("rank")
    picked = []
    for _, r in df.head(5000).iterrows():
        ok = True
        for p in picked:
            dkm = np.hypot((r.start_lat - p.start_lat) * M.KLA, (r.start_lon - p.start_lon) * M.KLO)
            dh = abs((r.heading_deg - p.heading_deg + 180) % 360 - 180)
            if (dkm < 1.0 and dh < 20 and abs(r.scale / p.scale - 1) < 0.1
                    and abs(r.gain / p.gain - 1) < 0.1 and abs(r.v_max / p.v_max - 1) < 0.1):
                ok = False
                break
        if ok:
            picked.append(r)
        if len(picked) == n:
            break
    return pd.DataFrame(picked).reset_index(drop=True)


def _init(ridge_path=None):
    global _T
    _T = M.Trial(ridge_path)


def _job(args):
    cid, j, lat, lon, scale, hdg, gain, vmax = args
    try:
        row = _T.evaluate(lat, lon, scale, np.radians(hdg), gain, vmax)
    except Exception:
        return None
    row.update(cand=cid, jitter=j)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--src", default="ridge_mc.csv")
    ap.add_argument("--out", default="ridge_jitter")
    ap.add_argument("--radius-km", type=float, default=1.0)
    ap.add_argument("--ridge", default=None)
    args = ap.parse_args()
    df = add_rank(pd.read_csv(os.path.join(HERE, args.src)))
    cands = distinct_top(df, args.top)
    cands.insert(0, "cand", range(len(cands)))
    rng = np.random.default_rng(7)
    jobs = []
    for _, c in cands.iterrows():
        jobs.append((int(c.cand), 0, c.start_lat, c.start_lon, c.scale, c.heading_deg, c.gain, c.v_max))
        for j in range(1, args.n + 1):
            r, th = args.radius_km * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            jobs.append((int(c.cand), j,
                         c.start_lat + r * np.sin(th) / M.KLA, c.start_lon + r * np.cos(th) / M.KLO,
                         c.scale * rng.uniform(0.95, 1.05), c.heading_deg + rng.uniform(-15, 15),
                         c.gain * rng.uniform(0.95, 1.05), c.v_max * rng.uniform(0.95, 1.05)))
    rp = args.ridge and os.path.join(HERE, args.ridge)
    with mp.Pool(args.workers, initializer=_init, initargs=(rp,)) as pool:
        rows = [r for r in pool.imap_unordered(_job, jobs, chunksize=4) if r is not None]
    J = add_rank(pd.DataFrame(rows))
    J.to_csv(os.path.join(HERE, f"{args.out}.csv"), index=False)
    jj = J[J.jitter > 0]
    g = jj.groupby("cand")
    summ = cands[["cand"] + KEYS + ["rho_km", "rank", "score", "dev_D", "dev_s1", "ride_D", "ride_s1", "cov_D", "cov_s1"]].copy()
    summ["jit_med_rank"] = g["rank"].median().values
    summ["jit_p75_rank"] = g["rank"].quantile(0.75).values
    summ["jit_med_ride_D"] = g.ride_D.median().values
    summ["jit_med_ride_s1"] = g.ride_s1.median().values
    summ["jit_best_rank"] = g["rank"].min().values
    summ["jit_frac_beats_paper"] = g["rank"].apply(lambda s: (s < PAPER_SCORE).mean()).values
    summ["jit_med_reach_D"] = g.reach_D.median().values
    summ["jit_med_reach_s1"] = g.reach_s1.median().values
    summ = summ.sort_values("jit_med_rank")
    summ.to_csv(os.path.join(HERE, f"{args.out}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
