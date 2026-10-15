"""Statistics for the current Fig. 9 run and anneal_check, scored on main_ridge_ne.npy.
Run from VF_Robot: venv/bin/python3 experiments/ocean_small_rho/compare_two.py"""
import os
import sys
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocean_ridge_mc as M

RIDGE = os.path.join(HERE, "main_ridge_ne.npy")
RUNS = {
    "current": (34.38493548452217, -120.40546793817956, 0.934190927524684,
                335.08707973270054, 1.940386764778898, 0.0332630274329872),
    "anneal_check": (34.39488, -120.41478, None, None, None, None),
}
BIFURCATION_LAT = 34.13


def fill_anneal_check():
    import pandas as pd
    d = pd.read_csv(os.path.join(HERE, "ridge_anneal_v1_nodur.csv"))
    r = d.loc[(d.cost - 63.375).abs().idxmin()]   # the row rendered as review/anneal_check (chain 5-3)
    assert abs(r.heading_deg - 204.035) < 0.01 and abs(r.gain - 2.573) < 0.001, r
    RUNS["anneal_check"] = (r.start_lat, r.start_lon, r.scale, r.heading_deg, r.gain, r.v_max)


def stats(T, name, p):
    lat, lon, scale, hdg, k, c = p
    row, (pD, rbD, pS, rbS) = T.evaluate(lat, lon, scale, np.radians(hdg), k, c, keep_paths=True)
    out = {"rho_km": row["rho_km"], "k": k, "c": c, "kc": k * c, "heading": hdg,
           "start": f"{lat:.4f} N, {-lon:.4f} W"}
    for tag, P in (("D", pD), ("s1", pS)):
        xy = np.column_stack([P[:, 1] * M.KLO, P[:, 0] * M.KLA])
        d, i = T.tree.query(xy)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        hours = (len(P) - 1) / 6
        south = np.nonzero(P[:, 0] <= BIFURCATION_LAT)[0]
        out.update({
            f"{tag} ride km": row[f"ride_{tag}"],
            f"{tag} mean dist km": d.mean(),
            f"{tag} 90th pct dist km": np.percentile(d, 90),
            f"{tag} max dist km": d.max(),
            f"{tag} time within 1 km %": 100 * (d <= 1.0).mean(),
            f"{tag} ridge covered within 1 km %": 100 * row[f"cov_{tag}"],
            f"{tag} path length km": step.sum(),
            f"{tag} mean speed km/h": step.sum() / max(hours, 1e-9),
            f"{tag} reaches 34.13 N (bifurcation) h": south[0] / 6 if len(south) else None,
            f"{tag} landfall h": row[f"land_{tag}_h"] or "at sea, 28 h",
            f"{tag} end": f"{P[-1, 0]:.3f} N, {-P[-1, 1]:.3f} W",
        })
    return out


_T = None


def _init():
    global _T
    _T = M.Trial(RIDGE)


def _job(p):
    try:
        return _T.evaluate(*p)
    except Exception:
        return None


def jitter(p, n=100, seed=7):
    lat, lon, scale, hdg, k, c = p
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n):
        r, th = np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        jobs.append((lat + r * np.sin(th) / M.KLA, lon + r * np.cos(th) / M.KLO,
                     scale * rng.uniform(.95, 1.05), np.radians(hdg + rng.uniform(-15, 15)),
                     k * rng.uniform(.95, 1.05), c * rng.uniform(.95, 1.05)))
    with mp.Pool(8, initializer=_init) as pool:
        rows = [r for r in pool.map(_job, jobs) if r is not None]
    res = {}
    for tag in ("D", "s1"):
        land = np.array([r[f"land_{tag}_h"] != "" for r in rows])
        endlat = np.array([r[f"end_{tag}_lat"] for r in rows])
        res[f"{tag} reaches island (lands south of 34.12 N) %"] = 100 * (land & (endlat <= 34.12)).mean()
        res[f"{tag} lands on mainland (north of 34.3 N) %"] = 100 * (land & (endlat > 34.3)).mean()
        res[f"{tag} median ride km"] = np.median([r[f"ride_{tag}"] for r in rows])
        res[f"{tag} median mean dist km"] = np.median([r[f"dev_{tag}"] for r in rows])
    return res


if __name__ == "__main__":
    fill_anneal_check()
    T = M.Trial(RIDGE)
    S = {n: stats(T, n, p) for n, p in RUNS.items()}
    J = {n: jitter(p) for n, p in RUNS.items()}
    keys = list(S["current"]) + list(J["current"])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"{'':42s} {'current':>22s} {'anneal_check':>22s}")
    for k in keys:
        a = S["current"].get(k, J["current"].get(k)); b = S["anneal_check"].get(k, J["anneal_check"].get(k))
        print(f"{k:42s} {fmt(a):>22s} {fmt(b):>22s}")
