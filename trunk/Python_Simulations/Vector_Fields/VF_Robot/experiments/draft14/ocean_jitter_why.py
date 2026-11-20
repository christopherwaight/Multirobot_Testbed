"""
ocean_jitter_why.py  (Draft_14, Section V-B numbers; supporting evidence for Fig. 9)

Why do perturbed Santa Barbara runs split? Replays the 100 perturbations of
ocean_small_rho/jitter_paper_current.py (same seed 7 and recipe: start within
1 km, heading +-15 deg, formation scale, gain and speed cap +-5%) plus the
nominal Fig. 9 settings, on current code (after the 2026-10-28 formation fix,
which the October run predates). For every ride it logs, from the cluster's own
fit at each step, the rotation-to-strain ratio |omega|/2r and the sign of D,
and scores the path against the main FTLE ridge (main_ridge.npy) as the
October study did.

Run from VF_Robot:
    venv/bin/python3 experiments/draft14/ocean_jitter_why.py --workers 2
Writes experiments/draft14/outputs/ocean_jitter_why.csv (one row per run and
tracker) and ocean_jitter_why_steps.npz (per-step ratio, sign, lat/lon).
"""
import argparse
import os
import sys
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "ocean_small_rho"))

NOMINAL = dict(start_lat=34.38493548452217, start_lon=-120.40546793817956,
               scale=0.934190927524684, heading_deg=335.08707973270054,
               gain=1.940386764778898, v_max=0.0332630274329872)
WINDOW_H = (2.0, 6.5)     # the early stretch Section V-B discusses


def jobs(n):
    import ocean_ridge_mc as M
    c = NOMINAL
    rng = np.random.default_rng(7)
    out = [(0, c["start_lat"], c["start_lon"], c["scale"], c["heading_deg"], c["gain"], c["v_max"])]
    for j in range(1, n + 1):
        r, th = np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        out.append((j, c["start_lat"] + r * np.sin(th) / M.KLA, c["start_lon"] + r * np.cos(th) / M.KLO,
                    c["scale"] * rng.uniform(.95, 1.05), c["heading_deg"] + rng.uniform(-15, 15),
                    c["gain"] * rng.uniform(.95, 1.05), c["v_max"] * rng.uniform(.95, 1.05)))
    return out


_T = None


def _init():
    global _T
    import ocean_ridge_mc as M
    _T = M.Trial()


def _fit_terms(c):
    """|omega|/2r, D and the measured flow from the cluster's own fit."""
    from src.control.pentagon_primitives import (_sample_vector_at_robots,
                                                 _get_relative_positions, _fit_vector_quadratic)
    u, v = _sample_vector_at_robots(c)
    tu, tv = _fit_vector_quadratic(_get_relative_positions(c), u, v)
    ux, uy, vx, vy = tu[1], tu[2], tv[1], tv[2]
    r = np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx))
    return abs(vx - uy) / (2 * max(r, 1e-12)), ux * vy - uy * vx, np.array([tu[0], tv[0]])


def _run(args):
    import ocean_ridge_mc as M
    j, lat, lon, scale, hdg, gain, vmax = args
    T = _T
    C = T.C
    C.set_formation_scale(T.cluster, scale)
    fd, fs = T.prims
    rows, steps = [], {}
    for name, base in (("D", lambda c: fd(c, v_max=vmax)),
                       ("s1", lambda c: fs(c, v_max=vmax, g_perp=C.G_PERP))):
        log = []

        def prim(c, base=base, log=log):
            ratio, det, flow = _fit_terms(c)
            vx, vy = base(c)
            cmd = np.array([vx, vy])
            # cosine between the command and the measured flow: < 0 rides upstream
            cos = float(cmd @ flow / max(np.linalg.norm(cmd) * np.linalg.norm(flow), 1e-12))
            tid = c.diagnostics[-1].get("tangent_id", "") if getattr(c, "diagnostics", None) else ""
            log.append((ratio, det, cos, tid))
            return vx * gain, vy * gain

        path, _, k_land = T.run(prim, lat, lon, np.radians(hdg))
        sc = M.score_path(path, T.tree, T.sd, T.dense)
        ratio = np.array([e[0] for e in log])[:len(path)]
        dsgn = np.array([e[1] for e in log])[:len(path)]
        cosf = np.array([e[2] for e in log])[:len(path)]
        tid = [e[3] for e in log][:len(path)]
        sw = np.array([tid[i] != tid[i - 1] and tid[i] != "" for i in range(1, len(tid))], bool)
        h = np.arange(len(path)) / 6.0
        w = (h >= WINDOW_H[0]) & (h <= WINDOW_H[1])
        rows.append(dict(
            jitter=j, tracker=name, start_lat=lat, start_lon=lon, scale=scale,
            heading_deg=hdg, gain=gain, v_max=vmax,
            land_h=(k_land / 6.0 if k_land is not None else np.nan),
            end_lat=path[-1, 0], end_lon=path[-1, 1], min_lat=path[:, 0].min(),
            lat_7h=path[min(42, len(path) - 1), 0],
            ride_km=sc["ride"], dev_km=sc["dev"], reach_km=sc["reach"],
            ratio_med_all=float(np.median(ratio)),
            ratio_med_win=float(np.median(ratio[w])) if w.any() else np.nan,
            dpos_frac_win=float(np.mean(dsgn[w] > 0)) if w.any() else np.nan,
            ratio_med_0_7=float(np.median(ratio[h < 7])),
            dpos_frac_0_7=float(np.mean(dsgn[h < 7] > 0)),
            upstream_frac_0_7=float(np.mean(cosf[h < 7] < 0)),
            first_upstream_h=(float(h[np.argmax(cosf < -0.5)]) if (cosf < -0.5).any() else np.nan),
            switches_0_7=int(sw[:min(42, len(sw))].sum()),
            switches_all=int(sw.sum())))
        steps[f"{name}_{j}_cos"] = cosf
        steps[f"{name}_{j}_ratio"] = ratio
        steps[f"{name}_{j}_dsign"] = np.sign(dsgn)
        steps[f"{name}_{j}_path"] = path
    return rows, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    import pandas as pd
    with mp.Pool(a.workers, initializer=_init) as p:
        res = p.map(_run, jobs(a.n), chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    steps = {k: v for _, ss in res for k, v in ss.items()}
    out = os.path.join(HERE, "outputs")
    pd.DataFrame(rows).to_csv(os.path.join(out, "ocean_jitter_why.csv"), index=False)
    np.savez_compressed(os.path.join(out, "ocean_jitter_why_steps.npz"), **steps)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
