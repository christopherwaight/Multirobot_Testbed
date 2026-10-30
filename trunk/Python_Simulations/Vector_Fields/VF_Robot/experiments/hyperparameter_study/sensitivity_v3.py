"""
E11: sensitivity of the version 3 robot model to its two uncertain parameters
(Paper_Writing/Systems Paper Revision/robot_model_note): the time constant tau
(0.28 s assumed, about 0.20 s from the hardware start transients) and the
friction levels, static v_c in {0.025, 0.035} m/s with kinetic v_k = v_c, v_c / 2 or 0.

For each (tau, v_c, v_k), with the plane fit and every other setting at hardware values:
    attraction on five fields (N 10^4) and on the vortex at k 0.3, 3, 10
    orbit at the paper's six radii on five fields (N 10^3)
    orbit radius sweep r_d 1e-4 to 10 m on the vortex (N 200)
    the 157 hardware starts x 50 headings, 10 s, linear and reconstructed fields
Each row carries its closed-form prediction: rest point (theory.rest_point_coulomb),
orbit radius with V = min(k_t - v_k, v_max), and the start condition k_t > v_c.

Run from VF_Robot:  venv/bin/python3 experiments/hyperparameter_study/sensitivity_v3.py [--workers 7]
Writes results_v3_sens/sens.csv and prints a summary.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vecsim as vs
import theory as th
import load
import design

OUT = os.path.join(HERE, "results_v3_sens")
DT = 0.1
MAP_SCALE = 1.6764            # hardware .npy files are OptiTrack metres / 1.6764 (see stiction_hw_check.py)
FIELDS = ("vortex", "saddle", "gyre", "vortex_nn", "saddle_nn")
VARIANTS = [(tau, vc, vc * r) for tau in (0.20, 0.28) for vc in (0.025, 0.035) for r in (1.0, 0.5, 0.0)]


def jobs():
    J = []
    for tau, vc, vk in VARIANTS:
        v = dict(tau=tau, vc=vc, vk=vk)
        for f in FIELDS:
            J.append(dict(v, kind="attract", field=f, k=1.0, N=10000))
            for rd in design.PAPER_RDS:
                J.append(dict(v, kind="orbit", field=f, rd=rd, N=1000))
        for k in (0.3, 3.0, 10.0):
            J.append(dict(v, kind="attract", field="vortex", k=k, N=10000))
        for rd in design.RDS:
            J.append(dict(v, kind="sweep", field="vortex", rd=rd, N=200))
        for f, hw in (("vortex", "vortex"), ("saddle", "saddle"), ("vortex_nn", "vortex"), ("saddle_nn", "saddle")):
            J.append(dict(v, kind="hw", field=f, hw=hw, k=1.0, N=50))
    return J


def job_id(j):
    return "E11|" + "|".join(f"{k}={j[k]}" for k in sorted(j))


def run(j):
    t0 = time.time()
    seed = design._seed(job_id(j))
    F = vs.make_field(j["field"])
    alpha = float(np.exp(-DT / j["tau"]))
    row = dict(j, alpha=alpha)
    if j["kind"] in ("attract", "hw"):
        starts = thetas = None
        T = 60.0
        if j["kind"] == "hw":
            hs = np.load(os.path.join(HERE, f"hardware_{j['hw']}_starts_finals.npy"))[:, :2] * MAP_SCALE
            starts = np.repeat(hs, j["N"], axis=0)
            thetas = np.random.default_rng(seed + 2).uniform(0, 2 * np.pi, len(starts))
            T = 10.0
        n = j["N"] if starts is None else len(starts)
        P = vs.params(n, alpha=alpha, vs=j["vc"], k=j["k"])
        M, _ = vs.simulate(F, "attract", "fit", P, T, dt=DT, seed=seed, starts=starts, thetas=thetas,
                           stiction_model="coulomb", vk=j["vk"])
        s = load.attract_summary(dict(rd=np.nan), M)
        fe = M["final_err"]
        row.update(n=n, conv=s["conv"], err_mean=s["err_mean"], err_med=s["err_med"], err_p95=s["err_p95"],
                   bias=s["bias"], prec=s["prec"], over_p95=s["over_p95"], all_mean=float(np.mean(fe)),
                   proj_med=float(np.median(M["final_proj"])))
        if j["field"] in vs.LINEAR:
            rp = th.rest_point_coulomb(M["start_dist"], j["k"], alpha, DT, j["vc"], vk=j["vk"])
            row["pred_proj_med"] = float(-np.nanmedian(rp))
    else:
        rd = j["rd"]
        T, w0 = design.orbit_timing(rd, 0.2, 0.5, alpha, DT)
        P = vs.params(j["N"], alpha=alpha, vs=j["vc"], rd=rd)
        M, _ = vs.simulate(F, "orbit", "fit", P, T, dt=DT, seed=seed, window=w0,
                           stiction_model="coulomb", vk=j["vk"])
        s = load.orbit_summary(dict(rd=rd), M)
        row.update(n=j["N"], ok=s["ok"], nostart=s["nostart"], err_mean=s["err_mean"],
                   err_between_sd=s["err_between_sd"], within_sd=s["within_sd"],
                   pred_start=float(0.2 > j["vc"]))
        if j["field"] in vs.LINEAR:
            row["pred_err"] = th.orbit_r_exact(rd, alpha, DT, 0.2, 0.5, 0.4, vc=j["vk"]) - rd
            row["pred_err_small"] = float(th.orbit_r_small_angle(rd, alpha, DT, 0.2, 0.5, 0.4, j["vk"]) - rd)
    row["secs"] = time.time() - t0
    return row


def hardware_rows():
    out = []
    for f in ("vortex", "saddle"):
        a = np.load(os.path.join(HERE, f"hardware_{f}_starts_finals.npy"))
        start, final = a[:, :2] * MAP_SCALE, a[:, 2:4] * MAP_SCALE
        u0 = -start / np.linalg.norm(start, axis=1, keepdims=True)
        mu = final.mean(0)
        out.append(dict(kind="hardware", field=f, n=len(final), all_mean=float(np.hypot(*final.T).mean()),
                        err_med=float(np.median(np.hypot(*final.T))), bias=float(np.hypot(*mu)),
                        prec=float(np.sqrt(((final - mu) ** 2).sum(1).mean())),
                        proj_med=float(np.median((final * u0).sum(1)))))
    return out


def hw_fit():
    """Kinetic-friction scan against the 157 hardware runs: each hardware start x 50 headings,
    10 s, tau 0.28, v_c = max(v_k, 0.025), on the reconstructed and linear fields. Compares the
    bias-free measures (precision about the mean rest point, median rest point along the
    approach line) and the KS distance between final-distance distributions."""
    from scipy.stats import ks_2samp
    rows = []
    for f, hw in (("vortex_nn", "vortex"), ("saddle_nn", "saddle"), ("vortex", "vortex"), ("saddle", "saddle")):
        a = np.load(os.path.join(HERE, f"hardware_{hw}_starts_finals.npy"))
        start, final = a[:, :2] * MAP_SCALE, a[:, 2:4] * MAP_SCALE
        u0 = -start / np.linalg.norm(start, axis=1, keepdims=True)
        hd = np.hypot(*final.T)
        F = vs.make_field(f)
        starts = np.repeat(start, 50, axis=0)
        uu = np.repeat(u0, 50, axis=0)
        thetas = np.random.default_rng(5).uniform(0, 2 * np.pi, len(starts))
        for vk in np.arange(0.005, 0.03001, 0.0025):
            P = vs.params(len(starts), alpha=np.exp(-DT / 0.28), vs=max(vk, 0.025), k=1.0)
            M, _ = vs.simulate(F, "attract", "fit", P, 10.0, dt=DT, seed=6, starts=starts, thetas=thetas,
                               stiction_model="coulomb", vk=vk)
            fin = np.stack([M["final_x"], M["final_y"]], 1) - F.target
            sd, mu = np.hypot(*fin.T), fin.mean(0)
            rows.append(dict(field=f, hw=hw, vk=round(vk, 4), mean=sd.mean(), bias_x=mu[0], bias_y=mu[1],
                             prec=np.sqrt(((fin - mu) ** 2).sum(1).mean()),
                             along_med=np.median((fin * uu).sum(1)), ks=ks_2samp(sd, hd).statistic))
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
    with open(os.path.join(OUT, "hw_fit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


FRAMES = {"identity": [[1, 0], [0, 1]], "rot90": [[0, -1], [1, 0]], "rot180": [[-1, 0], [0, -1]],
          "rot270": [[0, 1], [-1, 0]], "mirror x": [[-1, 0], [0, 1]], "mirror y": [[1, 0], [0, -1]],
          "swap x,y": [[0, 1], [1, 0]], "anti-swap": [[0, -1], [-1, 0]]}


def frames():
    """Frame check for the hardware rest offsets: for each 90-degree rotation or mirror M,
    map the hardware starts and finals through M, rerun the starts on the reconstructed
    field (v_c = v_k = 0.025, tau 0.28, 50 headings, 10 s) and compare the simulated mean
    rest offset with the mapped hardware offset (angle between them, distance apart)."""
    rows = []
    for f, hw in (("vortex_nn", "vortex"), ("saddle_nn", "saddle")):
        a = np.load(os.path.join(HERE, f"hardware_{hw}_starts_finals.npy"))
        start, final = a[:, :2] * MAP_SCALE, a[:, 2:4] * MAP_SCALE
        F = vs.make_field(f)
        for name, M in FRAMES.items():
            M = np.array(M, float)
            starts = np.repeat(start @ M.T, 50, axis=0)
            thetas = np.random.default_rng(5).uniform(0, 2 * np.pi, len(starts))
            P = vs.params(len(starts), alpha=np.exp(-DT / 0.28), vs=0.025, k=1.0)
            R, _ = vs.simulate(F, "attract", "fit", P, 10.0, dt=DT, seed=6, starts=starts, thetas=thetas,
                               stiction_model="coulomb")
            bs = np.array([R["final_x"].mean(), R["final_y"].mean()]) - F.target
            bh = final.mean(0) @ M.T
            ang = np.degrees(np.arccos(np.clip(bs @ bh / np.linalg.norm(bs) / np.linalg.norm(bh), -1, 1)))
            rows.append(dict(field=f, frame=name, hw_x=bh[0], hw_y=bh[1], sim_x=bs[0], sim_y=bs[1],
                             angle_deg=ang, miss=np.linalg.norm(bs - bh)))
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
    with open(os.path.join(OUT, "frames.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fit", action="store_true", help="only the kinetic-friction scan against hardware (hw_fit.csv)")
    ap.add_argument("--frames", action="store_true", help="only the hardware frame check (frames.csv)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.fit:
        hw_fit()
        return
    if a.frames:
        frames()
        return
    J = jobs()
    J.sort(key=lambda j: -j["N"] * (2 if j["kind"] != "attract" else 1))
    t0 = time.time()
    rows = hardware_rows()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(J)} jobs, {time.time() - t0:.0f} s", flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, "sens.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(J)} jobs in {time.time() - t0:.0f} s, wrote {os.path.join(OUT, 'sens.csv')}")


if __name__ == "__main__":
    main()
