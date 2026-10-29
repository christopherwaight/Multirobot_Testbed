"""
Small Monte Carlo spot check of the accuracy model for the 3-robot primitives
(vector plane fit and magnitude only) on linear fields, the single gyre, and
the curved saddles. Each cell runs N trials with a random start and a random
formation heading, and is printed beside a prediction made without the MC.

PREDICTIONS
  Orbit. The orbit law regulates rho = |p_hat - c| to r_d while moving at
  k_t = 0.2 m/s. With d = p_hat - (true center) split into a radial part d_r
  and a counterclockwise tangential part d_t, the radius obeys
      r_dot = -k_r (r - d_r - r_d) + k_t d_t / r + k_r e_dyn(r, alpha)
  where e_dyn is the outward push of the robot lag and of the straight
  0.1 s steps (first-order filter of a command rotating at k_t / r).
  Mean radius r* solves  r = r_d + e_dyn(r) + <d_r> + k_t <d_t> / (k_r r),
  with <.> the average over orbit angle and formation heading of the static
  estimator error. The angle-dependent part forces harmonics n that the
  radial loop passes with gain 1 / sqrt((k_r s)^2 + (n k_t / r)^2).
  Attraction. Final error is the larger of the estimator's fixed-point bias
  (where p_hat(c) = c, or for gradient descent where the fitted slope is
  zero) and the stiction stop of a 1-D model of the lagged robot.
  Basins. The rule in FieldSpec.basin (D = 0 diamond, potential-flow
  hyperbola), checked away from the boundary.

Run from VF_Robot:
    venv/bin/python3 experiments/mc_field_tests.py                # all groups, 100 runs
    venv/bin/python3 experiments/mc_field_tests.py --only G1 G2 --n-runs 20
Output: experiments/outputs/mc_field_tests/{runs.csv, cells.csv, summary.txt}
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _field_tests_common as ft
from src.control.kinematics import inverse_kinematics

OUT_DIR = os.path.join(ft.PROJECT_ROOT, "experiments", "outputs", "mc_field_tests")
RADII = [0.10, 0.20, 0.30, 0.40]
EPS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
ATTRACT_STEPS, ORBIT_STEPS, ORBIT_W0 = 400, 450, 150
SEED = 20261028
DEFAULT = (0.7, 0.025)     # alpha, stiction
LIGHT = (0.01, 0.0)


def cells():
    C = []

    def add(group, task, method, field, side, alpha, stiction, eps=0.0, r_d=None):
        C.append(dict(group=group, task=task, method=method, field=field, eps=eps, side=side,
                      alpha=alpha, stiction=stiction, r_d=r_d))
    for f in ("linear_center", "linear_saddle"):
        for a in (0.01, 0.5, 0.7, 0.9):
            for rd in RADII:
                add("L1", "orbit", "vector", f, 0.33, a, 0.025, r_d=rd)
    for rd in RADII:
        add("L2", "orbit", "vector", "linear_saddle", 0.10, *DEFAULT, r_d=rd)
        add("L3", "orbit", "vector", "linear_saddle", 0.33, 0.7, 0.0, r_d=rd)
        # alpha 0.9 with stiction never starts (0.1 x 0.2 m/s < 0.025), so test its lag without
        add("L5", "orbit", "vector", "linear_saddle", 0.33, 0.9, 0.0, r_d=rd)
    for a in (0.01, 0.7):
        for st in (0.0, 0.01, 0.025, 0.05):
            add("L4", "attract", "vector", "linear_saddle", 0.33, a, st)
    for st in (0.0, 0.025):
        add("L4", "attract", "vector", "linear_saddle", 0.10, 0.7, st)
    add("L4", "attract", "vector", "linear_center", 0.33, *DEFAULT)
    gcfg = [(0.10, *DEFAULT), (0.33, *DEFAULT), (0.10, *LIGHT), (0.33, *LIGHT), (0.33, 0.7, 0.0)]
    for side, a, st in gcfg:
        add("G1", "attract", "vector", "gyre", side, a, st)
        for rd in RADII:
            add("G2", "orbit", "vector", "gyre", side, a, st, r_d=rd)
    for fam in ("potential_saddle", "cellular_saddle"):
        for e in EPS:
            for rd in (0.2, 0.3):
                add("S1", "orbit", "vector", fam, 0.10, *LIGHT, eps=e, r_d=rd)
            add("S2", "orbit", "vector", fam, 0.33, *DEFAULT, eps=e, r_d=0.3)
    for e in (-2.0, -1.0, 1.0, 2.0):
        add("S3", "attract", "vector", "potential_saddle", 0.33, *DEFAULT, eps=e)
    for e in (1.0, 2.0):
        add("S3", "attract", "vector", "cellular_saddle", 0.33, *DEFAULT, eps=e)
    mfields = [("linear_saddle", 0.0), ("shear_saddle", 0.0), ("gyre", 0.0), ("potential_saddle", 1.0)]
    for f, e in mfields:
        for side in (0.10, 0.33):
            for rd in RADII:
                add("M1", "orbit", "magnitude", f, side, *DEFAULT, eps=e, r_d=rd)
            add("M3", "attract", "magnitude", f, side, *DEFAULT, eps=e)
    for rd in RADII:
        add("M2", "orbit", "vector", "shear_saddle", 0.33, *DEFAULT, r_d=rd)
    add("M3", "attract", "magnitude", "linear_saddle", 0.33, 0.7, 0.0)
    add("M3", "attract", "magnitude", "gyre", 0.33, 0.7, 0.0)
    add("M4", "attract", "vector", "shear_saddle", 0.33, *DEFAULT)
    return C


# ============================================================================
# Predictions (no simulation of the cluster)
# ============================================================================
THETAS = np.linspace(0, 2 * np.pi / 3, 4, endpoint=False)
PHIS = np.linspace(0, 2 * np.pi, 48, endpoint=False)


def lag_filter(alpha, w):
    """Discrete robot lag v' = alpha v + (1 - alpha) u for a command rotating w rad/step."""
    H = (1 - alpha) / (1 - alpha * np.exp(-1j * w))
    return abs(H), -np.angle(H)


def e_dyn(r, alpha):
    G, lag = lag_filter(alpha, ft.K_T * ft.DT / r)
    return (ft.K_T * np.sin(lag) + ft.DT * G * ft.K_T ** 2 / (2 * r)) / (ft.K_R * np.cos(lag))


def tri(c, theta, side):
    return np.array(inverse_kinematics(c[0], c[1], theta, side, np.pi / 3, side)).reshape(3, 2)


def d_parts(spec, method, side, r):
    dr = np.full((len(THETAS), len(PHIS)), np.nan)
    dt = np.full_like(dr, np.nan)
    for i, th in enumerate(THETAS):
        for j, ph in enumerate(PHIS):
            u = np.array([np.cos(ph), np.sin(ph)])
            try:
                with np.errstate(all="ignore"):
                    d = ft.estimate(spec, tri(r * u, th, side), method)
            except np.linalg.LinAlgError:
                continue
            dr[i, j] = d @ u
            dt[i, j] = d @ np.array([-u[1], u[0]])
    return dr, dt


def h_of_r(spec, method, side, alpha, r_d, r):
    dr, dt = d_parts(spec, method, side, r)
    if not np.isfinite(dr).all() or np.nanmax(np.hypot(dr, dt)) > 2.0:
        return np.nan, dr, dt
    F = dr.mean() + ft.K_T * dt.mean() / (ft.K_R * r)
    return r - r_d - e_dyn(r, alpha) - F, dr, dt


def predict_orbit(spec, method, side, alpha, r_d):
    grid = np.arange(0.02, 0.81, 0.01)
    hs = [h_of_r(spec, method, side, alpha, r_d, r)[0] for r in grid]
    root = None
    for k in range(len(grid) - 1):
        if np.isfinite(hs[k]) and np.isfinite(hs[k + 1]) and hs[k] < 0 <= hs[k + 1]:
            lo, hi = grid[k], grid[k + 1]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if h_of_r(spec, method, side, alpha, r_d, mid)[0] < 0:
                    lo = mid
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
            break
    if root is None:
        fin = [h for h in hs if np.isfinite(h)]
        status = "collapse" if fin and min(fin) > 0 else ("runaway" if fin else "singular")
        return dict(pred_r=np.nan, pred_std=np.nan, pred_a2=np.nan, pred_status=status)
    r = root
    _, dr, dt = h_of_r(spec, method, side, alpha, r_d, r)
    s = (h_of_r(spec, method, side, alpha, r_d, r + 0.005)[0]
         - h_of_r(spec, method, side, alpha, r_d, r - 0.005)[0]) / 0.01
    Om = ft.K_T / r
    var, a2 = [], []
    for i in range(len(THETAS)):
        f = ft.K_R * (dr[i] - dr[i].mean()) + (ft.K_T / r) * (dt[i] - dt[i].mean())
        Fh = np.fft.rfft(f) / len(f)
        x = []
        for n in range(1, len(Fh)):
            G, _ = lag_filter(alpha, n * Om * ft.DT)
            x.append(2 * abs(Fh[n]) * G / np.hypot(ft.K_R * s, n * Om))
        x = np.array(x)
        var.append(np.sum(x ** 2) / 2)
        a2.append(x[1])
    return dict(pred_r=r, pred_std=float(np.sqrt(np.mean(var))), pred_a2=float(np.mean(a2)),
                pred_status="ok")


def fixed_point_bias(spec, method, side):
    out = []
    for th in THETAS:
        c = np.array([0.003, -0.002])
        try:
            for _ in range(400):
                if method == "vector":
                    c = ft.estimate(spec, tri(c, th, side), method)
                else:
                    pos = tri(c, th, side)
                    a, b, _ = ft._plane(pos, np.array([np.hypot(*spec.fn(*p)) for p in pos]))
                    c = c - 0.02 * np.array([a, b]) / ft.SIGMA
        except np.linalg.LinAlgError:
            return np.nan
        out.append(np.linalg.norm(c))
    return float(np.mean(out))


def stiction_stop_1d(alpha, vs, k=1.0, cmd_max=0.5, vmax=0.3):
    """Attraction law on a line with the Omnibot lag, speed cap and stiction."""
    finals = []
    for d0 in np.linspace(0.12, 0.5, 20):
        d, v = d0, 0.0
        for _ in range(800):
            v = alpha * v + (1 - alpha) * (-np.sign(d) * min(k * abs(d), cmd_max))
            mag = abs(v)
            if mag > vmax:
                v = np.sign(v) * vmax
            if mag < vs:
                v = 0.0
            d += ft.DT * v
        finals.append(abs(d))
    return float(np.mean(finals)), float(np.max(finals))


# ============================================================================
# Worker
# ============================================================================
def near_boundary(spec, p, R):
    ref = spec.basin(*p)
    return any(spec.basin(p[0] + R * np.cos(a), p[1] + R * np.sin(a)) != ref
               for a in np.linspace(0, 2 * np.pi, 12, endpoint=False))


def run_cell(job):
    idx, cell, n_runs = job
    rng = np.random.default_rng(SEED + idx)
    spec = ft.make_field(cell["field"], cell["eps"])
    mag = cell["method"] == "magnitude"
    R = cell["side"] / np.sqrt(3)
    rows = []
    for k in range(n_runs):
        theta = rng.uniform(0, 2 * np.pi)
        if cell["task"] == "attract":
            h = spec.start_half
            start = rng.uniform(-h, h, 2)
            cl = ft.make_cluster(spec, start, cell["side"], cell["alpha"], cell["stiction"], theta, mag)
            r = ft.run(cl, ft.attract_primitive(cell["method"]), ATTRACT_STEPS, spec, cell["method"])
            m = ft.attract_metrics(r, spec, cell["side"])
            m.update(start_x=start[0], start_y=start[1], pred_origin=bool(spec.basin(*start)),
                     near_boundary=near_boundary(spec, start, R))
        else:
            ph = rng.uniform(0, 2 * np.pi)
            start = cell["r_d"] * np.array([np.cos(ph), np.sin(ph)])
            cl = ft.make_cluster(spec, start, cell["side"], cell["alpha"], cell["stiction"], theta, mag)
            r = ft.run(cl, ft.orbit_primitive(cell["method"], cell["r_d"]), ORBIT_STEPS, spec, cell["method"])
            m = ft.orbit_metrics(r, cell["side"], ORBIT_W0)
        m.update(cell_idx=idx, run=k, theta=theta)
        rows.append(m)
    if cell["task"] == "orbit":
        pred = predict_orbit(spec, cell["method"], cell["side"], cell["alpha"], cell["r_d"])
    else:
        pred = dict(pred_bias=fixed_point_bias(spec, cell["method"], cell["side"]))
        if cell["method"] == "vector":
            pred["pred_stop_mean"], pred["pred_stop_max"] = stiction_stop_1d(cell["alpha"], cell["stiction"])
    return idx, rows, pred


# ============================================================================
# Summaries
# ============================================================================
def summarize(cell, rows, pred):
    s = dict(cell)
    s["tag"] = ft.make_field(cell["field"], cell["eps"]).tag
    s["max_side_err"] = max(r["max_side_err"] for r in rows)
    s.update(pred)
    if cell["task"] == "orbit":
        ok = [r for r in rows if not r.get("failed", True)]
        s["n_ok"] = len(ok)
        for key in ("mean_r", "std_r", "ctr_off", "a2", "est_err"):
            v = np.array([r[key] for r in ok]) if ok else np.array([np.nan])
            s[key] = float(np.nanmean(v))
            s[key + "_sd"] = float(np.nanstd(v))
    else:
        s["n_origin"] = sum(r["reached_origin"] for r in rows)
        s["n_other"] = sum((not r["reached_origin"]) and r["reached"] not in ("none", "escaped") for r in rows)
        agree = [r["reached_origin"] == r["pred_origin"] for r in rows]
        far = [a for a, r in zip(agree, rows) if not r["near_boundary"]]
        s["agree_all"] = f"{sum(agree)}/{len(agree)}"
        s["agree_far"] = f"{sum(far)}/{len(far)}"
        errs = np.array([r["final_err"] for r in rows if r["reached_origin"]])
        s["err_med"] = float(np.median(errs)) if len(errs) else np.nan
        s["err_p95"] = float(np.percentile(errs, 95)) if len(errs) else np.nan
        s["err_max"] = float(errs.max()) if len(errs) else np.nan
    return s


def print_tables(summ, log):
    def f(x, w=7, p=3):
        return f"{x:{w}.{p}f}" if isinstance(x, float) and np.isfinite(x) else f"{'-':>{w}}"
    groups = sorted(set(s["group"] for s in summ), key=lambda g: (g[0], g[1:]))
    for g in groups:
        S = [s for s in summ if s["group"] == g]
        log("")
        log(f"=== {g} ({S[0]['task']}, {S[0]['method']}) ===")
        if S[0]["task"] == "orbit":
            log(f"{'field':>24} {'side':>5} {'alpha':>5} {'stic':>5} {'r_d':>4} {'ok':>4} | "
                f"{'mean r':>7} {'pred':>7} | {'std r':>6} {'pred':>6} | {'a2':>6} {'pred':>6} | "
                f"{'ctr off':>7} {'est err':>7} {'side err':>8}")
            for s in S:
                pr = s["pred_status"] if s["pred_status"] != "ok" else ""
                log(f"{s['tag']:>24} {s['side']:5.2f} {s['alpha']:5.2f} {s['stiction']:5.3f} {s['r_d']:4.1f} "
                    f"{s['n_ok']:4d} | {f(s['mean_r'])} {f(s['pred_r']) if not pr else pr:>7} | "
                    f"{f(s['std_r'], 6)} {f(s['pred_std'], 6)} | {f(s['a2'], 6)} {f(s['pred_a2'], 6)} | "
                    f"{f(s['ctr_off'])} {f(s['est_err'])} {s['max_side_err']:8.1e}")
        else:
            log(f"{'field':>24} {'side':>5} {'alpha':>5} {'stic':>5} | {'origin':>6} {'other':>5} "
                f"{'agree all':>9} {'agree far':>9} | {'err med':>7} {'p95':>7} {'max':>7} | "
                f"{'pred bias':>9} {'pred stop':>9} {'stop max':>8} {'side err':>8}")
            for s in S:
                log(f"{s['tag']:>24} {s['side']:5.2f} {s['alpha']:5.2f} {s['stiction']:5.3f} | "
                    f"{s['n_origin']:6d} {s['n_other']:5d} {s['agree_all']:>9} {s['agree_far']:>9} | "
                    f"{f(s['err_med'], 7, 4)} {f(s['err_p95'], 7, 4)} {f(s['err_max'], 7, 4)} | "
                    f"{f(s['pred_bias'], 9, 4)} {f(s.get('pred_stop_mean', np.nan), 9, 4)} "
                    f"{f(s.get('pred_stop_max', np.nan), 8, 4)} {s['max_side_err']:8.1e}")


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo spot check of the 3-robot accuracy model")
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--only", nargs="+", default=None, help="group names, e.g. L1 G2 S1 M1")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    C = cells()
    jobs = [(i, c, args.n_runs) for i, c in enumerate(C) if not args.only or c["group"] in args.only]
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    results = {}
    with Pool(args.workers) as pool:
        for n, (idx, rows, pred) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            results[idx] = (rows, pred)
            print(f"\r{n}/{len(jobs)} cells, {time.time() - t0:.0f} s", end="", flush=True)
    print()

    summ = [summarize(C[i], *results[i]) for i in sorted(results)]
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)
    log(f"mc_field_tests: {args.n_runs} runs per cell, {len(jobs)} cells, seed {SEED}, "
        f"{time.time() - t0:.0f} s")
    log("Orbit stats over 15 to 45 s; mean r and its prediction r*, std r (radius wobble), a2 "
        "(amplitude of the cos/sin 2 phi wobble). Attraction over 40 s; 'agree far' excludes starts "
        "within one formation circumradius of the predicted basin edge.")
    print_tables(summ, log)

    sfx = "" if not args.only else "_" + "_".join(args.only)
    with open(os.path.join(OUT_DIR, f"summary{sfx}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    keys = sorted({k for s in summ for k in s})
    with open(os.path.join(OUT_DIR, f"cells{sfx}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summ)
    rk = sorted({k for i in results for r in results[i][0] for k in r})
    with open(os.path.join(OUT_DIR, f"runs{sfx}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rk)
        w.writeheader()
        for i in sorted(results):
            w.writerows(results[i][0])
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
