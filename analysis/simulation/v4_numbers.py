"""Paper numbers from the v4 hyperparameter sweep (data_simulation_campaign/hyperparameter_v4/).

1. Noise covariance check (Sec. IV-A, item F): E6ideal attraction cells on the linear fields, ideal robot
   (no delay, lag, dead zone or cap). Mean final error against the covariance propagated through the sampled
   loop, noise_rms * sqrt(pi)/2 (mean of a 2-D Rayleigh), on cells with lambda R / sigma_n >= 3 that converged.
2. Orbit closed form across the E2 sweeps (Sec. VI-A, item M): mean radius on the linear fields against
   r* = (r_d + sqrt(r_d^2 + 4C))/2, C = T V^2/(g k_r), V = min(g k_t, v_max), T = L + tau + dt/2.
Writes analysis/simulation/v4_numbers.json and prints a summary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402,F401
import theory as th  # noqa: E402
import vecsim as vs  # noqa: E402

V4 = os.path.join(paths.ROOT, "data_simulation_campaign", "hyperparameter_v4")


def load():
    M = pd.read_csv(os.path.join(V4, "manifest.csv"))
    S = pd.read_csv(os.path.join(V4, "summary.csv"))
    return M.merge(S, on="file", suffixes=("", "_s"))


def lam(field):
    return float(np.sqrt(abs(np.linalg.det(vs.make_field(field).J0))))


def noise_check(X):
    d = X[(X.exp == "E6ideal") & (X.task == "attract") & X.field.isin(vs.LINEAR) & (X.sigma_n > 0)]
    rows = []
    for r in d.itertuples():
        L = lam(r.field)
        snr = L * (r.side / np.sqrt(3)) / r.sigma_n
        pred = th.noise_rms(r.sigma_n, L, r.k, 0.0, r.dt) * np.sqrt(np.pi) / 2
        rows.append(dict(field=r.field, side=r.side, sigma_n=r.sigma_n, snr=snr, conv=r.conv, err=r.err_mean, pred=pred,
                         ratio=r.err_mean / pred))
    R = pd.DataFrame(rows)
    q = R[(R.snr >= 3) & (R.conv > 0.5)]
    return R, dict(n=len(q), median_ratio=float(q.ratio.median()), min=float(q.ratio.min()), max=float(q.ratio.max()),
                   max_dev_pct=float(100 * (q.ratio - 1).abs().max()),
                   within_4pct=float(100 * ((q.ratio - 1).abs() < 0.04).mean()))


def closed_form(X):
    d = X[X.exp.str.startswith("E2") & X.field.isin(vs.LINEAR)].copy()
    tau = np.where(d.alpha > 0, -d.dt / np.log(d.alpha.clip(lower=1e-300)), 0.0)
    T = d.delay + tau + d.dt / 2
    V = np.minimum(d.gain * d.kt, d.vmax)
    C = T * V ** 2 / (d.gain * d.kr)
    d["r_cf"] = (d.rd + np.sqrt(d.rd ** 2 + 4 * C)) / 2
    d["gap"] = d.mean_r - d.r_cf
    d["capped"] = d.gain * d.kt > d.vmax
    d["stalled"] = d.kt < d.vs                     # orbit command inside the dead zone
    k = d[(d.ok > 0.99) & ~d.stalled & d.mean_r.notna()]
    out = dict(n=len(k), median_abs_mm=float(1000 * k.gap.abs().median()), p90_abs_mm=float(1000 * k.gap.abs().quantile(0.9)),
               max_abs_mm=float(1000 * k.gap.abs().max()), n_capped=int(k.capped.sum()), n_excluded=int(len(d) - len(k)))
    out["by_exp"] = {e: dict(n=len(g), median_abs_mm=float(1000 * g.gap.abs().median()), max_abs_mm=float(1000 * g.gap.abs().max()))
                     for e, g in k.groupby("exp")}
    return d, out


def main():
    X = load()
    R, noise = noise_check(X)
    d, cf = closed_form(X)
    missing = sorted(set(pd.read_csv(os.path.join(V4, "manifest.csv")).exp) - set(X.exp))
    res = dict(noise=noise, closed_form=cf, exps_without_summary=missing)
    json.dump(res, open(os.path.join(HERE, "v4_numbers.json"), "w"), indent=1)
    R.to_csv(os.path.join(HERE, "v4_noise_cells.csv"), index=False)
    d.to_csv(os.path.join(HERE, "v4_closed_form_cells.csv"), index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
