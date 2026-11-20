"""Continuous-time robot model fitted with the real wall-clock command times.

    tau dw/dt = -w + g u(t - L)      w: robot velocity, u: command held from its wall-clock send time
    measured v(k) = (p(t_{k+1}) - p(t_k)) / (t_{k+1} - t_k)   (interval-average velocity)

Simulated on a fine grid (H s) with the exact lag update; g is closed form (model is linear in g).
"""
import pickle
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, sys.path[0])
from fit import R_ALL, ratio  # noqa: E402

H = 0.0025
ROOT = sys.argv[2]


def wall_times(r):
    S = sio.loadmat(f"{ROOT}/{r['file']}", squeeze_me=True, variable_names=["timing"])
    t = S["timing"][:, 1]
    return t - t[0]


def prepare(runs):
    cache = {}
    out = []
    for r in runs:
        if r["file"] not in cache:
            cache[r["file"]] = wall_times(r)
        tw = cache[r["file"]]
        out.append((tw, r["u"], r["v"]))
    return out


def unit_response(data, L, tau):
    """Interval-average velocity of the unit-gain model, per run (list of (n,2))."""
    Tf = max(int(np.ceil(tw[-1] / H)) + 2 for tw, _, _ in data)
    N = len(data)
    tf = np.arange(Tf) * H
    Uf = np.zeros((N, Tf, 2))
    for i, (tw, u, _) in enumerate(data):
        idx = np.searchsorted(tw[:len(u)], tf - L, side="right") - 1
        ok = idx >= 0
        Uf[i, ok] = u[idx[ok]]
    a = np.exp(-H / tau) if tau > 0 else 0.0
    w = np.zeros((N, 2))
    X = np.zeros((N, Tf + 1, 2))
    for j in range(Tf):
        uj = Uf[:, j]
        # exact integral of w over the fine step, then the exact end value
        X[:, j + 1] = X[:, j] + uj * H + (w - uj) * (tau * (1 - a) if tau > 0 else 0.0)
        w = uj + (w - uj) * a
    res = []
    for i, (tw, u, v) in enumerate(data):
        pos = np.stack([np.interp(tw, np.arange(Tf + 1) * H, X[i, :, c]) for c in (0, 1)], 1)
        res.append(np.diff(pos, axis=0)[:len(v)] / np.diff(tw)[:len(v), None])
    return res


def score(data, R, skip=0, per=None):
    num = den = 0.0
    for (tw, u, v), s in zip(data, R):
        m = np.isfinite(v).all(1) & np.isfinite(s).all(1)
        m[:skip] = False
        num += (s[m] * v[m]).sum()
        den += (s[m] ** 2).sum()
    g = num / den
    se = n = 0.0
    for (tw, u, v), s in zip(data, R):
        m = np.isfinite(v).all(1)
        m[:skip] = False
        se += ((g * s[m] - v[m]) ** 2).sum()
        n += m.sum()
    return g, np.sqrt(se / n)


if __name__ == "__main__":
    healthy = [r for r in R_ALL if ratio(r) >= 0.75]
    data = prepare(healthy)
    Ls = np.round(np.arange(0.040, 0.1301, 0.005), 3)
    taus = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.28]
    table = {}
    print("RMSE (m/s) of the continuous model with wall-clock timing; rows L (s), cols tau (s)")
    print("  L\\tau " + " ".join(f"{t:7.2f}" for t in taus))
    for L in Ls:
        row = []
        for tau in taus:
            R = unit_response(data, L, tau)
            g, e = score(data, R)
            table[(L, tau)] = (g, e, R)
            row.append(e)
        print(f"  {L:5.3f} " + " ".join(f"{x:7.4f}" for x in row))
    best = min(table, key=lambda k: table[k][1])
    print("best", best, "g %.3f rmse %.4f" % table[best][:2])
    print("\nbest L for each tau (profile):")
    for tau in taus:
        L = min(Ls, key=lambda L: table[(L, tau)][1])
        g, e, _ = table[(L, tau)]
        print(f"  tau {tau:5.3f} s (alpha at 10 Hz {np.exp(-0.1/tau) if tau else 0:.3f}): best L {L:.3f} s, "
              f"L + tau {L + tau:.3f}, g {g:.3f}, rmse {e:.4f}")
    print("\nstartup check (median measured vs model, along u(0), steps 1-3):")
    meas = np.array([[v[j] @ (u[0] / np.linalg.norm(u[0])) / np.linalg.norm(u[0]) for j in (1, 2, 3)]
                     for tw, u, v in data if np.linalg.norm(u[0]) >= 0.08])
    print("  measured        ", np.round(np.median(meas, 0), 3))
    for key in [(0.075, 0.0), (0.06, 0.02), (0.05, 0.03), (0.1, 0.0), (0.1, 0.03), (0.04, 0.28)]:
        if key not in table:
            continue
        g, e, R = table[key]
        pr = np.array([[g * R[i][j] @ (u[0] / np.linalg.norm(u[0])) / np.linalg.norm(u[0]) for j in (1, 2, 3)]
                       for i, (tw, u, v) in enumerate(data) if np.linalg.norm(u[0]) >= 0.08])
        print(f"  L {key[0]:.3f} tau {key[1]:.2f}: ", np.round(np.median(pr, 0), 3), f" rmse {e:.4f}")
    with open("physics_table.pkl", "wb") as fh:
        pickle.dump({k: v[:2] for k, v in table.items()}, fh)
