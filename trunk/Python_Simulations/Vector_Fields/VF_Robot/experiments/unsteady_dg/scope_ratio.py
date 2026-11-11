"""
scope_ratio.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (candidate scope
          condition for the proposed IV-E to IV-G; nothing here is in the paper).
  Makes:  the rotation-to-strain ratio |omega| / 2r along the TRUE separatrix
          of every test field, next to the tracker outcomes of the Monte Carlo
          runs, to test whether strain dominance on the separatrix predicts
          success.
  Reads:  experiments/unsteady_dg/outputs/trials_n1000.csv,
          bent_n500.csv, bickley_n1000.csv
  Writes: stdout only.

  r = sqrt(s_n^2 + s_s^2) is the strain rate (half the strain-eigenvalue
  gap); |omega|/2 is the rotation rate. Ratio < 1: strain-dominated, which in
  incompressible flow is D < 0. Statistics are weighted by arc length. Points
  where r and omega both vanish (the steady gyre's crest) are skipped.

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/scope_ratio.py
"""
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.fields.environments.Double_Gyre import double_gyre_static
from experiments.unsteady_dg.mc_bent_double_gyre import bent_gyre
import experiments.unsteady_dg.mc_bickley_jet as bj

OUT = os.path.join(project_root, "experiments", "unsteady_dg", "outputs")
H = 1e-5
OMEGA_F = 2 * np.pi / 10


def ratio_at(fun, x, y):
    up, um = fun(x + H, y), fun(x - H, y)
    vp, vm = fun(x, y + H), fun(x, y - H)
    ux, vx = (up[0] - um[0]) / (2 * H), (up[1] - um[1]) / (2 * H)
    uy, vy = (vp[0] - vm[0]) / (2 * H), (vp[1] - vm[1]) / (2 * H)
    r = np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx))
    w = abs(vx - uy)
    if r < 1e-9 and w < 1e-9:
        return np.nan
    return w / (2 * r) if r > 1e-12 else np.inf


def stats(vals, weights):
    vals, weights = np.asarray(vals, float), np.asarray(weights, float)
    m = np.isfinite(vals)
    v, wt = vals[m], weights[m]
    order = np.argsort(v)
    v, wt = v[order], wt[order]
    cw = np.cumsum(wt) / wt.sum()
    q = lambda p: float(v[np.searchsorted(cw, p)])
    return {"median": q(0.5), "p90": q(0.9), "max": float(v.max()),
            "frac>0.5": float(wt[v > 0.5].sum() / wt.sum()),
            "frac>1": float(wt[v > 1.0].sum() / wt.sum())}


def gyre_curve(xfun, ys):
    xs = np.array([xfun(y) for y in ys])
    seg = np.hypot(np.gradient(xs), np.gradient(ys))
    return xs, seg


def main():
    ys = np.linspace(-0.495, 0.495, 991)
    rows = {}

    # steady and time-periodic gyre, averaged over 40 phases
    for eps in (0.0, 0.1, 0.25):
        cfg = {"A": 0.1, "eps": eps, "omega": OMEGA_F}
        vals, wts = [], []
        for t in np.linspace(0, 10, 40, endpoint=False):
            a = eps * np.sin(OMEGA_F * t)
            b = 1 - 2 * a
            xs0 = 2 / (b + np.sqrt(b * b + 4 * a)) - 1
            f = lambda x, y: double_gyre_static(x, y, t, cfg)
            for y in ys:
                vals.append(ratio_at(f, xs0, y))
                wts.append(1.0)
        rows[("time-periodic gyre" if eps else "steady gyre", eps)] = stats(vals, wts)

    # bent gyre
    for d in (0.1, 0.2):
        f = lambda x, y: bent_gyre(x, y, config={"delta": d})
        xs, seg = gyre_curve(lambda y: -d * np.cos(np.pi * y), ys)
        rows[("bent gyre", d)] = stats([ratio_at(f, x, y) for x, y in zip(xs, ys)], seg)

    # Bickley jet, along the traced arc (uniform in arc length)
    arc = bj.ARC[1:-1]
    rows[("Bickley jet", bj.EPS)] = stats([ratio_at(bj.bickley, x, y) for x, y in arc],
                                          np.ones(len(arc)))

    # outcomes
    def out(df):
        g = df.groupby("tracker")
        return {tr: (100 * gg.success.mean(), 100 * gg.retained.mean()) for tr, gg in g}

    tp = pd.read_csv(os.path.join(OUT, "trials_n1000.csv"), comment="#")
    bent = pd.read_csv(os.path.join(OUT, "bent_n500.csv"), comment="#")
    bick = pd.read_csv(os.path.join(OUT, "bickley_n1000.csv"), comment="#")
    outcomes = {
        ("steady gyre", 0.0): out(tp[tp.eps == 0.0]),
        ("time-periodic gyre", 0.1): out(tp[tp.eps == 0.1]),
        ("time-periodic gyre", 0.25): out(tp[tp.eps == 0.25]),
        ("bent gyre", 0.1): out(bent[bent.delta == 0.1]),
        ("bent gyre", 0.2): out(bent[bent.delta == 0.2]),
        ("Bickley jet", bj.EPS): out(bick),
    }

    print("|omega|/2r along the true separatrix (arc-length weighted), and outcomes")
    print(f"{'field':>19} {'par':>5} {'median':>7} {'p90':>6} {'max':>6} "
          f"{'>0.5':>6} {'>1':>6} | {'D succ':>6} {'D ret':>6} {'s1 succ':>7} {'s1 ret':>6}")
    for key, s in rows.items():
        o = outcomes[key]
        print(f"{key[0]:>19} {key[1]:5.2f} {s['median']:7.3f} {s['p90']:6.3f} "
              f"{s['max']:6.2f} {100 * s['frac>0.5']:5.1f}% {100 * s['frac>1']:5.1f}% | "
              f"{o['D'][0]:6.1f} {o['D'][1]:6.1f} {o['s1'][0]:7.1f} {o['s1'][1]:6.1f}")


if __name__ == "__main__":
    main()
