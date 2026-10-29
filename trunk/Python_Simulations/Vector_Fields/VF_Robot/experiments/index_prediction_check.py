"""
Does the sign of det(J_hat) from the FIRST plane fit predict the type (index)
of the critical point that critical_point_plane_fitting reaches?

    Index classifier (no motion needed)
        J_hat <- plane fit of u and v at the start
        predict saddle      if det J_hat < 0   (index -1)
        predict non-saddle  if det J_hat > 0   (index +1: center, node, focus)

Reason. The attraction law moves along -J^{-1} F (continuous Newton; the
saturation only rescales speed). Along that path F shrinks as F0 e^{-t} and
the step blows up where det J = 0, so the path cannot cross det J = 0 and det J
keeps its starting sign. At a critical point that sign is the index.

Fields: gyre, cellular_saddle eps 1 and 2 (see _field_tests_common.py), and a
damped Duffing field u = SIGMA y, v = SIGMA (x - x^3 / a^2 - delta y) with
a = 0.4, delta = 0.5 (saddle at 0, stable foci at +-a, det J = 0 at |x| = a/sqrt 3).
The Duffing field is not incompressible; it is here to include foci.

Run from VF_Robot:
    venv/bin/python3 experiments/index_prediction_check.py [--n-runs 100]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _field_tests_common as ft

A_D, DELTA = 0.4, 0.5
CONFIGS = [(0.33, 0.7, 0.025), (0.20, 0.7, 0.025), (0.10, 0.01, 0.0)]   # side, alpha, stiction
FIELDS = ["gyre", "cellular_1", "cellular_2", "duffing"]


def duffing():
    S = ft.SIGMA
    fn = lambda x, y: (float(S * y), float(S * (x - x ** 3 / A_D ** 2 - DELTA * y)))
    crit = [("saddle", 0.0, 0.0), ("focus(+)", A_D, 0.0), ("focus(-)", -A_D, 0.0)]
    return ft.FieldSpec("duffing", 0.0, fn, crit, lambda x, y: abs(x) < A_D / np.sqrt(3), 0.6)


def spec_of(name):
    if name == "duffing":
        return duffing()
    if name.startswith("cellular"):
        return ft.make_field("cellular_saddle", float(name.split("_")[1]))
    return ft.make_field(name)


def det_j(spec, x, y, h=1e-5):
    u1, v1 = spec.fn(x + h, y)
    u0, v0 = spec.fn(x - h, y)
    u3, v3 = spec.fn(x, y + h)
    u2, v2 = spec.fn(x, y - h)
    return ((u1 - u0) * (v3 - v2) - (u3 - u2) * (v1 - v0)) / (4 * h * h)


def job(args):
    name, side, alpha, st, seed, n = args
    spec = spec_of(name)
    g = np.linspace(-0.75, 0.75, 301)
    X, Y = np.meshgrid(g, g)
    SD = np.sign(ft.det_j_grid(spec, X, Y))
    rng = np.random.default_rng(seed)
    R = side / np.sqrt(3)
    out = []
    for _ in range(n):
        h = 0.45 if name == "gyre" else 0.6
        start = rng.uniform(-h, h, 2)
        cl = ft.make_cluster(spec, start, side, alpha, st, rng.uniform(0, 2 * np.pi))
        pos = ft.robot_positions(cl)
        uv = np.array([spec.fn(*p) for p in pos])
        a, b, _ = ft._plane(pos, uv[:, 0])
        d, e, _ = ft._plane(pos, uv[:, 1])
        pred = int(np.sign(a * e - b * d))
        r = ft.run(cl, ft.attract_primitive("vector"), 400, spec, "vector")
        got = None
        if not r.escaped:
            _, dist, i = ft.nearest_crit(spec, r.center[-1])
            if dist < 0.05:
                got = int(np.sign(det_j(spec, spec.crit[i][1], spec.crit[i][2])))
        mask = SD != np.sign(det_j(spec, *start))
        bd = np.min(np.hypot(X[mask] - start[0], Y[mask] - start[1])) if mask.any() else np.inf
        out.append((pred, got, bd / R))
    return name, side, alpha, st, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    n = ap.parse_args()
    jobs = [(f, s, a, st, 1000 + 17 * i + j, n.n_runs)
            for i, f in enumerate(FIELDS) for j, (s, a, st) in enumerate(CONFIGS)]
    with Pool(n.workers) as pool:
        res = pool.map(job, jobs)
    print(f"{'field':>11} {'side':>5} {'alpha':>5} | {'matched':>7} {'opposite':>8} {'no arrival':>10} "
          f"| miss (opposite or no arrival) by start distance to det J = 0, in R:  <1    1-2    >2")
    for name, side, alpha, st, out in res:
        m = sum(1 for p, g, _ in out if g is not None and p == g)
        o = sum(1 for p, g, _ in out if g is not None and p != g)
        na = sum(1 for _, g, _ in out if g is None)

        def miss(lo, hi):
            sel = [(p, g) for p, g, d in out if lo <= d < hi]
            return f"{sum(1 for p, g in sel if g is None or p != g)}/{len(sel)}"
        print(f"{name:>11} {side:5.2f} {alpha:5.2f} | {m:7d} {o:8d} {na:10d} | "
              f"{miss(0, 1):>66} {miss(1, 2):>6} {miss(2, np.inf):>6}")
