"""Step 2b: how long the 2026 hardware orbits take to settle, to choose the transient trim.

Per run, r(t) = |c(t) - p*|. The steady segment is the last 10 s (the last half of a 20 s run);
r_ss is its mean and the band is max(2 sd, 1 cm). The settling time is the first time after which
r stays inside r_ss +- band. Also writes the per-map centre fits used by every hardware script.

Run from the repo root:  trunk/.../VF_Robot/venv/bin/python3 analysis/hardware/settling_2026.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import paths  # noqa: E402
import hw2026  # noqa: E402
from centres import decode, field_zero  # noqa: E402

OUT = os.path.join(paths.OUT_HW, "outputs")
os.makedirs(OUT, exist_ok=True)


def centre_fits(runs):
    """Field zero per map from every 2026 run's readings, trained robots only (3R, 4R and pooled)."""
    fits = {}
    for field, c0 in (("vortex", (0.84, 0.87)), ("saddle", (0.81, 0.80))):
        for tag, sel in (("3R", 3), ("4R", 4), ("pooled", None)):
            P, F = [], []
            for r in runs:
                if r["field"] != field or (sel and r["nr"] != sel):
                    continue
                for k, name in enumerate(r["robots"]):
                    if name in hw2026.TRAINED:
                        P.append(r["pos"][:, k]); F.append(decode(r["hue"][:, k], r["sat"][:, k]))
            P, F = np.vstack(P), np.vstack(F)
            ok = np.isfinite(P).all(1) & np.isfinite(F).all(1)
            c, J, n = field_zero(P[ok], F[ok], c0)
            fits[f"{field}_{tag}"] = dict(p_star=c.tolist(), det=float(np.linalg.det(J)), readings=n)
    return fits


def settling(t, r):
    """Settling time of the 1 s moving mean of r; NaN if it is still outside the band at the end."""
    rs = pd.Series(r).rolling(10, center=True, min_periods=1).mean().to_numpy()
    steady = t >= max(t[-1] - 10.0, t[-1] / 2)
    rss, sd = rs[steady].mean(), rs[steady].std()
    band = max(2 * sd, 0.01)
    last_out = np.where(np.abs(rs - rss) > band)[0]
    if len(last_out) and last_out[-1] == len(t) - 1:
        return np.nan, rss, band
    ts = t[last_out[-1] + 1] if len(last_out) else t[0]
    return ts, rss, band


def main():
    cache = os.path.join(OUT, "runs2026.pkl")
    if os.path.exists(cache):
        runs = pickle.load(open(cache, "rb"))
    else:
        runs = hw2026.load_all()
        pickle.dump(runs, open(cache, "wb"))
    fits = centre_fits(runs)
    json.dump(fits, open(os.path.join(OUT, "centres_2026.json"), "w"), indent=1)
    for k, v in fits.items():
        print(f"{k:16s} p* = ({v['p_star'][0]:.4f}, {v['p_star'][1]:.4f})  det J {v['det']:+.3f}  readings {v['readings']}")

    rows = []
    for r in runs:
        if r["rd"] is None:
            continue
        ps = np.array(fits[f"{r['field']}_pooled"]["p_star"])
        rad = np.hypot(*(r["c"] - ps).T)
        ts, rss, band = settling(r["t"], rad)
        rows.append(dict(file=r["file"], field=r["field"], nr=r["nr"], rd=r["rd"], run=r["run"], T=r["t"][-1],
                         t_settle=ts, r_ss=rss, band=band,
                         **{f"bias_trim{k}": float(np.mean(rad[r["t"] >= k]) - r["rd"]) for k in (0, 2, 5, 10)}))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "settling_2026.csv"), index=False)
    print("\nsettling time (s) by field, robots, r_d: median / p90 / max")
    g = D.groupby(["field", "nr", "rd"])["t_settle"]
    print(pd.DataFrame({"median": g.median(), "p90": g.quantile(0.9), "max": g.max()}).round(1).to_string())
    print(f"\nall runs: median {D.t_settle.median():.1f} s, p75 {D.t_settle.quantile(0.75):.1f}, "
          f"p90 {D.t_settle.quantile(0.9):.1f}, p95 {D.t_settle.quantile(0.95):.1f}")
    print("\nmean radial bias (m) by trim, all runs pooled per field/robots/r_d:")
    print(D.groupby(["field", "nr", "rd"])[[f"bias_trim{k}" for k in (0, 2, 5, 10)]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
