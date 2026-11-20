"""
noise_summary.py  (Draft_14, Section IV-C numbers)

Prints every number Section IV-C quotes from the Fig. 7 sweep and the radius
sweep: 50 % retention crossings (log-interpolated), D/s1 ratios, retention at
sigma_uv = 0.01, the S6/S7 crossover, and the radius scaling.

Run from VF_Robot:
    venv/bin/python3 experiments/draft14/noise_summary.py            # 1000-trial run (tag d14)
    venv/bin/python3 experiments/draft14/noise_summary.py --tag d14_10k --rtag _10k
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "outputs", "mc_noise_both")


def load(tag):
    return pd.read_csv(os.path.join(D, f"noise_both_{tag}.csv"), comment="#")


def cross50(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if ys[0] < 0.5:
        return np.nan
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i])
            return float(np.exp(np.log(xs[i - 1]) + f * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d14", help="Fig. 7 tag, S<i>_<tag>")
    ap.add_argument("--rtag", default="", help="suffix of the radius tags, e.g. _10k")
    a = ap.parse_args()
    print(f"Fig. 7 sweep, tag {a.tag}: 50 % retention crossings and D/s1 ratios")
    for i in range(1, 8):
        d = load(f"S{i}_{a.tag}")
        out = []
        for ax, col in (("uv", "sigma_uv"), ("p", "sigma_p")):
            c = {}
            for tr in ("D", "s1"):
                g = d[(d.tracker == tr) & (d.axis == ax)].sort_values(col)
                c[tr] = cross50(g[col], g.success_straddle)
            out.append(f"{ax}: D {c['D']:.5f} s1 {c['s1']:.5f} ratio {c['D'] / c['s1']:.2f}")
        g = d[(d.axis == "uv") & np.isclose(d.sigma_uv, 0.01)]
        r01 = " ".join(f"{t} {100 * v:.1f}" for t, v in zip(g.tracker, g.success_straddle))
        print(f"  S{i}  " + " | ".join(out) + f" | retention at 0.01: {r01}")
    print("Retention D/s1 (%) at sigma_uv 0.004 and 0.006 from S6, S7 (crossover):")
    for i in (6, 7):
        d = load(f"S{i}_{a.tag}")
        for sg in (0.004, 0.005, 0.006):
            g = d[(d.axis == "uv") & np.isclose(d.sigma_uv, sg)]
            if len(g):
                print(f"  S{i} {sg}: " + " ".join(f"{t} {100 * v:.1f}" for t, v in zip(g.tracker, g.success_straddle)))
    print("Radius sweep, sigma_uv axis, 50 % retention crossing")
    for st in ("S3", "S4"):
        for tag, rho in ((f"{st}_r050{a.rtag}", 0.05), (f"{st}_{a.tag}", 0.075), (f"{st}_r1125{a.rtag}", 0.1125)):
            if not os.path.exists(os.path.join(D, f"noise_both_{tag}.csv")):
                print(f"  {tag}: missing")
                continue
            d = load(tag)
            res = []
            for tr in ("D", "s1"):
                g = d[(d.tracker == tr) & (d.axis == "uv")].sort_values("sigma_uv")
                res.append(f"{tr} {cross50(g.sigma_uv, g.success_straddle):.5f}")
            print(f"  {st} rho {rho:.4f}: " + "  ".join(res))


if __name__ == "__main__":
    main()
