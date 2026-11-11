"""
analyze.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex (proposed
          subsection in Draft_13_unsteady_dg_proposed.tex; not in the paper).
  Makes:  summary tables of experiments/unsteady_dg/outputs/trials_<tag>.csv.
  Reads:  trials_<tag>.csv from mc_unsteady_double_gyre.py.

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/analyze.py --tag n200
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, "outputs", f"trials_{args.tag}.csv"), comment="#")
    n = df.groupby(["eps", "tracker", "start"]).size().iloc[0]
    print(f"{len(df)} trials, {n} per cell\n")

    pct = lambda s: f"{100 * s.mean():5.1f}"
    print("Per start (percent of trials; track = median of per-trial mean |x_c - x_s|)")
    print(f"{'eps':>5} {'trk':>3} {'start':>5} {'succ':>6} {'ret':>6} {'rev':>6} "
          f"{'exit':>6} {'tout':>6} {'acq':>6} {'track':>7} {'hold':>6} {'holdmax':>8}")
    for (eps, tr, st), g in df.groupby(["eps", "tracker", "start"]):
        succ = g[g.success == 1]
        print(f"{eps:5.2f} {tr:>3} {st:>5} {pct(g.success)} {pct(g.retained)} "
              f"{pct(g.reversal)} {pct(g.outcome == 'exit')} {pct(g.outcome == 'timeout')} "
              f"{pct(g.acq_step >= 0)} {g.track_mean.median():7.4f} "
              f"{succ.hold_frac.median() if len(succ) else np.nan:6.2f} "
              f"{succ.hold_max.median() if len(succ) else np.nan:8.4f}")

    print("\nPooled over the seven starts, with Wilson 95% intervals on success")
    print(f"{'eps':>5} {'trk':>3} {'succ':>6} {'95% CI':>15} {'ret':>6} {'rev':>6} "
          f"{'exit':>6} {'tout':>6} {'track':>7} {'hold':>6} {'holdmax':>8}")
    for (eps, tr), g in df.groupby(["eps", "tracker"]):
        k, m = int(g.success.sum()), len(g)
        lo, hi = wilson(k, m)
        succ = g[g.success == 1]
        print(f"{eps:5.2f} {tr:>3} {pct(g.success)} [{100 * lo:5.1f}, {100 * hi:5.1f}] "
              f"{pct(g.retained)} {pct(g.reversal)} {pct(g.outcome == 'exit')} "
              f"{pct(g.outcome == 'timeout')} {g.track_mean.median():7.4f} "
              f"{succ.hold_frac.median() if len(succ) else np.nan:6.2f} "
              f"{succ.hold_max.median() if len(succ) else np.nan:8.4f}")

    print("\nPaired comparison (same heading and phase): trials where only one succeeds")
    for eps, g in df.groupby("eps"):
        p = g.pivot_table(index=["start", "trial"], columns="tracker", values="success")
        only_d = int(((p["D"] == 1) & (p["s1"] == 0)).sum())
        only_s = int(((p["D"] == 0) & (p["s1"] == 1)).sum())
        both = int(((p["D"] == 1) & (p["s1"] == 1)).sum())
        print(f"  eps {eps:4.2f}: both {both}, only D {only_d}, only s1 {only_s}, of {len(p)}")

    if "lost_y" in df.columns:
        print("\nWhere the straddle is lost (centroid y at the first lost step), "
              "successful trials that lost it")
        bins = [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6]
        for (eps, tr), g in df[(df.eps > 0) & (df.success == 1)].groupby(["eps", "tracker"]):
            ly = g.lost_y.dropna()
            if len(ly) == 0:
                print(f"  eps {eps:4.2f} {tr:>3}: none lost")
                continue
            h = np.histogram(ly, bins=bins)[0]
            print(f"  eps {eps:4.2f} {tr:>3}: n={len(ly)}, median y={ly.median():+.3f}, "
                  "counts " + ", ".join(f"[{a:+.1f},{b:+.1f}):{c}" for a, b, c in
                                        zip(bins[:-1], bins[1:], h)))
        print("\nReversals: did the cluster first ride down? (min_y below start y - 0.05)")
        starts = {"S1": 0.30, "S2": 0.25, "S3": 0.35, "S4": 0.0,
                  "S5": -0.25, "S6": -0.15, "S7": -0.20}
        for (eps, tr), g in df[(df.eps > 0) & (df.reversal == 1)].groupby(["eps", "tracker"]):
            sy = g.start.map(starts)
            rode = int((g.min_y < sy - 0.05).sum())
            print(f"  eps {eps:4.2f} {tr:>3}: {len(g)} reversals, {rode} rode down first, "
                  f"median steps {g.steps.median():.0f}")
        print("\nExits: where the centroid left (eps > 0)")
        for (eps, tr), g in df[(df.eps > 0) & (df.outcome == "exit")].groupby(["eps", "tracker"]):
            side = np.where(np.abs(g.final_x) > 1.0, "x-wall",
                            np.where(g.final_y < 0, "bottom", "top"))
            vals, cnt = np.unique(side, return_counts=True)
            print(f"  eps {eps:4.2f} {tr:>3}: {len(g)} exits, " +
                  ", ".join(f"{v}:{c}" for v, c in zip(vals, cnt)) +
                  f", median final (x,y)=({g.final_x.median():+.2f},{g.final_y.median():+.2f})")

    print("\nFailures by initial phase sin(omega t0) (eps > 0, both trackers)")
    df["s0"] = np.sin(2 * np.pi / 10 * df.t0)
    df["s0_bin"] = pd.cut(df.s0, [-1.0001, -0.5, 0.5, 1.0001], labels=["< -0.5", "mid", "> 0.5"])
    for (eps, tr), g in df[df.eps > 0].groupby(["eps", "tracker"]):
        row = g.groupby("s0_bin", observed=True).success.mean()
        print(f"  eps {eps:4.2f} {tr:>3}: " +
              ", ".join(f"sin={b}: {100 * v:5.1f}%" for b, v in row.items()))


if __name__ == "__main__":
    main()
