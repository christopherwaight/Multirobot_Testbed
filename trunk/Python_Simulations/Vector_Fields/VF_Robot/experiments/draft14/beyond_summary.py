"""
beyond_summary.py  (Draft_14, Section IV-E and Table II numbers)

Pooled noise-free outcomes per field and tracker (success, retention against
the separatrix, retention against the tracker's own true trench on the bent
gyre, reversal), the median fitted |w|/2r along the ride, and the extra
numbers the text quotes (time-periodic straddle-loss heights, bent-gyre
S6/S7 outcomes, Bickley s1 end offset from its trench).

Run from VF_Robot:
    venv/bin/python3 experiments/draft14/beyond_summary.py                 # tag d14 (500/300 per start)
    venv/bin/python3 experiments/draft14/beyond_summary.py --tag d14_10k   # overnight
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "experiments", "unsteady_dg", "outputs")


def pct(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d14")
    a = ap.parse_args()
    tp = pd.read_csv(os.path.join(OUT, f"trials_{a.tag}.csv"), comment="#")
    bt = pd.read_csv(os.path.join(OUT, f"bent_{a.tag}.csv"), comment="#")
    bk = pd.read_csv(os.path.join(OUT, f"bickley_{a.tag}.csv"), comment="#")
    pr_path = os.path.join(OUT, f"path_ratio_{a.tag}.csv")
    pr = pd.read_csv(pr_path) if os.path.exists(pr_path) else None

    def ratio(field, par, tr):
        if pr is None:
            return float("nan")
        g = pr[(pr.field == field) & np.isclose(pr.par, par) & (pr.tracker == tr)]
        return float(g.ratio_med.median()) if len(g) else float("nan")

    print(f"Table II rows (tag {a.tag}); trials per cell in brackets")
    rows = [("Steady gyre", tp[tp.eps == 0], "tp", 0.0),
            ("Time-periodic 0.1", tp[np.isclose(tp.eps, 0.1)], "tp", 0.1),
            ("Time-periodic 0.25", tp[np.isclose(tp.eps, 0.25)], "tp", 0.25),
            ("Bent 0.1", bt[np.isclose(bt.delta, 0.1)], "bent", 0.1),
            ("Bent 0.2", bt[np.isclose(bt.delta, 0.2)], "bent", 0.2),
            ("Bickley", bk, "bickley", 0.3)]
    for name, df, field, par in rows:
        cells = []
        for tr in ("D", "s1"):
            h = df[df.tracker == tr]
            own = pct(h.retained_own.mean()) if "retained_own" in h else "-"
            cells.append(f"{tr}[{len(h)}] succ {pct(h.success.mean())} ret {pct(h.retained.mean())} "
                         f"own {own} rev {pct(h.reversal.mean())} ratio {ratio(field, par, tr):.3f}")
        print(f"  {name:18s} " + " | ".join(cells))
    b0 = bt[bt.delta == 0]
    print(f"  (steady own-trench retention from bent delta=0: D {pct(b0[b0.tracker=='D'].retained_own.mean())}, "
          f"s1 {pct(b0[b0.tracker=='s1'].retained_own.mean())})")

    x = tp[np.isclose(tp.eps, 0.1) & (tp.tracker == "s1") & (tp.success == 1) & (tp.retained == 0)]
    print(f"Time-periodic 0.1, s1: {len(x)} successful trials lose the straddle; "
          f"{pct(np.mean(np.abs(x.lost_y) < 0.1))} % within 0.1 of the crest")
    b = bt[(bt.delta > 0) & (bt.tracker == "D") & bt.start.isin(["S6", "S7"])]
    print(f"Bent D from S6/S7: success {pct(b.success.mean())} %, outcomes {b.outcome.value_counts().to_dict()}, "
          f"median end ({b.final_x.median():.3f}, {b.final_y.median():.3f})")

    import experiments.unsteady_dg.fig_notes as N
    s = bk[bk.tracker == "s1"]
    xs = np.linspace(-0.49, 0.49, 99)
    _, ys1 = N.trench_bickley(xs, n=301)
    inside = np.abs(s.final_x) < 0.49
    off = np.abs(s.final_y[inside] - np.interp(s.final_x[inside], xs, ys1))
    print(f"Bickley s1: median end offset from its own trench {np.nanmedian(off):.4f} "
          f"({pct(inside.mean())} % of trials end inside the separatrix span)")


if __name__ == "__main__":
    main()
