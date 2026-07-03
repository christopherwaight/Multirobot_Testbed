"""
analyze_mc_sweeps.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex
  Makes:  the Results success-rate tables (separatrix and OW controllers)
          in plain text for user review before any number enters the
          paper; the cliff-vs-prediction comparison quoted in the
          estimator-accuracy Results prose; the heading-isotropy check
          (Lemma 1 predicts success independent of initial heading).
  Reads:  experiments/outputs/mc_separatrix/{summary,trials}_fixed.csv
          experiments/outputs/mc_ow/{summary,trials}_fixed.csv

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/analyze_mc_sweeps.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_LOOP_GRAD_THRESHOLD = 0.01   # sigma_uv where grad D error = signal


def table(df, value_col):
    piv = df.pivot(index="sigma_uv", columns="sigma_p", values=value_col)
    return piv.sort_index()


def cliff(piv, level=0.5):
    out = {}
    for p in piv.columns:
        col = piv[p]
        below = col[col < level]
        out[p] = below.index.min() if len(below) else None
    return out


def analyze(name, out_dir, success_col, extra_cols):
    spath = os.path.join(out_dir, "summary_fixed.csv")
    tpath = os.path.join(out_dir, "trials_fixed.csv")
    if not os.path.exists(spath):
        print(f"[{name}] no summary_fixed.csv yet, skipping")
        return
    s = pd.read_csv(spath, comment="#")
    print(f"\n===== {name}: success rate ({success_col}) =====")
    piv = table(s, success_col)
    print((piv * 100).round(1).to_string())
    cl = cliff(piv)
    print(f"50-percent cliff (first sigma_uv below 50%): "
          + ", ".join(f"sigma_p={p}: {v}" for p, v in cl.items()))
    base = cl.get(0.0)
    if base is not None:
        print(f"Cliff at sigma_p=0: sigma_uv = {base}; open-loop gradient "
              f"threshold = {OPEN_LOOP_GRAD_THRESHOLD}; ratio = "
              f"{base / OPEN_LOOP_GRAD_THRESHOLD:.1f}x")
    for c in extra_cols:
        print(f"\n--- {c} ---")
        print(table(s, c).round(4).to_string())

    if os.path.exists(tpath):
        t = pd.read_csv(tpath, comment="#")
        lo = t[(t.sigma_uv <= 0.01) & (t.sigma_p == 0.0)]
        if len(lo):
            q = pd.cut(lo.heading, bins=4,
                       labels=["Q1", "Q2", "Q3", "Q4"])
            iso = lo.groupby(q, observed=True)[
                lo.columns[lo.columns.str.startswith("success")][0]
            ].mean()
            print(f"\nHeading isotropy (low-noise trials, n={len(lo)}): "
                  + ", ".join(f"{k}={v:.1%}" for k, v in iso.items()))


def main():
    analyze("SEPARATRIX (Logic C)",
            os.path.join(HERE, "outputs", "mc_separatrix"),
            "success_traverse", ["success_straddle", "track_mean"])
    analyze("OKUBO-WEISS (Logic G Newton)",
            os.path.join(HERE, "outputs", "mc_ow"),
            "success_track", ["d_mean", "d_p95"])


if __name__ == "__main__":
    main()
