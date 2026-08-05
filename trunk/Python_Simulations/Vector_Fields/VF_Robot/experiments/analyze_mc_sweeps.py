"""
analyze_mc_sweeps.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_6a.tex
  Makes:  the Results success-rate tables (separatrix and OW controllers)
          in plain text for user review before any number enters the
          paper; the cliff-vs-prediction comparison quoted in the
          estimator-accuracy Results prose; the heading-isotropy check
          (Lemma 1 predicts success independent of initial heading);
          the ring-phase-dependent zero-noise tracking-error sweep
          (Draft_6a sec:disc_noise, ~line 1949, the "36 deg period,
          0.0006 to 0.0024" claim -- see revision/items.yaml D4 and
          RING_PHASE_36DEG_PROVENANCE).
  Reads:  experiments/outputs/mc_separatrix/trials_fixed.csv
          experiments/outputs/mc_oecs_traverse/trials_fixed.csv
          experiments/outputs/mc_ow/{summary,trials}_fixed.csv

  Both trials_fixed.csv files hold a fixed start (0, 0.35) with only
  heading and noise varied per trial (confirmed 2026-08-04), so the
  zero-noise rows (sigma_uv == sigma_p == 0, n=10,000 each) are a clean
  ring-phase sweep of closed-loop tracking error with no confound from
  start position -- no new simulation needed for that claim.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/analyze_mc_sweeps.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_LOOP_GRAD_THRESHOLD = 0.01   # sigma_uv where grad D error = signal


def atomic_write_json(path, obj):
    import tempfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def ring_phase_tracking_error(name, trials_path, n_bins=36):
    """
    Zero-noise ring-phase sweep of closed-loop |x_c| tracking error, from the
    fixed-start (0, 0.35) trials already on disk. Bins heading into n_bins
    over [0, 2*pi) and reports the binned track_mean range, plus a 36-deg
    fold to check the mirror-symmetry period claim of D4/M9.

    n_bins=36 (10-deg bins, >=230 trials/bin here) reproduces Draft_6a's
    "0.0006 to 0.0024" claim almost exactly (0.00054 to 0.00242, checked
    2026-08-04). Finer bins (120, 240) drift the minimum down toward zero
    as bin occupancy drops below ~60 trials -- that is sampling noise in
    the bin estimate, not a real dip, so do not use a finer bin count to
    "improve" this number.
    """
    if not os.path.exists(trials_path):
        return None
    t = pd.read_csv(trials_path, comment="#")
    zero = t[(t.sigma_uv == 0) & (t.sigma_p == 0)].copy()
    if len(zero) == 0:
        return None

    bin_width = 2 * np.pi / n_bins
    zero["bin"] = (zero.heading // bin_width).astype(int) % n_bins
    binned = zero.groupby("bin")["track_mean"].mean()

    # Fold onto a single 36-deg period (D4/M9: signed offset is 72-deg
    # periodic and sign-changes at each mirror phase, so |x_c| should be
    # close to 36-deg periodic) and re-bin at 3-deg resolution within it.
    period_deg = 36.0
    fold_bins = 12
    fold_width = period_deg / fold_bins
    zero["fold_bin"] = (np.degrees(zero.heading) % period_deg) // fold_width
    fold_binned = zero.groupby("fold_bin")["track_mean"].mean()

    return {
        "n_trials": int(len(zero)),
        "range_min": float(binned.min()),
        "range_max": float(binned.max()),
        "phase_deg_at_max": float(np.degrees(binned.idxmax() * bin_width)),
        "phase_deg_at_min": float(np.degrees(binned.idxmin() * bin_width)),
        "fold_36deg_range_min": float(fold_binned.min()),
        "fold_36deg_range_max": float(fold_binned.max()),
    }


def heading_quartiles(name, trials_path, success_col="success_traverse",
                      sigma_uv_max=0.01):
    if not os.path.exists(trials_path):
        return None
    t = pd.read_csv(trials_path, comment="#")
    lo = t[(t.sigma_uv <= sigma_uv_max) & (t.sigma_p == 0.0)]
    if len(lo) == 0:
        return None
    q = pd.cut(lo.heading, bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    iso = lo.groupby(q, observed=True)[success_col].mean()
    return {
        "n_trials": int(len(lo)),
        "quartile_success": {k: float(v) for k, v in iso.items()},
        "min": float(iso.min()),
        "max": float(iso.max()),
    }


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

    d_trials = os.path.join(HERE, "outputs", "mc_separatrix", "trials_fixed.csv")
    s1_trials = os.path.join(HERE, "outputs", "mc_oecs_traverse", "trials_fixed.csv")

    print("\n===== Ring-phase tracking-error sweep (zero noise, fixed start) =====")
    d_ring = ring_phase_tracking_error("D", d_trials)
    s1_ring = ring_phase_tracking_error("s1", s1_trials)
    for label, r in (("D", d_ring), ("s1", s1_ring)):
        if r:
            print(f"  {label}: n={r['n_trials']}, track_mean range "
                  f"[{r['range_min']:.4f}, {r['range_max']:.4f}], "
                  f"peak at {r['phase_deg_at_max']:.1f} deg, "
                  f"min at {r['phase_deg_at_min']:.1f} deg; "
                  f"36-deg fold range [{r['fold_36deg_range_min']:.4f}, "
                  f"{r['fold_36deg_range_max']:.4f}]")
    if d_ring:
        atomic_write_json(
            os.path.join(HERE, "outputs", "mc_separatrix", "ring_phase_sweep.json"),
            {"D_tracker": d_ring})
    if s1_ring:
        atomic_write_json(
            os.path.join(HERE, "outputs", "mc_oecs_traverse", "ring_phase_sweep.json"),
            {"s1_tracker": s1_ring})

    print("\n===== Heading quartiles (40,000-trial claim, sigma_uv <= 0.01, sigma_p = 0) =====")
    d_q = heading_quartiles("D", d_trials, "success_traverse")
    s1_q = heading_quartiles("s1", s1_trials, "success_traverse")
    for label, q in (("D", d_q), ("s1", s1_q)):
        if q:
            print(f"  {label}: n={q['n_trials']}, quartile success "
                  + ", ".join(f"{k}={v:.1%}" for k, v in q["quartile_success"].items())
                  + f"  (range {q['min']:.1%} to {q['max']:.1%})")
    if d_q:
        atomic_write_json(
            os.path.join(HERE, "outputs", "mc_separatrix", "heading_quartiles.json"),
            {"quartile_success": d_q["quartile_success"], "min": d_q["min"],
             "max": d_q["max"], "n_trials": d_q["n_trials"]})
    if s1_q:
        atomic_write_json(
            os.path.join(HERE, "outputs", "mc_oecs_traverse", "heading_quartiles.json"),
            {"quartile_success": s1_q["quartile_success"], "min": s1_q["min"],
             "max": s1_q["max"], "n_trials": s1_q["n_trials"]})


if __name__ == "__main__":
    main()
