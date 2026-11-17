"""
Read the 2026-11-16 simulation campaign as pandas tables.

    import load_campaign as lc
    C = lc.cells("E3delay")                 # one row per cell: settings (with tau) and summary metrics
    T = lc.trials(C.iloc[0])                # one row per trial of one cell
    T = lc.trials_of("E11", field="vortex", task="attract")   # trials of every matching cell

    venv/bin/python3 data_simulation_campaign/load_campaign.py --summarize   # (re)build summary.csv

Run from the VF_Robot folder with its venv, or put this folder on sys.path.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "hyperparameter_v4")
STUDY = os.path.join(os.path.dirname(HERE), "trunk", "Python_Simulations", "Vector_Fields", "VF_Robot",
                     "experiments", "hyperparameter_study")
sys.path.insert(0, STUDY)
import load as study_load  # noqa: E402


def _tau(alpha, dt):
    """Lag time constant (s) from the per-step lag coefficient alpha = exp(-dt / tau)."""
    alpha = np.asarray(alpha, float)
    with np.errstate(divide="ignore"):
        return np.where((alpha > 0) & (alpha < 1), -np.asarray(dt, float) / np.log(alpha), 0.0)


def cells(exp=None, sweep=SWEEP):
    """Cell table: manifest settings, tau, whether the cell has finished, and its
    summary metrics once summary.csv exists. exp filters by experiment prefix."""
    M = pd.read_csv(os.path.join(sweep, "manifest.csv"))
    for k, v in study_load.MISSING.items():
        if k not in M:
            M[k] = v
    M["tau"] = _tau(M["alpha"], M["dt"])
    M["done"] = [os.path.exists(os.path.join(sweep, "cells", f)) for f in M["file"]]
    path = os.path.join(sweep, "summary.csv")
    if os.path.exists(path):
        S = pd.read_csv(path)
        M = M.merge(S[["file"] + [c for c in S.columns if c not in M.columns]], on="file", how="left")
    if exp is not None:
        M = M[M["exp"].str.startswith(exp)]
    return M.reset_index(drop=True)


def trials(cell, sweep=SWEEP):
    """Per-trial table of one cell (a row of cells()). Columns are the cell's
    settings plus every per-trial metric; E7 random cells also carry each
    trial's drawn parameters as p_<name>."""
    with np.load(os.path.join(sweep, "cells", cell["file"])) as z:
        arrays = {k: z[k] for k in z.files if z[k].ndim == 1}
    n = len(arrays["escaped"])
    T = pd.DataFrame(arrays)
    for k in ("exp", "task", "method", "field", "eps", "family", "dt", "stiction", "hw", "tau",
              "side", "alpha", "vs", "vmax", "gain", "delay", "k", "vcmd_max", "kt", "kr", "rclamp",
              "rd", "sigma_n"):
        if k in cell and k not in T:
            T[k] = [cell[k]] * n
    T.insert(0, "trial", np.arange(n))
    T.insert(0, "cell", cell["file"])
    return T


def trials_of(exp, sweep=SWEEP, **match):
    """Per-trial table of every finished cell of an experiment whose settings
    equal the given values, e.g. trials_of("E2delay", field="vortex", method="fit")."""
    C = cells(exp, sweep)
    C = C[C["done"]]
    for k, v in match.items():
        C = C[np.isclose(C[k], v)] if isinstance(v, (int, float)) else C[C[k] == v]
    return pd.concat([trials(r, sweep) for _, r in C.iterrows()], ignore_index=True) if len(C) else pd.DataFrame()


def summarize(sweep=SWEEP):
    """One summary row per finished cell (the hyperparameter study's metrics), to summary.csv."""
    rows = [r for r in study_load.manifest(sweep) if os.path.exists(r["path"]) and not r["random"]]
    out = []
    for r in rows:
        s = study_load.summary(r)
        out.append(dict(file=r["file"], **s))
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(sweep, "summary.csv"), index=False)
    return S


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        S = summarize()
        print(f"summary.csv: {len(S)} cells")
    else:
        C = cells()
        print(C.groupby("exp")["done"].agg(["sum", "count"]).rename(columns={"sum": "done", "count": "cells"}))
