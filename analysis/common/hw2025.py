"""Load the 2025 hardware attraction runs used in the paper, with each session's fitted p*.

Sets (decision 2026-11-16: no subgroup averages, no crossed-mux runs, no 2025 orbits):
  vortex 3R  2025-08-06  data_hardware_campaign_07_31_2025/Vortex/three_robot/convergence, run1..80
  saddle 3R  2025-07-31  .../Saddle/three_robot/convergence, run1..80 (43 and 64 never existed);
             extra1 -> run 43 and extra2 -> run 64 (2026 runs, Extras/HANDLING_NOTES.txt), flagged
  vortex 4R  2025-10-09  data_hardware_campaign_10_09_2025/Vortex/four_robot/convergence/
             least_squares_run1-80 (run35 is not a run log; 79 runs)
p* per session is the zero of the field fitted to that session's own readings (centres.field_zero).
The readings live in the CSVs the reconstruction nets were trained from; each matches its session's
poses exactly (checked 2026-11-16).
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

import paths
from centres import decode, field_zero

SETS = {
    ("vortex", 3): os.path.join(paths.HW2025, "Vortex", "three_robot", "convergence"),
    ("saddle", 3): os.path.join(paths.HW2025, "Saddle", "three_robot", "convergence"),
    ("vortex", 4): os.path.join(paths.HW2025OCT, "Vortex", "four_robot", "convergence", "least_squares_run1-80"),
}
READINGS = {   # (csv, keep rows matching this session's poses)
    ("vortex", 3): os.path.join(paths.ROOT, "trunk", "robots_3", "vortex_tests", "3-robot-tests", "all_vortex_data_possible.csv"),
    ("saddle", 3): os.path.join(paths.VF, "saddle_predictors", "saddle_data.csv"),
    ("vortex", 4): os.path.join(paths.VF, "vortex_predictors", "all_vortex_data_possible.csv"),
}


def _runs(key):
    out = {}
    for f in glob.glob(os.path.join(SETS[key], "run*.mat")):
        n = int(re.search(r"run(\d+)\.mat$", f).group(1))
        S = sio.loadmat(f, squeeze_me=True, variable_names=["tout", "detected_critical_point"] +
                        [f"robot{k}_pose" for k in range(1, 5)])
        if "robot1_pose" not in S or "detected_critical_point" not in S:
            continue                                      # run35 (pacer state) and the like
        P = np.stack([S[f"robot{k}_pose"][:, :2] for k in range(1, key[1] + 1)], 1).astype(float)
        out[n] = dict(run=n, t=np.asarray(S["tout"], float), pos=P, phat=np.asarray(S["detected_critical_point"], float),
                      source="2025", file=os.path.relpath(f, paths.ROOT))
    return out


def session_readings(key, runs):
    """(P, hue, sat) of the readings CSV rows that belong to this session (matched to its poses)."""
    from scipy.spatial import cKDTree
    d = pd.read_csv(READINGS[key])
    tree = cKDTree(np.vstack([r["pos"][:, 0] for r in runs.values() if r["source"] == "2025"]))
    keep = tree.query(d.iloc[:, :2].values)[0] < 1e-6
    P = np.vstack([d.iloc[:, [5 * i, 5 * i + 1]].values[keep] for i in range(3)])
    H = np.concatenate([d.iloc[:, 5 * i + 3].values[keep] for i in range(3)])
    S = np.concatenate([d.iloc[:, 5 * i + 4].values[keep] for i in range(3)])
    ok = np.isfinite(P).all(1) & np.isfinite(H) & np.isfinite(S)
    return P[ok], H[ok], S[ok], int(keep.sum())


def session_centre(key, runs):
    """Fit p* from the readings of this session."""
    P, H, S, rows = session_readings(key, runs)
    c, J, n = field_zero(P, decode(H, S), (0.0, 0.0))
    return c, dict(rows_matched=rows, readings=n, det=float(np.linalg.det(J)))


def extras_adapted(c25, c26=(0.8123, 0.7973), theta_deg=90.0):
    """extra1 -> run 43, extra2 -> run 64 in the 2025 saddle frame (rotate about c26, move to c25);
    extra2 starts at sample 4, where it reaches group 7's start radius."""
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = {}
    for name, slot, k0 in (("extra1", 43, 0), ("extra2", 64, 4)):
        S = sio.loadmat(os.path.join(paths.EXTRAS, f"{name}.mat"), squeeze_me=True,
                        variable_names=["timing", "field_cp"] + [f"pose_r{k}" for k in (1, 2, 3)])
        P = np.stack([S[f"pose_r{k}"][:, :2] for k in (1, 2, 3)], 1).astype(float)
        tf = lambda X: (X - np.asarray(c26)) @ R.T + np.asarray(c25)
        t = S["timing"][:, 0].astype(float)
        out[slot] = dict(run=slot, t=t[k0:] - t[k0], pos=tf(P[k0:]), phat=tf(np.asarray(S["field_cp"], float)[k0:]),
                         source=f"2026 {name}", file=os.path.join("Extras", f"{name}.mat"))
    return out


def load(field, nr, with_extras=True):
    """Runs dict {run number: run}, p*, and fit info."""
    key = (field, nr)
    runs = _runs(key)
    c, info = session_centre(key, runs)
    if with_extras and key == ("saddle", 3):
        runs.update(extras_adapted(c))
    return runs, c, info
