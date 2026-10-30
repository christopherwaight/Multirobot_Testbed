"""
Loaders for the three-robot hardware logs in trunk/robots_3 (Simulink .mat files).

Positions are OptiTrack metres with the printed map's critical point at the
origin. The older paper tables divided by the map scale 1.6764 (map units);
everything here stays in metres.

Only three-robot runs are used (decision 2026-10-30):
  vortex: vortex_tests/3-robot-tests/run1..80 (attraction, 101 samples at 10 Hz)
          and vortex_tests/3-robot-tests/radius*.mat (orbit, 601 samples)
  saddle: saddle_tests/run1..80 (43 and 64 absent, 78 runs) and saddle_tests/radius*.mat
Orbit files: radius001 is r_d = 0.01 m, radiusN is N / 10 m.
Every top-level vortex_tests file (run1-81, run101-180, orbit0XX/1XX/2XX) logs a
four-robot formation and is not used.
"""
import os
import re

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
VF_ROOT = os.path.dirname(os.path.dirname(HERE))
TRUNK = os.path.abspath(os.path.join(VF_ROOT, "..", "..", ".."))
ROBOTS3 = os.path.join(TRUNK, "robots_3")
STUDY = os.path.join(VF_ROOT, "experiments", "hyperparameter_study")
MAP_SCALE = 1.6764


def load(path):
    """One log as a dict: t, c (N, 2), phat (N, 2), phat_avg or None,
    vdes (N, 2) or None, robots (N, n_r, 2), n_robots. None if the file holds
    no trajectory."""
    m = sio.loadmat(path)
    if "cluster_position" in m:
        c = m["cluster_position"][:, :2]
    elif "cluster_pose" in m:
        c = m["cluster_pose"][:, :2]
    else:
        return None
    robots = [m[f"robot{k}_pose"][:, :2] for k in range(1, 5) if f"robot{k}_pose" in m]
    R = np.stack(robots, 1)
    return dict(file=os.path.relpath(path, TRUNK), t=m["tout"].ravel(), c=c.astype(float),
                phat=m["detected_critical_point"].astype(float),
                phat_avg=m["detected_critical_point_avg"].astype(float)
                if "detected_critical_point_avg" in m else None,
                vdes=m["desired_velocities"][:, :2].astype(float) if "desired_velocities" in m else None,
                J=m["Jhat"].astype(float) if "Jhat" in m else None,
                robots=R.astype(float), n_robots=R.shape[1],
                centroid_is_robot_mean=bool(np.abs(R.mean(1) - c).max() < 1e-9))


FOLDERS = {"vortex": os.path.join(ROBOTS3, "vortex_tests", "3-robot-tests"),
           "saddle": os.path.join(ROBOTS3, "saddle_tests")}


def attraction_runs(field):
    folder = FOLDERS[field]
    runs = []
    for i in range(1, 81):
        p = os.path.join(folder, f"run{i}.mat")
        if os.path.exists(p):
            d = load(p)
            if d is not None:
                d["run"] = i
                runs.append(d)
    return runs


def _orbit_rd(name):
    m = re.fullmatch(r"radius(\d+)\.mat", name)
    if m:
        s = m.group(1)
        return 0.01 if s == "001" else int(s) / 10
    return None


def orbit_runs():
    runs = []
    for field in ("vortex", "saddle"):
        for name in sorted(os.listdir(FOLDERS[field])):
            rd = _orbit_rd(name)
            if rd is None:
                continue
            d = load(os.path.join(FOLDERS[field], name))
            d.update(field=field, series=field, rd=rd, name=name[:-4])
            runs.append(d)
    return runs


def check_against_study(field, runs):
    """The hyperparameter study's start/final file holds metres / 1.6764. For the
    saddle it is the same 78 runs; for the vortex it is the four-robot set, so
    a mismatch is expected there. Returns the largest mismatch in metres."""
    a = np.load(os.path.join(STUDY, f"hardware_{field}_starts_finals.npy")) * MAP_SCALE
    fin = np.array([r["c"][-1] for r in runs])
    if len(fin) != len(a):
        return np.inf
    d = np.linalg.norm(fin[:, None, :] - a[None, :, 2:], axis=2)
    return float(d.min(1).max())
