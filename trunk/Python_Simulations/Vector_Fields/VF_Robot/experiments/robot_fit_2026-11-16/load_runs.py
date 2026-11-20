"""Load every 2026-11 campaign run into per-robot (u, v) sequences.

u(k): global [vx vz] command sent to the robot at step k (cmd_robots_selected)
v(k): measured velocity (p(k+1) - p(k)) / dt(k), dt from the wall-clock column
Only variables listed in run_config.log_legend are read (see README_STALE_VARIABLES.md).
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import scipy.io as sio

SKIP = ("off_the_mat",)
MISPLACED = "Vortex/three_robot/vortex_001/vortex_4robots_radius_of_0_point_01_run1.mat"


def rd_from_name(rel):
    m = re.search(r"(?:radius_of_)?(?:zero_)?(0_)?point_(\d+)", rel)
    if not m:
        return np.nan
    return float("0." + m.group(2))


def load(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
        rel = os.path.relpath(f, root)
        if any(s in rel for s in SKIP) or rel == MISPLACED:
            continue
        S = sio.loadmat(f, squeeze_me=True, struct_as_record=False,
                        variable_names=["cmd_robots_selected", "timing", "run_config",
                                        "pose_r1", "pose_r2", "pose_r3", "pose_r4"])
        c, t = S["cmd_robots_selected"], S["timing"]
        nr = c.shape[1] // 3
        names = dict((str(a), str(b)) for a, b in S["run_config"].robots)
        dtw = np.diff(t[:, 1])
        dtw[(dtw <= 0.02) | (dtw > 0.5)] = np.nan
        field = rel.split("/")[0]
        mtime = os.path.getmtime(f)
        for r in range(1, nr + 1):
            p = S["pose_r%d" % r][:, :2]
            u = c[:-1, 3 * r - 3:3 * r - 1].astype(float)
            v = np.diff(p, axis=0) / dtw[:, None]
            v_nom = np.diff(p, axis=0) / 0.1
            out.append(dict(file=rel, field=field, nr=nr, rd=rd_from_name(rel),
                            robot=names["r%d" % r], slot=r, u=u, v=v, v_nom=v_nom,
                            dt=dtw, t0=t[0, 1], mtime=mtime))
    return out


if __name__ == "__main__":
    root, dst = sys.argv[1], sys.argv[2]
    runs = load(root)
    with open(dst, "wb") as fh:
        pickle.dump(runs, fh)
    print(len(runs), "robot-runs from", len({r["file"] for r in runs}), "files")
