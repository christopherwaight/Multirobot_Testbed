"""Load the 2026-11 hardware runs (data_hardware_campaign_11_11_2026, raw folder).

Only variables listed in each file's run_config.log_legend are used: the files are whole-workspace
saves with stale variables (README_STALE_VARIABLES.md). Robot count comes from cmd_robots_selected.
Skipped: Robot4_vortex_050_runs_off_the_mat.mat (failed run) and the misplaced copy of a 4-robot file
in Vortex/three_robot/vortex_001.
"""
import glob
import os
import re

import numpy as np
import scipy.io as sio

from paths import HW2026

TRAINED = ("celeste", "tidal", "pacific-blue")      # robots the sensor net was trained on
VARS = ["run_config", "timing", "cmd_robots_selected", "field_cp", "field_cp_jacobian", "field_jacobian",
        "sensor_hue", "sensor_sat", "cluster_state"] + [f"pose_r{r}" for r in range(1, 5)]


def files():
    out = []
    for f in sorted(glob.glob(os.path.join(HW2026, "**", "*.mat"), recursive=True)):
        rel = os.path.relpath(f, HW2026)
        if "off_the_mat" in rel or ("three_robot" in rel and "4robots" in rel):
            continue
        out.append(f)
    return out


def load(path):
    S = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=VARS)
    rc = S["run_config"]
    legend = set(rc.log_legend._fieldnames)
    nr = S["cmd_robots_selected"].shape[1] // 3
    cp_name = "field_cp" if nr == 3 else "field_cp_jacobian"
    assert cp_name in legend and "field_jacobian" in legend, path
    names = dict((str(a), str(b)) for a, b in rc.robots)
    rel = os.path.relpath(path, HW2026)
    m = re.search(r"point_(\d+)_run(\d+)", os.path.basename(path))
    pose = np.stack([S[f"pose_r{r}"] for r in range(1, nr + 1)], 1)          # (n, nr, 3) [x z heading]
    return dict(file=rel, field=rel.split(os.sep)[0].lower(), nr=nr,
                rd=float("0." + m.group(1)), run=int(m.group(2)),
                robots=[names[f"r{r}"] for r in range(1, nr + 1)],
                t=S["timing"][:, 0].astype(float), wall=(S["timing"][:, 1] - S["timing"][0, 1]).astype(float),
                pos=pose[:, :, :2].astype(float), heading=pose[:, :, 2].astype(float),
                c=pose[:, :, :2].mean(1).astype(float), phat=np.asarray(S[cp_name], float),
                J=np.asarray(S["field_jacobian"], float)[:, :4], jtype=np.asarray(S["field_jacobian"], float)[:, 7],
                hue=np.atleast_2d(S["sensor_hue"]).astype(float), sat=np.atleast_2d(S["sensor_sat"]).astype(float),
                cmd=np.asarray(S["cmd_robots_selected"], float))


def load_all():
    return [load(f) for f in files()]
