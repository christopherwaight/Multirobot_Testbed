"""Color-sensor readings of the printed vortex and saddle maps, in the frame of the 2026 nets.

Same readings, frame, split and 1 cm cells as vortex_predictors_2026 and saddle_predictors_2026: the frame comes
from their build_report.json (critical point at the origin) and the cells from the trainers' own grid().
"""
import json
import os
import sys

import numpy as np

import paths

sys.path.insert(0, os.path.join(paths.VF, "saddle_predictors_2026"))
sys.path.insert(0, os.path.join(paths.VF, "vortex_predictors_2026"))
import nn_trainer_saddle_2026 as SAD  # noqa: E402
import nn_trainer_vortex_2026 as VOR  # noqa: E402

NETDIR = {"vortex": "vortex_predictors_2026", "saddle": "saddle_predictors_2026"}


def prepare(field):
    """Readings in the net frame and the trainers' split (test26: every 5th 2026 run; val26: run % 5 == 1)."""
    rep = json.load(open(os.path.join(paths.VF, NETDIR[field], "build_report.json")))
    if field == "saddle":
        P25, H25, S25, row25, n25 = SAD.load25()
        P26, H26, S26, run26, robot26, _ = SAD.load26()
    else:
        P25, H25, S25, row25, n25 = VOR.load25()
        P26, H26, S26, run26, robot26 = VOR.load26()
    tr = np.isin(robot26, list(SAD.TRAINED))
    if field == "vortex":                       # 2025 saturation mapped onto the 2026 line, as the trainer does
        f25 = SAD.sat_fit(P25, S25, (0.0, 0.0))
        f26 = SAD.sat_fit(P26[tr], S26[tr], (0.84, 0.86))
        S25 = f26[2] + (S25 - f25[2]) * f26[3] / f25[3]
    a = rep["align_2026_to_2025"]
    th = np.radians(a["rotation_deg"])
    Q = (P26 - a["centre_2026_motive"]) * a["scale"]
    if a["mirror"]:
        Q = Q * [1, -1]
    Q = Q @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) + a["translation"]
    c = np.array(rep["critical_point_in_old_frame"])
    return dict(P25=P25 - c, H25=H25, S25=S25, Q26=Q - c, H26=H26, S26=S26, rep=rep,
                test25=row25 >= int(0.8 * n25), test26=tr & (run26 % 5 == 0), trn26=tr & (run26 % 5 != 0),
                val26=tr & (run26 % 5 == 1), red26=robot26 == "redwood")


def cells(d, use26):
    """1 cm cells (the nets' training data when use26 = d["trn26"]): positions, hue (sin, cos), 2S - 1."""
    X, Yh, Ys = SAD.grid(d["Q26"][use26], d["H26"][use26], d["S26"][use26],
                         d["P25"][~d["test25"]], d["H25"][~d["test25"]], d["S25"][~d["test25"]])
    return X.astype(float), Yh.astype(float), Ys.astype(float)
