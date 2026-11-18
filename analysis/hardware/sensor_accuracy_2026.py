"""Colour-sensor accuracy per robot on the printed maps, 2026-11 campaign.

Each reading is compared with the 2026 net of its map (vortex_predictors_2026, saddle_predictors_2026) at the
same point. The nets were trained on 1 cm cells of the three trained robots' readings, without every 5th run
and without redwood, so the trained robots are scored on their held-out runs and redwood on all of its runs.
The palette calibration of the sensors is the 2025 one, so this measures it in use.

Readings 0.10 to 0.65 m from the critical point: nearer the centre the printed direction turns quickly, and
position error dominates. Floor: each held-out reading predicted by the mean of training readings within
1.5 cm (the trainer's own floor), the best a map can do at this resolution.
Writes hardware/outputs/sensor_accuracy_2026.csv and .json.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import map_readings as mr  # noqa: E402
import paths  # noqa: E402

SAD = mr.SAD
LOADER = {"vortex": mr.VOR, "saddle": mr.SAD}
R_MIN, R_MAX = 0.10, 0.65
OUT = os.path.join(paths.OUT_HW, "outputs")


def load_net(field):
    d = os.path.join(paths.VF, mr.NETDIR[field])
    info = pickle.load(open(os.path.join(d, "model_info.pkl"), "rb"))
    nets = []
    for part in ("hue", "sat"):
        m = SAD.Wrapped(info[f"{part}_architecture"])
        m.load_state_dict(torch.load(os.path.join(d, f"{part}_model.pth"), map_location="cpu"))
        nets.append(m.eval())
    return nets


def score(field):
    d = mr.prepare(field)
    robot = LOADER[field].load26()[4]
    assert len(robot) == len(d["H26"])
    hm, sm = load_net(field)
    Q, H, S = d["Q26"], d["H26"], d["S26"]
    r = np.hypot(*Q.T)
    ring = (r >= R_MIN) & (r < R_MAX)
    hn, sn = SAD.predict(hm, sm, Q)
    eh = -SAD.dang(hn, H)                      # reading minus map, degrees
    es = S - sn                                # reading minus map, saturation units
    rows = []
    for name in sorted(set(robot)):
        held = (robot == name) & (d["red26"] | d["test26"]) & ring
        fh = SAD.floor(Q[held], H[held], S[held], Q[d["trn26"]], H[d["trn26"]], S[d["trn26"]])[0]
        m_read, m_map = S[held] - 0.3, sn[held] - 0.3
        rows.append(dict(field=field, robot=name, trained=name in SAD.TRAINED, n=int(held.sum()),
                         hue_median_deg=float(np.median(np.abs(eh[held]))),
                         hue_rmse_deg=float(np.sqrt(np.mean(eh[held] ** 2))),
                         hue_rmse_rad=float(np.radians(np.sqrt(np.mean(eh[held] ** 2)))),
                         hue_bias_deg=float(np.mean(eh[held])),
                         hue_floor_median_deg=float(np.median(np.abs(fh))),
                         sat_rmse_pct=float(100 * np.sqrt(np.mean(es[held] ** 2))),
                         sat_bias_pct=float(100 * np.mean(es[held])),
                         magnitude_scale=float(np.sum(m_read * m_map) / np.sum(m_map ** 2))))
    return rows


def main():
    T = pd.DataFrame([row for f in ("vortex", "saddle") for row in score(f)])
    T.to_csv(os.path.join(OUT, "sensor_accuracy_2026.csv"), index=False)
    json.dump(T.to_dict("records"), open(os.path.join(OUT, "sensor_accuracy_2026.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
