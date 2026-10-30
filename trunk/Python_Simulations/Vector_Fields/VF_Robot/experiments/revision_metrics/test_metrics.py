"""
Synthetic checks for metrics.py. Run from VF_Robot:
    venv/bin/python3 -m pytest experiments/revision_metrics/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as mt


def test_straight_approach():
    # centroid moves from (0.5, 0) to the origin at 0.1 m/s, then stays: e[j] = 0.5 - 0.01 j
    j = np.arange(101)
    x = np.maximum(0.5 - 0.01 * j, 0.0)
    c = np.stack([x, np.zeros_like(x)], 1)
    out = mt.attraction_trial(c, c * 0.0, dt=0.1)
    assert np.isclose(out["e_final"], 0.0)
    assert np.isclose(out["t_hit_5cm"], 4.5) and np.isclose(out["t_settle_5cm"], 4.5)
    assert np.isclose(out["t_hit_3cm"], 4.7)
    assert np.isclose(out["path_eff"], 1.0) and np.isclose(out["L_path"], 0.5)
    assert np.isclose(out["E_real"], 50 * 0.01 ** 2 / 0.1)
    assert not out["fail_3cm"] and not out["left_ws"]


def test_settle_after_reentry():
    err = np.array([0.1, 0.02, 0.06, 0.03, 0.01, 0.01])
    hit, settle = mt.settle_times(err, 0.05, dt=1.0)
    assert hit == 1.0 and settle == 3.0
    assert np.isnan(mt.settle_times(np.array([0.1, 0.01, 0.2]), 0.05)[1])


def test_failure_and_workspace():
    c = np.array([[0.0, 0.0], [0.9, 0.0], [0.04, 0.0]])
    out = mt.attraction_trial(c, c)
    assert out["left_ws"] and out["fail_5cm"]
    c2 = np.array([[0.2, 0.0], [0.045, 0.0]])
    out2 = mt.attraction_trial(c2, c2)
    assert out2["fail_4cm"] and not out2["fail_5cm"]


def test_circle_orbit():
    # radius 0.3 about the origin, period 12 s (120 samples), 601 samples, estimate exact
    j = np.arange(601)
    th = 2 * np.pi * j / 120
    c = 0.3 * np.stack([np.cos(th), np.sin(th)], 1)
    out, laps = mt.orbit_run(c, np.zeros_like(c), rd=0.3)
    assert abs(out["bias_w1"]) < 1e-12 and out["rmse_w1"] < 1e-12 and abs(out["ctrl_bias_w1"]) < 1e-12
    # 500 samples after the transient = 4.16 laps -> 4 complete laps of 12 s
    assert out["n_laps"] == 4
    assert np.allclose([L["period"] for L in laps], 12.0)
    assert np.isclose(out["speed_w1"], 0.3 * 2 * np.pi / 12.0, rtol=1e-3)


def test_offset_orbit_bias_signs():
    # orbit of radius 0.25 about (0.05, 0): r* varies, estimate at (0.05, 0) gives r_est = 0.25
    j = np.arange(601)
    th = -2 * np.pi * j / 100          # clockwise
    c = np.stack([0.05 + 0.25 * np.cos(th), 0.25 * np.sin(th)], 1)
    ph = np.tile([0.05, 0.0], (601, 1))
    out, _ = mt.orbit_run(c, ph, rd=0.2)
    assert np.isclose(out["ctrl_bias_w1"], 0.05)
    assert out["direction"] == -1.0 and out["n_laps"] == 4   # 499 intervals = 4.99 laps
    assert out["rmse_w1"] >= abs(out["bias_w1"]) and out["mae_w1"] <= out["rmse_w1"] + 1e-15


def test_classify_table_I():
    J = {"saddle": [0, -1, -1, 0], "center": [0, -1, 1, 0], "sink": [-1, 0, 0, -1], "source": [1, 0, 0, 1],
         "stable spiral": [-0.15, -0.4, 0.4, -0.15], "unstable spiral": [0.15, 0.4, -0.4, 0.15],
         "invalid": [1, 1, 1, 1]}
    for name, j in J.items():
        assert mt.TYPE_NAMES[int(mt.classify(j))] == name, name
    assert mt.classify([np.nan] * 4) == 0
    weak = [0.1, -1, 1, 0.1]                        # |tr| / sqrt(det) = 0.199
    assert mt.classify(weak, eps_c=0.2) == 6 and mt.classify(weak, eps_c=0.1) == 5
    star = [-1, 0.05, -0.05, -1]                    # sink with a small rotation: node band
    assert mt.classify(star, eps_c=0.2) == 2 and mt.classify(star, eps_c=0.05) == 4


def test_classification_scores():
    good, bad = [0, -1, -1, 0], [-1, 0, 0, -1]
    J = np.array([bad] * 5 + [good] * 15 + [[np.nan] * 4] * 2, float)
    s = mt.classification_scores(J, true_code=1, dt=0.1)
    assert np.isclose(s["cls_valid_frac"], 20 / 22) and np.isclose(s["cls_acc"], 15 / 20)
    assert np.isclose(s["cls_t_first"], 0.5) and np.isnan(s["cls_t_stable"])   # ends on invalid cycles
    s2 = mt.classification_scores(J[:20], true_code=1, dt=0.1, start=5)
    assert s2["cls_acc"] == 1.0 and np.isclose(s2["cls_t_stable"], 0.5)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
