"""Score every hardware run with the paper's metrics (revision_metrics/metrics.py).

Attraction: 2025 vortex 3R (80), saddle 3R (78 + extra1 and extra2 as runs 43 and 64), vortex 4R
(Oct 2025 least squares, 79). Orbits: every 2026 run (vortex and saddle, 3R and 4R). Every position is
measured from its session's fitted p* (common/hw2025.py, outputs/centres_2026.json). Orbit transient
3 s (metrics.TRANSIENT_STEPS). Classification on the 2026 orbit cycles (saddle readings reflected,
README_hardware 3.3). Temporal reading noise from the extras' parked final 5 s.

Run from the repo root (after settling_2026.py, which caches the 2026 runs and fits their centres):
    trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3 analysis/hardware/score_hardware.py
Outputs in analysis/hardware/outputs/.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import paths  # noqa: E402
import hw2025  # noqa: E402
import metrics as mt  # noqa: E402  (revision_metrics, on the path through paths)

OUT = os.path.join(paths.OUT_HW, "outputs")
TRUE_CODE = {"vortex": 6, "saddle": 1}          # printed maps: a center and a saddle


def inside_hull(P, x):
    hull = ConvexHull(P)
    return bool(Path(P[hull.vertices]).contains_point(x))


def attraction(field, nr):
    runs, ps, info = hw2025.load(field, nr)
    rows = []
    for n, r in sorted(runs.items()):
        P = r["pos"] - ps
        c = P.mean(1)
        row = mt.attraction_trial(c, r["phat"] - ps, target=(0.0, 0.0), robots=P)
        row.update(field=field, nr=nr, run=n, group=(n - 1) // 10 + 1, source=r["source"], file=r["file"],
                   inside_final=inside_hull(P[-1], np.zeros(2)),
                   final_speed=float(np.mean(np.hypot(*np.diff(c[-6:], axis=0).T)) / mt.DT))
        rows.append(row)
    return rows, ps, info


def summary_attraction(rows):
    S = mt.summarize_attraction(rows)
    S["inside_final_pct"] = 100 * np.mean([r["inside_final"] for r in rows])
    S["final_speed_med"] = float(np.median([r["final_speed"] for r in rows]))
    S["t_conv_med"] = S["t_settle_med_5cm"]
    return S


def orbits():
    runs = pickle.load(open(os.path.join(OUT, "runs2026.pkl"), "rb"))
    fits = json.load(open(os.path.join(OUT, "centres_2026.json")))
    rows, laps = [], []
    for r in runs:
        ps = np.array(fits[f"{r['field']}_pooled"]["p_star"])
        P = r["pos"] - ps
        row, lp = mt.orbit_run(P.mean(1), r["phat"] - ps, r["rd"], target=(0.0, 0.0), robots=P)
        J = r["J"].copy()
        if r["field"] == "saddle":
            J[:, 2:] *= -1            # reflect one decoded component (the saddle map is not mirrored)
        row.update(mt.classification_scores(J, TRUE_CODE[r["field"]], start=mt.TRANSIENT_STEPS))
        code = mt.classify(J[mt.TRANSIENT_STEPS:])
        row["cls_rotational"] = float(np.isin(code, (4, 5, 6)).mean())     # complex eigenvalues: center or spiral
        for trim in (2, 5, 10):
            o, _ = mt.orbit_run(P.mean(1), r["phat"] - ps, r["rd"], target=(0.0, 0.0), transient=10 * trim)
            row[f"bias_trim{trim}"], row[f"sdw_trim{trim}"] = o["bias_w1"], o["sd_within_w1"]
        row.update(field=r["field"], nr=r["nr"], run=r["run"], file=r["file"], T=float(r["t"][-1]),
                   lost=bool(row["left_ws"] or not (row["revs_w1"] >= 0.25)))
        rows.append(row)
        laps += [dict(L, field=r["field"], nr=r["nr"], rd=r["rd"], run=r["run"]) for L in lp]
    return rows, laps


def summary_orbits(O):
    out = []
    for (field, nr, rd), g in O.groupby(["field", "nr", "rd"]):
        k = g[~g.lost]
        b = k.bias_w1
        out.append(dict(field=field, nr=nr, rd=rd, n=len(g), lost=int(g.lost.sum()), bias=b.mean(),
                        s_b=b.std(ddof=1), s_w=k.sd_within_w1.mean(), rmse=k.rmse_w1.mean(), mean_r=k.mean_r_w1.mean(),
                        ctrl_bias=k.ctrl_bias_w1.mean(), est_err=k.est_err_w1.mean(), speed=k.speed_w1.mean(),
                        revs=k.revs_w1.median(), practical=bool(abs(b.mean()) <= 0.25 * rd), cls_acc=100 * k.cls_acc.mean(),
                        cls_rotational=100 * k.cls_rotational.mean(),
                        bias_trim2=k.bias_trim2.mean(), bias_trim5=k.bias_trim5.mean(), bias_trim10=k.bias_trim10.mean()))
    return pd.DataFrame(out)


def parked_noise():
    """Cycle-to-cycle change of each reading while the cluster is parked (extras, last 5 s)."""
    import scipy.io as sio
    rows = []
    for name in ("extra1", "extra2", "extra3"):
        S = sio.loadmat(os.path.join(paths.EXTRAS, f"{name}.mat"), squeeze_me=True, variable_names=["sensor_hue", "sensor_sat"])
        h, s = np.atleast_2d(S["sensor_hue"])[-50:], np.atleast_2d(S["sensor_sat"])[-50:]
        speed = 0.3 * (s - 0.3) / 0.7                      # field speed, m/s (saturation 1 = 0.3 m/s)
        ang = 2 * np.pi * h
        u, v = speed * np.sin(ang), -speed * np.cos(ang)
        dv = np.hypot(np.diff(u, axis=0), np.diff(v, axis=0))
        for k in range(dv.shape[1]):
            rows.append(dict(run=name, robot=k + 1, median_dv=float(np.median(dv[:, k])), mean_dv=float(dv[:, k].mean()),
                             sd_dhue_deg=float(np.degrees(np.std(np.diff(np.unwrap(ang[:, k]))))),
                             sd_dsat=float(np.std(np.diff(s[:, k])))))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"centres": {}, "attraction": {}}
    allrows = []
    for field, nr in (("vortex", 3), ("saddle", 3), ("vortex", 4)):
        rows, ps, info = attraction(field, nr)
        allrows += rows
        res["centres"][f"{field}_{nr}R_2025"] = dict(p_star=ps.tolist(), **info)
        res["attraction"][f"{field}_{nr}R"] = summary_attraction(rows)
        if field == "saddle":
            res["attraction"][f"{field}_{nr}R_without_extras"] = summary_attraction([r for r in rows if r["source"] == "2025"])
    A = pd.DataFrame(allrows)
    A.to_csv(os.path.join(OUT, "attraction_per_trial.csv"), index=False)
    O_rows, laps = orbits()
    O = pd.DataFrame(O_rows)
    O.to_csv(os.path.join(OUT, "orbit_per_run.csv"), index=False)
    pd.DataFrame(laps).to_csv(os.path.join(OUT, "orbit_per_lap.csv"), index=False)
    SO = summary_orbits(O)
    SO.to_csv(os.path.join(OUT, "orbit_summary.csv"), index=False)
    N = parked_noise()
    N.to_csv(os.path.join(OUT, "parked_noise.csv"), index=False)
    res["parked_noise_mean_dv"] = float(N.mean_dv.mean())          # m/s per cycle; most cycles repeat the last reading
    res["parked_noise_sd_dhue_deg"] = float(N.sd_dhue_deg.mean())
    res["classification_2026_orbits"] = {f"{f}_{n}R": dict(strict=float(100 * O[(O.field == f) & (O.nr == n) & ~O.lost].cls_acc.mean()),
                                                            rotational=float(100 * O[(O.field == f) & (O.nr == n) & ~O.lost].cls_rotational.mean()))
                                         for f in ("vortex", "saddle") for n in (3, 4)}
    json.dump(res, open(os.path.join(OUT, "hardware_summary.json"), "w"), indent=1, default=float)

    keys = ["n", "bias", "scatter", "e_mean", "e_sd", "e_med", "e_p95", "est_ss_mean", "t_conv_med", "E_real_mean",
            "success_3cm", "success_4cm", "success_5cm", "inside_final_pct", "final_speed_med", "left_ws"]
    print(pd.DataFrame({k: {kk: v.get(kk) for kk in keys} for k, v in res["attraction"].items()}).round(4).to_string())
    print("\n", SO.round(3).to_string())
    print("\nclassification (% of cycles, 2026 orbits):", res["classification_2026_orbits"])
    print(f"parked reading change per cycle, mean (m/s): {res['parked_noise_mean_dv']:.2e}; hue sd {res['parked_noise_sd_dhue_deg']:.2f} deg")


if __name__ == "__main__":
    main()
