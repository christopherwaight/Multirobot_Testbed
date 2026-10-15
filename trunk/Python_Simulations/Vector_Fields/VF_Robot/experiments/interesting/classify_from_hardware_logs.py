"""Offline critical-point classification from the 3-robot hardware logs.

Recomputes the affine-fit Jacobian every logged cycle from the three robots'
mocap positions and sensed hue/saturation, classifies it by eigenvalue
structure (critical-points paper, Table I), and scores it against the true
type of the printed field (vortex: center, saddle: saddle).

Data: trunk/robots_3/{saddle_tests, vortex_tests/3-robot-tests}/. Sensed
readings are MATLAB timeseries objects inside run*.mat, which scipy cannot
decode, so they are taken from all_vortex_data_possible3.csv (written by
all_data_combiner.m) and matched back to each run by its pose rows.

Decode (checked against the logs): magnitude = (sat - 0.3)/0.7 reproduces the
controller's logged detected_critical_point to ~1e-6 m. Direction follows
the per-map registration in SETS: angle = 2*pi*hue, with v negated for the
saddle map only. The critical point location is invariant to that
registration; the classification is not, so results are also reported for
the raw, unregistered fit the controller computed.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../robots_3"))
OUT = os.path.join(os.path.dirname(__file__), "outputs", "classify_hw")
# (folder, csv, true type, map-to-mocap y-reflection of the sensed vector).
# The reflection per map follows the author's reconstruction scripts:
# saddle_tests/interpreted_field_saddle.m negates v only (map y-axis opposite
# to mocap); vortex_tests/Copy_of_interpreted_field2.m applies a consistent
# reflection to positions and vectors, which needs no registration.
SETS = {
    "saddle": ("saddle_tests", "all_vortex_data_possible3.csv", "saddle", True),
    "vortex": ("vortex_tests/3-robot-tests", "all_vortex_data_possible3.csv", "center", False),
}
CENTER_TOLS = [0.1, 0.2, 0.3]  # |alpha|/|omega| below which a complex pair is a center
TERMINAL_SAMPLES = 20          # last 2 s at 10 Hz


def load_trials(folder, csv):
    C = pd.read_csv(os.path.join(ROOT, folder, csv)).values
    trials, pos = [], 0
    for i in range(1, 81):
        f = os.path.join(ROOT, folder, f"run{i}.mat")
        if not os.path.exists(f):
            continue
        d = sio.loadmat(f)
        r1 = d["robot1_pose"][:, :2]
        hits = np.where(np.all(np.abs(C[pos:, 0:2] - r1[0]) < 1e-6, axis=1))[0]
        s, n = pos + hits[0], len(r1)
        block = C[s:s + n]
        assert np.allclose(block[:, 0:2], r1, atol=1e-6)
        trials.append((i, block, d["detected_critical_point"]))
        pos = s + n
    return trials


def fit(block, k):
    P = [block[k, [0, 1]], block[k, [5, 6]], block[k, [10, 11]]]
    H = [block[k, 3], block[k, 8], block[k, 13]]
    S = [block[k, 4], block[k, 9], block[k, 14]]
    mag = [(s - 0.3) / 0.7 for s in S]
    u = [m * np.cos(2 * np.pi * h) for m, h in zip(mag, H)]
    v = [m * np.sin(2 * np.pi * h) for m, h in zip(mag, H)]
    A = np.array([[p[0], p[1], 1.0] for p in P])
    cu, cv = np.linalg.solve(A, u), np.linalg.solve(A, v)
    J = np.array([[cu[0], cu[1]], [cv[0], cv[1]]])
    c = np.array([cu[2], cv[2]])
    p_star = np.linalg.solve(J, -c)
    J_flip = np.diag([1.0, -1.0]) @ J  # v negated
    return J_flip, J, p_star, np.mean(P, axis=0)


def classify(J, tol):
    tr, det = np.trace(J), np.linalg.det(J)
    if det < 0:
        return "saddle"
    disc = tr ** 2 - 4 * det
    if disc < 0:
        alpha, omega = tr / 2, np.sqrt(-disc) / 2
        if abs(alpha) < tol * omega:
            return "center"
        return "stable spiral" if alpha < 0 else "unstable spiral"
    return "stable node" if tr < 0 else "unstable node"


def coarse(t):
    return {"saddle": "saddle", "center": "rotational", "stable spiral": "rotational",
            "unstable spiral": "rotational"}.get(t, "node")


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, rows = {}, []
    for name, (folder, csv, truth, flip) in SETS.items():
        trials = load_trials(os.path.join(folder), csv)
        cp_check, per = [], {}
        for i, block, det_logged in trials:
            for k in range(len(block)):
                Jf, J, ps, pc = fit(block, k)
                Jraw = J
                if not flip:
                    Jf = J
                cp_check.append(np.linalg.norm(ps - det_logged[k]))
                rec = {"field": name, "run": i, "k": k, "dist": float(np.linalg.norm(pc)),
                       "tr": float(np.trace(Jf)), "det": float(np.linalg.det(Jf))}
                for tol in CENTER_TOLS:
                    rec[f"type_{tol}"] = classify(Jf, tol)
                rec["type_raw"] = classify(Jraw, 0.2)
                rows.append(rec)
        df = pd.DataFrame([r for r in rows if r["field"] == name])
        res = {"trials": len(trials), "cycles": len(df),
               "cp_reproduction_median_m": float(np.nanmedian(cp_check)),
               "true_type": truth}
        coarse_truth = coarse(truth)
        df["coarse"] = df[f"type_{CENTER_TOLS[0]}"].map(coarse)
        term = df.groupby("run").tail(TERMINAL_SAMPLES)
        res["coarse_cycle_acc"] = float((df["coarse"] == coarse_truth).mean())
        res["coarse_terminal_acc"] = float((term["coarse"] == coarse_truth).mean())
        res["coarse_trial_majority_acc"] = float(
            term.groupby("run")["coarse"].agg(lambda s: s.mode()[0] == coarse_truth).mean())
        for tol in CENTER_TOLS:
            col = f"type_{tol}"
            res[f"strict_tol{tol}_cycle_acc"] = float((df[col] == truth).mean())
            res[f"strict_tol{tol}_terminal_acc"] = float((term[col] == truth).mean())
            res[f"strict_tol{tol}_trial_majority_acc"] = float(
                term.groupby("run")[col].agg(lambda s: s.mode()[0] == truth).mean())
        res["raw_unregistered_terminal_confusion"] = term["type_raw"].value_counts().to_dict()
        res["confusion_terminal_tol0.2"] = term["type_0.2"].value_counts().to_dict()
        res["confusion_all_tol0.2"] = df["type_0.2"].value_counts().to_dict()
        bins = [0, 0.1, 0.2, 0.4, 0.8, 5]
        df["dbin"] = pd.cut(df["dist"], bins)
        res["coarse_acc_by_distance_m"] = {
            str(b): [float((g["coarse"] == coarse_truth).mean()), int(len(g))]
            for b, g in df.groupby("dbin", observed=True)}
        summary[name] = res
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_cycle.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
