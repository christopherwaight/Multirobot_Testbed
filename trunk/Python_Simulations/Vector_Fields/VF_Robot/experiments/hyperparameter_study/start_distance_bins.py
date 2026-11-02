"""
Attraction outcome against start distance (Reviewer 3, "different starting distances").

Every attraction trial starts uniformly in a box about the target (half-width 0.5 m on the
linear and reconstructed fields, 0.45 m on the gyre), and each trial's start distance is stored
in results_v3/cells. This bins the E1hw (hardware-settings) trials by start distance and reports,
per field and estimator, the share that converged to the target (final error < 5 cm), the share
that reached a different critical point, and the median final error of the converged trials.

On the gyre the other critical points are saddles at (+-0.5, +-0.5) m, and det J = 0 on
|x| + |y| = 0.5, so the fall-off between 0.35 and 0.5 m is the det J = 0 boundary.

Run from VF_Robot: venv/bin/python3 experiments/hyperparameter_study/start_distance_bins.py
Writes results_v3/start_distance_bins.txt. Numbers await review before any use in the paper.
"""
import os

import numpy as np

import load

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "results_v3"
FIELDS = ("gyre", "curved_saddle", "vortex", "saddle_nn", "vortex_nn")
BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
CONV = 0.05
MIN_N = 200


def main():
    rows = [r for r in load.manifest(RES)
            if r["task"] == "attract" and r["exp"] == "E1hw" and r["field"] in FIELDS]
    lines = ["E1hw attraction by start distance. conv = final error < 5 cm; other = reached a different "
             "critical point; med = median final error of converged trials (mm).", ""]
    for r in sorted(rows, key=lambda r: (r["field"], r["method"])):
        M = load.trials(r)
        ok = (M["escaped"] == 0) & (M["max_side_err"] < 0.1)
        conv = ok & (M["final_err"] < CONV)
        other = ok & (M["nearest"] > 0) & (M["nearest_dist"] < CONV)
        sd = M["start_dist"]
        lines.append(f"{r['field']} / {r['method']}  (N = {len(sd)}, start {sd.min():.2f} to {sd.max():.2f} m)")
        for a, b in zip(BINS[:-1], BINS[1:]):
            m = (sd >= a) & (sd < b)
            if m.sum() < MIN_N:
                continue
            e = M["final_err"][m & conv]
            med = 1000 * np.median(e) if len(e) else np.nan
            lines.append(f"    [{a:.1f}, {b:.1f}) m  n = {m.sum():6d}  conv {100 * conv[m].mean():5.1f}%  "
                         f"other {100 * other[m].mean():5.1f}%  med {med:5.1f} mm")
    out = os.path.join(HERE, RES, "start_distance_bins.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
