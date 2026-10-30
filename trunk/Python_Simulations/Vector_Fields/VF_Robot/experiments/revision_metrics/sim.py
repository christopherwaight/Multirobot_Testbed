"""
Simulation runs for the revised Tables II and III, scored with metrics.py.

Uses the hyperparameter study's vectorized simulator (vecsim.py), which
reproduces OmniCluster with the 2026-10-28 heading fix. Robot model: Coulomb
friction (the OmniCluster default, chosen 2026-10-30), with vecsim's hardware
baseline otherwise: side 0.33 m, tau 0.28 s, v_c 0.025 m/s, 0.4 m/s robot cap,
k = 1 /s with a 0.5 m/s command cap, k_t 0.2 m/s, k_r 0.5 /s.

Fields: the six linear fields of Table I and the two reconstructed (MLP)
fields, as in the current Table II.
Attraction: 1000 trials per field, 10 s (100 steps), starts uniform in the
1 m x 1 m box about the critical point, random headings.
Orbit: 100 trials per field and r_d, 60 s (600 steps), starts on the circle
of radius r_d, random headings.
Classification (Reviewer 1, Comment 7): every trial is also scored on the type
label of its estimated Jacobian (metrics.classify) against the type of the
true Jacobian at the target. sigma_n adds white measurement noise to u and v.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(os.path.dirname(HERE), "hyperparameter_study")
sys.path.insert(0, STUDY)
import vecsim as vs  # noqa: E402

import metrics as mt  # noqa: E402

FIELDS = vs.LINEAR + vs.RECONSTRUCTED
PAPER_RDS = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50]
MODEL = "coulomb"


def _cls(J, truth, start, prefix=""):
    return {prefix + k: v for k, v in mt.classification_scores(J, truth, start=start).items()}


def run_attraction(field_name, n=1000, T=10.0, seed=0, sigma_n=0.0):
    F = vs.make_field(field_name)
    P = vs.params(n, sigma_n=sigma_n)
    truth = int(mt.classify(F.J0))
    out, _ = vs.simulate(F, "attract", "fit", P, T, seed=seed, stiction_model=MODEL, record=True)
    rec = out["rec"]
    rows = []
    for i in range(n):
        row = mt.attraction_trial(rec["c"][i], rec["phat"][i], target=F.target, vcmd=rec["vc"][i],
                                  robot_cmd_speed=rec["cmd_speed"][i], robot_vel_speed=rec["vel_speed"][i])
        row.update(field=field_name, trial=i, escaped=bool(out["escaped"][i]),
                   max_side_err=float(out["max_side_err"][i]), sigma_n=sigma_n, true_type=truth)
        steps = rec["J"].shape[1]
        row.update(_cls(rec["J"][i], truth, start=0, prefix="all_"))
        row.update(_cls(rec["J"][i], truth, start=int(0.8 * steps)))
        rows.append(row)
    return rows


def run_orbit(field_name, rd, n=100, T=60.0, seed=0, sigma_n=0.0):
    F = vs.make_field(field_name)
    P = vs.params(n, rd=rd, sigma_n=sigma_n)
    truth = int(mt.classify(F.J0))
    out, _ = vs.simulate(F, "orbit", "fit", P, T, seed=seed, stiction_model=MODEL, record=True)
    rec = out["rec"]
    rows, laps = [], []
    for i in range(n):
        row, lp = mt.orbit_run(rec["c"][i], rec["phat"][i], rd, target=F.target, vcmd=rec["vc"][i],
                               robot_vel_speed=rec["vel_speed"][i])
        row.update(field=field_name, trial=i, escaped=bool(out["escaped"][i]),
                   max_side_err=float(out["max_side_err"][i]), sigma_n=sigma_n, true_type=truth)
        row.update(_cls(rec["J"][i], truth, start=mt.TRANSIENT_STEPS))
        rows.append(row)
        for L in lp:
            L.update(field=field_name, trial=i, rd=rd)
        laps += lp
    return rows, laps
