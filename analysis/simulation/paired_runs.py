"""Re-simulate every hardware run from its own start (hardware-matched simulation), 3R and 4R.

Each run is simulated once, deterministically, from its first sample: the same centroid start
(relative to the session's p*), the formation the run held (held_shape: median over the run, in the
first sample's orientation; until 2026-11-17 the first sample's offsets, which were 1.5 to 4.5 cm larger), the same r_d
and duration, the 2026 controller and the 2026-11-16 robot model (rigidsim). Two fields:
  linear   the canonical vortex (-y, x) or saddle (-y, -x), in the Motive frame (orbit turn +1);
  recon    the 2026 reconstructed field, after mapping the start into its frame (common/frames.py;
           a mirrored frame flips the orbit sense to -1).
Scored with the same metrics as the hardware (metrics.py; 3 s orbit transient).
Run from the repo root after analysis/hardware/score_hardware.py. Outputs in analysis/simulation/paired/.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402
import rigidsim as rs  # noqa: E402
import vecsim as vs  # noqa: E402
import metrics as mt  # noqa: E402
import hw2025  # noqa: E402
import frames  # noqa: E402

OUT = os.path.join(paths.OUT_SIM, "paired")
HWOUT = os.path.join(paths.OUT_HW, "outputs")
RECON = {"vortex": "vortex_nn_2026", "saddle": "saddle_nn_2026"}


def held_shape(pos):
    """Formation the run held: robot offsets about the centroid at every sample, rotated onto the first
    sample's orientation, then the median over the run, re-centred (pos is samples x robots x 2)."""
    Q = pos - pos.mean(1, keepdims=True)
    Q = Q[np.isfinite(Q).all((1, 2))]
    ref = Q[0]
    phi = np.arctan2((Q[..., 0] * ref[:, 1] - Q[..., 1] * ref[:, 0]).sum(1), (Q * ref).sum((1, 2)))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    R = np.stack([c * Q[..., 0] - s * Q[..., 1], s * Q[..., 0] + c * Q[..., 1]], -1)
    M = np.median(R, 0)
    return M - M.mean(0)          # the median shape can sit up to 1 mm off its centroid


def simulate_batch(field_name, task, starts, offsets, T, rd=0.3, turn=1.0):
    n = len(starts)
    P = vs.params(n, rd=rd, turn=turn)
    return rs.simulate(vs.make_field(field_name), task, "fit", P, T, offsets, starts)


def orbits():
    runs = pickle.load(open(os.path.join(HWOUT, "runs2026.pkl"), "rb"))
    fits = json.load(open(os.path.join(HWOUT, "centres_2026.json")))
    rows = []
    groups = {}
    for r in runs:
        groups.setdefault((r["field"], r["nr"], r["rd"]), []).append(r)
    for (field, nr, rd), g in sorted(groups.items()):
        ps = np.array(fits[f"{field}_pooled"]["p_star"])
        start = np.array([r["c"][0] - ps for r in g])
        off = np.array([held_shape(r["pos"]) for r in g])
        T = max(r["t"][-1] for r in g)
        fr = frames.frame_2026(field, ps)
        res = {"linear": simulate_batch(field, "orbit", start, off, T, rd=rd, turn=1.0),
               "recon": simulate_batch(RECON[field], "orbit", fr(start + ps - ps), off @ fr.A.T, T, rd=rd,
                                       turn=-1.0 if fr.mirror else 1.0)}
        for i, r in enumerate(g):
            m = len(r["t"])
            row = dict(field=field, nr=nr, rd=rd, run=r["run"], T=float(r["t"][-1]))
            for tag, (C, PH) in res.items():
                o, _ = mt.orbit_run(C[i, :m], PH[i, :m], rd, target=(0.0, 0.0))
                for k in ("bias_w1", "mean_r_w1", "sd_within_w1", "rmse_w1", "ctrl_bias_w1", "est_err_w1", "speed_w1", "revs_w1"):
                    row[f"{tag}_{k}"] = o[k]
                row[f"{tag}_lost"] = bool(o["left_ws"] or not (o["revs_w1"] >= 0.25))
            rows.append(row)
    return pd.DataFrame(rows)


def attraction():
    rows = []
    for field, nr in (("vortex", 3), ("saddle", 3), ("vortex", 4)):
        runs, ps, _ = hw2025.load(field, nr)
        Pr, H, _, _ = hw2025.session_readings((field, nr), runs)
        fr, reg = frames.register(field, Pr - ps, H)
        keys = sorted(runs)
        start = np.array([runs[n]["pos"][0].mean(0) - ps for n in keys])
        off = np.array([held_shape(runs[n]["pos"]) for n in keys])
        T = 10.0
        res = {"linear": simulate_batch(field, "attract", start, off, T),
               "recon": simulate_batch(RECON[field], "attract", fr(start), off @ fr.A.T, T)}
        for i, n in enumerate(keys):
            row = dict(field=field, nr=nr, run=n, group=(n - 1) // 10 + 1, source=runs[n]["source"],
                       reg_rotation=reg["rotation_deg"], reg_mirror=reg["mirror"])
            for tag, (C, PH) in res.items():
                a = mt.attraction_trial(C[i], PH[i, :-1], target=(0.0, 0.0))
                for k in ("e_final", "est_ss", "t_settle_5cm", "E_real", "L_path"):
                    row[f"{tag}_{k}"] = a[k]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    O = orbits()
    O.to_csv(os.path.join(OUT, "paired_orbits.csv"), index=False)
    A = attraction()
    A.to_csv(os.path.join(OUT, "paired_attraction.csv"), index=False)
    HO = pd.read_csv(os.path.join(HWOUT, "orbit_per_run.csv"))
    HA = pd.read_csv(os.path.join(HWOUT, "attraction_per_trial.csv"))
    Om = O.merge(HO[["field", "nr", "rd", "run", "bias_w1", "mean_r_w1", "sd_within_w1", "rmse_w1"]].rename(
        columns=lambda c: c if c in ("field", "nr", "rd", "run") else f"hw_{c}"), on=["field", "nr", "rd", "run"])
    S = Om.groupby(["field", "nr", "rd"]).agg(hw_bias=("hw_bias_w1", "mean"), lin_bias=("linear_bias_w1", "mean"),
                                              rec_bias=("recon_bias_w1", "mean"), hw_sdw=("hw_sd_within_w1", "mean"),
                                              lin_sdw=("linear_sd_within_w1", "mean"), rec_sdw=("recon_sd_within_w1", "mean"),
                                              rec_lost=("recon_lost", "sum")).reset_index()
    S.to_csv(os.path.join(OUT, "paired_orbits_summary.csv"), index=False)
    Am = A.merge(HA[["field", "nr", "run", "e_final", "est_ss", "t_settle_5cm"]].rename(
        columns=lambda c: c if c in ("field", "nr", "run") else f"hw_{c}"), on=["field", "nr", "run"])
    SA = Am.groupby(["field", "nr"]).agg(hw_e=("hw_e_final", "mean"), lin_e=("linear_e_final", "mean"), rec_e=("recon_e_final", "mean"),
                                         hw_loc=("hw_est_ss", "mean"), rec_loc=("recon_est_ss", "mean"),
                                         hw_tconv=("hw_t_settle_5cm", "median"), lin_tconv=("linear_t_settle_5cm", "median"),
                                         rec_tconv=("recon_t_settle_5cm", "median")).reset_index()
    SA.to_csv(os.path.join(OUT, "paired_attraction_summary.csv"), index=False)
    print(S.round(3).to_string()); print(); print(SA.round(4).to_string())


if __name__ == "__main__":
    main()
