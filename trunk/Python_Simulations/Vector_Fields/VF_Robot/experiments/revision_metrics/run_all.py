"""
Run every revision metric on the hardware logs and on simulation, then write
results/ (per-trial CSVs, summary.json), tables/ (standalone LaTeX for review)
and figures/, and print the executive summary.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/run_all.py              # everything
    venv/bin/python3 experiments/revision_metrics/run_all.py --quick      # 50 attraction / 10 orbit trials
    venv/bin/python3 experiments/revision_metrics/run_all.py --skip-sim   # hardware only, reuses sim CSVs

The hardware orbit runs are the old 60 s logs and are a dry run of the
pipeline (labelled provisional) until the orbit experiments are redone.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hardware as hw  # noqa: E402
import metrics as mt  # noqa: E402
import sim  # noqa: E402
import theory as th  # noqa: E402  (hyperparameter_study, on the path through sim)

RES = os.path.join(HERE, "results")
SIGMA_HW = 0.02          # white measurement noise per component, sized to the testbed's fixed calibration
                         # error (Section V-A); a stress level, far above the measured cycle-to-cycle noise
TRUE_TYPE = {"vortex": 6, "saddle": 1}   # printed maps: a center and a saddle
ALPHA = float(np.exp(-0.1 / 0.28))
VC = 0.025
VMAX = 0.4
GAINS = {"paper (k_t 0.2, k_r 0.5)": (0.2, 0.5),
         "Simulink 3R (k_t 0.15, k_r 0.2)": (0.15, 0.2)}
# Published Table V (current paper), signed radial error and std. Its vortex column came from the
# four-robot orbit0XX logs, so only the saddle column is expected to match the three-robot runs.
PUBLISHED_V = {"vortex": {0.01: (0.246, 0.053), 0.10: (0.190, 0.012), 0.20: (0.143, 0.019),
                          0.30: (0.087, 0.015), 0.40: (0.032, 0.020), 0.50: (-0.035, 0.023)},
               "saddle": {0.01: (0.114, 0.011), 0.10: (0.072, 0.018), 0.20: (0.058, 0.037),
                          0.30: (0.089, 0.059), 0.40: (0.090, 0.079), 0.50: (0.060, 0.095)}}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def hardware_attraction():
    rows, summ, checks = [], {}, {}
    for field in ("saddle", "vortex"):
        runs = hw.attraction_runs(field)
        checks[field] = dict(n=len(runs), study_mismatch_m=hw.check_against_study(field, runs),
                             n_robots=sorted({r["n_robots"] for r in runs}),
                             centroid_is_robot_mean=all(r["centroid_is_robot_mean"] for r in runs))
        frows = []
        for r in runs:
            row = mt.attraction_trial(r["c"], r["phat"], robots=r["robots"], vcmd=r["vdes"])
            row.update(field=field, run=r["run"], file=r["file"], n_robots=r["n_robots"])
            if r["J"] is not None:     # new-campaign logs carry Jhat; the old logs do not
                row.update(mt.classification_scores(r["J"], TRUE_TYPE[field], start=int(0.8 * len(r["J"]))))
            if r["phat_avg"] is not None:
                pe = np.hypot(*(r["phat_avg"]).T)
                row["est_avg_final"] = pe[-1]
                row["est_avg_ss"] = np.nanmean(pe[int(0.8 * len(pe)):])
            if r["vdes"] is not None:      # logged command against realized centroid speed
                v_real = np.hypot(*np.diff(r["c"], axis=0).T) / mt.DT
                v_log = np.hypot(*r["vdes"][:-1].T)
                moving = v_real > 0.05
                row["vdes_over_vreal_med"] = np.median(v_log[moving] / v_real[moving]) if moving.any() else np.nan
            frows.append(row)
        summ[field] = mt.summarize_attraction(frows)
        for k in ("est_avg_final", "est_avg_ss", "vdes_over_vreal_med"):
            if k in frows[0]:
                summ[field][k + "_mean"] = float(np.nanmean([f[k] for f in frows]))
        rows += frows
    return pd.DataFrame(rows), summ, checks


def hardware_orbit():
    rows, laps = [], []
    for r in hw.orbit_runs():
        row, lp = mt.orbit_run(r["c"], r["phat"], r["rd"], vcmd=r["vdes"], robots=r["robots"])
        row.update(field=r["field"], series=r["series"], name=r["name"], n_robots=r["n_robots"])
        if r["J"] is not None:
            row.update(mt.classification_scores(r["J"], TRUE_TYPE[r["field"]], start=mt.TRANSIENT_STEPS))
        pub = PUBLISHED_V[r["field"]].get(round(r["rd"], 2))
        row["published_err"], row["published_sd"] = pub if pub else (np.nan, np.nan)
        rows.append(row)
        for L in lp:
            L.update(series=r["series"], name=r["name"], rd=r["rd"])
        laps += lp
    df = pd.DataFrame(rows)
    match = {}
    for s in sorted(df.series.unique()):
        d = df[(df.series == s) & df.published_err.notna()]
        if len(d):
            match[s] = dict(n=len(d), rms_diff_w2=float(np.sqrt(np.mean((d.bias_w2 - d.published_err) ** 2))),
                            rms_diff_w1=float(np.sqrt(np.mean((d.bias_w1 - d.published_err) ** 2))))
    return df, pd.DataFrame(laps), match


def theory_curves(rds):
    out = {"task_formula_paper_gains": [float(mt.orbit_r_task_formula(rd)) for rd in rds]}
    for label, (kt, kr) in GAINS.items():
        out["friction_scaled " + label] = [float(th.orbit_r_exact(rd, ALPHA, mt.DT, kt, kr, VMAX, vc=VC))
                                           for rd in rds]
    return out


def sim_all(n_att, n_orb):
    att, orb, laps = [], [], []
    for f in sim.FIELDS:
        t0 = time.time()
        att += sim.run_attraction(f, n=n_att)
        for rd in sim.PAPER_RDS:
            o, lp = sim.run_orbit(f, rd, n=n_orb)
            orb += o
            laps += lp
        print(f"  sim {f:15s} {time.time() - t0:6.1f} s", flush=True)
    return pd.DataFrame(att), pd.DataFrame(orb), pd.DataFrame(laps)


def sim_noisy(n_att, n_orb):
    """Every field again with white measurement noise SIGMA_HW, for classification under noise."""
    att, orb = [], []
    for f in sim.FIELDS:
        att += sim.run_attraction(f, n=n_att, sigma_n=SIGMA_HW, seed=1)
        for rd in sim.PAPER_RDS:
            orb += sim.run_orbit(f, rd, n=n_orb, sigma_n=SIGMA_HW, seed=1)[0]
        print(f"  noisy sim {f}", flush=True)
    return pd.DataFrame(att), pd.DataFrame(orb)


def summarize_classification(conds):
    """conds: {label: (attraction df, orbit df)} -> per field and task classification statistics."""
    out = {}
    for label, (att, orb) in conds.items():
        for task, d in (("attract", att), ("orbit", orb)):
            for f, g in d.groupby("field", sort=False):
                out.setdefault(label, {}).setdefault(f, {})[task] = dict(
                    n=len(g), true_type=mt.TYPE_NAMES[int(g.true_type.iloc[0])],
                    acc=g.cls_acc.mean(), acc_coarse=g.cls_acc_coarse.mean(),
                    acc_all=g.all_cls_acc.mean() if "all_cls_acc" in g else None,
                    final_correct=(g.cls_final == g.true_type).mean(),
                    t_first_med=g.cls_t_first.median(), valid=g.cls_valid_frac.mean(),
                    final_labels={mt.TYPE_NAMES[int(k)]: int(v) for k, v in g.cls_final.value_counts().items()})
    return out


def summarize_sim(att, orb):
    S = {"attraction": {}, "orbit": {}}
    for f, d in att.groupby("field", sort=False):
        S["attraction"][f] = mt.summarize_attraction(d.to_dict("records"))
        S["attraction"][f]["formation_broken"] = int((d.max_side_err >= 0.1).sum())
    for (f, rd), d in orb.groupby(["field", "rd"], sort=False):
        S["orbit"].setdefault(f, {})[f"{rd:.2f}"] = dict(
            n=len(d), bias_w1=d.bias_w1.mean(), between_sd_w1=d.bias_w1.std(ddof=1),
            sd_within_w1=d.sd_within_w1.mean(), rmse_w1=d.rmse_w1.mean(), mae_w1=d.mae_w1.mean(),
            ctrl_bias_w1=d.ctrl_bias_w1.mean(), bias_w2=d.bias_w2.mean(), sd_within_w2=d.sd_within_w2.mean(),
            mean_r_w1=d.mean_r_w1.mean(), n_laps_med=d.n_laps.median(), lap_mean_r_sd=d.lap_mean_r_sd.mean(),
            lap_period_mean=d.lap_period_mean.mean(), speed_w1=d.speed_w1.mean(),
            E_real_per_orbit=d.E_real_per_orbit.mean(), left_ws=int(d.left_ws.sum()),
            formation_broken=int((d.max_side_err >= 0.1).sum()))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    tag = "_quick" if a.quick else ""

    print("hardware attraction ...", flush=True)
    hw_att, hw_att_sum, checks = hardware_attraction()
    hw_att.to_csv(os.path.join(RES, "hw_attraction_per_trial.csv"), index=False)
    print("hardware orbit (provisional dry run) ...", flush=True)
    hw_orb, hw_laps, match = hardware_orbit()
    hw_orb.to_csv(os.path.join(RES, "hw_orbit_per_run.csv"), index=False)
    hw_laps.to_csv(os.path.join(RES, "hw_orbit_per_lap.csv"), index=False)

    sim_paths = [os.path.join(RES, f"sim_{k}{tag}.csv") for k in ("attraction_per_trial", "orbit_per_trial",
                                                                  "orbit_per_lap")]
    noisy_paths = [os.path.join(RES, f"sim_noisy_{k}{tag}.csv") for k in ("attraction_per_trial", "orbit_per_trial")]
    if a.skip_sim:
        att, orb, slaps = (pd.read_csv(p) for p in sim_paths)
        natt, norb = (pd.read_csv(p) for p in noisy_paths)
    else:
        print("simulation (Coulomb model) ...", flush=True)
        att, orb, slaps = sim_all(50 if a.quick else 1000, 10 if a.quick else 100)
        for df, p in zip((att, orb, slaps), sim_paths):
            df.to_csv(p, index=False)
        print(f"simulation with sensor noise {SIGMA_HW} (classification) ...", flush=True)
        natt, norb = sim_noisy(50 if a.quick else 1000, 10 if a.quick else 100)
        for df, p in zip((natt, norb), noisy_paths):
            df.to_csv(p, index=False)

    rds = sorted(set(sim.PAPER_RDS) | set(hw_orb.rd.round(2)))
    summary = dict(
        meta=dict(sim_model=sim.MODEL, quick=a.quick, alpha=ALPHA, v_c=VC, v_max=VMAX,
                  eps_conv=list(mt.EPS_CONV), workspace_half_m=mt.HALF_WORKSPACE,
                  transient_steps=mt.TRANSIENT_STEPS, units="OptiTrack metres (hardware), metres (sim)"),
        hw_checks=checks, hw_attraction=hw_att_sum, hw_orbit_series_match=match,
        sim=summarize_sim(att, orb), theory=dict(rd=rds, **theory_curves(rds)),
        classification=dict(eps_center=mt.EPS_CENTER, sigma_noise=SIGMA_HW,
                            sim=summarize_classification({"noise-free": (att, orb),
                                                          f"sigma {SIGMA_HW}": (natt, norb)})))
    with open(os.path.join(RES, f"summary{tag}.json"), "w") as fh:
        json.dump(_clean(summary), fh, indent=1)
    print(f"wrote {RES}")


if __name__ == "__main__":
    main()
