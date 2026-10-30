"""
Executive summary of the revision metrics: prints to the terminal and writes
results/executive_summary.txt. Reads results/summary.json and the per-run CSVs.
Numbers here are for review; none goes into the paper .tex without approval.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    res = y - yhat
    return 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2), np.sqrt(np.mean(res ** 2)), np.mean(res)


def theory_fit(S, hw_orb):
    T = S["theory"]
    rd_t = np.round(T["rd"], 2)
    keys = [k for k in T if k != "rd"]
    lines = []
    sets = {"hardware vortex": hw_orb[hw_orb.series == "vortex"],
            "hardware saddle": hw_orb[hw_orb.series == "saddle"]}
    sim_lin = S["sim"]["orbit"]["vortex"]
    sets["sim linear fields"] = pd.DataFrame({"rd": [float(k) for k in sim_lin],
                                              "mean_r_w1": [v["mean_r_w1"] for v in sim_lin.values()]})
    for name, d in sets.items():
        d = d[d.rd <= 0.5 + 1e-9]
        for k in keys:
            pred = [T[k][int(np.argmin(np.abs(rd_t - round(r, 2))))] for r in d.rd]
            R2, rms, mean = r2(d.mean_r_w1, pred)
            lines.append(f"  {name:32s} vs {k:45s} R^2 {R2:6.3f}  RMS resid {rms * 1000:5.1f} mm  mean resid {mean * 1000:+6.1f} mm")
    return lines


def main(tag=""):
    S = json.load(open(os.path.join(RES, f"summary{tag}.json")))
    hw_orb = pd.read_csv(os.path.join(RES, "hw_orbit_per_run.csv"))
    hw_att = pd.read_csv(os.path.join(RES, "hw_attraction_per_trial.csv"))
    L = []
    p = L.append
    p("REVISION METRICS: EXECUTIVE SUMMARY (review only; nothing is in the paper .tex)")
    p(f"sim model {S['meta']['sim_model']}, hardware in OptiTrack metres, eps = {S['meta']['eps_conv']} m")
    p("")
    p("DATA CHECKS")
    for f, c in S["hw_checks"].items():
        mm = c["study_mismatch_m"]
        study = f"matches the study start/final file to {mm:.0e} m" if mm is not None else \
            "differs from the study start/final file (that file holds the four-robot vortex runs)"
        p(f"  {f}: {c['n']} runs, {study}, robots per log {c['n_robots']}, "
          f"logged centroid = mean of logged robots: {c['centroid_is_robot_mean']}")
    p("  Published Table V against these runs (its vortex column came from four-robot logs):")
    for s, m in S["hw_orbit_series_match"].items():
        p(f"    {s:10s} RMS difference to published errors {m['rms_diff_w2'] * 1000:5.1f} mm (full record), "
          f"{m['rms_diff_w1'] * 1000:5.1f} mm (10 s excluded)")
    p("")
    p("HARDWARE ATTRACTION")
    for f, s in S["hw_attraction"].items():
        p(f"  {f:7s} n={s['n']}: bias {s['bias'] * 1000:.1f} mm, scatter {s['scatter'] * 1000:.1f} mm, "
          f"mean Euclidean error {s['e_mean'] * 1000:.1f} +/- {s['e_sd'] * 1000:.1f} mm, median {s['e_med'] * 1000:.1f} "
          f"[IQR {s['e_q1'] * 1000:.1f}-{s['e_q3'] * 1000:.1f}], p95 {s['e_p95'] * 1000:.1f}, max {s['e_max'] * 1000:.1f} mm")
        p(f"          localization error (last 20%) {s['est_ss_mean'] * 1000:.1f} mm; success 3/4/5 cm "
          f"{s['success_3cm']:.1f}/{s['success_4cm']:.1f}/{s['success_5cm']:.1f} %; settle time (5 cm) median "
          f"{s['t_settle_med_5cm']:.1f} s, p95 {s['t_settle_p95_5cm']:.1f} s; path efficiency {s['path_eff_med']:.3f}")
    emax, N = hw_att.e_final.max(), len(hw_att)
    keep = "; ".join(f"eps = 0.0{t} m keeps {int(sum(s['n'] - s[f'n_fail_{t}cm'] for s in S['hw_attraction'].values()))}/{N}"
                     for t in (3, 4, 5))
    p(f"  Smallest eps keeping all {N} runs successful: {emax * 1000:.1f} mm "
      f"({int(hw_att.left_ws.sum())} runs left the workspace). {keep}.")
    stalled = hw_att[(hw_att.e_final > 0.05)]
    if len(stalled):
        p(f"  Runs ending beyond 0.05 m: {', '.join(f'{r.field} run{r.run}' for r in stalled.itertuples())}")
    p("")
    p("SIMULATION ATTRACTION (1000 trials per field, 10 s)")
    for f, s in S["sim"]["attraction"].items():
        p(f"  {f:15s} bias {s['bias'] * 1000:5.1f} mm, scatter {s['scatter'] * 1000:5.1f}, mean error "
          f"{s['e_mean'] * 1000:5.1f} +/- {s['e_sd'] * 1000:4.1f}, max {s['e_max'] * 1000:5.1f} mm, success 3/4/5 cm "
          f"{s['success_3cm']:.0f}/{s['success_4cm']:.0f}/{s['success_5cm']:.0f} %, settle (5 cm) {s['t_settle_med_5cm']:.1f} s")
    p("")
    p("SIMULATION ORBIT (100 trials per field and r_d, 10 s excluded): bias / RMSE / ctrl-only bias, mm")
    for f in ("vortex", "vortex_nn", "saddle_nn"):
        row = "  ".join(f"{float(rd):.2f}: {v['bias_w1'] * 1000:+.0f}/{v['rmse_w1'] * 1000:.0f}/{v['ctrl_bias_w1'] * 1000:+.0f}"
                        for rd, v in S["sim"]["orbit"][f].items())
        p(f"  {f:10s} {row}")
    p("")
    p("HARDWARE ORBIT (PROVISIONAL dry run on the old 60 s logs, 10 s excluded): bias / RMSE / ctrl, mm; laps")
    for s in ("vortex", "saddle"):
        d = hw_orb[hw_orb.series == s].sort_values("rd")
        row = "  ".join(f"{r.rd:.2f}: {r.bias_w1 * 1000:+.0f}/{r.rmse_w1 * 1000:.0f}/{r.ctrl_bias_w1 * 1000:+.0f} ({int(r.n_laps)})"
                        for r in d.itertuples())
        p(f"  {s:10s} {row}")
    few = hw_orb[hw_orb.n_laps < 2]
    p(f"  Runs with fewer than two complete laps after 10 s: {', '.join(few.name) or 'none'}")
    sp = hw_orb[hw_orb.n_est_outside > 0]
    p(f"  Estimates outside the map (excluded from Ctrl.): "
      f"{', '.join(f'{r.field} {r.name} {int(r.n_est_outside)}/{int(r.n_samples)}' for r in sp.itertuples()) or 'none'}")
    p(f"  Mean orbit speed after 10 s: vortex {hw_orb[hw_orb.series == 'vortex'].speed_w1.mean():.3f} m/s, "
      f"saddle {hw_orb[hw_orb.series == 'saddle'].speed_w1.mean():.3f} m/s")
    p("")
    C = S["classification"]
    p(f"CLASSIFICATION, simulation (band eps = {C['eps_center']}): % of cycles labelled correctly, "
      "attraction last 2 s / orbit after 10 s / coarse (attraction)")
    for f in C["sim"][next(iter(C["sim"]))]:
        cells = []
        for cond, F in C["sim"].items():
            t = F[f]
            cells.append(f"{cond}: {100 * t['attract']['acc']:5.1f} / {100 * t['orbit']['acc']:5.1f} / {100 * t['attract']['acc_coarse']:5.1f}")
        p(f"  {f:15s} ({F[f]['attract']['true_type']:15s}) " + "   ".join(cells))
    p("  Hardware classification needs the new campaign's logged Jhat; the old logs do not have it.")
    p("")
    p("THEORY AGAINST MEASURED MEAN RADIUS (r_d 0.01-0.50, 10 s excluded)")
    L += theory_fit(S, hw_orb)
    p("")
    p("ANOMALIES")
    p("  1. Only three-robot runs are used. The vortex runs are vortex_tests/3-robot-tests (80 attraction, 6 orbit);")
    p("     every top-level vortex_tests log is a four-robot formation. Published Tables IV and V took their vortex")
    p("     rows from the four-robot logs (79 runs and orbit0XX).")
    p("  2. Logged desired_velocities run ~3x the realized centroid speed (up to 2 m/s), so they are not the")
    p("     command the robots received; hardware effort is reported as realized effort only.")
    p("  3. The orbit logs are 600 cycles (60 s), not the 3000 cycles stated in Section V; the published Table V")
    p("     statistics include the first 10 s.")
    p("  4. Under the Coulomb model the linear-field rest point is the friction ring v_c/k = 25 mm, so the")
    p("     simulated scatter (24.9 mm) replaces the old Table II's ~15 mm. Reconstructed-vortex runs that stop")
    p("     on that ring around a biased estimate end up to 55 mm away, so eps = 0.05 m gives 95 % there.")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(RES, f"executive_summary{tag}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main("_quick" if "--quick" in sys.argv else "")
