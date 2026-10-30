"""
Matched baselines for the rewritten Section VI-B (Reviewer 1 Comment 5, Reviewer 3, AE).

Three estimators feed the same attraction law (Eq. 10) and orbit law (Eq. 11)
with the same gains, robot model and formation, so only the estimate differs:
  fit        the paper's plane fit, p_hat = -J_hat^{-1} h_hat
  oracle     the true critical point (separates estimation from control)
  magnitude  speed only: a plane fit of |v_i| gives slope g and centroid value m,
             and p_hat = p_c - m g / |g|^2 feeds Eq. 10 and Eq. 11
             (vecsim method "magnitude_p"; vecsim "magnitude" is the unmatched
             unit-gain descent of -g, run for attraction as a reference only)
Fields: the six linear fields, the two reconstructed (MLP) fields, and the two
nonlinear fields of the hyperparameter study (gyre, curved saddle).
Scored with metrics.py. Writes results/baselines_*.csv, results/baselines_summary.json
and tables/table_baselines_revised.tex (review copy).

Run from VF_Robot:  venv/bin/python3 experiments/revision_metrics/baselines.py [--quick]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as mt  # noqa: E402
import sim  # noqa: E402  (puts hyperparameter_study on the path)
import vecsim as vs  # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables")
METHODS = ["fit", "oracle", "magnitude_p"]
EXTRA_ATTRACT = ["magnitude"]          # unit-gain descent of the speed gradient, not matched
FIELDS = vs.LINEAR + vs.RECONSTRUCTED + ["gyre", "curved_saddle"]
GROUPS = {"Linear (6)": vs.LINEAR, "Vortex (reconstructed)": ["vortex_nn"],
          "Saddle (reconstructed)": ["saddle_nn"], "Gyre": ["gyre"], "Curved saddle": ["curved_saddle"]}
RDS = [0.10, 0.30, 0.50]


def attraction(field, method, n, T=10.0, seed=0):
    F = vs.make_field(field)
    out, _ = vs.simulate(F, "attract", method, vs.params(n), T, seed=seed, stiction_model=sim.MODEL, record=True)
    rec, rows = out["rec"], []
    for i in range(n):
        ph = rec["phat"][i] if method != "magnitude" else np.full_like(rec["phat"][i], np.nan)
        r = mt.attraction_trial(rec["c"][i], ph, target=F.target, vcmd=rec["vc"][i],
                                robot_vel_speed=rec["vel_speed"][i])
        r.update(field=field, method=method, trial=i, escaped=bool(out["escaped"][i]),
                 other_cp=bool(out["nearest"][i] > 0 and out["nearest_dist"][i] < 0.05))
        rows.append(r)
    return rows


def orbit(field, method, rd, n, T=60.0, seed=0):
    F = vs.make_field(field)
    out, _ = vs.simulate(F, "orbit", method, vs.params(n, rd=rd), T, seed=seed, stiction_model=sim.MODEL,
                         record=True)
    rec, rows = out["rec"], []
    for i in range(n):
        r, _ = mt.orbit_run(rec["c"][i], rec["phat"][i], rd, target=F.target, vcmd=rec["vc"][i],
                            robot_vel_speed=rec["vel_speed"][i])
        r.update(field=field, method=method, trial=i, escaped=bool(out["escaped"][i]))
        rows.append(r)
    return rows


def summarize(att, orb):
    S = {}
    for g, fields in GROUPS.items():
        for m in METHODS + EXTRA_ATTRACT:
            a = att[att.field.isin(fields) & (att.method == m)]
            o = orb[orb.field.isin(fields) & (orb.method == m)]
            d = dict(n_att=len(a), success_5cm=100 * (1 - a.fail_5cm.mean()), e_mean=a.e_final.mean(),
                     e_p95=a.e_final.quantile(0.95), t_settle_med=a.t_settle_5cm.median(),
                     E_real=a.E_real.mean(), other_cp=100 * a.other_cp.mean(), n_orb=len(o),
                     est_att=a.est_ss.mean(), est_orb=o.est_err_w1.mean() if len(o) else np.nan,
                     est_out_orb=o.n_est_outside.mean() if len(o) else np.nan)
            for rd in RDS:
                x = o[np.isclose(o.rd, rd)]
                d[f"rmse_{rd:.2f}"] = x.rmse_w1.mean()
                d[f"bias_{rd:.2f}"] = x.bias_w1.mean()
                d[f"lost_{rd:.2f}"] = 100 * ((x.revs_w1 < 0.25) | x.escaped | x.left_ws).mean()
            S.setdefault(g, {})[m] = d
    return S


def f(x, nd=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def table(S, n_att, n_orb):
    names = {"fit": "Plane fit", "oracle": "Oracle", "magnitude_p": "Magnitude"}
    groups = [g for g in GROUPS if g in ("Linear (6)", "Vortex (reconstructed)", "Saddle (reconstructed)")]
    rows = []
    for g in groups:
        for k, m in enumerate(METHODS):
            d = S[g][m]
            cells = [g if k == 0 else "", names[m], f"{d['success_5cm']:.0f}", f(d["e_mean"]),
                     f(d["t_settle_med"], 1), f(d["E_real"])] + [f(d[f"rmse_{rd:.2f}"]) for rd in RDS] + [f(d["est_orb"])]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    head = (r"\begin{tabular}{llcccc|ccc|c}" "\n\\hline\n"
            r" & & \multicolumn{4}{c|}{\textbf{Attraction}} & \multicolumn{3}{c|}{\textbf{Orbit RMSE (m)}} & \textbf{Est.} \\" "\n"
            r"\textbf{Field} & \textbf{Estimate} & \textbf{Succ.\ (\%)} & \textbf{$\mu_e$ (m)} & \textbf{$T_{\mathrm{conv}}$ (s)} & "
            r"\textbf{$E_u$} & " + " & ".join(f"$r_d$ = {rd:.1f}" for rd in RDS) + r" & \textbf{err.\ (m)} \\" "\n\\hline\n")
    return ("% Generated by experiments/revision_metrics/baselines.py. Review copy: not for the paper until approved.\n"
            "\\begin{table}[!t]\n\\centering\n\\footnotesize\n"
            f"\\caption{{Matched baselines in simulation: the same attraction law (\\ref{{eq:10}}) and orbital law "
            f"(\\ref{{eq:11}}), gains, robot model, and formation, driven by three estimates of the critical point. "
            f"{n_att} attraction trials of 10~s and {n_orb} orbit trials of 60~s per field and setting. Success is ending "
            f"within 0.05~m; $\\mu_e$ is the mean final error; $T_{{\\mathrm{{conv}}}}$ is the median time to stay within "
            f"0.05~m; $E_u$ (m$^2$/s) is the realized quadratic effort; orbit RMSE of $r^* - r_d$ excludes the first 10~s; "
            f"Est.\\ err.\\ is the mean $\\|\\hat{{\\mathbf{{p}}}}^* - \\mathbf{{p}}^*\\|$ during the orbits.}}\n"
            "\\label{tab:baselines}\n\\setlength{\\tabcolsep}{2.5pt}\n" + head + "\n".join(rows)
            + "\n\\end{tabular}\n\\end{table}\n")


def main():
    quick = "--quick" in sys.argv
    n_att, n_orb = (50, 10) if quick else (500, 50)
    att, orb = [], []
    for fld in FIELDS:
        for m in METHODS:
            att += attraction(fld, m, n_att)
            for rd in RDS:
                orb += orbit(fld, m, rd, n_orb)
        for m in EXTRA_ATTRACT:
            att += attraction(fld, m, n_att)
        print(f"  baselines {fld}", flush=True)
    att, orb = pd.DataFrame(att), pd.DataFrame(orb)
    tag = "_quick" if quick else ""
    os.makedirs(RES, exist_ok=True)
    att.to_csv(os.path.join(RES, f"baselines_attraction{tag}.csv"), index=False)
    orb.to_csv(os.path.join(RES, f"baselines_orbit{tag}.csv"), index=False)
    S = summarize(att, orb)
    with open(os.path.join(RES, f"baselines_summary{tag}.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"table_baselines_revised{tag}.tex"), "w") as fh:
        fh.write(table(S, n_att, n_orb))
    for g, ms in S.items():
        for m, d in ms.items():
            print(f"{g:24s} {m:11s} succ {d['success_5cm']:5.1f}%  e_mean {d['e_mean'] * 1e3:6.1f} mm  p95 {d['e_p95'] * 1e3:6.1f}  "
                  f"T {d['t_settle_med']:4.1f} s  E {d['E_real']:.3f}  other_cp {d['other_cp']:4.1f}%  est {d['est_orb'] * 1e3:5.0f} mm | orbit RMSE "
                  + "  ".join(f"{rd:.1f}: {d[f'rmse_{rd:.2f}'] * 1e3:5.0f} mm (bias {d[f'bias_{rd:.2f}'] * 1e3:+5.0f}, lost {d[f'lost_{rd:.2f}']:.0f}%)"
                              for rd in RDS))


if __name__ == "__main__":
    main()
