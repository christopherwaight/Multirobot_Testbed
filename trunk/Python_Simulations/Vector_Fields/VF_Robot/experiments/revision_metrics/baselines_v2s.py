"""
Section VI-B rebuilt around the vector-to-scalar (V2S) primitive (2026-11-04).
Review numbers only: nothing here is written into the paper .tex.

Variants (vecsim methods for attraction / orbit), same robot model (Coulomb), gains,
and formation as Section IV:
  jacobian    fit / fit                  the proposed estimate in Eq. 10 and Eq. 11
  v2s         magnitude / v2s            V2S as published: descend the fitted speed
                                         gradient; orbit tangent to the speed contour,
                                         no radial term
  v2s_oracle  oracle / v2s_oracle        V2S given the true critical point: Eq. 10 on
                                         the truth; contour tangent plus k_r (r - r_d)
  v2s_est     magnitude_p / magnitude_p  V2S with its own speed estimate
                                         p_m = p_c - S_c g / |g|^2 in Eq. 10 and Eq. 11
Fields: the six linear fields (reported merged), the anisotropic node diag(-1, -0.2),
and the gyre cell in the 1 m box of Section IV.

Attraction (500 trials of 10 s, seed 0, same starts for every variant): success is the
target inside the final triangle (and the cluster on the map); e_f, T_conv, E_u and the
final speed (mean centroid speed over the last 1 s) use the trials that succeeded;
overshoot is the furthest the centroid passed the target along its first approach.
In the gyre, each trial's reached critical point is compared across variants, and the
Jacobian's type label (last 2 s) is scored against the type of the point it reached.
Orbit (50 runs of 60 s per r_d, seed 0): a run is lost if it leaves the map or turns
less than a quarter revolution about p* after 10 s; RMSE and bias use the kept runs.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/baselines_v2s.py [--quick]
    venv/bin/python3 experiments/revision_metrics/baselines_v2s.py --table   # table from the saved summary
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as mt  # noqa: E402
import nonlinear as nl  # noqa: E402
import sim  # noqa: E402
import triangle_rescore as tr  # noqa: E402
import vecsim as vs  # noqa: E402

RES = os.path.join(HERE, "results")
VARIANTS = {"jacobian": ("fit", "fit"), "v2s": ("magnitude", "v2s"),
            "v2s_oracle": ("oracle", "v2s_oracle"), "v2s_est": ("magnitude_p", "magnitude_p")}
FIELDS = vs.LINEAR + ["node_aniso", "gyre"]
GROUPS = {"linear": vs.LINEAR, "node_aniso": ["node_aniso"], "gyre": ["gyre"]}
RDS = [0.10, 0.30, 0.50]


def field(name):
    F = vs.make_field(name)
    F.start_half = 0.5
    return F


def attraction(name, variant, n):
    F = field(name)
    method = VARIANTS[variant][0]
    out, _ = vs.simulate(F, "attract", method, vs.params(n), 10.0, seed=0, stiction_model=sim.MODEL, record=True)
    rec, rows = out["rec"], []
    steps = rec["c"].shape[1] - 1
    for i in range(n):
        c, pos = rec["c"][i], rec["pos"][i]
        esc = bool(out["escaped"][i])
        ph = rec["phat"][i] if method not in ("magnitude", "oracle") else np.full_like(rec["phat"][i], np.nan)
        r = mt.attraction_trial(c, ph, target=F.target, vcmd=rec["vc"][i], robot_vel_speed=rec["vel_speed"][i])
        r["tri_final"] = tr.attraction_row(pos, F.target)["tri_final"] and not esc
        j = int(np.argmin(np.linalg.norm(F.crit[:, :2] - c[-1], axis=1)))
        r.update(field=name, variant=variant, trial=i, escaped=esc, reached_row=j, reached_index=float(F.crit[j, 2]),
                 overshoot=float(out["overshoot"][i]),
                 final_speed=float(np.mean(np.hypot(*np.diff(c[-11:], axis=0).T)) / mt.DT))
        if method == "fit":
            truth = int(mt.classify(nl.jac(F, *F.crit[j, :2])))
            r["type_acc"] = mt.classification_scores(rec["J"][i], truth, start=int(0.8 * steps))["cls_acc"]
            r["det_sign_start"] = float(np.sign(nl.det4(rec["J"][i, 0])))
        rows.append(r)
    return rows


def orbit(name, variant, rd, n):
    F = field(name)
    method = VARIANTS[variant][1]
    out, _ = vs.simulate(F, "orbit", method, vs.params(n, rd=rd), 60.0, seed=0, stiction_model=sim.MODEL,
                         record=True)
    rec, rows = out["rec"], []
    for i in range(n):
        r, _ = mt.orbit_run(rec["c"][i], rec["phat"][i], rd, target=F.target, vcmd=rec["vc"][i],
                            robot_vel_speed=rec["vel_speed"][i])
        esc = bool(out["escaped"][i])
        r.update(field=name, variant=variant, trial=i, escaped=esc,
                 kept=bool(not esc and not r["left_ws"] and np.isfinite(r["revs_w1"]) and r["revs_w1"] >= 0.25))
        rows.append(r)
    return rows


def summarize(att, orb):
    S = {}
    for g, fields in GROUPS.items():
        for v in VARIANTS:
            a = att[att.field.isin(fields) & (att.variant == v)]
            ok = a[a.tri_final]
            o = orb[orb.field.isin(fields) & (orb.variant == v)]
            d = dict(n_att=len(a), success=100 * a.tri_final.mean(), e_mean=ok.e_final.mean(),
                     t_conv=ok.t_settle_5cm.median(), E_u=ok.E_real.mean(), final_speed=ok.final_speed.mean(),
                     overshoot_med=ok.overshoot.median(), overshoot_p95=ok.overshoot.quantile(0.95),
                     est_att=ok.est_ss.mean(), est_orb=o.est_err_w1.mean())
            if "type_acc" in a and a.type_acc.notna().any():
                d["type_acc"] = 100 * a.type_acc.mean()
                d["det_sign_match"] = 100 * float((np.sign(a.det_sign_start) == np.sign(a.reached_index)).mean())
            for rd in RDS:
                x = o[np.isclose(o.rd, rd)]
                k = x[x.kept]
                d[f"lost_{rd:.2f}"] = 100 * (1 - x.kept.mean())
                d[f"rmse_{rd:.2f}"] = k.rmse_w1.mean() if len(k) else None
                d[f"bias_{rd:.2f}"] = k.bias_w1.mean() if len(k) else None
                d[f"Eorb_{rd:.2f}"] = k.E_real_per_orbit.mean() if len(k) else None
            S.setdefault(g, {})[v] = d
    g = att[att.field == "gyre"]
    jac = g[g.variant == "jacobian"].set_index("trial").reached_row
    S["gyre_same_point_as_jacobian"] = {v: 100 * float((g[g.variant == v].set_index("trial").reached_row == jac).mean())
                                        for v in VARIANTS}
    return S


NAMES = {"jacobian": "Jacobian", "v2s": "V2S", "v2s_oracle": "V2S + oracle", "v2s_est": "V2S + speed est."}
GNAMES = {"linear": "Linear (6)", "node_aniso": "Aniso.\\ node", "gyre": "Gyre"}


def table(S):
    """Single-column review table: tables/table_v2s.tex."""
    def f(x, nd=3):
        return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    rows = []
    for g in GROUPS:
        for k, v in enumerate(VARIANTS):
            d = S[g][v]
            cells = [GNAMES[g] if k == 0 else "", NAMES[v], f"{d['success']:.0f}" if d["success"] < 99.5 or d["success"] == 100 else f"{d['success']:.1f}",
                     f(d["e_mean"]), f(d["t_conv"], 1), f(d["E_u"])]
            cells += ["--" if d[f"lost_{rd:.2f}"] > 50 else f(d[f"rmse_{rd:.2f}"]) for rd in RDS]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    head = (r"\begin{tabular}{ll|cccc|ccc}" "\n\\hline\n"
            r" & & \multicolumn{4}{c|}{\textbf{Attraction}} & \multicolumn{3}{c}{\textbf{Orbit RMSE (m)}} \\" "\n"
            r"\textbf{Field} & \textbf{Navigation} & \textbf{Succ.} & \textbf{$\bar{e}_f$} & \textbf{$T_{\mathrm{conv}}$} & \textbf{$E_u$} & \multicolumn{3}{c}{\textbf{at $r_d$ (m)}} \\" "\n"
            r" & & \textbf{(\%)} & \textbf{(m)} & \textbf{(s)} & \textbf{(m$^2$/s)} & 0.1 & 0.3 & 0.5 \\" "\n\\hline\n")
    body = head + "\n".join(rows) + "\n\\end{tabular}"
    with open(os.path.join(HERE, "tables", "table_v2s.tex"), "w") as fh:
        fh.write("% Generated by experiments/revision_metrics/baselines_v2s.py --table. Review copy.\n" + body + "\n")
    print(body)


def main():
    if "--table" in sys.argv:
        table(json.load(open(os.path.join(RES, "baselines_v2s_summary.json"))))
        return
    quick = "--quick" in sys.argv
    n_att, n_orb = (50, 10) if quick else (500, 50)
    att, orb = [], []
    for f in FIELDS:
        for v in VARIANTS:
            att += attraction(f, v, n_att)
            for rd in RDS:
                orb += orbit(f, v, rd, n_orb)
        print(f"  {f}", flush=True)
    att, orb = pd.DataFrame(att), pd.DataFrame(orb)
    tag = "_quick" if quick else ""
    att.to_csv(os.path.join(RES, f"baselines_v2s_attraction{tag}.csv"), index=False)
    orb.to_csv(os.path.join(RES, f"baselines_v2s_orbit{tag}.csv"), index=False)
    S = summarize(att, orb)
    with open(os.path.join(RES, f"baselines_v2s_summary{tag}.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
