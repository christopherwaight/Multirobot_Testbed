"""
Nonlinear fields under the Table II / III protocol, for merging them into the
same tables as the eight linear and reconstructed fields (Section IV restructure,
2026-11-04). Review numbers only: nothing here is written into the paper .tex.

Fields (vecsim): the gyre cell, and the curved saddle v = -y + chi y^3 at chi = 1
in its irrotational (vecsim family "potential") and rotational ("cellular") forms.
Robot model and gains are sim.py's (Coulomb, side 0.33 m, k = 1, k_t 0.2, k_r 0.5).

Attraction: 1000 trials, 10 s, seed 0, starts uniform in the 1 m x 1 m box and
random headings. The gyre's vecsim default box (+-0.45 m) is widened to +-0.5 m,
so every field uses the same draws as sim.py's linear fields. Each trial is scored
two ways.
    target  every metric of metrics.attraction_trial against the field's target
            (the origin), the triangle rule, and classification against its type.
    any     the critical point nearest the final centroid: distance to it, the
            3/4/5 cm rule, the triangle rule, and classification against its type.
Tags per trial: formation_in_region (the true det J at all three start robot positions
has the target's sign; for the gyre and rotational saddle, all three robots inside the
diamond |x| + |y| < h, which excludes robots past the diamond's tips in a neighboring
cell; the merged table's bracket), in_region (the true det J at the start centroid has the target's
sign: inside the |x| + |y| < 0.5 m diamond for the gyre, inside |x| + |y| < pi /
(2 sqrt 6) m for the rotational saddle, everywhere for the irrotational saddle),
the sign of det(J_hat) on the first cycle, and the start distance.
A noisy repeat (white measurement noise 0.02 m/s, seed 1) scores classification.

Orbit: 100 runs per r_d in sim.PAPER_RDS, 60 s, seed 0, as sim.run_orbit. A run is
kept if the cluster never escaped or left the map and made at least 0.25
revolutions about the target after the 10 s transient (the hyperparameter study's
ORBIT_FAIL_REVS rule). The triangle orbit rule is also scored.
Curvature sweep: the same orbit protocol at chi in {-1, -0.5, 0.5} for both forms
(chi = 1 is in the main run, chi = 0 is the linear saddle).
Formation-size sweep: the attraction protocol at sides 0.10, 0.20, 0.33 and 0.50 m,
localization error over the trials that succeeded.
Summary blocks per field: all trials, in_region / out_region, and success (the
trials whose final triangle holds the target), the subset the merged table's
accuracy rows use.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/nonlinear.py           # full
    venv/bin/python3 experiments/revision_metrics/nonlinear.py --quick   # 50 attraction / 10 orbit
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
import sim  # noqa: E402
import metrics as mt  # noqa: E402
import triangle_rescore as tr  # noqa: E402
import vecsim as vs  # noqa: E402  (on the path through sim)

RES = os.path.join(HERE, "results")
SIGMA_NOISE = 0.02
KEEP_REVS = 0.25
START_HALF = 0.5
MAIN = [("gyre", dict(name="gyre")),
        ("curved_saddle_irrot", dict(name="curved_saddle", eps=1.0, family="potential")),
        ("curved_saddle_rot", dict(name="curved_saddle", eps=1.0, family="cellular"))]
SIDES = (0.10, 0.20, 0.33, 0.50)
DIAMOND = {"gyre": 0.5, "curved_saddle_rot": np.pi / (2 * np.sqrt(6.0))}   # det J = 0 on |x| + |y| = h
SWEEP = [(f"curved_saddle_{lab}_chi{e:+g}", dict(name="curved_saddle", eps=e, family=fam), e, lab)
         for fam, lab in (("potential", "irrot"), ("cellular", "rot")) for e in (-1.0, -0.5, 0.5)]


def make(kw):
    F = vs.make_field(**kw)
    F.start_half = START_HALF
    return F


def jac(F, x, y, h=1e-5):
    """True Jacobian [du/dx, du/dy, dv/dx, dv/dy] by central differences."""
    up, vp = F.fn(np.array([x + h]), np.array([y]))
    um, vm = F.fn(np.array([x - h]), np.array([y]))
    uq, vq = F.fn(np.array([x]), np.array([y + h]))
    un, vn = F.fn(np.array([x]), np.array([y - h]))
    return np.array([(up - um)[0], (uq - un)[0], (vp - vm)[0], (vq - vn)[0]]) / (2 * h)


def det4(J):
    return J[..., 0] * J[..., 3] - J[..., 1] * J[..., 2]


def attraction(label, kw, n, sigma_n=0.0, seed=0):
    F = make(kw)
    truth = int(mt.classify(F.J0))
    P = vs.params(n, sigma_n=sigma_n)
    out, _ = vs.simulate(F, "attract", "fit", P, 10.0, seed=seed, stiction_model=sim.MODEL, record=True)
    rec = out["rec"]
    steps = rec["J"].shape[1]
    tsign = np.sign(det4(F.J0.reshape(4)))
    rows = []
    for i in range(n):
        c, ph, pos = rec["c"][i], rec["phat"][i], rec["pos"][i]
        esc = bool(out["escaped"][i])
        row = mt.attraction_trial(c, ph, target=F.target, vcmd=rec["vc"][i],
                                  robot_cmd_speed=rec["cmd_speed"][i], robot_vel_speed=rec["vel_speed"][i])
        row.update(field=label, trial=i, escaped=esc, max_side_err=float(out["max_side_err"][i]),
                   sigma_n=sigma_n, true_type=truth, start_x=c[0, 0], start_y=c[0, 1])
        row.update(sim._cls(rec["J"][i], truth, start=int(0.8 * steps)))
        tri = tr.attraction_row(pos, F.target)
        row.update(tri_final=tri["tri_final"] and not esc, tri_settle_s=tri["tri_settle_s"])
        row["in_region"] = bool(np.sign(det4(jac(F, *c[0]))) == tsign)
        h = DIAMOND.get(label)
        row["formation_in_region"] = bool((np.abs(pos[0]).sum(1) < h).all()) if h else \
            bool(all(np.sign(det4(jac(F, *v))) == tsign for v in pos[0]))
        row["det_hat_sign_start"] = float(np.sign(det4(rec["J"][i, 0])))
        row["cls_start"] = int(mt.classify(rec["J"][i, 0]))
        hit, idx, dist = tr.reached(F, c[-1])
        j = int(np.argmin(np.linalg.norm(F.crit[:, :2] - c[-1], axis=1)))
        cp = F.crit[j, :2]
        any_truth = int(mt.classify(jac(F, *cp)))
        row.update(reached_target=hit, reached_index=idx, reached_x=cp[0], reached_y=cp[1], any_true_type=any_truth)
        e = np.hypot(*(c[-1] - cp))
        row["any_e_final"] = e
        left = row["left_ws"] or esc
        for eps in mt.EPS_CONV:
            row[f"any_fail_{round(eps * 100):d}cm"] = bool(left or not (e <= eps))
        row["any_tri_final"] = tr.attraction_row(pos, cp)["tri_final"] and not esc
        row.update({"any_" + k: v for k, v in mt.classification_scores(rec["J"][i], any_truth,
                                                                         start=int(0.8 * steps)).items()})
        rows.append(row)
    return rows


def orbit(label, kw, rd, n, extra=None):
    F = make(kw)
    truth = int(mt.classify(F.J0))
    P = vs.params(n, rd=rd)
    out, _ = vs.simulate(F, "orbit", "fit", P, 60.0, seed=0, stiction_model=sim.MODEL, record=True)
    rec = out["rec"]
    rows = []
    for i in range(n):
        row, _ = mt.orbit_run(rec["c"][i], rec["phat"][i], rd, target=F.target, vcmd=rec["vc"][i],
                              robot_vel_speed=rec["vel_speed"][i])
        esc = bool(out["escaped"][i])
        row.update(field=label, trial=i, escaped=esc, max_side_err=float(out["max_side_err"][i]), true_type=truth)
        row.update(sim._cls(rec["J"][i], truth, start=mt.TRANSIENT_STEPS))
        t = tr.orbit_row(rec["pos"][i], rd, F.target)
        row.update(touch_frac=t["touch_frac"], touch_ge95=t["touch_ge95"] and not esc)
        row["kept"] = bool(not esc and not row["left_ws"] and np.isfinite(row["revs_w1"])
                           and row["revs_w1"] >= KEEP_REVS)
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def summarize(att, natt, orb, sweep, size):
    S = {"attraction": {}, "orbit": {}, "sweep": {}}
    for f, d in att.groupby("field", sort=False):
        nd = natt[natt.field == f]
        block = {}
        for sub, m, nm in (("all", np.ones(len(d), bool), np.ones(len(nd), bool)),
                           ("in_region", d.in_region.values, nd.in_region.values),
                           ("out_region", ~d.in_region.values, ~nd.in_region.values),
                           ("formation_in_region", d.formation_in_region.values, nd.formation_in_region.values),
                           ("straddle", d.in_region.values & ~d.formation_in_region.values,
                            nd.in_region.values & ~nd.formation_in_region.values),
                           ("success", d.tri_final.values, nd.tri_final.values)):
            g, gn = d[m], nd[nm]
            if len(g) < 2:
                block[sub] = dict(n=int(len(g)))
                continue
            s = mt.summarize_attraction(g.to_dict("records"))
            keep = ("n", "bias", "scatter", "e_mean", "e_sd", "e_med", "e_p95", "est_ss_mean", "t_settle_med_5cm",
                    "E_real_mean", "success_3cm", "success_4cm", "success_5cm", "left_ws")
            b = {k: s[k] for k in keep}
            b.update(tri_success=100 * g.tri_final.mean(), reached_target=100 * g.reached_target.mean(),
                     any_success_3cm=100 * (1 - g.any_fail_3cm.mean()), any_success_4cm=100 * (1 - g.any_fail_4cm.mean()),
                     any_success_5cm=100 * (1 - g.any_fail_5cm.mean()), any_tri_success=100 * g.any_tri_final.mean(),
                     cls_target=100 * g.cls_acc.mean(), cls_any=100 * g.any_cls_acc.mean(),
                     cls_target_noise=100 * gn.cls_acc.mean() if len(gn) else None,
                     cls_any_noise=100 * gn.any_cls_acc.mean() if len(gn) else None,
                     cls_target_coarse_noise=100 * gn.cls_acc_coarse.mean() if len(gn) else None)
            dsign = np.sign(np.where(g.reached_index > 0, 1, -1))
            b["det_hat_sign_predicts_type"] = 100 * float((g.det_hat_sign_start.values == dsign).mean())
            b["det_hat_sign_mismatch_n"] = int((g.det_hat_sign_start.values != dsign).sum())
            b["det_hat_wrong_sign_n"] = int((g.det_hat_sign_start.values != np.sign(det4(make(
                dict(MAIN)[f]).J0.reshape(4)))).sum()) if f in dict(MAIN) else None
            block[sub] = b
        bins = np.arange(0, 0.75, 0.1)
        sd = np.hypot(d.start_x, d.start_y)
        band = pd.cut(sd, bins)
        block["by_start_distance"] = {str(k): dict(n=int(len(v)), reached_target=100 * float(v.reached_target.mean()),
                                                   tri_success=100 * float(v.tri_final.mean()))
                                      for k, v in d.groupby(band, observed=True)}
        S["attraction"][f] = block
    for (f, rd), d in orb.groupby(["field", "rd"], sort=False):
        k = d[d.kept]
        S["orbit"].setdefault(f, {})[f"{rd:.2f}"] = dict(
            n=int(len(d)), kept_pct=100 * float(d.kept.mean()), touch95_pct=100 * float(d.touch_ge95.mean()),
            bias_kept=float(k.bias_w1.mean()) if len(k) else None,
            sd_within_kept=float(k.sd_within_w1.mean()) if len(k) else None,
            between_sd_kept=float(k.bias_w1.std(ddof=1)) if len(k) > 1 else None,
            ctrl_kept=float(k.ctrl_bias_w1.mean()) if len(k) else None,
            bias_all=float(d.bias_w1.mean()), cls=100 * float(d.cls_acc.mean()))
    for (f, rd), d in sweep.groupby(["field", "rd"], sort=False):
        k = d[d.kept]
        S["sweep"].setdefault(f, {})[f"{rd:.2f}"] = dict(
            chi=float(d.chi.iloc[0]), form=d.form.iloc[0], n=int(len(d)), kept_pct=100 * float(d.kept.mean()),
            bias_kept=float(k.bias_w1.mean()) if len(k) else None, ctrl_kept=float(k.ctrl_bias_w1.mean()) if len(k) else None)
    S["size"] = {}
    for (f, side), d in size.groupby(["field", "side"], sort=False):
        g = d[d.tri_final]
        S["size"].setdefault(f, {})[f"{side:.2f}"] = dict(
            n=int(len(d)), tri_success=100 * float(d.tri_final.mean()), tri_success_in_region=100 * float(
                d[d.in_region].tri_final.mean()), loc_med=float(g.est_ss.median()), loc_mean=float(g.est_ss.mean()),
            bias=float(np.hypot(g.final_x.mean(), g.final_y.mean())), e_mean=float(g.e_final.mean()))
    return S


def size_sweep(n):
    """Attraction at several formation sides (Reviewer 3: formation size against localization accuracy)."""
    rows = []
    for label, kw in MAIN:
        for side in SIDES:
            F = make(kw)
            P = vs.params(n, side=side)
            out, _ = vs.simulate(F, "attract", "fit", P, 10.0, seed=0, stiction_model=sim.MODEL, record=True)
            rec = out["rec"]
            tsign = np.sign(det4(F.J0.reshape(4)))
            for i in range(n):
                c = rec["c"][i]
                r = mt.attraction_trial(c, rec["phat"][i], target=F.target)
                esc = bool(out["escaped"][i])
                r.update(field=label, side=side, trial=i,
                         tri_final=tr.attraction_row(rec["pos"][i], F.target)["tri_final"] and not esc,
                         in_region=bool(np.sign(det4(jac(F, *c[0]))) == tsign))
                rows.append(r)
    return pd.DataFrame(rows)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_att, n_orb = (50, 10) if a.quick else (1000, 100)
    tag = "_quick" if a.quick else ""
    att, natt, orb, sw = [], [], [], []
    for label, kw in MAIN:
        t0 = time.time()
        att += attraction(label, kw, n_att)
        natt += attraction(label, kw, n_att, sigma_n=SIGMA_NOISE, seed=1)
        for rd in sim.PAPER_RDS:
            orb += orbit(label, kw, rd, n_orb)
        print(f"  {label:24s} {time.time() - t0:6.1f} s", flush=True)
    for label, kw, chi, form in SWEEP:
        t0 = time.time()
        for rd in sim.PAPER_RDS:
            sw += orbit(label, kw, rd, n_orb, extra=dict(chi=chi, form=form, rd=rd))
        print(f"  {label:24s} {time.time() - t0:6.1f} s", flush=True)
    t0 = time.time()
    sz = size_sweep(n_att)
    print(f"  {'size sweep':24s} {time.time() - t0:6.1f} s", flush=True)
    att, natt, orb, sw = (pd.DataFrame(x) for x in (att, natt, orb, sw))
    for df, name in ((att, "attraction"), (natt, "noisy_attraction"), (orb, "orbit"), (sw, "chi_sweep_orbit"),
                     (sz, "size_sweep_attraction")):
        df.to_csv(os.path.join(RES, f"nonlinear_{name}_per_trial{tag}.csv"), index=False)
    S = _clean(summarize(att, natt, orb, sw, sz))
    S["meta"] = dict(model=sim.MODEL, n_attraction=n_att, n_orbit=n_orb, start_half=START_HALF,
                     sigma_noise=SIGMA_NOISE, keep_revs=KEEP_REVS, side=0.33)
    with open(os.path.join(RES, f"nonlinear_summary{tag}.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
