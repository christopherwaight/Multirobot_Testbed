"""
Numbers quoted in hyperparameter_study.md (version 3) that analyze.py does not
print. Run after analyze.py, from VF_Robot:

    venv/bin/python3 experiments/hyperparameter_study/report_numbers.py [--res results_v3]

Writes <res>/report_numbers.txt.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vecsim as vs
import theory as th
import load
import analyze as an

MAP_SCALE = 1.6764


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_v3")
    a = ap.parse_args()
    res = a.res
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    S = an.add_orbit_predictions(an.build_summary(res))
    rows = {r["id"]: r for r in load.manifest(res)}
    man = load.manifest(res)

    # ---- trial counts
    cnt = {}
    for r in man:
        e = "E10" if r["exp"].startswith("E10") else ("E1" if r["exp"].startswith("E1") else r["exp"][:2])
        n = r["N"] * (157 if r["hw"] else 1)
        cnt[e] = cnt.get(e, 0) + n
    log("trials by experiment: " + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in sorted(cnt.items())) +
        f"; total {sum(cnt.values()) / 1e6:.2f}M in {len(man)} cells")

    # ---- paper settings
    al = float(np.exp(-0.1 / 0.28))
    T = th.t_eff(al, 0.1)
    log(f"paper settings: alpha {al:.4f}, T_eff {T:.4f} s, k_c {float(th.k_crit(al, 0.1)):.3f} /s, "
        f"k_max {float(th.k_max(al, 0.1)):.1f} /s, 4 k T_eff at k=1 {4 * T:.3f}, "
        f"overshoot ratio at k=1 {float(th.overshoot_ratio_discrete(1.0, al, 0.1)) * 100:.4f}%")

    # ---- hardware stop side
    for f in ("vortex", "saddle"):
        h = np.load(os.path.join(HERE, f"hardware_{f}_starts_finals.npy"))
        st, fi = h[:, :2] * MAP_SCALE, h[:, 2:4] * MAP_SCALE
        u0 = -st / np.linalg.norm(st, axis=1, keepdims=True)
        along = (fi * u0).sum(1)
        log(f"hardware {f}: {np.mean(along < 0) * 100:.0f}% stop on the approach side, mean along-approach "
            f"{along.mean() * 1000:+.1f} mm, mean rest point ({fi[:, 0].mean() * 1000:+.1f}, {fi[:, 1].mean() * 1000:+.1f}) mm")
    for f in ("vortex", "saddle"):
        d = an.sel(S, exp="E10hw", stiction="coulomb", field=f)
        M = load.trials(rows[d["id"].values[0]])
        log(f"E10hw coulomb {f}: {np.mean(M['final_proj'] < 0) * 100:.1f}% short on approach side, "
            f"mean along {M['final_proj'].mean() * 1000:+.1f} mm")

    # ---- E1hw rest at paper settings
    lin = S[(S["exp"] == "E1hw") & (S["task"] == "attract") & (S["method"] == "fit") & S["field"].isin(vs.LINEAR)]
    log(f"E1hw linear fit: conv {lin['conv'].min() * 100:.1f}-{lin['conv'].max() * 100:.1f}%, mean err "
        f"{lin['err_mean'].min() * 1000:.2f}-{lin['err_mean'].max() * 1000:.2f} mm")

    # ---- start-up dip, E8o caps off, tau 1, k_r 0.25, r0 0.6, dt 0.1
    for caps in ("off", "hw"):
        d = S[(S["exp"] == "E8o" + caps) & (np.abs(S["alpha"] - np.exp(-0.1)) < 1e-9) & (S["kr"] == 0.25) &
              (S["r0"] == 0.6) & (S["dt"] == 0.1)]
        if len(d):
            M = load.trials(rows[d["id"].values[0]])
            log(f"E8o{caps} tau 1 k_r 0.25 r0 0.6: mean r {np.nanmedian(M['mean_r']):.3f}, min r after start "
                f"{np.nanmedian(M['r_tmin']):.3f}, dip {(np.nanmedian(M['mean_r']) - np.nanmedian(M['r_tmin'])) * 1000:.0f} mm, "
                f"pred r* {float(th.orbit_r_exact(0.3, np.exp(-0.1), 0.1, 0.2, 0.25, 1e9 if caps == 'off' else 0.4, vc=0.025)):.3f}")

    # ---- E9a ranges
    E9 = an.sel(S, exp="E9a", alpha=0.7)
    for k in (1.0, 30.0, 300.0):
        d = E9[E9["k"] == k]
        meds = []
        for r in d.sort_values("vmax").itertuples():
            M = load.trials(rows[r.id])
            meds.append((r.vmax, np.median(np.where(M["escaped"] > 0, np.inf, M["final_err"])) * 1000))
        log(f"E9a alpha 0.7 k {k:g}: median final error by v_max " + ", ".join(f"{v:g}: {m:.2f} mm" for v, m in meds))

    # ---- orbit closed form, validity region
    o = S[(S["task"] == "orbit") & S["exact_est"] & (S["sigma_n"] == 0) & (S["ok"] > 0.5) & np.isfinite(S["pred_r"])].copy()
    V = th.orbit_speed(o["kt"], o["vmax"], o["vc"])
    o["phase"] = V / o["pred_r_small"] * th.t_eff(o["alpha"], o["dt"])
    o["valid"] = (o["phase"] <= 0.5) & (o["kr"] * (o["pred_r_small"] - o["rd"]) <= o["rclamp"])
    v = o[o["valid"]]
    for col, lab in (("pred_r_small", "small-angle form (quoted)"), ("pred_r", "exact-phase form")):
        gap = (v["mean_r"] - v[col]).abs()
        w = v.loc[gap.idxmax()]
        rel = gap / (v[col] - v["rd"]).abs().clip(lower=1e-6)
        log(f"orbit closed form, {lab}, designed cells in the validity region: n={len(v)} of {len(o)}, r_d "
            f"{v['rd'].min():g} to {v['rd'].max():g} m, median |gap| {gap.median() * 1000:.2f} mm, 95% "
            f"{gap.quantile(.95) * 1000:.2f} mm, max {gap.max() * 1000:.2f} mm at {w['exp']} k_t {w['kt']:g} r_d "
            f"{w['rd']:g} alpha {w['alpha']:.3g} dt {w['dt']:g} v_c {w['vc']:g} (r* {w[col]:.3f}); relative median "
            f"{rel.median() * 100:.2f}%, 95% {rel.quantile(.95) * 100:.1f}%, max {rel.max() * 100:.1f}%")
    outside = o[~o["valid"]]
    log(f"   outside validity: n={len(outside)}, sim larger than formula in "
        f"{(outside['mean_r'] > outside['pred_r']).mean() * 100:.0f}%")
    d = an.sel(S, exp="E1hw", task="orbit", method="fit", field="vortex").sort_values("rd")
    log("E1hw vortex orbit error (sim) " + ", ".join(f"{r:g}: {e:+.3f}" for r, e in zip(d["rd"], d["err_mean"])) +
        " | closed form " + ", ".join(f"{p - r:+.3f}" for r, p in zip(d["rd"], d["pred_r"])))
    d = an.sel(S, exp="E2dt", method="fit", field="vortex", rd=0.3).sort_values("dt")
    log("E2dt r_d 0.3 error by dt " + ", ".join(f"{a:g}: {e:.3f}" for a, e in zip(d["dt"], d["err_mean"])))

    # ---- table 3: smallest orbit
    def small(alpha, kt, kr, vc, vmax=0.4):
        C = float(th.orbit_C(alpha, 0.1, kt, kr, vmax, vc))
        rd = [C / e - e for e in (0.05, 0.01)]
        return C, np.sqrt(C), ["any" if x <= 0 else f"{x:.2f}" for x in rd]
    for lab, args in (("hardware (alpha 0.70, k_t 0.2, k_r 0.5, v_c 0.025)", (al, 0.2, 0.5, 0.025)),
                      ("same, v2 model (no friction, cap 0.3)", (0.7, 0.2, 0.5, 0.0, 0.3)),
                      ("lag off (alpha 0)", (0.0, 0.2, 0.5, 0.025)),
                      ("k_t 0.1", (al, 0.1, 0.5, 0.025)), ("k_r 2", (al, 0.2, 2.0, 0.025)),
                      ("k_t 0.1 and k_r 2", (al, 0.1, 2.0, 0.025)),
                      ("tau 0.20 (hardware transient fit)", (np.exp(-0.5), 0.2, 0.5, 0.025))):
        C, rmin, rd = small(*args)
        log(f"table 3 {lab}: C {C:.4f} m^2, r_min {rmin:.3f} m, r_d for error <= 5 cm {rd[0]}, <= 1 cm {rd[1]}")

    # ---- E3k gain
    D = an.sel(S, exp="E3k", field="vortex", method="fit", alpha=0.7)
    for r in D.sort_values("k").itertuples():
        M = load.trials(rows[r.id])
        log(f"E3k vortex alpha 0.7 k {r.k:g}: conv {r.conv * 100:.1f}%, mean err (conv) {r.err_mean * 1000:.2f} mm, "
            f"median all {np.median(M['final_err']) * 1000:.2f} mm, overshoot p95 {r.over_p95 * 100:.2f} cm")

    # ---- E4, E5 size and curvature
    for f, fam in (("gyre", "potential"), ("vortex_nn", "potential"), ("saddle_nn", "potential")):
        d = an.sel(S, exp="E4ideal", task="attract", method="fit", field=f, family=fam).sort_values("side")
        log(f"E4ideal {f} attract median err by side " + ", ".join(f"{s:g}: {e * 1000:.2f}" for s, e in zip(d["side"], d["err_med"])))
    d = an.sel(S, exp="E4ideal", task="orbit", method="fit", field="gyre", rd=0.3).sort_values("side")
    log("E4ideal gyre orbit r_d 0.3 err by side " + ", ".join(f"{s:g}: {e:+.3f}" for s, e in zip(d["side"], d["err_mean"])))
    for fam in ("potential", "cellular"):
        d = an.sel(S, exp="E5ideal", field="curved_saddle", family=fam, side=0.1, rd=0.3).sort_values("eps")
        log(f"E5ideal {fam} r_d 0.3 side 0.1: " + ", ".join(
            f"eps {e:+g}: err {m:+.3f} a2 {w_:.3f} ok {o_:.2f}" for e, m, w_, o_ in zip(d["eps"], d["err_mean"], d["a2"], d["ok"])))
    r0 = float(th.orbit_r_exact(0.3, 0.0, 0.1, 0.2, 0.5, 1e9))
    Om = 0.2 / r0
    log(f"E5 wobble prediction at eps 1, r {r0:.3f}: {2 * r0 ** 3 * np.sqrt(0.25 + Om ** 2) / np.sqrt(0.25 + 4 * Om ** 2):.3f} m")

    # ---- E6 noise
    lam = an.lam
    rat = []
    for r in S[(S["exp"] == "E6ideal") & (S["task"] == "attract") & S["field"].isin(vs.LINEAR) & (S["sigma_n"] > 0)].itertuples():
        L = lam(r.field)
        if L * (r.side / np.sqrt(3)) / r.sigma_n >= 3 and r.conv > 0.5:
            rat.append(r.err_mean / (th.noise_rms(r.sigma_n, L, 1.0, 0.0, 0.1) * np.sqrt(np.pi) / 2))
    rat = np.array(rat)
    log(f"E6ideal noise theory: n={len(rat)} cells with lambda R / sigma >= 3, median ratio {np.median(rat):.3f}, "
        f"range {rat.min():.2f}-{rat.max():.2f}, within 4%: {np.mean(np.abs(rat - 1) < 0.04) * 100:.0f}%")
    for sg in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1):
        d = S[(S["exp"] == "E6hw") & (S["task"] == "attract") & S["field"].isin(vs.LINEAR) & (S["side"] == 0.33) &
              (S["sigma_n"] == sg)]
        log(f"E6hw linear side 0.33 sigma {sg:g}: mean err {d['err_mean'].min() * 1000:.1f}-{d['err_mean'].max() * 1000:.1f} mm, "
            "conv " + ", ".join(f"{f}: {c * 100:.0f}%" for f, c in zip(d["field"], d["conv"])))
    for f in vs.RECONSTRUCTED:
        for e_ in ("E6hw", "E6ideal"):
            d = an.sel(S, exp=e_, task="attract", field=f).sort_values("sigma_n")
            log(f"{e_} {f}: mean err by sigma " + ", ".join(f"{s:g}: {e * 1000:.1f}" for s, e in zip(d["sigma_n"], d["err_mean"])) +
                " | bias " + ", ".join(f"{b * 1000:.1f}" for b in d["bias"]))

    # ---- E1 estimators
    for m in ("fit", "oracle", "magnitude"):
        d = an.sel(S, exp="E1hw", task="orbit", method=m, field="gyre").sort_values("rd")
        log(f"E1hw gyre orbit {m}: " + ", ".join(f"{r:g}: {e:+.3f} ok {o_:.2f}" for r, e, o_ in zip(d["rd"], d["err_mean"], d["ok"])))
    d = an.sel(S, exp="E1hw", task="attract", method="magnitude")
    log("E1hw magnitude attract median err: " + ", ".join(f"{f}: {e * 1000:.1f} mm" for f, e in zip(d["field"], d["err_med"])))
    d = an.sel(S, exp="E1hw", task="orbit", method="magnitude", field="vortex").sort_values("rd")
    log("E1hw magnitude vortex orbit mean r: " + ", ".join(f"{r:g}: {m_:.3f}" for r, m_ in zip(d["rd"], d["mean_r"])))
    same = []
    for f in vs.LINEAR:
        for r in an.sel(S, exp="E1hw", task="orbit", method="fit", field=f).itertuples():
            o_ = an.sel(S, exp="E1hw", task="orbit", method="oracle", field=f, rd=r.rd)
            same.append(abs(r.err_mean - o_["err_mean"].values[0]))
    log(f"E1hw linear fields, fit vs oracle orbit error: max |diff| {max(same) * 1000:.3f} mm")

    # ---- random design convergence past k_max
    R = an.random_frame(res)
    if R is not None:
        A = R[(R["task"] == "attract") & (R["method"] != "magnitude") & R["field"].isin(vs.LINEAR) &
              (R["sigma_n"] == 0) & (R["vs"] == 0)].copy()
        A["kr_"] = A["k"] / th.k_max(A["alpha"].values, A["dt"].values)
        A["conv"] = (A["escaped"] == 0) & (A["max_side_err"] < 0.1) & (A["final_err"] < load.CONV)
        for lo, hi in ((0.5, 1), (1, 2), (2, 4), (4, 10), (10, 40)):
            m = (A["kr_"] >= lo) & (A["kr_"] < hi)
            log(f"E7 linear, no stiction, k/k_max {lo}-{hi}: conv {A['conv'][m].mean() * 100:.1f}% (n={m.sum()})")
        Ar = R[(R["task"] == "attract") & R["field"].isin(vs.LINEAR) & (R["sigma_n"] == 0) & (R["vs"] > 0)]
        log(f"E7 random attraction trials with friction: {len(Ar):,}; orbit trials total "
            f"{(R['task'] == 'orbit').sum():,}")
    open(os.path.join(HERE, res, "report_numbers.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
