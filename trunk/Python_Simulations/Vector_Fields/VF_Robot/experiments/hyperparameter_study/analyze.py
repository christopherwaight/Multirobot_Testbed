"""
Figures, tables and theory checks for the hyperparameter study.

    venv/bin/python3 experiments/hyperparameter_study/analyze.py [--res results]

Writes figures/fig*.png, results/summary.csv, results/tables.md, and prints
the validation numbers quoted in hyperparameter_study.md.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vecsim as vs
import theory as th
import load
import design

FIG = os.path.join(HERE, "figures")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NICE = {"sink": "sink", "source": "source", "vortex": "vortex", "sinking_vortex": "sinking vortex",
        "spewing_vortex": "spewing vortex", "saddle": "saddle", "gyre": "gyre",
        "curved_saddle": "curved saddle", "vortex_nn": "vortex (recon.)", "saddle_nn": "saddle (recon.)"}

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titlesize": 9, "axes.titleweight": "bold"})


def lam(field):
    return float(np.sqrt(abs(np.linalg.det(vs.make_field(field).J0)))) if field in vs.LINEAR else 1.0


# ============================================================================
def build_summary(res):
    path = os.path.join(HERE, res, "summary.csv")
    rows = [r for r in load.manifest(res) if not r["random"] and os.path.exists(r["path"])]
    if os.path.exists(path):
        S = pd.read_csv(path)
        if len(S) == len(rows):
            return S
    recs = []
    for r in rows:
        s = load.summary(r)
        rec = {k: r[k] for k in ("id", "file", "exp", "task", "method", "field", "eps", "family", "N", "dt",
                                 "T", "window", "stiction", "hw")}
        rec.update({k: r[k] for k in vs.PARAM_KEYS})
        rec.update(s)
        recs.append(rec)
    S = pd.DataFrame(recs)
    S.to_csv(path, index=False)
    return S


def add_orbit_predictions(S):
    o = S["task"] == "orbit"
    S.loc[o, "pred_r"] = [th.orbit_r_exact(r.rd, r.alpha, r.dt, r.kt, r.kr, r.vmax) for r in S[o].itertuples()]
    S.loc[o, "pred_r_small"] = th.orbit_r_small_angle(S.loc[o, "rd"], S.loc[o, "alpha"], S.loc[o, "dt"],
                                                      S.loc[o, "kt"], S.loc[o, "kr"], S.loc[o, "vmax"])
    if "stiction" not in S:
        S["stiction"] = "legacy"
    S["stiction"] = S["stiction"].fillna("legacy")
    leg = S["stiction"] == "legacy"
    S.loc[o, "pred_start"] = np.where(leg[o], (1 - S.loc[o, "alpha"]) * S.loc[o, "kt"] >= S.loc[o, "vs"],
                                      S.loc[o, "kt"] >= S.loc[o, "vs"])
    S["exact_est"] = (S["method"] == "oracle") | ((S["method"] == "fit") & S["field"].isin(vs.LINEAR))
    return S


def sel(S, **kw):
    m = np.ones(len(S), bool)
    for k, v in kw.items():
        m &= (S[k] == v).values if not isinstance(v, (list, tuple)) else S[k].isin(v).values
    return S[m]


def legend_and_labels(ax, lines, labels, loc="best"):
    ax.legend(lines, labels, loc=loc, fontsize=7.5, handlelength=1.8)


# ============================================================================
# Figure 1: orbit radius vs commanded radius, lag; collapse onto theory
# ============================================================================
def fig1(S, R7, out):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axs[0]
    E = sel(S, exp="E2alpha", field="vortex", method="fit")
    E9 = sel(S, exp="E2vs", field="vortex", method="fit", alpha=0.9, vs=0.0)
    series = [(a, sel(E, alpha=a)) for a in (0.0, 0.5, 0.7, 0.8)] + [(0.9, E9)]
    rr = np.logspace(-4, 1, 200)
    hs, ls = [], []
    for i, (a, D) in enumerate(series):
        D = D.sort_values("rd")
        col = SEQ[i]
        pred = th.orbit_r_small_angle(rr, a, 0.1, 0.2, 0.5) - rr
        ax.plot(rr, pred, color=col, lw=1.2, alpha=0.9)
        h = ax.plot(D["rd"], D["err_mean"], MARK[i], color=col, mec="white", mew=0.6, ms=5)[0]
        hs.append(h)
        ls.append(f"alpha {a:g}" + (" (no stiction)" if a == 0.9 else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("commanded radius r_d (m)")
    ax.set_ylabel("mean radius error r - r_d (m)")
    ax.set_title("(a) Orbit error vs r_d and lag (vortex)")
    legend_and_labels(ax, hs, ls, "lower left")
    ax.set_ylim(top=0.8)
    ax.text(0.02, 0.45, "lines: closed form\nmarkers: simulation", color=INK2, fontsize=7)

    ax = axs[1]
    hs, ls = [], []
    D = sel(S, exp="E2kt", field="vortex", method="fit")
    for i, kt in enumerate((0.1, 0.2, 0.3)):
        d = sel(D, kt=kt).sort_values("rd")
        ax.plot(rr, th.orbit_r_small_angle(rr, 0.7, 0.1, kt, 0.5) - rr, color=CAT[i], lw=1.2)
        hs.append(ax.plot(d["rd"], d["err_mean"], MARK[i], color=CAT[i], mec="white", mew=0.6)[0])
        ls.append(f"k_t {kt:g} m/s")
    D = sel(S, exp="E2kr", field="vortex", method="fit")
    for j, kr in enumerate((0.1, 5.0)):
        d = sel(D, kr=kr).sort_values("rd")
        ax.plot(rr, th.orbit_r_small_angle(rr, 0.7, 0.1, 0.2, kr) - rr, color=CAT[3 + j], lw=1.2, ls="--")
        hs.append(ax.plot(d["rd"], d["err_mean"], MARK[3 + j], color=CAT[3 + j], mec="white", mew=0.6)[0])
        ls.append(f"k_r {kr:g} /s")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("commanded radius r_d (m)")
    ax.set_ylim(1e-4, 1.0)
    if (D["stiction"] == "legacy").all():
        ax.text(1.5e-4, 1.6e-4, "k_t 0.05 never starts: (1 - alpha) k_t < v_s", color=INK2, fontsize=7)
    ax.set_title("(b) Gains (alpha 0.7, vortex)")
    legend_and_labels(ax, hs, ls, "lower left")

    ax = axs[2]
    X = sel(S, task="orbit")
    X = X[X["exact_est"] & (X["sigma_n"] == 0) & (X["ok"] > 0.5) & np.isfinite(X["pred_r"])]
    X = X[X["kt"] <= X["vmax"]]
    pm = X["pred_r_small"] - X["rd"]
    mm = X["err_mean"]
    if R7 is not None and len(R7):
        v = R7["valid"]
        ax.loglog(R7["pred_e"][~v], R7["meas_e"][~v], ".", color="#b5b4af", ms=1.5, alpha=0.4,
                  label=f"random, outside validity ({(~v).sum():,})")
        ax.loglog(R7["pred_e"][v], R7["meas_e"][v], ".", color=CAT[1], ms=1.5, alpha=0.3,
                  label=f"random, inside validity ({v.sum():,})")
    ax.loglog(pm, mm, "o", color=CAT[0], ms=3, alpha=0.8, mec="none", label=f"designed cells ({len(X)})")
    lim = [1e-5, 2]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted r* - r_d (m)")
    ax.set_ylabel("simulated mean r - r_d (m)")
    ax.set_title("(c) Theory vs simulation, exact estimate")
    ax.legend(loc="upper left", fontsize=7.5, markerscale=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_orbit_radius.png"))
    plt.close(fig)


# ============================================================================
# Figure 2: attraction gain, lag, stiction
# ============================================================================
def fig2(S, res, out):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    rows = {r["id"]: r for r in load.manifest(res)}
    ax = axs[0]
    D = sel(S, exp="E3k", field="vortex", method="fit")
    hs, ls = [], []
    for i, a in enumerate((0.0, 0.5, 0.7, 0.9)):
        d = sel(D, alpha=a).sort_values("k")
        pred = []
        for r in d.itertuples():
            M = load.trials(rows[r.id])
            _, ov, _ = th.reduced_1d(M["start_dist"], r.k, a, r.dt, r.vs, r.vmax, r.vcmd_max, 60.0, model=r.stiction)
            pred.append(np.median(ov))
        ax.plot(d["k"], pred, color=SEQ[i + 1], lw=1.2)
        hs.append(ax.plot(d["k"], d["over_med"], MARK[i], color=SEQ[i + 1], mec="white", mew=0.6)[0])
        ls.append(f"alpha {a:g}")
        kc = th.k_crit(a, 0.1)
        ax.axvline(kc, color=SEQ[i + 1], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("attraction gain k (1/s)")
    ax.set_ylabel("median overshoot past target (m)")
    ax.set_title("(a) Overshoot (dotted: k_c onset)")
    legend_and_labels(ax, hs, ls, "upper left")

    ax = axs[1]
    D = sel(S, exp="E3vs", field="vortex", method="fit")
    hs, ls = [], []
    for i, a in enumerate((0.0, 0.5, 0.7, 0.9)):
        d = sel(D, alpha=a).sort_values("vs")
        pred, meas = [], []
        for r in d.itertuples():
            M = load.trials(rows[r.id])
            fe, _, _ = th.reduced_1d(M["start_dist"], 1.0, a, r.dt, r.vs, 0.3, 0.5, 60.0, model=r.stiction)
            pred.append(np.median(fe))
            meas.append(np.median(M["final_err"]))
        x = d["vs"].values
        ax.plot(x, np.array(pred) * 100, color=SEQ[i + 1], lw=1.2)
        hs.append(ax.plot(x, np.array(meas) * 100, MARK[i], color=SEQ[i + 1], mec="white", mew=0.6)[0])
        ls.append(f"alpha {a:g}")
    ax.plot([0, 0.1], [0, 10], color=INK2, lw=0.8, ls="--")
    ax.text(0.045, 6.0, "v_s / k", color=INK2, fontsize=7)
    ax.set_xlabel("stiction threshold v_s (m/s)")
    ax.set_ylim(-0.5, 12)
    if (D["stiction"] == "legacy").all():
        ax.text(0.099, 0.4, "alpha 0.9 is off scale for v_s >= 0.05:\nit stalls from rest once\n(1 - alpha) x 0.5 m/s < v_s",
                color=INK2, fontsize=6.5, ha="right")
    ax.set_ylabel("median final error (cm)")
    ax.set_title("(b) Stiction stop at k = 1")
    legend_and_labels(ax, hs, ls, "upper left")

    ax = axs[2]
    D = sel(S, exp="E3k", field="vortex", method="fit")
    A = sorted(D["alpha"].unique())
    K = sorted(D["k"].unique())
    rows_ = {r["id"]: r for r in load.manifest(res)}
    Z = np.array([[np.median(load.trials(rows_[sel(D, alpha=a, k=k)["id"].values[0]])["final_err"]) * 1000
                   for k in K] for a in A])
    from matplotlib.colors import LogNorm
    im = ax.pcolormesh(np.arange(len(K) + 1), np.arange(len(A) + 1), np.clip(Z, 0.05, 500), cmap="Blues",
                       norm=LogNorm(0.05, 500))
    for i in range(len(A)):
        for j in range(len(K)):
            z = Z[i, j]
            lab = "<.05" if z < 0.05 else (f"{z:.0f}" if z >= 10 else (f"{z:.1f}" if z >= 1 else f"{z:.2f}"))
            ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=6.5,
                    color="white" if z > 20 else INK)
    ax.set_xticks(np.arange(len(K)) + 0.5)
    ax.set_xticklabels([f"{k:g}" for k in K])
    ax.set_yticks(np.arange(len(A)) + 0.5)
    ax.set_yticklabels([f"{a:g}" for a in A])
    ax.grid(False)
    for i, a in enumerate(A):
        km = float(th.k_max(a, 0.1))
        j = np.interp(np.log(km), np.log(K), np.arange(len(K))) + 0.5 if km <= K[-1] else None
        if j is not None:
            ax.plot([j, j], [i, i + 1], color=CAT[7], lw=2)
        kc = th.k_crit(a, 0.1)
        jc = np.interp(np.log(kc), np.log(K), np.arange(len(K))) + 0.5 if K[0] <= kc <= K[-1] else None
        if jc is not None:
            ax.plot([jc, jc], [i, i + 1], color=CAT[3], lw=2)
    ax.set_xlabel("attraction gain k (1/s)")
    ax.set_ylabel("lag alpha")
    ax.set_title("(c) Median final error (mm)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_attraction.png"))
    plt.close(fig)


# ============================================================================
# Figure 3: formation size, curvature
# ============================================================================
SIZE_FIELDS = [("vortex", "potential", "vortex (linear)"), ("gyre", "potential", "gyre"),
               ("curved_saddle", "potential", "curved saddle, irrotational"),
               ("curved_saddle", "cellular", "curved saddle, rotational"),
               ("vortex_nn", "potential", "vortex (recon.)"), ("saddle_nn", "potential", "saddle (recon.)")]


def fig3(S, out):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    sides = np.array(design.SIDES)
    ax = axs[0]
    hs, ls = [], []
    for i, (f, fam, lab) in enumerate(SIZE_FIELDS):
        d = sel(S, exp="E4ideal", task="attract", method="fit", field=f, family=fam).sort_values("side")
        y = np.where(d["conv"] > 0.5, d["err_med"], np.nan)
        hs.append(ax.plot(d["side"], np.maximum(y, 1e-7), "-" + MARK[i], color=CAT[i], lw=1.4, mec="white",
                          mew=0.6)[0])
        ls.append(lab)
        if f in ("gyre", "curved_saddle"):
            F = vs.make_field(f, 1.0, fam)
            b = [th.fixed_point_bias(F, "fit", s) for s in sides]
            ax.plot(sides, b, "x", color=CAT[i], ms=6, mew=1.5)
    ax.plot(sides, 0.0066 * (sides / 0.33) ** 3, color=INK2, lw=0.8, ls="--")
    ax.text(0.12, 2e-5, "slope 3", color=INK2, fontsize=7)
    ax.text(0.011, 2.2e-6, "linear: 1e-6 is the controller's\nstop radius, not an estimate error", color=INK2, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 2)
    ax.set_xlabel("formation side (m)")
    ax.set_ylabel("median final error (m)")
    ax.set_title("(a) Localization vs size (x: fixed-point theory)")
    fig.legend(hs, ls, loc="upper center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, 1.07))

    ax = axs[1]
    for i, (f, fam, lab) in enumerate(SIZE_FIELDS):
        d = sel(S, exp="E4ideal", task="orbit", method="fit", field=f, family=fam, rd=0.3).sort_values("side")
        y = np.where(d["ok"] > 0.5, d["err_mean"], np.nan)
        ax.plot(d["side"], y, "-" + MARK[i], color=CAT[i], lw=1.4, mec="white", mew=0.6)
    lin = th.orbit_r_exact(0.3, 0.0, 0.1, 0.2, 0.5) - 0.3
    ax.axhline(lin, color=INK2, lw=0.8, ls="--")
    ax.text(0.012, lin + 0.004, "exact-estimate value", color=INK2, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("formation side (m)")
    ax.set_ylabel("mean radius error (m), r_d = 0.3")
    ax.set_title("(b) Orbit vs size (no lag, no stiction)")

    eps = np.array(design.EPSS)
    for j, (key, ylab, title) in enumerate([("err_mean", "mean radius error (m)", "(c) Curvature: mean radius"),
                                            ("a2", "cos 2 phi wobble amplitude (m)", "(d) Curvature: orbit shape")]):
        ax = axs[2 + j]
        hs, ls = [], []
        for i, fam in enumerate(("potential", "cellular")):
            d = sel(S, exp="E5ideal", field="curved_saddle", family=fam, side=0.1, rd=0.3).sort_values("eps")
            good = (d["ok"] > 0.5) & (d["mean_r"] < 0.6)
            y = np.where(good, d[key], np.nan)
            top = 0.17 if key == "err_mean" else 0.06
            for e_, g_ in zip(d["eps"], good):
                if not g_:
                    ax.plot(e_, top, MARK[i], color=CAT[2 + i], mfc="white", mew=1.2, ms=6)
            hs.append(ax.plot(d["eps"], y, MARK[i], color=CAT[2 + i], mec="white", mew=0.6, ms=6)[0])
            ls.append("irrotational" if fam == "potential" else "rotational")
            if key == "err_mean":
                p = []
                for e in eps:
                    F = vs.make_field("curved_saddle", e, fam)
                    p.append(th.orbit_r_nonlinear(F, "fit", 0.1, 0.3, 0.0, 0.1, 0.2, 0.5, 1e9) - 0.3)
                p = np.array(p)
                p[np.abs(eps) > 1] = np.nan
                ax.plot(eps, p, color=CAT[2 + i], lw=1.2)
        ax.set_ylim(-0.1, 0.19) if key == "err_mean" else ax.set_ylim(-0.003, 0.065)
        ax.text(2.9 if key == "err_mean" else 0, 0.10 if key == "err_mean" else 0.052, "hollow, top edge: orbit lost\n(captured by a neighbor or runaway)",
                ha="center", color=INK2, fontsize=6.5)
        ax.set_xlabel("eps  (v = -y + eps y^3)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        legend_and_labels(ax, hs, ls, "center left" if key == "err_mean" else "center left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_size_curvature.png"))
    plt.close(fig)


# ============================================================================
# Figure 4: sensor noise
# ============================================================================
def fig4(S, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    sig = np.array([s for s in design.SIGMAS if s > 0])
    for ax, f in zip(axs, ("vortex", "sinking_vortex")):
        L = lam(f)
        hs, ls = [], []
        for i, side in enumerate((0.1, 0.33, 1.0)):
            d = sel(S, exp="E6ideal", task="attract", field=f, side=side).sort_values("sigma_n")
            d = d[d["sigma_n"] > 0]
            hs.append(ax.plot(d["sigma_n"], d["err_mean"], MARK[i], color=CAT[i], mec="white", mew=0.6, ms=6)[0])
            ls.append(f"side {side:g} m")
            R = side / np.sqrt(3)
            snr_edge = L * R / 3
            ax.axvline(snr_edge, color=CAT[i], lw=0.8, ls=":")
        ax.plot(sig, [th.noise_rms(s, L, 1.0, 0.0, 0.1) * np.sqrt(np.pi) / 2 for s in sig], color=INK2, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sensor noise sigma_n (field units)")
        ax.set_title(f"({'a' if f == 'vortex' else 'b'}) {NICE[f]}, |det J|^(1/2) = {L:.2f}")
        legend_and_labels(ax, hs, ls, "upper left")
    axs[0].set_ylabel("mean final error, converged trials (m)")
    axs[1].text(0.095, 2.2e-4, "line: covariance + loop theory\ndotted: lambda R / sigma_n = 3",
                color=INK2, fontsize=7, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_noise.png"))
    plt.close(fig)


# ============================================================================
# Figure 5: estimators
# ============================================================================
def fig5(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    F = vs.FIELDS
    hs, ls = [], []
    for j, m in enumerate(("fit", "oracle", "magnitude")):
        y = [sel(S, exp="E1hw", task="attract", method=m, field=f)["err_med"].values[0] for f in F]
        c = [sel(S, exp="E1hw", task="attract", method=m, field=f)["conv"].values[0] for f in F]
        x = np.arange(len(F)) + (j - 1) * 0.22
        hs.append(ax.plot(x, np.maximum(y, 1e-5), MARK[j], color=CAT[j], mec="white", mew=0.6, ms=6, ls="none")[0])
        ls.append({"fit": "plane fit", "oracle": "true critical point", "magnitude": "magnitude only"}[m])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(F)))
    ax.set_xticklabels([NICE[f] for f in F], rotation=55, ha="right")
    ax.set_ylabel("median final error (m), hardware settings")
    ax.set_title("(a) Attraction, converged trials")
    legend_and_labels(ax, hs, ls, "upper left")
    for ax, f in zip(axs[1:], ("gyre", "saddle_nn")):
        hs, ls = [], []
        for j, m in enumerate(("fit", "oracle", "magnitude")):
            d = sel(S, exp="E1hw", task="orbit", method=m, field=f).sort_values("rd")
            y = np.where(d["ok"] > 0.5, d["err_mean"], np.nan)
            hs.append(ax.errorbar(d["rd"], y, yerr=np.where(d["ok"] > 0.5, d["within_sd"], np.nan), fmt="-" + MARK[j],
                                  color=CAT[j], mec="white", mew=0.6, ms=5, lw=1.4, capsize=2)[0])
            ls.append({"fit": "plane fit", "oracle": "true critical point", "magnitude": "magnitude only"}[m])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("commanded radius r_d (m)")
        ax.set_ylabel("mean radius error (m), bars: within-trial sd")
        ax.set_title(f"({'b' if f == 'gyre' else 'c'}) Orbit, {NICE[f]}")
        legend_and_labels(ax, hs, ls, "best")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_estimators.png"))
    plt.close(fig)


# ============================================================================
# Figure 6: random design and index prediction
# ============================================================================
def random_frame(res):
    recs = []
    for r in load.manifest(res):
        if not r["random"] or not os.path.exists(r["path"]):
            continue
        M = load.trials(r)
        n = len(M["escaped"])
        d = dict(field=[r["field"]] * n, method=[r["method"]] * n, task=[r["task"]] * n,
                 dt=np.full(n, r["dt"]), family=[r["family"]] * n, eps=np.full(n, r["eps"]),
                 stiction=[r["stiction"]] * n)
        for k in vs.PARAM_KEYS:
            d[k] = M["p_" + k]
        for k in ("escaped", "max_side_err", "path"):
            d[k] = M[k]
        if r["task"] == "attract":
            for k in ("final_err", "overshoot", "start_dist", "nearest", "nearest_dist", "nearest_index", "det0"):
                d[k] = M[k]
        else:
            for k in ("mean_r", "std_r", "revs"):
                d[k] = M[k]
        recs.append(pd.DataFrame(d))
    return pd.concat(recs, ignore_index=True) if recs else None


def fig6(R, S, res, out):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    A = R[R["task"] == "attract"].copy()
    A = A[A["method"] != "magnitude"]
    A["kratio"] = A["k"] / th.k_max(A["alpha"].values, A["dt"].values)
    A["ok"] = (A["escaped"] == 0) & (A["max_side_err"] < 0.1)
    A["conv"] = A["ok"] & (A["final_err"] < load.CONV)
    lin = A[A["field"].isin(vs.LINEAR) & (A["sigma_n"] == 0) & (A["vs"] == 0)]
    ax = axs[0]
    bins = np.logspace(-3, 1.5, 19)
    idx = np.digitize(lin["kratio"], bins)
    xs, ys, ns = [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 50:
            xs.append(np.sqrt(bins[b - 1] * bins[b]))
            ys.append(lin["conv"][m].mean() * 100)
    ax.plot(xs, ys, "-o", color=CAT[0], mec="white", mew=0.6)
    ax.axvline(1, color=CAT[7], lw=1, ls="--")
    ax.text(1.15, 99, "k = k_max", color=INK, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("k / k_max,  k_max = 2(1 + alpha) / (dt (1 - alpha))")
    ax.set_ylabel("% converged within 5 cm")
    ax.set_title("(a) Sampled-loop stability (random trials)")

    ax = axs[1]
    st = A[A["field"].isin(vs.LINEAR) & (A["sigma_n"] == 0) & (A["vs"] > 0) & (A["kratio"] < 0.5)].copy()
    fac = np.where(st["stiction"] == "legacy", 1 - st["alpha"], 1.0)
    st["start_ratio"] = fac * np.minimum(st["k"] * st["start_dist"], st["vcmd_max"]) / st["vs"]
    bins = np.logspace(-1.5, 2, 22)
    idx = np.digitize(st["start_ratio"], bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 50:
            xs.append(np.sqrt(bins[b - 1] * bins[b]))
            ys.append((st["path"][m] > 1e-9).mean() * 100)
    ax.plot(xs, ys, "-s", color=CAT[1], mec="white", mew=0.6)
    ax.axvline(1, color=CAT[7], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("|first command| / v_s" if (st["stiction"] != "legacy").all() else "(1 - alpha) |first command| / v_s")
    ax.set_ylabel("% of trials that move at all")
    ax.set_title("(b) Stiction start condition")

    ax = axs[2]
    rows = {r["id"]: r for r in load.manifest(res)}
    recs = []
    for r in sel(S, task="attract", method="fit").itertuples():
        if r.field not in ("gyre", "curved_saddle") or not r.exp.startswith(("E1", "E4", "E5")):
            continue
        if r.field == "curved_saddle" and not (r.family == "cellular" and r.eps > 0):
            continue
        M = load.trials(rows[r.id])
        F = vs.make_field(r.field, r.eps, r.family)
        n = len(M["det0"])
        rng = np.random.default_rng(design._seed(r.id))
        rng.uniform(0, 2 * np.pi, n)
        st0 = F.target + rng.uniform(-F.start_half, F.start_half, (n, 2))
        a = 0.5 if r.field == "gyre" else np.pi / (2 * np.sqrt(6 * r.eps))
        dist = np.abs(a - np.abs(st0).sum(1)) / np.sqrt(2) / (r.side / np.sqrt(3))
        arrived = (M["nearest_dist"] < load.CONV) & (M["escaped"] == 0)
        recs.append(pd.DataFrame(dict(dist=dist, match=np.sign(M["det0"]) == M["nearest_index"], arrived=arrived)))
    X = pd.concat(recs)
    X = X[X["arrived"]]
    bins = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 100])
    idx = np.digitize(X["dist"], bins)
    xs = [f"{bins[b - 1]:g}-{bins[b]:g}" if bins[b] < 100 else f">{bins[b - 1]:g}" for b in range(1, len(bins))]
    ys = [X["match"][idx == b].mean() * 100 if (idx == b).sum() else np.nan for b in range(1, len(bins))]
    ns = [(idx == b).sum() for b in range(1, len(bins))]
    ax.bar(np.arange(len(xs)), ys, color=CAT[2], width=0.7)
    for i, y in enumerate(ys):
        ax.text(i, (y if np.isfinite(y) else 0) + 0.15, f"{y:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f"{x_}\n{n_ // 1000}k" if n_ >= 1000 else f"{x_}\n{n_}" for x_, n_ in zip(xs, ns)],
                       fontsize=6.5)
    ax.set_ylim(97, 100.8)
    ax.set_xlabel("start distance to det J = 0, in formation circumradii")
    ax.set_ylabel("% where sign det J_hat(start) = type reached")
    ax.set_title("(c) Destination type from the first fit")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_random_index.png"))
    plt.close(fig)
    return X


# ============================================================================
def tables(S, out_md):
    L = []
    L.append("## Attraction, hardware settings (E1hw), per-trial final distance to the true critical point\n")
    L.append("| field | conv % | mean (m) | median | p95 | bias | precision | t to 5 cm (s) | overshoot p95 | oracle median | magnitude conv % / median |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in vs.FIELDS:
        a = sel(S, exp="E1hw", task="attract", method="fit", field=f).iloc[0]
        o = sel(S, exp="E1hw", task="attract", method="oracle", field=f).iloc[0]
        g = sel(S, exp="E1hw", task="attract", method="magnitude", field=f).iloc[0]
        L.append(f"| {NICE[f]} | {a.conv * 100:.1f} | {a.err_mean:.4f} | {a.err_med:.4f} | {a.err_p95:.4f} | "
                 f"{a.bias:.4f} | {a.prec:.4f} | {a.t5_med:.1f} | {a.over_p95:.4f} | {o.err_med:.4f} | "
                 f"{g.conv * 100:.1f} / {g.err_med:.4f} |")
    for tag in ("hw", "ideal"):
        L.append(f"\n## Orbit, {'hardware settings' if tag == 'hw' else 'robot non-idealities off'} (E1{tag}), "
                 "mean radius error (m), between-trial sd, within-trial sd, % trials kept\n")
        L.append("| field | method | " + " | ".join(f"r_d {r:g}" for r in design.PAPER_RDS) + " |")
        L.append("|---|---|" + "---|" * len(design.PAPER_RDS))
        for f in vs.FIELDS:
            for m in ("fit", "oracle", "magnitude"):
                d = sel(S, exp="E1" + tag, task="orbit", method=m, field=f).sort_values("rd")
                cells = []
                for r in d.itertuples():
                    if r.ok < 0.5:
                        cells.append(f"fail ({r.ok * 100:.0f}% kept)")
                    else:
                        cells.append(f"{r.err_mean:+.3f} ({r.err_between_sd:.3f}, {r.within_sd:.3f}){'' if r.ok > 0.99 else f' {r.ok * 100:.0f}%'}")
                L.append(f"| {NICE[f]} | {m} | " + " | ".join(cells) + " |")
    open(out_md, "w").write("\n".join(L) + "\n")


def checks(S, R, X):
    out = []
    o = S[(S["task"] == "orbit") & S["exact_est"] & (S["sigma_n"] == 0) & (S["ok"] > 0.5)]
    d = (o["mean_r"] - o["pred_r"]).abs()
    out.append(f"orbit r*, exact estimate, designed cells: n={len(o)}, median |err| {d.median() * 1000:.2f} mm, "
               f"95% {d.quantile(.95) * 1000:.2f} mm, max {d.max() * 1000:.1f} mm")
    ds = (o["mean_r"] - o["pred_r_small"]).abs()
    out.append(f"   small-angle form: median {ds.median() * 1000:.2f} mm, 95% {ds.quantile(.95) * 1000:.2f} mm")
    oo = S[(S["task"] == "orbit")]
    agree = ((oo["nostart"] > 0.5) == (~oo["pred_start"].astype(bool))).mean()
    out.append(f"orbit start condition (model-specific): agrees in {agree * 100:.1f}% of {len(oo)} cells")
    if R is not None:
        O = R[(R["task"] == "orbit") & (R["sigma_n"] == 0) & ((R["method"] == "oracle") |
              ((R["method"] == "fit") & R["field"].isin(vs.LINEAR)))]
        started = O["path"] > 1e-9
        pred = np.where(O["stiction"] == "legacy", (1 - O["alpha"]) * O["kt"] >= O["vs"], O["kt"] >= O["vs"])
        out.append(f"random design start condition: {((started == pred).mean()) * 100:.2f}% of {len(O):,} trials")
    if X is not None:
        out.append(f"index prediction (gyre + rotational saddles): {X['match'].mean() * 100:.2f}% of {len(X):,} arrived trials; "
                   f">= 1 circumradius from det J = 0: {X[X['dist'] >= 1]['match'].mean() * 100:.2f}% "
                   f"(n={int((X['dist'] >= 1).sum()):,})")
    return out


def random_orbit_frame(R):
    O = R[(R["task"] == "orbit") & (R["sigma_n"] == 0) & ((R["method"] == "oracle") |
          ((R["method"] == "fit") & R["field"].isin(vs.LINEAR)))].copy()
    O = O[(O["escaped"] == 0) & (O["max_side_err"] < 0.1) & (O["revs"] >= 0.25) & (O["path"] > 1e-9)]
    O = O[(O["kt"] <= O["vmax"])]
    sub = O.sample(min(len(O), 20000), random_state=0)
    sub["pred_e"] = th.orbit_r_small_angle(sub.rd, sub.alpha, sub.dt, sub.kt, sub.kr, sub.vmax) - sub.rd
    sub["meas_e"] = sub["mean_r"] - sub["rd"]
    phase = sub.kt / (sub.rd + sub.pred_e) * (th.t_lag(sub.alpha, sub.dt) + sub.dt / 2)
    sub["valid"] = (phase <= 0.5) & (sub.kr * sub.pred_e <= sub.rclamp)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    global FIG
    if a.res.startswith("results_v2"):
        FIG = os.path.join(HERE, "figures_v2")
    os.makedirs(FIG, exist_ok=True)
    S = add_orbit_predictions(build_summary(a.res))
    R = random_frame(a.res)
    R7 = random_orbit_frame(R) if R is not None else None
    if R7 is not None:
        e = (R7["meas_e"] - R7["pred_e"]).abs()
        print(f"random orbit trials (exact estimate, k_t <= v_max): n={len(R7):,}, median |err| {e.median() * 1000:.2f} mm, "
              f"95% {e.quantile(.95) * 1000:.2f} mm")
        v = R7[R7.valid]
        ev = (v.meas_e - v.pred_e).abs()
        print(f"   inside validity (phase <= 0.5 rad, k_r e <= clamp): n={len(v):,} ({len(v) / len(R7) * 100:.1f}%), "
              f"median {ev.median() * 1000:.2f} mm, 95% {ev.quantile(.95) * 1000:.2f} mm, "
              f"relative 95% {(ev / v.pred_e).quantile(.95) * 100:.1f}%")
    fig1(S, R7, FIG)
    fig2(S, a.res, FIG)
    fig3(S, FIG)
    fig4(S, FIG)
    fig5(S, FIG)
    X = fig6(R, S, a.res, FIG) if R is not None else None
    tables(S, os.path.join(HERE, a.res, "tables.md"))
    for line in checks(S, R, X):
        print(line)
    if a.res.startswith("results_v2"):
        import analyze_v2
        analyze_v2.run(S, a.res, FIG)


if __name__ == "__main__":
    main()
