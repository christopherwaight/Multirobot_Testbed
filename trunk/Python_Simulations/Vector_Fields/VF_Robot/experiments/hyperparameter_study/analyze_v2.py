"""
Version 2 figures (called from analyze.py --res results_v2):
  fig7  stiction models across sample periods, and against hardware
  fig8  overshoot and undershoot as functions of k, alpha (tau) and dt
  fig9  raised speed limits
and the numbers quoted in hyperparameter_study.md.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import vecsim as vs
import theory as th
import load
from analyze import sel, legend_and_labels, CAT, SEQ, MARK, INK, INK2

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("legacy", "legacy (test on lagged velocity)"), ("scaled", "scaled threshold (1 - alpha) v_s"),
          ("deadzone", "command dead zone (ZOH exact)")]


def _trials(res, rid):
    rows = {r["id"]: r for r in load.manifest(res)}
    return load.trials(rows[rid])


def hardware():
    out = {}
    for f in ("vortex", "saddle"):
        a = np.load(os.path.join(HERE, f"hardware_{f}_starts_finals.npy"))
        out[f] = np.hypot(a[:, 2], a[:, 3])
    return out


def fig7(S, res, out, log):
    rows = {r["id"]: r for r in load.manifest(res)}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    hs, ls = [], []
    for i, (m, lab) in enumerate(MODELS):
        for j, k in enumerate((1.0, 10.0)):
            d = sel(S, exp="E10", task="attract", stiction=m, k=k, vs=0.025).sort_values("dt")
            meas = [np.mean(load.trials(rows[r])["final_err"]) for r in d["id"]]
            pred = []
            for r in d.itertuples():
                M = load.trials(rows[r.id])
                fe, _, _ = th.reduced_1d(M["start_dist"], k, r.alpha, r.dt, 0.025, 0.3, 0.5, 60.0, model=m)
                pred.append(np.mean(fe))
            ax.plot(d["dt"], pred, color=CAT[i], lw=1.2, ls="-" if k == 1 else "--")
            h = ax.plot(d["dt"], meas, MARK[i], color=CAT[i], mec="white", mew=0.6,
                        mfc=CAT[i] if k == 1 else "white", ms=6)[0]
            if k == 1:
                hs.append(h)
                ls.append(lab)
            log(f"E10 attract {m:8s} k {k:4g}: mean final error by dt " +
                ", ".join(f"{a:g}: {b * 1000:.1f} mm" for a, b in zip(d["dt"], meas)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample period dt (s), tau fixed at 0.28 s")
    ax.set_ylabel("mean final error (m), v_s = 0.025")
    ax.set_title("(a) Attraction stop vs dt (filled k=1, hollow k=10)")
    fig.legend(hs, ls, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.36, 1.07))

    ax = axs[1]
    hs, ls = [], []
    for i, (m, lab) in enumerate(MODELS):
        for j, kt in enumerate((0.05, 0.2)):
            d = sel(S, exp="E10", task="orbit", stiction=m, kt=kt, vs=0.025).sort_values("dt")
            moved = [np.mean(load.trials(rows[r])["path"] > 1e-9) * 100 for r in d["id"]]
            off = (i - 1) * 2.0
            h = ax.plot(d["dt"], np.array(moved) + off, "-" + MARK[i], color=CAT[i], mec="white", mew=0.6,
                        ls="-" if kt == 0.2 else "--", mfc=CAT[i] if kt == 0.2 else "white", lw=1.4)[0]
            if kt == 0.2:
                hs.append(h)
                ls.append(m)
            log(f"E10 orbit {m:8s} k_t {kt}: % started by dt " + ", ".join(f"{a:g}: {b:.0f}" for a, b in zip(d["dt"], moved)))
    ax.set_xscale("log")
    ax.set_ylim(-8, 108)
    ax.set_xlabel("sample period dt (s), tau fixed at 0.28 s")
    ax.set_ylabel("% of orbit trials that start")
    ax.set_title("(b) Orbit start (solid k_t 0.2, dashed 0.05)")
    legend_and_labels(ax, hs, ls, "center right")

    ax = axs[2]
    hw = hardware()
    hwall = np.sort(np.concatenate([hw["vortex"], hw["saddle"]]))
    ax.plot(hwall * 100, np.arange(1, len(hwall) + 1) / len(hwall) * 100, color=INK, lw=2.4,
            label=f"hardware ({len(hwall)} runs)")
    log(f"hardware final distance: mean {hwall.mean() * 1000:.1f} mm, median {np.median(hwall) * 1000:.1f} mm, "
        f"p95 {np.percentile(hwall, 95) * 1000:.1f} mm (vortex {hw['vortex'].mean() * 1000:.1f}, saddle {hw['saddle'].mean() * 1000:.1f})")
    for i, (m, lab) in enumerate(MODELS):
        vals = []
        for f in ("vortex", "saddle"):
            d = sel(S, exp="E10hw", stiction=m, field=f)
            vals.append(load.trials(rows[d["id"].values[0]])["final_err"])
        v = np.sort(np.concatenate(vals))
        ax.plot(v * 100, np.arange(1, len(v) + 1) / len(v) * 100, color=CAT[i], lw=1.6, label=f"sim, {m}")
        log(f"E10hw {m:8s} linear fields from hardware starts: mean {v.mean() * 1000:.1f} mm, median {np.median(v) * 1000:.1f}, "
            f"p95 {np.percentile(v, 95) * 1000:.1f}")
        vals = []
        for f in ("vortex_nn", "saddle_nn"):
            d = sel(S, exp="E10hw", stiction=m, field=f)
            vals.append(load.trials(rows[d["id"].values[0]])["final_err"])
        v2 = np.sort(np.concatenate(vals))
        if m == "deadzone":
            ax.plot(v2 * 100, np.arange(1, len(v2) + 1) / len(v2) * 100, color=CAT[i], lw=1.2, ls=":",
                    label="sim, dead zone, reconstructed fields")
        log(f"E10hw {m:8s} reconstructed fields: mean {v2.mean() * 1000:.1f} mm, median {np.median(v2) * 1000:.1f}")
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("final distance to critical point (cm), 10 s runs")
    ax.set_ylabel("% of runs at or below")
    ax.set_title("(c) Hardware vs model, same 157 starts")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_stiction_models.png"))
    plt.close(fig)


def fig8(S, res, out, log):
    rows = {r["id"]: r for r in load.manifest(res)}
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
    ax = axs[0]
    D = sel(S, exp="E8aoff", vs=0.0)
    kk = np.logspace(-1, 1.8, 200)
    hs, ls = [], []
    for i, dt in enumerate((0.01, 0.05, 0.1, 0.2)):
        a = np.exp(-dt / 0.28)
        od = th.overshoot_ratio_discrete(kk, a, dt, steps=4000) * 100
        ax.plot(kk, np.where(od <= 100, od, np.nan), color=SEQ[i + 1], lw=1.2)
        km = float(th.k_max(a, dt))
        if km < kk[-1]:
            ax.axvline(km, color=SEQ[i + 1], lw=0.8, ls=":")
        d = D[(D["dt"] == dt) & (np.abs(D["alpha"] - a) < 1e-9)].sort_values("k")
        y = []
        for r in d.itertuples():
            M = load.trials(rows[r.id])
            ok = M["escaped"] == 0
            y.append(np.median((M["overshoot"] / M["start_dist"])[ok]) * 100 if ok.mean() > 0.5 else np.nan)
        hs.append(ax.plot(d["k"], y, MARK[i], color=SEQ[i + 1], mec="white", mew=0.6, ms=6)[0])
        ls.append(f"dt {dt:g} s (alpha {a:.2f})")
    z = 1 / (2 * np.sqrt(kk * 0.28))
    cont = np.where(z < 1, np.exp(-np.pi * z / np.sqrt(np.clip(1 - z * z, 1e-12, None))), 0) * 100
    ax.plot(kk, cont, color=INK, lw=1, ls="--")
    hs.append(plt.Line2D([], [], color=INK, ls="--"))
    ls.append("continuous, dt -> 0")
    ax.set_xscale("log")
    ax.set_xlabel("attraction gain k (1/s), tau = 0.28 s")
    ax.set_ylabel("overshoot / start distance (%)")
    ax.set_ylim(-3, 103)
    ax.text(60, 4, "dotted: k_max (dt 0.2),\nunstable beyond", color=INK2, fontsize=6.5, ha="right")
    ax.set_title("(a) Overshoot, no caps (lines: theory)")
    legend_and_labels(ax, hs, ls, "upper left")

    ax = axs[1]
    taus = [0.0, 0.1, 0.28, 0.5, 1.0]
    dts = [0.01, 0.02, 0.05, 0.1, 0.2]
    Z = np.full((len(taus), len(dts)), np.nan)
    for i, tau in enumerate(taus):
        for j, dt in enumerate(dts):
            a = np.exp(-dt / tau) if tau > 0 else 0.0
            d = D[(D["dt"] == dt) & (np.abs(D["alpha"] - a) < 1e-9) & (D["k"] == 3.0)]
            if len(d):
                M = load.trials(rows[d["id"].values[0]])
                ok = M["escaped"] == 0
                Z[i, j] = np.median((M["overshoot"] / M["start_dist"])[ok]) * 100 if ok.mean() > 0.5 else np.inf
    im = ax.pcolormesh(np.arange(len(dts) + 1), np.arange(len(taus) + 1), np.clip(Z, 0, 100), cmap="Blues",
                       vmin=0, vmax=100)
    for i in range(len(taus)):
        for j in range(len(dts)):
            z_ = Z[i, j]
            ax.text(j + 0.5, i + 0.5, "unst." if np.isinf(z_) else f"{z_:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if (np.isinf(z_) or z_ > 55) else INK)
    ax.set_xticks(np.arange(len(dts)) + 0.5)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.set_yticks(np.arange(len(taus)) + 0.5)
    ax.set_yticklabels([f"{t:g}" for t in taus])
    ax.grid(False)
    ax.set_xlabel("sample period dt (s)")
    ax.set_ylabel("actuator time constant tau (s)")
    ax.set_title("(b) Overshoot % at k = 3, no caps")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    log("E8a overshoot % at k=3 (rows tau " + str(taus) + ", cols dt " + str(dts) + "):\n" +
        "\n".join("   " + " ".join(f"{z_:6.1f}" for z_ in row) for row in Z))

    ax = axs[2]
    D2 = sel(S, exp="E8aoff", vs=0.025, dt=0.1)
    hs, ls = [], []
    for i, tau in enumerate(taus):
        a = np.exp(-0.1 / tau) if tau > 0 else 0.0
        d = D2[np.abs(D2["alpha"] - a) < 1e-9].sort_values("k")
        d = d[d["k"] < th.k_max(a, 0.1)]
        y = [np.median(load.trials(rows[r])["final_proj"]) * 100 for r in d["id"]]
        hs.append(ax.plot(d["k"], y, "-" + MARK[i], color=CAT[i], mec="white", mew=0.6, lw=1.2)[0])
        ls.append(f"tau {tau:g} s")
        kk2 = np.logspace(np.log10(0.3), np.log10(30), 100)
        T = th.t_eff(a, 0.1)
        od = kk2 * T <= 0.25
        ax.plot(kk2[od], -th.undershoot_deadzone(kk2[od], a, 0.1, 0.025) * 100, color=CAT[i], lw=0.9, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("attraction gain k (1/s), dt = 0.1, v_s = 0.025")
    ax.set_ylabel("rest point along approach (cm); < 0 short")
    ax.set_ylim(-9, 4)
    ax.set_title("(c) Undershoot (dotted: closed form)")
    legend_and_labels(ax, hs, ls, "lower right")

    ax = axs[3]
    hs, ls = [], []
    allm, allp = [], []
    for caps, mfc in (("off", None), ("hw", "white")):
        for r in sel(S, exp="E8o" + caps).itertuples():
            M = load.trials(rows[r.id])
            ok = (M["escaped"] == 0) & np.isfinite(M["mean_r"])
            if ok.mean() < 0.5:
                continue
            rs = M["mean_r"][ok]
            off = np.median(np.abs(r.r0 - rs))
            if off < 0.1:
                continue
            if r.r0 > 0.3:
                ov = np.maximum(0, rs - M["r_tmin"][ok]) / np.maximum(r.r0 - rs, 1e-6)
            else:
                ov = np.maximum(0, M["r_tmax"][ok] - rs) / np.maximum(rs - r.r0, 1e-6)
            p = float(th.overshoot_ratio_discrete(r.kr, r.alpha, r.dt))
            i = 0 if r.r0 < 0.3 else 1
            ax.plot(p * 100, np.median(ov) * 100, MARK[i], color=CAT[i], mfc=mfc or CAT[i],
                    mec=CAT[i] if mfc else "white", mew=0.8, ms=5)
            if caps == "off":
                allm.append(np.median(ov))
                allp.append(p)
    hs = [plt.Line2D([], [], marker=MARK[0], color=CAT[0], ls="none"),
          plt.Line2D([], [], marker=MARK[1], color=CAT[1], ls="none"),
          plt.Line2D([], [], marker="o", color=INK2, mfc="white", ls="none")]
    ls = ["start inside (r0 = 0.05)", "start outside (r0 = 0.6)", "hardware caps (hollow)"]
    ax.plot([0, 60], [0, 60], color=INK2, lw=1, ls="--")
    ax.set_xlabel("sampled-loop theory, k -> k_r (%)")
    ax.set_ylabel("simulated radial overshoot (%)")
    ax.set_title("(d) Orbit radial transient, |r0 - r*| >= 0.1 m")
    legend_and_labels(ax, hs, ls, "upper left")
    if allm:
        dd = np.abs(np.array(allm) - np.array(allp)) * 100
        log(f"E8o radial overshoot vs sampled theory (caps off, offset >= 0.1 m): n={len(dd)}, "
            f"median |diff| {np.median(dd):.1f} pp, 90% {np.percentile(dd, 90):.1f} pp, corr {np.corrcoef(allm, allp)[0, 1]:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_overshoot_undershoot.png"))
    plt.close(fig)

    # numbers: discrete overshoot theory vs simulation, all caps-off cells without stiction
    err = []
    for r in sel(S, exp="E8aoff", vs=0.0).itertuples():
        M = load.trials(rows[r.id])
        ok = M["escaped"] == 0
        if ok.mean() < 0.99:
            continue
        p = float(th.overshoot_ratio_discrete(r.k, r.alpha, r.dt, steps=8000))
        if np.isfinite(p):
            err.append(abs(np.median((M["overshoot"] / M["start_dist"])[ok]) - p))
    log(f"E8a overshoot ratio, discrete theory vs simulation (caps off, no stiction): n={len(err)} cells, "
        f"max |diff| {np.max(err) * 100:.3f} percentage points")
    unst = []
    for r in sel(S, exp="E8aoff", vs=0.0).itertuples():
        M = load.trials(rows[r.id])
        unst.append((r.k > th.k_max(r.alpha, r.dt), M["escaped"].mean() > 0.5))
    unst = np.array(unst)
    log(f"E8a caps off: k > k_max predicts divergence in {np.mean(unst[:, 0] == unst[:, 1]) * 100:.1f}% of {len(unst)} cells")


def fig9(S, res, out, log):
    rows = {r["id"]: r for r in load.manifest(res)}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    hs, ls = [], []
    for i, vm in enumerate((0.1, 0.3, 1.0, 3.0, 1e9)):
        d = sel(S, exp="E9a", vmax=vm, alpha=0.7).sort_values("k")
        y, esc = [], []
        for r in d.itertuples():
            M = load.trials(rows[r.id])
            fe = np.where(M["escaped"] > 0, np.inf, M["final_err"])
            y.append(np.median(fe))
            esc.append(M["escaped"].mean())
        y = np.array(y)
        hs.append(ax.plot(d["k"], np.where(np.isfinite(y), y, 2.0), "-" + MARK[i], color=SEQ[min(i, 4)] if i < 4 else CAT[7],
                          mec="white", mew=0.6, lw=1.3)[0])
        ls.append("no cap" if vm > 1e8 else f"v_max {vm:g} m/s")
        log(f"E9a alpha .7 v_max {vm:g}: median final err by k " + ", ".join(
            f"{k:g}: {'escaped' if not np.isfinite(v) else f'{v * 1000:.2f} mm'}" for k, v in zip(d["k"], y)))
    ax.axvline(float(th.k_max(0.7, 0.1)), color=CAT[7], lw=1, ls="--")
    ax.text(float(th.k_max(0.7, 0.1)) * 0.6, 1.2, "k_max", color=INK, fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=3)
    ax.set_xlabel("attraction gain k (1/s), alpha 0.7, v_s 0.025")
    ax.set_ylabel("median final error (m); top row = diverged")
    ax.set_title("(a) Speed cap turns instability into a cycle")
    legend_and_labels(ax, hs, ls, "upper left")

    ax = axs[1]
    xs, ys = [], []
    for r in sel(S, exp="E9a").itertuples():
        if r.vmax > 1e8 or r.k < 2 * float(th.k_max(r.alpha, 0.1)):
            continue
        M = load.trials(rows[r.id])
        xs.append(r.vmax * 0.1)
        ys.append(np.median(M["final_err"]))
    ax.loglog(xs, ys, "o", color=CAT[0], mec="white", mew=0.6, ms=6, label="k >= 2 k_max")
    xx = np.array([0.005, 0.5])
    ax.loglog(xx, xx / 2, color=INK2, lw=1, ls="--", label="v_max dt / 2")
    ax.set_xlabel("v_max dt (m)")
    ax.set_ylabel("median final distance (m)")
    ax.set_title("(b) Limit-cycle size above k_max")
    ax.legend(loc="upper left", fontsize=7.5)
    if xs:
        r_ = np.array(ys) / (np.array(xs) / 2)
        log(f"E9a limit cycle: final distance / (v_max dt / 2) median {np.median(r_):.2f}, range {r_.min():.2f}-{r_.max():.2f} (n={len(r_)})")

    ax = axs[2]
    hs, ls = [], []
    for i, vm in enumerate((0.3, 1.0, 3.0, 1e9)):
        d = sel(S, exp="E9o", vmax=vm, rd=1.0, rclamp=1e9).sort_values("kt")
        y = np.where(d["ok"] > 0.5, d["err_mean"], np.nan)
        hs.append(ax.plot(d["kt"], y, MARK[i], color=CAT[i], mec="white", mew=0.6, ms=6)[0])
        ls.append("no cap" if vm > 1e8 else f"v_max {vm:g}")
        kt = np.linspace(0.1, 2.0, 100)
        ax.plot(kt, th.orbit_r_small_angle(1.0, 0.7, 0.1, kt, 0.5, vm) - 1.0, color=CAT[i], lw=1)
        log(f"E9o r_d 1 v_max {vm:g}: err by k_t " + ", ".join(f"{a:g}: {b:+.4f} (pred {th.orbit_r_small_angle(1.0, .7, .1, a, .5, vm) - 1:+.4f})"
                                                           for a, b in zip(d["kt"], y)))
    ax.set_yscale("log")
    ax.set_xlabel("tangential speed k_t (m/s), r_d = 1 m")
    ax.set_ylabel("mean radius error (m)")
    ax.set_title("(c) Orbit with raised caps (lines: closed form)")
    legend_and_labels(ax, hs, ls, "upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig9_speed_limits.png"))
    plt.close(fig)


def run(S, res, out):
    lines = []

    def log(s):
        print(s)
        lines.append(s)
    fig7(S, res, out, log)
    fig8(S, res, out, log)
    fig9(S, res, out, log)
    open(os.path.join(HERE, res, "v2_numbers.txt"), "w").write("\n".join(lines) + "\n")
