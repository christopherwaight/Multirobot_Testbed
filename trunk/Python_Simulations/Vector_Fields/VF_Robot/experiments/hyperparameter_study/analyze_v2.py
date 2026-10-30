"""
Version 2 and 3 figures (called from analyze.py --res results_v2 or results_v3):
  fig7  stiction models across sample periods, and against hardware
  fig8  overshoot and undershoot as functions of k, alpha (tau) and dt
  fig9  raised speed limits
and the numbers quoted in hyperparameter_study.md (v2_numbers.txt, v3_numbers.txt).
Version 3 adds the Coulomb model, which is then the main model.
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
          ("deadzone", "command dead zone (ZOH exact)"), ("coulomb", "Coulomb friction behind the lag")]
# hardware_*_starts_finals.npy hold OptiTrack metres divided by the printed map
# size (66 in); see stiction_hw_check.py.
MAP_SCALE = 1.6764


def models(S):
    return [(m, lab) for m, lab in MODELS if (S["stiction"] == m).any()]


def main_model(S):
    return "coulomb" if (S["stiction"] == "coulomb").any() else "deadzone"


def _trials(res, rid):
    rows = {r["id"]: r for r in load.manifest(res)}
    return load.trials(rows[rid])


def hardware():
    out = {}
    for f in ("vortex", "saddle"):
        a = np.load(os.path.join(HERE, f"hardware_{f}_starts_finals.npy"))
        out[f] = np.hypot(a[:, 2], a[:, 3]) * MAP_SCALE
    return out


def fig7(S, res, out, log):
    rows = {r["id"]: r for r in load.manifest(res)}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    hs, ls = [], []
    for i, (m, lab) in enumerate(models(S)):
        for j, k in enumerate((1.0, 10.0)):
            d = sel(S, exp="E10", task="attract", stiction=m, k=k, vs=0.025).sort_values("dt")
            meas = [np.mean(load.trials(rows[r])["final_err"]) for r in d["id"]]
            pred = []
            for r in d.itertuples():
                M = load.trials(rows[r.id])
                fe, _, _ = th.reduced_1d(M["start_dist"], k, r.alpha, r.dt, 0.025, r.vmax, r.vcmd_max, 60.0, model=m)
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
    fig.legend(hs, ls, loc="upper center", ncol=len(hs), fontsize=7.5, bbox_to_anchor=(0.36, 1.07))

    ax = axs[1]
    hs, ls = [], []
    for i, (m, lab) in enumerate(models(S)):
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
    mm = main_model(S)
    for i, (m, lab) in enumerate(models(S)):
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
        if m == mm:
            ax.plot(v2 * 100, np.arange(1, len(v2) + 1) / len(v2) * 100, color=CAT[i], lw=1.2, ls=":",
                    label=f"sim, {m}, reconstructed fields")
        log(f"E10hw {m:8s} reconstructed fields: mean {v2.mean() * 1000:.1f} mm, median {np.median(v2) * 1000:.1f}")
    ax.set_xlim(0, 6.5)
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
        if (d["stiction"] == "coulomb").all() and len(d):
            kk2 = kk2[kk2 < th.k_max(a, 0.1)]
            d0 = load.trials(rows[d["id"].values[0]])["start_dist"][:400]
            rp = th.rest_point_coulomb(d0[None, :], kk2[:, None], a, 0.1, 0.025)
            ax.plot(kk2, -np.median(rp, 1) * 100, color=CAT[i], lw=0.9, ls=":")
            pr = [np.median(th.rest_point_coulomb(load.trials(rows[r_])["start_dist"], k_, a, 0.1, 0.025))
                  for r_, k_ in zip(d["id"], d["k"])]
            log(f"E8a Coulomb rest point tau {tau:g}: sim " + ", ".join(f"k {k_:g}: {y_:+.2f} cm" for k_, y_ in zip(d["k"], y)) +
                " | closed form " + ", ".join(f"{-p_ * 100:+.2f}" for p_ in pr))
        else:
            T = th.t_eff(a, 0.1)
            od = kk2 * T <= 0.25
            ax.plot(kk2[od], -th.undershoot_deadzone(kk2[od], a, 0.1, 0.025) * 100, color=CAT[i], lw=0.9, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("attraction gain k (1/s), dt = 0.1, " +
                  ("v_c = 0.025" if (D2["stiction"] == "coulomb").all() else "v_s = 0.025"))
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
        a0 = d["alpha"].iloc[0]
        vc0 = d["vs"].iloc[0] if (d["stiction"] == "coulomb").all() else 0.0
        ax.plot(kt, th.orbit_r_small_angle(1.0, a0, 0.1, kt, 0.5, vm, vc0) - 1.0, color=CAT[i], lw=1)
        log(f"E9o r_d 1 v_max {vm:g}: err by k_t " + ", ".join(
            f"{a:g}: {b:+.4f} (pred {th.orbit_r_small_angle(1.0, a0, .1, a, .5, vm, vc0) - 1:+.4f})"
            for a, b in zip(d["kt"], y)))
    ax.set_yscale("log")
    ax.set_xlabel("tangential speed k_t (m/s), r_d = 1 m")
    ax.set_ylabel("mean radius error (m)")
    ax.set_title("(c) Orbit with raised caps (lines: closed form)")
    legend_and_labels(ax, hs, ls, "upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig9_speed_limits.png"))
    plt.close(fig)


HW_ORBITS = {"vortex": [("orbit001", .01), ("orbit010", .1), ("orbit020", .2), ("orbit030", .3), ("orbit040", .4),
                        ("orbit050", .5)],
              "saddle": [("radius001", .01), ("radius01", .1), ("radius02", .2), ("radius03", .3), ("radius04", .4),
                         ("radius05", .5)]}


def hardware_orbits():
    """Hardware 3-robot orbit runs (trunk/robots_3, metres, 60 s at 0.1 s). Returns per field the
    radius error over the full record (the paper's table), after 20 s (as the simulation), the
    within-run sd after 20 s and the tangential speed after 20 s."""
    import scipy.io as sio
    import warnings
    root = os.path.join(HERE, *[".."] * 5, "robots_3")
    out = {}
    for f, files in HW_ORBITS.items():
        rows = []
        for fn, rd in files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = sio.loadmat(os.path.join(root, f"{f}_tests", fn + ".mat"))
            c = d["cluster_position"][:, :2] if "cluster_position" in d else d["cluster_pose"][:, :2]
            t = d["tout"][:, 0]
            late = t >= 20
            r = np.hypot(*c.T)
            om = np.polyfit(t[late], np.unwrap(np.arctan2(c[:, 1], c[:, 0]))[late], 1)[0]
            rows.append(dict(rd=rd, full=r.mean() - rd, late=r[late].mean() - rd, sd=r[late].std(),
                             V=abs(om) * r[late].mean(), T=t[-1]))
        out[f] = rows
    return out


def fig10(res, out, log):
    """E11 robot-model sensitivity (sensitivity_v3.py) and the hardware orbit runs."""
    import pandas as pd
    path = os.path.join(HERE, "results_v3_sens", "sens.csv")
    if not os.path.exists(path):
        return
    E = pd.read_csv(path)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    H = E[E["kind"] == "hw"]
    hw = E[E["kind"] == "hardware"].set_index("field")
    for i, (f, lab) in enumerate((("vortex", "linear fields"), ("vortex_nn", "vortex (recon.)"),
                                  ("saddle_nn", "saddle (recon.)"))):
        for j, tau in enumerate((0.20, 0.28)):
            d = H[(H["field"] == f) & (np.abs(H["tau"] - tau) < 1e-9)].sort_values("vk")
            d = d.groupby("vk", as_index=False)["all_mean"].mean()
            ax.plot(d["vk"], d["all_mean"] * 1000, MARK[i] if j == 0 else "-", color=CAT[i], mec="white", mew=0.6,
                    lw=1.2, ms=6, label=lab if j == 0 else None)
    vk = np.linspace(0, 0.035, 2)
    ax.plot(vk, vk * 1000, color=INK2, lw=0.8, ls="--")
    ax.text(0.031, 28.5, "v_k / k", color=INK2, fontsize=7)
    hv, hsd = hw.loc["vortex", "all_mean"] * 1000, hw.loc["saddle", "all_mean"] * 1000
    ax.axhline((hv + hsd) / 2, color=INK, lw=1, ls=":")
    ax.text(0.0135, (hv + hsd) / 2 - 2.6, f"hardware: vortex {hv:.1f},\nsaddle {hsd:.1f}", color=INK, fontsize=6.5)
    ax.set_xlabel("kinetic friction v_k (m/s), k = 1")
    ax.set_ylabel("mean final distance (mm), 157 hardware starts")
    ax.set_title("(a) Rest distance is set by v_k")
    ax.legend(loc="lower right", fontsize=7)
    log("E11 hardware starts, mean final distance (mm) by field, tau, v_c, v_k:\n" + "\n".join(
        f"   {r.field:10s} tau {r.tau:.2f} v_c {r.vc:.3f} v_k {r.vk:.4f}: mean {r.all_mean * 1000:5.1f}, bias {r.bias * 1000:5.1f}, "
        f"precision {r.prec * 1000:5.1f}, along-approach median {r.proj_med * 1000:+6.1f}"
        for r in H.sort_values(["field", "tau", "vc", "vk"]).itertuples()))
    log("hardware (metres): " + "; ".join(f"{f} mean {hw.loc[f, 'all_mean'] * 1000:.1f}, bias {hw.loc[f, 'bias'] * 1000:.1f}, "
                                          f"precision {hw.loc[f, 'prec'] * 1000:.1f}, along-approach median "
                                          f"{hw.loc[f, 'proj_med'] * 1000:+.1f} mm" for f in ("vortex", "saddle")))

    ax = axs[1]
    O = E[(E["kind"] == "orbit") & (E["field"] == "vortex") & (np.abs(E["vc"] - 0.025) < 1e-9)]
    hs, ls = [], []
    rr = np.linspace(0.01, 0.5, 100)
    for i, (tau, vk_) in enumerate(((0.28, 0.0), (0.28, 0.025), (0.20, 0.0), (0.20, 0.025))):
        d = O[(np.abs(O["tau"] - tau) < 1e-9) & (np.abs(O["vk"] - vk_) < 1e-9)].sort_values("rd")
        a = np.exp(-0.1 / tau)
        ax.plot(rr, th.orbit_r_small_angle(rr, a, 0.1, 0.2, 0.5, 0.4, vk_) - rr, color=SEQ[4 - i], lw=1)
        hs.append(ax.plot(d["rd"], d["err_mean"], MARK[i], color=SEQ[4 - i], mec="white", mew=0.6, ms=6)[0])
        ls.append(f"tau {tau:g} s, v_k {vk_:g}")
    ax.set_xlabel("commanded radius r_d (m), vortex, plane fit")
    ax.set_ylabel("mean radius error (m)")
    ax.set_title("(b) Orbit error vs tau and v_k (lines: closed form)")
    legend_and_labels(ax, hs, ls, "upper right")
    W = E[E["kind"].isin(["orbit", "sweep"]) & E["field"].isin(vs.LINEAR)].copy()
    W["pred_err"] = [th.orbit_r_small_angle(r.rd, r.alpha, 0.1, 0.2, 0.5, 0.4, r.vk) - r.rd for r in W.itertuples()]
    gap = (W["err_mean"] - W["pred_err"]).abs()
    log(f"E11 orbit closed form with V = min(k_t - v_k, v_max), linear field: n={len(W)}, median |gap| "
        f"{gap.median() * 1000:.2f} mm, max {gap.max() * 1000:.2f} mm; start condition k_t > v_c held in "
        f"{((W['nostart'] < 0.5) == (W['pred_start'] > 0.5)).mean() * 100:.0f}% of cells")

    ax = axs[2]
    HO = hardware_orbits()
    for i, f in ((1, "vortex"), (2, "saddle")):     # colors as panel (a)
        rows = HO[f]
        ax.plot([r["rd"] for r in rows], [r["late"] for r in rows], MARK[i], color=CAT[i], ms=7, mec="white",
                mew=0.6, label=f"hardware {f}")
        for j, tau in enumerate((0.28, 0.20)):
            d = E[(E["kind"] == "orbit") & (E["field"] == f + "_nn") & (np.abs(E["tau"] - tau) < 1e-9) &
                  (np.abs(E["vc"] - 0.025) < 1e-9) & (np.abs(E["vk"] - 0.025) < 1e-9)].sort_values("rd")
            ax.plot(d["rd"], d["err_mean"], color=CAT[i], lw=1.3, ls="-" if j == 0 else "--",
                    label=f"sim {f} (recon.), tau {tau:g}")
        log(f"hardware orbit {f}: " + "; ".join(
            f"r_d {r['rd']:g}: full record {r['full']:+.3f}, after 20 s {r['late']:+.3f} (sd {r['sd']:.3f}), V {r['V']:.3f} m/s"
            for r in rows) + f"; run length {rows[0]['T']:.0f} s")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("commanded radius r_d (m)")
    ax.set_ylabel("mean radius error (m), after 20 s")
    ax.set_title("(c) Hardware orbits vs model (v_k 0.025)")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig10_robot_model.png"))
    plt.close(fig)


def run(S, res, out):
    lines = []

    def log(s):
        print(s)
        lines.append(s)
    fig7(S, res, out, log)
    fig8(S, res, out, log)
    fig9(S, res, out, log)
    if res.startswith("results_v3"):
        fig10(res, out, log)
    tag = "v3" if res.startswith("results_v3") else "v2"
    open(os.path.join(HERE, res, f"{tag}_numbers.txt"), "w").write("\n".join(lines) + "\n")
