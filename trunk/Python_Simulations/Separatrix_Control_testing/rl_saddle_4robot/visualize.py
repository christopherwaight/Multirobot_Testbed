"""
Figures for the 4-robot PPO saddle-seeking experiment.

    --estimator    why the formation has to rotate, and what boxes the rate in
    --gallery      the six randomized field families
    --training     PPO learning curves with stage boundaries
    --comparison   final distance by controller, and by field family
    --rollout      trajectories over the field
    --mechanism    what the policy does about the blind spot and saturation
    --tradeoff     formation size against achieved rotation rate
    --anim         one rollout as a GIF
    --all          everything for which the inputs exist

Colours come from a validated categorical palette, assigned in fixed slot order
and never cycled.  Scalar fields use a two-hue diverging map centred on the
saddle value, never a rainbow.  Every series is direct-labelled or legended, so
identity never rests on colour alone.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import saddle_fields as sf
from estimator import (plane_fit_estimate, square_positions,
                       eigenframe_misalignment)

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Validated categorical palette, fixed slot order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
     "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"

V_MAX, STICTION, ALPHA = 0.3, 0.025, 0.7


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": MUTED,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "font.size": 9,
    })


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def omega_bounds(R):
    """Feasible rotation band. Both bounds scale as 1/R.

    Upper: tangential speed omega*R must not exceed v_max.
    Lower: one momentum-filter step realizes only (1-alpha) of the command, and
    anything below the stiction threshold is zeroed before it can accumulate,
    so a command under stiction/((1-alpha) R) yields no rotation at all.
    """
    R = np.asarray(R, float)
    return STICTION / ((1.0 - ALPHA) * R), V_MAX / R


def field_grid(fld, n=140, pad=0.0):
    (x0, x1), (y0, y1) = fld.domain_bounds()
    xs = np.linspace(x0 + pad, x1 - pad, n)
    ys = np.linspace(y0 + pad, y1 - pad, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(fld.phi)(X, Y)
    return X, Y, Z


def draw_field(ax, fld, n=140, levels=28):
    X, Y, Z = field_grid(fld, n)
    z0 = fld.phi(*fld.saddle)
    m = max(abs(np.nanmin(Z) - z0), abs(np.nanmax(Z) - z0), 1e-9)
    ax.contourf(X, Y, Z, levels=levels, cmap="RdBu_r",
                vmin=z0 - m, vmax=z0 + m, alpha=0.85)
    ax.contour(X, Y, Z, levels=levels, colors=MUTED,
               linewidths=0.35, alpha=0.5)
    ax.plot(*fld.saddle, marker="x", color=INK, markersize=11,
            markeredgewidth=2.2, zorder=12)
    ax.set_aspect("equal")
    ax.grid(False)


# --------------------------------------------------------------------------
# 1. Estimator mechanism
# --------------------------------------------------------------------------

def fig_estimator():
    """Why the formation must rotate, and what boxes the rotation rate in."""
    _style()
    th_field = np.deg2rad(35.0)
    Rf = np.array([[np.cos(th_field), -np.sin(th_field)],
                   [np.sin(th_field), np.cos(th_field)]])
    H_true = Rf @ np.diag([3.0, -1.0]) @ Rf.T
    c = np.array([0.35, 0.20])

    def phi(x, y):
        r = np.array([x, y])
        return 0.5 * r @ H_true @ r

    g_true = H_true @ c
    step_true = -np.linalg.solve(H_true, g_true)
    ang_true = np.arctan2(step_true[1], step_true[0])

    ths = np.linspace(0, 180, 361)
    mags, dir_err = [], []
    for td in ths:
        xy = square_positions(c, 0.15, np.deg2rad(td))
        z = np.array([phi(p[0], p[1]) for p in xy])
        H, g = plane_fit_estimate(xy, z)
        mags.append(np.hypot(H[0, 0], H[0, 1]))
        s = -np.linalg.pinv(H) @ g
        if np.linalg.norm(s) < 1e-12:
            dir_err.append(np.nan)
        else:
            a = np.arctan2(s[1], s[0]) - ang_true
            dir_err.append(abs(np.degrees(np.arctan2(np.sin(a), np.cos(a)))))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # (a) sensing gain
    ax = axes[0]
    ax.plot(ths, mags, color=C[0], label="|m|, recovered curvature scale")
    for k in (0, 1):
        b = np.rad2deg(th_field) + 45 + 90 * k
        if b <= 180:
            ax.axvline(b, color=C[7], linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(b + 2, ax.get_ylim()[1] * 0.92, "blind spot",
                    color=C[7], fontsize=8, rotation=90, va="top")
    ax.axvline(np.rad2deg(th_field), color=MUTED, linewidth=1.0, alpha=0.9)
    ax.text(np.rad2deg(th_field) + 2, max(mags) * 0.05, "field axis",
            color=INK2, fontsize=8, rotation=90)
    ax.set_xlabel("formation angle (deg)")
    ax.set_ylabel("|m|")
    ax.set_title("a.  Only one scalar survives, and it vanishes\n"
                 "at 45 deg of misalignment", loc="left")
    ax.legend(loc="upper right")

    # (b) direction error
    ax = axes[1]
    ax.plot(ths, dir_err, color=C[1], label="angle between estimated and\n"
                                            "true Newton direction")
    ax.axhline(90, color=MUTED, linewidth=1.0, linestyle=":")
    ax.text(3, 93, "orthogonal to the correct move", color=INK2, fontsize=8)
    ax.set_xlabel("formation angle (deg)")
    ax.set_ylabel("direction error (deg)")
    ax.set_ylim(-5, 185)
    ax.set_title("b.  The step DIRECTION depends on the\n"
                 "formation angle, not just its magnitude", loc="left")
    ax.legend(loc="upper right")

    # (c) feasible rotation band
    ax = axes[2]
    Rs = np.linspace(0.06, 0.45, 300)
    lo, hi = omega_bounds(Rs)
    ax.fill_between(Rs, lo, hi, color=C[2], alpha=0.20, linewidth=0)
    ax.plot(Rs, hi, color=C[2], label=r"$\omega_{max}=v_{max}/R$")
    ax.plot(Rs, lo, color=C[3], label=r"$\omega_{min}$ (stiction dead zone)")
    ax.text(0.30, np.interp(0.30, Rs, (lo + hi) / 2) * 1.02, "feasible",
            color=INK2, fontsize=9, ha="center")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.set_xlabel("formation ring radius R (m)")
    ax.set_ylabel("rotation rate (rad/s)")
    ax.set_title("c.  Both bounds scale as 1/R, so the band is\n"
                 "3.6x wide at every size", loc="left")
    ax.legend(loc="upper right")

    fig.suptitle("The 4-robot Hessian estimate, and what the plant allows",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig1_estimator_mechanism.png")


# --------------------------------------------------------------------------
# 2. Field gallery
# --------------------------------------------------------------------------

def fig_gallery(draws=2, seed=7):
    _style()
    rng = np.random.default_rng(seed)
    fams = sf.FAMILY_NAMES
    fig, axes = plt.subplots(draws, len(fams),
                             figsize=(3.05 * len(fams), 3.25 * draws))
    axes = np.atleast_2d(axes)
    for j, fam in enumerate(fams):
        for i in range(draws):
            ax = axes[i, j]
            fld = sf.FAMILIES[fam](rng)
            draw_field(ax, fld, n=110, levels=24)
            for r in (sf.START_R_MIN, sf.START_R_MAX):
                ax.add_patch(plt.Circle(fld.saddle, r, fill=False,
                                        color=INK2, linestyle=":",
                                        linewidth=0.9, alpha=0.7))
            lam = np.sort(fld.eigvals)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(fam, loc="center", fontsize=10, color=INK)
            ax.set_xlabel(f"eig ({lam[0]:+.2f}, {lam[1]:+.2f})",
                          fontsize=7.5, color=INK2)
    handles = [
        Line2D([], [], color=INK, marker="x", linestyle="none",
               markersize=9, markeredgewidth=2, label="true saddle"),
        Line2D([], [], color=INK2, linestyle=":", label="episode start annulus"),
    ]
    fig.suptitle(f"{len(fams)} randomized field families, every saddle known in closed form",
                 fontsize=13, fontweight="bold")
    # tight_layout does not reserve space for a figure-level legend, so the
    # room is made explicitly afterwards.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.11 if draws > 1 else 0.20)
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005))
    return _save(fig, "fig2_field_gallery.png")


# --------------------------------------------------------------------------
# 3. Training curves
# --------------------------------------------------------------------------

def fig_training(path=os.path.join(OUT_DIR, "ppo_history.jsonl")):
    if not os.path.exists(path):
        print(f"  skipping training curves, no {path}")
        return None
    _style()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("  skipping training curves, history is empty")
        return None

    t = np.array([r["timesteps"] for r in rows])
    stage = np.array([r["stage"] for r in rows])
    series = [("ep_rew_mean", "mean episode return", C[0]),
              ("success_rate", "success rate", C[1]),
              ("ep_len_mean", "mean episode length (steps)", C[2])]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
    for ax, (key, label, col) in zip(axes, series):
        y = np.array([r.get(key, np.nan) for r in rows], float)
        ax.plot(t / 1e6, y, color=col, label=label)
        for s in np.unique(stage)[1:]:
            ax.axvline(t[stage == s][0] / 1e6, color=MUTED,
                       linewidth=1.0, linestyle="--", alpha=0.9)
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel(label)
        ax.legend(loc="lower right")
        if key == "success_rate":
            ax.set_ylim(-0.03, 1.03)
    axes[0].set_title("a.  return", loc="left")
    axes[1].set_title("b.  success rate", loc="left")
    axes[2].set_title("c.  episode length", loc="left")
    fig.suptitle("PPO training. Dashed rules are curriculum stage boundaries "
                 "(quadratic only, then three families, then all six)",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig3_training.png")


# --------------------------------------------------------------------------
# 4. Controller comparison
# --------------------------------------------------------------------------

def _short(label):
    return (label.replace("rot-hessian ", "rot-Hess ")
                 .replace(", default gains", " (default)")
                 .replace(", 0-dynamics ceiling", " (0-dyn ceiling)")
                 .split(" (k_rot")[0])


def fig_comparison(path=os.path.join(OUT_DIR, "evaluation.json")):
    data = _load(path)
    if not data:
        print(f"  skipping comparison, no {path}")
        return None
    _style()
    runs = data["runs"]
    labels = list(runs)
    med = [runs[k]["summary"]["e_final_median"] for k in labels]
    order = np.argsort(med)[::-1]
    labels = [labels[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 0.62 * len(labels) + 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # -- median with interquartile range -------------------------------
    ax = axes[0]
    ypos = np.arange(len(labels))
    for i, k in enumerate(labels):
        s = runs[k]["summary"]
        col = C[i % 3] if "PPO" not in k else C[0]
        col = C[0] if "PPO" in k else (C[2] if "0-dyn" in k else C[1])
        ax.plot([s["e_final_p25"], s["e_final_p75"]], [i, i],
                color=col, linewidth=6, alpha=0.35, solid_capstyle="round")
        ax.plot([s["e_final_median"]], [i], marker="o", markersize=9,
                color=col, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(s["e_final_p75"] + 0.06, i,
                f"{s['e_final_median']:.2f}   succ {s['success_rate']:.0%}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0.15, color=INK, linestyle=":", linewidth=1.2)
    ax.text(0.16, len(labels) - 0.35, "success tolerance 0.15 m",
            fontsize=8, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(k) for k in labels], fontsize=8.5)
    ax.set_xlabel("final distance to true saddle (m), median and IQR")
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.set_title("a.  Held-out performance. Marker is the median, "
                 "bar is the interquartile range", loc="left")

    # -- per family ----------------------------------------------------
    ax = axes[1]
    fams = [f for f in sf.FAMILY_NAMES
            if any(f in runs[k]["per_family"] for k in labels)]
    x = np.arange(len(fams))
    # The comparison that matters is the best analytic law against the best
    # policies. Ordering by whatever happens to sort first pulled in the
    # default-gain and 0-dynamics comparators instead, which are already
    # covered in panel a.
    best_analytic = [k for k in labels if "BEST gains" in k][:1]
    ppo_runs = sorted([k for k in labels if "PPO" in k],
                      key=lambda k: runs[k]["summary"]["e_final_median"])[:3]
    show = best_analytic + ppo_runs
    w = 0.8 / max(len(show), 1)
    for i, k in enumerate(show):
        vals = [runs[k]["per_family"].get(f, {}).get("e_final_median", np.nan)
                for f in fams]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w * 0.86,
               color=C[i], label=_short(k), edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(0.15, color=INK, linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7.5)
    ax.set_ylabel("median final distance (m)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("b.  By field family", loc="left")

    fig.suptitle(f"Controller comparison, {data['n_eval']} held-out fields, "
                 "identical plant and reward throughout",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig4_comparison.png")


# --------------------------------------------------------------------------
# 4b. Paired difference against the analytic law
# --------------------------------------------------------------------------

def fig_paired(path=os.path.join(OUT_DIR, "evaluation.json"), n_boot=10000):
    """Forest plot of per-field success difference vs the analytic baseline.

    Every controller is scored on identical fields, so the meaningful quantity
    is the paired difference, not two independent rates.  Field difficulty
    dominates the variance here; pairing removes it.
    """
    data = _load(path)
    if not data:
        print(f"  skipping paired plot, no {path}")
        return None
    runs = data["runs"]
    ref_key = next((k for k in runs if "BEST gains" in k), None)
    if ref_key is None or "per_episode" not in runs[ref_key]:
        print("  skipping paired plot, no per-episode data")
        return None
    _style()
    rng = np.random.default_rng(0)

    # Align on SEED, not array position.  Runs were scored at different n
    # (1000 vs 2500), and index-aligning would silently drop every run whose
    # length differs from the reference.  Each comparison then uses whatever
    # fields the two runs actually share.
    ref_pe = runs[ref_key]["per_episode"]
    ref_map = dict(zip(ref_pe["seed"], ref_pe["success"]))

    def paired(v):
        pe = v["per_episode"]
        common = [(ref_map[s], ok) for s, ok in zip(pe["seed"], pe["success"])
                  if s in ref_map]
        if not common:
            return None, None
        a = np.array([c[0] for c in common], float)
        b = np.array([c[1] for c in common], float)
        return a, b

    rows, n_used = [], 0
    for k, v in runs.items():
        if k == ref_key or "per_episode" not in v:
            continue
        a, s = paired(v)
        if a is None:
            continue
        n_used = max(n_used, len(a))
        d = s - a
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        rows.append((f"{k}  (n={len(d)})", d.mean(),
                     *np.percentile(boot, [2.5, 97.5]),
                     (boot > 0).mean(), s.mean()))
    if not rows:
        return None
    # oracle: either controller succeeds, the upper bound on any switching rule
    best_key = max((k for k in runs if k != ref_key and "per_episode" in runs[k]),
                   key=lambda k: np.mean(runs[k]["per_episode"]["success"]))
    a, s_best = paired(runs[best_key])
    orc = np.maximum(a, s_best) - a
    idx = rng.integers(0, len(orc), size=(n_boot, len(orc)))
    bo = orc[idx].mean(axis=1)
    rows.append(("ORACLE: either succeeds (upper bound)", orc.mean(),
                 *np.percentile(bo, [2.5, 97.5]), (bo > 0).mean(),
                 np.maximum(a, s_best).mean()))
    ref = np.zeros(n_used)  # only used for the title's field count

    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 2.6))
    for i, (k, mu, lo, hi, p, succ) in enumerate(rows):
        oracle = k.startswith("ORACLE")
        col = C[2] if oracle else (C[0] if mu > 0 else C[1])
        ax.plot([lo, hi], [i, i], color=col, linewidth=6, alpha=0.35,
                solid_capstyle="round")
        ax.plot([mu], [i], marker="o", markersize=10, color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(hi + 0.004, i, f"{succ:.1%}   P(better)={p:.0%}",
                va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK, linestyle=":", linewidth=1.4)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_short(r[0]) for r in rows], fontsize=9)
    ax.set_xlabel("paired difference in success rate vs the tuned analytic law "
                  "(positive = better)")
    ax.grid(axis="y", visible=False)
    n = len(ref)
    ax.set_title(f"Marker is the mean paired difference, bar is the 95% "
                 f"bootstrap CI over {n} fields.\n"
                 f"A bar crossing the dotted line is a tie.", loc="left")
    fig.suptitle(f"Head-to-head against the analytic law, {n} paired held-out fields",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig9_paired_difference.png")


# --------------------------------------------------------------------------
# Log helpers
# --------------------------------------------------------------------------

def _rebuild_field(log):
    """Reconstruct the exact field a logged episode ran on."""
    rng = np.random.default_rng(log["seed"])
    return sf.sample_field(rng, log.get("families"))


def _pick_runs(logs, prefer=("PPO", "BEST gains", "single, default")):
    """Order runs so the interesting comparison comes first.

    The do-nothing floor is useful in the aggregate table but makes a poor
    second trace in a time series, so it is pushed to the back.
    """
    keys = list(logs)
    out = []
    for p in prefer:
        for k in keys:
            if p in k and logs.get(k) and k not in out:
                out.append(k)
                break
    for k in keys:
        if k not in out and logs.get(k):
            out.append(k)
    return out


# --------------------------------------------------------------------------
# 5. Rollouts
# --------------------------------------------------------------------------

def fig_rollout(path=os.path.join(OUT_DIR, "evaluation_logs.json"), n=6):
    logs = _load(path)
    if not logs:
        print(f"  skipping rollouts, no {path}")
        return None
    _style()
    key = _pick_runs(logs)[0]
    eps = logs[key][:n]
    if not eps:
        print("  skipping rollouts, no episodes stored")
        return None

    ncol = 3
    nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, ep in zip(axes, eps):
        fld = _rebuild_field(ep)
        draw_field(ax, fld, n=120, levels=22)
        traj = np.asarray(ep["centroid"])
        ax.plot(traj[:, 0], traj[:, 1], color=C[0], linewidth=2.0,
                zorder=9, label="centroid path")

        robots = np.asarray(ep["robots"])
        stride = max(1, len(robots) // 9)
        for t in range(0, len(robots), stride):
            q = robots[t][[0, 1, 2, 3, 0]]
            ax.plot(q[:, 0], q[:, 1], color=C[1], linewidth=1.0,
                    alpha=0.55, zorder=8)
        q = robots[-1][[0, 1, 2, 3, 0]]
        ax.plot(q[:, 0], q[:, 1], color=C[1], linewidth=2.0, zorder=10,
                label="formation")

        ax.plot(*traj[0], marker="*", color=C[0], markersize=15,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=11)
        ax.plot(*traj[-1], marker="*", color=C[7], markersize=15,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ep['family']}", loc="left", fontsize=9.5)
        ax.set_xlabel(f"final distance {ep['e'][-1]:.3f} m, "
                      f"{len(traj)} steps", fontsize=8, color=INK2)

    for ax in axes[len(eps):]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], color=C[0], label="centroid path"),
        Line2D([], [], color=C[1], label="formation, every ~9th step"),
        Line2D([], [], color=C[0], marker="*", linestyle="none",
               markersize=12, label="start"),
        Line2D([], [], color=C[7], marker="*", linestyle="none",
               markersize=12, label="end"),
        Line2D([], [], color=INK, marker="x", linestyle="none",
               markersize=9, markeredgewidth=2, label="true saddle"),
    ]
    fig.suptitle(f"{key} rollouts on held-out fields",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10 if nrow > 1 else 0.16)
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.005))
    return _save(fig, "fig5_rollouts.png")


# --------------------------------------------------------------------------
# 6. Mechanism during a rollout
# --------------------------------------------------------------------------

def _misalignment_series(ep, fld):
    """Formation misalignment against the LOCAL field eigenframe.

    Computed from the true Hessian at each visited point rather than from the
    Hessian at the saddle, since the eigenframe rotates across the domain for
    every family except the pure quadratic.
    """
    out = []
    for c, th in zip(np.asarray(ep["centroid"]), ep["theta"]):
        H = sf.fd_hessian(fld.phi, c[0], c[1])
        out.append(np.degrees(eigenframe_misalignment(th, H)))
    return np.asarray(out)


def fig_mechanism(path=os.path.join(OUT_DIR, "evaluation_logs.json"), idx=0):
    logs = _load(path)
    if not logs:
        print(f"  skipping mechanism, no {path}")
        return None
    _style()
    keys = _pick_runs(logs)[:2]
    if not keys:
        return None

    fig, axes = plt.subplots(4, 1, figsize=(13.4, 11.4), sharex=True)
    for i, k in enumerate(keys):
        if idx >= len(logs[k]):
            continue
        ep = logs[k][idx]
        fld = _rebuild_field(ep)
        t = np.arange(len(ep["e"])) * 0.1
        col = C[i]
        lab = _short(k)

        axes[0].plot(t, ep["e"], color=col, label=lab)
        axes[1].plot(t, ep["radius"], color=col, label=lab)

        mis = np.abs(_misalignment_series(ep, fld))
        axes[2].plot(t, mis, color=col, label=lab)
        axes[3].plot(t, np.abs(ep["omega_ach"]), color=col, label=f"{lab}, achieved")
        axes[3].plot(t, np.abs(ep["omega_cmd"]), color=col, linewidth=1.0,
                     linestyle=":", alpha=0.8, label=f"{lab}, commanded")

        if i == 0:
            lo, hi = omega_bounds(np.asarray(ep["radius"]))
            axes[3].fill_between(t, lo, hi, color=C[2], alpha=0.16,
                                 linewidth=0, zorder=0)
            axes[3].plot(t, hi, color=C[2], linewidth=1.0, alpha=0.9,
                         label=r"$\omega_{max}=v_{max}/R(t)$")
            axes[3].plot(t, lo, color=C[3], linewidth=1.0, alpha=0.9,
                         label=r"$\omega_{min}$ dead zone")

    axes[0].axhline(0.15, color=INK, linestyle=":", linewidth=1.1)
    axes[0].set_ylabel("distance to saddle (m)")
    axes[0].set_title("a.  Does it get there", loc="left")

    axes[1].set_ylabel("formation radius R (m)")
    axes[1].set_title("b.  What it does with the size knob", loc="left")

    axes[2].axhline(45, color=C[7], linestyle="--", linewidth=1.2)
    axes[2].text(0.4, 46, "blind spot, estimate is zero here",
                 color=C[7], fontsize=8)
    axes[2].set_ylabel("|misalignment| (deg)")
    axes[2].set_ylim(0, 50)
    axes[2].set_title("c.  Misalignment against the local field eigenframe",
                      loc="left")

    axes[3].set_ylabel("|rotation rate| (rad/s)")
    axes[3].set_xlabel("time (s)")
    axes[3].set_yscale("log")
    axes[3].set_ylim(bottom=1e-2)
    axes[3].set_title("d.  Commanded against achieved rotation, with the band "
                      "the plant allows", loc="left")

    # Legends sit outside the axes: these traces run the full width, so an
    # inset legend lands on the data.
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0),
                  ncol=1, fontsize=7.5)

    fig.suptitle("What the controller does about the blind spot and the rate limit",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save(fig, "fig6_mechanism.png")


# --------------------------------------------------------------------------
# 7. Size against rotation
# --------------------------------------------------------------------------

def fig_tradeoff(path=os.path.join(OUT_DIR, "evaluation_logs.json")):
    logs = _load(path)
    if not logs:
        print(f"  skipping tradeoff, no {path}")
        return None
    _style()
    keys = _pick_runs(logs)[:2]

    fig, axes = plt.subplots(1, len(keys), figsize=(7.4 * len(keys), 5.4),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        R, W, T = [], [], []
        for ep in logs[k]:
            R.extend(ep["radius"])
            W.extend(np.abs(ep["omega_ach"]))
            T.extend(np.linspace(0, 1, len(ep["radius"])))
        R, W, T = np.asarray(R), np.asarray(W), np.asarray(T)

        rr = np.linspace(max(R.min() * 0.9, 0.05), R.max() * 1.05, 200)
        lo, hi = omega_bounds(rr)
        ax.fill_between(rr, lo, hi, color=C[2], alpha=0.15, linewidth=0)
        ax.plot(rr, hi, color=C[2], label=r"$\omega_{max}=v_{max}/R$")
        ax.plot(rr, lo, color=C[3], label=r"$\omega_{min}$ dead zone")

        sc = ax.scatter(R, np.maximum(W, 1e-3), c=T, cmap="viridis",
                        s=7, alpha=0.55, linewidths=0, zorder=5)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("fraction of episode elapsed", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        ax.set_yscale("log")
        ax.set_xlabel("formation ring radius R (m)")
        ax.set_ylabel("achieved |rotation rate| (rad/s)")
        ax.set_title(f"{_short(k)}", loc="left")
        ax.legend(loc="upper right")

    fig.suptitle("Does the controller ride the actuator boundary? "
                 "Every logged step, coloured by time",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig7_tradeoff.png")


# --------------------------------------------------------------------------
# 8. Animation
# --------------------------------------------------------------------------

def fig_anim(path=os.path.join(OUT_DIR, "evaluation_logs.json"), idx=0):
    logs = _load(path)
    if not logs:
        print(f"  skipping animation, no {path}")
        return None
    from matplotlib.animation import FuncAnimation, PillowWriter
    _style()
    key = _pick_runs(logs)[0]
    ep = logs[key][idx]
    fld = _rebuild_field(ep)
    robots = np.asarray(ep["robots"])
    traj = np.asarray(ep["centroid"])
    stride = max(1, len(robots) // 200)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    draw_field(ax, fld, n=120, levels=22)
    (line,) = ax.plot([], [], color=C[0], linewidth=2.0, zorder=9)
    (quad,) = ax.plot([], [], color=C[1], linewidth=2.0, zorder=10)
    ttl = ax.set_title("", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    def update(f):
        t = min(f * stride, len(robots) - 1)
        line.set_data(traj[:t + 1, 0], traj[:t + 1, 1])
        q = robots[t][[0, 1, 2, 3, 0]]
        quad.set_data(q[:, 0], q[:, 1])
        ttl.set_text(f"{key}  |  {ep['family']}  |  t={t*0.1:5.1f}s  "
                     f"e={ep['e'][t]:.3f} m  R={ep['radius'][t]:.3f} m")
        return line, quad, ttl

    anim = FuncAnimation(fig, update, frames=len(robots) // stride + 1,
                         blit=False)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "fig8_rollout.gif")
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("estimator", "gallery", "training", "comparison", "rollout",
              "mechanism", "tradeoff", "anim", "all"):
        p.add_argument(f"--{f}", action="store_true")
    args = p.parse_args()
    any_flag = any(getattr(args, f) for f in
                   ("estimator", "gallery", "training", "comparison",
                    "rollout", "mechanism", "tradeoff", "anim", "all"))
    if not any_flag:
        args.all = True

    print("figures ->", FIG_DIR)
    if args.all or args.estimator:
        fig_estimator()
    if args.all or args.gallery:
        fig_gallery()
    if args.all or args.training:
        fig_training()
    if args.all or args.comparison:
        fig_comparison()
    if args.all or args.rollout:
        fig_rollout()
    if args.all or args.mechanism:
        fig_mechanism()
    if args.all or args.tradeoff:
        fig_tradeoff()
    if args.anim:
        fig_anim()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
