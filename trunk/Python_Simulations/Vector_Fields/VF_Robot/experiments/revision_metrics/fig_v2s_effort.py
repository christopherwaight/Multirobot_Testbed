"""
Draft replacement for Fig. 8 (VI-B rebuilt around the vector-to-scalar primitive, 2026-11-04).
Analytical vortex, ten starts evenly spaced on a circle (0.4 m for attraction, r_d for the
orbit), Coulomb model and gains of sim.py, seed 0. Mean and +-1 std across the ten runs.

Draft A: (a) attraction centroid speed, (b) orbit distance from p* at r_d = 0.3 m.
Draft B: cumulative control effort E_u(t) for the same runs.
Writes figures/fig_v2s_effort_A.{pdf,png} and fig_v2s_effort_B.{pdf,png}.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/fig_v2s_effort.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim  # noqa: E402
import vecsim as vs  # noqa: E402

OUT = os.path.join(HERE, "figures")
FIELD, R0, RD, N = "vortex", 0.4, 0.3, 10
ATT = [("fit", "Jacobian", "#2468c8", "-"), ("magnitude", "V2S", "#e8692c", (0, (5, 2)))]
ORB = [("fit", "Jacobian", "#2468c8", "-"), ("v2s", "V2S", "#e8692c", (0, (5, 2))),
       ("v2s_oracle", "V2S + oracle", "#2a9d5c", (0, (1.2, 1.4)))]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def run(task, method, r0, T):
    F = vs.make_field(FIELD)
    ph = 2 * np.pi * np.arange(N) / N
    starts = F.target + r0 * np.stack([np.cos(ph), np.sin(ph)], 1)
    kw = dict(rd=RD) if task == "orbit" else {}
    out, _ = vs.simulate(F, task, method, vs.params(N, **kw), T, seed=0, starts=starts, stiction_model=sim.MODEL,
                         record=True)
    c = out["rec"]["c"]
    t = np.arange(c.shape[1]) * 0.1
    step = np.linalg.norm(np.diff(c, axis=1), axis=2)
    speed = np.concatenate([np.zeros((N, 1)), step / 0.1], 1)
    effort = np.concatenate([np.zeros((N, 1)), np.cumsum(step ** 2 / 0.1, axis=1)], 1)
    r = np.linalg.norm(c - F.target, axis=2)
    return t, speed, effort, r


def band(ax, t, y, col, ls, label):
    m, s = np.nanmean(y, 0), np.nanstd(y, 0)
    ax.fill_between(t, m - s, m + s, color=col, alpha=0.18, lw=0)
    ax.plot(t, m, color=col, ls=ls, lw=1.3, label=label)


def main():
    os.makedirs(OUT, exist_ok=True)
    att = {m: run("attract", m, R0, 10.0) for m, *_ in ATT}
    orb = {m: run("orbit", m, RD, 30.0) for m, *_ in ORB}
    for m, lab, *_ in ATT:
        print(f"attraction {lab:12s} E_u {att[m][2][:, -1].mean():.3f}  final r {att[m][3][:, -1].mean():.4f}")
    for m, lab, *_ in ORB:
        print(f"orbit      {lab:12s} E_u(30 s) {orb[m][2][:, -1].mean():.3f}  mean r 10-30 s {np.nanmean(orb[m][3][:, 100:]):.3f}")

    # Draft A
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw=dict(wspace=0.38))
    for m, lab, col, ls in ATT:
        t, sp, E, _ = att[m]
        band(a, t, sp, col, ls, f"{lab}, $E_u$ = {E[:, -1].mean():.2f}")
    a.set_xlim(0, 6); a.set_xlabel("Time (s)", labelpad=1); a.set_ylabel("Centroid speed (m/s)", labelpad=1)
    a.legend(frameon=False, fontsize=6, loc="upper right"); a.set_title("(a) Attraction", fontsize=8, pad=2)
    for m, lab, col, ls in ORB:
        t, _, _, r = orb[m]
        r = np.where(r > 1.0, np.nan, r)
        band(b, t, r, col, ls, lab)
    b.axhline(RD, color="0.3", lw=0.6, ls=(0, (4, 2)))
    b.set_xlim(0, 30); b.set_ylim(0, 1.0); b.set_xlabel("Time (s)", labelpad=1)
    b.set_ylabel(r"$\|\mathbf{p}_c - \mathbf{p}^*\|$ (m)", labelpad=1)
    b.legend(frameon=False, fontsize=6, loc="upper left"); b.set_title(f"(b) Orbit, $r_d$ = {RD} m", fontsize=8, pad=2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_v2s_effort_A.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)

    # Draft B
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw=dict(wspace=0.38))
    for m, lab, col, ls in ATT:
        t, _, E, _ = att[m]
        band(a, t, E, col, ls, lab)
    for m, lab, col, ls in ORB:
        t, _, E, r = orb[m]
        E = np.where(np.maximum.accumulate(r, axis=1) > 0.838, np.nan, E)
        band(b, t, E, col, ls, lab)
    for ax, ttl in ((a, "(a) Attraction"), (b, f"(b) Orbit, $r_d$ = {RD} m")):
        ax.set_xlabel("Time (s)", labelpad=1); ax.set_title(ttl, fontsize=8, pad=2)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    a.set_ylabel("Cumulative $E_u$ (m$^2$/s)", labelpad=1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_v2s_effort_B.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("wrote drafts A and B in", OUT)


if __name__ == "__main__":
    main()
