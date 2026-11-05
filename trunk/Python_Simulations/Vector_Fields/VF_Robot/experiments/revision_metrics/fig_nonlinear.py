"""
Fig. 4 of the vector field revision (Section IV restructure, 2026-11-04): the gyre
cell with the three-robot formation drawn to scale. Replaces the untraced
fig_nonlinear_flows.pdf. Review output only, written to figures/; copy it into the
paper folder by hand.

(a) Attraction: the 1000 starts of nonlinear.py (Coulomb model, seed 0, 1 m box),
    colored by the critical point reached, with two trials drawn with their
    triangles. One starts inside the det J = 0 diamond and reaches the center. The
    other also starts inside the diamond, but its triangle straddles the edge, the
    fit gives det(J_hat) < 0, and it goes to a corner saddle.
(b) Orbit: one run at r_d = 0.3 m, which stays inside the diamond, and one at
    r_d = 0.4 m that starts inside it, reaches it, and is lost (nonlinear.py's keep
    rule), drawn until it leaves the plotted area.

Prints the check behind the text: of the in-diamond starts with det(J_hat) < 0,
how many had a vertex outside the diamond.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/fig_nonlinear.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonlinear as nl  # noqa: E402
import sim  # noqa: E402
import vecsim as vs  # noqa: E402

OUT = os.path.join(HERE, "figures")
BLUE, ORANGE, GRAY = "#2468c8", "#e8692c", "#b4b4b4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42})
L = 0.68


def diamond(ax, h=0.5):
    ax.plot([h, 0, -h, 0, h], [0, h, 0, -h, 0], ls=(0, (4, 2.5)), color="0.25", lw=0.8, zorder=2)


def stream(ax, F):
    g = np.linspace(-L, L, 120)
    X, Y = np.meshgrid(g, g)
    U, V = F.fn(X, Y)
    ax.streamplot(g, g, U, V, color=GRAY, linewidth=0.45, density=0.9, arrowsize=0.5, zorder=0)


def crit(ax, F):
    for x, y, idx in F.crit:
        if abs(x) <= L and abs(y) <= L:
            ax.plot(x, y, "o" if idx > 0 else "s", color="k", ms=4.0 if idx > 0 else 3.6, zorder=6)


def triangle(ax, P, color, alpha=1.0, z=5):
    t = np.vstack([P, P[:1]])
    ax.fill(P[:, 0], P[:, 1], color=color, alpha=0.12 * alpha, lw=0, zorder=z - 1)
    ax.plot(t[:, 0], t[:, 1], color=color, lw=0.8, alpha=alpha, zorder=z)
    ax.plot(P[:, 0], P[:, 1], "o", ms=2.0, mfc=color, mec="k", mew=0.3, alpha=alpha, zorder=z + 1)


def path(ax, rec, i, color, snaps, end=None, robots=True):
    c = rec["c"][i][:end]
    pos = rec["pos"][i][:end]
    if robots:
        for k in range(3):
            ax.plot(pos[:, k, 0], pos[:, k, 1], color=color, lw=0.35, alpha=0.55, zorder=3)
    ax.plot(c[:, 0], c[:, 1], color=color, lw=1.1, zorder=4)
    ax.plot(c[0, 0], c[0, 1], "o", ms=3.5, mfc="white", mec=color, mew=0.9, zorder=7)
    for s in snaps:
        triangle(ax, pos[s], color)


def frame(ax, title):
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel("$x$ (m)", labelpad=1)
    ax.set_title(title, fontsize=8, pad=3)


def main():
    os.makedirs(OUT, exist_ok=True)
    F = nl.make(dict(name="gyre"))
    att = pd.read_csv(os.path.join(HERE, "results", "nonlinear_attraction_per_trial.csv"))
    g = att[att.field == "gyre"].reset_index(drop=True)
    P = vs.params(len(g))
    out, _ = vs.simulate(F, "attract", "fit", P, 10.0, seed=0, stiction_model=sim.MODEL, record=True)
    rec = out["rec"]
    assert np.allclose(rec["c"][:, 0, 0], g.start_x), "trials do not match nonlinear.py"

    v0 = rec["pos"][:, 0]
    out_vertex = (np.abs(v0[..., 0]) + np.abs(v0[..., 1]) > 0.5).any(1)
    wrong = g.in_region & (g.det_hat_sign_start < 0)
    print(f"in-diamond starts with det(J_hat) < 0: {int(wrong.sum())}, of which {int((wrong & out_vertex).sum())} "
          f"had a vertex outside the diamond; in-diamond starts with det(J_hat) > 0 and a vertex outside: "
          f"{int((g.in_region & (g.det_hat_sign_start > 0) & out_vertex).sum())}")

    r0 = np.hypot(g.start_x, g.start_y)
    ang = np.degrees(np.arctan2(g.start_y, g.start_x))
    good = g.index[g.in_region & g.reached_target & (r0 > 0.28) & (r0 < 0.34) & (ang > 100) & (ang < 140)]
    bad = g.index[wrong & out_vertex & (ang > -80) & (ang < -10)]
    i_good = int(good[0])
    i_bad = int(bad[np.argmin(r0[bad])])
    print(f"(a) trials: center {i_good} (start r {r0[i_good]:.3f} m), saddle {i_bad} (start r {r0[i_bad]:.3f} m)")

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95), gridspec_kw=dict(wspace=0.12))
    ax = axs[0]
    stream(ax, F)
    diamond(ax)
    ok = g.reached_target.values
    ax.scatter(g.start_x[ok], g.start_y[ok], s=0.6, color=BLUE, alpha=0.45, lw=0, zorder=1)
    ax.scatter(g.start_x[~ok], g.start_y[~ok], s=0.6, color=ORANGE, alpha=0.45, lw=0, zorder=1)
    crit(ax, F)
    path(ax, rec, i_good, BLUE, snaps=[0, -1])
    path(ax, rec, i_bad, ORANGE, snaps=[0, -1])
    frame(ax, "Attraction")
    ax.set_ylabel("$y$ (m)", labelpad=1)

    ax = axs[1]
    stream(ax, F)
    diamond(ax)
    crit(ax, F)
    runs = {}
    for rd in (0.3, 0.4):
        Po = vs.params(100, rd=rd)
        o, _ = vs.simulate(F, "orbit", "fit", Po, 60.0, seed=0, stiction_model=sim.MODEL, record=True)
        runs[rd] = o["rec"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rd * np.cos(th), rd * np.sin(th), color="0.35", lw=0.5, ls=(0, (1, 1.5)), zorder=2)
    orb = pd.read_csv(os.path.join(HERE, "results", "nonlinear_orbit_per_trial.csv"))
    go = orb[(orb.field == "gyre")]
    k3 = int(go[np.isclose(go.rd, 0.3) & go.kept].trial.iloc[0])
    lost4 = go[np.isclose(go.rd, 0.4) & ~go.kept].trial.values
    c0 = runs[0.4]["c"][lost4, 0]
    inside4 = lost4[np.abs(c0[:, 0]) + np.abs(c0[:, 1]) < 0.44]
    k4 = int(inside4[0])
    path(ax, runs[0.3], k3, BLUE, snaps=[250], end=400, robots=False)
    c4 = runs[0.4]["c"][k4]
    s4 = np.flatnonzero(np.abs(c4[:, 0]) + np.abs(c4[:, 1]) > 0.5)
    exit4 = np.flatnonzero((np.abs(c4) > L).any(1))
    cut = int(exit4[0]) + 1 if len(exit4) else len(c4)
    path(ax, runs[0.4], k4, ORANGE, snaps=[0], end=cut, robots=False)
    print(f"(b) runs: r_d 0.3 kept {k3}, r_d 0.4 lost {k4}, crosses the diamond at step {s4[0] if len(s4) else None}")
    frame(ax, "Orbit")
    ax.set_yticklabels([])

    for a, lab in zip(axs, ("(a)", "(b)")):
        a.text(0.5, -0.36, lab, transform=a.transAxes, ha="center", va="top", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_nonlinear_triangles.{ext}"), bbox_inches="tight", pad_inches=0.02,
                    dpi=300)
    print("wrote", os.path.join(OUT, "fig_nonlinear_triangles.pdf"))


if __name__ == "__main__":
    main()
