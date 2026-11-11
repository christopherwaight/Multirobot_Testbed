"""
fig_notes.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13_unsteady_dg_proposed.tex
          (the side notes; nothing here is in Draft_13).
  Makes:  the four figures of the notes (numbered there in order of
          appearance), and prints every number the notes quote from them.
            notes_structures.png   true D and s1 trenches vs the separatrix
            notes_crest_loss.png   where the s1 tracker loses the straddle
            notes_rides.png        one ride per start, both trackers
            notes_scope.png        retention and success vs fitted |w|/2r
  Reads:  experiments/unsteady_dg/outputs/trials_n1000.csv, bent_n500.csv,
          bickley_n1000.csv, path_ratio_n50.csv
  Writes: Paper_Writing/Separatrix_and_OW_Paper/figures/notes_unsteady/*.png

Colors: s1 tracker blue #2a78d6, D tracker orange #eb6834 (validated pair,
dataviz palette slots 1 and 2), with the paper's marker convention as the
second encoding (D circles and solid lines, s1 triangles and dashed lines).

Run (from VF_Robot):
  venv/bin/python3 experiments/unsteady_dg/fig_notes.py
"""
import io
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve()
VF = HERE.parents[2]
REPO = HERE.parents[6]
sys.path.insert(0, str(VF))
os.chdir(VF)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import experiments._mc_common as mc
import experiments.unsteady_dg.mc_bent_double_gyre as bent
import experiments.unsteady_dg.mc_bickley_jet as bj
import experiments.unsteady_dg.mc_unsteady_double_gyre as tpg

OUT = REPO / "Paper_Writing" / "Separatrix_and_OW_Paper" / "figures" / "notes_unsteady"
DATA = VF / "experiments" / "unsteady_dg" / "outputs"
C_D, C_S = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d7d2"
ROT_FILL = "#ece8e1"
RHO = 0.075
OMEGA_F = 2 * np.pi / 10
H = 1e-5

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- fields
def f_steady(x, y):
    return double_gyre_static(x, y, 0.0, {"A": 0.1, "eps": 0.0, "omega": OMEGA_F})


def f_tp_peak(x, y, eps=0.25):
    return double_gyre_static(x, y, 2.5, {"A": 0.1, "eps": eps, "omega": OMEGA_F})


def f_bent(d):
    return lambda x, y: bent.bent_gyre(x, y, config={"delta": d})


def xs_tp_peak(eps=0.25):
    a = eps * np.sin(OMEGA_F * 2.5)
    b = 1 - 2 * a
    return 2 / (b + np.sqrt(b * b + 4 * a)) - 1


def jac(fun, x, y):
    up, um = fun(x + H, y), fun(x - H, y)
    vp, vm = fun(x, y + H), fun(x, y - H)
    return ((up[0] - um[0]) / (2 * H), (vp[0] - vm[0]) / (2 * H),
            (up[1] - um[1]) / (2 * H), (vp[1] - vm[1]) / (2 * H))


def D_s1(fun, x, y):
    ux, uy, vx, vy = jac(fun, x, y)
    D = ux * vy - uy * vx
    s1 = 0.5 * (ux + vy) - np.hypot(0.5 * (ux - vy), 0.5 * (uy + vx))
    return D, s1


def nearest_min(vals, coords, ref):
    i = np.where((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:]))[0] + 1
    if len(i) == 0:
        return np.nan
    return coords[i[np.argmin(np.abs(coords[i] - ref))]]


def trench_gyre(fun, xsep, ys, half=0.3, n=601):
    """For each y, the nearest local minimum of D and of s1 across x."""
    xd, xs1 = [], []
    for y in ys:
        x0 = xsep(y)
        xx = np.linspace(x0 - half, x0 + half, n)
        vals = np.array([D_s1(fun, x, y) for x in xx])
        xd.append(nearest_min(vals[:, 0], xx, x0))
        xs1.append(nearest_min(vals[:, 1], xx, x0))
    return np.array(xd), np.array(xs1)


def trench_bickley(xs, half=0.15, n=601):
    yd, ys1 = [], []
    for x in xs:
        y0 = float(bj.arc_y(x))
        yy = np.linspace(y0 - half, y0 + half, n)
        vals = np.array([D_s1(bj.bickley, x, y) for y in yy])
        yd.append(nearest_min(vals[:, 0], yy, y0))
        ys1.append(nearest_min(vals[:, 1], yy, y0))
    return np.array(yd), np.array(ys1)


def background(ax, fun, xlim, ylim, n=90):
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    U = np.zeros_like(X); V = np.zeros_like(X); Dg = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            U[i, j], V[i, j] = fun(X[i, j], Y[i, j])
            Dg[i, j] = D_s1(fun, X[i, j], Y[i, j])[0]
    ax.contourf(X, Y, Dg, levels=[0, np.inf], colors=[ROT_FILL])
    ax.streamplot(X, Y, U, V, color=GRID, density=1.0, linewidth=0.6, arrowsize=0.6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")


def ring(ax, x, y):
    ax.add_patch(Circle((x, y), RHO, fill=False, ec=MUTED, lw=0.8, ls=(0, (1, 1.5))))


def offsets(a, ref, mask=None):
    d = np.abs(a - ref)
    if mask is not None:
        d = d[mask]
    good = np.isfinite(d)
    return (np.nanmax(d) if good.any() else np.nan,
            np.nanmean(d) if good.any() else np.nan, int((~good).sum()), len(d))


# ---------------------------------------------------------------- Fig. N1
def fig_structures():
    ys = np.linspace(-0.49, 0.49, 197)
    cases = [
        ("(a) Steady double gyre", f_steady, lambda y: 0.0 * y, (-0.45, 0.45), (-0.5, 0.5)),
        ("(b) Time-periodic, $\\epsilon=0.25$", f_tp_peak,
         lambda y: xs_tp_peak() + 0.0 * y, (-0.2, 0.65), (-0.5, 0.5)),
        ("(c) Bent, $\\delta=0.2$", f_bent(0.2), lambda y: -0.2 * np.cos(np.pi * y),
         (-0.6, 0.3), (-0.5, 0.5)),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.5),
                            gridspec_kw={"width_ratios": [0.9, 0.85, 0.9, 1.55]})
    print("\nFig. N1 trench offsets from the separatrix (|y| >= 0.1 for gyres)")
    for ax, (title, fun, xsep, xl, yl) in zip(axs[:3], cases):
        background(ax, fun, xl, yl)
        xd, xs1 = trench_gyre(fun, xsep, ys)
        xsv = np.array([xsep(y) for y in ys])
        ax.plot(xsv, ys, color=INK, lw=2.2, label="separatrix")
        ax.plot(xd, ys, color=C_D, lw=1.6, label="$D$ trench")
        ax.plot(xs1, ys, color=C_S, lw=1.6, ls="--", label="$s_1$ trench")
        x_sad = xsep(np.array([-0.5, 0.5]))
        ax.plot(x_sad, [-0.5, 0.5], "x", color=INK, ms=7, mew=1.6, clip_on=False)
        ring(ax, xsep(0.25) + 0.0, 0.25)
        ax.set_title(title, loc="left")
        ax.set_xlabel("$x$")
        m = np.abs(ys) >= 0.1
        for name, arr in (("D", xd), ("s1", xs1)):
            mx, mn, miss, n = offsets(arr, xsv, m)
            print(f"  {title[:4]} {name:>2}: max {mx:.4f}  mean {mn:.4f}  "
                  f"no local min at {miss} of {n} heights")
    axs[0].set_ylabel("$y$")
    # Bickley
    ax = axs[3]
    xl, yl = (-0.62, 0.62), (-0.12, 0.46)
    background(ax, bj.bickley, xl, yl, n=110)
    xs = np.linspace(-0.49, 0.49, 197)
    yd, ys1 = trench_bickley(xs)
    ax.plot(bj.ARC[:, 0], bj.ARC[:, 1], color=INK, lw=2.2)
    ax.plot(xs, yd, color=C_D, lw=1.6)
    ax.plot(xs, ys1, color=C_S, lw=1.6, ls="--")
    ax.plot([bj.P1[0], bj.P2[0]], [bj.P1[1], bj.P2[1]], "x", color=INK, ms=7, mew=1.6)
    ax.plot(0.0, bj.Y_CENTER, "o", ms=4, color=MUTED)
    ring(ax, 0.0, float(bj.arc_y(0.0)))
    ax.set_title("(d) Bickley jet", loc="left")
    ax.set_xlabel("$x$")
    ya = np.array([float(bj.arc_y(x)) for x in xs])
    for name, arr in (("D", yd), ("s1", ys1)):
        mx, mn, miss, n = offsets(arr, ya)
        print(f"  (d)  {name:>2}: max {mx:.4f}  mean {mn:.4f}  "
              f"no local min within 0.15 at {miss} of {n} positions")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "notes_structures.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. N2
def ride(kind, par, tracker, start, max_steps=500):
    with contextlib.redirect_stdout(io.StringIO()):
        if kind == "tp":
            field = AnalyticalField(double_gyre_static)
            starts, p1, p2 = tpg.STARTS, None, None
        elif kind == "bent":
            field = AnalyticalField(bent.bent_gyre)
            starts, p1, p2 = bent.STARTS, bent.P1, bent.P2
        else:
            field = AnalyticalField(bj.bickley)
            starts, p1, p2 = bj.STARTS, bj.P1, bj.P2
        cluster = PentagonCluster(mc.FORMATION_CONFIG, field)
    if kind == "tp":
        field.config = {**field.config, "eps": par}
        field.t = 0.0
    elif kind == "bent":
        field.config = {"delta": par}
    np.random.seed(0)
    x0, y0 = starts[start]
    cluster.reset(x0, y0, heading_offset=0.0)
    if tracker == "D":
        prim = lambda c: tuple(mc.GAIN * v for v in separatrix_logic_c_step(c, v_max=mc.V_MAX))
    else:
        prim = lambda c: tuple(mc.GAIN * v for v in
                               oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=1.0))
    path, outcome = [(x0, y0)], "timeout"
    for _ in range(max_steps):
        cluster.move(prim)
        if kind == "tp":
            field.step(cluster.timestep)
            xs = tpg.sep_x(field.t, par, OMEGA_F)
            p1, p2 = (xs, -0.5), (xs, 0.5)
        cx, cy = cluster.get_centroid()
        path.append((cx, cy))
        if np.hypot(cx - p1[0], cy - p1[1]) < mc.SADDLE_CONTACT_D:
            outcome = "p1"; break
        if np.hypot(cx - p2[0], cy - p2[1]) < mc.SADDLE_CONTACT_D:
            outcome = "p2"; break
        ylo = bj.Y_SADDLE - 0.6 if kind == "bickley" else -0.6
        yhi = bj.Y_SADDLE + 0.6 if kind == "bickley" else 0.6
        if abs(cx) > 1.0 or cy < ylo or cy > yhi:
            outcome = "exit"; break
    return np.array(path), outcome


def fig_rides():
    cols = [("tp", 0.1, "Time-periodic, $\\epsilon=0.1$", (-0.45, 0.45), (-0.62, 0.62)),
            ("bent", 0.1, "Bent, $\\delta=0.1$", (-0.6, 0.6), (-0.62, 0.62)),
            ("bent", 0.2, "Bent, $\\delta=0.2$", (-0.6, 0.6), (-0.62, 0.62)),
            ("bickley", bj.EPS, "Bickley jet", (-1.02, 1.02), (-0.3, 0.95))]
    fig, axs = plt.subplots(2, 4, figsize=(10.4, 6.6), sharey=False)
    print("\nFig. N2 ride outcomes, heading 0 (tp phase 0)")
    ys = np.linspace(-0.5, 0.5, 201)
    for j, (kind, par, title, xl, yl) in enumerate(cols):
        for i, tr in enumerate(("D", "s1")):
            ax = axs[i, j]
            col, ls, mk = (C_D, "-", "o") if tr == "D" else (C_S, "--", "^")
            if kind == "tp":
                amp = tpg.sep_x(2.5, par, OMEGA_F)
                ax.axvspan(-amp, amp, color=ROT_FILL, lw=0)
                ax.axvline(0, color=INK, lw=0.8, ls=":")
                p_sad = [(0, -0.5), (0, 0.5)]
            elif kind == "bent":
                ax.plot(-par * np.cos(np.pi * ys), ys, color=INK, lw=2.0)
                xd, xs1 = trench_gyre(f_bent(par), lambda y: -par * np.cos(np.pi * y),
                                      ys[::4], n=301)
                ax.plot(xd if tr == "D" else xs1, ys[::4], color=col, lw=1.0, ls=ls, alpha=0.7)
                p_sad = [bent.P1, bent.P2]
            else:
                ax.plot(bj.ARC[:, 0], bj.ARC[:, 1], color=INK, lw=2.0)
                xs = np.linspace(-0.49, 0.49, 50)
                yd, ys1 = trench_bickley(xs, n=301)
                ax.plot(xs, yd if tr == "D" else ys1, color=col, lw=1.0, ls=ls, alpha=0.7)
                p_sad = [bj.P1, bj.P2]
            ax.plot(*zip(*p_sad), "x", color=INK, ms=7, mew=1.6)
            starts = {"tp": tpg.STARTS, "bent": bent.STARTS, "bickley": bj.STARTS}[kind]
            outs = []
            for s in starts:
                path, outc = ride(kind, par, tr, s)
                outs.append(f"{s}:{outc}")
                ax.plot(path[:, 0], path[:, 1], color=col, lw=1.1, ls=ls)
                ax.plot(*path[0], "o", ms=3.5, color=MUTED)
                ax.plot(*path[-1], mk, ms=6, color=col,
                        mfc=col if outc == "p1" else "white", mew=1.2)
            print(f"  {title[:12]:>12} {tr:>2}: " + " ".join(outs))
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
            letter = "abcdefgh"[4 * i + j]
            ax.set_title(f"({letter}) {title}, {'$D$' if tr == 'D' else '$s_1$'}", loc="left")
            if i == 1:
                ax.set_xlabel("$x$")
            if j == 0:
                ax.set_ylabel("$y$")
    fig.tight_layout()
    fig.savefig(OUT / "notes_rides.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. N3
def fig_scope():
    pr = pd.read_csv(DATA / "path_ratio_n50.csv")
    tp = pd.read_csv(DATA / "trials_n1000.csv", comment="#")
    be = pd.read_csv(DATA / "bent_n500.csv", comment="#")
    bk = pd.read_csv(DATA / "bickley_n1000.csv", comment="#")
    conds = [("steady", "tp", 0.0, tp[tp.eps == 0.0]),
             ("TP 0.1", "tp", 0.1, tp[tp.eps == 0.1]),
             ("TP 0.25", "tp", 0.25, tp[tp.eps == 0.25]),
             ("bent 0.1", "bent", 0.1, be[be.delta == 0.1]),
             ("bent 0.2", "bent", 0.2, be[be.delta == 0.2]),
             ("Bickley", "bickley", bj.EPS, bk)]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    print("\nFig. N3 points (fitted ratio median, retention, success)")
    for ax, metric, ylabel in ((axs[0], "retained", "Straddle retention (%)"),
                               (axs[1], "success", "Downstream-saddle success (%)")):
        ax.axvspan(0.15, 0.20, color=ROT_FILL, lw=0)
        ax.text(0.175, 110, "Santa Barbara\n(fitted, Sec. V-B)", ha="center", va="bottom",
                fontsize=7, color=MUTED)
        for tr, col, mk in (("D", C_D, "o"), ("s1", C_S, "^")):
            xs_, ys_ = [], []
            for name, f, p, df in conds:
                r = pr[(pr.field == f) & (np.isclose(pr.par, p)) & (pr.tracker == tr)].ratio_med.median()
                v = 100 * df[df.tracker == tr][metric].mean()
                hollow = name == "TP 0.25"
                ax.plot(r, v, mk, ms=7, color=col, mfc="white" if hollow else col,
                        mew=1.4, zorder=3)
                if not hollow:
                    xs_.append(r); ys_.append(v)
                if metric == "retained":
                    print(f"  {name:>8} {tr:>2}: ratio {r:.3f}  retention {v:5.1f}  "
                          f"success {100 * df[df.tracker == tr].success.mean():5.1f}")
                label, off = name, ((6, 3) if tr == "D" else (6, -10))
                if tr == "s1" and name == "steady":
                    label = None
                if metric == "retained" and tr == "s1" and name == "Bickley":
                    label = None
                if metric == "retained" and tr == "s1" and name == "bent 0.2":
                    label, off = "bent 0.2, Bickley", (-100, -4)
                if label:
                    ax.annotate(label, (r, v), xytext=off, textcoords="offset points",
                                fontsize=7, color=MUTED)
        ax.set_xlabel("Median fitted $|\\hat\\omega|/2\\hat r$ along the ride")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 0.68); ax.set_ylim(-6, 126)
        ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_D, marker="o", ls="none", label="$D$ tracker"),
               Line2D([], [], color=C_S, marker="^", ls="none", label="$s_1$ tracker"),
               Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none",
                      label="speed-limited (TP 0.25)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "notes_scope.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. N4
def fig_crest_loss():
    tp = pd.read_csv(DATA / "trials_n1000.csv", comment="#")
    g = tp[(tp.eps == 0.1) & (tp.tracker == "s1") & (tp.success == 1)].lost_y.dropna()
    gd = tp[(tp.eps == 0.1) & (tp.tracker == "D") & (tp.success == 1)].lost_y.dropna()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.axvspan(-0.1, 0.1, color=ROT_FILL, lw=0)
    bins = np.arange(-0.5, 0.501, 0.025)
    ax.hist(g, bins=bins, color=C_S, ec="white", lw=0.8)
    ax.set_xlabel("Centroid $y$ at the first lost straddle")
    ax.set_ylabel("Trials")
    ax.grid(True, axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    frac = float(np.mean(np.abs(g) < 0.1))
    print(f"\nFig. N4: s1 losses n={len(g)}, within |y|<0.1: {100 * frac:.1f}%; "
          f"D losses n={len(gd)}")
    fig.tight_layout()
    fig.savefig(OUT / "notes_crest_loss.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_structures()
    fig_rides()
    fig_scope()
    fig_crest_loss()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
