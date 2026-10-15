"""
objectivity_rotating_mc.py

Rotating-observer Monte Carlo for the Separatrix paper, Sec. IV-D.

Both trackers (D tracker = Primitive 7, s1 tracker = Primitive 11) start
from S1 = (-0.10, 0.30), the off-structure start of Figs. 5 to 7, on the
noise-free steady double gyre seen from five observer frames: inertial, and
frames rotating at Omega = +/-0.10 and +/-0.23 about the origin
(src/fields/environments/Rotating_Frame.py, positive Omega = the paper's
clockwise frame). Each frame and tracker gets the same 10^4 initial headings,
drawn uniformly from [0, 2 pi) as in the paper's other sweeps. Rotating-frame
runs are pulled back to inertial coordinates before scoring.

Why these rates: the frame carries each point of the flow at Omega |p|,
largest at the far saddle (|p| = 0.5). The s1 tracker's capture saturates the
velocity magnitude at k c_max = 0.12, so holding that saddle needs
0.5 Omega < 0.12, i.e. Omega < 0.24. sat is a tanh, so the bound is strict:
at 0.24 the s1 tracker already settles 0.064 from p1*. 0.23 is the largest
two-decimal rate inside it. The D tracker's capture saturates each channel,
so near the bound it has up to sqrt(2) more speed than the s1 tracker.

Trials stop on a stall rule (see STALL_EPS). Per trial it records the centroid's first descent resampled on a y grid (for
the mean and SD bands), the largest offset from x = 0 on the ride, the
retention measure of _mc_common.py (steps without a straddle before
far-saddle contact), the IV-A acquisition step, the final position, and
whether each tracker's capture latch fired. Heading-0 runs with robot
histories are saved for the pentagon snapshots.

PAPER TRACEABILITY: writes experiments/outputs/oecs/objectivity_rotating.png
and objectivity_rotating_summary.csv. The paper copy is installed by hand as
Paper_Writing/Separatrix_and_OW_Paper/figures/objectivity_rotating.png, the
proposed replacement for fig:oecs_objectivity in Draft_13.tex. It supersedes
traverse_objectivity_demo.py's single run from (0.05, 0.40).

Run (from VF_Robot):
  venv/bin/python3 experiments/objectivity_rotating_mc.py            # simulate (cached) and plot
  venv/bin/python3 experiments/objectivity_rotating_mc.py --plot-only
  venv/bin/python3 experiments/objectivity_rotating_mc.py --n-trials 72   # smoke test, separate cache
"""
import sys
import os
import csv
import argparse
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

START      = (-0.10, 0.30)          # S1
FRAMES     = (0.0, -0.10, 0.10, -0.23, 0.23)
N_TRIALS   = 10_000
SEED       = 20261014
# Stall rule: a trial stops once the pulled-back centroid stays within
# STALL_EPS = 0.2 rho of its mean over the last STALL_K steps, or after IV-A's
# 500 steps. IV-A's saddle rule (within 0.06 of a saddle for 20 steps) misses
# the D tracker at |Omega| = 0.10, which parks 0.069 from p1*. Without a stop,
# long runs meet a simulator artifact: holding the far saddle in a rotating
# frame keeps the cluster moving, robots that dip below the omnibot stiction
# threshold (0.025) lag the rest, and PentagonCluster.move has no feedback on
# theta_c, so the pentagon rotates and collapses from about step 380. With
# the stall rule a 24-heading sample stopped every run by step 145.
SIM_STEPS  = 500
STALL_EPS  = 0.015
STALL_K    = 20
STALL_FROM = 30                     # never test before this step
RING_OK    = 0.0735                 # intact if every outer robot stays within 2% of rho
FAR_SADDLE = np.array([0.0, -0.5])
NEAR_SADDLE = np.array([0.0, 0.5])
Y_GRID     = np.linspace(0.30, -0.48, 400)   # first-descent resampling grid
RIDE_Y     = (0.15, -0.40)          # ride window for the offset measure
SNAP_OMEGA = 0.23                   # pentagon snapshots at +/- this rate, heading 0
N_WORKERS  = max(1, (os.cpu_count() or 2) - 1)

OUT_DIR   = os.path.join(HERE, "outputs", "oecs")
CACHE_DIR = os.path.join(OUT_DIR, "objectivity_rotating_mc")


# ============================================================================
# SIMULATION
# ============================================================================

def _primitive(name):
    from traverse_objectivity_demo import traverser_primitive, logic_c_primitive
    return logic_c_primitive if name == "D" else traverser_primitive


def simulate(name, omega, heading, robots=False):
    """One noise-free run from START. Returns pulled-back centroid (and robot)
    histories and the step at which the tracker's capture latch fired."""
    from traverse_objectivity_demo import FORMATION_CONFIG, DT
    from src.robot.pentagon_cluster import PentagonCluster
    from src.fields.field_types import AnalyticalField
    from src.fields.environments.Double_Gyre import double_gyre_static
    from src.fields.environments.Rotating_Frame import (
        make_rotating_frame, pull_back_trajectory)

    fn = (double_gyre_static if omega == 0.0
          else make_rotating_frame(double_gyre_static, omega))
    field = AnalyticalField(fn)
    cl = PentagonCluster(FORMATION_CONFIG, field)
    cl.reset(*START, heading_offset=heading)
    field.reset_clock()
    prim = _primitive(name)
    latch = "_logicc_captured" if name == "D" else "_oecs_captured"
    cap_step = -1
    recent = []
    for k in range(SIM_STEPS):
        cl.move(prim)
        field.step(cl.timestep)
        if cap_step < 0 and getattr(cl, latch, False):
            cap_step = k
        cx, cy = cl.get_centroid()
        th = omega * (k + 1) * DT
        cx, cy = (np.cos(th) * cx + np.sin(th) * cy,
                  -np.sin(th) * cx + np.cos(th) * cy)
        if abs(cx) > 1.05 or abs(cy) > 0.60:
            break
        recent.append((cx, cy))
        if k >= STALL_FROM and len(recent) > STALL_K:
            w = np.array(recent[-(STALL_K + 1):])
            if np.max(np.linalg.norm(w - w.mean(0), axis=1)) < STALL_EPS:
                break
    C = np.asarray(cl.get_center_history(), float)
    R = np.asarray(cl.get_robot_history(), float) if robots else None
    if omega != 0.0:
        t = (np.arange(len(C)) + 1) * DT
        C = pull_back_trajectory(C, t, omega)
        if robots:
            R = np.stack([pull_back_trajectory(R[:, i, :], t, omega)
                          for i in range(R.shape[1])], axis=1)
    return C, R, cap_step


def first_descent(C):
    """The centroid path up to the first step y stops falling, or y < Y_GRID[-1]."""
    up = np.where(np.diff(C[:, 1]) > -1e-4)[0]
    end = up[0] + 1 if len(up) else len(C)
    below = np.where(C[:end, 1] < Y_GRID[-1])[0]
    return C[:(below[0] + 1 if len(below) else end)]


def held(C, p, tol=0.06, k=20):
    run = 0
    for inside in np.linalg.norm(C - p, axis=1) < tol:
        run = run + 1 if inside else 0
        if run >= k:
            return True
    return False


def acquisition(C, tol=0.05, k=10):
    """IV-A: first of ten consecutive steps with the centroid within 0.05 of x = 0."""
    run = 0
    for i, inside in enumerate(np.abs(C[:, 0]) < tol):
        run = run + 1 if inside else 0
        if run == k:
            return i - k + 1
    return -1


def score(name, omega, heading):
    from traverse_objectivity_demo import straddle_loss
    C, R, cap_step = simulate(name, omega, heading, robots=True)
    lost, _ = straddle_loss(C, R)
    ring = np.linalg.norm(R - R.mean(1, keepdims=True), axis=2)
    ring_min = float(np.sort(ring, axis=1)[:, 1].min())   # skip the center robot
    D1 = first_descent(C)
    x_grid = np.interp(Y_GRID[::-1], D1[::-1, 1], D1[::-1, 0],
                       left=np.nan, right=np.nan)[::-1]
    ride = D1[(D1[:, 1] < RIDE_Y[0]) & (D1[:, 1] > RIDE_Y[1])]
    offset = float(ride[np.argmax(np.abs(ride[:, 0])), 0]) if len(ride) else np.nan
    acq = acquisition(C)
    return dict(
        x_grid=x_grid.astype(np.float32),
        offset=offset,
        lost_steps=len(lost),
        acq_step=acq,
        acq_y=float(C[acq, 1]) if acq >= 0 else np.nan,
        final=C[-1],
        n_steps=len(C),
        held_far=held(C, FAR_SADDLE),
        near_far=bool(np.any(np.linalg.norm(C - NEAR_SADDLE, axis=1) < 0.06)),
        final_dist=float(np.linalg.norm(C[-1] - FAR_SADDLE)),
        tail_dist=float(np.linalg.norm(C[-100:] - FAR_SADDLE, axis=1).max()),
        stopped_by_stall=len(C) < SIM_STEPS,
        cap_step=cap_step,
        ring_min=ring_min,
    )


def _work(args):
    name, omega, idx, headings = args
    return idx, [score(name, omega, h) for h in headings]


def run_combo(name, omega, headings, pool):
    chunks = np.array_split(np.arange(len(headings)), N_WORKERS * 8)
    jobs = [(name, omega, c, headings[c]) for c in chunks if len(c)]
    rows = [None] * len(headings)
    for idx, res in pool.imap_unordered(_work, jobs):
        for i, r in zip(idx, res):
            rows[i] = r
    out = {"heading": headings}
    for key in rows[0]:
        out[key] = np.array([r[key] for r in rows])
    return out


def cache_path(name, omega, n):
    tag = "" if n == N_TRIALS else f"_n{n}"
    return os.path.join(CACHE_DIR, f"{name}_{omega:+.2f}{tag}.npz")


def simulate_all(n, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    headings = np.random.default_rng(SEED).uniform(0.0, 2 * np.pi, n)
    todo = [(nm, om) for nm in ("D", "s1") for om in FRAMES
            if not (use_cache and os.path.exists(cache_path(nm, om, n)))]
    if todo:
        with mp.get_context("spawn").Pool(N_WORKERS) as pool:
            for nm, om in todo:
                res = run_combo(nm, om, headings, pool)
                np.savez_compressed(cache_path(nm, om, n), **res)
                print(f"  {nm:2s} Omega={om:+.2f}: {n} trials cached", flush=True)
    for om in (SNAP_OMEGA, -SNAP_OMEGA):
        for nm in ("D", "s1"):
            p = os.path.join(CACHE_DIR, f"snap_{nm}_{om:+.2f}.npz")
            if not (use_cache and os.path.exists(p)):
                C, R, _ = simulate(nm, om, 0.0, robots=True)
                np.savez_compressed(p, C=C, R=R)


def load(name, omega, n):
    return dict(np.load(cache_path(name, omega, n)))


# ============================================================================
# SUMMARY
# ============================================================================

def summarize(n):
    rows = []
    print(f"\n{'tracker':7s} {'Omega':>6s} {'ride offset med [p5,p95]':>27s} "
          f"{'straddle lost':>13s} {'lost steps med':>14s} {'acq step med':>12s} "
          f"{'held p1*':>8s} {'final dist med':>14s} {'latch':>6s} {'reached p2*':>11s}")
    for nm in ("D", "s1"):
        for om in FRAMES:
            r = load(nm, om, n)
            off = r["offset"]
            row = dict(
                tracker=nm, omega=om, n=n,
                offset_median=np.nanmedian(off),
                offset_p5=np.nanpercentile(off, 5), offset_p95=np.nanpercentile(off, 95),
                abs_offset_median=np.nanmedian(np.abs(off)),
                straddle_lost_frac=np.mean(r["lost_steps"] > 0),
                lost_steps_median=np.median(r["lost_steps"]),
                lost_steps_median_if_lost=(np.median(r["lost_steps"][r["lost_steps"] > 0])
                                           if np.any(r["lost_steps"] > 0) else 0),
                acq_step_median=np.median(r["acq_step"]),
                acq_y_median=np.nanmedian(r["acq_y"]),
                held_far_frac=np.mean(r["held_far"]),
                final_dist_median=np.median(r["final_dist"]),
                final_dist_p95=np.percentile(r["final_dist"], 95),
                tail_dist_median=np.median(r["tail_dist"]),
                latch_frac=np.mean(r["cap_step"] >= 0),
                reached_near_saddle_frac=np.mean(r["near_far"]),
                final_x_median=np.median(r["final"][:, 0]),
                final_y_median=np.median(r["final"][:, 1]),
                stall_stop_frac=np.mean(r["stopped_by_stall"]),
                stop_step_median=np.median(r["n_steps"]),
                stop_step_max=int(np.max(r["n_steps"])),
                formation_broken_frac=np.mean(r["ring_min"] < RING_OK),
                ring_min_min=float(r["ring_min"].min()),
                max_sd_on_ride=float(np.nanmax(np.nanstd(r["x_grid"], 0)[Y_GRID < RIDE_Y[0]])),
            )
            rows.append(row)
            print(f"{nm:7s} {om:+6.2f} {row['offset_median']:+.4f} "
                  f"[{row['offset_p5']:+.4f},{row['offset_p95']:+.4f}] "
                  f"{100*row['straddle_lost_frac']:12.1f}% {row['lost_steps_median_if_lost']:14.0f} "
                  f"{row['acq_step_median']:12.0f} {100*row['held_far_frac']:7.1f}% "
                  f"{row['final_dist_median']:14.4f} {100*row['latch_frac']:5.1f}% "
                  f"{100*row['reached_near_saddle_frac']:10.1f}%  ring min {row['ring_min_min']:.3f}")
    tag = "" if n == N_TRIALS else f"_n{n}"
    path = os.path.join(OUT_DIR, f"objectivity_rotating_summary{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Summary written to {path}")


# ============================================================================
# FIGURE
# ============================================================================

X_VIEW, Y_VIEW = (-0.32, 0.32), (-0.62, 0.42)    # Fig. 6's view
# Blue for a counterclockwise frame, red for clockwise, darker for faster.
# Checked with the dataviz palette validator (all pass on white; the light
# blue is 2.99:1, relieved by the legend).
STYLE = {0.0: ("#1a1a1a", "inertial"),
         -0.10: ("#5598e7", r"$\Omega=-0.10$"), 0.10: ("#e2615e", r"$\Omega=+0.10$"),
         -0.23: ("#1c5cab", r"$\Omega=-0.23$"), 0.23: ("#a82a29", r"$\Omega=+0.23$")}
LEGEND_ORDER = (-0.23, -0.10, 0.0, 0.10, 0.23)


def _outline(P):
    c = P.mean(0)
    mid = np.argmin(np.linalg.norm(P - c, axis=1))
    ring = np.delete(P, mid, axis=0)
    ring = ring[np.argsort(np.arctan2(ring[:, 1] - c[1], ring[:, 0] - c[0]))]
    return np.vstack([ring, ring[:1]])


def _one_sided_depth(R):
    rx = R[:, :, 0]
    straddle = (rx.min(1) < 0) & (rx.max(1) > 0)
    return np.where(straddle, 0.0, np.minimum(np.abs(rx.min(1)), np.abs(rx.max(1))))


def snapshot_steps():
    """D tracker at its deepest straddle loss on the ride, the -Omega run over
    the whole ride and the +Omega run above y = 0 so the two snapshots sit at
    different heights. The s1 tracker is shown at the same heights."""
    out = {}
    for om, y_min in ((-SNAP_OMEGA, RIDE_Y[1]), (SNAP_OMEGA, 0.0)):
        z = np.load(os.path.join(CACHE_DIR, f"snap_D_{om:+.2f}.npz"))
        C, R = z["C"], z["R"]
        ride = len(first_descent(C))
        depth = _one_sided_depth(R[:ride])
        depth[(C[:ride, 1] > RIDE_Y[0]) | (C[:ride, 1] < y_min)] = 0.0
        assert depth.max() > 0, f"D {om:+.2f} keeps the straddle in the snapshot window"
        k = int(np.argmax(depth))
        out[("D", om)] = k
        zs = np.load(os.path.join(CACHE_DIR, f"snap_s1_{om:+.2f}.npz"))
        Cs = zs["C"]
        rs = len(first_descent(Cs))
        out[("s1", om)] = int(np.argmin(np.abs(Cs[:rs, 1] - C[k, 1])))
    return out


def plot(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    from src.fields.environments.Double_Gyre import double_gyre_static

    gx = np.linspace(*X_VIEW, 80)
    gy = np.linspace(-0.5, 0.5, 120)
    GX, GY = np.meshgrid(gx, gy)
    U = np.empty_like(GX)
    V = np.empty_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            U[i, j], V[i, j] = double_gyre_static(GX[i, j], GY[i, j])

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 6})
    # Sized so the equal-aspect axes fill the canvas: panel width from the
    # margins, panel height from the view's aspect, then fixed inch margins
    # for the titles above and the x labels and legend below.
    W, left, right, wspace = 3.45, 0.13, 0.99, 0.08
    panel_w = W * (right - left) / (2 + wspace)
    panel_h = panel_w * (Y_VIEW[1] - Y_VIEW[0]) / (X_VIEW[1] - X_VIEW[0])
    top_in, bottom_in = 0.16, 0.42
    H = panel_h + top_in + bottom_in
    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    fig.subplots_adjust(left=left, right=right, wspace=wspace,
                        top=1 - top_in / H, bottom=bottom_in / H)

    snaps = snapshot_steps()
    for ax, (name, title) in zip(axes, (("D", "$D$ tracker"), ("s1", "$s_1$ tracker"))):
        ax.streamplot(GX, GY, U, V, color="0.82", density=0.8, linewidth=0.4,
                      arrowsize=0.5)
        ax.axvline(0.0, color="0.35", ls="--", lw=0.6)
        ax.plot(*FAR_SADDLE, marker="x", color="k", ms=5, mew=1.2, zorder=9)   # above the end squares
        for om in FRAMES:
            col = STYLE[om][0]
            r = load(name, om, n)
            X = r["x_grid"].astype(float)
            cover = np.mean(~np.isnan(X), 0)
            with np.errstate(invalid="ignore"):
                m, s = np.nanmean(X, 0), np.nanstd(X, 0)
            m[cover < 0.5] = np.nan
            s[cover < 0.5] = np.nan
            ax.fill_betweenx(Y_GRID, m - s, m + s, color=col, alpha=0.25, lw=0, zorder=3)
            ax.plot(m, Y_GRID, color=col, lw=1.1, zorder=4)
            if om != 0.0:
                fx, fy = np.median(r["final"][:, 0]), np.median(r["final"][:, 1])
                ax.plot(fx, fy, marker="s", color=col, ms=3, mec="black", mew=0.4, zorder=7)
        for om in (SNAP_OMEGA, -SNAP_OMEGA):
            P = np.load(os.path.join(CACHE_DIR, f"snap_{name}_{om:+.2f}.npz"))["R"][snaps[(name, om)]]
            ring = _outline(P)
            ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.7, zorder=5)
            ax.plot(P[:, 0], P[:, 1], "o", color=STYLE[om][0], ms=2.6, mec="black",
                    mew=0.4, zorder=6)
        ax.plot(*START, marker="*", color="lime", ms=6, mec="black", mew=0.5, zorder=8)
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xlim(*X_VIEW)
        ax.set_ylim(*Y_VIEW)
        ax.set_aspect("equal")
        ax.set_xticks([-0.25, 0, 0.25])
        ax.set_xlabel("$x$", labelpad=1)
        ax.tick_params(length=2, pad=1)
    axes[0].set_ylabel("$y$", labelpad=1)

    handles = [(Patch(color=STYLE[om][0], alpha=0.25, lw=0),
                Line2D([], [], color=STYLE[om][0], lw=1.1)) for om in LEGEND_ORDER]
    fig.legend(handles, [STYLE[om][1] for om in LEGEND_ORDER],
               handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower center",
               ncol=5, frameon=False, handlelength=1.4, columnspacing=0.7,
               handletextpad=0.3, bbox_to_anchor=(0.5 * (left + right), 0.0),
               borderaxespad=0.15)
    tag = "" if n == N_TRIALS else f"_n{n}"
    path = os.path.join(OUT_DIR, f"objectivity_rotating{tag}.png")
    fig.savefig(path, dpi=400)
    print(f"Figure written to {path}")
    for key, k in snaps.items():
        C = np.load(os.path.join(CACHE_DIR, f"snap_{key[0]}_{key[1]:+.2f}.npz"))["C"]
        print(f"  snapshot {key[0]:2s} {key[1]:+.2f}: step {k}, centroid ({C[k, 0]:+.3f}, {C[k, 1]:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        simulate_all(a.n_trials, use_cache=not a.no_cache)
    summarize(a.n_trials)
    plot(a.n_trials)
