"""
mc_zone_of_convergence.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_12.tex (candidate, not yet in the draft)
  Makes:  figures/zone_of_convergence.png, a noise-free Monte Carlo map of where
          each tracker converges to the far saddle, with the noise-sweep starts
          marked on it.
  Reads:  nothing (new simulation runs).
  Writes: experiments/outputs/mc_zone/zone_trials.csv   one row per trial
          experiments/outputs/mc_zone/zone_starts.csv   per named start

EXPERIMENT
  P(x, y) = E over heading h ~ U[0, 2 pi) of
            1{ centroid reaches within 0.06 of (0, -0.5), no formation collapse }
  Same trackers, gains, step cap and stop rules as mc_sweep_noise_both_trackers.py
  (imported from it), with sigma_uv = sigma_p = 0.
    - Map: N starts uniform over START_BOX and a random heading each. D and s1
      see the same starts and headings. Binned into CELL x CELL cells; p_hat = k/n
      with SE = sqrt(p_hat (1 - p_hat) / n).
    - Named starts: the straddling start and S1 to S6 of Draft_12, N_HEADINGS
      random headings each, for the per-start noise-free rate.
  Outcome classes, one per trial, so none is forced into pass or fail:
    far_band    far saddle reached after the centroid held |x| < 0.05 for 10 steps
    far_direct  far saddle reached without a 10-step band hold (a direct approach;
                the trial stops at contact, which can cut the hold short)
    top         ended within 0.06 of the top saddle (0, 0.5)
    collapsed   formation collapse (RMS pair-distance error > 0.30)
    exit        left the domain before the step cap
    timeout     500 steps without any of the above

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_zone_of_convergence.py --trials 20000 --workers 3
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import experiments._mc_common as mc
import experiments.mc_sweep_noise_both_trackers as sw

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_zone")
os.makedirs(OUT_DIR, exist_ok=True)
REPO_ROOT = project_root
for _ in range(4):
    REPO_ROOT = os.path.dirname(REPO_ROOT)
FIG_DIR = os.path.join(REPO_ROOT, "Paper_Writing", "Separatrix_and_OW_Paper", "figures")

CELL = 0.1
N_HEADINGS = 500
# The seven-start set, in order from the top of the domain. The old six-start set
# (orig, S1 (-0.15, 0.30), S2 (0.05, 0.40), S3 (0, 0), S4 (0.10, -0.20),
# S5 (0.15, 0.25), S6 (-0.20, -0.30)) was used for the zone map of 2026-10-05.
NAMED_STARTS = {
    "S1": (-0.10, 0.30),    # upper, 0.10 off, left
    "S2": (0.15, 0.25),     # upper, 0.15 off, right
    "S3": (0.0, 0.35),      # upper, on the line (Fig. 7 start)
    "S4": (0.0, 0.0),       # origin
    "S5": (0.0, -0.25),     # lower, on the line
    "S6": (0.15, -0.15),    # lower, 0.15 off, right
    "S7": (0.10, -0.20),    # lower, 0.10 off, right
}
TOP_SADDLE = (0.0, 0.5)
CLASSES = ["far_band", "far_direct", "top", "collapsed", "exit", "timeout"]
COLUMNS = ["tracker", "start_x", "start_y", "heading", "steps", "t_band",
           "final_x", "final_y", "outcome"]


def _worker(spec):
    tr = spec.pop("tracker")
    name = spec.pop("name", "")
    spec["primitive"] = sw._prim_d if tr == "D" else sw._prim_s1
    y_exit = 0.52
    if tr == "s1":
        spec["target"] = sw.S1_TARGET
        spec["y_exit"] = sw.S1_Y_EXIT
        y_exit = sw.S1_Y_EXIT
    r = mc.run_trial(spec)
    fx, fy = r["final_x"], r["final_y"]
    if np.hypot(fx - mc.SADDLE[0], fy - mc.SADDLE[1]) < mc.SADDLE_CONTACT_D and not r["collapsed"]:
        out = "far_band" if r["t_band"] >= 0 else "far_direct"
    elif r["collapsed"]:
        out = "collapsed"
    elif np.hypot(fx - TOP_SADDLE[0], fy - TOP_SADDLE[1]) < mc.SADDLE_CONTACT_D:
        out = "top"
    elif r["steps"] < mc.SIM_STEPS and (abs(fx) > 1.0 or abs(fy) > y_exit):
        out = "exit"
    else:
        out = "timeout"
    return {"tracker": tr, "name": name, "start_x": r["start_x"], "start_y": r["start_y"],
            "heading": r["heading"], "steps": r["steps"], "t_band": r["t_band"],
            "final_x": fx, "final_y": fy, "outcome": out}


def make_specs(n_map, seed, n_head):
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = mc.START_BOX
    xs = rng.uniform(x0, x1, n_map)
    ys = rng.uniform(y0, y1, n_map)
    hs = rng.uniform(0.0, 2 * np.pi, n_map)
    specs = []
    for tr in ("D", "s1"):
        for k in range(n_map):
            specs.append({"tracker": tr, "name": "", "start": (float(xs[k]), float(ys[k])),
                          "heading": float(hs[k]), "sigma_uv": 0.0, "sigma_p": 0.0,
                          "seed": int((seed + 7919 * k) % (2**31))})
    hrng = np.random.default_rng(seed + 1)
    hh = hrng.uniform(0.0, 2 * np.pi, n_head)
    for name, st in NAMED_STARTS.items():
        for tr in ("D", "s1"):
            for k in range(n_head):
                specs.append({"tracker": tr, "name": name, "start": st,
                              "heading": float(hh[k]), "sigma_uv": 0.0, "sigma_p": 0.0,
                              "seed": int((seed + 104729 + 7919 * k) % (2**31))})
    return specs


PANELS = (("far saddle reached", ("far_band", "far_direct")),
          ("far saddle reached after band hold", ("far_band",)))


def cell_grid():
    x0, x1, y0, y1 = mc.START_BOX
    nx, ny = int(round((x1 - x0) / CELL)), int(round((y1 - y0) / CELL))
    xc = x0 + CELL * (np.arange(nx) + 0.5)
    yc = y0 + CELL * (np.arange(ny) + 0.5)
    return x0, x1, y0, y1, nx, ny, xc, yc


def grids(rows):
    """p_hat[tracker][panel index] as (ny, nx) arrays, plus the count grid n."""
    x0, x1, y0, y1, nx, ny, xc, yc = cell_grid()
    out = {}
    for tr in ("D", "s1"):
        mine = [r for r in rows if r["tracker"] == tr and r["name"] == ""]
        sx = np.array([r["start_x"] for r in mine])
        sy = np.array([r["start_y"] for r in mine])
        ix = np.clip(((sx - x0) / CELL).astype(int), 0, nx - 1)
        iy = np.clip(((sy - y0) / CELL).astype(int), 0, ny - 1)
        n = np.zeros((ny, nx))
        np.add.at(n, (iy, ix), 1)
        ps = []
        for _, keep in PANELS:
            k = np.zeros((ny, nx))
            np.add.at(k, (iy, ix), np.array([r["outcome"] in keep for r in mine], dtype=float))
            ps.append(np.where(n > 0, k / np.maximum(n, 1), np.nan))
        out[tr] = {"p": ps, "n": n}
    return out


def neighbourhood_min(p, i, j):
    """Minimum p_hat over the 3 x 3 block of cells around (row i, column j)."""
    blk = p[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
    return float(np.nanmin(blk))


def cell_of(x, y):
    x0, x1, y0, y1, nx, ny, xc, yc = cell_grid()
    return (min(max(int((y - y0) / CELL), 0), ny - 1),
            min(max(int((x - x0) / CELL), 0), nx - 1))


def candidates(g, min_off=0.15):
    """Cells where both trackers reach the far saddle after a band hold in
    >= 95% of headings, with every neighbouring cell >= 90% for both (so the
    start is not on the edge of the zone), at least min_off from x = 0
    (2 rho = 0.15), ranked by the weaker tracker's p_hat, then by distance off
    the line."""
    x0, x1, y0, y1, nx, ny, xc, yc = cell_grid()
    out = []
    for i in range(ny):
        for j in range(nx):
            if abs(xc[j]) < min_off:
                continue
            pd, ps = g["D"]["p"][1], g["s1"]["p"][1]
            if np.isnan(pd[i, j]) or np.isnan(ps[i, j]):
                continue
            if pd[i, j] < 0.95 or ps[i, j] < 0.95:
                continue
            if neighbourhood_min(pd, i, j) < 0.90 or neighbourhood_min(ps, i, j) < 0.90:
                continue
            out.append((float(xc[j]), float(yc[i]), float(pd[i, j]), float(ps[i, j]),
                        neighbourhood_min(pd, i, j), neighbourhood_min(ps, i, j)))
    out.sort(key=lambda c: (-min(c[2], c[3]), -abs(c[0])))
    return out


def plot(rows, path, g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1, nx, ny, xc, yc = cell_grid()
    panels = PANELS
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharex=True, sharey=True)
    for i, (tr, tr_name) in enumerate((("D", "$D$ tracker"), ("s1", "$s_1$ tracker"))):
        for j, (title, keep) in enumerate(panels):
            p = g[tr]["p"][j]
            ax = axes[i][j]
            im = ax.pcolormesh(np.append(xc - CELL / 2, x1), np.append(yc - CELL / 2, y1),
                               p, cmap="viridis", vmin=0, vmax=1, shading="flat")
            ax.contour(xc, yc, np.nan_to_num(p), levels=[0.5], colors="w", linewidths=1.2)
            ax.contour(xc, yc, np.nan_to_num(p), levels=[0.95], colors="w",
                       linewidths=0.8, linestyles="--")
            for name, (sx_, sy_) in NAMED_STARTS.items():
                ax.plot(sx_, sy_, marker="*" if name == "orig" else "o", markersize=6,
                        markerfacecolor="none", markeredgecolor="r", markeredgewidth=1.1)
                ax.annotate("" if name == "orig" else name, (sx_, sy_), xytext=(3, 3),
                            textcoords="offset points", fontsize=6.5, color="r")
            ax.set_title(f"{tr_name}, {title}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 1:
                ax.set_xlabel("$x_0$", fontsize=8)
            if j == 0:
                ax.set_ylabel("$y_0$", fontsize=8)
            ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("$\\hat{p}$ (heading-averaged)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")


def named_only(rows, head, suffix=""):
    """--skip-map: per-start outcome table, no map, no figure, no overwrite of the map files."""
    print("\nPer named start, noise-free, random headings")
    print(f"{'start':<6}{'trk':<4}{'n':>7}" + "".join(f"{c:>11}" for c in CLASSES)
          + f"{'far total':>11}{'steps med':>11}{'steps max':>11}")
    path = os.path.join(OUT_DIR, f"zone_starts_seven{suffix}.csv")
    with open(path, "w") as f:
        f.write(head + "start,x,y,tracker,n," + ",".join(CLASSES)
                + ",far_total,far_steps_median,far_steps_max\n")
        for name, (sx_, sy_) in NAMED_STARTS.items():
            for tr in ("D", "s1"):
                mine = [r for r in rows if r["name"] == name and r["tracker"] == tr]
                n = len(mine)
                fr = [sum(r["outcome"] == c for r in mine) / n for c in CLASSES]
                far = fr[0] + fr[1]
                st = [r["steps"] for r in mine if r["outcome"] in ("far_band", "far_direct")]
                med, mx = (int(np.median(st)), max(st)) if st else (-1, -1)
                print(f"{name:<6}{tr:<4}{n:7d}" + "".join(f"{v:11.4f}" for v in fr)
                      + f"{far:11.4f}{med:11d}{mx:11d}")
                f.write(f"{name},{sx_},{sy_},{tr},{n}," + ",".join(f"{v:.4f}" for v in fr)
                        + f",{far:.4f},{med},{mx}\n")
    print(f"\nSaved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000, help="map starts per tracker")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--headings", type=int, default=N_HEADINGS,
                    help="random headings per named start")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--skip-map", action="store_true",
                    help="run only the named starts; writes zone_starts_seven.csv and "
                         "leaves zone_trials.csv, zone_candidates.csv and the figure alone")
    ap.add_argument("--only", default="",
                    help="with --skip-map, comma-separated names to run (e.g. S1,S5,S6); "
                         "writes zone_starts_seven_<names>.csv")
    args = ap.parse_args()
    if args.skip_map:
        args.trials = 0
    if args.only:
        keep = args.only.split(",")
        for k in list(NAMED_STARTS):
            if k not in keep:
                del NAMED_STARTS[k]

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    specs = make_specs(args.trials, args.seed, args.headings)
    print(f"{len(specs)} trials, {args.workers} workers", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, specs, chunksize=32), 1):
            rows.append(r)
            if i % 5000 == 0:
                print(f"  {i}/{len(specs)}", flush=True)

    head = (f"# generated_by: experiments/mc_zone_of_convergence.py\n# git_commit: {commit}\n"
            f"# date: {stamp}\n# map_trials_per_tracker: {args.trials}  seed: {args.seed}\n"
            f"# start_box: {mc.START_BOX}  cell: {CELL}  headings_per_named_start: {args.headings}\n"
            f"# noise: sigma_uv = sigma_p = 0\n")
    if args.skip_map:
        named_only(rows, head, "_" + args.only.replace(",", "_") if args.only else "")
        return
    with open(os.path.join(OUT_DIR, "zone_trials.csv"), "w") as f:
        f.write(head + ",".join(["name"] + COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) if c != "name" else r["name"] for c in ["name"] + COLUMNS) + "\n")

    g = grids(rows)
    print("\nPer named start, noise-free, random headings. 'cell' is the map p_hat of the")
    print("cell holding the start (far saddle after band hold); 'nbr' is the minimum over")
    print("that cell and its neighbours. A low nbr with a high own rate marks an edge.")
    print(f"{'start':<6}{'trk':<4}" + "".join(f"{c:>11}" for c in CLASSES)
          + f"{'far total':>11}{'cell':>7}{'nbr':>7}")
    with open(os.path.join(OUT_DIR, "zone_starts.csv"), "w") as f:
        f.write(head + "start,tracker,n," + ",".join(CLASSES) + ",far_total,cell_far_band,nbr_min_far_band\n")
        for name, (sx_, sy_) in NAMED_STARTS.items():
            ci, cj = cell_of(sx_, sy_)
            for tr in ("D", "s1"):
                mine = [r for r in rows if r["name"] == name and r["tracker"] == tr]
                n = len(mine)
                fr = [sum(r["outcome"] == c for r in mine) / n for c in CLASSES]
                far = fr[0] + fr[1]
                p = g[tr]["p"][1]
                cell, nbr = float(p[ci, cj]), neighbourhood_min(p, ci, cj)
                print(f"{name:<6}{tr:<4}" + "".join(f"{v:11.3f}" for v in fr)
                      + f"{far:11.3f}{cell:7.2f}{nbr:7.2f}")
                f.write(f"{name},{tr},{n}," + ",".join(f"{v:.4f}" for v in fr)
                        + f",{far:.4f},{cell:.4f},{nbr:.4f}\n")

    cand = candidates(g)
    print(f"\nCandidate replacement starts: {len(cand)} cells where both trackers reach the far")
    print("saddle after a band hold in >= 95% of headings, all neighbours >= 90%, and |x| >= 0.15.")
    print(f"{'x':>7}{'y':>7}{'p_D':>7}{'p_s1':>7}{'nbr_D':>7}{'nbr_s1':>8}")
    for c in cand[:25]:
        print(f"{c[0]:7.2f}{c[1]:7.2f}{c[2]:7.2f}{c[3]:7.2f}{c[4]:7.2f}{c[5]:8.2f}")
    with open(os.path.join(OUT_DIR, "zone_candidates.csv"), "w") as f:
        f.write(head + "x,y,p_D,p_s1,nbr_min_D,nbr_min_s1\n")
        for c in cand:
            f.write(",".join(f"{v:.4f}" for v in c) + "\n")

    plot(rows, os.path.join(FIG_DIR, "zone_of_convergence.png"), g)


if __name__ == "__main__":
    main()
