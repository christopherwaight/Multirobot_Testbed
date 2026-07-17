"""
basin_map.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the zero-noise basin map figure and the Section VII-E basin
          characterization (SIM-2 of the T-RO hardening plan).
          Prediction under test: the traverse basin is the
          strain-dominated region (D < 0), whose area fraction over the
          domain is exactly 1/2, so the published 47.1% uniform-start
          rate is an area fraction, not a stochastic failure rate.
  Reads:  nothing. Writes experiments/outputs/basin_map/
          trials_grid_<variant>.csv, trials_heading_<variant>.csv, and
          prints the classification summary per variant.

EXPERIMENT
  Zero-noise Logic C from every cell center of a 200 x 100 grid over
  the domain [-1,1] x [-0.5,0.5], heading 0.  Outcome classes:
    TRAVERSE   reached the terminal saddle without formation collapse
    EXIT       crossed the termination bounds
    CORE-PARK  timed out with analytic D(final) > 0 near a gyre core
    TIMEOUT    any other timeout
  Two termination variants share seeds:
    strict     the published V-D2 rules (500 steps, |x|>1.0, |y|>0.52);
               note the |x|>1.0 bound sits exactly on the wall trenches
               of D, so boundary-trench rides are censored as EXIT
    relaxed    800 steps, |x|>1.1, |y|>0.6: same success definition,
               but wall-branch rides may close the heteroclinic cycle
  The per-row CSV records the analytic D, ||grad D||,
  distance-to-nearest-core at the final centroid, and the closest
  approach to each saddle, so classes can be revisited without
  rerunning.
  Robustness subsample: 20 x 10 grid, 10 random-heading draws per cell.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/basin_map.py --workers 8
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

from src.control.pentagon_primitives import separatrix_logic_c_step
import experiments._mc_common as mc

NX, NY = 200, 100
X_MIN, X_MAX, Y_MIN, Y_MAX = -1.0, 1.0, -0.5, 0.5
SUB_NX, SUB_NY, SUB_DRAWS = 20, 10, 10
GYRE_CORES = [(-0.5, 0.0), (0.5, 0.0)]

VARIANTS = {
    "strict": {},
    "relaxed": {"max_steps": 800, "x_exit": 1.1, "y_exit": 0.6},
}

A_DG = 0.1
K_D = np.pi**4 * A_DG**2 / 2.0
K_GRAD = np.pi**5 * A_DG**2

# CORE-PARK thresholds (recorded in the CSV header; the raw analytic
# values are in the rows so these can be changed in analysis).
CORE_D_MIN = 0.0
CORE_GRAD_MAX = 0.5     # 16% of the gradient scale pi^5 A^2 = 3.06


def d_analytic(x, y):
    return -K_D * (np.cos(2 * np.pi * (x + 1.0))
                   + np.cos(2 * np.pi * (y + 0.5)))


def grad_d_norm(x, y):
    gx = np.sin(2 * np.pi * (x + 1.0))
    gy = np.sin(2 * np.pi * (y + 0.5))
    return K_GRAD * float(np.hypot(gx, gy))


def _prim(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX, eps_raw=mc.EPS_RAW,
                                     eps_dim=mc.EPS_DIM)
    return vx * mc.GAIN, vy * mc.GAIN


def classify(row, x_exit, y_exit):
    if row["success_traverse"]:
        return "TRAVERSE"
    if abs(row["final_x"]) > x_exit or abs(row["final_y"]) > y_exit:
        return "EXIT"
    if (row["final_D"] > CORE_D_MIN and row["final_gradD"] < CORE_GRAD_MAX):
        return "CORE-PARK"
    return "TIMEOUT"


def _worker(spec):
    # Track closest approach to both saddles so the analysis can tell
    # "rode the network out of bounds" apart from "left the structure".
    tracker = {"min_d_top": np.inf, "min_d_bot": np.inf}

    def prim(c):
        cx, cy = c.get_centroid()
        tracker["min_d_top"] = min(tracker["min_d_top"],
                                   float(np.hypot(cx, cy - 0.5)))
        tracker["min_d_bot"] = min(tracker["min_d_bot"],
                                   float(np.hypot(cx, cy + 0.5)))
        return _prim(c)

    spec["primitive"] = prim
    row = mc.run_trial(spec)
    fx, fy = row["final_x"], row["final_y"]
    row["final_D"] = d_analytic(fx, fy)
    row["final_gradD"] = grad_d_norm(fx, fy)
    row["dist_core"] = min(np.hypot(fx - cx, fy - cy)
                           for cx, cy in GYRE_CORES)
    row["min_d_top"] = tracker["min_d_top"]
    row["min_d_bot"] = tracker["min_d_bot"]
    row["outcome"] = classify(row, spec.get("x_exit", 1.0),
                              spec.get("y_exit", 0.52))
    return row


COLUMNS = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
           "steps", "success_traverse", "collapsed", "final_x", "final_y",
           "final_D", "final_gradD", "dist_core", "min_d_top", "min_d_bot",
           "outcome"]


def provenance(tag, over):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return [
        "# generated_by: experiments/basin_map.py",
        f"# git_commit: {commit}",
        f"# date: {datetime.now(timezone.utc).isoformat()}",
        f"# mode: {tag}  max_steps: {over.get('max_steps', mc.SIM_STEPS)}"
        f"  x_exit: {over.get('x_exit', 1.0)}"
        f"  y_exit: {over.get('y_exit', 0.52)}",
        f"# grid: {NX}x{NY} over [{X_MIN},{X_MAX}]x[{Y_MIN},{Y_MAX}]"
        f"  subsample: {SUB_NX}x{SUB_NY}x{SUB_DRAWS}",
        "# controller: separatrix_logic_c_step (Logic C), zero noise",
        f"# collapse_rms: {mc.COLLAPSE_RMS}"
        f"  saddle_contact_d: {mc.SADDLE_CONTACT_D}",
        f"# core_park: final_D > {CORE_D_MIN} and final_gradD <"
        f" {CORE_GRAD_MAX}",
    ]


def write_csv(path, tag, over, rows):
    with open(path, "w") as f:
        f.write("\n".join(provenance(tag, over)) + "\n")
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")


def summarize(variant, grid_rows, sub_rows, xs, ys):
    by_start = {(r["start_x"], r["start_y"]): r for r in grid_rows}
    outcome_grid = np.empty((NX, NY), dtype=object)
    traverse = np.zeros((NX, NY), dtype=bool)
    strain = np.zeros((NX, NY), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r = by_start[(float(x), float(y))]
            outcome_grid[i, j] = r["outcome"]
            traverse[i, j] = r["outcome"] == "TRAVERSE"
            strain[i, j] = d_analytic(x, y) < 0

    print(f"\n=== [{variant}] grid classification (heading 0) ===",
          flush=True)
    for cls in ("TRAVERSE", "CORE-PARK", "EXIT", "TIMEOUT"):
        n = int(np.sum(outcome_grid == cls))
        print(f"  {cls:10s} {n:6d}  ({100.0 * n / (NX * NY):.2f}%)",
              flush=True)
    print(f"  strain fraction (D<0 cell centers): "
          f"{100.0 * strain.mean():.2f}%", flush=True)
    agree = traverse == strain
    print(f"  TRAVERSE == (D<0) agreement: {100.0 * agree.mean():.2f}%",
          flush=True)

    mism = np.argwhere(~agree)
    if len(mism):
        # Chebyshev distance from each mismatched cell to the nearest
        # cell of opposite analytic D sign (1 = touching the boundary).
        dists = []
        for (i, j) in mism:
            s = strain[i, j]
            found = None
            for radius in range(1, max(NX, NY)):
                i0, i1 = max(0, i - radius), min(NX, i + radius + 1)
                j0, j1 = max(0, j - radius), min(NY, j + radius + 1)
                if np.any(strain[i0:i1, j0:j1] != s):
                    found = radius
                    break
            dists.append(found if found is not None else -1)
        dists = np.array(dists)
        print(f"  mismatched cells: {len(mism)}  "
              f"boundary distance (cells): median {np.median(dists):.0f}, "
              f"max {dists.max()}", flush=True)

    sub_traverse = np.mean([r["outcome"] == "TRAVERSE" for r in sub_rows])
    print(f"  random-heading subsample TRAVERSE fraction: "
          f"{100.0 * sub_traverse:.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.path.join(project_root, "experiments", "outputs",
                           "basin_map")
    os.makedirs(out_dir, exist_ok=True)

    xs = X_MIN + (np.arange(NX) + 0.5) * (X_MAX - X_MIN) / NX
    ys = Y_MIN + (np.arange(NY) + 0.5) * (Y_MAX - Y_MIN) / NY
    sub_xs = X_MIN + (np.arange(SUB_NX) + 0.5) * (X_MAX - X_MIN) / SUB_NX
    sub_ys = Y_MIN + (np.arange(SUB_NY) + 0.5) * (Y_MAX - Y_MIN) / SUB_NY

    with Pool(args.workers) as pool:
        for variant, over in VARIANTS.items():
            grid_specs = []
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    grid_specs.append(dict({"sigma_uv": 0.0, "sigma_p": 0.0,
                                            "start": (float(x), float(y)),
                                            "heading": 0.0,
                                            "seed": 1_000_000 + i * NY + j},
                                           **over))
            sub_specs = []
            idx = 0
            for x in sub_xs:
                for y in sub_ys:
                    for k in range(SUB_DRAWS):
                        sub_specs.append(dict({"sigma_uv": 0.0,
                                               "sigma_p": 0.0,
                                               "start": (float(x), float(y)),
                                               "seed": 2_000_000 + idx},
                                              **over))
                        idx += 1

            print(f"[{variant}] grid: {len(grid_specs)} trials", flush=True)
            grid_rows = list(pool.imap_unordered(_worker, grid_specs,
                                                 chunksize=32))
            write_csv(os.path.join(out_dir, f"trials_grid_{variant}.csv"),
                      f"grid_heading0_{variant}", over, grid_rows)
            print(f"[{variant}] subsample: {len(sub_specs)} trials",
                  flush=True)
            sub_rows = list(pool.imap_unordered(_worker, sub_specs,
                                                chunksize=32))
            write_csv(os.path.join(out_dir, f"trials_heading_{variant}.csv"),
                      f"subsample_random_heading_{variant}", over, sub_rows)
            summarize(variant, grid_rows, sub_rows, xs, ys)

    # Reconciliation: analytic strain fraction over the old random-start
    # box of the Monte Carlo sweeps (published 47.1% at zero noise).
    bx = np.linspace(mc.START_BOX[0], mc.START_BOX[1], 2001)
    by = np.linspace(mc.START_BOX[2], mc.START_BOX[3], 1001)
    BX, BY = np.meshgrid(bx, by)
    frac_box = np.mean(d_analytic(BX, BY) < 0)
    print(f"\nanalytic strain fraction over START_BOX {mc.START_BOX}: "
          f"{100.0 * frac_box:.2f}%", flush=True)


if __name__ == "__main__":
    main()
