"""
mc_sweep_acquisition.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Makes:  the acquisition-under-noise result of Section VI-A (SIM-1 of
          the T-RO hardening plan): Monte Carlo noise sweep of Logic C
          from two NON-straddling clean-run starts, so the acquisition
          capability gets the same statistics as the on-structure
          traverse.
  Reads:  nothing. Writes experiments/outputs/mc_acquisition/
          trials_<name>.csv and summary_<name>.csv.

EXPERIMENT
  Same harness, success definition, and seed discipline as
  mc_sweep_separatrix.py (fixed-start mode), with the start moved off
  the structure (both 2.67 rho from the nearest trench segment,
  rho = 0.075):
    S1 (-0.45, +0.30)  near the left gyre center, weak strain
                       (s1 = -0.12 s^-1, D = +0.31, rotation-dominated)
    S6 (-0.20, -0.30)  gyre-interior strain region (D = -0.30)
  Grid: sigma_uv in SIGMA_UV_BASE x sigma_p in {0, 0.01}, random
  heading per trial.  Each row also records the minimum analytic
  ||grad D|| along the centroid path (mechanism check: acquisition
  failures should correlate with transits of low-gradient regions).

Run (development, 100 trials/cell):
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_acquisition.py --trials 100
Final run for the paper: --trials 1000 --workers 8.
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

SIGMA_UV = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_P = [0.0, 0.01]
# name -> (start, seed offset); offsets follow the mc_sweep_ow.py "+17"
# convention so no cell shares seeds with the other sweeps.
STARTS = {
    "S1": ((-0.45, 0.30), 23),
    "S6": ((-0.20, -0.30), 29),
}
# strict: identical termination to the published V-D2 sweeps.
# relaxed: same success definition, but the trial is not killed for
# riding a boundary trench (the walls of the domain ARE valley lines of
# D, and the strict |x_c| > 1.0 bound sits exactly on them) or for the
# few-mm overshoot of the top-saddle turn; wall-branch rides need the
# longer horizon to close the heteroclinic cycle.
VARIANTS = {
    "strict": {},
    "relaxed": {"max_steps": 800, "x_exit": 1.1, "y_exit": 0.6},
}

A_DG = 0.1
K_GRAD = np.pi**5 * A_DG**2
TRENCH_X = (-1.0, 0.0, 1.0)
TRENCH_Y = (-0.5, 0.5)


def _grad_d(x, y):
    return K_GRAD * np.array([np.sin(2 * np.pi * (x + 1.0)),
                              np.sin(2 * np.pi * (y + 0.5))])


def _worker(spec):
    tracker = {"min_grad": np.inf, "min_net": np.inf}

    def prim(c):
        cx, cy = c.get_centroid()
        g = _grad_d(cx, cy)
        n = float(np.hypot(g[0], g[1]))
        if n < tracker["min_grad"]:
            tracker["min_grad"] = n
        d_net = min(min(abs(cx - tx) for tx in TRENCH_X),
                    min(abs(cy - ty) for ty in TRENCH_Y))
        if d_net < tracker["min_net"]:
            tracker["min_net"] = d_net
        vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX,
                                         eps_raw=mc.EPS_RAW,
                                         eps_dim=mc.EPS_DIM)
        return vx * mc.GAIN, vy * mc.GAIN

    spec["primitive"] = prim
    row = mc.run_trial(spec)
    row["min_gradD_path"] = tracker["min_grad"]
    row["min_dist_network"] = tracker["min_net"]
    return row


COLUMNS = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
           "t_band", "steps", "success_traverse", "success_band",
           "success_straddle", "first_straddle", "collapsed", "track_mean",
           "track_p95", "shape_rms_max", "effort", "final_x", "final_y",
           "min_gradD_path", "min_dist_network"]


def provenance(n_trials, name, start, offset, variant, over):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return [
        "# generated_by: experiments/mc_sweep_acquisition.py",
        f"# git_commit: {commit}",
        f"# date: {datetime.now(timezone.utc).isoformat()}",
        f"# trials_per_cell: {n_trials}",
        f"# variant: {variant}  max_steps: "
        f"{over.get('max_steps', mc.SIM_STEPS)}  x_exit: "
        f"{over.get('x_exit', 1.0)}  y_exit: {over.get('y_exit', 0.52)}",
        "# controller: separatrix_logic_c_step (Logic C)",
        f"# start_mode: fixed  start_{name}: {start}  seed_offset: {offset}",
        f"# band_x: {mc.BAND_X}  collapse_rms: {mc.COLLAPSE_RMS}"
        f"  saddle_contact_d: {mc.SADDLE_CONTACT_D}",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.path.join(project_root, "experiments", "outputs",
                           "mc_acquisition")
    os.makedirs(out_dir, exist_ok=True)

    with Pool(args.workers) as pool:
        for variant, over in VARIANTS.items():
            for name, (start, offset) in STARTS.items():
                rows = []
                summary = []
                for s_uv in SIGMA_UV:
                    for s_p in SIGMA_P:
                        base = int(1e6 * s_uv * 1000 + 1e3 * s_p * 1000
                                   + offset) % (2**31)
                        specs = [dict({"sigma_uv": s_uv, "sigma_p": s_p,
                                       "start": start,
                                       "seed": (base + 7919 * t) % (2**31)},
                                      **over)
                                 for t in range(args.trials)]
                        out = list(pool.imap_unordered(_worker, specs,
                                                       chunksize=16))
                        rows.extend(out)
                        st = np.mean([r["success_traverse"] for r in out])
                        sb = np.mean([r["success_band"] for r in out])
                        ss = np.mean([r["success_straddle"] for r in out])
                        tm = np.mean([r["track_mean"] for r in out])
                        tp = np.mean([r["track_p95"] for r in out])
                        net = np.mean([r["min_dist_network"] < 0.05
                                       for r in out])
                        summary.append((s_uv, s_p, st, sb, ss, tm, tp, net))
                        print(f"{variant}/{name} sigma_uv={s_uv:g} "
                              f"sigma_p={s_p:g}: traverse {st:.3f}  "
                              f"band {sb:.3f}  network {net:.3f}",
                              flush=True)

                head = provenance(args.trials, name, start, offset,
                                  variant, over)
                tag = f"{name}_{variant}"
                with open(os.path.join(out_dir, f"trials_{tag}.csv"),
                          "w") as f:
                    f.write("\n".join(head) + "\n")
                    f.write(",".join(COLUMNS) + "\n")
                    for r in rows:
                        f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")
                with open(os.path.join(out_dir, f"summary_{tag}.csv"),
                          "w") as f:
                    f.write("\n".join(head) + "\n")
                    f.write("sigma_uv,sigma_p,success_traverse,success_band,"
                            "success_straddle,track_mean,track_p95,"
                            "reach_network\n")
                    for row in summary:
                        f.write(",".join(f"{v:.6g}" for v in row) + "\n")
                print(f"written: {out_dir}/trials_{tag}.csv, "
                      f"summary_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
