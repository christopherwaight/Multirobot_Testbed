"""
mc_sweep_oecs_traverse.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the objective-tracker success-rate table of the Results section
          (tab:mc_success panel (b)), replacing the core-seeker sweep of
          mc_sweep_oecs.py now that Primitive 11 (oecs_separatrix_step) is
          Controller 2 in the paper.
  Reads:  nothing. Writes experiments/outputs/mc_oecs_traverse/
          trials_fixed.csv (one row per trial) and summary_fixed.csv.

EXPERIMENT
  Primitive 11 (oecs_separatrix_step) over the same 2D noise grid
  sigma_uv x sigma_p as mc_sweep_oecs.py and mc_sweep_separatrix.py, from
  the same fixed straddling start (0, 0.35), random initial heading per
  trial.

  TARGET IS BOTH SADDLES, NOT ONE (found empirically, not assumed): an
  early dev run against a single hardcoded TOP-saddle target showed
  0% "success" at sigma_uv=sigma_p=0, which traced to a real property of
  the controller, not a bug -- the tangent's initial sign is resolved
  ONCE, on the first TRACK step, by projecting onto the ambient flow at
  the start (the one flow-consulting step that keeps the ride
  frame-objective thereafter). At (0, 0.35) the flow points DOWN
  (toward the far saddle, 0.85 away) even though the near saddle is only
  0.15 away, so the ride commits downward from step 0 and the team
  captures at the BOTTOM saddle every time at zero noise -- confirmed by
  direct trace, not a truncated/early-exit artifact (steps column reads
  500 for every zero-noise row, final_y in [-0.521, -0.486]). This flow-
  direction sign is ALSO noise-sensitive right at this exact start (a
  70/30 bottom/top split appears at sigma_uv=0.001, full flip to ~100%
  top by sigma_uv=0.005+): a legitimate, reportable property of the
  flow-seeded tangent rule, not a defect in the noise sweep. So success
  here means capture at EITHER saddle (target passed as a list;
  _mc_common.run_trial checks distance to the nearest), matching what
  the CAPTURE branch is actually designed to hold: whichever 2D minimum
  of s1 the ride's committed direction leads to.

  EXIT BOX WIDENED FROM THE _mc_common.py DEFAULT (found empirically, not
  assumed): the shared default y_exit = 0.52 is sized for Logic C, which
  deliberately continues PAST the saddle onto the wall trench, so "clearly
  past y=0.5" is the right stop condition for that controller. This
  controller's CAPTURE dynamics produce a small transient overshoot past
  y=0.5 before settling back (the same saturated gradient descent that
  holds the well also approaches it, so it can cross the target itself by
  a few cm before the restoring term catches it) -- traced directly: of 54
  trials scored as sigma_p=0.005 failures under the default y_exit=0.52,
  every single one exited AT y in [0.520, 0.529], a hair past the box,
  while still converging (most already had |x| well under 0.5); relaxing
  only the exit box (same 500-step budget, same controller, nothing else
  changed) captured 30/54 of them. y_exit = 0.60 is used here, wide enough
  for the observed overshoot with margin, without disabling the exit
  check that still catches genuine divergence.

  AUTO-EXTEND doubles the top sigma_uv until the worst-corner success
  rate falls below 10%, matching the other sweeps.

Run (development, 1000 trials/cell):
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/mc_sweep_oecs_traverse.py --trials 1000 --workers 8
Final run for the paper: --trials 10000 (see FINAL_SWEEPS_HOWTO.md).
"""
import argparse
import os
import sys
import subprocess
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.control.pentagon_primitives import oecs_separatrix_step
import experiments._mc_common as mc

SIGMA_UV_BASE = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_P_BASE = [0.0, 0.005, 0.01, 0.02, 0.05]
EXTEND_UNTIL = 0.10
MAX_EXTENSIONS = 3

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
TARGET = [(0.0, 0.5), (0.0, -0.5)]   # either saddle counts as capture
Y_EXIT = 0.60   # widened from _mc_common's 0.52 default; see docstring

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "mc_oecs_traverse")
os.makedirs(OUT_DIR, exist_ok=True)


def _prim(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP,
                                  s_trim=S_TRIM, r_band=R_BAND,
                                  g_capture=G_CAPTURE, s_capture=None)
    return vx * mc.GAIN, vy * mc.GAIN


def _worker(spec):
    spec["primitive"] = _prim
    return mc.run_trial(spec)


def cell_specs(sigma_uv, sigma_p, n_trials):
    base = int(1e6 * sigma_uv * 1000 + 1e3 * sigma_p * 1000) % (2**31)
    return [{"sigma_uv": sigma_uv, "sigma_p": sigma_p,
             "start": mc.FIXED_START, "target": TARGET, "y_exit": Y_EXIT,
             "seed": (base + 7919 * t) % (2**31)} for t in range(n_trials)]


TRIAL_COLS = ["sigma_uv", "sigma_p", "seed", "start_x", "start_y", "heading",
              "t_band", "steps", "success_traverse", "success_band",
              "success_straddle", "first_straddle", "collapsed",
              "track_mean", "track_p95", "shape_rms_max", "effort",
              "final_x", "final_y"]

# Per-cell checkpoint.  A 10,000-trial run takes hours and previously wrote
# nothing until the very end, so any interruption threw the whole run away
# (lost a 42-minute run at cell 41 of 60 on 2026-07-24).  Every completed
# cell is now appended here immediately; a restart with the same --trials
# skips the cells already present.  Delete this file to force a clean run.
CKPT_PATH = os.path.join(OUT_DIR, "checkpoint_trials.csv")


def _fmt(r):
    return ",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                    for c in TRIAL_COLS)


# Column indices for the five quantities the summary needs, so the
# checkpoint can be aggregated by streaming without building dicts.
_IDX = {c: i for i, c in enumerate(TRIAL_COLS)}
_SUMMARY_COLS = ("success_traverse", "success_band", "success_straddle",
                 "track_mean", "track_p95")


def load_checkpoint(n_trials):
    """Return per-cell summary aggregates for cells already complete.

    Streams the checkpoint and keeps only running sums, never the trial
    rows themselves. Holding all rows in memory is what killed two 10,000
    trial runs on an 8 GB machine (600,000 records as Python dicts is
    1-2 GB resident, on top of eight worker processes); the rows are
    already durable on disk, so the parent has no reason to carry them.

    A cell counts only if it holds exactly n_trials rows, so a cell that
    was interrupted mid-write is discarded and simply rerun.
    """
    summary = {}
    if not os.path.exists(CKPT_PATH):
        return summary

    # Deduplicate on (cell, seed). Two sweeps writing the same checkpoint
    # concurrently (which happened on 2026-07-25, when a stale run was left
    # alive alongside its replacement) append the same trials twice. Trials
    # are deterministic in the seed, so the duplicates are identical and the
    # data is still correct, but a cell would then hold 20,000 rows and be
    # rejected by the exactly-n_trials rule below. Keying on the seed makes
    # the loader idempotent, so a duplicated cell counts once.
    counts, sums, seen = {}, {}, set()
    with open(CKPT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("sigma_uv,"):
                continue
            parts = line.split(",")
            if len(parts) != len(TRIAL_COLS):
                continue
            try:
                key = (float(parts[_IDX["sigma_uv"]]),
                       float(parts[_IDX["sigma_p"]]))
                vals = [float(parts[_IDX[c]]) for c in _SUMMARY_COLS]
            except ValueError:
                continue
            trial_key = (parts[_IDX["sigma_uv"]], parts[_IDX["sigma_p"]],
                         parts[_IDX["seed"]])
            if trial_key in seen:
                continue
            seen.add(trial_key)
            counts[key] = counts.get(key, 0) + 1
            acc = sums.setdefault(key, [0.0] * len(_SUMMARY_COLS))
            for i, v in enumerate(vals):
                acc[i] += v

    for key in sorted(counts):
        n = counts[key]
        if n != n_trials:
            print(f"  checkpoint: cell {key} has {n} of {n_trials} "
                  f"trials, will rerun", flush=True)
            continue
        summary[key] = tuple(s / n for s in sums[key])
    if summary:
        print(f"  checkpoint: resuming, {len(summary)} cells already done",
              flush=True)
    return summary


def write_trials_from_checkpoint(path, header, n_trials):
    """Stream the checkpoint into trials_fixed.csv, keeping only complete cells.

    Never holds more than one cell's rows in memory.
    """
    # Same (cell, seed) dedup as load_checkpoint: a concurrently written
    # checkpoint can hold each trial twice, and the output must carry one
    # copy of each.
    counts, seen = {}, set()
    with open(CKPT_PATH) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(TRIAL_COLS):
                continue
            try:
                key = (float(parts[_IDX["sigma_uv"]]),
                       float(parts[_IDX["sigma_p"]]))
            except ValueError:
                continue
            trial_key = (parts[_IDX["sigma_uv"]], parts[_IDX["sigma_p"]],
                         parts[_IDX["seed"]])
            if trial_key in seen:
                continue
            seen.add(trial_key)
            counts[key] = counts.get(key, 0) + 1
    keep = {k for k, n in counts.items() if n == n_trials}

    written, emitted = 0, set()
    with open(path, "w") as out, open(CKPT_PATH) as f:
        out.write(header)
        out.write(",".join(TRIAL_COLS) + "\n")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(TRIAL_COLS):
                continue
            try:
                key = (float(parts[_IDX["sigma_uv"]]),
                       float(parts[_IDX["sigma_p"]]))
            except ValueError:
                continue
            trial_key = (parts[_IDX["sigma_uv"]], parts[_IDX["sigma_p"]],
                         parts[_IDX["seed"]])
            if key in keep and trial_key not in emitted:
                emitted.add(trial_key)
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
    return written


def run_cells(pool, cells, n_trials, summary):
    for s_uv, s_p in cells:
        if (s_uv, s_p) in summary:
            st, sb, ss, tm, tp = summary[(s_uv, s_p)]
            print(f"  sigma_uv={s_uv:<6} sigma_p={s_p:<6} "
                  f"capture={st:5.1%} band={sb:5.1%} straddle={ss:5.1%} "
                  f"track_mean={tm:.4f} p95={tp:.4f}  (from checkpoint)",
                  flush=True)
            continue
        specs = cell_specs(s_uv, s_p, n_trials)
        out = list(pool.imap_unordered(_worker, specs, chunksize=16))
        # Append before summarizing: if this dies mid-cell the partial rows
        # are discarded on reload (count != n_trials) and the cell reruns.
        # `out` is dropped at the end of the iteration; the parent keeps only
        # the aggregate, so peak memory is one cell rather than the whole run.
        with open(CKPT_PATH, "a") as f:
            for r in out:
                f.write(_fmt(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        st = np.mean([r["success_traverse"] for r in out])
        sb = np.mean([r["success_band"] for r in out])
        ss = np.mean([r["success_straddle"] for r in out])
        tm = np.mean([r["track_mean"] for r in out])
        tp = np.mean([r["track_p95"] for r in out])
        summary[(s_uv, s_p)] = (st, sb, ss, tm, tp)
        print(f"  sigma_uv={s_uv:<6} sigma_p={s_p:<6} "
              f"capture={st:5.1%} band={sb:5.1%} straddle={ss:5.1%} "
              f"track_mean={tm:.4f} p95={tp:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sigma_uv_levels = list(SIGMA_UV_BASE)
    summary = load_checkpoint(args.trials)

    with Pool(args.workers) as pool:
        print(f"Base grid ({len(sigma_uv_levels)}x{len(SIGMA_P_BASE)} "
              f"cells, {args.trials} trials/cell):")
        run_cells(pool, [(u, p) for u in sigma_uv_levels
                         for p in SIGMA_P_BASE], args.trials, summary)

        for _ in range(MAX_EXTENSIONS):
            corner = summary[(sigma_uv_levels[-1], SIGMA_P_BASE[-1])]
            if corner[0] <= EXTEND_UNTIL:
                break
            new_uv = sigma_uv_levels[-1] * 2
            sigma_uv_levels.append(new_uv)
            print(f"Corner success {corner[0]:.1%} > {EXTEND_UNTIL:.0%}, "
                  f"extending sigma_uv to {new_uv}:")
            run_cells(pool, [(new_uv, p) for p in SIGMA_P_BASE],
                      args.trials, summary)

    header = (f"# generated_by: experiments/mc_sweep_oecs_traverse.py\n"
              f"# git_commit: {commit}\n# date: {stamp}\n"
              f"# trials_per_cell: {args.trials}  steps: {mc.SIM_STEPS}  "
              f"controller: oecs_separatrix_step (Primitive 11)\n"
              f"# g_perp: {G_PERP}  s_trim: {S_TRIM}  r_band: {R_BAND}  "
              f"g_capture: {G_CAPTURE}  target: {TARGET}\n"
              f"# start: {mc.FIXED_START}  band_x: {mc.BAND_X}  "
              f"collapse_rms: {mc.COLLAPSE_RMS}\n")

    n_written = write_trials_from_checkpoint(
        os.path.join(OUT_DIR, "trials_fixed.csv"), header, args.trials)

    with open(os.path.join(OUT_DIR, "summary_fixed.csv"), "w") as f:
        f.write(header)
        f.write("sigma_uv,sigma_p,success_capture,success_band,"
                "success_straddle,track_mean,track_p95\n")
        for (u, p), (st, sb, ss, tm, tp) in sorted(summary.items()):
            f.write(f"{u},{p},{st:.4f},{sb:.4f},{ss:.4f},"
                    f"{tm:.5f},{tp:.5f}\n")

    print(f"Wrote {OUT_DIR}/trials_fixed.csv ({n_written} rows) "
          f"and summary_fixed.csv")


if __name__ == "__main__":
    main()
