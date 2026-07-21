"""
rescore_single_target.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_Separatrix_5A.tex
  Corrects: tab:mc_success panel (b) and its surrounding noise-robustness prose.
  Reads:  experiments/outputs/mc_separatrix/trials_fixed.csv (Controller 1, unchanged)
          experiments/outputs/mc_oecs_traverse/trials_fixed.csv (Controller 2, rescored)
  Writes: experiments/outputs/mc_oecs_traverse/summary_single_target.csv

WHY THIS SCRIPT EXISTS (found 2026-07-21, reviewing the finished 10k sweep)
  Both Monte Carlo sweeps start every trial from the SAME fixed straddling point,
  (0, 0.35). At that point the true, noise-free ambient flow points DOWN, toward the
  FAR saddle (0, -0.5), 0.85 away -- not toward the near saddle (0, +0.5), only 0.15
  away. Controller 1 (the D tracker) always targets the far saddle, by design, at
  every noise level; its own summary_fixed.csv was already scored this way.

  Controller 2 (the s1 tracker, oecs_separatrix_step) resolves the sign of its ride
  tangent from the ambient flow ONCE, on the very first tracking step
  (Section sec:oecs_controller of the paper; this is what keeps TRACK frame-objective
  thereafter). At (0, 0.35) that one-time sign read is only weakly determined: the
  differential u-signal across the 6 sample points spans about 0.12, comparable in
  size to sigma_uv = 0.005 measurement noise applied independently to each robot
  BEFORE the fit. Traced directly: at sigma_uv = 0 the committed direction matches
  Controller 1's own target in 10000/10000 trials; between sigma_uv = 0.001 and 0.005
  the resolved direction FLIPS almost completely (9190/10000 far-saddle at 0.001, only
  7/10000 at 0.005), and stays flipped (>94% near-saddle) at every higher noise level
  tested. mc_sweep_oecs_traverse.py's own summary (success_traverse column) scores a
  trial as successful if it reaches EITHER saddle, which was the right call while
  characterizing the controller's own CAPTURE mechanism in isolation, but it means the
  reported "task success" silently switches from "traversed to the far saddle" (low
  noise) to "traversed to the near saddle" (moderate-to-high noise) -- two different
  tasks stitched into one column, with the easier one taking over exactly where the
  harder one's success rate would otherwise have kept falling.

  The paper's own framing is TWO SURROGATES FOR THE SAME OBJECT, the separatrix, not
  two different terminal targets. The fair, single-surrogate comparison scores BOTH
  controllers against the SAME fixed target, the far saddle (0, -0.5): the one point a
  true noise-free ride from (0, 0.35) actually reaches either way. Under this scoring,
  a trial whose tangent-sign flipped to the near saddle is NOT an easier success, it is
  a FAILURE to ride the separatrix, exactly as a straddle-retention failure or a
  formation collapse already is.

WHAT THIS SCRIPT DOES (pure re-derivation, no new simulation)
  Reads the existing per-trial CSV (already has final_x, final_y, collapsed for all
  10000 trials/cell); recomputes success as contact with the FAR saddle specifically
  (hypot(final_x, final_y - (-0.5)) < SADDLE_CONTACT_D and not collapsed). Aggregates
  per (sigma_uv, sigma_p) cell exactly as mc_sweep_oecs_traverse.py's own summary does.
  Also prints, per cell, the fraction of trials whose final_y sign matches the far
  saddle's (the "task-flip rate" itself), so the sharp threshold is directly visible
  and reproducible rather than quoted from an ad hoc one-off computation.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/rescore_single_target.py
"""
import csv
import os

SADDLE_FAR = (0.0, -0.5)     # the single target: what a true noise-free ride reaches
SADDLE_CONTACT_D = 0.06

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS_PATH = os.path.join(project_root, "experiments", "outputs",
                          "mc_oecs_traverse", "trials_fixed.csv")
OUT_PATH = os.path.join(project_root, "experiments", "outputs",
                       "mc_oecs_traverse", "summary_single_target.csv")


def read_trials(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line.strip())
    header = rows[0].split(",")
    idx = {name: i for i, name in enumerate(header)}
    data = []
    for line in rows[1:]:
        parts = line.split(",")
        data.append({name: parts[i] for name, i in idx.items()})
    return data


def main():
    trials = read_trials(TRIALS_PATH)

    cells = {}
    for t in trials:
        key = (float(t["sigma_uv"]), float(t["sigma_p"]))
        cells.setdefault(key, []).append(t)

    rows_out = []
    print(f"{'sigma_uv':>9} {'sigma_p':>8} {'far_saddle_rate':>16} "
          f"{'old_either_saddle':>18} {'single_target_success':>22} "
          f"{'straddle':>9}")
    for key in sorted(cells.keys()):
        s_uv, s_p = key
        group = cells[key]
        n = len(group)

        old_success = sum(1 for t in group
                          if t["success_traverse"] == "1") / n

        far_hits = 0
        far_hits_straddled = 0
        matches_far_sign = 0
        for t in group:
            fx, fy = float(t["final_x"]), float(t["final_y"])
            collapsed = t["collapsed"] == "1"
            d_far = (fx - SADDLE_FAR[0]) ** 2 + (fy - SADDLE_FAR[1]) ** 2
            reached_far = d_far ** 0.5 < SADDLE_CONTACT_D and not collapsed
            if reached_far:
                far_hits += 1
                # success_straddle in the CSV was computed against the OLD
                # "either saddle" stop condition: for a trial that reached
                # the far saddle, that stop point coincides with reaching
                # THIS target too, so the straddle-to-stop history is exactly
                # the straddle-to-far-saddle history and can be reused as is.
                # For a trial that reached the NEAR saddle instead, the run
                # stopped there and never continued on to see whether it
                # would also straddle en route to the far one -- there is no
                # straddle-to-far-saddle data for that trial, so it is scored
                # as 0, not carried over from the old either-saddle number
                # (this is the fix: straddle can now only be nonzero on
                # trials that already succeeded on the corrected target,
                # so success_straddle_corrected <= success_single_target
                # always, and P(straddle | success) is well-defined).
                if t["success_straddle"] == "1":
                    far_hits_straddled += 1
            if fy < 0:   # far saddle's y is negative; near saddle's is positive
                matches_far_sign += 1

        single_target_success = far_hits / n
        far_sign_rate = matches_far_sign / n
        straddle = far_hits_straddled / n

        rows_out.append({
            "sigma_uv": s_uv, "sigma_p": s_p,
            "success_single_target": round(single_target_success, 4),
            "success_either_saddle_OLD": round(old_success, 4),
            "far_saddle_sign_rate": round(far_sign_rate, 4),
            "success_straddle": round(straddle, 4),
        })
        print(f"{s_uv:>9} {s_p:>8} {far_sign_rate:>15.1%} "
              f"{old_success:>17.1%} {single_target_success:>21.1%} "
              f"{straddle:>8.1%}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
