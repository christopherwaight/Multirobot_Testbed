"""
Closed-loop check of the robot model against the 157 hardware attraction runs
(79 vortex, 78 saddle, 10 s each). Each hardware start is re-run with 50
random formation headings under each (stiction rule, v_c, k, tau) and the final
centroid position is compared with hardware (mean distance, bias, precision),
all in metres.

Run from VF_Robot:  venv/bin/python3 experiments/hyperparameter_study/stiction_hw_check.py
Writes results_stiction/hw_check.csv and prints the table.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vecsim as vs

OUT = os.path.join(HERE, "results_stiction")
REPS = 50
DT = 0.1
# hardware_*_starts_finals.npy hold OptiTrack positions divided by the printed
# map size (66 in), as in combined_distance_from_origin_plotter.m. OptiTrack
# reads metres, and the robots, formation and reconstructed fields all work in
# metres, so undo that division here.
MAP_SCALE = 1.6764


def hardware(field):
    a = np.load(os.path.join(HERE, f"hardware_{field}_starts_finals.npy"))
    start, final = a[:, :2] * MAP_SCALE, a[:, 2:4] * MAP_SCALE
    u0 = -start / np.linalg.norm(start, axis=1, keepdims=True)   # approach direction, target at origin
    return start, final, (final * u0).sum(1)


def run(field, model, v, k, tau, seed):
    start, _, _ = hardware(field.split("_")[0])
    starts = np.repeat(start, REPS, axis=0)
    n = len(starts)
    thetas = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    P = vs.params(n, alpha=np.exp(-DT / tau), vs=v, k=k)
    M, _ = vs.simulate(vs.make_field(field), "attract", "fit", P, 10.0, dt=DT, seed=seed,
                       starts=starts, thetas=thetas, stiction_model=model)
    return np.stack([M["final_x"], M["final_y"]], 1) - vs.make_field(field).target, M["final_proj"], M["moved"]


def stats(final):
    """Mean final distance, bias (distance of the mean rest point) and
    precision (RMS distance about the mean rest point), as in the paper's table."""
    mu = final.mean(0)
    return (np.hypot(*final.T).mean(), np.hypot(*mu),
            np.sqrt(((final - mu) ** 2).sum(1).mean()))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    hw = {f: hardware(f) for f in ("vortex", "saddle")}
    hdr = ("model", "v", "k", "tau", "field", "mean_mm", "bias_mm", "precision_mm", "along_approach_mm")
    rows = []
    for f in ("vortex", "saddle"):
        m, b, p = stats(hw[f][1])
        rows.append(("hardware", "", "", "", f, m * 1e3, b * 1e3, p * 1e3, hw[f][2].mean() * 1e3))
    configs = [("legacy", 0.025, 1.0, 0.28), ("deadzone", 0.025, 1.0, 0.28)]
    for v in (0.025, 0.035):
        for tau in (0.20, 0.28):
            configs.append(("coulomb", v, 1.0, tau))
    for seed, (model, v, k, tau) in enumerate(configs):
        for f in ("vortex", "saddle", "vortex_nn", "saddle_nn"):
            final, proj, moved = run(f, model, v, k, tau, seed)
            m, b, p = stats(final)
            rows.append((model, v, k, tau, f, m * 1e3, b * 1e3, p * 1e3, proj.mean() * 1e3))
    print(" ".join(f"{h:>12s}" for h in hdr))
    for row in rows:
        print(" ".join(f"{x:>12}" if isinstance(x, str) else f"{x:12.3f}" for x in row), flush=True)
    with open(os.path.join(OUT, "hw_check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
