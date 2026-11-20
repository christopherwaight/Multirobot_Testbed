"""
pilot_gperp.py  (Draft_14 pilot, not a figure producer)

Question: under the 2026-11-16 robot model (delay 0.055 s, gain 0.86), does
the s1 tracker's across-trench loop, whose gain k g_perp d2s1/dn2 reaches
3 pi^4 A = 29 near the saddles against the sampled-loop bound of about 16,
lose performance at g_perp = 1, and does g_perp = 0.5 recover it?

Same protocol as mc_sweep_noise_both_trackers.py (starts, stop rules, seeds,
scoring). Rows: model x tracker x start x sigma_uv.

Run from VF_Robot:  venv/bin/python3 experiments/draft14/pilot_gperp.py --workers 6
Writes experiments/draft14/outputs/pilot_gperp.csv
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)
os.chdir(root)

import experiments._mc_common as mc
from experiments.mc_sweep_noise_both_trackers import cell_specs
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

STARTS = {"S1": (-0.10, 0.30), "S2": (0.15, 0.25), "S3": (0.0, 0.35), "S4": (0.0, 0.0),
          "S5": (0.0, -0.25), "S6": (0.15, -0.15), "S7": (0.10, -0.20)}


def prim_d(c):
    vx, vy = separatrix_logic_c_step(c, v_max=mc.V_MAX)
    return vx * mc.GAIN, vy * mc.GAIN


def prim_s1_g1(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=1.0)
    return vx * mc.GAIN, vy * mc.GAIN


def prim_s1_g05(c):
    vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=0.5)
    return vx * mc.GAIN, vy * mc.GAIN


PRIMS = {"D": prim_d, "s1_g1": prim_s1_g1, "s1_g05": prim_s1_g05}


def worker(spec):
    spec["primitive"] = PRIMS[spec.pop("name")]
    r = mc.run_trial(spec)
    far = (np.hypot(r["final_x"] - SADDLE_FAR[0], r["final_y"] - SADDLE_FAR[1])
           < SADDLE_CONTACT_D and not r["collapsed"])
    return far, far and bool(r["success_straddle"]), r["final_y"] > 0.4, r["track_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clean-trials", type=int, default=50)
    ap.add_argument("--noisy-trials", type=int, default=1000)
    ap.add_argument("--models", default="legacy,2026")
    ap.add_argument("--out", default="pilot_gperp.csv")
    a = ap.parse_args()
    cells = []
    for model in a.models.split(","):
        for name in PRIMS:
            for s in STARTS:
                cells.append((model, name, s, 0.0, a.clean_trials))
            for s in ("S3", "S6"):
                for sig in (0.002, 0.005):
                    cells.append((model, name, s, sig, a.noisy_trials))
    out = os.path.join(here, "outputs", a.out)
    with open(out, "w") as f, Pool(a.workers) as pool:
        f.write("model,tracker,start,sigma_uv,n,success,retention,reversal,track_mean\n")
        for model, name, s, sig, n in cells:
            specs = cell_specs("D" if name == "D" else "s1", sig, 0.0, n, STARTS[s])
            for sp in specs:
                sp.pop("tracker")
                sp["name"] = name
                sp["robot"] = mc.ROBOT_MODELS[model]
            res = pool.map(worker, specs, chunksize=8)
            succ = np.mean([r[0] for r in res])
            ret = np.mean([r[1] for r in res])
            rev = np.mean([r[2] for r in res])
            trk = np.median([r[3] for r in res])
            line = f"{model},{name},{s},{sig},{n},{succ:.4f},{ret:.4f},{rev:.4f},{trk:.4f}"
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
