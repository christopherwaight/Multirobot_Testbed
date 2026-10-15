"""Noise-free s1 control behind Draft_12 Sec. IV-C ("without noise ... 99.2%"):
10^4 random-heading trials per start, same harness and stop rule as
mc_sweep_noise_both_trackers.py. Writes overnight/noisefree_s1.csv."""
import os, sys
from multiprocessing import Pool
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root); os.chdir(root)
import numpy as np
import experiments.mc_sweep_noise_both_trackers as sw
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D
from experiments.mc_mechanism_starts import STARTS
if __name__ == "__main__":
    out = os.path.join(root, "experiments", "primitive11_simplification", "overnight", "noisefree_s1.csv")
    with open(out, "w") as f, Pool(7) as pool:
        f.write("start,x,y,trials,success,success_straddle\n")
        for name, (x, y) in STARTS.items():
            res = pool.map(sw._worker, sw.cell_specs("s1", 0.0, 0.0, 10000, (x, y)), chunksize=16)
            far = [np.hypot(r["final_x"] - SADDLE_FAR[0], r["final_y"] - SADDLE_FAR[1]) < SADDLE_CONTACT_D
                   and not r["collapsed"] for r in res]
            strad = [a and bool(r["success_straddle"]) for a, r in zip(far, res)]
            f.write(f"{name},{x},{y},{len(res)},{np.mean(far):.4f},{np.mean(strad):.4f}\n"); f.flush()
            print(name, f"success {np.mean(far):.4f} straddle {np.mean(strad):.4f}", flush=True)
