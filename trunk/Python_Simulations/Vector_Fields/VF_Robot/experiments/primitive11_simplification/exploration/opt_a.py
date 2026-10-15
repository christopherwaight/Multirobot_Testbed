import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from collections import Counter
import noise_norelease as NR
from variant_q import oecs_separatrix_step_q
import main_separatrix_traverse as M
def opt_a(c, **kw):
    kw.pop('r_band', None)
    return oecs_separatrix_step_q(c, c_out=np.inf, r_band=-1.0, seed_init=False, **kw)
NR.PRIMS["opt_a"] = opt_a
if __name__ == "__main__":
    for sig in (0.002, 0.005):
        jobs = [("opt_a", sx, sy, sig, 0.0, 1000 + 7919*t) for _, sx, sy in M.STARTS for t in range(50)]
        with Pool(7) as pool:
            res = pool.map(NR.trial, jobs, chunksize=10)
        c = Counter(r[1] for r in res)
        s4 = Counter(r[1] for r, j in zip(res, jobs) if j[1] == 0.0 and j[2] == 0.0)
        print(f"opt_a sigma_uv={sig} (350): far {c['far']} near {c['near']} corner {c['corner']} exited {c['exited']} elsewhere {c['elsewhere']} | S4 (50): {dict(s4)}")
