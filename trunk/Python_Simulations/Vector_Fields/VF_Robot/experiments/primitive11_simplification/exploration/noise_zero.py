import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from collections import Counter
import noise_norelease as NR
from variant_q import oecs_separatrix_step_q
import main_separatrix_traverse as M

def zero_hold(c, **kw):
    return oecs_separatrix_step_q(c, c_out=np.inf, hold_zero=True, **kw)
NR.PRIMS["zero, no release"] = zero_hold

if __name__ == "__main__":
    for sig in (0.002, 0.005):
        jobs = [("zero, no release", sx, sy, sig, 0.0, 1000 + 7919*t)
                for _, sx, sy in M.STARTS for t in range(50)]
        with Pool(7) as pool:
            res = pool.map(NR.trial, jobs, chunksize=10)
        c = Counter(r[1] for r in res); cap = [r for r in res if r[4]]
        ed = np.array([r[2] for r in cap])
        print(f"\nsigma_uv = {sig} (350 trials): far {c['far']} near {c['near']} corner {c['corner']} exited {c['exited']} elsewhere {c['elsewhere']}"
              f" | captured {len(cap)}, entry dist to nearest saddle median {np.median(ed):.3f}, >0.1: {(ed>0.1).sum()}, >0.3: {(ed>0.3).sum()}")
        for name, sx, sy in M.STARTS:
            rr = [r for r in res if (r[0], ) and abs(r[2] or 0) >= 0 and True][:0]
        per = {}
        for (name, sx, sy) in M.STARTS:
            pass
