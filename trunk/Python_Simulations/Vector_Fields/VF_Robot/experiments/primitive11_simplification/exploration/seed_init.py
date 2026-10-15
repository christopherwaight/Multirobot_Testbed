import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import groupby
from multiprocessing import Pool
from collections import Counter
import main_separatrix_traverse as M
import noise_norelease as NR
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from variant_q import oecs_separatrix_step_q

def seeded(c, **kw):
    kw.pop('r_band', None)
    return oecs_separatrix_step_q(c, c_out=np.inf, r_band=-1.0, seed_init=True, **kw)
NR.PRIMS["seeded"] = seeded

if __name__ == "__main__":
    print("CLEAN")
    for name, sx, sy in M.STARTS:
        with contextlib.redirect_stdout(io.StringIO()):
            cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
        cl.reset(sx, sy)
        def prim(c):
            vx, vy = seeded(c, v_max=M.V_MAX, g_perp=M.G_PERP)
            return vx * M.GAIN, vy * M.GAIN
        for k in range(M.SIM_STEPS):
            cl.move(prim); cx, cy = cl.get_centroid()
            if abs(cx) > 1.0 or abs(cy) > 0.60: break
        d = [x for x in cl.diagnostics if 'mode' in x]; h = np.asarray(cl.get_center_history())
        segs, k = [], 0
        for m, g in groupby(d, key=lambda x: x['mode']):
            n = len(list(g)); segs.append(f"{m} {k}-{k+n-1}"); k += n
        print(f"  {name} end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}) " + " > ".join(segs))
    for sig in (0.002, 0.005):
        jobs = [("seeded", sx, sy, sig, 0.0, 1000 + 7919*t) for _, sx, sy in M.STARTS for t in range(50)]
        with Pool(7) as pool:
            res = pool.map(NR.trial, jobs, chunksize=10)
        c = Counter(r[1] for r in res)
        s4 = Counter(r[1] for r, j in zip(res, jobs) if j[1] == 0.0 and j[2] == 0.0)
        print(f"NOISY sigma_uv = {sig} (350): far {c['far']} near {c['near']} corner {c['corner']} exited {c['exited']} elsewhere {c['elsewhere']}   | S4 only (50): {dict(s4)}")
