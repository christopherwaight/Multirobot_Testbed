import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from dvar import d_step
for cq in (0.5, 0.25, 0.1, 0.05):
    out = []
    for name, sx, sy in M.STARTS:
        with contextlib.redirect_stdout(io.StringIO()):
            cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
        cl.reset(sx, sy); log = []
        prim = lambda c: tuple(M.GAIN * x for x in d_step(c, M.V_MAX, 'crest', cq, log, False))
        contact = -1
        for k in range(M.SIM_STEPS):
            cl.move(prim); cx, cy = cl.get_centroid()
            if abs(cx) > 1.0 or abs(cy) > 0.52: break
            if contact < 0 and np.hypot(cx, cy + 0.5) < 0.06: contact = k
            elif contact >= 0 and k - contact >= M.POST_SADDLE_STEPS: break
        h = np.asarray(cl.get_center_history()); far = float(np.min(np.hypot(h[:,0], h[:,1] + 0.5)))
        out.append(f"{name} {'ok' if far < 0.06 else 'MISS'} ({far:.3f}, FLOW {Counter(log)['FLOW']})")
    print(f"gyre, crest within {cq} rho: " + "; ".join(out))
