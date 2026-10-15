import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import groupby
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from variant_q import oecs_separatrix_step_q

def run(fn, sx, sy):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy)
    def prim(c):
        vx, vy = fn(c, v_max=M.V_MAX, g_perp=M.G_PERP, s_trim=M.S_TRIM,
                    r_band=M.R_BAND, s_capture=None)
        return vx * M.GAIN, vy * M.GAIN
    for k in range(M.SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: break
    return cl

for name, sx, sy in M.STARTS:
    out = {}
    for label, fn in (("original", P.oecs_separatrix_step), ("variant ", oecs_separatrix_step_q)):
        cl = run(fn, sx, sy)
        d = [x for x in cl.diagnostics if 'mode' in x]
        h = np.asarray(cl.get_center_history()); out[label] = h
        segs, k = [], 0
        for m, g in groupby(d, key=lambda x: x['mode']):
            n = len(list(g)); segs.append(f"{m} {k}-{k+n-1}"); k += n
        extra = ""
        if 'q' in d[0]:
            qs = [x['q'] for x in d if x['mode'] == 'CAPTURE']
            extra = f"  q at capture entry {qs[0]:.3f}, releases {getattr(cl, '_q_releases', 0)}" if qs else "  never captured"
        ymin = h[:,1].min()
        print(f"{name} {label} lowest y {ymin:+.3f}, end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}) " + " > ".join(segs) + extra)
    a, b = out["original"], out["variant "]
    L = min(len(a), len(b)); print(f"     max path difference over first {L} steps: {np.max(np.linalg.norm(a[:L]-b[:L], axis=1)):.4f}")
