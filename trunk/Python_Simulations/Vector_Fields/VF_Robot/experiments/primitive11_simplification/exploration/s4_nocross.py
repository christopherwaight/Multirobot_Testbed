import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import groupby
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from variant_q import oecs_separatrix_step_q

for label, rb in (("with CROSS (r_band 0.05)", 0.05), ("no CROSS", -1.0)):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(0.0, 0.0)
    def prim(c):
        vx, vy = oecs_separatrix_step_q(c, v_max=M.V_MAX, g_perp=M.G_PERP, r_band=rb, c_out=np.inf)
        return vx * M.GAIN, vy * M.GAIN
    for k in range(M.SIM_STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: break
    d = [x for x in cl.diagnostics if 'mode' in x]; h = np.asarray(cl.get_center_history())
    print(f"\n{label}: {len(d)} steps, end ({h[-1,0]:+.4f}, {h[-1,1]:+.4f})")
    k = 0
    for m, g in groupby(d, key=lambda x: x['mode']):
        g = list(g)
        tid = sorted(set(x.get('tangent_id') for x in g if x.get('tangent_id')))
        print(f"   steps {k:3d}-{k+len(g)-1:3d} {m:8s} r {g[0]['r']:.4f} -> {g[-1]['r']:.4f}  y {h[k,1]:+.3f} -> {h[k+len(g)-1,1]:+.3f}  {tid if tid else ''}")
        k += len(g)
    print("   first 6 steps (x, y, tangent):", [(round(h[i,0],4), round(h[i,1],4), d[i].get('tangent_id', d[i]['mode'])) for i in range(6)])
