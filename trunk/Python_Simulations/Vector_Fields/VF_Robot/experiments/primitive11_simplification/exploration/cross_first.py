import os, sys, io, contextlib
VF = os.getcwd()
sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
from itertools import groupby
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField

src = open(P.__file__).read()
anchor = "    # -- 1. ACQUIRE: no structure yet ---"
assert src.count(anchor) == 1
cross = '''    _tp = getattr(cluster, '_oecs_prev_tangent', None)
    if r < r_band and _tp is None:
        _log('CROSS', flow=flow.copy())
        fn = float(np.linalg.norm(flow))
        t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
        step_tan = v_max * np.tanh(1.0) * t_hat
        sx_, sy_ = _descend_s1()
        cluster._oecs_prev_tangent = t_hat
        d_ = step_tan + np.array([sx_, sy_])
        return float(d_[0]), float(d_[1])

'''
ns = {}
exec(compile(src.replace(anchor, cross + anchor), "variant", "exec"), ns)
variant = ns["oecs_separatrix_step"]

for label, fn in (("as written", P.oecs_separatrix_step), ("CROSS first", variant)):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(0.0, 0.0)
    def prim(c):
        vx, vy = fn(c, v_max=M.V_MAX, g_perp=M.G_PERP, s_trim=M.S_TRIM,
                    r_band=M.R_BAND, s_capture=None)
        return vx * M.GAIN, vy * M.GAIN
    for k in range(M.SIM_STEPS):
        cl.move(prim)
        cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.52: break
    d = [x for x in cl.diagnostics if 'mode' in x]
    h = np.asarray(cl.get_center_history())
    print(f"\n{label}: {len(d)} steps, ends at ({h[-1,0]:+.3f}, {h[-1,1]:+.3f})")
    if d[0]['mode'] == 'CROSS':
        print("   step 0 fitted flow:", d[0]['flow'], " s1", round(d[0]['s1'],4), " r", round(d[0]['r'],4))
    k = 0
    for m, g in groupby(d, key=lambda x: x['mode']):
        g = list(g)
        print(f"   steps {k:3d}-{k+len(g)-1:3d} {m:8s} s1 {g[0]['s1']:+.4f} -> {g[-1]['s1']:+.4f}  r {g[0]['r']:.4f} -> {g[-1]['r']:.4f}  centroid ({h[k,0]:+.3f},{h[k,1]:+.3f}) -> ({h[k+len(g)-1,0]:+.3f},{h[k+len(g)-1,1]:+.3f})")
        k += len(g)
