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
acq = """    # -- 1. ACQUIRE: no structure yet -----------------------------------
    if not banded:
        _log('ACQUIRE')
        return _descend_s1()
"""
assert src.count(acq) == 1
cross = '''    _tp = getattr(cluster, '_oecs_prev_tangent', None)
    if r < r_band and _tp is None:
        _log('CROSS')
        fn = float(np.linalg.norm(flow))
        t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
        step_tan = v_max * np.tanh(1.0) * t_hat
        sx_, sy_ = _descend_s1()
        cluster._oecs_prev_tangent = t_hat
        d_ = step_tan + np.array([sx_, sy_])
        return float(d_[0]), float(d_[1])
'''
ns = {}
exec(compile(src.replace(acq, cross), "variant", "exec"), ns)
variant = ns["oecs_separatrix_step"]

for name, sx, sy in M.STARTS:
    for label, fn in (("as written ", P.oecs_separatrix_step), ("variant    ", variant)):
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
            if abs(cx) > 1.0 or abs(cy) > 0.52: break
        d = [x for x in cl.diagnostics if 'mode' in x]
        h = np.asarray(cl.get_center_history())
        segs, k = [], 0
        for m, g in groupby(d, key=lambda x: x['mode']):
            n = len(list(g)); segs.append(f"{m} {k}-{k+n-1}"); k += n
        print(f"{name} {label} end ({h[-1,0]:+.3f},{h[-1,1]:+.3f})  " + " > ".join(segs))
