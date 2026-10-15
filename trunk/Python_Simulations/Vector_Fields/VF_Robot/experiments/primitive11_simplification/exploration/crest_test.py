import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step
from dvar import d_step
# 1. where the crest test fires along x = 0
with contextlib.redirect_stdout(io.StringIO()):
    cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
print("along x=0: y, Newton distance / rho, lam1*lam2 sign")
on_y = []
for y in np.linspace(-0.5, 0.5, 201):
    cl.reset(0.0, y)
    u, v = P._sample_vector_at_robots(cl); rel = P._get_relative_positions(cl)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv); lam = np.linalg.eigvalsh(H)
    rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
    nd = np.linalg.norm(np.linalg.solve(H, g)) / rho
    if lam[0] * lam[1] < 0 and nd < 1: on_y.append(y)
    if round(y, 3) in (0.5, 0.45, 0.3, 0.1, 0.05, 0.0, -0.05, -0.1, -0.3, -0.45, -0.48, -0.5):
        print(f"   {y:+.3f}  {nd:8.3f}  {'saddle' if lam[0]*lam[1] < 0 else 'definite'}")
print("crest test fires on x=0 for y in:", (f"[{min(on_y):+.3f}, {max(on_y):+.3f}], {len(on_y)} of 201" if on_y else "never"),
      "| separate pieces:", sum(1 for i in range(1, len(on_y)) if on_y[i] - on_y[i-1] > 0.006) + (1 if on_y else 0))
# 2. clean gyre starts
print("\nclean gyre, crest test as FLOW band, FLOW_DRIFT kept")
for name, sx, sy in M.STARTS:
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy); log = []
    prim = lambda c: tuple(M.GAIN * x for x in d_step(c, M.V_MAX, 'crest', 1.0, log, True))
    contact = -1
    for k in range(M.SIM_STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.52: break
        if contact < 0 and np.hypot(cx, cy + 0.5) < 0.06: contact = k
        elif contact >= 0 and k - contact >= M.POST_SADDLE_STEPS: break
    h = np.asarray(cl.get_center_history()); far = float(np.min(np.hypot(h[:,0], h[:,1] + 0.5)))
    print(f"  {name}: far saddle closest {far:.3f}, end ({h[-1,0]:+.2f},{h[-1,1]:+.2f}) | {dict(Counter(log))}")
# 3. SBC
def run(p):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        c = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
        return S.run_traj(field, c, p, *S.START)
ref = run(lambda c: tuple(S.CONTROL_GAIN * x for x in separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)))
log = []; path = run(lambda c: tuple(S.CONTROL_GAIN * x for x in d_step(c, S.V_MAX, 'crest', 1.0, log, True)))
d = np.hypot((ref[:,0]-path[:,0])*111.32, (ref[:,1]-path[:,1])*111.32*np.cos(np.radians(34.2)))
print(f"\nSBC, crest test: max separation (24 h) {d[:144].max():.2f} km, at 24 h {d[143]:.2f} km, end ({path[-1,0]:.3f}N, {path[-1,1]:.3f}) | {dict(Counter(log))}")
