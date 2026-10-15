import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
import fsad
import main_ocean_hfr_2km_shared_start as S
import traverse_objectivity_demo as T
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.fields.environments.Double_Gyre import double_gyre_static
from src.fields.environments.Rotating_Frame import make_rotating_frame, pull_back_trajectory
km = lambda a, b: np.hypot((a[:,0]-b[:,0])*111.32, (a[:,1]-b[:,1])*111.32*np.cos(np.radians(34.2)))
def sbc(step):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
        log = []; path = S.run_traj(field, cl, lambda c: tuple(S.CONTROL_GAIN * x for x in step(c, log)), *S.START)
    return path, log
print("SBC")
for name, base, var in (("s1", lambda c, l: fsad.s1_step(c, S.V_MAX, S.G_PERP, 'q', log=l), lambda c, l: fsad.s1_step(c, S.V_MAX, S.G_PERP, 'fs', log=l)),
                        ("D",  lambda c, l: fsad.d_step(c, S.V_MAX, capture=None, log=l), lambda c, l: fsad.d_step(c, S.V_MAX, capture='fs', log=l))):
    pb, lb = sbc(base); pv, lv = sbc(var); d = km(pb, pv)
    k0 = next((k for k, m in enumerate(lv) if m == 'CAPTURE'), None)
    print(f"  {name}: flow-saddle capture fired at step {k0} ({'never' if k0 is None else f'{k0/6:.1f} h, at ({pv[k0,0]:.3f}N, {pv[k0,1]:.3f})'}); "
          f"max separation from repo version {d.max():.2f} km; repo end ({pb[-1,0]:.3f}N, {pb[-1,1]:.3f}), variant end ({pv[-1,0]:.3f}N, {pv[-1,1]:.3f})")
print("\nROTATING frame, start (0.05, 0.40), Omega = 0.23, 400 steps")
def rot_run(step, rotating):
    fn = make_rotating_frame(double_gyre_static, T.OMEGA_ROT) if rotating else double_gyre_static
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(fn); cl = PentagonCluster(T.FORMATION_CONFIG, field)
    cl.reset(*T.START); field.reset_clock(); log = []
    for k in range(T.SIM_STEPS):
        cl.move(lambda c: tuple(T.CONTROL_GAIN * x for x in step(c, log))); field.step(cl.timestep)
        cx, cy = cl.get_centroid()
        if rotating:
            th = T.OMEGA_ROT * (k + 1) * T.DT
            cx, cy = (np.cos(th) * cx + np.sin(th) * cy, -np.sin(th) * cx + np.cos(th) * cy)
        if abs(cx) > 1.05 or abs(cy) > 0.60: break
    h = np.asarray(cl.get_center_history())
    if rotating: h = pull_back_trajectory(h, (np.arange(len(h)) + 1) * T.DT, T.OMEGA_ROT)
    return h, log
for name, step in (("s1, q capture", lambda c, l: fsad.s1_step(c, T.V_MAX, T.G_PERP, 'q', log=l)),
                   ("s1, flow-saddle capture", lambda c, l: fsad.s1_step(c, T.V_MAX, T.G_PERP, 'fs', log=l)),
                   ("D, flow-saddle capture", lambda c, l: fsad.d_step(c, T.V_MAX, capture='fs', log=l))):
    hi, li = rot_run(step, False); hr, lr = rot_run(step, True)
    ki = next((k for k, m in enumerate(li) if m == 'CAPTURE'), None); kr = next((k for k, m in enumerate(lr) if m == 'CAPTURE'), None)
    L = min(len(hi), len(hr)); gap = np.linalg.norm(hi[L-1] - hr[L-1])
    print(f"  {name:24s} inertial: capture step {ki}, end ({hi[-1,0]:+.3f},{hi[-1,1]:+.3f}) | rotating: capture step {kr}, end ({hr[-1,0]:+.3f},{hr[-1,1]:+.3f}) | final gap {gap:.4f}")
