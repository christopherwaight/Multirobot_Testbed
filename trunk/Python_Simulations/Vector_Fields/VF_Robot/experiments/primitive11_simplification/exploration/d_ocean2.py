import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step
from dvar import d_step
def run(p):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
        path = S.run_traj(field, cl, p, *S.START)
    return path, cl
def mk(band, cq, drift, log):
    return lambda c: tuple(S.CONTROL_GAIN * x for x in d_step(c, S.V_MAX, band, cq, log, drift))
ref, _ = run(lambda c: tuple(S.CONTROL_GAIN * x for x in separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)))
km = lambda a, b: np.hypot((a[:,0]-b[:,0])*111.32, (a[:,1]-b[:,1])*111.32*np.cos(np.radians(34.2)))
for label, band, cq, drift in (("current band, with FLOW_DRIFT (reimplementation)", 'orig', None, True),
                               ("current band, no FLOW_DRIFT", 'orig', None, False),
                               ("qD<1, with FLOW_DRIFT", 'qD', 1.0, True),
                               ("qD<0.3, with FLOW_DRIFT", 'qD', 0.3, True),
                               ("qD<0.03, with FLOW_DRIFT", 'qD', 0.03, True)):
    log = []; path, cl = run(mk(band, cq, drift, log)); q = np.array(cl._qcap_log); d = km(ref, path)
    print(f"{label:52s} max sep (24 h) {d[:144].max():6.2f} km, at 24 h {d[143]:6.2f} km | {dict(Counter(log))} | rho|gradD|/|D| min {q.min():.3f}, <0.03: {(q<0.03).sum()}")
