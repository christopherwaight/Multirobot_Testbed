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
paths = {}
for label in ("current", "qD<1", "qD<0.03"):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
    log = []
    if label == "current":
        def p(c):
            vx, vy = separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)
            return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN
    else:
        cq = 1.0 if label == "qD<1" else 0.03
        def p(c, cq=cq):
            vx, vy = d_step(c, S.V_MAX, 'qD', cq, log)
            return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN
    with contextlib.redirect_stdout(io.StringIO()):
        path = S.run_traj(field, cl, p, *S.START)
    paths[label] = path
    modes = log if log else [d['mode'] for d in cl.diagnostics if 'mode' in d]
    q = np.array(getattr(cl, '_qcap_log', [np.nan]))
    print(f"{label:8s} end ({path[-1,0]:.4f}N, {path[-1,1]:.4f}), at 24 h ({path[143,0]:.4f}N, {path[143,1]:.4f}) | modes {dict(Counter(modes))}"
          + (f" | rho|gradD|/|D| min {np.nanmin(q):.3f}, steps below 0.03: {(q<0.03).sum()}, below 0.1: {(q<0.1).sum()}" if log else ""))
ref = paths["current"]
for l in ("qD<1", "qD<0.03"):
    b = paths[l]; d = np.hypot((ref[:,0]-b[:,0])*111.32, (ref[:,1]-b[:,1])*111.32*np.cos(np.radians(34.2)))
    print(f"{l} vs published D path: max separation {d[:144].max():.2f} km over the first 24 h, at 24 h {d[143]:.2f} km")
