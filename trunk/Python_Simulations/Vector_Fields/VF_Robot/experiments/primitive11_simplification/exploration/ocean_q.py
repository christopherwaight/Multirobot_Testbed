import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import groupby
import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import oecs_separatrix_step
from variant_q import oecs_separatrix_step_q
paths, diags = {}, {}
for label, fn in (("original", oecs_separatrix_step), ("variant", oecs_separatrix_step_q)):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                             stiction_threshold=S.STICTION_THRESHOLD)
    def p(c):
        vx, vy = fn(c, v_max=S.V_MAX, g_perp=S.G_PERP, s_trim=S.S_TRIM, r_band=S.R_BAND,
                    g_capture=S.G_CAPTURE, s_capture=None)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN
    with contextlib.redirect_stdout(io.StringIO()):
        paths[label] = S.run_traj(field, cl, p, *S.START)
    d = [x for x in cl.diagnostics if 'mode' in x]; diags[label] = d
    print(label, " > ".join(f"{m} x{len(list(g))}" for m, g in groupby(x['mode'] for x in d)))
q = np.array([x['q'] for x in diags['variant']])
print(f"variant q along ride: min {q.min():.3f} at step {q.argmin()}, 5th pct {np.percentile(q,5):.3f}, median {np.median(q):.3f}; steps with q < 0.03: {(q<0.03).sum()}, q < 0.06: {(q<0.06).sum()}")
a, b = paths['original'], paths['variant']
print("max path difference (deg lat/lon):", np.max(np.abs(a - b)), " identical:", np.array_equal(a, b))
