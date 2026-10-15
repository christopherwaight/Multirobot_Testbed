"""Save the s1 tracker's Santa Barbara centroid path (lat, lon) for the Fig. 9 trial.
Usage: ocean_path.py <out.npy> [old]   'old' runs the pre-rewrite Primitive 11."""
import os, sys, io, contextlib, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
if len(sys.argv) > 2 and sys.argv[2] == "old":
    spec = importlib.util.spec_from_file_location("p_old", os.path.join(here, "pentagon_primitives_pre_rewrite.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
else:
    import src.control.pentagon_primitives as mod
with contextlib.redirect_stdout(io.StringIO()):
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                         stiction_threshold=S.STICTION_THRESHOLD)
def p(c):
    vx, vy = mod.oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP, s_trim=S.S_TRIM,
                                      r_band=S.R_BAND, g_capture=S.G_CAPTURE, s_capture=None)
    return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN
with contextlib.redirect_stdout(io.StringIO()):
    path = S.run_traj(field, cl, p, *S.START)
np.save(sys.argv[1], path)
modes = [d['mode'] for d in cl.diagnostics if 'mode' in d]
print(f"saved {sys.argv[1]}: {len(path)} points, end {path[-1]}, modes {sorted(set(modes))}")
