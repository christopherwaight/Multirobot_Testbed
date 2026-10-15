import os, sys, io, contextlib
root = os.getcwd(); sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments"))
import numpy as np
import main_ocean_hfr_2km_shared_start as S
import src.control.pentagon_primitives as P
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
with contextlib.redirect_stdout(io.StringIO()):
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
    sx, sy = S._latlon_to_world(*S.START, field.config)
    S.set_formation_scale(cl, S.FORMATION_SCALE); cl.reset(sx, sy, heading_offset=np.radians(S.HEADING_DEG)); field.reset_clock()
u, v = P._sample_vector_at_robots(cl); rel = P._get_relative_positions(cl)
tu, tv = P._fit_vector_quadratic(rel, u, v)
s1, g, e1, e2, r = P._strain_quantities(tu, tv)
f = np.array([tu[0], tv[0]]); d1, d2 = abs(g @ e1), abs(g @ e2)
t = e1 if d1 > d2 else e2
ang = np.degrees(np.arccos(abs(f @ t) / np.linalg.norm(f)))
print(f"step 0: |flow| {np.linalg.norm(f):.4f}, flow dir {f/np.linalg.norm(f)}, argmax tangent {t} (d1 {d1:.3f}, d2 {d2:.3f}), angle flow-to-tangent axis {ang:.1f} deg, r {r:.3f}")
