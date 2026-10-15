import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
with contextlib.redirect_stdout(io.StringIO()):
    cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
np.random.seed(0)
for lab, suv, sp in (("sigma_uv 0.002", 0.002, 0), ("sigma_p 0.002", 0, 0.002)):
    print(lab)
    for y in (-0.25, -0.35, -0.45, -0.5):
        cl.reset(0.0, y); cl.measurement_noise_std = suv; cl.position_noise_std = sp
        qs, gs = [], []
        for _ in range(2000):
            u, v = P._sample_vector_at_robots(cl); rel = P._get_relative_positions(cl)
            tu, tv = P._fit_vector_quadratic(rel, u, v)
            s1, g, *_ = P._strain_quantities(tu, tv)
            rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
            gs.append(np.linalg.norm(g)); qs.append(rho * np.linalg.norm(g) / abs(s1))
        qs = np.array(qs); gs = np.array(gs)
        print(f"  (0,{y:+.2f}) |grad s1| median {np.median(gs):.3f}  q median {np.median(qs):.3f}, 5-95% [{np.percentile(qs,5):.3f}, {np.percentile(qs,95):.3f}], P(q<0.03) {np.mean(qs<0.03):.2f}, P(q>0.06) {np.mean(qs>0.06):.2f}")
