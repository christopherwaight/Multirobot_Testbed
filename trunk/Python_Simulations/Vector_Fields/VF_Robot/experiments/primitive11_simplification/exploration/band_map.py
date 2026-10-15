import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
with contextlib.redirect_stdout(io.StringIO()):
    cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
def at(x, y):
    cl.reset(x, y)
    u, v = P._sample_vector_at_robots(cl); rel = P._get_relative_positions(cl)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    D = P._det_value(tu, tv); g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv)
    rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
    Hn = np.linalg.norm(H, 'fro'); lam = np.linalg.eigvalsh(H)
    band = abs(D) < M.EPS_RAW or abs(D) / max(Hn, 1e-12) < M.EPS_DIM
    qD = abs(D) / (rho * np.linalg.norm(g) + 0.5 * rho**2 * np.max(np.abs(lam)) + 1e-15)
    return D, band, qD, lam
print("along x = 0:   y      D        band   q_D      lam1     lam2")
for y in (0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.0, -0.03, -0.05, -0.1, -0.2, -0.3, -0.45, -0.5):
    D, b, q, lam = at(0.0, y); print(f"            {y:+.2f}  {D:+.5f}  {str(b):5s}  {q:7.3f}  {lam[0]:+.3f}  {lam[1]:+.3f}")
ys = np.linspace(-0.5, 0.5, 201)
for lab, idx in (("band", 1), ("q_D<1", 2)):
    on = [y for y in ys if (at(0.0, y)[1] if idx == 1 else at(0.0, y)[2] < 1)]
    print(f"{lab} fires on x=0 for y in [{min(on):+.3f}, {max(on):+.3f}] ({len(on)} of 201 points)" if on else f"{lab}: never")
xs = np.linspace(-0.3, 0.3, 61)
for lab, idx in (("band", 1), ("q_D<1", 2)):
    on = [x for x in xs if (at(x, 0.0)[1] if idx == 1 else at(x, 0.0)[2] < 1)]
    print(f"{lab} fires across y=0 for x in [{min(on):+.3f}, {max(on):+.3f}]" if on else f"{lab}: never across")
