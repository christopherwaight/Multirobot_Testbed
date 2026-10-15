import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
with contextlib.redirect_stdout(io.StringIO()):
    cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
def qat(x, y):
    cl.reset(x, y)
    u, v = P._sample_vector_at_robots(cl); rel = P._get_relative_positions(cl)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    s1, g, e1, e2, r = P._strain_quantities(tu, tv)
    rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
    return s1, np.linalg.norm(g), r, rho * np.linalg.norm(g) / max(abs(s1), 1e-12), rho
xs = np.linspace(-0.95, 0.95, 77); ys = np.linspace(-0.48, 0.48, 41)
Q = np.zeros((len(ys), len(xs))); S1 = np.zeros_like(Q)
with contextlib.redirect_stdout(io.StringIO()):
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            s1, gn, r, q, rho = qat(x, y); Q[i, j] = q; S1[i, j] = s1
print("rho =", round(rho, 4))
for c in (0.02, 0.05, 0.1, 0.2):
    m = (Q < c) & (S1 < 0)
    pts = [(round(xs[j], 2), round(ys[i], 2)) for i, j in zip(*np.where(m))]
    print(f"q < {c}: {m.sum()} grid points;", pts[:12], "..." if len(pts) > 12 else "")
print("\nq along x = 0:")
for y in (0.45, 0.35, 0.25, 0.1, 0.02, -0.02, -0.1, -0.25, -0.35, -0.45, -0.48, -0.5):
    s1, gn, r, q, _ = qat(0.0, y); print(f"  y {y:+.2f}  s1 {s1:+.3f} |g| {gn:.3f} r {r:.3f} q {q:.3f}")
print("\nq near gyre centres and walls:")
for p in ((0.5, 0), (-0.5, 0), (0.45, 0.05), (0.55, -0.05), (0.5, 0.48), (0.25, -0.48), (1.0, 0.0), (0.95, -0.48)):
    s1, gn, r, q, _ = qat(*p); print(f"  {p}  s1 {s1:+.3f} |g| {gn:.3f} r {r:.3f} q {q:.3f}")
