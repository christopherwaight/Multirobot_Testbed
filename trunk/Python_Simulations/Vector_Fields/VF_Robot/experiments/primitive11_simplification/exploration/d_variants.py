import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
from collections import Counter
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
EPS = 1e-9

def d_step(c, v_max, band, cq, log):
    """Logic C with: band = 'orig' (eps_raw/eps_dim) or 'qD' (q_D < cq); FLOW_DRIFT
    removed (band with non-saddle H falls through to CAPTURE); fallback renamed CAPTURE."""
    u, v = P._sample_vector_at_robots(c); rel = P._get_relative_positions(c)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    D = P._det_value(tu, tv); g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv)
    lam, V = np.linalg.eigh(H); f = np.array([tu[0], tv[0]])
    rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
    if band == 'orig':
        on = abs(D) < M.EPS_RAW or abs(D) / max(np.linalg.norm(H, 'fro'), 1e-12) < M.EPS_DIM
    else:
        on = abs(D) / (rho * np.linalg.norm(g) + 0.5 * rho**2 * np.max(np.abs(lam)) + 1e-15) < cq
    saddle = lam[0] * lam[1] < -EPS and np.min(np.abs(lam)) >= EPS
    if not saddle:
        log.append('CAPTURE')
        c0 = -(V[:, 0] @ g) / (lam[0] if abs(lam[0]) > EPS else EPS)
        c1 = -(V[:, 1] @ g) / (lam[1] if abs(lam[1]) > EPS else EPS)
        d = v_max*np.tanh(c0/v_max)*V[:, 0] + v_max*np.tanh(c1/v_max)*V[:, 1]
        return float(d[0]), float(d[1])
    i_neg = 0 if lam[0] < lam[1] else 1; i_pos = 1 - i_neg
    if on:
        log.append('FLOW')
        vn, vp = V[:, i_neg], V[:, i_pos]
        d = v_max*np.tanh(float(f @ vn)/v_max)*vn + v_max*np.tanh(-float(g @ vp)/lam[i_pos]/v_max)*vp
        return float(d[0]), float(d[1])
    vn = V[:, i_neg]
    if float(g @ vn) > 0: vn = -vn
    if float(f @ vn) > 0:
        log.append('SLIDE'); c0 = -(V[:, 0] @ g)/abs(lam[0]); c1 = -(V[:, 1] @ g)/abs(lam[1])
    else:
        log.append('ATTRACT'); c0 = -(V[:, 0] @ g)/lam[0]; c1 = -(V[:, 1] @ g)/lam[1]
    d = v_max*np.tanh(c0/v_max)*V[:, 0] + v_max*np.tanh(c1/v_max)*V[:, 1]
    return float(d[0]), float(d[1])

configs = [("current band (eps_raw, eps_dim)", 'orig', None)] + [(f"q_D < {cq}", 'qD', cq) for cq in (1.0, 0.3, 0.1, 0.03, 0.0)]
for label, band, cq in configs:
    print(f"\n{label}")
    for name, sx, sy in M.STARTS:
        with contextlib.redirect_stdout(io.StringIO()):
            cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
        cl.reset(sx, sy); log = []
        prim = lambda c: tuple(M.GAIN * x for x in d_step(c, M.V_MAX, band, cq, log))
        contact = -1
        for k in range(M.SIM_STEPS):
            cl.move(prim); cx, cy = cl.get_centroid()
            if abs(cx) > 1.0 or abs(cy) > 0.52: break
            if contact < 0 and np.hypot(cx, cy + 0.5) < 0.06: contact = k
            elif contact >= 0 and k - contact >= M.POST_SADDLE_STEPS: break
        h = np.asarray(cl.get_center_history())
        far = float(np.min(np.hypot(h[:,0], h[:,1] + 0.5))); kf = int(np.argmin(np.hypot(h[:,0], h[:,1] + 0.5)))
        xs = np.abs(h[:, 0]); kb = next((i for i in range(len(xs)-10) if (xs[i:i+10] < 0.05).all()), -1)
        maxx = np.max(xs[kb:kf]) if 0 <= kb < kf else float('nan')
        cnt = Counter(log)
        print(f"  {name}: far saddle closest {far:.3f}, max|x| on ride {maxx:.3f}, end ({h[-1,0]:+.2f},{h[-1,1]:+.2f}) | "
              + " ".join(f"{m} {cnt[m]}" for m in ("FLOW", "SLIDE", "ATTRACT", "CAPTURE")))
