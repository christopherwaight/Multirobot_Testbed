import numpy as np
import src.control.pentagon_primitives as P
EPS_RAW, EPS_DIM = 1e-3, 0.025
EPS = 1e-9

def d_step(c, v_max, band, cq, log, drift=False, fallback='signed'):
    """Logic C with: band = 'orig' (eps_raw/eps_dim) or 'qD' (q_D < cq); FLOW_DRIFT
    removed (band with non-saddle H falls through to CAPTURE); fallback renamed CAPTURE."""
    u, v = P._sample_vector_at_robots(c); rel = P._get_relative_positions(c)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    D = P._det_value(tu, tv); g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv)
    lam, V = np.linalg.eigh(H); f = np.array([tu[0], tv[0]])
    qcap = None
    rho = np.sort(np.linalg.norm(rel, axis=1))[1:].mean()
    c._qcap_log = getattr(c, '_qcap_log', []); c._qcap_log.append(rho * np.linalg.norm(g) / max(abs(D), 1e-15))
    if band == 'orig':
        on = abs(D) < EPS_RAW or abs(D) / max(np.linalg.norm(H, 'fro'), 1e-12) < EPS_DIM
    elif band == 'crest':
        on = lam[0] * lam[1] < 0 and np.linalg.norm(np.linalg.solve(H, g)) < cq * rho
    else:
        on = abs(D) / (rho * np.linalg.norm(g) + 0.5 * rho**2 * np.max(np.abs(lam)) + 1e-15) < cq
    saddle = lam[0] * lam[1] < -EPS and np.min(np.abs(lam)) >= EPS
    if drift and on and (lam[0] * lam[1] >= -EPS or np.min(np.abs(lam)) < EPS):
        log.append('FLOW_DRIFT')
        n = np.linalg.norm(f)
        if n < 1e-12: return 0.0, 0.0
        sc = v_max * np.tanh(n / v_max) / n
        return float(f[0] * sc), float(f[1] * sc)
    if not saddle:
        log.append('CAPTURE')
        if fallback == 'abs':
            c0 = -(V[:, 0] @ g) / max(abs(lam[0]), EPS)
            c1 = -(V[:, 1] @ g) / max(abs(lam[1]), EPS)
        else:
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

