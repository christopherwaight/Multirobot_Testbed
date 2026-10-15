"""Flow-saddle endpoint detection (critical-point test) for both trackers.
fs test: det J < 0 and |J^-1 v0| < cfs * rho, from the same single fit."""
import numpy as np
import src.control.pentagon_primitives as P
EPS = 1e-9

def _fit(c):
    u, v = P._sample_vector_at_robots(c); rel = P._get_relative_positions(c)
    tu, tv = P._fit_vector_quadratic(rel, u, v)
    rho = float(np.mean(np.sort(np.linalg.norm(rel, axis=1))[1:]))
    return tu, tv, rho

def _flow_saddle(tu, tv, rho, cfs):
    J = np.array([[tu[1], tu[2]], [tv[1], tv[2]]]); f = np.array([tu[0], tv[0]])
    dJ = np.linalg.det(J)
    if dJ >= 0: return False, np.inf
    dist = float(np.linalg.norm(np.linalg.solve(J, f)))
    return dist < cfs * rho, dist

def _sat_vec(a, v_max):
    n = float(np.linalg.norm(a))
    if n < 1e-12: return 0.0, 0.0
    s = v_max * np.tanh(n / v_max) / n
    return float(a[0] * s), float(a[1] * s)

def s1_step(c, v_max=0.04, g_perp=1.0, entry='q', c_in=0.03, cfs=1.0, log=None):
    tu, tv, rho = _fit(c)
    s1, g, e1, e2, r = P._strain_quantities(tu, tv); f = np.array([tu[0], tv[0]])
    if entry == 'q':
        hit = s1 < 0 and rho * np.linalg.norm(g) / max(abs(s1), 1e-12) < c_in
    else:
        hit = _flow_saddle(tu, tv, rho, cfs)[0]
    if getattr(c, '_cap', False) or hit:
        c._cap = True; log is not None and log.append('CAPTURE')
        return _sat_vec(-g_perp * g, v_max)
    d1, d2 = abs(float(g @ e1)), abs(float(g @ e2))
    tp = getattr(c, '_tp', None); ref = tp if tp is not None else f
    t = e1 if d1 > d2 else e2; t = t if float(t @ ref) >= 0 else -t
    gp = g - float(g @ t) * t; gn = float(np.linalg.norm(gp))
    sp = np.zeros(2) if gn < 1e-12 else v_max * np.tanh(-g_perp * gn / v_max) * gp / gn
    c._tp = t; log is not None and log.append('TRACK')
    d = sp + v_max * np.tanh(1.0) * t
    return float(d[0]), float(d[1])

def d_step(c, v_max=0.04, c_crest=0.1, capture=None, cfs=1.0, log=None):
    tu, tv, rho = _fit(c)
    g = P._det_gradient(tu, tv); H = P._det_hessian(tu, tv); f = np.array([tu[0], tv[0]])
    lam, V = np.linalg.eigh(H)
    def absnewton():
        c0 = -(V[:, 0] @ g) / max(abs(lam[0]), EPS); c1 = -(V[:, 1] @ g) / max(abs(lam[1]), EPS)
        d = v_max*np.tanh(c0/v_max)*V[:, 0] + v_max*np.tanh(c1/v_max)*V[:, 1]
        return float(d[0]), float(d[1])
    if capture == 'fs' and (getattr(c, '_cap', False) or _flow_saddle(tu, tv, rho, cfs)[0]):
        c._cap = True; log is not None and log.append('CAPTURE'); return absnewton()
    if lam[0] * lam[1] >= -EPS or np.min(np.abs(lam)) < EPS:
        log is not None and log.append('ATTRACT_FALLBACK'); return absnewton()
    i_neg = 0 if lam[0] < lam[1] else 1; i_pos = 1 - i_neg
    if float(np.linalg.norm(np.linalg.solve(H, g))) < c_crest * rho:
        vn, vp = V[:, i_neg], V[:, i_pos]
        d = v_max*np.tanh(float(f @ vn)/v_max)*vn + v_max*np.tanh(-float(g @ vp)/lam[i_pos]/v_max)*vp
        log is not None and log.append('FLOW'); return float(d[0]), float(d[1])
    vn = V[:, i_neg]
    if float(g @ vn) > 0: vn = -vn
    if float(f @ vn) > 0:
        c0 = -(V[:, 0] @ g)/abs(lam[0]); c1 = -(V[:, 1] @ g)/abs(lam[1]); m = 'SLIDE'
    else:
        c0 = -(V[:, 0] @ g)/lam[0]; c1 = -(V[:, 1] @ g)/lam[1]; m = 'ATTRACT'
    log is not None and log.append(m)
    d = v_max*np.tanh(c0/v_max)*V[:, 0] + v_max*np.tanh(c1/v_max)*V[:, 1]
    return float(d[0]), float(d[1])
