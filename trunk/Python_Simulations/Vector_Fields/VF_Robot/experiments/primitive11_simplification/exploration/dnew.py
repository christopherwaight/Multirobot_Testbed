"""Scratch D tracker, three modes (2026-10-09 screen): CAPTURE by the Newton step
to the fitted stagnation point (attraction law of [10]), FLOW at a crest, and
TRACK, which merges SLIDE and ATTRACT by orienting w1 along the flow and has no
ATTRACT_FALLBACK. abs_l2 selects |lambda2| or signed lambda2 in the across
channel. merge_only=True keeps the repo's capture action and fallback and only
merges SLIDE/ATTRACT, to check the merge reproduces the repo bit for bit."""
import numpy as np
from src.control.pentagon_primitives import (
    _sample_vector_at_robots, _get_relative_positions, _fit_vector_quadratic,
    _det_gradient, _det_hessian)

EPS = 1e-9


def d_step(c, v_max=0.04, c_crest=0.1, abs_l2=True, merge_only=False, log=None,
           newton=None, fallback=None):
    newton = (not merge_only) if newton is None else newton
    fallback = merge_only if fallback is None else fallback
    u, v = _sample_vector_at_robots(c)
    rel = _get_relative_positions(c)
    tu, tv = _fit_vector_quadratic(rel, u, v)
    g = _det_gradient(tu, tv)
    H = _det_hessian(tu, tv)
    rho = float(np.mean(np.sort(np.linalg.norm(rel, axis=1))[1:]))
    f = np.array([tu[0], tv[0]])
    lam, V = np.linalg.eigh(H)
    sat = lambda x: v_max * np.tanh(x / v_max)

    def _log(m):
        if log is not None:
            log.append(m)

    def _descend():
        c0 = -(V[:, 0] @ g) / max(abs(lam[0]), EPS)
        c1 = -(V[:, 1] @ g) / max(abs(lam[1]), EPS)
        d = sat(c0) * V[:, 0] + sat(c1) * V[:, 1]
        return float(d[0]), float(d[1])

    J = np.array([[tu[1], tu[2]], [tv[1], tv[2]]])
    detJ = np.linalg.det(J)
    at_saddle = detJ < 0 and float(np.linalg.norm(np.linalg.solve(J, f))) < rho
    if getattr(c, '_dnew_cap', False) or at_saddle:
        c._dnew_cap = True
        _log('CAPTURE')
        if not newton:
            return _descend()
        if abs(detJ) < EPS:
            return 0.0, 0.0
        d = sat(-np.linalg.solve(J, f))
        return float(d[0]), float(d[1])

    if fallback and (lam[0] * lam[1] >= -EPS or np.min(np.abs(lam)) < EPS):
        _log('ATTRACT_FALLBACK')
        return _descend()

    w1, w2 = V[:, 0], V[:, 1]
    l1, l2 = lam
    l2d = (abs(l2) if abs_l2 else l2)
    l2d = l2d if abs(l2d) >= EPS else np.copysign(EPS, l2d)
    across = -(w2 @ g) / l2d

    if l1 * l2 < 0 and float(np.linalg.norm(np.linalg.solve(H, g))) < c_crest * rho:
        along = f @ w1
        _log('FLOW')
    else:
        if f @ w1 < 0:
            w1 = -w1
        along = abs(w1 @ g) / max(abs(l1), EPS)
        _log('TRACK' if l1 * l2 < 0 else 'TRACK_DEFINITE')
    d = sat(along) * w1 + sat(across) * w2
    return float(d[0]), float(d[1])
