"""Primitive 11 variant: CROSS checked first, ACQUIRE removed, CAPTURE by the
dimensionless test q = rho*|grad s1|/|s1| with hysteresis (c_in, c_out)."""
import numpy as np
import src.control.pentagon_primitives as P

C_IN, C_OUT = 0.03, 0.06

def oecs_separatrix_step_q(cluster, v_max=0.04, g_perp=1.0, r_band=0.05,
                           c_in=C_IN, c_out=C_OUT, capture=True, hold_zero=False, seed_init=False, **_ignored):
    u_arr, v_arr = P._sample_vector_at_robots(cluster)
    rel = P._get_relative_positions(cluster)
    theta_u, theta_v = P._fit_vector_quadratic(rel, u_arr, v_arr)
    s1, grad_s1, e1, e2, r = P._strain_quantities(theta_u, theta_v)
    flow = np.array([theta_u[0], theta_v[0]])
    rho = float(np.sort(np.linalg.norm(rel, axis=1))[1:].mean())
    g_norm = float(np.linalg.norm(grad_s1))
    q = rho * g_norm / max(abs(s1), 1e-12)

    def _log(mode, **extra):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            e = {'mode': mode, 's1': float(s1), 'r': float(r), 'q': q}
            e.update(extra); diag.append(e)

    def _descend_s1():
        a = -g_perp * grad_s1
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return 0.0, 0.0
        scale = v_max * np.tanh(n / v_max) / n
        return float(a[0] * scale), float(a[1] * scale)

    t_prev = getattr(cluster, '_oecs_prev_tangent', None)
    prev_hat = (t_prev / np.linalg.norm(t_prev)
                if t_prev is not None and np.linalg.norm(t_prev) > 1e-12 else None)

    # 1. CROSS
    if r < r_band and prev_hat is None:
        _log('CROSS')
        fn = float(np.linalg.norm(flow))
        t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
        step_tan = v_max * np.tanh(1.0) * t_hat
        px, py = _descend_s1()
        cluster._oecs_prev_tangent = t_hat
        d = step_tan + np.array([px, py])
        return float(d[0]), float(d[1])

    # 2. CAPTURE (q test, hysteresis)
    captured = getattr(cluster, '_q_captured', False) if capture else False
    if captured and q > c_out:
        captured = False
        cluster._q_captured = False
        _log('RELEASE_EVENT')
        cluster.diagnostics.pop() if getattr(cluster, 'diagnostics', None) else None
        cluster._q_releases = getattr(cluster, '_q_releases', 0) + 1
    if capture and (captured or (s1 < 0 and q < c_in)):
        cluster._q_captured = True
        _log('CAPTURE')
        return (0.0, 0.0) if hold_zero else _descend_s1()

    # 3. TRACK (unchanged from Primitive 11, margin gate off)
    d1 = abs(float(grad_s1 @ e1)); d2 = abs(float(grad_s1 @ e2))
    if seed_init and prev_hat is None:
        fn = float(np.linalg.norm(flow))
        t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
    else:
        t_raw = e1 if d1 > d2 else e2
        sign_ref = prev_hat if prev_hat is not None else flow
        t_hat = t_raw if float(t_raw @ sign_ref) >= 0 else -t_raw
    g_perp_vec = grad_s1 - float(grad_s1 @ t_hat) * t_hat
    gpn = float(np.linalg.norm(g_perp_vec))
    if gpn < 1e-12:
        step_perp = np.zeros(2)
    else:
        step_perp = v_max * np.tanh(-g_perp * gpn / v_max) * (g_perp_vec / gpn)
    step_tan = v_max * np.tanh(1.0) * t_hat
    cluster._oecs_prev_tangent = t_hat
    _log('TRACK', tangent_id=('e1' if d1 > d2 else 'e2'))
    d = step_perp + step_tan
    return float(d[0]), float(d[1])
