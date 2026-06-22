"""
Sink vector fields.

All three variants (sink1, sink2, sink3) have a nominal center at (0, 0).
When loaded with config/fields/sink.yaml (eps > 0), the center wobbles:
    x_cp(t) = eps * cos(omega * t)
    y_cp(t) = eps * sin(omega * t)
With eps=0 (default) the center is fixed at the origin.

roi_finder2_simple is a control helper (not a field) and is left untouched
at the bottom of the file.
"""
import numpy as np


_DEFAULT_EPS   = 0.0
_DEFAULT_OMEGA = 0.628318530717958


def _wobble_center(t, config):
    cfg   = config or {}
    eps   = cfg.get("eps",   _DEFAULT_EPS)
    omega = cfg.get("omega", _DEFAULT_OMEGA)
    return eps * np.cos(omega * t), eps * np.sin(omega * t)


def sink1(x, y, t=0.0, config=None):
    x_sink, y_sink = _wobble_center(t, config)
    x_centre = x - x_sink
    y_centre = y - y_sink

    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    strength = -1.0
    v_r = strength * r

    u = v_r * x_centre / r
    v = v_r * y_centre / r

    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    return u, v


def sink2(x, y, t=0.0, config=None):
    x_sink, y_sink = _wobble_center(t, config)
    x_centre = x - x_sink
    y_centre = y - y_sink

    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    strength = -1.0
    u = strength * x_centre / r
    v = strength * y_centre / r
    return u, v


def sink3(x, y, t=0.0, config=None):
    center_x, center_y = _wobble_center(t, config)

    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    u = -0.15 * x_centre / r2**2
    v = -0.15 * y_centre / r2**2
    return u, v


sink1.config_name = "sink"
sink2.config_name = "sink"
sink3.config_name = "sink"


def roi_finder2_simple(
    cluster,
    mode="auto",
    tiny=1e-12, frac=0.5,
    alpha=0.3, beta_gain=0.5,
    k_r=0.5, k_t=0.3,
    r_d=0.15,
):
    """
    Estimate the critical point from plane-fit Jacobian and drive toward it
    (or orbit it).  This helper is not a field function; it is a control
    primitive bundled here for historical reasons.
    """
    import numpy as np

    readings = cluster.sample_field_at_robots()
    pos = np.array([r.position for r in cluster.robots])
    u_vals = np.array([rd[0] for rd in readings])
    v_vals = np.array([rd[1] for rd in readings])

    A = np.column_stack([pos[:, 0], pos[:, 1], np.ones(len(pos))])
    try:
        theta_u, _, _, _ = np.linalg.lstsq(A, u_vals, rcond=None)
        theta_v, _, _, _ = np.linalg.lstsq(A, v_vals, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    a, b, c = theta_u
    d, e, f = theta_v

    J = np.array([[a, b], [d, e]])
    det = np.linalg.det(J)
    if abs(det) < tiny:
        return 0.0, 0.0

    h = np.array([c, f])
    p_star = -np.linalg.solve(J, h)

    p_c = cluster.get_centroid()
    r_vec = p_star - p_c
    r_norm = np.linalg.norm(r_vec)

    if mode == "converge" or (mode == "auto" and r_norm > r_d * (1 + frac)):
        if r_norm < tiny:
            return 0.0, 0.0
        vx = alpha * r_vec[0]
        vy = alpha * r_vec[1]
    else:
        if r_norm < tiny:
            return 0.0, 0.0
        r_hat = r_vec / r_norm
        r_perp = np.array([-r_hat[1], r_hat[0]])
        err_r = r_norm - r_d
        vx = k_t * r_perp[0] - k_r * err_r * r_hat[0]
        vy = k_t * r_perp[1] - k_r * err_r * r_hat[1]

    return vx, vy
