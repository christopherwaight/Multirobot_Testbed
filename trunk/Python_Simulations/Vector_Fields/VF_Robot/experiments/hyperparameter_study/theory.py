"""
Closed-form and reduced-order predictions for the hyperparameter study.
Nothing here runs the cluster simulator.

ORBIT (Eq. 11 with an exact estimate, i.e. any linear field or the oracle)
    The tangential command k_t t rotates at k_t / r. The robot lag
    v' = alpha v + (1 - alpha) u delays it by a phase `lag` (gain G), which
    turns part of k_t outward; each straight sampled step of length k_t dt
    also moves the cluster outward by (k_t dt)^2 / (2 r). The radial loop
    cancels both at a steady radius r* > r_d:
        k_r (r* - r_d) cos(lag) = V sin(lag) + dt G V^2 / (2 r*),  V = min(k_t, v_max)
    Small-angle form: lag = (k_t / r) T_lag with T_lag = alpha dt / (1 - alpha), so
        r* = (r_d + sqrt(r_d^2 + 4 C)) / 2,   C = V^2 (T_lag + dt / 2) / k_r
    giving radius error C / r_d for r_d >> sqrt(C) and a floor r_min = sqrt(C)
    as r_d -> 0 (the r = 0 singularity of Eq. 11 is never reached).
    Stiction: a robot at rest receives (1 - alpha) of its command, so the
    orbit never starts when (1 - alpha) k_t < v_s.

ORBIT, estimate with error d = p_hat - p* (nonlinear fields, magnitude-only)
    r* = r_d + e_dyn(r*) + <d_r> + k_t <d_t> / (k_r r*), averaged over orbit angle.

ATTRACTION (Eq. 10, exact estimate): per axis, with e = p_c - p*,
    e' = (1 - dt (1 - alpha) k) e + dt alpha v,   v' = alpha v - (1 - alpha) k e
    poles: z^2 - (1 + alpha - dt (1 - alpha) k) z + alpha = 0
    overshoot (complex or negative poles) for k > k_c = (1 - sqrt alpha) / (dt (1 + sqrt alpha))
    unstable for k > k_max = 2 (1 + alpha) / (dt (1 - alpha))
    Saturation turns divergence into a bounded oscillation. Motion is along a
    straight line, so reduced_1d is exact for linear fields.

NOISE (white sensor noise sigma_n on u and v, conformal J with scale lam)
    Fitting about the centroid makes the slope and centroid-value errors
    uncorrelated for any triangle (sum of offsets is zero), so
        Cov(p_hat) = (sigma_n^2 / 3) (1 + 2 |e|^2 / R^2) J^{-1} J^{-T},  R = side / sqrt 3
    and the closed loop is the per-axis system above driven by that noise
    (discrete Lyapunov equation).
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import vecsim as vs

PI = np.pi


def t_lag(alpha, dt):
    return alpha * dt / (1 - alpha)


def lag_filter(alpha, w):
    H = (1 - alpha) / (1 - alpha * np.exp(-1j * w))
    return np.abs(H), -np.angle(H)


def e_dyn(r, alpha, dt, kt, kr, vmax=0.3):
    V = np.minimum(kt, vmax)
    G, lag = lag_filter(alpha, V * dt / r)
    return (V * np.sin(lag) + dt * G * V ** 2 / (2 * r)) / (kr * np.cos(lag))


def orbit_C(alpha, dt, kt, kr, vmax=0.3):
    V = np.minimum(kt, vmax)
    return V ** 2 * (t_lag(alpha, dt) + dt / 2) / kr


def orbit_r_small_angle(rd, alpha, dt, kt, kr, vmax=0.3):
    C = orbit_C(alpha, dt, kt, kr, vmax)
    return (rd + np.sqrt(rd ** 2 + 4 * C)) / 2


def orbit_r_exact(rd, alpha, dt, kt, kr, vmax=0.3, extra=None):
    """Solve r = rd + e_dyn(r) (+ extra(r)) by bisection. Scalars."""
    def h(r):
        out = r - rd - e_dyn(r, alpha, dt, kt, kr, vmax)
        return out - (extra(r) if extra else 0.0)
    lo = max(rd, 1e-6)
    hi = rd + 2.0
    if not (h(hi) > 0):
        return np.nan
    if h(lo) > 0:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def orbit_starts(alpha, kt, vs_):
    return (1 - alpha) * kt >= vs_


# ---------------------------------------------------------------- attraction
def k_crit(alpha, dt):
    sa = np.sqrt(alpha)
    return (1 - sa) / (dt * (1 + sa))


def k_max(alpha, dt):
    return np.where(alpha < 1, 2 * (1 + alpha) / (dt * (1 - alpha)), np.inf)


def reduced_1d(d0, k, alpha, dt, vs_, vmax, vcmd_max, T, noise_std=0.0, seed=0, model="legacy"):
    """Attraction along one line, vectorized over start distances d0 > 0, with
    the given stiction model. Returns final |d|, overshoot (max distance past
    the target), moved flag. Exact for linear fields with an exact estimate."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d0, float).copy()
    v = np.zeros_like(d)
    over = np.zeros_like(d)
    moved = np.zeros(d.shape, bool)
    for _ in range(int(round(T / dt))):
        est = noise_std * rng.standard_normal(d.shape) if noise_std else 0.0
        err = est - d
        cmd = np.sign(err) * np.minimum(k * np.abs(err), vcmd_max)
        if model == "deadzone":
            cmd = np.where(np.abs(cmd) < vs_, 0.0, cmd)
        v = alpha * v + (1 - alpha) * cmd
        mag = np.abs(v)
        v = np.where(mag > vmax, np.sign(v) * vmax, v)
        if model == "legacy":
            v = np.where(mag < vs_, 0.0, v)
        elif model == "scaled":
            v = np.where(mag < (1 - alpha) * vs_, 0.0, v)
        moved |= v != 0
        d = d + dt * v
        over = np.maximum(over, -d)
    return np.abs(d), over, moved


def noise_rms(sigma_n, lam, k, alpha, dt):
    """Stationary 2-D rms position error at the target from white sensor noise
    (no stiction, no saturation), estimate variance sigma_n^2 / (3 lam^2) per axis."""
    s2 = sigma_n ** 2 / (3 * lam ** 2)
    a = 1 - alpha
    A = np.array([[1 - dt * a * k, dt * alpha], [-a * k, alpha]])
    B = np.array([[dt * a * k], [a * k]])
    P = solve_discrete_lyapunov(A, B @ B.T * s2)
    return float(np.sqrt(2 * P[0, 0]))


def estimate_rms(sigma_n, lam, e, side):
    R = side / np.sqrt(3)
    return sigma_n * np.sqrt((1 + 2 * e ** 2 / R ** 2) / 3) * np.sqrt(2) / lam


# ---------------------------------------------------------------- static estimator maps
THETAS = np.linspace(0, 2 * PI / 3, 6, endpoint=False)
PHIS = np.linspace(0, 2 * PI, 72, endpoint=False)


def static_estimates(F, method, side, centers, thetas=THETAS):
    """Estimate p_hat for formations centered at `centers` (m, 2) with each heading.
    Returns array (len(thetas), m, 2), NaN where singular."""
    out = []
    for th in thetas:
        m = len(centers)
        pos = vs.equilateral(centers, np.full(m, th), np.full(m, side))
        c = pos.mean(1)
        with np.errstate(all="ignore"):
            U, V = F.fn(pos[..., 0], pos[..., 1])
            U, V = np.asarray(U, float), np.asarray(V, float)
            if method == "fit":
                gux, guy, ub = vs._fit(pos, c, U)
                gvx, gvy, vb = vs._fit(pos, c, V)
                det = gux * gvy - guy * gvx
                ph = c - np.stack([(gvy * ub - guy * vb) / det, (-gvx * ub + gux * vb) / det], 1)
            elif method == "magnitude":
                M = np.hypot(U, V)
                gx, gy, mb = vs._fit(pos, c, M)
                ph = c - (mb / (gx * gx + gy * gy))[:, None] * np.stack([gx, gy], 1)
            else:
                ph = np.broadcast_to(F.target, (m, 2)).copy()
        out.append(ph)
    return np.array(out)


def orbit_bias_terms(F, method, side, r):
    """<d_r>, <d_t> over orbit angle and heading for an orbit of radius r about the target."""
    u = np.stack([np.cos(PHIS), np.sin(PHIS)], 1)
    ph = static_estimates(F, method, side, F.target + r * u)
    d = ph - F.target
    dr = (d * u).sum(-1)
    dt = (d * np.stack([-u[:, 1], u[:, 0]], 1)).sum(-1)
    if not np.isfinite(dr).all() or np.abs(d).max() > 2.0:
        return np.nan, np.nan
    return dr.mean(), dt.mean()


def orbit_r_nonlinear(F, method, side, rd, alpha, dt, kt, kr, vmax=0.3):
    """Mean orbit radius with estimator error, by scanning for the stable root of
    h(r) = r - rd - e_dyn(r) - <d_r> - kt <d_t> / (kr r)."""
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.2, 39), np.linspace(0.2, 1.0, 81)]))

    def h(r):
        a, b = orbit_bias_terms(F, method, side, r)
        if not np.isfinite(a):
            return np.nan
        return r - rd - e_dyn(r, alpha, dt, kt, kr, vmax) - a - kt * b / (kr * r)
    hs = np.array([h(r) for r in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(hs[i]) and np.isfinite(hs[i + 1]) and hs[i] < 0 <= hs[i + 1]:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if h(mid) < 0 else (lo, mid)
            return 0.5 * (lo + hi)
    return np.nan


def fixed_point_bias(F, method, side):
    """Distance from the target to the estimator's fixed point near it (p_hat(c) = c
    for the plane fit; zero fitted slope of |v| for magnitude), mean over headings."""
    out = []
    for th in THETAS:
        c = F.target + np.array([0.003, -0.002])
        for _ in range(300):
            if method == "fit":
                c = static_estimates(F, "fit", side, c[None], [th])[0, 0]
            else:
                pos = vs.equilateral(c[None], np.array([th]), np.array([side]))
                U, V = F.fn(pos[..., 0], pos[..., 1])
                gx, gy, _ = vs._fit(pos, pos.mean(1), np.hypot(np.asarray(U), np.asarray(V)))
                c = c - 0.02 * np.array([gx[0], gy[0]])
            if not np.isfinite(c).all():
                return np.nan
        out.append(np.hypot(*(c - F.target)))
    return float(np.mean(out))


# ---------------------------------------------------------------- version 2: stiction models, overshoot
"""
STICTION, dead-zone model (exact ZOH discretization of tau dv/dt = -v + DZ(u)):
    a robot moves only when its command is at least v_s, at any dt. Attraction
    from rest starts iff min(k d0, v_cmd_max) >= v_s; an orbit starts iff k_t >= v_s.
    Once k d < v_s the command dies at d = v_s / k and the robot coasts a further
    v T_eff, with T_eff = alpha dt / (1 - alpha) + dt / 2 (about tau). In the
    overdamped regime (k T_eff <= 1/4) the velocity at that point is lambda d with
    lambda = (1 - sqrt(1 - 4 k T_eff)) / (2 T_eff), so the cluster stops short by
        d_stop = (v_s / k) (1 + sqrt(1 - 4 k T_eff)) / 2       (v_s / 2k at critical damping)
OVERSHOOT (linear regime, caps off): per axis the loop is the second-order system
    T_eff e'' + e' + k e = 0, damping zeta = 1 / (2 sqrt(k T_eff)), so from rest
        overshoot / start distance = exp(-pi zeta / sqrt(1 - zeta^2))   for k T_eff > 1/4
    overshoot_ratio_discrete gives the exact sampled value. The orbit's radial
    loop has the same form with k -> k_r, measured from the offset |r0 - r*|.
"""


def t_eff(alpha, dt):
    return t_lag(alpha, dt) + dt / 2


def start_condition(model, alpha, cmd0, vs_):
    """Does a robot at rest move under its first command magnitude cmd0?"""
    if model == "legacy":
        return (1 - alpha) * cmd0 >= vs_
    return cmd0 >= vs_            # deadzone and scaled agree here


def undershoot_deadzone(k, alpha, dt, vs_):
    T = t_eff(alpha, dt)
    disc = np.clip(1 - 4 * k * T, 0, None)
    return (vs_ / k) * (1 + np.sqrt(disc)) / 2


def overshoot_ratio_continuous(k, alpha, dt):
    T = np.asarray(t_eff(alpha, dt), float)
    k = np.asarray(k, float)
    with np.errstate(all="ignore"):
        z = 1 / (2 * np.sqrt(k * T))
        out = np.where(z < 1, np.exp(-PI * z / np.sqrt(np.clip(1 - z * z, 1e-12, None))), 0.0)
    return np.where(T > 0, out, 0.0)


def overshoot_ratio_discrete(k, alpha, dt, steps=20000):
    """Exact sampled loop from rest at unit distance, no caps, no stiction:
    largest excursion past the target (inf if unstable)."""
    k, alpha, dt = np.broadcast_arrays(np.asarray(k, float), np.asarray(alpha, float), np.asarray(dt, float))
    e = np.ones(k.shape)
    v = np.zeros(k.shape)
    mx = np.zeros(k.shape)
    for _ in range(steps):
        v = alpha * v - (1 - alpha) * k * e
        e = e + dt * v
        mx = np.maximum(mx, -e)
        if np.all(np.abs(e) < 1e-12) or np.any(np.abs(e) > 1e6):
            break
    return np.where(np.abs(e) > 1e3, np.inf, mx)
