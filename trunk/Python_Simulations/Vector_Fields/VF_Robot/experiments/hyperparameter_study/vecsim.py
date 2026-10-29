"""
Vectorized 3-robot cluster simulator for the hyperparameter study.

Steps many independent trials at once, each with its own parameters, and
reproduces the reference code path:
    OmniCluster.move (with the 2026-10-28 heading fix), Omnibot lag / speed
    cap / stiction, critical_point_plane_fitting (paper Eqs. 4-10) and
    critical_point_orbiter_plane_fitting (Eq. 11).
Gains that the reference hard-codes (k = 1, 0.5 m/s command cap, k_t = 0.2,
k_r = 0.5, radial clamp 0.3) are per-trial parameters here. validate_vecsim.py
checks trajectories against the reference classes.

Differences from the reference, all numerical:
  - the inverse Jacobian is analytic instead of a 1e-6 forward difference;
  - the plane fit is solved about the centroid (mathematically identical to
    Eq. 5, better conditioned for tiny formations far from the origin).

Estimators ("method"):
  fit        paper plane fit, p_hat = c - J_hat^{-1} h_hat
  oracle     the true critical point (separates estimation from control)
  magnitude  |v| only: attraction is scalar_gradient_descent on |v|; orbit
             uses p_hat = c - m g / |g|^2 from a plane fit of |v|
"""
import os
import numpy as np

PI = np.pi
HERE = os.path.dirname(os.path.abspath(__file__))
VF_ROOT = os.path.dirname(os.path.dirname(HERE))

A_GYRE = 0.1
LINEAR_J = {
    "sink": [[-1.0, 0.0], [0.0, -1.0]],
    "source": [[1.0, 0.0], [0.0, 1.0]],
    "vortex": [[0.0, -1.0], [1.0, 0.0]],
    "sinking_vortex": [[-0.15, -0.4], [0.4, -0.15]],
    "spewing_vortex": [[0.15, 0.4], [-0.4, 0.15]],
    "saddle": [[0.0, -1.0], [-1.0, 0.0]],
}
LINEAR = list(LINEAR_J)
NONLINEAR = ["gyre", "curved_saddle"]
RECONSTRUCTED = ["vortex_nn", "saddle_nn"]
FIELDS = LINEAR + NONLINEAR + RECONSTRUCTED


# ============================================================================
# Fields
# ============================================================================
class Field:
    """fn(x, y) -> (u, v) on arrays. crit rows are (x, y, index), target first."""

    def __init__(self, name, kind, fn, crit, J0, start_half, max_rd, note=""):
        self.name, self.kind, self.fn = name, kind, fn
        self.crit = np.array(crit, dtype=float)
        self.target = self.crit[0, :2].copy()
        self.J0 = np.array(J0, dtype=float)       # Jacobian at the target
        self.start_half, self.max_rd, self.note = start_half, max_rd, note


def _linear(name):
    J = np.array(LINEAR_J[name])

    def fn(x, y):
        return J[0, 0] * x + J[0, 1] * y, J[1, 0] * x + J[1, 1] * y
    idx = np.sign(np.linalg.det(J))
    return Field(name, "linear", fn, [(0.0, 0.0, idx)], J, 0.5, 10.0)


def _gyre():
    A = A_GYRE

    def fn(x, y):
        return (PI * A * np.cos(PI * x) * np.sin(PI * y),
                -PI * A * np.sin(PI * x) * np.cos(PI * y))
    crit = [(0.0, 0.0, 1.0)]
    for m in range(-3, 4):
        for n in range(-3, 4):
            if (m, n) != (0, 0):
                crit.append((m, n, 1.0))
            crit.append((m + 0.5, n + 0.5, -1.0))
    s = PI ** 2 * A
    return Field("gyre", "nonlinear", fn, crit, [[0, s], [-s, 0]], 0.45, 1.0,
                 "det J = 0 on |x| + |y| = 0.5")


def _curved_saddle(eps=1.0, family="potential"):
    """Unit-strength saddle (u = x, v = -y at the origin) with v = -y + eps y^3
    on the inflow axis. potential: u - iv = z + eps z^3 (irrotational).
    cellular: psi = S(x) S(y), S'' = -6 eps S (vorticity 12 eps psi)."""
    crit = [(0.0, 0.0, -1.0)]
    if family == "potential":
        def fn(x, y):
            z = x + 1j * y
            f = z + eps * z ** 3
            return f.real, -f.imag
        if eps > 0:
            crit += [(0.0, 1 / np.sqrt(eps), -1.0), (0.0, -1 / np.sqrt(eps), -1.0)]
        elif eps < 0:
            crit += [(1 / np.sqrt(-eps), 0.0, -1.0), (-1 / np.sqrt(-eps), 0.0, -1.0)]
    else:
        k = np.sqrt(6 * abs(eps))
        if eps > 0:
            S, dS = (lambda x: np.sin(k * x) / k), (lambda x: np.cos(k * x))
            h = PI / k
            for m in range(-4, 5):
                for n in range(-4, 5):
                    if (m, n) != (0, 0) and max(abs(m), abs(n)) * h < 3:
                        crit.append((m * h, n * h, -1.0))
                    if max(abs(m + 0.5), abs(n + 0.5)) * h < 3:
                        crit.append(((m + 0.5) * h, (n + 0.5) * h, 1.0))
        elif eps < 0:
            S, dS = (lambda x: np.sinh(k * x) / k), (lambda x: np.cosh(k * x))
        else:
            S, dS = (lambda x: x), (lambda x: np.ones_like(x))

        def fn(x, y):
            with np.errstate(over="ignore", invalid="ignore"):
                return S(x) * dS(y), -dS(x) * S(y)
    return Field(f"curved_saddle_{family}_{eps:+g}", "nonlinear", fn, crit,
                 [[1, 0], [0, -1]], 0.5, 1.0)


class _NN:
    """Batched replica of field_types.NNField.get_value."""

    def __init__(self, predictor_dir):
        import torch
        from src.fields.field_types import NNField
        self.torch = torch
        ref = NNField(predictor_dir)
        self.hue, self.sat = ref.hue_model, ref.sat_model
        torch.set_num_threads(1)

    def __call__(self, x, y):
        torch = self.torch
        shp = np.shape(x)
        xy = torch.from_numpy(np.stack([np.ravel(x), np.ravel(y)], 1).astype(np.float32))
        with torch.no_grad():
            h = self.hue(xy).numpy().astype(np.float64)
            s = self.sat(xy).numpy()[:, 0].astype(np.float64)
        ang = (np.arctan2(h[:, 0], h[:, 1]) + 2 * PI) % (2 * PI)
        mag = ((s + 1) / 2 - 0.3) / 0.7
        ang = ang + PI / 2
        return (mag * np.cos(ang)).reshape(shp), (mag * np.sin(ang)).reshape(shp)


def _reconstructed(name):
    import sys
    if VF_ROOT not in sys.path:
        sys.path.insert(0, VF_ROOT)
    fn = _NN("vortex_predictors" if name == "vortex_nn" else "saddle_predictors")
    J = LINEAR_J["vortex" if name == "vortex_nn" else "saddle"]
    idx = 1.0 if name == "vortex_nn" else -1.0
    return Field(name, "reconstructed", fn, [(0.0, 0.0, idx)], J, 0.5, 0.5,
                 "MLP fit to testbed color-sensor data; truth = printed map's critical point")


def make_field(name, eps=1.0, family="potential"):
    if name in LINEAR_J:
        return _linear(name)
    if name == "gyre":
        return _gyre()
    if name == "curved_saddle":
        return _curved_saddle(eps, family)
    if name in RECONSTRUCTED:
        return _reconstructed(name)
    raise ValueError(name)


# ============================================================================
# Simulation
# ============================================================================
DEFAULTS = dict(side=0.33, alpha=float(np.exp(-0.1 / 0.28)), vs=0.025, vmax=0.4, k=1.0, vcmd_max=0.5,
                kt=0.2, kr=0.5, rclamp=0.3, rd=0.3, sigma_n=0.0, kf=0.5, kb=0.01, r0=np.nan)
STICTION_MODELS = ("legacy", "deadzone", "scaled", "coulomb")
PARAM_KEYS = list(DEFAULTS)


def params(n, **over):
    """Per-trial parameter arrays: defaults (hardware baseline), overridden by
    scalars or arrays of length n."""
    P = {}
    for k, v in DEFAULTS.items():
        P[k] = np.broadcast_to(np.asarray(over.get(k, v), dtype=float), (n,)).copy()
    return P


def equilateral(c, theta, side):
    """Robot positions as kinematics.inverse_kinematics(c, theta, side, 60 deg, side)."""
    b = PI / 3
    L = np.stack([np.stack([side, np.zeros_like(side)], -1),
                  np.zeros((len(side), 2)),
                  np.stack([side * np.cos(b), side * np.sin(b)], -1)], 1)
    L = L - L.mean(1, keepdims=True)
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    X = ct * L[..., 0] - st * L[..., 1] + c[:, None, 0]
    Y = st * L[..., 0] + ct * L[..., 1] + c[:, None, 1]
    return np.stack([X, Y], -1)


def _fit(pos, c, f):
    """Plane fit about the centroid: returns (gx, gy, value at centroid)."""
    d = pos - c[:, None, :]
    Sxx = (d[..., 0] ** 2).sum(1)
    Syy = (d[..., 1] ** 2).sum(1)
    Sxy = (d[..., 0] * d[..., 1]).sum(1)
    det = Sxx * Syy - Sxy ** 2
    bx = (d[..., 0] * f).sum(1)
    by = (d[..., 1] * f).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = (Syy * bx - Sxy * by) / det
        gy = (Sxx * by - Sxy * bx) / det
    return gx, gy, f.mean(1)


def simulate(field, task, method, P, T, dt=0.1, seed=0, starts=None, thetas=None,
             window=20.0, escape=None, keep=0, keep_stride=5, stiction_model="coulomb",
             vk=None):
    """Run len(P['side']) trials for T seconds. task 'attract' or 'orbit'.
    starts: (n, 2) centroid starts; default uniform box (attract) or on the
    circle of radius r0 (default r_d) for orbits.
    stiction_model: as Omnibot, default "coulomb" to match OmniCluster
    (run_study passes each cell's rule explicitly). "legacy" zeroes the lagged velocity below v_s;
    "deadzone" zeroes commands below v_s before the lag (exact ZOH
    discretization, sample-period invariant); "scaled" is legacy with
    threshold (1 - alpha) v_s; "coulomb" is static friction v_s plus kinetic
    friction vk (scalar or per-trial array, default vk = v_s) behind the lag.
    Returns (metrics dict of arrays, sample trajectories)."""
    assert stiction_model in STICTION_MODELS, stiction_model
    rng = np.random.default_rng(seed)
    n = len(P["side"])
    tgt = field.target
    if thetas is None:
        thetas = rng.uniform(0, 2 * PI, n)
    if starts is None:
        if task == "attract":
            starts = tgt + rng.uniform(-field.start_half, field.start_half, (n, 2))
        else:
            ph = rng.uniform(0, 2 * PI, n)
            r0 = np.where(np.isfinite(P["r0"]), P["r0"], P["rd"])
            starts = tgt + r0[:, None] * np.stack([np.cos(ph), np.sin(ph)], 1)
    if escape is None:
        escape = 3.0 if field.kind != "linear" else max(50.0, 5 * float(P["rd"].max()))
    pos = equilateral(np.asarray(starts, float), np.asarray(thetas, float), P["side"])
    vel = np.zeros_like(pos)
    side, al, vs, vmax = P["side"], P["alpha"], P["vs"], P["vmax"]
    vk = vs if vk is None else np.broadcast_to(np.asarray(vk, float), (n,))
    if stiction_model == "coulomb":
        assert np.all((vk >= 0) & (vk <= vs)), "coulomb needs 0 <= vk <= vs"
    stuck = np.ones((n, 3), bool)
    al3 = al[:, None, None]
    n_steps = int(round(T / dt))
    w0 = int(round(window / dt))
    c0 = pos.mean(1)
    u0 = tgt - c0
    u0 = u0 / np.maximum(np.linalg.norm(u0, axis=1, keepdims=True), 1e-12)
    crit = field.crit

    M = dict(escaped=np.zeros(n, bool), fallback=np.zeros(n), max_side_err=np.zeros(n),
             path=np.zeros(n), sat=np.zeros(n), moved=np.zeros(n, bool))
    if task == "attract":
        M.update(t5=np.full(n, np.nan), t2=np.full(n, np.nan), t1=np.full(n, np.nan),
                 last_out2=np.zeros(n), overshoot=np.zeros(n), det0=np.full(n, np.nan),
                 est0_err=np.full(n, np.nan), min_dist=np.full(n, np.inf))
    else:
        H = 7
        XtX = np.zeros((n, H, H))
        Xtr = np.zeros((n, H))
        M.update(r_tmin=np.full(n, np.inf), r_tmax=np.zeros(n))
        M.update(cnt=np.zeros(n), sr=np.zeros(n), sr2=np.zeros(n), rmin=np.full(n, np.inf),
                 rmax=np.zeros(n), sc=np.zeros((n, 2)), ang=np.zeros(n), srho=np.zeros(n),
                 sest=np.zeros(n), nest=np.zeros(n))
        phi_prev = None
    traj = np.full((keep, n_steps // keep_stride + 1, 2), np.nan) if keep else None
    active = np.ones(n, bool)

    for kstep in range(n_steps):
        c = pos.mean(1)
        d12 = pos[:, 1] - pos[:, 0]
        d23 = pos[:, 2] - pos[:, 1]
        d31 = pos[:, 0] - pos[:, 2]
        p = np.hypot(d12[:, 0], d12[:, 1])
        q = np.hypot(d23[:, 0], d23[:, 1])
        r3 = np.hypot(d31[:, 0], d31[:, 1])
        beta = np.arccos(np.clip((p * p + q * q - r3 * r3) / (2 * p * q + 1e-10), -1, 1))
        th = np.arctan2(d12[:, 1], d12[:, 0]) + PI
        with np.errstate(all="ignore"):
            U, V = field.fn(pos[..., 0], pos[..., 1])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        if np.any(P["sigma_n"] > 0):
            U = U + P["sigma_n"][:, None] * rng.standard_normal(U.shape)
            V = V + P["sigma_n"][:, None] * rng.standard_normal(V.shape)

        # ---- estimate
        fb = np.zeros(n, bool)
        if method == "oracle":
            phat = np.broadcast_to(tgt, (n, 2)).copy()
            detJ = np.full(n, np.nan)
        elif method == "fit":
            gux, guy, ub = _fit(pos, c, U)
            gvx, gvy, vb_ = _fit(pos, c, V)
            detJ = gux * gvy - guy * gvx
            with np.errstate(all="ignore"):
                sx = (gvy * ub - guy * vb_) / detJ
                sy = (-gvx * ub + gux * vb_) / detJ
            phat = c - np.stack([sx, sy], 1)
            fb = ~np.isfinite(phat).all(1)
        else:
            m = np.hypot(U, V)
            gx, gy, mb = _fit(pos, c, m)
            gg = gx * gx + gy * gy
            with np.errstate(all="ignore"):
                phat = c - (mb / gg)[:, None] * np.stack([gx, gy], 1)
            detJ = np.full(n, np.nan)
            fb = ~np.isfinite(phat).all(1) | (gg < 1e-18)

        # ---- navigation command
        w = phat - c
        dist = np.hypot(w[:, 0], w[:, 1])
        with np.errstate(all="ignore"):
            wh = w / dist[:, None]
        if task == "attract":
            if method == "magnitude":
                vc = -np.stack([gx, gy], 1)
                vc[~np.isfinite(vc).all(1)] = 0.0
                fb[:] = False
            else:
                spd = np.minimum(P["k"] * dist, P["vcmd_max"])
                vc = wh * spd[:, None]
                vc[dist < 1e-6] = 0.0
        else:
            tang = np.stack([wh[:, 1], -wh[:, 0]], 1)
            vr = np.clip(P["kr"] * (dist - P["rd"]), -P["rclamp"], P["rclamp"])
            vc = P["kt"][:, None] * tang + vr[:, None] * wh
            small = dist < 1e-6
            if small.any():
                a = rng.uniform(0, 2 * PI, small.sum())
                vc[small] = np.stack([np.cos(a), np.sin(a)], 1)
        if fb.any():   # reference falls back to vector_sum on a singular solve
            if method == "magnitude":
                vc[fb] = 0.0
            else:
                vc[fb] = np.stack([U[fb].sum(1), V[fb].sum(1)], 1)
        M["fallback"] += fb

        if task == "attract" and kstep == 0:
            M["det0"] = detJ
            M["est0_err"] = np.hypot(*(phat - tgt).T)

        # ---- formation control (cluster space) and robot commands
        ep = side - p
        eq = side - q
        eb = (PI / 3 - beta + PI) % (2 * PI) - PI
        vp, vq, vbt = P["kf"] * ep, P["kf"] * eq, P["kb"] * eb
        cb_, sb_ = np.cos(beta), np.sin(beta)
        lx = np.stack([2 / 3 * vp - 1 / 3 * vq * cb_ + 1 / 3 * vbt * q * sb_,
                       -1 / 3 * vp - 1 / 3 * vq * cb_ + 1 / 3 * vbt * q * sb_,
                       -1 / 3 * vp + 2 / 3 * vq * cb_ - 2 / 3 * vbt * q * sb_], 1)
        ly = np.stack([-1 / 3 * vq * sb_ - 1 / 3 * vbt * q * cb_,
                       -1 / 3 * vq * sb_ - 1 / 3 * vbt * q * cb_,
                       2 / 3 * vq * sb_ + 2 / 3 * vbt * q * cb_], 1)
        ct, st = np.cos(th)[:, None], np.sin(th)[:, None]
        cmd = np.stack([ct * lx - st * ly + vc[:, None, 0], st * lx + ct * ly + vc[:, None, 1]], -1)

        # ---- Omnibot: lag, speed cap, stiction (pre-cap magnitude)
        if stiction_model == "deadzone":
            cmd[np.hypot(cmd[..., 0], cmd[..., 1]) < vs[:, None]] = 0.0
        new = al3 * vel + (1 - al3) * cmd
        if stiction_model == "coulomb":
            um = np.hypot(cmd[..., 0], cmd[..., 1])
            stuck &= ~(vel != 0).any(-1)
            stuck &= ~(um > vs[:, None])
            fm = np.hypot(new[..., 0], new[..., 1])
            drop = ((1 - al) * vk)[:, None]
            through = fm <= drop
            with np.errstate(all="ignore"):
                new = new * np.where(through | stuck, 0.0, 1 - drop / fm)[..., None]
            stuck |= through & (um <= vs[:, None])
        mag = np.hypot(new[..., 0], new[..., 1])
        over = mag > vmax[:, None]
        with np.errstate(all="ignore"):
            scale = np.where(over, vmax[:, None] / mag, 1.0)
        new = new * scale[..., None]
        if stiction_model == "legacy":
            new[mag < vs[:, None]] = 0.0
        elif stiction_model == "scaled":
            new[mag < ((1 - al) * vs)[:, None]] = 0.0
        newpos = pos + dt * new
        bad = ~np.isfinite(newpos).all((1, 2)) | (np.abs(newpos - tgt).max((1, 2)) > escape)
        M["escaped"] |= bad & active
        active &= ~bad
        a3 = active[:, None, None]
        pos = np.where(a3, newpos, pos)
        vel = np.where(a3, new, 0.0)
        stuck |= ~active[:, None]

        # ---- metrics
        sp = np.hypot(vel[..., 0], vel[..., 1])
        M["path"] += dt * sp.mean(1)
        M["sat"] += (sp >= vmax[:, None] * (1 - 1e-9)).mean(1)
        M["moved"] |= sp.max(1) > 0
        s1 = np.hypot(*(pos[:, 1] - pos[:, 0]).T)
        s2 = np.hypot(*(pos[:, 2] - pos[:, 1]).T)
        s3 = np.hypot(*(pos[:, 0] - pos[:, 2]).T)
        se = np.max(np.abs(np.stack([s1, s2, s3], 1) / side[:, None] - 1), 1)
        M["max_side_err"] = np.maximum(M["max_side_err"], np.where(active, se, 0))
        cn = pos.mean(1)
        e = cn - tgt
        rr = np.hypot(e[:, 0], e[:, 1])
        t = (kstep + 1) * dt
        if task == "attract":
            for key, thr in (("t5", 0.05), ("t2", 0.02), ("t1", 0.01)):
                M[key] = np.where(np.isnan(M[key]) & (rr < thr), t, M[key])
            M["last_out2"] = np.where(rr >= 0.02, t, M["last_out2"])
            M["overshoot"] = np.maximum(M["overshoot"], (e * u0).sum(1))
            M["min_dist"] = np.minimum(M["min_dist"], rr)
        else:
            phi = np.arctan2(e[:, 1], e[:, 0])
            if kstep < w0:
                M["r_tmin"] = np.where(active, np.minimum(M["r_tmin"], rr), M["r_tmin"])
                M["r_tmax"] = np.where(active, np.maximum(M["r_tmax"], rr), M["r_tmax"])
            if kstep >= w0:
                ok = active
                X = np.stack([np.ones(n), np.cos(phi), np.sin(phi), np.cos(2 * phi),
                              np.sin(2 * phi), np.cos(4 * phi), np.sin(4 * phi)], 1)
                X[~ok] = 0.0
                XtX += X[:, :, None] * X[:, None, :]
                Xtr += X * np.where(ok, rr, 0.0)[:, None]
                M["cnt"] += ok
                M["sr"] += np.where(ok, rr, 0)
                M["sr2"] += np.where(ok, rr * rr, 0)
                M["rmin"] = np.where(ok, np.minimum(M["rmin"], rr), M["rmin"])
                M["rmax"] = np.where(ok, np.maximum(M["rmax"], rr), M["rmax"])
                M["sc"] += np.where(ok[:, None], e, 0)
                dphi = (phi - phi_prev + PI) % (2 * PI) - PI
                M["ang"] += np.where(ok, dphi, 0)
                fin = ok & np.isfinite(phat).all(1)
                M["srho"] += np.where(fin, dist, 0)
                M["sest"] += np.where(fin, np.hypot(*(phat - tgt).T), 0)
                M["nest"] += fin
            phi_prev = phi
        if keep and kstep % keep_stride == 0:
            traj[:, kstep // keep_stride] = cn[:keep]

    c = pos.mean(1)
    out = {k: v for k, v in M.items() if k not in ("sc",)}
    out["final_x"], out["final_y"] = c[:, 0], c[:, 1]
    out["path"] = M["path"]
    out["sat"] = M["sat"] / n_steps
    out["fallback"] = M["fallback"] / n_steps
    if task == "attract":
        d = np.hypot(*(c[:, None, :] - crit[None, :, :2]).transpose(2, 0, 1))
        i = d.argmin(1)
        out["final_err"] = np.hypot(*(c - tgt).T)
        out["final_proj"] = ((c - tgt) * u0).sum(1)     # > 0 past the target, < 0 short of it
        out["nearest"] = i
        out["nearest_dist"] = d[np.arange(n), i]
        out["nearest_index"] = crit[i, 2]
        out["start_dist"] = np.hypot(*(c0 - tgt).T)
    else:
        cnt = np.maximum(M["cnt"], 1)
        out["mean_r"] = M["sr"] / cnt
        out["std_r"] = np.sqrt(np.maximum(M["sr2"] / cnt - out["mean_r"] ** 2, 0))
        out["ctr_x"], out["ctr_y"] = M["sc"][:, 0] / cnt, M["sc"][:, 1] / cnt
        out["ctr_off"] = np.hypot(out["ctr_x"], out["ctr_y"])
        out["revs"] = np.abs(M["ang"]) / (2 * PI)
        out["rho_mean"] = M["srho"] / np.maximum(M["nest"], 1)
        out["est_err"] = M["sest"] / np.maximum(M["nest"], 1)
        with np.errstate(all="ignore"):
            coef = np.full((n, 7), np.nan)
            okc = M["cnt"] > 20
            if okc.any():
                reg = XtX[okc] + 1e-9 * np.eye(7)
                coef[okc] = np.linalg.solve(reg, Xtr[okc][..., None])[..., 0]
        out["a1"] = np.hypot(coef[:, 1], coef[:, 2])
        out["a2"] = np.hypot(coef[:, 3], coef[:, 4])
        out["c2"] = coef[:, 3]
        out["a4"] = np.hypot(coef[:, 5], coef[:, 6])
        for k in ("cnt", "sr", "sr2", "ang", "srho", "sest", "nest"):
            out.pop(k, None)
    return out, traj
