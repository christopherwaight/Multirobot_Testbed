"""
Shared pieces for the 3-robot field tests:
    plane_fit_field_test.py        vector plane fit (critical_point_plane_fitting / _orbiter_)
    vector_to_scalar_field_test.py magnitude only (|v| sensed, scalar plane fit)
    mc_field_tests.py              small Monte Carlo spot checks of both

FIELDS (all scaled by SIGMA = pi^2 A, the gyre's center rate, so every field
has the same local Jacobian magnitude at its target critical point (0, 0)):

  gyre              one Shadden double-gyre cell centered at the origin.
                    u = pi A cos(pi x) sin(pi y),  v = -pi A sin(pi x) cos(pi y).
                    Center at (0,0), saddles at (+-0.5, +-0.5). det J = 0 on
                    the diamond |x| + |y| = 0.5.
  linear_center     u = SIGMA y,  v = -SIGMA x            (clockwise, like the gyre)
  linear_saddle     u = SIGMA x,  v = -SIGMA y
  shear_saddle      u = SIGMA (x + SHEAR y),  v = -SIGMA y.
                    Linear and incompressible with uniform vorticity, so a
                    steady Euler flow. J^T J is anisotropic, which the
                    magnitude-only estimator cannot see past.
  potential_saddle  irrotational: complex velocity u - i v = SIGMA (z + eps z^3).
                    On the y axis v = -SIGMA (y - eps y^3); on the x axis
                    u = SIGMA (x + eps x^3). A steady inviscid flow with no
                    forcing (Bernoulli). Extra saddles at z^2 = -1/eps,
                    det J = -|f'|^2 <= 0 vanishes only at the points
                    1 + 3 eps z^2 = 0. eps -> -eps is a 90 degree rotation
                    plus flow reversal, which the estimators cannot tell apart.
  cellular_saddle   rotational: psi = S(x) S(y) with S'' = -6 eps S,
                    u = SIGMA S(x) S'(y),  v = -SIGMA S'(x) S(y).
                    Vorticity is 12 eps psi, a function of psi, so it is also
                    a steady Euler flow with no forcing. On the y axis
                    v = -SIGMA (y - eps y^3 + ...). eps > 0 is a gyre corner
                    (sin), with det J = 0 on |x| + |y| = pi / (2k),
                    k = sqrt(6 eps). eps < 0 is sinh, no other critical points.

No numbers from these scripts go into any .tex without review.
"""
import io
import os
import sys
import contextlib
from dataclasses import dataclass, field as dc_field

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField, AnalyticalScalarField, load_field_config
from src.control import primitives as ocp
from src.control.kinematics import inverse_kinematics

FORMATION_CONFIG = "config/formations/equilateral_default.yaml"
A_GYRE = load_field_config("double_gyre").get("A", 0.1)
SIGMA = np.pi ** 2 * A_GYRE
SHEAR = 1.0
DT = 0.1

# Orbit law constants, identical to critical_point_orbiter_plane_fitting
K_T, K_R, MAX_RADIAL = 0.2, 0.5, 0.3

FIELD_NAMES = ["gyre", "linear_center", "linear_saddle", "shear_saddle",
               "potential_saddle", "cellular_saddle"]


# ============================================================================
# Fields
# ============================================================================
@dataclass
class FieldSpec:
    name: str
    eps: float
    fn: callable                       # (x, y) -> (u, v)
    crit: list                         # [(label, x, y)], target first at (0, 0)
    basin: callable                    # (x, y) -> True if predicted to reach (0, 0)
    start_half: float                  # half width of the attraction start box
    note: str = ""
    lim: float = 0.65                  # plot half width

    @property
    def tag(self):
        return self.name if self.name not in ("potential_saddle", "cellular_saddle") \
            else f"{self.name}_eps{self.eps:+g}"


def make_field(name, eps=0.0):
    s = SIGMA
    if name == "gyre":
        A = A_GYRE

        def fn(x, y):
            return (float(np.pi * A * np.cos(np.pi * x) * np.sin(np.pi * y)),
                    float(-np.pi * A * np.sin(np.pi * x) * np.cos(np.pi * y)))
        crit = [("center", 0.0, 0.0)] + [(f"saddle({sx:+.1f},{sy:+.1f})", sx, sy)
                                         for sx in (0.5, -0.5) for sy in (0.5, -0.5)] \
            + [(f"center({cx:+.0f},{cy:+.0f})", cx, cy)
               for cx, cy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return FieldSpec(name, 0.0, fn, crit, lambda x, y: abs(x) + abs(y) < 0.5, 0.45,
                         "det J = 0 on |x|+|y| = 0.5")
    if name == "linear_center":
        return FieldSpec(name, 0.0, lambda x, y: (float(s * y), float(-s * x)),
                         [("center", 0.0, 0.0)], lambda x, y: True, 0.5)
    if name == "linear_saddle":
        return FieldSpec(name, 0.0, lambda x, y: (float(s * x), float(-s * y)),
                         [("saddle", 0.0, 0.0)], lambda x, y: True, 0.5)
    if name == "shear_saddle":
        return FieldSpec(name, 0.0, lambda x, y: (float(s * (x + SHEAR * y)), float(-s * y)),
                         [("saddle", 0.0, 0.0)], lambda x, y: True, 0.5,
                         f"shear {SHEAR}")
    if name == "potential_saddle":
        def fn(x, y):
            z = complex(x, y)
            f = s * (z + eps * z ** 3)
            return float(f.real), float(-f.imag)
        crit = [("saddle", 0.0, 0.0)]
        if eps > 0:
            crit += [("saddle(0,+)", 0.0, 1 / np.sqrt(eps)), ("saddle(0,-)", 0.0, -1 / np.sqrt(eps))]
            basin = lambda x, y: 3 * eps * y * y - eps * x * x < 1
            note = f"f'=0 at (0, +-{1 / np.sqrt(3 * eps):.3f})"
        elif eps < 0:
            e = -eps
            crit += [("saddle(+,0)", 1 / np.sqrt(e), 0.0), ("saddle(-,0)", -1 / np.sqrt(e), 0.0)]
            basin = lambda x, y: 3 * e * x * x - e * y * y < 1
            note = f"f'=0 at (+-{1 / np.sqrt(3 * e):.3f}, 0)"
        else:
            basin, note = (lambda x, y: True), "linear"
        return FieldSpec(name, eps, fn, crit, basin, 0.6, note)
    if name == "cellular_saddle":
        k = np.sqrt(6 * abs(eps))
        if eps > 0:
            S, dS = (lambda x: np.sin(k * x) / k), (lambda x: np.cos(k * x))
        elif eps < 0:
            S, dS = (lambda x: np.sinh(k * x) / k), (lambda x: np.cosh(k * x))
        else:
            S, dS = (lambda x: x), (lambda x: 1.0)

        def fn(x, y):
            with np.errstate(over="ignore", invalid="ignore"):
                return float(s * S(x) * dS(y)), float(-s * dS(x) * S(y))
        crit = [("saddle", 0.0, 0.0)]
        if eps > 0:
            h = np.pi / k
            for m in range(-3, 4):
                for n in range(-3, 4):
                    if (m, n) != (0, 0) and abs(m * h) < 1.3 and abs(n * h) < 1.3:
                        crit.append((f"saddle({m},{n})", m * h, n * h))
                    cx, cy = (m + 0.5) * h, (n + 0.5) * h
                    if abs(cx) < 1.3 and abs(cy) < 1.3:
                        crit.append((f"center({m},{n})", cx, cy))
            basin = lambda x, y: abs(x) + abs(y) < np.pi / (2 * k)
            note = f"det J = 0 on |x|+|y| = {np.pi / (2 * k):.3f}"
        else:
            basin, note = (lambda x, y: True), "no other critical points"
        return FieldSpec(name, eps, fn, crit, basin, 0.6, note)
    raise ValueError(f"unknown field {name!r}; choose from {FIELD_NAMES}")


def magnitude_fn(fn):
    return lambda x, y: float(np.hypot(*fn(x, y)))


# ============================================================================
# Cluster
# ============================================================================
def make_cluster(spec, start, side, alpha, stiction, theta=np.pi, magnitude=False):
    """OmniCluster with an equilateral formation of the given side, placed at
    `start` with heading `theta`. magnitude=True senses |v| only."""
    if magnitude:
        field = AnalyticalScalarField(magnitude_fn(spec.fn))
    else:
        field = AnalyticalField(spec.fn)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = OmniCluster(FORMATION_CONFIG, field, momentum_alpha=alpha,
                         stiction_threshold=stiction)
    cl.desired_p = cl.desired_q = side
    pos = inverse_kinematics(start[0], start[1], theta, side, cl.desired_beta, side)
    for i, robot in enumerate(cl.robots):
        robot.set_position(pos[2 * i], pos[2 * i + 1])
        robot.velocity = np.zeros(2)
    cl.center_history, cl.robot_history, cl.velocity_history, cl.diagnostics = [], [], [], []
    return cl


def robot_positions(cl):
    return np.array(cl.get_robot_positions(), dtype=float).reshape(3, 2)


# ============================================================================
# Estimators
# ============================================================================
def _plane(pos, vals):
    return np.linalg.solve(np.column_stack([pos, np.ones(3)]), vals)


def vector_estimate(pos, uv):
    """Plane fit of u and v, critical point p* = -J^{-1} h (paper Eqs. 4-9)."""
    a, b, c = _plane(pos, uv[:, 0])
    d, e, f = _plane(pos, uv[:, 1])
    return np.linalg.solve(np.array([[a, b], [d, e]]), np.array([-c, -f]))


def magnitude_estimate(pos, m):
    """Plane fit of |v|, then the point on the plane's zero line nearest the
    centroid, p = c - m_c g / |g|^2."""
    a, b, c0 = _plane(pos, m)
    g = np.array([a, b])
    cen = pos.mean(axis=0)
    gg = g @ g
    if gg < 1e-18:
        raise np.linalg.LinAlgError("flat magnitude plane")
    return cen - (g @ cen + c0) * g / gg


def estimate(spec, pos, method):
    if method == "vector":
        return vector_estimate(pos, np.array([spec.fn(*p) for p in pos]))
    return magnitude_estimate(pos, np.array([np.hypot(*spec.fn(*p)) for p in pos]))


# ============================================================================
# Primitives
# ============================================================================
def orbit_law(est, centroid, r_d):
    """Orbit law of critical_point_orbiter_plane_fitting (paper Eq. 11),
    same constants, applied to any center estimate."""
    w = est - centroid
    rho = np.linalg.norm(w)
    if rho < 1e-6:
        return 0.0, 0.0
    wh = w / rho
    v = K_T * np.array([wh[1], -wh[0]]) + np.clip(K_R * (rho - r_d), -MAX_RADIAL, MAX_RADIAL) * wh
    return float(v[0]), float(v[1])


def magnitude_orbiter(cluster, r_d):
    pos = robot_positions(cluster)
    try:
        est = magnitude_estimate(pos, np.array(cluster.sample_field_at_robots(), dtype=float))
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return orbit_law(est, pos.mean(axis=0), r_d)


def attract_primitive(method):
    return ocp.critical_point_plane_fitting if method == "vector" else ocp.scalar_gradient_descent


def orbit_primitive(method, r_d):
    if method == "vector":
        return lambda c: ocp.critical_point_orbiter_plane_fitting(c, desired_radius=r_d)
    return lambda c: magnitude_orbiter(c, r_d)


# ============================================================================
# Running and metrics
# ============================================================================
@dataclass
class Run:
    center: np.ndarray
    robots: np.ndarray
    est: np.ndarray
    escaped: bool = False
    extra: dict = dc_field(default_factory=dict)


def run(cluster, primitive, steps, spec, method, escape=3.0):
    """Step the cluster, logging the centroid and the method's center estimate.
    Stops early if any robot passes |x| or |y| = escape."""
    est = np.full((steps, 2), np.nan)
    escaped = False
    for k in range(steps):
        pos = robot_positions(cluster)
        if np.abs(pos).max() > escape or not np.isfinite(pos).all():
            escaped = True
            break
        try:
            with np.errstate(all="ignore"):
                est[k] = estimate(spec, pos, method)
        except np.linalg.LinAlgError:
            pass
        with np.errstate(all="ignore"):
            cluster.move(primitive)
    n = len(cluster.center_history)
    return Run(cluster.get_center_history(), cluster.get_robot_history(), est[:n], escaped)


def side_error(robots, side):
    if len(robots) == 0:
        return np.array([np.inf])
    s = np.stack([np.linalg.norm(robots[:, i] - robots[:, (i + 1) % 3], axis=1)
                  for i in range(3)], axis=1)
    return np.abs(s / side - 1).max(axis=1)


def nearest_crit(spec, p):
    d = [np.hypot(p[0] - x, p[1] - y) for _, x, y in spec.crit]
    i = int(np.argmin(d))
    return spec.crit[i][0], d[i], i


def attract_metrics(r, spec, side, tol=0.05):
    out = dict(escaped=r.escaped, max_side_err=float(side_error(r.robots, side).max()))
    if r.escaped or len(r.center) == 0:
        out.update(reached="escaped", reached_origin=False, final_err=np.nan, t_2cm=np.nan)
        return out
    c = r.center[-1]
    label, dist, i = nearest_crit(spec, c)
    reached = label if dist < tol else "none"
    d0 = np.linalg.norm(r.center - np.array(spec.crit[i][1:]), axis=1)
    hit = np.where(d0 < 0.02)[0]
    out.update(reached=reached, reached_origin=(i == 0 and dist < tol), final_err=float(dist),
               final_x=float(c[0]), final_y=float(c[1]),
               t_2cm=float(hit[0] * DT) if len(hit) else np.nan)
    return out


def orbit_metrics(r, side, w0):
    out = dict(escaped=r.escaped, max_side_err=float(side_error(r.robots, side).max()))
    C = r.center[w0:]
    if r.escaped or len(C) < 20:
        out.update(failed=True)
        return out
    rad = np.linalg.norm(C, axis=1)
    phi = np.arctan2(C[:, 1], C[:, 0])
    X = np.column_stack([np.ones_like(phi), np.cos(2 * phi), np.sin(2 * phi),
                         np.cos(4 * phi), np.sin(4 * phi)])
    coef = np.linalg.lstsq(X, rad, rcond=None)[0]
    est_err = np.linalg.norm(r.est[w0:], axis=1)
    swept = abs(np.unwrap(phi)[-1] - np.unwrap(phi)[0])
    out.update(failed=bool(out["max_side_err"] > 0.1 or rad.max() > 1.5 or swept < 2 * np.pi),
               revolutions=float(swept / (2 * np.pi)),
               mean_r=float(rad.mean()), std_r=float(rad.std()),
               ctr_off=float(np.linalg.norm(C.mean(axis=0))),
               c2=float(coef[1]), a2=float(np.hypot(coef[1], coef[2])),
               a4=float(np.hypot(coef[3], coef[4])),
               est_err=float(np.nanmedian(est_err)) if np.isfinite(est_err).any() else np.nan)
    return out


# ============================================================================
# Plotting and the single-configuration suite used by the two scripts
# ============================================================================
ATTRACT_STARTS = [(0.30, 0.20), (-0.35, 0.25), (0.10, -0.40),
                  (-0.25, -0.30), (0.38, -0.05), (0.40, 0.40)]
ORBIT_RADII = [0.10, 0.20, 0.30, 0.40]
ORBIT_START_X = 0.25


def det_j_grid(spec, X, Y, h=1e-5):
    D = np.empty_like(X)
    for idx in np.ndindex(X.shape):
        x, y = X[idx], Y[idx]
        ux1, vx1 = spec.fn(x + h, y)
        ux0, vx0 = spec.fn(x - h, y)
        uy1, vy1 = spec.fn(x, y + h)
        uy0, vy0 = spec.fn(x, y - h)
        D[idx] = ((ux1 - ux0) * (vy1 - vy0) - (uy1 - uy0) * (vx1 - vx0)) / (4 * h * h)
    return D


def draw_field(ax, spec, magnitude=False):
    import matplotlib.pyplot as plt  # noqa: F401
    L = spec.lim
    g = np.linspace(-L, L, 25)
    X, Y = np.meshgrid(g, g)
    UV = np.array([[spec.fn(x, y) for x, y in zip(rx, ry)] for rx, ry in zip(X, Y)])
    if magnitude:
        M = np.hypot(UV[..., 0], UV[..., 1])
        ax.contourf(X, Y, M, levels=15, cmap="Greys", alpha=0.35)
    else:
        ax.quiver(X, Y, UV[..., 0], UV[..., 1], color="0.5", alpha=0.6)
    gf = np.linspace(-L, L, 121)
    XF, YF = np.meshgrid(gf, gf)
    D = det_j_grid(spec, XF, YF)
    if D.min() < 0 < D.max():
        ax.contour(XF, YF, D, levels=[0], colors="tab:red", linestyles=":", linewidths=1)
    B = np.vectorize(lambda x, y: float(spec.basin(x, y)))(XF, YF)
    if 0 < B.mean() < 1:
        ax.contour(XF, YF, B, levels=[0.5], colors="tab:purple", linestyles="--", linewidths=1)
    for label, x, y in spec.crit:
        if abs(x) <= L and abs(y) <= L:
            ax.plot(x, y, "k+" if label.startswith("center") else "kx", ms=12 if x == y == 0 else 8, mew=2)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")


def build_parser(description, default_out):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--field", default="gyre", choices=FIELD_NAMES)
    p.add_argument("--eps", type=float, default=1.0,
                   help="curvature for potential_saddle / cellular_saddle (ignored otherwise)")
    p.add_argument("--side", type=float, default=0.33, help="triangle side length (m)")
    p.add_argument("--alpha", type=float, default=0.7, help="robot velocity lag (momentum_alpha)")
    p.add_argument("--stiction", type=float, default=0.025, help="stiction threshold (m/s)")
    p.add_argument("--attract-steps", type=int, default=300)
    p.add_argument("--orbit-steps", type=int, default=1500)
    p.add_argument("--radii", type=float, nargs="+", default=ORBIT_RADII)
    p.add_argument("--out-dir", default=default_out)
    return p


def run_suite(method, args):
    """Attraction from ATTRACT_STARTS and orbits at args.radii for one configuration.
    Writes two figures and a summary table; returns the summary lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = make_field(args.field, args.eps)
    mag = method == "magnitude"
    cfg = f"{spec.tag}_side{args.side:g}_a{args.alpha:g}_st{args.stiction:g}"
    out = os.path.join(args.out_dir, cfg)
    os.makedirs(out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    title = "magnitude only" if mag else "vector plane fit"
    log(f"{title}: field {spec.tag} ({spec.note}), side {args.side} m, "
        f"alpha {args.alpha}, stiction {args.stiction} m/s")
    log("")
    log(f"ATTRACTION ({'scalar_gradient_descent on |v|' if mag else 'critical_point_plane_fitting'}), "
        f"{args.attract_steps * DT:.0f} s")
    log(f"{'start':>16} {'basin rule':>10} {'final centroid':>20} {'nearest':>18} "
        f"{'dist':>7} {'t to 2 cm':>9} {'max side err':>12}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, start in zip(axes.flat, ATTRACT_STARTS):
        cl = make_cluster(spec, start, args.side, args.alpha, args.stiction, magnitude=mag)
        r = run(cl, attract_primitive(method), args.attract_steps, spec, method)
        m = attract_metrics(r, spec, args.side)
        if r.escaped:
            log(f"({start[0]:+.2f},{start[1]:+.2f})  escaped")
        else:
            name, dist, _ = nearest_crit(spec, r.center[-1])
            t2 = "never" if np.isnan(m["t_2cm"]) else f"{m['t_2cm']:.1f} s"
            log(f"({start[0]:+.2f},{start[1]:+.2f}) {'origin' if spec.basin(*start) else 'other':>10} "
                f"({r.center[-1][0]:+.4f},{r.center[-1][1]:+.4f}) {name:>18} {dist:7.4f} "
                f"{t2:>9} {m['max_side_err']:12.1e}")
        draw_field(ax, spec, magnitude=mag)
        if len(r.robots):
            for k, col in enumerate(["tab:blue", "tab:orange", "tab:green"]):
                ax.plot(r.robots[:, k, 0], r.robots[:, k, 1], color=col, lw=1, alpha=0.5, ls="--")
            ax.plot(r.center[:, 0], r.center[:, 1], "k-", lw=2)
            ax.plot(*r.center[0], "*", color="lime", ms=15, mec="k")
            ax.plot(*r.center[-1], "X", color="red", ms=12, mec="k")
        ax.set_title(f"start ({start[0]:+.2f}, {start[1]:+.2f})")
    fig.suptitle(f"{title}, attraction, {cfg}\nred dotted: det J = 0, purple dashed: "
                 "predicted basin edge of (0,0)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "attract.png"), dpi=130)
    plt.close(fig)

    w0, w1 = 100, args.orbit_steps
    log("")
    log(f"ORBIT ({'magnitude estimate + Eq. 11 law' if mag else 'critical_point_orbiter_plane_fitting'}), "
        f"start ({ORBIT_START_X}, 0), stats over {w0 * DT:.0f} to {w1 * DT:.0f} s")
    log(f"{'r_d':>5} {'mean r':>7} {'std r':>6} {'ctr off':>8} {'cos2phi':>8} "
        f"{'med est err':>11} {'max side err':>12} {'status':>8}")
    fig, axes = plt.subplots(1, len(args.radii), figsize=(5 * len(args.radii), 5.4))
    for ax, rd in zip(np.atleast_1d(axes), args.radii):
        cl = make_cluster(spec, (ORBIT_START_X, 0.0), args.side, args.alpha, args.stiction, magnitude=mag)
        r = run(cl, orbit_primitive(method, rd), args.orbit_steps, spec, method)
        m = orbit_metrics(r, args.side, w0)
        if "mean_r" in m:
            log(f"{rd:5.2f} {m['mean_r']:7.3f} {m['std_r']:6.3f} {m['ctr_off']:8.3f} {m['c2']:+8.3f} "
                f"{m['est_err']:11.4f} {m['max_side_err']:12.1e} {'FAILED' if m['failed'] else 'ok':>8}")
        else:
            log(f"{rd:5.2f}  escaped")
        draw_field(ax, spec, magnitude=mag)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rd * np.cos(th), rd * np.sin(th), color="tab:red", lw=1, ls="--")
        if len(r.center):
            ax.plot(r.center[:, 0], r.center[:, 1], "k-", lw=1.1)
            ax.plot(r.est[w0:, 0], r.est[w0:, 1], ".", color="tab:purple", ms=2)
            ax.plot(*r.center[0], "*", color="lime", ms=15, mec="k")
        ax.set_title(f"r_d = {rd:.2f}" + (f", mean r = {m['mean_r']:.3f}" if "mean_r" in m else ""))
    fig.suptitle(f"{title}, orbit, {cfg} (red dashed: desired circle, purple: estimates)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "orbit.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")
    return lines
