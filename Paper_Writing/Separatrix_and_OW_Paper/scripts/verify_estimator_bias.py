"""
verify_estimator_bias.py

Verification table for the Section III-D rewrite. Writes NO figure and NO .tex.
Prints a table for manual review; numbers enter the paper only by hand.

Claims under test, in the order the section makes them:

  Stage 1 (linear)       coefficient std matches gamma_q * sigma_eff / rho^q,
                         and the gradient block is isotropic at
                         (4/10) sigma^2 rho^-2 I.

  Stage 2 (bilinear)     every term of the paper's (25)-(27) pairs one u-coefficient
                         with one v-coefficient. Sensor noise is independent across
                         components, so E[D_hat] equals the estimator's OWN noise-free
                         fit, exactly, at every noise level.

                         The baseline matters. theta_hat = theta_fit + Phi^-1 eta, where
                         theta_fit already carries truncation bias, so unbiasedness is
                         relative to theta_fit and NOT to the true field value. The two
                         error sources separate cleanly for the D family: truncation is
                         pure bias with no variance, noise is pure variance with no bias.
                         Measured against the true value instead, the residual that
                         survives at every noise level is the O(1) floor e.

  Stage 3 (non-poly)     s1 = mu - ||(a,b)||. The norm is convex and the (a,b) noise
                         is isotropic, so r is Rice-distributed and biased UP, making
                         s1 biased DOWN by sigma_r^2/(2r) away from degeneracy and
                         sqrt(pi/2)*sigma_r at r = 0, with
                         sigma_r = gamma_1 * sigma_eff / (sqrt(2) rho).

  eps2 vs eps1           position noise corrupts robot i's u and v through the SAME
                         displacement, correlating them and breaking the independence
                         Stage 2 relies on. So position noise biases the D family at
                         O(sigma_p^2) where sensor noise does not.

Ground truth is closed form, asserted against central differences at import time
because the previously committed `_analytic_grad_det` had three of six second
derivatives wrong.

Usage:
    venv/bin/python3 scripts/verify_estimator_bias.py
    venv/bin/python3 scripts/verify_estimator_bias.py --n-trials 200000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_DIR, VFR_ROOT  # noqa: E402  (also fixes sys.path for src.*)

import numpy as np  # noqa: E402

from src.robot.pentagon_cluster import PentagonCluster  # noqa: E402
from src.fields.field_types import AnalyticalField  # noqa: E402
from src.fields.environments.Double_Gyre import double_gyre_static  # noqa: E402
from src.control.pentagon_primitives import (  # noqa: E402
    _fit_vector_quadratic, _det_value, _det_gradient, _det_hessian,
    _strain_quantities, _get_relative_positions,
)

# Double gyre at t=0 with the repo defaults (A=0.1, eps=0 so it is static).
A_DEFAULT = 0.1
FORMATION = "config/formations/pentagon_small.yaml"

PARAMS = {
    "seed": 20260729,
    "n_trials": 10000,          # matches the Monte Carlo cell size used elsewhere
    # fine grid between 0.001 and 0.01 to resolve where the H_D eigendirections
    # go from usable to random, which is also the D tracker's failure cliff
    "sigma_uv_vals": [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.02, 0.05],
    "sigma_p_vals": [0.005, 0.01, 0.02, 0.05],
    "points": {
        # r = pi*A*|cos(pi(x+1)) cos(pi(y+0.5))|, and the shear part is
        # identically zero for this field, so r -> 0 on y = 0 and on x = -0.5.
        "generic":       [-0.3, 0.1],    # the centroid the old script used
        "near_degenerate": [-0.3, 0.01],  # r small but nonzero
        "degenerate":    [-0.3, 0.0],    # r = 0 exactly
    },
    "formation_config": FORMATION,
}


# ---------------------------------------------------------------------------
# Closed-form field derivatives (verified against finite differences below)
# ---------------------------------------------------------------------------
# Streamfunction psi = A sin(pi xf) sin(pi yf) with xf = x+1, yf = y+0.5 gives
#     u = -pi A sin(pi xf) cos(pi yf)
#     v =  pi A cos(pi xf) sin(pi yf)

def _trig(x, y, A=A_DEFAULT):
    k = np.pi
    xf, yf = x + 1.0, y + 0.5
    return (k, A * k, np.sin(k * xf), np.cos(k * xf), np.sin(k * yf), np.cos(k * yf))


def field_derivs(x, y, A=A_DEFAULT):
    """All derivatives of (u, v) through third order, as a dict."""
    k, P, Sx, Cx, Sy, Cy = _trig(x, y, A)
    return {
        "u":     -P * Sx * Cy,
        "v":      P * Cx * Sy,
        # first
        "ux":    -P * k * Cx * Cy,
        "uy":     P * k * Sx * Sy,
        "vx":    -P * k * Sx * Sy,
        "vy":     P * k * Cx * Cy,
        # second
        "uxx":    P * k**2 * Sx * Cy,
        "uxy":    P * k**2 * Cx * Sy,
        "uyy":    P * k**2 * Sx * Cy,
        "vxx":   -P * k**2 * Cx * Sy,
        "vxy":   -P * k**2 * Sx * Cy,
        "vyy":   -P * k**2 * Cx * Sy,
        # third
        "uxxx":   P * k**3 * Cx * Cy,
        "uxxy":  -P * k**3 * Sx * Sy,
        "uxyy":   P * k**3 * Cx * Cy,
        "uyyy":  -P * k**3 * Sx * Sy,
        "vxxx":   P * k**3 * Sx * Sy,
        "vxxy":  -P * k**3 * Cx * Cy,
        "vxyy":   P * k**3 * Sx * Sy,
        "vyyy":  -P * k**3 * Cx * Cy,
    }


def true_D(x, y):
    d = field_derivs(x, y)
    return d["ux"] * d["vy"] - d["uy"] * d["vx"]


def true_grad_D(x, y):
    d = field_derivs(x, y)
    Dx = d["uxx"] * d["vy"] + d["ux"] * d["vxy"] - d["uxy"] * d["vx"] - d["uy"] * d["vxx"]
    Dy = d["uxy"] * d["vy"] + d["ux"] * d["vyy"] - d["uyy"] * d["vx"] - d["uy"] * d["vxy"]
    return np.array([Dx, Dy])


def true_hess_D(x, y):
    d = field_derivs(x, y)
    Dxx = (d["uxxx"] * d["vy"] + 2 * d["uxx"] * d["vxy"] + d["ux"] * d["vxxy"]
           - d["uxxy"] * d["vx"] - 2 * d["uxy"] * d["vxx"] - d["uy"] * d["vxxx"])
    Dyy = (d["uxyy"] * d["vy"] + 2 * d["uxy"] * d["vyy"] + d["ux"] * d["vyyy"]
           - d["uyyy"] * d["vx"] - 2 * d["uyy"] * d["vxy"] - d["uy"] * d["vxyy"])
    Dxy = (d["uxxy"] * d["vy"] + d["uxx"] * d["vyy"] + d["ux"] * d["vxyy"]
           - d["uxyy"] * d["vx"] - d["uyy"] * d["vxx"] - d["uy"] * d["vxxy"])
    return np.array([[Dxx, Dxy], [Dxy, Dyy]])


def true_strain(x, y):
    """Returns (s1, r, e1) at (x, y) for the true field."""
    d = field_derivs(x, y)
    mu = 0.5 * (d["ux"] + d["vy"])
    a = 0.5 * (d["ux"] - d["vy"])
    b = 0.5 * (d["uy"] + d["vx"])
    r = float(np.hypot(a, b))
    S = np.array([[d["ux"], b], [b, d["vy"]]])
    _, V = np.linalg.eigh(S)
    return float(mu - r), r, V[:, 0]


# ---------------------------------------------------------------------------
# Self-check: closed forms against central differences
# ---------------------------------------------------------------------------

def _self_check(verbose=True):
    """Assert the closed forms above against finite differences of the field."""
    def u(x, y):
        return double_gyre_static(x, y)[0]

    def v(x, y):
        return double_gyre_static(x, y)[1]

    def d1(f, x, y, ax, h=1e-5):
        return ((f(x + h, y) - f(x - h, y)) / (2 * h) if ax == 0
                else (f(x, y + h) - f(x, y - h)) / (2 * h))

    def d2(f, x, y, ax, ay, h=1e-4):
        if ax == 2:
            return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
        if ay == 2:
            return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
        return (f(x + h, y + h) - f(x + h, y - h)
                - f(x - h, y + h) + f(x - h, y - h)) / (4 * h * h)

    failures = []
    for (x, y) in [(-0.3, 0.1), (-0.62, 0.27), (0.15, -0.33)]:
        d = field_derivs(x, y)
        checks = {
            "u": u(x, y), "v": v(x, y),
            "ux": d1(u, x, y, 0), "uy": d1(u, x, y, 1),
            "vx": d1(v, x, y, 0), "vy": d1(v, x, y, 1),
            "uxx": d2(u, x, y, 2, 0), "uxy": d2(u, x, y, 1, 1), "uyy": d2(u, x, y, 0, 2),
            "vxx": d2(v, x, y, 2, 0), "vxy": d2(v, x, y, 1, 1), "vyy": d2(v, x, y, 0, 2),
        }
        for k, ref in checks.items():
            if abs(d[k] - ref) > 1e-3 * max(1.0, abs(ref)):
                failures.append((x, y, k, d[k], ref))

        # Third derivatives, and the D-chain, via differences of the closed forms.
        h = 1e-5
        for k, (base, ax) in {
            "uxxx": ("uxx", 0), "uxxy": ("uxx", 1),
            "uxyy": ("uxy", 1), "uyyy": ("uyy", 1),
            "vxxx": ("vxx", 0), "vxxy": ("vxx", 1),
            "vxyy": ("vxy", 1), "vyyy": ("vyy", 1),
        }.items():
            fwd = field_derivs(x + h, y)[base] if ax == 0 else field_derivs(x, y + h)[base]
            bwd = field_derivs(x - h, y)[base] if ax == 0 else field_derivs(x, y - h)[base]
            ref = (fwd - bwd) / (2 * h)
            if abs(d[k] - ref) > 1e-3 * max(1.0, abs(ref)):
                failures.append((x, y, k, d[k], ref))

        # grad D and H_D against differences of true_D / true_grad_D.
        gD_ref = np.array([(true_D(x + h, y) - true_D(x - h, y)) / (2 * h),
                           (true_D(x, y + h) - true_D(x, y - h)) / (2 * h)])
        if np.max(np.abs(true_grad_D(x, y) - gD_ref)) > 1e-3 * max(1.0, np.max(np.abs(gD_ref))):
            failures.append((x, y, "grad_D", true_grad_D(x, y), gD_ref))

        H_ref = np.zeros((2, 2))
        H_ref[:, 0] = (true_grad_D(x + h, y) - true_grad_D(x - h, y)) / (2 * h)
        H_ref[:, 1] = (true_grad_D(x, y + h) - true_grad_D(x, y - h)) / (2 * h)
        H_ref = 0.5 * (H_ref + H_ref.T)
        if np.max(np.abs(true_hess_D(x, y) - H_ref)) > 1e-3 * max(1.0, np.max(np.abs(H_ref))):
            failures.append((x, y, "H_D", true_hess_D(x, y), H_ref))

    if failures:
        print("SELF-CHECK FAILED:")
        for f in failures:
            print("   ", f)
        raise SystemExit(1)
    if verbose:
        print("  self-check: closed-form derivatives agree with finite differences "
              "at 3 points, through third order, including grad D and H_D.")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def _make_geometry(xc, yc):
    field = AnalyticalField(double_gyre_static)
    cluster = PentagonCluster(PARAMS["formation_config"], field)
    cluster.reset(xc, yc)
    rel = np.asarray(_get_relative_positions(cluster))
    abs_pos = rel + np.array([xc, yc])
    return rel, abs_pos


def _rice_mean_excess(r0, sigma):
    """E[||(a,b) + noise||] - r0 for isotropic 2D Gaussian noise of scale sigma.

    Exact Rice mean: E[R] = sigma sqrt(pi/2) L_{1/2}(-K), K = r0^2/(2 sigma^2),
    with L_{1/2}(-K) = exp(-K/2)[(1+K) I0(K/2) + K I1(K/2)]. Reduces to
    sigma sqrt(pi/2) at r0 = 0 and to r0 + sigma^2/(2 r0) for r0 >> sigma.
    """
    from scipy.special import ive  # exponentially scaled I_n, stable for large K
    if sigma <= 0:
        return 0.0
    K = r0 * r0 / (2.0 * sigma * sigma)
    h = K / 2.0
    # ive(n, h) = I_n(h) exp(-h), so exp(-K/2) I_n(K/2) = ive(n, h)
    lag = (1.0 + K) * ive(0, h) + K * ive(1, h)
    return float(sigma * np.sqrt(np.pi / 2.0) * lag - r0)


def _angle_deg(e, e_ref):
    """Angle between two axes in [0, 90] deg; eigenvector sign is arbitrary."""
    c = abs(float(np.dot(e, e_ref)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _clean_readings(abs_pos):
    u = np.array([double_gyre_static(px, py)[0] for px, py in abs_pos])
    v = np.array([double_gyre_static(px, py)[1] for px, py in abs_pos])
    return u, v


def noise_free_fit(xc, yc):
    """The estimator's own output with no noise: carries truncation bias only."""
    rel, abs_pos = _make_geometry(xc, yc)
    u0, v0 = _clean_readings(abs_pos)
    th_u, th_v = _fit_vector_quadratic(rel, u0, v0)
    s1, _g, e1, _e2, r = _strain_quantities(th_u, th_v)
    return {
        "D": _det_value(th_u, th_v),
        "gD": _det_gradient(th_u, th_v),
        "H": _det_hessian(th_u, th_v),
        "s1": s1, "r": r, "e1": e1,
        "theta_u": th_u, "theta_v": th_v,
    }


def run_point(xc, yc, sigma_uv, sigma_p, n_trials, rng):
    """Signed-error Monte Carlo at one point and one noise setting.

    Errors are recorded against the noise-free FIT (suffix-free keys), which is
    what the unbiasedness claim is about, and against the TRUE field value
    (keys prefixed 't_'), which additionally carries truncation.
    """
    rel, abs_pos = _make_geometry(xc, yc)
    u0, v0 = _clean_readings(abs_pos)

    base = noise_free_fit(xc, yc)
    D_t = base["D"]
    gD_t = base["gD"]
    H_t = base["H"]
    s1_t, r_t, e1_t = base["s1"], base["r"], base["e1"]

    D_true_ = true_D(xc, yc)
    gD_true_ = true_grad_D(xc, yc)
    H_true_ = true_hess_D(xc, yc)
    s1_true_, r_true_, _e1_true = true_strain(xc, yc)

    keys = ("D", "gDx", "gDy", "Hxx", "Hxy", "Hyy", "s1", "r")
    rec = {k: [] for k in keys}
    rec.update({f"t_{k}": [] for k in keys})
    rec.update({k: [] for k in ("ang", "t_ang", "Hang", "a2", "a3", "a5", "a4", "a6")})

    def _principal(M):
        """Eigenvector of the most negative eigenvalue of a symmetric 2x2."""
        _w, V = np.linalg.eigh(M)
        return V[:, 0]

    eH_true = _principal(H_true_)

    for _ in range(n_trials):
        if sigma_p > 0.0:
            xi = rng.normal(0.0, sigma_p, size=(6, 2))
            um, vm = _clean_readings(abs_pos + xi)
        else:
            um, vm = u0.copy(), v0.copy()
        if sigma_uv > 0.0:
            um = um + rng.normal(0.0, sigma_uv, size=6)
            vm = vm + rng.normal(0.0, sigma_uv, size=6)

        th_u, th_v = _fit_vector_quadratic(rel, um, vm)
        D = _det_value(th_u, th_v)
        g = _det_gradient(th_u, th_v)
        H = _det_hessian(th_u, th_v)
        s1, _gs1, e1, _e2, r = _strain_quantities(th_u, th_v)

        # against the noise-free fit: isolates what noise alone does
        rec["D"].append(D - D_t)
        rec["gDx"].append(g[0] - gD_t[0])
        rec["gDy"].append(g[1] - gD_t[1])
        rec["Hxx"].append(H[0, 0] - H_t[0, 0])
        rec["Hxy"].append(H[0, 1] - H_t[0, 1])
        rec["Hyy"].append(H[1, 1] - H_t[1, 1])
        rec["s1"].append(s1 - s1_t)
        rec["r"].append(r - r_t)
        rec["ang"].append(_angle_deg(e1, e1_t))

        # against the true field: noise plus truncation
        rec["t_D"].append(D - D_true_)
        rec["t_gDx"].append(g[0] - gD_true_[0])
        rec["t_gDy"].append(g[1] - gD_true_[1])
        rec["t_Hxx"].append(H[0, 0] - H_true_[0, 0])
        rec["t_Hxy"].append(H[0, 1] - H_true_[0, 1])
        rec["t_Hyy"].append(H[1, 1] - H_true_[1, 1])
        rec["t_s1"].append(s1 - s1_true_)
        rec["t_r"].append(r - r_true_)
        rec["t_ang"].append(_angle_deg(e1, _e1_true))
        rec["Hang"].append(_angle_deg(_principal(H), eH_true))

        # raw coefficients, code slot order [f, fx, fy, fxx, fxy, fyy]
        for name, idx in (("a2", 1), ("a3", 2), ("a5", 3), ("a4", 4), ("a6", 5)):
            rec[name].append(th_u[idx])

    out = {}
    for k, arr in rec.items():
        a = np.asarray(arr, dtype=float)
        out[k] = {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "sem": float(a.std(ddof=1) / np.sqrt(len(a))),
        }
    d0 = field_derivs(xc, yc)
    out["_truth"] = {
        "D_true": float(D_true_), "D_fit": float(D_t),
        "s1_true": float(s1_true_), "s1_fit": float(s1_t),
        "r_true": float(r_true_), "r_fit": float(r_t),
        "trunc_D": float(D_t - D_true_),
        "trunc_Hxx": float(H_t[0, 0] - H_true_[0, 0]),
        "trunc_Hyy": float(H_t[1, 1] - H_true_[1, 1]),
        "H_true_xx": float(H_true_[0, 0]), "H_true_yy": float(H_true_[1, 1]),
        "J_fro": float(np.linalg.norm(
            np.array([[d0["ux"], d0["uy"]], [d0["vx"], d0["vy"]]]), "fro")),
    }
    return out


def _t_ratio(stat):
    """Signed mean in units of its own standard error."""
    return stat["mean"] / stat["sem"] if stat["sem"] > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Estimator bias verification (no figure).")
    ap.add_argument("--n-trials", type=int, default=PARAMS["n_trials"])
    ap.add_argument("--seed", type=int, default=PARAMS["seed"])
    ap.add_argument("--json-out", type=Path,
                    default=PAPER_DIR / "scripts" / "verify_estimator_bias.json")
    args = ap.parse_args()

    PARAMS["n_trials"] = args.n_trials
    PARAMS["seed"] = args.seed
    rng = np.random.default_rng(args.seed)
    N = args.n_trials

    print(f"\nverify_estimator_bias  (N = {N} per cell, seed = {args.seed})")
    print(f"repo: {VFR_ROOT.name}   formation: {FORMATION}")
    _self_check()

    rel, _ = _make_geometry(-0.3, 0.1)
    rho = float(np.median([n for n in np.linalg.norm(rel, axis=1) if n > 1e-9]))
    g1, g2m, g2p = 2 / np.sqrt(10), 4 / np.sqrt(10), 8 / np.sqrt(10)
    print(f"ring radius rho = {rho:.6f}   gamma_1 = {g1:.4f}  "
          f"gamma_2(mixed) = {g2m:.4f}  gamma_2(pure) = {g2p:.4f}")

    results = {}

    # ---------------- Stage 1: ladder and isotropy ----------------
    print("\n" + "=" * 78)
    print("STAGE 1  coefficient noise ladder   sigma_q = gamma_q sigma_eff / rho^q")
    print("=" * 78)
    print(f"{'sigma_uv':>9} {'coef':>5} {'predicted':>12} {'measured':>12} {'rel err %':>10}")
    lad = {}
    for s in PARAMS["sigma_uv_vals"]:
        R = run_point(-0.3, 0.1, s, 0.0, N, rng)
        for name, gam, q in (("a2", g1, 1), ("a3", g1, 1),
                             ("a5", g2p, 2), ("a4", g2m, 2), ("a6", g2p, 2)):
            pred = gam * s / rho**q
            meas = R[name]["std"]
            err = 100 * (meas - pred) / pred
            print(f"{s:9.3f} {name:>5} {pred:12.4f} {meas:12.4f} {err:+10.2f}")
            lad[f"{name}@{s}"] = {"pred": pred, "meas": meas, "rel_err_pct": err}
        results[f"ladder_sigma{s}"] = R
    results["ladder_table"] = lad

    # ---------------- Stage 2: unbiasedness under eps1 ----------------
    print("\n" + "=" * 78)
    print("STAGE 2  D family under SENSOR noise, measured against the NOISE-FREE FIT.")
    print("         Claim: noise adds no bias, so the signed mean is 0.")
    print("         t = mean / SEM.  |t| < 3 is consistent with exact unbiasedness.")
    print("=" * 78)
    print(f"{'sigma_uv':>9} {'qty':>5} {'signed mean':>14} {'SEM':>12} {'t':>7} "
          f"{'|mean|/std':>11}")
    worst = 0.0
    for s in PARAMS["sigma_uv_vals"]:
        R = results[f"ladder_sigma{s}"]
        for q in ("D", "gDx", "gDy", "Hxx", "Hxy", "Hyy"):
            st = R[q]
            frac = abs(st["mean"]) / st["std"] if st["std"] > 0 else float("nan")
            worst = max(worst, abs(_t_ratio(st)))
            print(f"{s:9.3f} {q:>5} {st['mean']:+14.6g} {st['sem']:12.4g} "
                  f"{_t_ratio(st):+7.2f} {frac:11.4f}")
    print(f"\n  largest |t| over all cells = {worst:.2f}   "
          f"(N = {N}, so SEM resolves a bias of about {3/np.sqrt(N)*100:.1f}% of one std)")

    # ---------------- truncation: pure bias, no variance ----------------
    print("\n" + "=" * 78)
    print("TRUNCATION  noise-free fit minus true field. Deterministic, no Monte Carlo.")
    print("            This is the residual that survives at every noise level,")
    print("            and for H_D it is the O(1) floor e.")
    print("=" * 78)
    print(f"{'point':>16} {'D_true':>11} {'D_fit':>11} {'trunc D':>11} "
          f"{'H_xx true':>11} {'H_xx fit':>11} {'floor/|H|':>10}")
    trunc = {}
    for label, (px, py) in PARAMS["points"].items():
        b = noise_free_fit(px, py)
        Ht = true_hess_D(px, py)
        Dt = true_D(px, py)
        rel_floor = (abs(b["H"][0, 0] - Ht[0, 0]) / abs(Ht[0, 0])
                     if abs(Ht[0, 0]) > 0 else float("nan"))
        print(f"{label:>16} {Dt:11.5f} {b['D']:11.5f} {b['D']-Dt:+11.5f} "
              f"{Ht[0,0]:11.4f} {b['H'][0,0]:11.4f} {rel_floor:10.3f}")
        trunc[label] = {"D_true": Dt, "D_fit": b["D"], "trunc_D": b["D"] - Dt,
                        "H_xx_true": float(Ht[0, 0]), "H_xx_fit": float(b["H"][0, 0]),
                        "rel_floor_Hxx": rel_floor}
    results["truncation"] = trunc

    # ---------------- Stage 3: s1 bias ----------------
    print("\n" + "=" * 78)
    print("STAGE 3  s1 = mu - r is biased DOWN because r is Rice-biased UP")
    print("         predicted:  -sqrt(pi/2)*sigma_r at r=0;  -sigma_r^2/(2r) for r >> sigma_r")
    print("         sigma_r = gamma_1 sigma_eff / (sqrt(2) rho)")
    print("=" * 78)
    print(f"{'point':>16} {'r_fit':>9} {'sigma_uv':>9} {'sigma_r':>9} "
          f"{'s1 bias':>12} {'Rice exact':>12} {'asympt':>12} {'ratio':>7}")
    s3 = {}
    for label, (px, py) in PARAMS["points"].items():
        r_fit = noise_free_fit(px, py)["r"]
        for s in PARAMS["sigma_uv_vals"]:
            R = run_point(px, py, s, 0.0, N, rng)
            sig_r = g1 * s / (np.sqrt(2) * rho)
            pred = -_rice_mean_excess(r_fit, sig_r)
            asy = (-np.sqrt(np.pi / 2) * sig_r if r_fit < 1e-12
                   else -sig_r**2 / (2 * r_fit))
            st = R["s1"]
            ratio = st["mean"] / pred if pred != 0 else float("nan")
            print(f"{label:>16} {r_fit:9.5f} {s:9.3f} {sig_r:9.5f} "
                  f"{st['mean']:+12.6f} {pred:+12.6f} {asy:+12.6f} {ratio:7.3f}")
            s3[f"{label}@{s}"] = {"r_fit": r_fit, "sigma_r": sig_r,
                                  "s1_bias": st["mean"], "rice_exact": pred,
                                  "asymptotic": asy, "sem": st["sem"],
                                  "eig_ang_vs_fit_deg": R["ang"]["mean"],
                                  "eig_ang_vs_true_deg": R["t_ang"]["mean"]}
    results["stage3"] = s3

    # ---------------- eigenframe accuracy ----------------
    print("\n" + "=" * 78)
    print("EIGENFRAME  mean angular error of e1 (deg), the 1/r exposure")
    print("=" * 78)
    print(f"{'point':>16} {'r_fit':>9} " + "".join(
        f"{('s=' + str(s)):>11}" for s in PARAMS["sigma_uv_vals"]))
    for label, (px, py) in PARAMS["points"].items():
        r_fit = noise_free_fit(px, py)["r"]
        row = "".join(f"{s3[f'{label}@{s}']['eig_ang_vs_true_deg']:11.3f}"
                      for s in PARAMS["sigma_uv_vals"])
        print(f"{label:>16} {r_fit:9.5f} " + row)

    # ---------------- H_D eigendirections ----------------
    print("\n" + "=" * 78)
    print("H_D EIGENDIRECTIONS  mean angle (deg) between the fitted and true principal")
    print("       eigenvector of H_D. The paper's claim is that the O(1) floor lands on")
    print("       the eigenvalues and leaves the directions usable.")
    print("       For this field H_D is diagonal everywhere, so the true axes are exactly")
    print("       the coordinate axes and the directions are degenerate where Dxx = Dyy.")
    print("=" * 78)
    print("       The s=0 column is truncation ALONE and is the claim's best case.")
    print(f"{'point':>16} {'H_xx true':>10} {'H_yy true':>10} {'s=0':>10}" + "".join(
        f"{('s=' + str(s)):>10}" for s in PARAMS["sigma_uv_vals"]))
    hang = {}
    for label, (px, py) in PARAMS["points"].items():
        Ht = true_hess_D(px, py)
        _w, Vt = np.linalg.eigh(Ht)
        _w, Vf = np.linalg.eigh(noise_free_fit(px, py)["H"])
        a0 = _angle_deg(Vf[:, 0], Vt[:, 0])
        hang[f"{label}@0"] = a0
        row = f"{a0:10.3f}"
        for s in PARAMS["sigma_uv_vals"]:
            R = (results[f"ladder_sigma{s}"] if label == "generic"
                 else run_point(px, py, s, 0.0, N, rng))
            row += f"{R['Hang']['mean']:10.3f}"
            hang[f"{label}@{s}"] = R["Hang"]["mean"]
        print(f"{label:>16} {Ht[0,0]:10.4f} {Ht[1,1]:10.4f} " + row)
    results["H_eigendirections"] = hang

    # ---------------- eps2: position noise ----------------
    print("\n" + "=" * 78)
    print("EPS2  position noise ALONE. Shared displacement correlates a robot's u and v,")
    print("      breaking the independence Stage 2 needs, so D should acquire a bias")
    print("      that grows like sigma_p^2 (contrast: sensor noise gives none).")
    print("=" * 78)
    print(f"{'sigma_p':>9} {'D signed mean':>16} {'SEM':>12} {'t':>8} "
          f"{'bias/sigma_p^2':>16} {'sigma_eff pred':>15} {'meas D std':>12}")
    e2 = {}
    Jf = None
    for sp in PARAMS["sigma_p_vals"]:
        R = run_point(-0.3, 0.1, 0.0, sp, N, rng)
        Jf = R["_truth"]["J_fro"]
        st = R["D"]
        sig_eff = np.sqrt(0.5 * Jf**2 * sp**2)
        print(f"{sp:9.3f} {st['mean']:+16.6g} {st['sem']:12.4g} {_t_ratio(st):+8.1f} "
              f"{st['mean'] / sp**2:+16.4f} {sig_eff:15.5f} {st['std']:12.5f}")
        e2[str(sp)] = {"D_mean": st["mean"], "sem": st["sem"], "D_std": st["std"],
                       "sigma_eff_pred": sig_eff}
    results["eps2"] = e2
    print(f"\n  ||J||_F at the test point = {Jf:.5f}; "
          f"sigma_eff^2 = sigma_uv^2 + 0.5||J||_F^2 sigma_p^2")

    # ---------------- eq (21) at the reading level ----------------
    print("\n" + "=" * 78)
    print("SIGMA_EFF  eq (21) is a claim about the per-robot READING error, not about D.")
    print("           A displaced robot commits J*xi to first order, so per robot the")
    print("           u-channel error has variance sigma_p^2 ||grad u||^2, and averaged")
    print("           over both channels sigma_eff^2 = sigma_uv^2 + 0.5||J||_F^2 sigma_p^2.")
    print("=" * 78)
    rel, abs_pos = _make_geometry(-0.3, 0.1)
    u0, v0 = _clean_readings(abs_pos)
    print(f"{'sigma_p':>9} {'meas read std':>15} {'predicted':>12} {'rel err %':>10} "
          f"{'corr(du,dv)':>12} {'pred corr':>10}")
    seff = {}
    for sp in PARAMS["sigma_p_vals"]:
        du, dv = [], []
        for _ in range(N):
            um, vm = _clean_readings(abs_pos + rng.normal(0.0, sp, size=(6, 2)))
            du.extend(um - u0)
            dv.extend(vm - v0)
        du, dv = np.asarray(du), np.asarray(dv)
        meas = float(np.sqrt(0.5 * (du.var(ddof=1) + dv.var(ddof=1))))
        # predicted, averaged over the six robot locations
        gs = [field_derivs(px, py) for px, py in abs_pos]
        pred = float(np.sqrt(np.mean([0.5 * sp**2 * (g["ux"]**2 + g["uy"]**2
                                                     + g["vx"]**2 + g["vy"]**2)
                                      for g in gs])))
        corr = float(np.corrcoef(du, dv)[0, 1])
        pcorr = float(np.mean([(g["ux"] * g["vx"] + g["uy"] * g["vy"])
                               / np.sqrt((g["ux"]**2 + g["uy"]**2)
                                         * (g["vx"]**2 + g["vy"]**2)) for g in gs]))
        print(f"{sp:9.3f} {meas:15.6f} {pred:12.6f} {100*(meas-pred)/pred:+10.2f} "
              f"{corr:12.4f} {pcorr:10.4f}")
        seff[str(sp)] = {"meas": meas, "pred": pred,
                         "rel_err_pct": 100 * (meas - pred) / pred,
                         "corr_du_dv": corr, "pred_corr": pcorr}
    results["sigma_eff_reading"] = seff
    print("\n  A nonzero corr(du,dv) is the mechanism: one displacement moves both")
    print("  channels at that robot, which is what sensor noise never does.")

    args.json_out.write_text(json.dumps(
        {"params": PARAMS, "rho": rho, "results": results}, indent=2, default=str))
    print(f"\n  json -> {args.json_out.relative_to(PAPER_DIR)}")
    print("  Numbers are for review. Nothing was written to the .tex.\n")


if __name__ == "__main__":
    main()
