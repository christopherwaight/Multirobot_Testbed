"""
estimator_check.py  (Draft_14, Section II-C numbers; no robots, fit only)

Validates the six-robot quadratic fit on the steady double gyre against the
closed forms of Section II-E, at the nominal radius and two others:
  - noise-free truncation error of the fitted J, second derivatives, D,
    grad D, H_D and grad s1 at centroids along the separatrix x = 0
  - Monte Carlo spread of the same quantities under measurement noise,
    against the linear prediction sigma * gain / rho^q for the coefficients

Run from VF_Robot:  venv/bin/python3 experiments/draft14/estimator_check.py
Writes experiments/draft14/outputs/estimator_check.txt
"""
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, root)
os.chdir(root)

from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (_fit_vector_quadratic, _det_gradient,
                                             _det_hessian, _strain_quantities)

A = 0.1
PI = np.pi
RHOS = (0.05, 0.075, 0.1125)
HEIGHTS = np.linspace(-0.45, 0.45, 19)
N_MC = 4000
SIGMAS = (0.001, 0.002, 0.005)


def analytic(x, y):
    """J, D, grad D, H_D, grad s1 of the steady gyre at world (x, y)."""
    xf, yf = x + 1.0, y + 0.5
    cx, sx, cy, sy = np.cos(PI * xf), np.sin(PI * xf), np.cos(PI * yf), np.sin(PI * yf)
    ux, uy = -PI**2 * A * cx * cy, PI**2 * A * sx * sy
    J = np.array([[ux, uy], [-uy, -ux]])
    D = -0.5 * PI**4 * A**2 * (np.cos(2 * PI * xf) + np.cos(2 * PI * yf))
    gD = PI**5 * A**2 * np.array([np.sin(2 * PI * xf), np.sin(2 * PI * yf)])
    HD = 2 * PI**6 * A**2 * np.diag([np.cos(2 * PI * xf), np.cos(2 * PI * yf)])
    # s1 = -pi^2 A |cx cy|; gradient away from its kinks
    sg = np.sign(cx * cy)
    gs1 = PI**3 * A * sg * np.array([sx * cy, cx * sy])
    return J, D, gD, HD, gs1


def ring(rho, heading):
    ang = heading + 2 * PI * np.arange(5) / 5
    return np.vstack([[0.0, 0.0], np.c_[rho * np.cos(ang), rho * np.sin(ang)]])


def fit_at(xc, yc, rho, heading, sigma=0.0, rng=None):
    rel = ring(rho, heading)
    uv = np.array([double_gyre_static(xc + rx, yc + ry) for rx, ry in rel])
    if sigma > 0:
        uv = uv + rng.normal(0.0, sigma, uv.shape)
    tu, tv = _fit_vector_quadratic(rel, uv[:, 0], uv[:, 1])
    J = np.array([[tu[1], tu[2]], [tv[1], tv[2]]])
    D = np.linalg.det(J)
    s1, gs1, _, _, _ = _strain_quantities(tu, tv)
    return dict(J=J, sec=np.r_[tu[3:], tv[3:]], D=D, gD=_det_gradient(tu, tv),
                HD=_det_hessian(tu, tv), gs1=gs1)


def main():
    rng = np.random.default_rng(20261119)
    out = []
    p = out.append
    p("Noise-free truncation error along the separatrix x = 0, |y| <= 0.45 (19 heights,")
    p("8 headings). Relative error = ||fit - true|| / max over the separatrix of ||true||.")
    for rho in RHOS:
        errs = {k: [] for k in ("J", "D", "gD", "HD", "gs1")}
        scale = {k: 0.0 for k in errs}
        for y in HEIGHTS:
            J, D, gD, HD, gs1 = analytic(0.0, y)
            tru = dict(J=J, D=D, gD=gD, HD=HD, gs1=gs1)
            for k in scale:
                scale[k] = max(scale[k], np.linalg.norm(tru[k]))
            for h in np.linspace(0, 2 * PI, 8, endpoint=False):
                f = fit_at(0.0, y, rho, h)
                for k in errs:
                    if k == "gs1" and abs(np.cos(PI * (y + 0.5))) < 0.2:
                        continue        # skip the kink of s1 at y = 0
                    errs[k].append(np.linalg.norm(f[k] - tru[k]))
        p(f"  rho = {rho:.4f}: " + ", ".join(
            f"{k} max {np.max(errs[k]) / scale[k]:.2%} median {np.median(errs[k]) / scale[k]:.2%}"
            for k in errs))
    p("")
    p("Measurement noise at (0, 0.25), N = %d fits per cell, std of fitted quantity." % N_MC)
    p("Pure second derivative: predicted 2.53 sigma / rho^2 vs Monte Carlo.")
    for rho in RHOS:
        for sig in SIGMAS:
            fits = [fit_at(0.0, 0.25, rho, rng.uniform(0, 2 * PI), sig, rng) for _ in range(N_MC)]
            sec = np.array([f["sec"] for f in fits])          # (uxx, uxy, uyy, vxx, vxy, vyy)
            pure = np.r_[sec[:, 0], sec[:, 2], sec[:, 3], sec[:, 5]]
            pure_sd = np.std(pure - np.repeat(sec.mean(0)[[0, 2, 3, 5]], N_MC))
            gD = np.array([f["gD"] for f in fits])
            gs1 = np.array([f["gs1"] for f in fits])
            _, _, gD_t, _, gs1_t = analytic(0.0, 0.25)
            p(f"  rho {rho:.4f} sigma {sig:.4f}: pure 2nd-deriv std {pure_sd:.3f} "
              f"(pred {2.53 * sig / rho**2:.3f}); |grad D| true {np.linalg.norm(gD_t):.3f}, "
              f"rms err {np.sqrt(np.mean(np.sum((gD - gD_t)**2, 1))):.3f}; "
              f"|grad s1| true {np.linalg.norm(gs1_t):.3f}, rms err "
              f"{np.sqrt(np.mean(np.sum((gs1 - gs1_t)**2, 1))):.3f}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(here, "outputs", "estimator_check.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
