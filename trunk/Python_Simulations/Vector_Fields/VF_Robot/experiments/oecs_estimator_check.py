"""
oecs_estimator_check.py

Open-loop diagnosis of the rate-of-strain (OECS) estimator quantities on
the steady double gyre.  Produces the numbers quoted in the paper's
OECS estimation remark and results text:

  1. Fitted shear strain b = (u_y + v_x)/2 residual (analytic value 0).
  2. Eigenvalues of the fitted H_s1: negative semidefinite at every test
     point (structural concavity of the quadratic model's s1), ~0 on the
     double gyre.  This is why the OECS tracker uses no s1 Hessian.
  3. Transverse restoring slope (grad s1 . e1)/offset vs the analytic
     transverse trench curvature pi^4 A |cos(pi y_f)|.
  4. Frame (non-)invariance under a rotating observer: shift in fitted s1
     and pulled-back e2 angle vs shift in fitted det(J), Omega = 0.3,
     t = 2.0 s, material point (0.03, 0.25).
  5. Noise ladder: median errors of s1, e2 angle, grad s1 direction, and
     (for comparison) the H_D eigenvector angle Logic C uses as its
     tangent, over 20000 noise draws per level.  Shows the OECS tangent
     (first-order coefficients) is one noise rung more robust than the
     H_D tangent (second-order coefficients).

Paper traceability: numbers for the OECS estimation remark and the
results subsection; CSV output experiments/outputs/oecs/
oecs_estimator_check_noise.csv, full printout
oecs_estimator_check.txt in the same folder.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import csv
import numpy as np

from src.control.pentagon_primitives import (
    _fit_vector_quadratic, _strain_quantities, _det_value, _det_hessian
)
from src.fields.environments.Double_Gyre import double_gyre_static

A = 0.1
RHO = 0.075
TEST_POINTS = [(0.03, 0.25), (-0.35, -0.20), (0.25, 0.42)]
SEED = 20260709
N_DRAWS = 20000
SIGMA_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05]

OUT_DIR = os.path.join(project_root, "experiments", "outputs", "oecs")


def pentagon_rel():
    ang = np.arange(5) * 2.0 * np.pi / 5.0
    ring = np.column_stack([RHO * np.cos(ang), RHO * np.sin(ang)])
    return np.vstack([np.zeros((1, 2)), ring])


def sample_uv(cx, cy, rel, field=double_gyre_static):
    u = np.array([field(cx + r[0], cy + r[1])[0] for r in rel])
    v = np.array([field(cx + r[0], cy + r[1])[1] for r in rel])
    return u, v


def fitted_H_s1(theta_u, theta_v):
    """Hessian of the fitted model's s1 (diagnostic only; never used
    by the controller)."""
    _, ux, uy, uxx, uxy, uyy = theta_u
    _, vx, vy, vxx, vxy, vyy = theta_v
    a, b = 0.5 * (ux - vy), 0.5 * (uy + vx)
    r = max(np.hypot(a, b), 1e-12)
    ga = 0.5 * np.array([uxx - vxy, uxy - vyy])
    gb = 0.5 * np.array([uxy + vxx, uyy + vxy])
    gr = (a * ga + b * gb) / r
    return -(np.outer(ga, ga) + np.outer(gb, gb) - np.outer(gr, gr)) / r


def angle_between(v1, v2):
    """Unsigned angle between two direction fields (sign-blind), degrees."""
    c = abs(float(np.clip(v1 @ v2, -1.0, 1.0)))
    return float(np.degrees(np.arccos(c)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rel = pentagon_rel()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("OECS estimator diagnosis (steady double gyre, A = 0.1, rho = 0.075)")
    emit("=" * 72)

    # -- 1 + 2: shear residual and H_s1 eigenvalues -------------------------
    emit("\n[1,2] Shear strain residual and fitted H_s1 eigenvalues:")
    for (x, y) in TEST_POINTS:
        theta_u, theta_v = _fit_vector_quadratic(rel, *sample_uv(x, y, rel))
        b = 0.5 * (theta_u[2] + theta_v[1])
        lam = np.linalg.eigvalsh(fitted_H_s1(theta_u, theta_v))
        emit(f"  ({x:+.2f},{y:+.2f}): b = {b:+.2e}   "
             f"H_s1 eig = [{lam[0]:+.4f}, {lam[1]:+.4f}]  (analytic "
             f"transverse curvature is +6.9 to +9.7 on the trench)")

    # -- 3: transverse restoring slope --------------------------------------
    dx = 0.03
    theta_u, theta_v = _fit_vector_quadratic(rel, *sample_uv(dx, 0.25, rel))
    _, grad_s1, e1, _, _ = _strain_quantities(theta_u, theta_v)
    slope = float(grad_s1 @ (e1 * np.sign(e1[0]))) / dx
    curv_true = np.pi**4 * A * abs(np.cos(np.pi * 0.75))
    emit(f"\n[3] Transverse restoring slope at (0.03, 0.25): fitted "
         f"{slope:.2f} per unit offset vs analytic curvature {curv_true:.2f} "
         f"(right sign, same order; Newton normalization unavailable).")

    # -- 4: frame invariance -------------------------------------------------
    omega_rot, t = 0.3, 2.0
    th = omega_rot * t
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def rotated_field(x, y):
        xi = Q.T @ np.array([x, y])
        uv = np.array(double_gyre_static(xi[0], xi[1]))
        return tuple(Q @ uv + omega_rot * np.array([-y, x]))

    p = np.array([0.03, 0.25])
    tu0, tv0 = _fit_vector_quadratic(rel, *sample_uv(p[0], p[1], rel))
    s1_0, _, _, e2_0, _ = _strain_quantities(tu0, tv0)
    pp = Q @ p
    tu1, tv1 = _fit_vector_quadratic(
        rel, *sample_uv(pp[0], pp[1], rel, field=rotated_field))
    s1_1, _, _, e2_1, _ = _strain_quantities(tu1, tv1)
    D0, D1 = _det_value(tu0, tv0), _det_value(tu1, tv1)
    emit(f"\n[4] Rotating observer (Omega = {omega_rot}, t = {t} s), same "
         f"material point:")
    emit(f"    s1: {s1_0:+.5f} -> {s1_1:+.5f}  (shift {abs(s1_1-s1_0):.1e})")
    emit(f"    e2 pulled back: angle error "
         f"{angle_between(Q.T @ e2_1, e2_0):.4f} deg")
    emit(f"    det(J): {D0:+.5f} -> {D1:+.5f}  "
         f"(shift {abs(D1-D0):.3f}, {100*abs(D1-D0)/abs(D0):.0f}%)")

    # -- 5: noise ladder ------------------------------------------------------
    emit(f"\n[5] Noise ladder at (-0.35, -0.20), {N_DRAWS} draws/level "
         f"(median errors):")
    emit("    sigma_uv | s1 rel err | e2 angle (deg) | grad_s1 dir (deg) "
         "| H_D eigvec angle (deg)")
    rng = np.random.default_rng(SEED)
    cx, cy = -0.35, -0.20
    u0, v0 = sample_uv(cx, cy, rel)
    tu_c, tv_c = _fit_vector_quadratic(rel, u0, v0)
    s1_c, g_c, _, e2_c, _ = _strain_quantities(tu_c, tv_c)
    H_c = _det_hessian(tu_c, tv_c)
    _, Vc = np.linalg.eigh(H_c)
    w1_c = Vc[:, 0]
    g_hat_c = g_c / np.linalg.norm(g_c)

    rows = []
    for sigma in SIGMA_LEVELS:
        s1_err, e2_ang, g_ang, hd_ang = [], [], [], []
        for _ in range(N_DRAWS):
            un = u0 + rng.normal(0.0, sigma, 6)
            vn = v0 + rng.normal(0.0, sigma, 6)
            tu, tv = _fit_vector_quadratic(rel, un, vn)
            s1, g, _, e2, _ = _strain_quantities(tu, tv)
            s1_err.append(abs(s1 - s1_c) / abs(s1_c))
            e2_ang.append(angle_between(e2, e2_c))
            gn = np.linalg.norm(g)
            if gn > 1e-12:
                g_ang.append(angle_between(g / gn, g_hat_c))
            _, V = np.linalg.eigh(_det_hessian(tu, tv))
            hd_ang.append(angle_between(V[:, 0], w1_c))
        row = {'sigma_uv': sigma,
               's1_rel_err_med': round(float(np.median(s1_err)), 4),
               'e2_angle_deg_med': round(float(np.median(e2_ang)), 3),
               'grad_s1_angle_deg_med': round(float(np.median(g_ang)), 2),
               'hd_eigvec_angle_deg_med': round(float(np.median(hd_ang)), 2)}
        rows.append(row)
        emit(f"    {sigma:8.3f} | {row['s1_rel_err_med']:10.4f} | "
             f"{row['e2_angle_deg_med']:14.3f} | "
             f"{row['grad_s1_angle_deg_med']:17.2f} | "
             f"{row['hd_eigvec_angle_deg_med']:22.2f}")

    csv_path = os.path.join(OUT_DIR, "oecs_estimator_check_noise.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    txt_path = os.path.join(OUT_DIR, "oecs_estimator_check.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nWrote {csv_path}")
    emit(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
