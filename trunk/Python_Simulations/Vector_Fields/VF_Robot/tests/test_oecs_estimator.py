"""
Test the rate-of-strain (OECS) estimator quantities against the analytic
double gyre.  Companion to Primitive 10 (oecs_trap_step) in
pentagon_primitives.py.

Checks:
1. s1 matches the closed form s1 = -pi^2 A |cos(pi x_f) cos(pi y_f)|.
2. The shear strain b = (u_y + v_x)/2 vanishes on the double gyre.
3. The eigenframe is axis-aligned and its compression/stretch identity
   SWAPS between the upper and lower halves of the separatrix.
4. grad_s1 . e1 is a restoring signal toward the s1 trench with a slope
   of the analytic order.
5. The fitted H_s1 (never used by the controller, computed here only to
   document the structural limit) is negative semidefinite.
6. Objectivity: refitting in a rotated frame leaves s1 invariant and
   rotates e2 with the frame.

Run from the VF_Robot root:
  MPLBACKEND=Agg ./venv/bin/python3 -m pytest tests/test_oecs_estimator.py -q
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.pentagon_primitives import (
    _fit_vector_quadratic, _strain_quantities
)
from src.fields.environments.Double_Gyre import double_gyre_static

A = 0.1
RHO = 0.075


def _pentagon_rel(phase=0.0):
    """Pentagon-plus-center relative positions, optional ring phase."""
    ang = np.arange(5) * 2.0 * np.pi / 5.0 + phase
    ring = np.column_stack([RHO * np.cos(ang), RHO * np.sin(ang)])
    return np.vstack([np.zeros((1, 2)), ring])


def _fit_at(cx, cy, field=double_gyre_static, phase=0.0):
    rel = _pentagon_rel(phase)
    u = np.array([field(cx + r[0], cy + r[1])[0] for r in rel])
    v = np.array([field(cx + r[0], cy + r[1])[1] for r in rel])
    return _fit_vector_quadratic(rel, u, v)


def _s1_analytic(x, y):
    xf, yf = x + 1.0, y + 0.5
    return -np.pi**2 * A * abs(np.cos(np.pi * xf) * np.cos(np.pi * yf))


def test_s1_matches_analytic():
    for (x, y) in [(0.03, 0.25), (-0.2, -0.3), (0.4, 0.1), (0.0, 0.35)]:
        theta_u, theta_v = _fit_at(x, y)
        s1, _, _, _, _ = _strain_quantities(theta_u, theta_v)
        assert abs(s1 - _s1_analytic(x, y)) < 0.02, (x, y, s1)


def test_shear_strain_vanishes_on_double_gyre():
    for (x, y) in [(0.03, 0.25), (-0.35, -0.2), (0.25, 0.42)]:
        theta_u, theta_v = _fit_at(x, y)
        b = 0.5 * (theta_u[2] + theta_v[1])
        assert abs(b) < 1e-3, (x, y, b)


def test_eigenframe_identity_swap():
    # Upper separatrix branch: compression along x (e1 ~ x-axis).
    theta_u, theta_v = _fit_at(0.03, 0.25)
    _, _, e1, e2, _ = _strain_quantities(theta_u, theta_v)
    assert abs(abs(e1[0]) - 1.0) < 1e-2
    assert abs(abs(e2[1]) - 1.0) < 1e-2
    # Lower branch: identity swaps, compression along y.
    theta_u, theta_v = _fit_at(0.03, -0.25)
    _, _, e1, e2, _ = _strain_quantities(theta_u, theta_v)
    assert abs(abs(e1[1]) - 1.0) < 1e-2
    assert abs(abs(e2[0]) - 1.0) < 1e-2


def test_transverse_gradient_is_restoring():
    # At x = +0.03 off the upper separatrix trench, descending s1 along the
    # compression axis must push back toward x = 0, with a slope of the
    # analytic transverse curvature's order (pi^4 A |c_y| = 6.89 there).
    dx = 0.03
    theta_u, theta_v = _fit_at(dx, 0.25)
    _, grad_s1, e1, _, _ = _strain_quantities(theta_u, theta_v)
    d_s1_dx = float(grad_s1 @ (e1 * np.sign(e1[0])))  # component along +x
    assert d_s1_dx > 0.0            # descent moves toward -x, i.e. the trench
    slope = d_s1_dx / dx
    assert 1.0 < slope < 10.0, slope


def test_fitted_H_s1_is_negative_semidefinite():
    # Documented structural limit: under the quadratic model, s1 is the
    # negative norm of an affine map plus an affine term, so its Hessian is
    # NSD for every field.  The controller does not use it; this test pins
    # the fact the paper remark states.
    for (x, y) in [(0.03, 0.25), (-0.35, -0.2), (0.25, 0.42)]:
        theta_u, theta_v = _fit_at(x, y)
        _, ux, uy, uxx, uxy, uyy = theta_u
        _, vx, vy, vxx, vxy, vyy = theta_v
        a, b = 0.5 * (ux - vy), 0.5 * (uy + vx)
        r = max(np.hypot(a, b), 1e-12)
        ga = 0.5 * np.array([uxx - vxy, uxy - vyy])
        gb = 0.5 * np.array([uxy + vxx, uyy + vxy])
        gr = (a * ga + b * gb) / r
        H_s1 = -(np.outer(ga, ga) + np.outer(gb, gb) - np.outer(gr, gr)) / r
        lam = np.linalg.eigvalsh(H_s1)
        assert lam[1] < 1e-9, (x, y, lam)


def test_objectivity_of_s1_and_e2():
    # Refit the same material point through a rotated observer frame:
    # x' = Q x, v' = Q v(Q^T x') + Omega * perp(x').  s1 must be invariant
    # and e2 must pull back to the inertial e2.
    omega_rot, t = 0.3, 2.0
    th = omega_rot * t
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def rotated_field(x, y):
        xi = Q.T @ np.array([x, y])
        uv = np.array(double_gyre_static(xi[0], xi[1]))
        return tuple(Q @ uv + omega_rot * np.array([-y, x]))

    p = np.array([0.03, 0.25])
    theta_u0, theta_v0 = _fit_at(p[0], p[1])
    s1_0, _, _, e2_0, _ = _strain_quantities(theta_u0, theta_v0)

    pp = Q @ p
    theta_u1, theta_v1 = _fit_at(pp[0], pp[1], field=rotated_field)
    s1_1, _, _, e2_1, _ = _strain_quantities(theta_u1, theta_v1)

    assert abs(s1_1 - s1_0) < 1e-3
    e2_back = Q.T @ e2_1
    assert abs(abs(float(e2_back @ e2_0)) - 1.0) < 1e-3

    # And the non-objective quantity moves: det(J) shifts by Omega*omega
    # + Omega^2, which is order 0.1 here.
    D0 = theta_u0[1] * theta_v0[2] - theta_u0[2] * theta_v0[1]
    D1 = theta_u1[1] * theta_v1[2] - theta_u1[2] * theta_v1[1]
    assert abs(D1 - D0) > 0.05


if __name__ == "__main__":
    test_s1_matches_analytic()
    test_shear_strain_vanishes_on_double_gyre()
    test_eigenframe_identity_swap()
    test_transverse_gradient_is_restoring()
    test_fitted_H_s1_is_negative_semidefinite()
    test_objectivity_of_s1_and_e2()
    print("All OECS estimator tests passed.")
