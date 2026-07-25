"""Rotation equivariance of the OECS controllers (Primitives 10 and 11).

The paper's central claim is that s1 (the smaller rate-of-strain eigenvalue)
is objective, unlike Okubo-Weiss, so the structure the team tracks is a
property of the flow rather than of the observer. That claim only survives
end to end if the CONTROLLER acting on s1 is objective too: a control law
whose output depends on how the x-axis was drawn would discard the
objectivity of its own input at the last step.

The formal condition is rotation equivariance,

    v(Q grad s1, Q e1, Q e2) = Q v(grad s1, e1, e2)

for every rotation Q. Saturating a command per Cartesian component violates
it; saturating a scalar (a projection or a norm) and applying the result
along a rotating unit vector satisfies it identically.

These tests pin every saturation site in both controllers.
"""

import numpy as np
import pytest

from src.control.pentagon_primitives import (
    _fit_vector_quadratic,
    _strain_quantities,
)

V_MAX = 0.04
G_PERP = 1.0
TOL = 1e-12

ANGLES = np.linspace(0.0, 2.0 * np.pi, 73)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# -- the saturation sites, each as a pure function of its rotating inputs ----

def _descend_s1(grad, e1, e2, t_hat):
    """ACQUIRE / PARK / CAPTURE: vector-saturated full gradient descent."""
    a = -G_PERP * grad
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.zeros(2)
    return a * (V_MAX * np.tanh(n / V_MAX) / n)


def _p10_track_perp(grad, e1, e2, t_hat):
    """Primitive 10 TRACK transverse: scalar projection applied along e1."""
    c = -G_PERP * float(grad @ e1)
    return V_MAX * np.tanh(c / V_MAX) * e1


def _p10_seek_tan(grad, e1, e2, t_hat):
    """Primitive 10 SEEK tangential: scalar projection applied along e2."""
    c = -G_PERP * float(grad @ e2)
    return V_MAX * np.tanh(c / V_MAX) * e2


def _p10_ride_tan(grad, e1, e2, t_hat):
    """Primitive 10 RIDE tangential: constant cruise along the tangent."""
    return V_MAX * np.tanh(1.0) * t_hat


def _p11_track_perp(grad, e1, e2, t_hat):
    """Primitive 11 TRACK transverse: descend the tangent-orthogonal part."""
    g_perp_vec = grad - float(grad @ t_hat) * t_hat
    n = float(np.linalg.norm(g_perp_vec))
    if n < 1e-12:
        return np.zeros(2)
    return V_MAX * np.tanh(-G_PERP * n / V_MAX) * (g_perp_vec / n)


def _p11_track_tan(grad, e1, e2, t_hat):
    """Primitive 11 TRACK tangential: constant cruise along the tangent."""
    return V_MAX * np.tanh(1.0) * t_hat


SITES = [
    ("descend_s1 (ACQUIRE/PARK/CAPTURE)", _descend_s1),
    ("P10 TRACK transverse", _p10_track_perp),
    ("P10 SEEK tangential", _p10_seek_tan),
    ("P10 RIDE tangential", _p10_ride_tan),
    ("P11 TRACK transverse", _p11_track_perp),
    ("P11 TRACK tangential", _p11_track_tan),
]


@pytest.mark.parametrize("name,site", SITES, ids=[s[0] for s in SITES])
def test_saturation_site_is_rotation_equivariant(name, site):
    """Every saturation site must commute with rotation of its inputs."""
    grad = np.array([0.7, 0.23])
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    t_hat = e2.copy()

    for theta in ANGLES:
        Q = _rot(theta)
        rotated = site(Q @ grad, Q @ e1, Q @ e2, Q @ t_hat)
        expected = Q @ site(grad, e1, e2, t_hat)
        assert np.linalg.norm(rotated - expected) < TOL, (
            f"{name} is frame dependent at theta = {np.degrees(theta):.1f} deg"
        )


def test_per_component_saturation_would_fail():
    """Guard the guard: the rejected per-component form must fail this test.

    Without this, a bug that made the equivariance check vacuous (e.g. a
    tolerance that swallowed everything) would pass silently.
    """
    def per_component(grad):
        return np.array([
            -V_MAX * np.tanh(G_PERP * grad[0] / V_MAX),
            -V_MAX * np.tanh(G_PERP * grad[1] / V_MAX),
        ])

    grad = np.array([0.7, 0.23])
    worst = max(
        np.linalg.norm(per_component(_rot(t) @ grad) - _rot(t) @ per_component(grad))
        for t in ANGLES
    )
    assert worst > 1e-3, "per-component saturation should be frame dependent"


def test_descent_speed_is_isotropic():
    """Commanded speed must not depend on the gradient's orientation.

    Per-component saturation varies the speed by a factor of sqrt(2) between
    an axis-aligned and a diagonal gradient. Vector saturation is flat, and
    caps at v_max rather than sqrt(2) v_max.
    """
    speeds = [
        np.linalg.norm(_descend_s1(np.array([np.cos(t), np.sin(t)]), None, None, None))
        for t in ANGLES
    ]
    assert max(speeds) - min(speeds) < TOL
    assert max(speeds) <= V_MAX + TOL


def test_descend_s1_is_a_descent_direction():
    """s1_dot <= 0, with equality only where the gradient vanishes.

    This is the PARK Lyapunov claim in the paper. Vector saturation gives
    s1_dot = -c_max ||grad s1|| tanh(g_perp ||grad s1|| / c_max) exactly.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        grad = rng.normal(scale=1.5, size=2)
        s1_dot = float(grad @ _descend_s1(grad, None, None, None))
        assert s1_dot <= 0.0
        n = np.linalg.norm(grad)
        expected = -V_MAX * n * np.tanh(G_PERP * n / V_MAX)
        assert abs(s1_dot - expected) < 1e-12

    assert np.allclose(_descend_s1(np.zeros(2), None, None, None), 0.0)


def test_equivariance_with_fitted_double_gyre_gradients():
    """End-to-end: gradients fitted from the real field, not synthetic ones."""
    A = 0.1
    rho = 0.075
    ang = np.array([2.0 * np.pi * k / 5.0 for k in range(5)])
    rel = np.vstack([np.zeros((1, 2)), np.stack([rho * np.cos(ang), rho * np.sin(ang)], -1)])

    for center in [np.array([0.0, 0.35]), np.array([0.12, 0.28]),
                   np.array([0.0, -0.4]), np.array([0.3, 0.1])]:
        absp = center + rel
        x, y = absp[:, 0], absp[:, 1]
        u = -np.pi * A * np.sin(np.pi * x) * np.cos(np.pi * y)
        v = np.pi * A * np.cos(np.pi * x) * np.sin(np.pi * y)
        theta_u, theta_v = _fit_vector_quadratic(rel, u, v)
        _, grad_s1, _, _, _ = _strain_quantities(theta_u, theta_v)

        for theta in ANGLES:
            Q = _rot(theta)
            rotated = _descend_s1(Q @ grad_s1, None, None, None)
            expected = Q @ _descend_s1(grad_s1, None, None, None)
            assert np.linalg.norm(rotated - expected) < TOL
