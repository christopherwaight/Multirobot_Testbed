"""
Rotating-frame re-expression of a base vector field.

Under the time-dependent Euclidean frame change x' = Q(t) x, with
Q(t) = R(omega_rot * t) a rotation by angle omega_rot * t, a velocity
field v(x, t) transforms to

    v'(x', t) = Q(t) v(Q(t)^T x', t) + omega_rot * perp(x'),

where perp(x, y) = (-y, x) is the solid-body swirl contributed by the
frame's own rotation (Qdot Q^T x').  The primed field describes the SAME
physical flow seen by an observer on a carousel: material behavior is
identical, but any non-objective diagnostic changes.  For trace-free J,

    det(J') = det(J) + omega_rot * vorticity + omega_rot^2,

so the Okubo-Weiss / det(J) landscape an observer computes depends on the
observer, while the rate-of-strain tensor transforms as S' = Q S Q^T:
eigenvalues invariant, eigenvectors co-rotating.  This module exists to
demonstrate exactly that split on the double gyre (Primitive 10 vs
Logic C; see experiments/oecs_objectivity_demo.py).

Pull-back convention: a trajectory x'(t) recorded in the rotating frame
corresponds to the inertial-frame trajectory x(t) = Q(t)^T x'(t).
"""
import numpy as np


def make_rotating_frame(base_field_fn, omega_rot):
    """
    Wrap a base field function in a rotating observer frame.

    Args:
        base_field_fn: field with signature (x, y[, t, ...]) -> (u, v).
                       Called here as base_field_fn(x, y, t), so steady
                       fields simply ignore t.
        omega_rot:     frame rotation rate (rad/s), positive
                       counterclockwise.

    Returns:
        Field function (x, y, t=0.0) -> (u, v) in the rotating frame.
        AnalyticalField passes its internal clock to the t argument.
    """
    def rotating_field(x, y, t=0.0):
        th = omega_rot * t
        c, s = np.cos(th), np.sin(th)
        # Inertial coordinates of the queried point: x = Q^T x'.
        xi = c * x + s * y
        yi = -s * x + c * y
        u0, v0 = base_field_fn(xi, yi, t)
        # v' = Q v + omega_rot * perp(x').
        u = c * u0 - s * v0 - omega_rot * y
        v = s * u0 + c * v0 + omega_rot * x
        return float(u), float(v)

    rotating_field.omega_rot = omega_rot
    rotating_field.base_field_fn = base_field_fn
    return rotating_field


def make_rotated_field(base_field_fn, theta):
    """
    Wrap a base field function in a FIXED spatial rotation by angle theta.

    Unlike make_rotating_frame above, this is a static Euclidean change of
    variables, not a time-dependent observer: Q = R(theta) is constant, so
    Qdot = 0 and there is no swirl term.  v_theta(p) = Q v(Q^T p).  This
    tilts the strain eigenframe (e1, e2) off whatever axes they sit on in
    the base field, while every analytic quantity of the base field (s1,
    the separatrix, singular points, ...) carries over exactly under the
    same rotation.  It exists to test control-law code that was only ever
    exercised on fields whose eigenframe happens to sit on the coordinate
    axes (e.g. the double gyre, where the shear strain vanishes
    identically) -- a bug that cancels when e1/e2 are axis-aligned can
    still be present and will show up once the frame is tilted.

    Args:
        base_field_fn: field with signature (x, y[, t, ...]) -> (u, v).
                       Called here as base_field_fn(x, y, t), so steady
                       fields simply ignore t.
        theta:         fixed rotation angle (rad), positive
                       counterclockwise.

    Returns:
        Field function (x, y, t=0.0) -> (u, v), the base field rotated by
        theta.  Ground truth for any base-field closed form (position of
        the separatrix, s1, singular points, ...) is obtained by applying
        the same R(theta) to that closed form; no time-dependent pull-back
        is needed since theta is constant.
    """
    c, s = np.cos(theta), np.sin(theta)

    def rotated_field(x, y, t=0.0):
        # Inertial (base-field) coordinates of the queried point: Q^T p.
        xi = c * x + s * y
        yi = -s * x + c * y
        u0, v0 = base_field_fn(xi, yi, t)
        # v_theta = Q v0 (no swirl term: Qdot = 0 for a fixed rotation).
        u = c * u0 - s * v0
        v = s * u0 + c * v0
        return float(u), float(v)

    rotated_field.theta = theta
    rotated_field.base_field_fn = base_field_fn
    return rotated_field


def pull_back_trajectory(traj, times, omega_rot):
    """
    Map a rotating-frame trajectory back to inertial coordinates.

    Args:
        traj:      (N, 2) positions in the rotating frame
        times:     (N,) simulation times of each sample
        omega_rot: frame rotation rate used to generate the field

    Returns:
        (N, 2) inertial-frame positions x = Q(t)^T x'.
    """
    traj = np.asarray(traj, dtype=float)
    times = np.asarray(times, dtype=float)
    th = omega_rot * times
    c, s = np.cos(th), np.sin(th)
    out = np.empty_like(traj)
    out[:, 0] = c * traj[:, 0] + s * traj[:, 1]
    out[:, 1] = -s * traj[:, 0] + c * traj[:, 1]
    return out
