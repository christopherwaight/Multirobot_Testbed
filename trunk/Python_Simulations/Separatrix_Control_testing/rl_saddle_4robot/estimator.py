"""
The 4-robot C(4,3) plane-fit gradient and Hessian estimator.

Reimplemented from saddle_point_4_robot.ipynb, cell 1, class
SaddlePointNavigator, methods estimate_gradient_from_plane and
estimate_hessian.  Factored out here because three consumers need it: the
baseline controller, the 'raw+est' observation ablation, and the mechanism
figure.

    Take all four 3-robot subsets of the square.  Fit a plane to each and read
    off its gradient.  That gives four gradient samples at four sub-centroids.
    Least-squares a linear model through those four samples; its coefficient
    matrix is the Hessian estimate.

What this estimator recovers, measured rather than assumed.  Run this file
directly to reproduce every claim below numerically.

  * The GRADIENT is exact for a quadratic field, at every formation angle.

  * The HESSIAN estimate is, exactly,

        H_est = m * Ref(theta_f),      m = M * cos(2 (theta_f - theta_field))

    where theta_f is the FORMATION angle, theta_field is the field's principal
    axis, M is a field curvature scale, and Ref(t) is reflection about the line
    at angle t:

        Ref(t) = [[cos 2t,  sin 2t],
                  [sin 2t, -cos 2t]]

    Verified to machine precision.  Three consequences, and the third is the
    one that is easy to get backwards:

      - H_est is always traceless and always symmetric.  The map from the four
        readings to (H, offset) has rank 3: two degrees of freedom go to the
        gradient, leaving exactly ONE scalar for curvature.

      - That one scalar is m.  Its magnitude vanishes at 45 degrees of
        misalignment, the sensing blind spot, and its SIGN flips every 90
        degrees of formation rotation.

      - The eigenvectors of H_est are the FORMATION's own axes, not the field's.
        Rotating the formation rotates the reported eigenframe with it.  The
        estimate therefore carries no direct information about where the field's
        eigenframe points; that information reaches the controller only through
        the scalar m.

  * Consequently the saddle-seeking Newton step is exactly

        -H_est^-1 g  =  -(1/m) Ref(theta_f) g

    the negative gradient reflected about the formation's own axis, scaled by
    1/m.  The step DIRECTION is set by the gradient and the formation angle
    alone.  The field's curvature enters only as a magnitude and a sign.

This is why the formation has to rotate, and it is a stronger reason than a
blind spot alone.  Turning the formation is the controller's only means of
steering the step direction at all, and alignment with the field eigenframe is
what makes -Ref(theta_f) g coincide with the true Newton direction.  It also
predicts that a NON-rotating formation still converges whenever its fixed angle
happens to sit near a field eigen-axis, which is roughly half of random
initializations.  baselines.py measures exactly that.
"""
from itertools import combinations

import numpy as np

# The four 3-robot subsets, fixed once so the ordering is stable.
_TRIPLES = list(combinations(range(4), 3))


def plane_gradient(pts, vals):
    """Gradient of the plane through 3 points. pts is (3, 2), vals is (3,)."""
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(3)])
    return np.linalg.lstsq(A, vals, rcond=None)[0][:2]


def plane_fit_estimate(xy, z):
    """Estimate (H, gradient) at the formation centroid.

    Args:
        xy: (4, 2) robot positions in world coordinates
        z:  (4,)   scalar readings

    Returns:
        H: (2, 2) Hessian estimate, always symmetric and traceless
        g: (2,)   gradient estimate, exact for a quadratic field
    """
    xy = np.asarray(xy, float)
    z = np.asarray(z, float)

    grads = np.empty((4, 2))
    cents = np.empty((4, 2))
    for k, tri in enumerate(_TRIPLES):
        idx = list(tri)
        grads[k] = plane_gradient(xy[idx], z[idx])
        cents[k] = xy[idx].mean(axis=0)

    A = np.column_stack([cents[:, 0], cents[:, 1], np.ones(4)])
    cx = np.linalg.lstsq(A, grads[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, grads[:, 1], rcond=None)[0]
    H = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    return H, grads.mean(axis=0)


def newton_step(H, g, max_step=1.0):
    """Saddle-seeking Newton step -H^-1 g, magnitude-limited."""
    if abs(np.linalg.det(H)) > 1e-10:
        s = -np.linalg.solve(H, g)
    else:
        s = -np.linalg.pinv(H) @ g
    n = np.linalg.norm(s)
    if not np.isfinite(n):
        return np.zeros(2)
    return s / n * max_step if n > max_step else s


def formation_angle(xy):
    """Formation orientation, the heading of the robot-1 to robot-3 diagonal.

    Matches forward_kinematics_quad's th_c convention in
    src/control/quad_kinematics.py.
    """
    xy = np.asarray(xy, float)
    d = xy[2] - xy[0]
    return float(np.arctan2(d[1], d[0]))


def eigenframe_misalignment(theta, H_true):
    """Signed misalignment of the formation against the field eigenframe.

    The estimator's sensitivity goes as cos(2 * this), and the whole structure
    is pi/2-periodic in the formation angle, so the result is wrapped to
    [-pi/4, pi/4].  Zero means aligned and maximally sensitive; +/- pi/4 means
    sitting in the blind spot.
    """
    lam, V = np.linalg.eigh(np.asarray(H_true, float))
    principal = V[:, int(np.argmax(lam))]        # the positive-curvature axis
    a = theta - np.arctan2(principal[1], principal[0])
    return float((a + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def estimator_gain(theta, H_true):
    """cos(2 * misalignment): the factor the estimate is scaled by.

    1.0 when the formation is aligned with the field eigenframe, 0.0 in the
    45-degree blind spot.
    """
    return float(np.cos(2.0 * eigenframe_misalignment(theta, H_true)))


def square_positions(centroid, radius, theta):
    """Four corners of a square, in the same order the estimator expects."""
    a = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]) + theta
    return np.asarray(centroid, float)[None, :] + \
        radius * np.column_stack([np.cos(a), np.sin(a)])


# --------------------------------------------------------------------------
# Verification of every claim in the docstring
# --------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 74)
    print("1. GRADIENT IS EXACT, HESSIAN IS TRACELESS, BLIND SPOT AT 45 DEG")
    print("=" * 74)
    H_true = np.array([[2.0, 0.0], [0.0, -2.0]])
    c = np.array([0.7, -0.4])

    def phi(x, y):
        r = np.array([x, y])
        return 0.5 * r @ H_true @ r

    print(f"true H = diag(2, -2),  true grad at {tuple(c)} = "
          f"{H_true @ c}\n")
    print(f"{'theta':>7s} {'Hest':>34s} {'trace':>9s} {'|eig|':>8s} "
          f"{'grad err':>10s}")
    print("-" * 74)
    for td in [0, 15, 30, 45, 60, 75, 90]:
        xy = square_positions(c, 0.25, np.deg2rad(td))
        z = np.array([phi(p[0], p[1]) for p in xy])
        H, g = plane_fit_estimate(xy, z)
        lam = abs(np.linalg.eigvalsh((H + H.T) / 2))[1]
        print(f"{td:7d} [[{H[0,0]:+7.4f},{H[0,1]:+7.4f}],"
              f"[{H[1,0]:+7.4f},{H[1,1]:+7.4f}]] {np.trace(H):+9.2e} "
              f"{lam:8.4f} {np.linalg.norm(g - H_true @ c):10.2e}")
    print("\n  trace is zero to machine precision at every angle")
    print("  |eig| follows 6*|cos(2 theta)| and vanishes exactly at 45 deg")

    print()
    print("=" * 74)
    print("2. RANK OF THE MAP  readings -> (H, offset)")
    print("=" * 74)
    xy = square_positions(c, 0.25, np.deg2rad(17.0))
    z0 = np.array([phi(p[0], p[1]) for p in xy])

    def flat(z):
        H, _ = plane_fit_estimate(xy, z)
        A = np.column_stack([np.array([np.mean(xy[list(t)], axis=0)
                                       for t in _TRIPLES])[:, 0],
                             np.array([np.mean(xy[list(t)], axis=0)
                                       for t in _TRIPLES])[:, 1],
                             np.ones(4)])
        grads = np.array([plane_gradient(xy[list(t)], z[list(t)]) for t in _TRIPLES])
        cx = np.linalg.lstsq(A, grads[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, grads[:, 1], rcond=None)[0]
        return np.concatenate([H.ravel(), [cx[2], cy[2]]])

    J = np.zeros((6, 4))
    h = 1e-6
    for i in range(4):
        zp, zm = z0.copy(), z0.copy()
        zp[i] += h
        zm[i] -= h
        J[:, i] = (flat(zp) - flat(zm)) / (2 * h)
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  singular values: {sv}")
    print(f"  rank: {np.linalg.matrix_rank(J, tol=1e-8)} of a possible 4")
    print("  4 readings, minus 1 for the additive constant, leaves 3.")
    print("  Rows for H00 and H11 are exact negatives -> traceless.")
    print("  Rows for H01 and H10 are identical       -> symmetric.")

    print()
    print("=" * 74)
    print("3. THE EIGENFRAME IS THE FORMATION'S, NOT THE FIELD'S")
    print("=" * 74)
    th_field = np.deg2rad(40.0)
    Rf = np.array([[np.cos(th_field), -np.sin(th_field)],
                   [np.sin(th_field), np.cos(th_field)]])
    H2 = Rf @ np.diag([3.0, -1.0]) @ Rf.T

    def phi2(x, y):
        r = np.array([x, y])
        return 0.5 * r @ H2 @ r

    print("  field principal axis is fixed at 40 deg throughout\n")
    print(f"  {'formation':>10s} {'H_est axis':>11s} {'|m|':>8s} "
          f"{'6|cos2(tf-40)|':>15s}")
    print("  " + "-" * 50)
    for td in [0, 20, 40, 65, 85, 130]:
        xy = square_positions(np.array([0.3, 0.2]), 0.25, np.deg2rad(td))
        z = np.array([phi2(p[0], p[1]) for p in xy])
        H, _ = plane_fit_estimate(xy, z)
        lam, V = np.linalg.eigh((H + H.T) / 2)
        v = V[:, int(np.argmax(lam))]
        ax = np.rad2deg(np.arctan2(v[1], v[0])) % 180
        print(f"  {td:10.0f} {ax:11.2f} {np.max(np.abs(lam)):8.4f} "
              f"{6*abs(np.cos(2*np.deg2rad(td - 40))):15.4f}")
    print("\n  The axis column tracks the FORMATION angle, not the field's 40.")
    print("  Only the magnitude column carries field information, and it is")
    print("  the single scalar m = M cos(2 (theta_f - theta_field)).")

    print()
    print("=" * 74)
    print("4. NEWTON STEP IS EXACTLY  -(1/m) Ref(theta_f) g")
    print("=" * 74)
    print(f"  {'formation':>10s} {'-H^-1 g':>22s} {'-(1/m)Ref g':>22s} {'err':>10s}")
    print("  " + "-" * 68)
    worst = 0.0
    for td in [0, 20, 40, 65, 130]:
        t = np.deg2rad(td)
        xy = square_positions(np.array([0.3, 0.2]), 0.25, t)
        z = np.array([phi2(p[0], p[1]) for p in xy])
        H, g = plane_fit_estimate(xy, z)
        step = -np.linalg.solve(H, g)
        Ref = np.array([[np.cos(2 * t), np.sin(2 * t)],
                        [np.sin(2 * t), -np.cos(2 * t)]])
        m = np.sign(H[0, 0] * np.cos(2 * t) + H[0, 1] * np.sin(2 * t)) * \
            np.hypot(H[0, 0], H[0, 1])
        pred = -(1.0 / m) * Ref @ g
        err = float(np.linalg.norm(step - pred))
        worst = max(worst, err)
        print(f"  {td:10.0f} [{step[0]:+8.4f},{step[1]:+8.4f}] "
              f"[{pred[0]:+8.4f},{pred[1]:+8.4f}] {err:10.2e}")
    print(f"\n  worst residual {worst:.2e}: the identity is exact.")
    print("  Step DIRECTION is set by g and the formation angle alone.")
    print("  Field curvature enters only through the scalar m.")

    print()
    print("=" * 74)
    print("5. THE ROTATION RATE IS BOXED IN FROM BOTH SIDES")
    print("=" * 74)
    print("  Upper bound: tangential speed omega*R must stay under v_max=0.3.")
    print("  Lower bound: one filter step realizes only (1-alpha)=0.3 of the")
    print("  command, and anything under the 0.025 m/s stiction threshold is")
    print("  zeroed, so the velocity never accumulates.  A command below")
    print("  0.025/((1-alpha) R) produces NO rotation at all, not slow rotation.")
    print()
    print(f"  {'R (m)':>8s} {'omega_min':>11s} {'omega_max':>11s} {'ratio':>8s}")
    print("  " + "-" * 42)
    for R in (0.08, 0.10, 0.15, 0.25, 0.40):
        lo = 0.025 / (0.3 * R)
        hi = 0.3 / R
        print(f"  {R:8.2f} {lo:11.3f} {hi:11.3f} {hi/lo:8.2f}")
    print("\n  Both bounds scale as 1/R, so the usable band is only 3.6x wide")
    print("  at every size, and shrinking the formation shifts it upward.")

    print()
    print("=" * 74)
    print("6. A FIFTH ROBOT AT THE CENTROID RECOVERS THE TRACE, EXACTLY")
    print("=" * 74)
    print("  For four ring points at radius R spaced 90 degrees apart,")
    print("  sum_i r_i r_i^T = 2 R^2 I, so for a local quadratic")
    print()
    print("      mean(z_ring) - z_centre = R^2 tr(H) / 4")
    print()
    print("  which inverts to tr(H) = 4 (mean(z_ring) - z_centre) / R^2.")
    print("  Adding that back to the deviatoric part the 4-robot fit already")
    print("  recovers gives the FULL Hessian, so det(H) carries a sign and")
    print("  saddle-vs-extremum becomes decidable.  With four robots alone")
    print("  det(H_est) = -(a^2 + b^2) < 0 always: every critical point looks")
    print("  like a saddle, which is why the online handover rule in")
    print("  baselines.handover cannot work.")
    print()
    print(f"  {'formation':>10s}{'tr true':>10s}{'tr est':>9s}"
          f"{'det true':>10s}{'verdict':>22s}")
    print("  " + "-" * 62)
    cases = [(np.array([[3., 0.], [0., -1.]]), "saddle"),
             (np.array([[2., 1.5], [1.5, -0.5]]), "saddle"),
             (np.array([[1., 0.], [0., 2.]]), "EXTREMUM")]
    for td in (10.0, 35.0, 80.0):
        for Ht, truth in cases:
            def phi_c(x, y, H=Ht):
                r = np.array([x, y])
                return 0.5 * r @ H @ r
            c = np.array([0.4, -0.3])
            Rr = 0.15
            xy = square_positions(c, Rr, np.deg2rad(td))
            z = np.array([phi_c(p[0], p[1]) for p in xy])
            tr_est = 4.0 * (z.mean() - phi_c(c[0], c[1])) / Rr ** 2
            Hd, _ = plane_fit_estimate(xy, z)
            Hfull = Hd + 0.5 * tr_est * np.eye(2)
            got = "saddle" if np.linalg.det(Hfull) < 0 else "EXTREMUM"
            ok = "ok" if got == truth else "WRONG"
            print(f"  {td:10.0f}{np.trace(Ht):10.4f}{tr_est:9.4f}"
                  f"{np.linalg.det(Ht):10.4f}{got + ' ' + ok:>22s}")
    print()
    print("  Trace is exact at every formation angle, and the saddle/extremum")
    print("  verdict is correct in all nine cases.  Note the recovered det has")
    print("  the right SIGN but not the right magnitude, because the deviatoric")
    print("  part still carries the cos(2 theta) attenuation from section 3.")
    print("  Sign is what the controller needs.")
    print("  Near convergence the servo wants a small omega, which falls in")
    print("  the dead zone.  That is the endgame failure the notebook's law")
    print("  cannot express, and the reason the size knob matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
