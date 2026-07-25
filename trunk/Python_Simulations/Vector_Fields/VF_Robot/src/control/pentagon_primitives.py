"""
Control primitives for the 6-robot pentagon cluster.

Scalar-field primitives (1-6) are ported from the notebook:
  trunk/Python_Simulations/Separatrix_Control_testing/saddle_point_6_robot2.ipynb
Each takes a PentagonCluster and returns (vx_c, vy_c).  They call
cluster.sample_field_at_robots() to get 6 scalar readings, then fit a full
second-order model recovering the gradient and Hessian at the centroid.

Primitive 7 is a vector-field primitive ported from:
  trunk/Python_Simulations/separatrix_interactive_v6r.ipynb
It queries cluster.field.get_value() directly to obtain (u, v) readings and
fits two independent quadratics (one per component) to estimate the Jacobian
and its det(J) landscape.

Primitive summary:
  1. newton_step              -- scalar: plain Newton, straight-line approach
  2. sign_decomposed_step     -- scalar: per-component sign control (chatters on trench)
  3. anisotropic_newton_step  -- scalar: per-eigendirection gains, no chatter
  4. tanh_sliding_step        -- scalar: boundary-layer SMC, smooth reach
  5. adaptive_eigen_step      -- scalar: single v_max, automatic curvature normalization,
                                 cubic convergence near saddle (recommended)
  6. adaptive_eigen_step_abs  -- scalar: |lambda_i| in denominator: snap perp,
                                 descend along trench (Dauphin-style)
  7. separatrix_logic_c_step  -- vector: hybrid trench navigator with FLOW-band override
                                 on the separatrix (Logic C from separatrix_interactive_v6r)
  8. logic_g_zero_flow_pentagon -- vector: heuristic Logic G port; tracks det(J)=0
                                 Okubo-Weiss diamond using gradient/flow blend.
                                 Fixed orientation (no omega).
  9. logic_g_newton_contour_pentagon -- vector: Newton-step Logic G; drives
                                 det(J) -> 0 via Newton step along gradient +
                                 perp-gradient tangential drift. Fixed orientation.
 10. oecs_trap_step            -- vector: objective (frame-invariant) tracker of
                                 attracting OECS/TRAP curves built from the
                                 rate-of-strain tensor S = (J + J^T)/2. Rides
                                 trenches of the smaller strain eigenvalue s1
                                 using S's own eigenframe; one knob (s_capture)
                                 switches ride-along vs core-seek-and-park.
 11. oecs_separatrix_step      -- vector: objective (frame-invariant) traverser of
                                 the FULL separatrix network, the S-based analogue
                                 of separatrix_logic_c_step (Primitive 7). Rides the
                                 transverse trench of s1 with the snap taken
                                 orthogonal to the direction of travel (not a fixed
                                 eigenvector), so one law holds both the attracting
                                 and repelling halves; a flow-assisted CROSS mode
                                 carries the team through isotropic points (S = 0)
                                 where the eigenframe itself is undefined.
"""
import numpy as np


# -----------------------------------------------------------------------
# Quadratic estimator
# -----------------------------------------------------------------------

def _quadratic_basis(rx, ry):
    """Basis vector for a single relative position (rx, ry).

    Paper cross-reference (Draft_5c.tex, Eq. 7): the paper writes the
    fitted coefficients as a_1..a_6 for u and b_1..b_6 for v, in place of
    the derivative-style names used here:

        u0, ux, uy, uxx, uxy, uyy  ->  a_1, a_2, a_3, a_5, a_4, a_6
        v0, vx, vy, vxx, vxy, vyy  ->  b_1, b_2, b_3, b_5, b_4, b_6

    Note the slot order differs. This basis is [1, x, y, x^2/2, xy, y^2/2],
    so the xy term sits in slot 5; the paper's basis is
    [1, x, y, xy, x^2/2, y^2/2], with xy in slot 4. Both are internally
    consistent and give identical fits, but a_4 is the xy coefficient
    while theta[4] here is also xy, whereas theta[3] is xx and maps to
    a_5. Mind the swap when comparing code against the paper's Eq. 8-9.

    The one-half factors on x^2 and y^2 (absent on xy) are load-bearing in
    both conventions: they make the recovered coefficients equal the Taylor
    derivatives directly, and Lemma 1's noise gains (8/sqrt(10) for the
    pure second-order terms, 4/sqrt(10) for the mixed one) assume them.
    """
    return np.array([1.0, rx, ry, rx**2 / 2.0, rx * ry, ry**2 / 2.0])


def _fit_quadratic(relative_positions, readings):
    """
    Fit a local quadratic scalar field model to 6 robot readings.

    Well-posedness: the 6x6 Phi matrix is singular exactly when the six
    sample points lie on a common conic (circle, ellipse, line pair, ...).
    Six robots on one ring (e.g. a regular hexagon) is therefore always
    singular; the pentagon-plus-center formation is not, because the only
    conic through the five ring robots is their circumcircle, which
    misses the center robot. Conditioning degrades smoothly as the
    formation deforms toward a conic, hence the det check below.

    Args:
        relative_positions: (6, 2) array of (rx, ry) relative to centroid
        readings: (6,) array of scalar field values

    Returns:
        theta: (6,) coefficient vector [phi_c, dphi/dx, dphi/dy,
                                        d2phi/dx2, d2phi/dxdy, d2phi/dy2]
    """
    Phi = np.array([_quadratic_basis(relative_positions[j, 0],
                                     relative_positions[j, 1])
                    for j in range(6)])
    s = np.array(readings, dtype=float)

    if abs(np.linalg.det(Phi)) > 1e-10:
        return np.linalg.solve(Phi, s)
    return np.linalg.lstsq(Phi, s, rcond=None)[0]


def _extract_grad_hessian(theta):
    """
    Extract gradient and Hessian from quadratic fit coefficients.

    Returns:
        grad: (2,) gradient vector [dphi/dx, dphi/dy]
        H:    (2, 2) Hessian matrix
    """
    grad = theta[1:3].copy()
    H = np.array([[theta[3], theta[4]],
                  [theta[4], theta[5]]])
    return grad, H


def _get_relative_positions(cluster):
    """Return (6, 2) array of robot positions relative to centroid."""
    centroid = cluster.get_centroid()
    coords = cluster.get_robot_positions()
    rel = np.array([[coords[2*i] - centroid[0], coords[2*i+1] - centroid[1]]
                    for i in range(6)])
    return rel


def _estimate(cluster):
    """Fit quadratic model to current cluster state. Returns (grad, H)."""
    readings = cluster.sample_field_at_robots()
    rel_pos = _get_relative_positions(cluster)
    theta = _fit_quadratic(rel_pos, readings)
    return _extract_grad_hessian(theta)


# -----------------------------------------------------------------------
# Hessian eigenbasis: saddle frame
# -----------------------------------------------------------------------

def _trench_frame(H):
    """
    Decompose H into saddle eigenbasis.

    Returns:
        ((lambda_perp, v_perp), (lambda_along, v_along))
          where lambda_perp > 0 (convex, across trench)
          and   lambda_along < 0 (concave, along trench)
        or None if H does not have mixed-sign eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    lam0, lam1 = eigenvalues
    v0 = eigenvectors[:, 0]
    v1 = eigenvectors[:, 1]

    if lam0 * lam1 >= 0:
        return None  # not a saddle

    if lam0 > 0:
        return (lam0, v0), (lam1, v1)
    return (lam1, v1), (lam0, v0)


# -----------------------------------------------------------------------
# Primitive 1: Newton step
# -----------------------------------------------------------------------

def newton_step(cluster, step_size=0.02, max_step=1.0):
    """
    Plain Newton step: delta = -H^{-1} * grad, scaled by step_size.

    Converges in a straight line but does not preferentially collapse
    onto the trench.
    """
    grad, H = _estimate(cluster)
    try:
        delta = np.linalg.solve(H, -grad)
    except np.linalg.LinAlgError:
        delta = np.zeros(2)

    norm = np.linalg.norm(delta)
    if norm > max_step:
        delta = delta / norm * max_step
    delta = step_size * delta

    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 2: Sign-decomposed step (first-order sliding mode)
# -----------------------------------------------------------------------

def sign_decomposed_step(cluster, step_size=0.02):
    """
    Per-component sign of the Newton step.

    Reaches the trench quickly but chatters (sign flipping) once on it.
    First-order sliding mode in the standard x/y basis.
    """
    grad, H = _estimate(cluster)
    try:
        delta = np.linalg.solve(H, -grad)
    except np.linalg.LinAlgError:
        delta = np.zeros(2)

    step = step_size * np.sign(delta)
    return float(step[0]), float(step[1])


# -----------------------------------------------------------------------
# Primitive 3: Anisotropic Newton step (eigenbasis reweighting)
# -----------------------------------------------------------------------

def anisotropic_newton_step(cluster, gain_perp=0.2, gain_along=0.02, max_step=1.0):
    """
    Decompose Newton step into trench eigenbasis and apply separate gains.

    gain_perp:  gain for the convex (across-trench) direction -- large value
                for fast snap to the trench.
    gain_along: gain for the concave (along-trench) direction -- small value
                for gentle climb to the saddle.

    No chatter. Eigenvector sign flips are harmless because the step
    direction is determined by the signed curvature-normalized component.
    Falls back to plain newton_step if H is not a saddle.
    """
    grad, H = _estimate(cluster)
    frame = _trench_frame(H)
    if frame is None:
        return newton_step(cluster, step_size=gain_along, max_step=max_step)

    (lam_perp, v_perp), (lam_along, v_along) = frame

    c_perp  = -(v_perp  @ grad) / lam_perp
    c_along = -(v_along @ grad) / lam_along

    delta = gain_perp * c_perp * v_perp + gain_along * c_along * v_along

    norm = np.linalg.norm(delta)
    if norm > max_step:
        delta = delta / norm * max_step

    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 4: Tanh sliding step (boundary-layer SMC)
# -----------------------------------------------------------------------

def tanh_sliding_step(cluster, reach_gain=0.1, boundary=0.2,
                      gain_along=0.02, max_step=1.0):
    """
    Saturated reaching law in the trench eigenbasis.

    Far from trench: tanh saturates -> bounded constant-speed reach.
    Near trench (|s| < boundary): linear region -> exponential decay.
    Along-trench direction: gentle Newton step.

    Falls back to plain newton_step if H is not a saddle.
    """
    grad, H = _estimate(cluster)
    frame = _trench_frame(H)
    if frame is None:
        return newton_step(cluster, step_size=reach_gain, max_step=max_step)

    (lam_perp, v_perp), (lam_along, v_along) = frame

    s = -(v_perp @ grad) / lam_perp
    step_perp  = reach_gain * np.tanh(s / boundary) * v_perp
    step_along = gain_along * (v_along @ grad) / lam_along * v_along
    delta = step_perp + step_along

    norm = np.linalg.norm(delta)
    if norm > max_step:
        delta = delta / norm * max_step

    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 5: Adaptive eigen step (recommended)
# -----------------------------------------------------------------------

def adaptive_eigen_step(cluster, v_max=0.04):
    """
    Self-tuning eigenstep with a single parameter v_max.

    For each eigendirection i:
      c_i = -(v_i . grad) / lambda_i    (curvature-normalized Newton step)
      s_i = v_max * tanh(c_i / v_max)   (hyperbolic saturation)
      delta += s_i * v_i

    Behavior:
      Far from saddle (|c_i| >> v_max): s_i ~ v_max * sign(c_i)
        -> bounded constant-speed reach, no gain tuning.
      Near saddle (|c_i| << v_max): tanh(x) ~ x, so s_i ~ c_i
        -> recovers exact Newton with cubic convergence (superlinear).

    One parameter (v_max) sets both the reach speed far out and the
    boundary-layer width near the saddle. No separate per-direction gains.

    Falls back to plain newton_step (with step_size=v_max) if H is not
    a clean saddle.
    """
    grad, H = _estimate(cluster)
    frame = _trench_frame(H)
    if frame is None:
        return newton_step(cluster, step_size=v_max)

    (lam_perp, v_perp), (lam_along, v_along) = frame

    c_perp  = -(v_perp  @ grad) / lam_perp
    c_along = -(v_along @ grad) / lam_along

    s_perp  = v_max * np.tanh(c_perp  / v_max)
    s_along = v_max * np.tanh(c_along / v_max)

    delta = s_perp * v_perp + s_along * v_along

    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 6: Adaptive eigen step with |lambda| denominator (Dauphin)
# -----------------------------------------------------------------------

def adaptive_eigen_step_abs(cluster, v_max=0.04):
    """
    Case 5 with |lambda_i| in the denominator (Dauphin-style).

    For each eigendirection i:
      c_i = -(v_i . grad) / |lambda_i|    (sign-flipped along trench)
      s_i = v_max * tanh(c_i / v_max)
      delta += s_i * v_i

    Under the local quadratic model g ~ H r:
      perpendicular (lambda > 0):  c_perp  ~ -r_perp   (snap to trench)
      along trench  (lambda < 0):  c_along ~ +r_along  (descend away from saddle)

    Replaces the saddle attractor of adaptive_eigen_step with a snap-then-
    descend controller: perp direction still attracts the formation to the
    trench (the positive-eigenvalue eigendirection of H), the along direction
    repels from the saddle and slides the formation down the trench toward
    whatever it leads to.

    On fields whose trench has no finite minimum (the bimodal Gaussian used
    in main_mod_dauph.py is one), the formation descends along the trench
    indefinitely; v_max bounds the per-step velocity.

    Falls back to plain newton_step (with step_size=v_max) if H is not a
    clean saddle, matching adaptive_eigen_step's fallback.
    """
    grad, H = _estimate(cluster)
    frame = _trench_frame(H)
    if frame is None:
        return newton_step(cluster, step_size=v_max)

    (lam_perp, v_perp), (lam_along, v_along) = frame

    c_perp  = -(v_perp  @ grad) / abs(lam_perp)
    c_along = -(v_along @ grad) / abs(lam_along)

    s_perp  = v_max * np.tanh(c_perp  / v_max)
    s_along = v_max * np.tanh(c_along / v_max)

    delta = s_perp * v_perp + s_along * v_along

    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 7: Separatrix Logic C (vector field)
# -----------------------------------------------------------------------
# Ported from separatrix_interactive_v6r.ipynb, classes HybridStepperBase
# and LogicCStepper.  Operates on a VECTOR field: queries cluster.field
# directly rather than cluster.sample_field_at_robots().
#
# The core idea: fit two independent 6-coef quadratics (one for u, one for v)
# to recover the Jacobian and its determinant landscape.  det(J) is a scalar
# field whose trench is the separatrix.  Logic C then navigates this scalar
# landscape using the local flow direction (theta_u[0], theta_v[0]) to
# choose between two eigenstep bodies:
#   Logic A: signed lambda denominator -> attracts toward the trench saddle.
#   Logic B: |lambda| denominator      -> descends along the trench.
# A FLOW-band override kicks in when det(J) is near zero (on the separatrix).
# -----------------------------------------------------------------------


def _sample_vector_at_robots(cluster):
    """
    Sample (u, v) from the vector field at each robot's position.

    Noise model (Paper_Draft_2A, Section VI Noise Model; both hooks default
    to 0 for backward compatibility):

    Position noise `cluster.position_noise_std`: Gaussian N(0, sigma_p^2)
    is added independently to each robot's (x, y) before querying the field,
    i.e. the field is sampled at a perturbed location while the quadratic
    fit uses the nominal robot positions.  True robot state is unchanged.

    Measurement noise `cluster.measurement_noise_std`: Gaussian
    N(0, sigma_uv^2) is added independently to each robot's u and v reading
    after the field query.  Same additive-sensor-noise model as the
    `noise_std` hook in the archived 3-robot field_types.py and as
    Michini et al. 2014 (T-RO), Eq. (6).

    Returns:
        u_arr: (6,) array of u-component readings
        v_arr: (6,) array of v-component readings
    """
    coords = cluster.get_robot_positions()
    pos_sigma  = getattr(cluster, 'position_noise_std', 0.0)
    meas_sigma = getattr(cluster, 'measurement_noise_std', 0.0)
    u_arr = np.zeros(6)
    v_arr = np.zeros(6)
    for i in range(6):
        x = coords[2*i]     + (np.random.randn() * pos_sigma if pos_sigma > 0.0 else 0.0)
        y = coords[2*i + 1] + (np.random.randn() * pos_sigma if pos_sigma > 0.0 else 0.0)
        u_arr[i], v_arr[i] = cluster.field.get_value(x, y)
        if meas_sigma > 0.0:
            u_arr[i] += np.random.randn() * meas_sigma
            v_arr[i] += np.random.randn() * meas_sigma
    return u_arr, v_arr


def _fit_vector_quadratic(rel_pos, u_readings, v_readings):
    """
    Fit a 6-parameter quadratic to each field component independently.

    Uses the same _quadratic_basis already defined above.  Builds the 6x6
    design matrix once, solves for theta_u and theta_v separately.

    Args:
        rel_pos:     (6, 2) relative robot positions (rx, ry) w.r.t. centroid
        u_readings:  (6,)  u-component samples
        v_readings:  (6,)  v-component samples

    Returns:
        theta_u, theta_v: each (6,) coefficient vectors
                          [f_c, df/dx, df/dy, d2f/dx2, d2f/dxdy, d2f/dy2]
    """
    Phi = np.array([_quadratic_basis(rel_pos[j, 0], rel_pos[j, 1])
                    for j in range(6)])
    if abs(np.linalg.det(Phi)) > 1e-10:
        theta_u = np.linalg.solve(Phi, u_readings)
        theta_v = np.linalg.solve(Phi, v_readings)
    else:
        theta_u = np.linalg.lstsq(Phi, u_readings, rcond=None)[0]
        theta_v = np.linalg.lstsq(Phi, v_readings, rcond=None)[0]
    return theta_u, theta_v


def _det_value(theta_u, theta_v):
    """det(J) at the centroid from quadratic fit coefficients."""
    _, ux, uy, _, _, _ = theta_u
    _, vx, vy, _, _, _ = theta_v
    return ux * vy - uy * vx


def _det_gradient(theta_u, theta_v):
    """Gradient of det(J) w.r.t. position, evaluated at the centroid."""
    _, ux, uy, uxx, uxy, uyy = theta_u
    _, vx, vy, vxx, vxy, vyy = theta_v
    D_x = uxx * vy + ux * vxy - uxy * vx - uy * vxx
    D_y = uxy * vy + ux * vyy - uyy * vx - uy * vxy
    return np.array([D_x, D_y])


def _det_hessian(theta_u, theta_v):
    """Hessian of det(J) w.r.t. position, evaluated at the centroid."""
    _, _, _, uxx, uxy, uyy = theta_u
    _, _, _, vxx, vxy, vyy = theta_v
    D_xx = 2.0 * uxx * vxy - 2.0 * uxy * vxx
    D_xy = uxx * vyy - uyy * vxx
    D_yy = 2.0 * uxy * vyy - 2.0 * uyy * vxy
    return np.array([[D_xx, D_xy], [D_xy, D_yy]])


def separatrix_logic_c_step(cluster, v_max=0.04, eps_raw=1e-3, eps_dim=0.025):
    """
    Logic C: flow-projected-on-trench selector for vector fields.

    Fits two quadratics to recover the local Jacobian, then navigates the
    det(J) landscape.  On the separatrix (det(J) ~ 0) a FLOW step follows
    the local vector field tangent to the trench.  Away from it, the local
    flow direction at the centroid (theta_u[0], theta_v[0]) is projected
    onto the along-trench eigenvector of H_det to select:
      flow points along descent -> Logic B (slide along trench)
      flow points against descent -> Logic A (attract toward saddle)

    Args:
        cluster:  PentagonCluster with a vector field attached
        v_max:    per-direction saturation speed (m/s)
        eps_raw:  raw det(J) threshold for the FLOW band
        eps_dim:  dimensionless det(J)/||H_det||_F threshold for FLOW band

    Returns:
        (vx_c, vy_c): centroid velocity command
    """
    eps = 1e-9

    # -- Estimation --------------------------------------------------------
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)

    det_val  = _det_value(theta_u, theta_v)
    grad_det = _det_gradient(theta_u, theta_v)
    H_det    = _det_hessian(theta_u, theta_v)

    # -- FLOW-band check ---------------------------------------------------
    H_norm = np.linalg.norm(H_det, 'fro')
    on_raw = abs(det_val) < eps_raw
    on_dim = (abs(det_val) / max(H_norm, 1e-12)) < eps_dim

    # Per-step mode logging (2026-07-02): if the cluster carries a
    # `diagnostics` list (PentagonCluster does; reset() clears it), record
    # which selector branch fired and the quantities that decided it.
    # Backward compatible: silent when the attribute is absent/None.
    def _log(mode):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            lam_d = np.linalg.eigvalsh(H_det)
            diag.append({
                'mode': mode, 'det': float(det_val),
                'det_ratio': float(abs(det_val) / max(H_norm, 1e-12)),
                'lam1': float(lam_d[0]), 'lam2': float(lam_d[1]),
            })

    if on_raw or on_dim:
        # FLOW step: follow local field along trench, snap perp to trench.
        f = np.array([theta_u[0], theta_v[0]])
        lam, V = np.linalg.eigh(H_det)
        if lam[0] * lam[1] >= -eps or np.min(np.abs(lam)) < eps:
            # Degenerate H_det: plain saturated flow step.
            _log('FLOW_DRIFT')
            n = np.linalg.norm(f)
            if n < 1e-12:
                return 0.0, 0.0
            scale = v_max * np.tanh(n / v_max) / n
            return float(f[0] * scale), float(f[1] * scale)
        i_neg = 0 if lam[0] < lam[1] else 1
        i_pos = 1 - i_neg
        v_neg = V[:, i_neg]
        v_pos = V[:, i_pos]
        c_along = float(f @ v_neg)
        r_perp  = float(grad_det @ v_pos) / lam[i_pos]
        c_perp  = -r_perp
        s_along = v_max * np.tanh(c_along / v_max)
        s_perp  = v_max * np.tanh(c_perp  / v_max)
        delta = s_along * v_neg + s_perp * v_pos
        _log('FLOW')
        return float(delta[0]), float(delta[1])

    # -- A/B selector ------------------------------------------------------
    lam, V = np.linalg.eigh(H_det)

    if lam[0] * lam[1] >= -eps:
        # Not a saddle of det(J): fall back to Logic A.
        _log('ATTRACT_FALLBACK')
        c0 = -(V[:, 0] @ grad_det) / (lam[0] if abs(lam[0]) > eps else eps)
        c1 = -(V[:, 1] @ grad_det) / (lam[1] if abs(lam[1]) > eps else eps)
        delta = (v_max * np.tanh(c0 / v_max) * V[:, 0] +
                 v_max * np.tanh(c1 / v_max) * V[:, 1])
        return float(delta[0]), float(delta[1])

    i_neg = 0 if lam[0] < lam[1] else 1
    v_neg = V[:, i_neg]

    # Sign-stabilize v_neg so it points in the direction of decreasing det(J).
    if float(grad_det @ v_neg) > 0:
        v_neg = -v_neg

    f = np.array([theta_u[0], theta_v[0]])

    if float(f @ v_neg) > 0:
        # Flow points along trench descent -> Logic B (|lambda| denominator).
        mode_ab = 'SLIDE'
        c0 = -(V[:, 0] @ grad_det) / abs(lam[0])
        c1 = -(V[:, 1] @ grad_det) / abs(lam[1])
    else:
        # Flow points against descent -> Logic A (signed lambda denominator).
        mode_ab = 'ATTRACT'
        c0 = -(V[:, 0] @ grad_det) / lam[0]
        c1 = -(V[:, 1] @ grad_det) / lam[1]

    _log(mode_ab)
    delta = (v_max * np.tanh(c0 / v_max) * V[:, 0] +
             v_max * np.tanh(c1 / v_max) * V[:, 1])
    return float(delta[0]), float(delta[1])


def separatrix_logic_park_step(cluster, v_max=0.04, eps_raw=1e-3,
                                eps_dim=0.025, d_capture=None):
    """
    Logic Park (added 2026-07-02): Logic C plus an estimator-aware PARK
    mode. Kept as a SEPARATE primitive so separatrix_logic_c_step remains
    unchanged as the fallback/revert path (user decision 2026-07-02).

    Motivation (Paper_Draft_2A Sections III-E and VII): the theory's
    eigenvalue-based capture test (H_D positive definite near a well of
    det(J)) never fires in practice, because the quadratic model's H_D
    estimate is structurally biased near the wells: the estimated
    eigenvalues come out indefinite and roughly 10x too small. The PARK
    test instead uses only the quantities the estimator delivers reliably
    (det(J)_hat and its gradient):

        if det(J)_hat < d_capture:   # deep inside a well of det(J)
            command componentwise tanh-saturated gradient DESCENT on
            det(J), which converges to the well minimum, i.e. the flow
            saddle at the end of the separatrix.

    d_capture is the park-vs-traverse knob: None (default) disables the
    test and the primitive is behaviorally identical to Logic C (verified
    trajectory-exact, noise-free, in experiments/park_vs_traverse_demo.py);
    a finite value (e.g. -0.5 for the A=0.1 double gyre, about half the
    well depth pi^4 A^2) parks the team at the terminal saddle.

    The estimation block and the selector below the PARK test are verbatim
    Logic C, operating on the SAME single fit per step (no double
    sampling, so noisy behavior is directly comparable to Logic C).

    Args:
        cluster:   PentagonCluster with a vector field attached
        v_max:     per-direction saturation speed (m/s)
        eps_raw:   raw det(J) threshold for the FLOW band
        eps_dim:   dimensionless det(J)/||H_det||_F threshold for FLOW band
        d_capture: capture threshold on det(J)_hat; None = never park

    Returns:
        (vx_c, vy_c): centroid velocity command
    """
    eps = 1e-9

    # -- Estimation (identical to Logic C) ---------------------------------
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)

    det_val  = _det_value(theta_u, theta_v)
    grad_det = _det_gradient(theta_u, theta_v)
    H_det    = _det_hessian(theta_u, theta_v)

    H_norm = np.linalg.norm(H_det, 'fro')

    def _log(mode):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            lam_d = np.linalg.eigvalsh(H_det)
            diag.append({
                'mode': mode, 'det': float(det_val),
                'det_ratio': float(abs(det_val) / max(H_norm, 1e-12)),
                'lam1': float(lam_d[0]), 'lam2': float(lam_d[1]),
            })

    # -- PARK: estimator-aware capture (the one addition over Logic C) -----
    if d_capture is not None and det_val < d_capture:
        _log('PARK')
        vx = -v_max * np.tanh(grad_det[0] / v_max)
        vy = -v_max * np.tanh(grad_det[1] / v_max)
        return float(vx), float(vy)

    # -- FLOW-band check (verbatim Logic C from here down) ------------------
    on_raw = abs(det_val) < eps_raw
    on_dim = (abs(det_val) / max(H_norm, 1e-12)) < eps_dim

    if on_raw or on_dim:
        f = np.array([theta_u[0], theta_v[0]])
        lam, V = np.linalg.eigh(H_det)
        if lam[0] * lam[1] >= -eps or np.min(np.abs(lam)) < eps:
            _log('FLOW_DRIFT')
            n = np.linalg.norm(f)
            if n < 1e-12:
                return 0.0, 0.0
            scale = v_max * np.tanh(n / v_max) / n
            return float(f[0] * scale), float(f[1] * scale)
        i_neg = 0 if lam[0] < lam[1] else 1
        i_pos = 1 - i_neg
        v_neg = V[:, i_neg]
        v_pos = V[:, i_pos]
        c_along = float(f @ v_neg)
        r_perp  = float(grad_det @ v_pos) / lam[i_pos]
        c_perp  = -r_perp
        s_along = v_max * np.tanh(c_along / v_max)
        s_perp  = v_max * np.tanh(c_perp  / v_max)
        delta = s_along * v_neg + s_perp * v_pos
        _log('FLOW')
        return float(delta[0]), float(delta[1])

    lam, V = np.linalg.eigh(H_det)

    if lam[0] * lam[1] >= -eps:
        _log('ATTRACT_FALLBACK')
        c0 = -(V[:, 0] @ grad_det) / (lam[0] if abs(lam[0]) > eps else eps)
        c1 = -(V[:, 1] @ grad_det) / (lam[1] if abs(lam[1]) > eps else eps)
        delta = (v_max * np.tanh(c0 / v_max) * V[:, 0] +
                 v_max * np.tanh(c1 / v_max) * V[:, 1])
        return float(delta[0]), float(delta[1])

    i_neg = 0 if lam[0] < lam[1] else 1
    v_neg = V[:, i_neg]

    if float(grad_det @ v_neg) > 0:
        v_neg = -v_neg

    f = np.array([theta_u[0], theta_v[0]])

    if float(f @ v_neg) > 0:
        mode_ab = 'SLIDE'
        c0 = -(V[:, 0] @ grad_det) / abs(lam[0])
        c1 = -(V[:, 1] @ grad_det) / abs(lam[1])
    else:
        mode_ab = 'ATTRACT'
        c0 = -(V[:, 0] @ grad_det) / lam[0]
        c1 = -(V[:, 1] @ grad_det) / lam[1]

    _log(mode_ab)
    delta = (v_max * np.tanh(c0 / v_max) * V[:, 0] +
             v_max * np.tanh(c1 / v_max) * V[:, 1])
    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitives 8 and 9: Logic G Okubo-Weiss contour tracker (vector field)
# Ported from separatrix_interactive_v002.ipynb
# -----------------------------------------------------------------------

def _lg_normalise(v, eps=1e-12):
    """Return v / ||v||, or v unchanged if ||v|| < eps."""
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def _lg_perp_aligned_with(direction, reference):
    """
    Return the unit perpendicular to `direction` whose dot with `reference`
    is non-negative (agrees with `reference`).
    """
    perp = np.array([-direction[1], direction[0]])
    if np.dot(perp, reference) < 0:
        perp = -perp
    return perp


def _estimate_det_and_grad_pentagon(cluster):
    """
    Estimate det(J), grad(det(J)), Hessian of det(J), and flow for
    a 6-robot PentagonCluster.

    Reuses the existing quadratic fit infrastructure from Primitive 7.

    Returns:
        (det_val, grad_det, H_det, flow, theta_u, theta_v)
          det_val:  scalar, det of the Jacobian at the centroid
          grad_det: (2,) raw gradient of det(J) at the centroid
          H_det:    (2,2) Hessian of det(J) at the centroid
          flow:     (2,) field vector at the centroid (theta_u[0], theta_v[0])
          theta_u, theta_v: quadratic fit coefficient vectors (for downstream use)
    """
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)
    det_val  = _det_value(theta_u, theta_v)
    grad_det = _det_gradient(theta_u, theta_v)
    H_det    = _det_hessian(theta_u, theta_v)
    flow     = np.array([theta_u[0], theta_v[0]])
    return det_val, grad_det, H_det, flow, theta_u, theta_v


# -----------------------------------------------------------------------
# Primitive 8: heuristic Logic G (fixed orientation)
# -----------------------------------------------------------------------

def logic_g_zero_flow_pentagon(cluster, step_size=0.04, correction_weight=0.0):
    """
    Logic G contour tracker for 6-robot PentagonCluster (heuristic version).

    Drives the cluster centroid onto the det(J)=0 Okubo-Weiss boundary using
    the heuristic gradient/flow blend from separatrix_interactive_v002.

    Fixed orientation (no omega returned).  Uses the pentagon's existing
    quadratic-fit infrastructure for det_val and grad_det.

    Args:
        cluster:           PentagonCluster with a vector field attached.
        step_size:         Commanded speed magnitude (m/s), default 0.04.
        correction_weight: Flow blend weight w in [0, 1].  0.0 = pure gradient
                           direction (v002 default).

    Returns:
        (vx_c, vy_c)
    """
    det_val, grad_det, _, flow, _, _ = _estimate_det_and_grad_pentagon(cluster)

    # -- Zero-seeking direction ---------------------------------------------
    g_norm = np.linalg.norm(grad_det)
    if g_norm < 1e-10:
        direction = _lg_normalise(flow)
    else:
        zero_dir  = -np.sign(det_val) * grad_det / g_norm
        flow_unit = _lg_normalise(flow)
        cos_sim   = float(np.dot(zero_dir, flow_unit))
        if cos_sim >= 0:
            blend     = ((1.0 - correction_weight) * zero_dir
                         + correction_weight * flow_unit)
            direction = _lg_normalise(blend)
        else:
            direction = _lg_perp_aligned_with(zero_dir, flow_unit)

    vx = step_size * float(direction[0])
    vy = step_size * float(direction[1])
    return vx, vy


# -----------------------------------------------------------------------
# Primitive 9: elegant Newton-step Logic G (fixed orientation)
# -----------------------------------------------------------------------

def logic_g_newton_contour_pentagon(cluster, v_max=0.04, eps_grad=1e-6,
                                    use_halley=False):
    """
    Newton-step contour tracker for 6-robot PentagonCluster.

    Drives det(J) -> 0 using a Newton step perpendicular to the contour and
    slides along the exact contour tangent perp(grad_D).  No heuristic flow
    blending.  Fixed orientation (no omega).

    Perpendicular step:
      dp_perp = -(D / ||grad_D||^2) * grad_D     (linear Newton toward D=0)
      Saturated via v_max * tanh(c_per / v_max) in the descent direction.

    Tangential step:
      t_hat = rot90(grad_D) / ||grad_D||   (exact level-set tangent),
      sign-stabilized so the cluster orbits with the flow direction,
      saturated via v_max * tanh(c_tan / v_max).

    History (2026-07-02): an earlier version used the eigenvector of H_det
    with the smallest |eigenvalue| as the tangent.  A head-to-head across
    the double gyre (noise-free and noisy) and the 2km Santa Barbara HFR
    field showed the perp-gradient tangent tracks 2x to 10^4x tighter and
    circulates farther in every arena, so the eigen-tangent was removed.
    The tangent of a level set is perpendicular to the gradient by
    definition; the eigenvector only approximates it and is ill-defined
    where the H_det eigenvalue magnitudes coincide (which happens exactly
    on the double-gyre diamond).

    Degenerate fallback (||grad_D|| < eps_grad): drift with flow.

    Args:
        cluster:    PentagonCluster with a vector field attached.
        v_max:      Per-direction saturation speed (m/s), default 0.04.
        eps_grad:   Gradient norm threshold below which flow-drift fallback fires.
        use_halley: If True, use Halley quadratic correction instead of linear
                    Newton for the perpendicular step.  Default False.

    Returns:
        (vx_c, vy_c)
    """
    det_val, grad_det, H_det, flow, _, _ = _estimate_det_and_grad_pentagon(cluster)

    g_norm = float(np.linalg.norm(grad_det))

    # -- Degenerate fallback ------------------------------------------------
    if g_norm < eps_grad:
        f_norm = float(np.linalg.norm(flow))
        if f_norm < 1e-12:
            return 0.0, 0.0
        scale = v_max * float(np.tanh(f_norm / v_max)) / f_norm
        return float(flow[0] * scale), float(flow[1] * scale)

    # -- Tangent direction: perp(grad_D), the exact level-set tangent -------
    t_hat = np.array([-grad_det[1], grad_det[0]]) / g_norm

    # Sign-stabilize: t_hat should point in the forward-flow direction.
    if float(np.dot(flow, t_hat)) < 0:
        t_hat = -t_hat

    # -- Perpendicular Newton step toward D=0 ------------------------------
    if use_halley:
        kappa = float(grad_det @ H_det @ grad_det) / (g_norm ** 2)
        disc  = g_norm ** 2 - 2.0 * float(det_val) * kappa
        eps_k = 1e-8
        if abs(kappa) >= eps_k and disc >= 0:
            s_mag = (g_norm - float(np.sqrt(disc))) / kappa
            dp_perp = s_mag * float(np.sign(det_val)) * (-grad_det / g_norm)
        else:
            dp_perp = -(float(det_val) / g_norm ** 2) * grad_det
    else:
        dp_perp = -(float(det_val) / g_norm ** 2) * grad_det

    # Descent unit direction: -sign(D) * grad_D / ||grad_D||
    u_hat   = -float(np.sign(det_val)) * grad_det / g_norm
    c_per   = float(np.dot(dp_perp, u_hat))
    s_per   = v_max * float(np.tanh(c_per / v_max))
    step_per = s_per * u_hat

    # -- Tangential drift along contour ------------------------------------
    c_tan    = float(np.dot(flow, t_hat))
    s_tan    = v_max * float(np.tanh(c_tan / v_max))
    step_tan = s_tan * t_hat

    delta = step_per + step_tan
    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 10: OECS / TRAP tracker (objective rate-of-strain)
# -----------------------------------------------------------------------
# Tracks attracting objective Eulerian coherent structures (OECSs), the
# instantaneous limits of LCSs (Serra and Haller 2016), operationalized as
# TRAPs (Serra et al. 2020).  Everything is built from the rate-of-strain
# tensor S = (J + J^T)/2, which is objective: under any time-dependent
# rotation Q(t) of the observer frame, S transforms as Q S Q^T, so its
# eigenvalues are invariant and its eigenvectors co-rotate with the frame.
# det(J), vorticity, and the Okubo-Weiss field are NOT objective (they are
# only Galilean invariant), so the Logic C separatrix tracker above can be
# fooled by a rotating frame; this primitive cannot.
#
# Structure tracked: trenches (valley lines) of s1, the smaller eigenvalue
# of S.  s1 < 0 is the local instantaneous compression rate; the attracting
# OECS/TRAP is the curve through a local minimum of s1 whose tangent is the
# stretching eigenvector e2 and whose normal is the compression eigenvector
# e1.  The TRAP core is the s1 minimum itself.
#
# STRUCTURAL LIMIT OF THE QUADRATIC FIT (2026-07-09, verified numerically):
# under the quadratic velocity model the entries of S are affine in
# position, so the model's s1 = mu - ||(a, b)|| is the negative norm of an
# affine map: a CONCAVE function whose Hessian is negative semidefinite for
# every field, and identically ~0 on the double gyre (where the shear
# strain b == 0).  The true trench's positive transverse curvature lives in
# THIRD velocity derivatives the fit cannot see.  A Newton step on
# (grad s1, H_s1), the naive analogue of the Logic C machinery, is
# therefore impossible in principle, not merely inaccurate.  This tracker
# uses no H_s1: the frame comes from S's eigenvectors (first-order fitted
# coefficients, noise gain (2/sqrt(10)) sigma/rho, one full rung BETTER
# than the H_D eigenvector tangent Logic C uses), and transverse
# regulation is saturated gradient descent of s1 along e1 (gradient-rate,
# not Newton-rate; the restoring slope was verified against the analytic
# transverse curvature on the double gyre).
# -----------------------------------------------------------------------


def _strain_quantities(theta_u, theta_v):
    """
    Rate-of-strain quantities at the centroid from the two quadratic fits.

    S = [[u_x, b], [b, v_y]] with b = (u_y + v_x)/2.  Writing
    mu = (u_x + v_y)/2 (mean strain, zero for incompressible flow) and
    a = (u_x - v_y)/2 (deviatoric normal strain), the eigenvalues are
    s1 = mu - r and s2 = mu + r with r = sqrt(a^2 + b^2).  Under the
    quadratic model a, b, mu are affine in position, so grad s1 is exact
    for the fitted field:  grad s1 = grad mu - (a grad a + b grad b)/r.

    Returns:
        s1:      smaller eigenvalue of S (compression rate, < 0 where
                 attraction dominates)
        grad_s1: (2,) spatial gradient of s1 at the centroid
        e1:      (2,) unit compression eigenvector (normal to the
                 attracting curve); sign arbitrary
        e2:      (2,) unit stretching eigenvector (tangent to the
                 attracting curve); sign arbitrary
        r:       deviatoric strain magnitude; r ~ 0 means S is nearly
                 isotropic and the eigenframe is unreliable
    """
    _, ux, uy, uxx, uxy, uyy = theta_u
    _, vx, vy, vxx, vxy, vyy = theta_v

    mu = 0.5 * (ux + vy)
    a  = 0.5 * (ux - vy)
    b  = 0.5 * (uy + vx)
    r  = float(np.hypot(a, b))
    s1 = mu - r

    grad_mu = 0.5 * np.array([uxx + vxy, uxy + vyy])
    grad_a  = 0.5 * np.array([uxx - vxy, uxy - vyy])
    grad_b  = 0.5 * np.array([uxy + vxx, uyy + vxy])
    grad_s1 = grad_mu - (a * grad_a + b * grad_b) / max(r, 1e-12)

    S = np.array([[ux, b], [b, vy]])
    _, V = np.linalg.eigh(S)      # ascending: column 0 <-> s1, column 1 <-> s2
    e1 = V[:, 0]
    e2 = V[:, 1]
    return float(s1), grad_s1, e1, e2, r


def oecs_trap_step(cluster, v_max=0.04, g_perp=1.0, s_trim=0.05,
                   s_capture=None, eps_degen=1e-3):
    """
    Objective attracting-OECS/TRAP tracker (Primitive 10).

    Selector, executed once per control cycle after the estimation step:

      1. ACQUIRE  -- before first structure contact, or when S is nearly
                     isotropic (r < eps_degen): componentwise saturated
                     gradient descent on s1, toward stronger attraction.
      2. TRIM     -- after first contact, if s1 > -s_trim the local
                     attraction no longer dominates (structure end or
                     tensor degeneracy): hold position.  This is the
                     Serra-Haller termination rule playing the role that
                     the capture test plays in Logic C.
      3. PARK     -- (core-seek only, s_capture set) if s1 < s_capture the
                     team is inside the TRAP core well: componentwise
                     saturated gradient descent on s1 settles at the s1
                     minimum, the objective saddle.
      4. TRACK    -- otherwise: transverse channel descends s1 along the
                     compression eigenvector e1 (snaps onto the trench);
                     tangential channel drives along the stretching
                     eigenvector e2.  Tangent sign: core-seek descends s1
                     along the curve (toward the core); ride keeps
                     continuity with the previous step's tangent (seeded
                     from the flow direction once, then never consults the
                     flow again -- eigenvector direction fields have no
                     global sign, so continuity is the only objective
                     orientation rule).

    Every command is built from s1, grad s1, e1, e2 only, all of which
    transform equivariantly under time-dependent frame rotations. The
    TRACK and TRIM commands are therefore exactly frame-covariant. ACQUIRE
    and PARK saturate the gradient componentwise in observer axes, so they
    are covariant only where the saturation is inactive; they still descend
    the invariant s1 in every frame, so the closed loop tracks the same
    material structure in any frame up to that acquisition transient.

    One knob mirrors Logic Park: s_capture = None (default) rides the
    structure and trims at its end; a finite value (e.g. -0.9 for the
    A = 0.1 double gyre, whose TRAP cores sit at s1 = -pi^2 A ~ -0.987)
    parks the team at the TRAP core.

    Per-run state is kept on the cluster (cleared implicitly by building a
    fresh PentagonCluster per run, the pattern all experiments use):
      cluster._oecs_prev_tangent  -- (2,) last commanded tangent direction
      cluster._oecs_banded        -- True once s1 first dropped below -s_trim

    Args:
        cluster:    PentagonCluster with a vector field attached
        v_max:      per-direction saturation speed (m/s)
        g_perp:     gradient gain on both s1-descent channels
        s_trim:     attraction-dominance threshold (s1 > -s_trim => TRIM)
        s_capture:  TRAP-core park threshold on s1; None = ride mode
        eps_degen:  deviatoric strain magnitude below which the eigenframe
                    is treated as degenerate

    Returns:
        (vx_c, vy_c): centroid velocity command
    """
    # -- Estimation (same single fit per step as Logic C) -------------------
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)

    s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
    flow = np.array([theta_u[0], theta_v[0]])

    banded = getattr(cluster, '_oecs_banded', False)
    if s1 < -s_trim and not banded:
        banded = True
        cluster._oecs_banded = True

    def _log(mode):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            diag.append({
                'mode': mode, 's1': float(s1), 'r': float(r),
                'g_perp_comp': float(grad_s1 @ e1),
                'g_tan_comp': float(grad_s1 @ e2),
            })

    def _descend_s1():
        vx = -v_max * np.tanh(g_perp * grad_s1[0] / v_max)
        vy = -v_max * np.tanh(g_perp * grad_s1[1] / v_max)
        return float(vx), float(vy)

    # -- 1. ACQUIRE: no structure yet, or eigenframe degenerate -------------
    if r < eps_degen or not banded:
        _log('ACQUIRE')
        return _descend_s1()

    # -- 2. TRIM: attraction stopped dominating on-structure ----------------
    if s1 > -s_trim:
        _log('TRIM')
        return 0.0, 0.0

    # -- 3. PARK: inside the TRAP core well (core-seek only) ----------------
    if s_capture is not None and s1 < s_capture:
        _log('PARK')
        return _descend_s1()

    # -- 4. TRACK: transverse snap along e1 + tangential drive along e2 -----
    # Transverse: descend s1 along the compression direction.  Quadratic in
    # e1, so the eigenvector's sign ambiguity cancels.
    c_perp = -g_perp * float(grad_s1 @ e1)
    step_perp = v_max * np.tanh(c_perp / v_max) * e1

    if s_capture is not None:
        # Core-seek: descend s1 along the curve, slowing into the core.
        mode = 'SEEK'
        c_tan = -g_perp * float(grad_s1 @ e2)
        step_tan = v_max * np.tanh(c_tan / v_max) * e2
        tangent = e2 if c_tan >= 0 else -e2
    else:
        # Ride: constant cruise along e2, sign by continuity.
        mode = 'RIDE'
        prev = getattr(cluster, '_oecs_prev_tangent', None)
        ref = prev if prev is not None else flow
        t_hat = e2 if float(e2 @ ref) >= 0 else -e2
        step_tan = v_max * np.tanh(1.0) * t_hat
        tangent = t_hat

    cluster._oecs_prev_tangent = tangent
    _log(mode)
    delta = step_perp + step_tan
    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Primitive 11: objective separatrix traverser (rate-of-strain, full network)
# -----------------------------------------------------------------------
# The S-based analogue of separatrix_logic_c_step (Primitive 7): rides the
# FULL separatrix network -- both halves, through the saddle -- using only
# the objective quantities s1, grad s1, and the strain eigenframe (e1, e2).
# Primitive 10 above targets a single attracting OECS segment and parks at
# its TRAP core; this primitive targets the trench of s1 itself and keeps
# going, the way Logic C keeps going past a saddle onto the next trench
# segment.
#
# THE KEY GEOMETRIC FACT (verified numerically on the double gyre, A = 0.1,
# 2026-07-21; do not re-derive, unit-tested): the scalar s1 has a genuine
# TRANSVERSE TRENCH (minimum) along the ENTIRE separatrix, both the
# attracting upper half and the repelling lower half (d^2 s1/dx^2 = +8.68 at
# y = +/-0.35, +6.89 at +/-0.25, +3.01 at +/-0.10, measured across x = 0 for
# every y).  What alternates between the two halves is NOT the sign of the
# trench; it is which eigenvector is TANGENT to it.  Walking the separatrix
# from the top saddle to the bottom saddle, e2 (stretching, attracting OECS
# tangent) is aligned with the line on the upper half and e1 (compression)
# is aligned with it on the lower half; they trade places exactly at the
# origin, the isotropic point where S = 0 and s1 = s2 = 0, so neither
# eigenvector is defined there.  (This reconciles with the material
# attracting/repelling classification in the paper's double-gyre discussion,
# which is a statement about which eigenvector is tangent, not about the
# sign of the s1 trench.)
#
# Consequently the transverse channel here does NOT snap along a fixed
# eigenvector the way Primitive 10 does (that only holds the half where the
# hardcoded eigenvector happens to be the transverse one).  It descends s1
# along whatever direction is orthogonal to the CURRENT RIDE TANGENT, which
# is correct on both halves because the trench never becomes a ridge.
#
# A SECOND geometric fact, found empirically running this controller against
# separatrix_logic_c_step: the s1-trench network, like the D-trench network
# it coincides with here, is a GRID, not a single curve -- the wall lines
# y = +/-0.5 are themselves s1-trenches (transverse to e1 there), crossing
# the separatrix at the saddles.  A tangent rule based on continuity alone
# (whichever eigenvector best matches the PREVIOUS tangent) has no reason to
# prefer continuing through a saddle over turning onto the crossing wall
# branch once noise or the approach angle nudges it that way, and once
# aboard the wall branch, continuity has no restoring force back toward the
# saddle: the ride sails off along the wall forever.
#
# The first fix tried was to choose the tangent each step by best alignment
# with the ambient flow (mirroring Logic C's f @ v_neg test, eq. d_selector),
# with continuity fixing only the sign.  This worked cleanly in the inertial
# frame, but was found to fail under a ROTATING observer: the frame's own
# solid-body swirl term does not vanish at a trench-network crossing the way
# the true flow does (it is a constant local vector there, proportional to
# POSITION, not to distance from the crossing), and at Omega = 0.2 it was
# large enough to flip which eigenvector the flow preferred a FULL
# TRENCH-LENGTH away from any saddle, not just near one -- verified
# numerically (true flow favors one eigenvector 2:1, the swirl-corrupted
# flow favors the other, from the very first tracking step).  No flow-based
# rule can be made robust to this, because the corruption scales with the
# same distance from the origin that the rule needs to operate over.
#
# THE ACTUAL FIX uses no flow at all.  Away from a stationary point of s1,
# grad s1 points almost entirely ALONG the trench (its transverse component
# vanishes there by construction, leaving only the along-track slope), so of
# the two eigenvectors, the TANGENT is whichever has the LARGER
# |grad s1 . e_i|, and the TRANSVERSE direction (already used by the channel
# below) is whichever has the SMALLER one -- the same two numbers pick both
# channels.  This was verified against the fitted estimator: it correctly
# resolves the vertical tangent through the origin eigenvector swap and
# correctly prefers the separatrix branch over the crossing wall branch at
# both saddles, using only s1, grad s1, and the eigenframe.  It is immune to
# the swirl problem above by construction, since it never reads the flow.
#
# The CROSS mode is kept only for the case r is degenerate on the very first
# tracking step with no previous tangent yet established (no continuity
# reference exists); there it falls back to the ambient flow, the one input
# in the whole controller not built from s1 alone, exactly as Logic C's
# FLOW-band fallback (FLOW_DRIFT) falls back to a plain saturated flow step
# when its own Hessian degenerates.  This branch is essentially unreachable
# on the double gyre once tracking has begun.
#
# OBJECTIVITY ACCOUNTING: ACQUIRE, PARK, CAPTURE, and BOTH channels of TRACK
# (transverse and tangent selection) are built only from s1, grad s1, and the
# co-rotating eigenvectors e1, e2, which transform equivariantly under a
# time-dependent rotation of the observer frame; continuity fixes only a
# sign.  The controller is therefore frame-objective throughout except for
# the essentially-unreachable CROSS fallback, a STRONGER claim than the
# flow-selected-tangent design it replaced, which was Galilean-objective but
# not fully frame-objective (see experiments/traverse_objectivity_demo.py for
# the closed-loop verification, both designs' results, and the Omega=0.2
# failure of the flow-based version).
# -----------------------------------------------------------------------


def oecs_separatrix_step(cluster, v_max=0.04, g_perp=1.0, s_trim=0.05,
                         r_band=0.05, g_capture=0.15, s_capture=None):
    """
    Objective separatrix traverser (Primitive 11).

    Selector, executed once per control cycle after the estimation step:

      1. ACQUIRE  -- before first structure contact: componentwise
                     saturated gradient descent on s1, toward stronger
                     attraction.  Homes onto the s1-trench from anywhere in
                     the basin; no straddling start required.
      2. PARK     -- (optional, s_capture set) if s1 < s_capture the team is
                     inside a core well: componentwise saturated gradient
                     descent on s1 settles at the local s1 minimum.  With
                     s_capture = None (default) this branch never fires and
                     the team relies on CAPTURE (below) instead.
      3. CAPTURE  -- UNCONDITIONAL hold at a 2D stationary point of s1: fires
                     whenever the full gradient magnitude |grad s1| drops
                     below g_capture while r stays large (so this is not
                     confused with an isotropic point, where r is small and
                     CROSS applies instead).  A flow saddle is not merely
                     where two 1D trenches cross; it is a genuine 2D local
                     MINIMUM of s1 (the TRAP core of Primitive 10, verified
                     s1 = -pi^2 A there), so grad s1 vanishes in every
                     direction, not just transversally, while on a generic
                     1D trench point |grad s1| stays large (zero only
                     transverse to the trench, full strength along it).
                     Checking the gradient directly, rather than trying to
                     keep choosing a tangent, matters because the ambient
                     flow ALSO collapses at the same point (a stagnation
                     point), so any tangent-selection rule that consulted
                     the flow would be starved of a reliable direction
                     exactly where it needs one.  Descends s1 the same way
                     PARK does; unlike PARK it needs no user-supplied s1
                     threshold, so it holds ANY network minimum by default,
                     the way TRIM in Primitive 10 holds an attracting
                     segment's end without a supplied threshold.
      4. CROSS    -- after first contact, whenever the eigenframe is
                     degenerate (r < r_band) AND no previous tangent has
                     yet been established: neither the eigenframe nor
                     continuity gives a usable direction (only the very
                     first tracking step after ACQUIRE can reach this).
                     Rides the ambient flow, the one input in the whole
                     controller not built from s1 alone; kept as the
                     honest fallback for a generic field, essentially
                     unreachable on the double gyre.
      5. TRACK    -- otherwise: ride the trench of s1.  The tangent is
                     whichever of e1, e2 has the LARGER |grad s1 . e_i|
                     (continuity with the previous tangent fixes only the
                     residual sign ambiguity); the transverse channel
                     descends s1 along the component of grad s1 orthogonal
                     to that tangent, using the SMALLER |grad s1 . e_i|,
                     so the same two numbers pick both channels and the
                     same law holds the trench on both the attracting and
                     the repelling half.  Away from a stationary point of
                     s1, grad s1 points mostly ALONG the trench (the
                     transverse derivative vanishes there by construction),
                     so the tangent is the eigenvector grad s1 is LESS
                     orthogonal to, not more -- this is what carries the
                     ride through a trench-NETWORK CROSSING (e.g. a saddle,
                     where the separatrix meets a wall trench and a rule
                     that only ever compared to the previous step could
                     lock onto the wrong branch) and through the
                     isotropic-point eigenvector swap alike, using no flow
                     at all: CAPTURE above is what stops this branch from
                     ever having to resolve the choice right at a genuine
                     stationary point, where grad s1 itself vanishes and
                     the discriminator loses power.

    Objectivity note: ACQUIRE, PARK, CAPTURE, and BOTH channels of TRACK
    (transverse and tangent selection) are built only from s1, grad s1, and
    the co-rotating eigenvectors e1, e2, all of which transform
    equivariantly under a time-dependent rotation of the observer frame
    (Proposition equivariance for Primitive 10 applies verbatim to these
    terms); continuity with the previous tangent, itself propagated from
    these same equivariant quantities, fixes only a sign. The only place
    the ambient flow enters at all is CROSS, reached solely on the very
    first tracking step if the eigenframe is degenerate there with no
    established tangent yet -- essentially unreachable once tracking has
    begun. This is a STRONGER objectivity claim than an earlier design that
    selected TRACK's tangent by best alignment with the flow at every step
    (matching Logic C's own flow-projection test, eq. d_selector): that
    design was found, empirically, to fail under a rotating observer whose
    swirl term does not vanish at a trench-network crossing the way the
    true flow does, so it can bias which branch gets selected even a full
    trench-length from the crossing (verified at Omega = 0.2). The
    gradient-based rule here uses no flow for branch selection, so it
    carries no such corruption in any frame; see
    experiments/traverse_objectivity_demo.py for the closed-loop
    verification.

    Per-run state is kept on the cluster (cleared implicitly by building a
    fresh PentagonCluster per run, the pattern all experiments use):
      cluster._oecs_prev_tangent  -- (2,) last commanded tangent direction
      cluster._oecs_banded        -- True once s1 first dropped below -s_trim

    Args:
        cluster:    PentagonCluster with a vector field attached
        v_max:      per-direction saturation speed (m/s)
        g_perp:     gradient gain on the transverse, CAPTURE, and ACQUIRE
                    channels
        s_trim:     attraction-dominance threshold marking first contact
                    (s1 < -s_trim latches `banded`)
        r_band:     strain-magnitude threshold below which the eigenframe is
                    treated as unreliable and CROSS takes over
        g_capture:  full-gradient-magnitude threshold below which CAPTURE
                    holds position at a 2D stationary point of s1 (a network
                    minimum / saddle); unconditional, unlike s_capture
        s_capture:  optional core-park threshold on s1 (PARK); None = rely
                    on CAPTURE at network minima and ride to the domain edge
                    otherwise

    Returns:
        (vx_c, vy_c): centroid velocity command
    """
    # -- Estimation (same single fit per step as Logic C and Primitive 10) --
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)

    s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
    flow = np.array([theta_u[0], theta_v[0]])

    banded = getattr(cluster, '_oecs_banded', False)
    if s1 < -s_trim and not banded:
        banded = True
        cluster._oecs_banded = True

    def _log(mode):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            diag.append({
                'mode': mode, 's1': float(s1), 'r': float(r),
            })

    def _descend_s1():
        # Original (per-component tanh saturation, distorts direction near
        # saturation):
        # vx = -v_max * np.tanh(g_perp * grad_s1[0] / v_max)
        # vy = -v_max * np.tanh(g_perp * grad_s1[1] / v_max)
        # return float(vx), float(vy)
        a = -g_perp * grad_s1
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return 0.0, 0.0
        scale = v_max * np.tanh(n / v_max) / n
        return float(a[0] * scale), float(a[1] * scale)

    # -- 1. ACQUIRE: no structure yet -----------------------------------
    if not banded:
        _log('ACQUIRE')
        return _descend_s1()

    # -- 2. PARK: optional core well, checked before CROSS/TRACK --------
    if s_capture is not None and s1 < s_capture:
        _log('PARK')
        return _descend_s1()

    # -- 3. CAPTURE: hold at a 2D stationary point of s1 --
    # A flow saddle is not merely a crossing of two 1D trenches; it is a
    # genuine 2D local MINIMUM of s1 (verified: true H_s1 eigenvalues both
    # +9.74 at the double-gyre saddle, matching the TRAP-core value
    # s1 = -pi^2 A of eq. s1_analytic_main), so grad s1 vanishes in EVERY
    # direction there, not just transversally. On a generic 1D trench point
    # grad s1 is large (it is exactly zero only transverse to the trench,
    # full-strength along it); only near a 2D minimum does the FULL gradient
    # collapse. This is the distinguishing test, and it matters because the
    # flow itself also collapses at the same point (a stagnation point of
    # the flow), which otherwise starves TRACK's flow-selected tangent of a
    # reliable direction exactly where it needs one -- empirically, without
    # this branch, some approaches to a saddle pick the flow-ambiguous wrong
    # eigenvector and slide off onto the crossing wall trench instead of
    # stopping. Checking the gradient directly sidesteps the ambiguity, but
    # the gradient test ALONE is not sufficient: a shallow, incidental flat
    # spot of s1 well away from any real structure (e.g. near a gyre center)
    # can also pass |grad s1| < g_capture for a step under position noise,
    # even though s1 there sits near 0, nowhere close to the true well depth
    # -- verified directly (s1 in [-0.06, +0.04] at such a spot vs. -0.76 to
    # -1.16 at the true saddle, at the SAME |grad s1| magnitudes). So
    # CAPTURE also requires s1 < -4*s_trim: a deep-well test using the same
    # scale parameter that already marks first structure contact, verified
    # to hold with comfortable margin everywhere the gradient test can
    # plausibly fire near a true saddle (s1 already below -0.9 once
    # |grad s1| is within an order of magnitude of g_capture) while
    # rejecting the shallow false positive. This is not gated on a
    # field-specific analytic value (unlike PARK/s_capture, which needs the
    # user to know the field's well depth in advance), so it still holds ANY
    # network minimum by default, generic across fields, exactly as the
    # TRIM test of Primitive 10 holds an attracting segment's end without a
    # user-supplied threshold.
    #
    # LATCH (mirrors `banded` above), RELEASED ON THE SAME CERTIFICATE PARK
    # ALREADY USES: under measurement/position noise, grad_s1's norm can
    # flicker back above g_capture for a step even after genuinely arriving
    # at the well, and every such excursion re-enters TRACK's tangent
    # discriminator in the same near-degenerate eigenframe region the
    # CAPTURE test exists to avoid -- verified to occasionally reselect the
    # wrong eigenvector there and slide onto the crossing wall under
    # position noise as small as sigma_p = 0.005. An UNCONDITIONAL latch
    # (tried first) fixes that but creates a worse failure: held forever
    # with no way to re-diagnose, so a later noise-driven excursion that
    # drifts the position entirely off the well kept applying pure grad-s1
    # descent through open field far from any structure -- not a stable
    # hold, an unmodulated walk down whatever the local gradient points at,
    # verified to run the team out past +/-2 domain widths within 1000
    # steps this way (s1 itself drifting toward and past 0 en route, the
    # tell that the fitted field is no longer describing a real well).
    # FIX, not a new estimator or threshold: Section stability's own PARK
    # argument already shows s1 is a Lyapunov function for this exact
    # descent under exact estimates (s1_dot <= 0, equality only at a
    # critical point) -- so gradient descent cannot run away WHILE s1
    # actually still reads like a trench, and only does so once the
    # estimate no longer does. The latch is released the instant s1 rises
    # back above -s_trim, the SAME threshold `banded` already uses to
    # define "no longer on structure": while below it, the well is real and
    # the latch may bridge a flicker of the transverse gradient test;
    # above it, the well is lost and the state machine falls through to
    # re-diagnose from TRACK, exactly as if CAPTURE had never latched.
    captured = getattr(cluster, '_oecs_captured', False)
    if captured and s1 > -s_trim:
        captured = False
        cluster._oecs_captured = False
    g_norm = float(np.linalg.norm(grad_s1))
    if captured or (r > r_band and g_norm < g_capture and s1 < -4.0 * s_trim):
        cluster._oecs_captured = True
        _log('CAPTURE')
        return _descend_s1()

    t_prev = getattr(cluster, '_oecs_prev_tangent', None)
    prev_hat = (t_prev / np.linalg.norm(t_prev)
               if t_prev is not None and np.linalg.norm(t_prev) > 1e-12
               else None)

    # -- 3. CROSS: eigenframe degenerate AND no continuity to fall back on
    # This is the genuine degeneracy case: r small (eigenframe undefined or
    # noisy) with no previous tangent yet established (only the very first
    # tracking step after ACQUIRE can reach this). Ride the ambient flow,
    # oriented arbitrarily since no reference exists; this is the one branch
    # that is not built purely from s1, grad s1, and the eigenframe, kept as
    # the honest fallback for a generic field. On the double gyre this
    # branch is essentially unreachable in practice (ACQUIRE's gradient
    # descent lands on-trench well before r drops this low with no prior
    # tangent).
    if r < r_band and prev_hat is None:
        _log('CROSS')
        fn = float(np.linalg.norm(flow))
        t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
        step_tan = v_max * np.tanh(1.0) * t_hat
        step_perp_x, step_perp_y = _descend_s1()
        step_perp = np.array([step_perp_x, step_perp_y])
        cluster._oecs_prev_tangent = t_hat
        delta = step_perp + step_tan
        return float(delta[0]), float(delta[1])

    # -- 4. TRACK: ride the s1 trench, tangent chosen WITHOUT the flow ---
    # The tangent of a trench is, by definition, the direction along which
    # the along-track derivative of s1 is small: away from a stationary
    # point, grad s1 points mostly ALONG the trench (the transverse
    # derivative vanishes there by construction, leaving only the
    # along-track slope), so of the two eigenvectors, the tangent is
    # whichever has the LARGER |grad s1 . e_i| and the transverse direction
    # is whichever has the SMALLER one -- the flip of the transverse
    # channel below, using the same two numbers. This replaces an earlier
    # design that chose the tangent by best alignment with the ambient
    # flow (mirroring Logic C's f @ v_neg test): that worked in the
    # inertial frame, but under a ROTATING observer the frame's own
    # solid-body swirl does not vanish at a trench-network crossing the
    # way the true flow does (it is a constant local vector there,
    # proportional to POSITION, not to proximity to the crossing), and at
    # Omega = 0.2 it was large enough, even a full trench-length away from
    # any saddle, to flip which eigenvector the flow preferred -- verified
    # numerically (true flow favors e2 by a 2:1 margin at one test point,
    # the swirl-corrupted flow favors e1). The gradient-based rule uses NO
    # flow at all, so it carries no such corruption in any frame, and it
    # was verified against the fitted estimator to correctly track the
    # separatrix tangent through the origin eigenvector swap and to
    # correctly resolve the wall-vs-separatrix crossing at both saddles.
    # Continuity with the previous tangent supplies only the sign, exactly
    # as it always did; the direction itself no longer depends on it.
    d1 = abs(float(grad_s1 @ e1))
    d2 = abs(float(grad_s1 @ e2))
    t_raw = e1 if d1 > d2 else e2
    sign_ref = prev_hat if prev_hat is not None else flow
    t_hat = t_raw if float(t_raw @ sign_ref) >= 0 else -t_raw

    # Transverse: descend s1 along the component of grad s1 orthogonal to
    # the tangent, not along a fixed eigenvector.  s1 is a genuine
    # transverse minimum along the whole network, so this single law snaps
    # onto the trench whether the tangent above resolved to e1 or e2.
    g_perp_vec = grad_s1 - float(grad_s1 @ t_hat) * t_hat
    g_perp_norm = float(np.linalg.norm(g_perp_vec))
    if g_perp_norm < 1e-12:
        step_perp = np.zeros(2)
    else:
        n_hat = g_perp_vec / g_perp_norm
        c_perp = -g_perp * g_perp_norm
        step_perp = v_max * np.tanh(c_perp / v_max) * n_hat

    step_tan = v_max * np.tanh(1.0) * t_hat

    cluster._oecs_prev_tangent = t_hat
    _log('TRACK')
    delta = step_perp + step_tan
    return float(delta[0]), float(delta[1])
