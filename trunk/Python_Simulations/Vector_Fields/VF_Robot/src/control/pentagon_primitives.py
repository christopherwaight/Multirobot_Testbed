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
"""
import numpy as np


# -----------------------------------------------------------------------
# Quadratic estimator
# -----------------------------------------------------------------------

def _quadratic_basis(rx, ry):
    """Basis vector for a single relative position (rx, ry)."""
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
