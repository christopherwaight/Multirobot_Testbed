"""
Control primitives for the 6-robot pentagon cluster.

Each primitive takes a PentagonCluster and returns the centroid velocity
(vx_c, vy_c). The six robots sample the field and a full quadratic is fit
about the centroid, recovering first and second derivatives.

Scalar primitives (1-6) are ported from
trunk/Python_Simulations/Separatrix_Control_testing/saddle_point_6_robot2.ipynb.
They fit one quadratic to cluster.sample_field_at_robots() and act on its
gradient and Hessian.

Vector primitives (7-11) are ported from separatrix_interactive_v6r.ipynb
and separatrix_interactive_v002.ipynb. They query cluster.field.get_value()
for (u, v), fit one quadratic per component, and act on det(J) or on the
rate-of-strain tensor S = (J + J^T)/2.

  1. newton_step                  plain Newton, straight-line approach
  2. sign_decomposed_step         per-component sign, chatters on the trench
  3. anisotropic_newton_step      per-eigendirection gains, no chatter
  4. tanh_sliding_step            boundary-layer SMC, smooth reach
  5. adaptive_eigen_step          one knob v_max, Newton near the saddle
                                  (recommended)
  6. adaptive_eigen_step_abs      |lambda| denominator, snap then descend
                                  the trench (Dauphin-style)
  7. separatrix_logic_c_step      det(J) trench navigator with FLOW band
     separatrix_logic_park_step   Logic C plus PARK at a det(J) well
  8. logic_g_zero_flow_pentagon   heuristic det(J) = 0 tracker
  9. logic_g_newton_contour_pentagon  Newton step onto det(J) = 0 plus
                                  tangent drift
 10. oecs_trap_step               objective tracker of attracting OECS/TRAP
                                  curves, the trenches of s1 (smaller
                                  eigenvalue of S)
 11. oecs_separatrix_step         objective traverser of the full separatrix
                                  network along the s1 trench (S analogue
                                  of 7)
"""
import numpy as np


# -----------------------------------------------------------------------
# Quadratic estimator
# -----------------------------------------------------------------------

def _quadratic_basis(rx, ry):
    """Basis [1, x, y, x^2/2, xy, y^2/2] at relative position (rx, ry).

    The paper (Draft_5c.tex, Eq. 7) names the coefficients a_1..a_6 for u
    and b_1..b_6 for v and orders its basis [1, x, y, xy, x^2/2, y^2/2], so
    theta[3] (xx) is a_5 and theta[4] (xy) is a_4. Mind the swap against
    Eq. 8-9. The one-half factors make the coefficients equal the Taylor
    derivatives, and Lemma 1's noise gains (8/sqrt(10) pure, 4/sqrt(10)
    mixed) assume them.
    """
    return np.array([1.0, rx, ry, rx**2 / 2.0, rx * ry, ry**2 / 2.0])


def _fit_quadratic(relative_positions, readings):
    """
    Fit a quadratic to 6 scalar readings at (6, 2) centroid-relative
    positions. Returns theta ordered as in _quadratic_basis.

    Phi is singular exactly when the six points lie on a common conic, so
    six robots on one ring always fail. The pentagon-plus-center formation
    does not, since the ring's circumcircle misses the center robot.
    Conditioning degrades as the formation deforms toward a conic, hence
    the det check.
    """
    Phi = np.array([_quadratic_basis(relative_positions[j, 0],
                                     relative_positions[j, 1])
                    for j in range(6)])
    s = np.array(readings, dtype=float)

    if abs(np.linalg.det(Phi)) > 1e-10:
        return np.linalg.solve(Phi, s)
    return np.linalg.lstsq(Phi, s, rcond=None)[0]


def _extract_grad_hessian(theta):
    """Return (grad, H) from quadratic fit coefficients."""
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
    Split H into ((lambda_perp, v_perp), (lambda_along, v_along)) with
    lambda_perp > 0 across the trench and lambda_along < 0 along it.
    Returns None if H is not a saddle.
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
    Plain Newton step -H^{-1} grad, scaled by step_size. Converges in a
    straight line but does not collapse onto the trench first.
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
    Per-component sign of the Newton step (first-order sliding mode in
    x/y). Reaches the trench quickly but chatters once on it.
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
    Newton step in the trench eigenbasis with separate gains, a large
    gain_perp for a fast snap onto the trench and a small gain_along for a
    gentle climb to the saddle. No chatter, and eigenvector sign flips are
    harmless. Falls back to newton_step if H is not a saddle.
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
    Saturated reaching law in the trench eigenbasis. Across the trench,
    tanh gives a bounded constant-speed reach far out and exponential decay
    inside |s| < boundary. Along the trench, a gentle Newton step. Falls
    back to newton_step if H is not a saddle.
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
    Self-tuning eigenstep with one parameter v_max. For each eigendirection

      c_i = -(v_i . grad) / lambda_i,   delta += v_max tanh(c_i / v_max) v_i

    Far out the step saturates at v_max. Near the saddle tanh is linear and
    the step recovers Newton. v_max sets both the reach speed and the
    boundary-layer width. Falls back to newton_step(step_size=v_max) if H
    is not a saddle.
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
    adaptive_eigen_step with |lambda_i| in the denominator (Dauphin-style).
    Under g ~ H r this gives c_perp ~ -r_perp, a snap onto the trench, and
    c_along ~ +r_along, descent along the trench away from the saddle. On
    fields whose trench has no finite minimum (the bimodal Gaussian in
    main_mod_dauph.py) the team descends indefinitely at a speed bounded by
    v_max. Falls back to newton_step(step_size=v_max) if H is not a saddle.
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
# Ported from separatrix_interactive_v6r.ipynb (HybridStepperBase,
# LogicCStepper). Two quadratics, one for u and one for v, give the
# Jacobian, and det(J) is a scalar field whose trench is the separatrix.
# The local flow chooses between Logic A (signed lambda, attract toward the
# trench saddle) and Logic B (|lambda|, descend along the trench), with a
# FLOW-band override near det(J) = 0.
# -----------------------------------------------------------------------


def _sample_vector_at_robots(cluster):
    """
    Sample (u, v) at each robot. Returns (u_arr, v_arr), each (6,).

    Two noise hooks (Paper_Draft_2A Sec. VI), both 0 by default.
    cluster.position_noise_std perturbs each robot's (x, y) before the field
    query, while the fit uses the nominal positions and the true state is
    unchanged. cluster.measurement_noise_std adds Gaussian noise to each u
    and v reading, the same model as the archived 3-robot field_types.py
    and Michini et al. 2014 (T-RO), Eq. (6).
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
    """Fit _quadratic_basis to u and v separately. Returns (theta_u, theta_v)."""
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


def separatrix_logic_c_step(cluster, v_max=0.04, eps_raw=1e-3, eps_dim=0.025,
                            guard='sign'):
    """
    Logic C, the flow-projected-on-trench selector for vector fields.

    Inside the FLOW band (det(J) ~ 0) the team follows the local flow along
    the trench and snaps across it. Outside, the flow at the centroid is
    projected on the along-trench eigenvector of H_det. Flow along descent
    selects Logic B (slide along the trench), and flow against it selects
    Logic A (attract toward the saddle).

    Args:
        v_max:    per-direction saturation speed (m/s)
        eps_raw:  raw det(J) threshold for the FLOW band
        eps_dim:  det(J)/||H_det||_F threshold for the FLOW band
        guard:    'sign' (default, every published number) or
                  'transverse' (EXPERIMENTAL, NOT WORKING, see
                  revision/items.yaml B1). 'sign' requires the fitted H_det
                  to be indefinite. On the double gyre the true H_det is
                  positive definite for |y| > 0.25, so the law traverses
                  there only because truncation makes the fitted H_det
                  traceless. 'transverse' picks the transverse eigenvector
                  by the larger |grad(D) . w| instead. On the clean six-start
                  benchmark (2026-08-04) it traversed 0 of 6 against 6 of 6
                  for 'sign', even after r_perp was fixed to use |lambda|.
                  Some other step (likely the v_neg sign stabilization or
                  the SLIDE/ATTRACT split) still assumes i_pos carries the
                  positive eigenvalue. The resolution was to keep 'sign' and
                  qualify Theorem 1 in the text.
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

    # Per-step mode log, recorded only if the cluster has a `diagnostics` list.
    def _log(mode):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            lam_d = np.linalg.eigvalsh(H_det)
            diag.append({
                'mode': mode, 'det': float(det_val),
                'det_ratio': float(abs(det_val) / max(H_norm, 1e-12)),
                'lam1': float(lam_d[0]), 'lam2': float(lam_d[1]),
            })

    def _roles(lam, V):
        """
        Return (i_tan, i_trans) for guard='transverse', taking the
        transverse eigenvector as the one with larger |grad(D) . w|.
        Returns None for guard='sign' (callers use eigenvalue sign) and
        'degenerate' when grad(D) vanishes or the eigenvalues coincide.
        """
        if guard == 'sign':
            return None
        proj = np.array([abs(float(grad_det @ V[:, 0])), abs(float(grad_det @ V[:, 1]))])
        if np.max(proj) < eps or abs(lam[0] - lam[1]) < eps:
            return 'degenerate'
        i_trans = 0 if proj[0] > proj[1] else 1
        i_tan = 1 - i_trans
        return i_tan, i_trans

    if on_raw or on_dim:
        # FLOW step: follow local field along trench, snap perp to trench.
        f = np.array([theta_u[0], theta_v[0]])
        lam, V = np.linalg.eigh(H_det)
        roles = _roles(lam, V)
        if roles == 'degenerate' or (roles is None and (
                lam[0] * lam[1] >= -eps or np.min(np.abs(lam)) < eps)):
            # Degenerate H_det: plain saturated flow step.
            _log('FLOW_DRIFT')
            n = np.linalg.norm(f)
            if n < 1e-12:
                return 0.0, 0.0
            scale = v_max * np.tanh(n / v_max) / n
            return float(f[0] * scale), float(f[1] * scale)
        if roles is None:
            i_neg = 0 if lam[0] < lam[1] else 1
            i_pos = 1 - i_neg
            lam_pos_restore = lam[i_pos]  # positive, guard checked the sign
        else:
            i_neg, i_pos = roles  # tangent, transverse
            # lam[i_pos] may be negative here, but the transverse channel
            # must restore toward the trench, so use the magnitude.
            lam_pos_restore = abs(lam[i_pos])
        v_neg = V[:, i_neg]
        v_pos = V[:, i_pos]
        c_along = float(f @ v_neg)
        r_perp  = float(grad_det @ v_pos) / lam_pos_restore
        c_perp  = -r_perp
        s_along = v_max * np.tanh(c_along / v_max)
        s_perp  = v_max * np.tanh(c_perp  / v_max)
        delta = s_along * v_neg + s_perp * v_pos
        _log('FLOW')
        return float(delta[0]), float(delta[1])

    # -- A/B selector ------------------------------------------------------
    lam, V = np.linalg.eigh(H_det)
    roles = _roles(lam, V)

    if roles == 'degenerate' or (roles is None and lam[0] * lam[1] >= -eps):
        # Not a saddle of det(J): fall back to Logic A.
        _log('ATTRACT_FALLBACK')
        c0 = -(V[:, 0] @ grad_det) / (lam[0] if abs(lam[0]) > eps else eps)
        c1 = -(V[:, 1] @ grad_det) / (lam[1] if abs(lam[1]) > eps else eps)
        delta = (v_max * np.tanh(c0 / v_max) * V[:, 0] +
                 v_max * np.tanh(c1 / v_max) * V[:, 1])
        return float(delta[0]), float(delta[1])

    if roles is None:
        i_neg = 0 if lam[0] < lam[1] else 1
    else:
        i_neg = roles[0]  # tangent
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
    Logic C plus an estimator-aware PARK mode (2026-07-02). Kept separate so
    separatrix_logic_c_step stays unchanged as the revert path.

    The eigenvalue-based capture test of Paper_Draft_2A (Secs. III-E, VII)
    never fires in practice because the fitted H_D near a det(J) well comes
    out indefinite and roughly 10x too small. PARK uses only det(J) and its
    gradient, which the estimator delivers reliably. When det(J) < d_capture
    it commands componentwise saturated gradient descent on det(J), which
    settles at the well minimum, the flow saddle ending the separatrix.

    d_capture = None (default) never parks and is trajectory-exact with
    Logic C (experiments/park_vs_traverse_demo.py). A finite value, e.g.
    -0.5 for the A = 0.1 double gyre (about half the well depth pi^4 A^2),
    parks at the terminal saddle. Everything below PARK is verbatim Logic C
    on the same single fit. Other args as in separatrix_logic_c_step.
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
    """Unit perpendicular to `direction` with non-negative dot on `reference`."""
    perp = np.array([-direction[1], direction[0]])
    if np.dot(perp, reference) < 0:
        perp = -perp
    return perp


def _estimate_det_and_grad_pentagon(cluster):
    """
    Run the Primitive 7 fit. Returns (det_val, grad_det, H_det, flow,
    theta_u, theta_v) at the centroid, with flow = (theta_u[0], theta_v[0]).
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
    Heuristic Logic G from separatrix_interactive_v002. Drives the centroid
    onto the det(J) = 0 Okubo-Weiss boundary at speed step_size, heading
    toward det(J) = 0 and blending in the flow by correction_weight in
    [0, 1] (0 is pure gradient, the v002 default). When that heading opposes
    the flow it moves perpendicular to it instead. Fixed orientation.
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
    Newton-step contour tracker. Drives det(J) -> 0 with a Newton step
    across the contour and slides along the exact level-set tangent
    perp(grad_D), oriented with the flow. Each channel saturates as
    v_max tanh(c / v_max). No heuristic flow blending, fixed orientation.

      dp_perp = -(D / ||grad_D||^2) grad_D   (Halley step if use_halley)
      t_hat   = rot90(grad_D) / ||grad_D||

    Drifts with the flow when ||grad_D|| < eps_grad.

    History (2026-07-02). An earlier version took the tangent from the H_det
    eigenvector with the smallest |eigenvalue|. On the double gyre (clean
    and noisy) and the 2 km Santa Barbara HFR field, perp(grad_D) tracked
    2x to 10^4x tighter and circulated farther. The eigenvector is also
    undefined where the eigenvalue magnitudes coincide, which happens on the
    double-gyre diamond.
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
# Tracks attracting OECSs (Serra and Haller 2016), operationalized as TRAPs
# (Serra et al. 2020), using the rate-of-strain tensor S = (J + J^T)/2.
# S is objective (S -> Q S Q^T in a rotating frame), while det(J),
# vorticity, and Okubo-Weiss are only Galilean invariant, so a rotating
# frame can fool Logic C but not this primitive. The attracting curve runs
# along a trench of s1, the smaller eigenvalue of S, with tangent e2 and
# normal e1. The TRAP core is the s1 minimum.
#
# Structural limit (2026-07-09). Under the quadratic model S is affine in
# position, so the fitted s1 = mu - ||(a, b)|| is concave for every field,
# with Hessian ~0 on the double gyre. The trench's transverse curvature
# lives in third derivatives, so a Newton step on H_s1 is impossible in
# principle. The frame comes from S's eigenvectors instead (noise gain
# (2/sqrt(10)) sigma/rho, one rung better than Logic C's H_D tangent), and
# transverse regulation is saturated gradient descent of s1 along e1.
# -----------------------------------------------------------------------


def _strain_quantities(theta_u, theta_v):
    """
    Rate-of-strain quantities at the centroid. With mu = (u_x + v_y)/2,
    a = (u_x - v_y)/2, b = (u_y + v_x)/2, and r = sqrt(a^2 + b^2), the
    eigenvalues of S are s1 = mu - r and s2 = mu + r, and under the
    quadratic model grad s1 = grad mu - (a grad a + b grad b)/r exactly.

    Returns (s1, grad_s1, e1, e2, r). e1 (compression, normal to the
    attracting curve) and e2 (stretching, tangent to it) have arbitrary
    sign. Small r means S is nearly isotropic and the eigenframe unreliable.
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
    Objective attracting-OECS/TRAP tracker (Primitive 10). Modes, checked
    in order each cycle:

      ACQUIRE  before first contact, or r < eps_degen. Saturated gradient
               descent on s1 toward stronger attraction.
      TRIM     after contact, s1 > -s_trim. Attraction no longer dominates
               (the Serra-Haller termination rule), so hold position.
      PARK     s_capture set and s1 < s_capture. Descend s1 to the core.
      TRACK    descend s1 along e1 (snap onto the trench) and drive along
               e2. Core-seek (s_capture set) descends s1 along the curve.
               Ride cruises at constant speed with its sign fixed by
               continuity, seeded once from the flow, since an eigenvector
               field has no global sign.

    Commands use only s1, grad s1, e1, and e2, so the team tracks the same
    material structure in any rotating frame. s_capture = None rides the
    structure and trims at its end. A finite value (e.g. -0.9 for the
    A = 0.1 double gyre, whose cores sit at s1 = -pi^2 A ~ -0.987) parks at
    the TRAP core.

    Per-run state (_oecs_prev_tangent, _oecs_banded) lives on the cluster
    and is cleared by building a fresh PentagonCluster per run.

    Args:
        v_max:      per-direction saturation speed (m/s)
        g_perp:     gradient gain on both s1-descent channels
        s_trim:     attraction-dominance threshold (s1 > -s_trim => TRIM)
        s_capture:  TRAP-core park threshold on s1; None = ride mode
        eps_degen:  r below which the eigenframe is treated as degenerate
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
        # Saturate the magnitude, not each component. Per-component tanh
        # depends on the axes (up to 39 deg of direction skew and a sqrt(2)
        # speed spread) and would break objectivity. This form is exactly
        # rotation equivariant. Primitive 11 uses the same fix.
        a = -g_perp * grad_s1
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return 0.0, 0.0
        scale = v_max * np.tanh(n / v_max) / n
        return float(a[0] * scale), float(a[1] * scale)

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
# The S-based analogue of separatrix_logic_c_step. It rides the full
# separatrix network, both halves and through the saddles, using only s1,
# grad s1, and the eigenframe (e1, e2). Primitive 10 stops on one
# attracting segment, and this one keeps going as Logic C does.
#
# Geometry (double gyre, A = 0.1, 2026-07-21, unit-tested). s1 has a
# transverse trench along the whole separatrix, on the attracting and
# repelling halves alike (d^2 s1/dx^2 = +8.68, +6.89, +3.01 at
# |y| = 0.35, 0.25, 0.10). What alternates is which eigenvector is tangent,
# e2 on the upper half and e1 on the lower. They swap at the origin, an
# isotropic point (S = 0) where neither is defined. The network is also a
# grid, since the walls y = +/-0.5 are s1 trenches crossing the separatrix
# at the saddles.
#
# Tangent rule. Away from a stationary point, grad s1 lies almost along the
# trench, so the tangent is the eigenvector with the larger |grad s1 . e_i|
# and the transverse direction is the other. This resolves both the origin
# swap and the wall-vs-separatrix choice at the saddles. Two earlier rules
# failed. Continuity alone could turn onto a wall at a saddle and never
# come back. Best alignment with the flow worked in the inertial frame but
# failed at Omega = 0.2, where the frame's swirl term, which grows with
# position, flipped the preferred eigenvector a full trench-length from any
# saddle.
#
# Two modes (2026-10-08 rewrite). TRACK rides the trench with the tangent
# rule above. CAPTURE holds at a 2D minimum of s1, entered when s1 changes by
# less than a fraction c_in across the footprint, q = rho |grad s1| / |s1|.
# q is dimensionless, so c_in needs no knowledge of the field's strain rate.
# q is large on the ride, at the gyre centres, and at the isotropic origin
# (s1 ~ 0 there), and small only near a saddle. CAPTURE never releases.
# Under noise q often dips below c_in before the saddle, and the hold, a
# descent of s1, then slides the team down the trench to the saddle anyway.
# A release on q flickered on noise (grad s1 comes from second-order
# coefficients) and a zero-velocity hold stopped short, so both were
# rejected (experiments/primitive11_simplification/README.md).
#
# Objectivity. q, the descent, and the TRACK tangent after the first step use
# only s1, grad s1, and the co-rotating eigenvectors, with continuity fixing
# a sign. The measured flow enters once, as the sign of the first step's
# tangent, so the controller is frame-objective past that seed step
# (closed-loop check in experiments/traverse_objectivity_demo.py). Taking the
# first tangent along the flow instead matched this under noise on the double
# gyre but sent the Santa Barbara run north, so it was not adopted.
# -----------------------------------------------------------------------


LEGACY_THRESHOLDS = ('s_trim', 'r_band', 'g_capture', 's_capture')


def oecs_separatrix_step(cluster, v_max=0.04, g_perp=1.0, c_in=0.03,
                         **legacy):
    """
    Objective separatrix traverser (Primitive 11). Two modes, checked in
    order each cycle:

      CAPTURE  once q = rho |grad s1| / |s1| < c_in with s1 < 0, descend s1
               for the rest of the run (latched, never released).
      TRACK    tangent t is whichever of e1, e2 has the larger
               |grad s1 . e_i|, signed by the previous t (by the measured
               flow on the first step). Cruise along t and descend the part
               of grad s1 orthogonal to it.

    rho is the formation's mean ring radius, measured each step. Per-run
    state (_oecs_prev_tangent, _oecs_captured) lives on the cluster and is
    cleared by building a fresh PentagonCluster per run.

    The thresholds of the previous version (s_trim, r_band, g_capture,
    s_capture) are accepted and ignored so existing callers still run. Its
    diagnostic arguments (override_seed, override_argmax, margin_gate,
    sigma_eff, rho, capture) are rejected.

    Args:
        v_max:   per-direction saturation speed (m/s)
        g_perp:  gradient gain on the transverse and CAPTURE channels
        c_in:    CAPTURE threshold on q (dimensionless)
    """
    unknown = set(legacy) - set(LEGACY_THRESHOLDS)
    if unknown:
        raise TypeError(f"oecs_separatrix_step no longer accepts {sorted(unknown)}")

    # -- Estimation (same single fit per step as Logic C and Primitive 10) --
    u_arr, v_arr = _sample_vector_at_robots(cluster)
    rel_pos = _get_relative_positions(cluster)
    theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)

    s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
    flow = np.array([theta_u[0], theta_v[0]])
    rho = float(np.mean(np.sort(np.linalg.norm(rel_pos, axis=1))[1:]))
    q = rho * float(np.linalg.norm(grad_s1)) / max(abs(s1), 1e-12)

    def _log(mode, **extra):
        diag = getattr(cluster, 'diagnostics', None)
        if diag is not None:
            entry = {'mode': mode, 's1': float(s1), 'r': float(r), 'q': q}
            entry.update(extra)
            diag.append(entry)

    # -- 1. CAPTURE: latched descent of s1 at a 2D minimum -----------------
    if getattr(cluster, '_oecs_captured', False) or (s1 < 0 and q < c_in):
        cluster._oecs_captured = True
        _log('CAPTURE')
        # Magnitude saturation, rotation equivariant (see Primitive 10). It is
        # the form of the paper's PARK Lyapunov result,
        #     s1_dot = -c_max ||grad s1|| tanh(g_perp ||grad s1|| / c_max).
        a = -g_perp * grad_s1
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return 0.0, 0.0
        scale = v_max * np.tanh(n / v_max) / n
        return float(a[0] * scale), float(a[1] * scale)

    # -- 2. TRACK: ride the s1 trench ---------------------------------------
    d1 = abs(float(grad_s1 @ e1))
    d2 = abs(float(grad_s1 @ e2))
    t_prev = getattr(cluster, '_oecs_prev_tangent', None)
    # The first step has no previous tangent and signs against the measured
    # flow, the one non-objective input. It fixes the ride direction once.
    sign_ref = t_prev if t_prev is not None else flow
    t_raw = e1 if d1 > d2 else e2
    t_hat = t_raw if float(t_raw @ sign_ref) >= 0 else -t_raw

    # Transverse: descend s1 orthogonal to the tangent rather than along a
    # fixed eigenvector, so one law holds whether the tangent is e1 or e2.
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
    _log('TRACK', tangent_id=('e1' if d1 > d2 else 'e2'))
    delta = step_perp + step_tan
    return float(delta[0]), float(delta[1])


# -----------------------------------------------------------------------
# Previous Primitive 11 (five modes: ACQUIRE, PARK, CAPTURE, CROSS, TRACK),
# replaced 2026-10-08 and kept commented out for reference.
# -----------------------------------------------------------------------
#
# GAMMA2 = 8.0 / np.sqrt(10.0)  # eq:gain_ladder noise gain on the mixed a4/b4-style coefficient
#
#
# def oecs_separatrix_step(cluster, v_max=0.04, g_perp=1.0, s_trim=0.05,
#                          r_band=0.05, g_capture=0.15, s_capture=None,
#                          override_seed=None, override_argmax=None,
#                          margin_gate=False, sigma_eff=None, rho=None,
#                          capture=True):
#     """
#     Objective separatrix traverser (Primitive 11). Modes, checked in order
#     each cycle:
#
#       ACQUIRE  before first contact. Saturated gradient descent on s1
#                homes onto the trench from anywhere in the basin.
#       PARK     s_capture set and s1 < s_capture. Descend s1 to the well.
#       CAPTURE  |grad s1| < g_capture, r > r_band, and s1 < -4 s_trim. Hold
#                at a 2D minimum of s1 (a flow saddle) by descending s1,
#                latched until s1 > -s_trim. Unlike PARK it needs no
#                field-specific threshold.
#       CROSS    r < r_band with no previous tangent, reachable only on the
#                first tracking step. Rides the flow, the one non-objective
#                input.
#       TRACK    the tangent is whichever of e1, e2 has the larger
#                |grad s1 . e_i|, signed by continuity. The transverse
#                channel descends the part of grad s1 orthogonal to it,
#                which holds the trench on both halves.
#
#     Per-run state (_oecs_prev_tangent, _oecs_banded, _oecs_captured) lives
#     on the cluster and is cleared by building a fresh PentagonCluster per
#     run. The diagnostic arguments default to no effect.
#
#     Args:
#         v_max:      per-direction saturation speed (m/s)
#         g_perp:     gradient gain on the transverse, CAPTURE, and ACQUIRE
#                     channels
#         s_trim:     latch depth, called s_latch in the Separatrix paper
#                     (Draft_11.tex, Table I and Sec. III-C). The name comes
#                     from Primitive 10's TRIM mode, which this primitive
#                     lacks. It latches the ride (s1 < -s_trim), gates CAPTURE
#                     (s1 < -4 s_trim), and releases it (s1 > -s_trim).
#         r_band:     r below which the eigenframe is unreliable (CROSS)
#         g_capture:  |grad s1| below which CAPTURE holds
#         s_capture:  optional PARK threshold on s1; None relies on CAPTURE
#         override_seed:    diagnostic. A (2,) vector that replaces the flow
#                     seed for this step, e.g. the noise-free flow, to isolate
#                     the seed channel's noise (Draft_6a sec:disc_noise).
#         override_argmax:  diagnostic. A (grad_s1, e1, e2) tuple that replaces
#                     the TRACK discriminator inputs for this step.
#         margin_gate:  if True (with sigma_eff, rho, and a previous tangent),
#                     TRACK keeps the previous tangent while |d1 - d2| is
#                     below the noise floor gamma_2 sigma_eff / rho^2
#                     (eq:gain_ladder).
#         sigma_eff:  measurement-noise std for margin_gate
#         rho:        formation ring radius for margin_gate
#         capture:    False skips CAPTURE entirely, giving a pure traverser
#                     (diagnostic only). The paper uses True.
#     """
#     # -- Estimation (same single fit per step as Logic C and Primitive 10) --
#     u_arr, v_arr = _sample_vector_at_robots(cluster)
#     rel_pos = _get_relative_positions(cluster)
#     theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_arr, v_arr)
#
#     s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
#     flow = np.array([theta_u[0], theta_v[0]])
#
#     if override_seed is not None:
#         flow = np.asarray(override_seed, dtype=float)
#     if override_argmax is not None:
#         grad_s1, e1, e2 = override_argmax
#
#     banded = getattr(cluster, '_oecs_banded', False)
#     if s1 < -s_trim and not banded:
#         banded = True
#         cluster._oecs_banded = True
#
#     def _log(mode, **extra):
#         diag = getattr(cluster, 'diagnostics', None)
#         if diag is not None:
#             entry = {'mode': mode, 's1': float(s1), 'r': float(r)}
#             entry.update(extra)
#             diag.append(entry)
#
#     def _descend_s1():
#         # Magnitude saturation, rotation equivariant (see Primitive 10). It is
#         # the form of the paper's PARK Lyapunov result,
#         #     s1_dot = -c_max ||grad s1|| tanh(g_perp ||grad s1|| / c_max),
#         # and caps the speed at v_max.
#         a = -g_perp * grad_s1
#         n = float(np.linalg.norm(a))
#         if n < 1e-12:
#             return 0.0, 0.0
#         scale = v_max * np.tanh(n / v_max) / n
#         return float(a[0] * scale), float(a[1] * scale)
#
#     # -- 1. ACQUIRE: no structure yet -----------------------------------
#     if not banded:
#         _log('ACQUIRE')
#         return _descend_s1()
#
#     # -- 2. PARK: optional core well, checked before CROSS/TRACK --------
#     if s_capture is not None and s1 < s_capture:
#         _log('PARK')
#         return _descend_s1()
#
#     # -- 3. CAPTURE: hold at a 2D stationary point of s1 --
#     # A flow saddle is a 2D minimum of s1 (true H_s1 eigenvalues both +9.74
#     # on the double gyre, s1 = -pi^2 A per eq. s1_analytic_main), so the
#     # full gradient vanishes there, while on a 1D trench it vanishes only
#     # across. The flow stagnates there too, so testing grad s1 avoids
#     # choosing a tangent where every cue is weak. The s1 < -4*s_trim gate
#     # rejects shallow flat spots near the gyre centers that pass the
#     # gradient test under noise (s1 in [-0.06, +0.04] there, -0.76 to -1.16
#     # at the saddle).
#     #
#     # The latch bridges noise flicker of |grad s1| above g_capture, which
#     # otherwise re-entered TRACK and slid onto the wall at sigma_p = 0.005.
#     # It releases when s1 > -s_trim, the same test as `banded`. A latch that
#     # never released walked the team past +/-2 domain widths within 1000
#     # steps. While s1 still reads as a well, the PARK Lyapunov argument
#     # (Section stability) keeps the descent from running away.
#     captured = getattr(cluster, '_oecs_captured', False) if capture else False
#     if captured and s1 > -s_trim:
#         captured = False
#         cluster._oecs_captured = False
#     g_norm = float(np.linalg.norm(grad_s1))
#     if capture and (captured or (r > r_band and g_norm < g_capture
#                                  and s1 < -4.0 * s_trim)):
#         cluster._oecs_captured = True
#         _log('CAPTURE')
#         return _descend_s1()
#
#     t_prev = getattr(cluster, '_oecs_prev_tangent', None)
#     prev_hat = (t_prev / np.linalg.norm(t_prev)
#                if t_prev is not None and np.linalg.norm(t_prev) > 1e-12
#                else None)
#
#     # -- 4. CROSS: eigenframe degenerate AND no continuity to fall back on
#     # Rides the flow, the only input not built from s1. Essentially
#     # unreachable on the double gyre once ACQUIRE has landed on the trench.
#     if r < r_band and prev_hat is None:
#         _log('CROSS')
#         fn = float(np.linalg.norm(flow))
#         t_hat = flow / fn if fn > 1e-9 else np.array([1.0, 0.0])
#         step_tan = v_max * np.tanh(1.0) * t_hat
#         step_perp_x, step_perp_y = _descend_s1()
#         step_perp = np.array([step_perp_x, step_perp_y])
#         cluster._oecs_prev_tangent = t_hat
#         delta = step_perp + step_tan
#         return float(delta[0]), float(delta[1])
#
#     # -- 5. TRACK: ride the s1 trench, tangent chosen WITHOUT the flow ---
#     # Tangent rule as in the block comment above. Continuity (or the flow,
#     # before any tangent exists) supplies only the sign.
#     d1 = abs(float(grad_s1 @ e1))
#     d2 = abs(float(grad_s1 @ e2))
#     margin_held = False
#     if (margin_gate and sigma_eff is not None and rho is not None
#             and prev_hat is not None):
#         threshold = GAMMA2 * sigma_eff / rho**2
#         if abs(d1 - d2) < threshold:
#             t_hat = prev_hat
#             margin_held = True
#     if not margin_held:
#         t_raw = e1 if d1 > d2 else e2
#         sign_ref = prev_hat if prev_hat is not None else flow
#         t_hat = t_raw if float(t_raw @ sign_ref) >= 0 else -t_raw
#
#     # Transverse: descend s1 orthogonal to the tangent rather than along a
#     # fixed eigenvector, so one law holds whether the tangent is e1 or e2.
#     g_perp_vec = grad_s1 - float(grad_s1 @ t_hat) * t_hat
#     g_perp_norm = float(np.linalg.norm(g_perp_vec))
#     if g_perp_norm < 1e-12:
#         step_perp = np.zeros(2)
#     else:
#         n_hat = g_perp_vec / g_perp_norm
#         c_perp = -g_perp * g_perp_norm
#         step_perp = v_max * np.tanh(c_perp / v_max) * n_hat
#
#     step_tan = v_max * np.tanh(1.0) * t_hat
#
#     cluster._oecs_prev_tangent = t_hat
#     _log('TRACK', tangent_id=('e1' if d1 > d2 else 'e2'))
#     delta = step_perp + step_tan
#     return float(delta[0]), float(delta[1])
