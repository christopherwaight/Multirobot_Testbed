"""
Randomized scalar fields with closed-form saddle points.

Eight families.  Each one returns a SaddleField carrying the field callable,
the exact saddle location, and the exact Hessian at that saddle.  The RL
reward needs the true saddle, so "closed form" is a hard requirement here,
not a convenience: a numerically-located saddle would put its own error floor
underneath every number the experiment produces.

    1. quadratic            pure rotated saddle, the estimator's best case
    2. log_sum_exp          the field from saddle_point_4_robot.ipynb, generalized
    3. gaussian_pair        col between two hills, gradient dies off far away
    4. cubic_perturbed      family 1 plus a cubic that vanishes to 2nd order at s
    5. streamfunction_quad  checkerboard 4-Gaussian, Hessian exactly traceless
    6. double_gyre_psi      periodic streamfunction, extrema inside the domain
    7. quartic_wells        family 1 plus a quartic double-well on the weak axis,
                             producing two genuine extra minima at a KNOWN offset
    8. rational_envelope    quadratic saddle under a Lorentzian amplitude decay;
                             far field flattens and folds back, producing
                             secondary critical points beyond the saddle

Why families 5 and 6 both give traceless Hessians on purpose: the 4-robot
C(4,3) plane-fit estimator can only ever report a traceless Hessian (see
README).  On these two families that structural bias is exactly right, so they
isolate how much of the difficulty is the missing trace as opposed to the
cos(2*theta) sensing blind spot.

Families 7 and 8 are a second, DIFFERENT axis of stress: instead of asking
whether the estimator can see the saddle at all, they place genuine extra
critical points at an EXACTLY KNOWN offset from it, of two different
mechanisms (polynomial confinement vs. multiplicative envelope decay) so a
policy that arrives at the wrong stationary point can be diagnosed rather than
just observed.  See saddle_point_4robot.ipynb.eval note: two families
(gaussian_pair, streamfunction_quad) are hard enough that even the
best-tuned hand-derived controller fails on them (median final distance ~3,
domain-exit range) -- not a PPO-specific weakness, an information limit of
the 4-robot single-snapshot estimator on those particular shapes.  Accepted
as a known limitation rather than something either controller is expected to
solve; see README, "Known limitation" section.

Geometry convention.  The domain is a box of half-width `domain_half` centred
on the saddle, and episode starts are drawn from an annulus around it.  The
policy never observes absolute position (the observation is built from
robot-relative geometry only), so centring the domain on the saddle leaks
nothing; it just keeps the out-of-bounds test and the start distribution
consistent across families.

Amplitude convention.  Every field is rescaled so the RMS gradient magnitude on
a ring of radius `GRAD_REF_RADIUS` around the saddle equals a randomized
`grad_scale`.  Without this the six families differ in magnitude by orders of
magnitude and the randomization is swamped by that rather than by the shape
differences we actually care about.
"""
import numpy as np

# --------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------

DOMAIN_HALF      = 3.0    # half-width of the square domain, metres
START_R_MIN      = 1.0    # episode start annulus, inner radius
START_R_MAX      = 2.5    # episode start annulus, outer radius
GRAD_REF_RADIUS  = 1.5    # radius at which grad_scale is enforced
GRAD_SCALE_RANGE = (0.5, 2.0)

# Furthest a robot can be from the saddle while still inside the domain.
DOMAIN_REACH = DOMAIN_HALF * np.sqrt(2.0)

FAMILY_NAMES = [
    "quadratic",
    "log_sum_exp",
    "gaussian_pair",
    "cubic_perturbed",
    "streamfunction_quad",
    "double_gyre_psi",
    "quartic_wells",
    "rational_envelope",
]


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

class SaddleField:
    """A scalar field with a known saddle.

    Attributes:
        phi:      callable (x, y) -> float
        saddle:   (2,) array, exact saddle location
        hess:     (2, 2) array, exact Hessian at the saddle
        family:   str, which generator produced it
        params:   dict of the sampled parameters, for logging and figures
    """

    __slots__ = ("phi", "saddle", "hess", "family", "params")

    def __init__(self, phi, saddle, hess, family, params):
        self.phi = phi
        self.saddle = np.asarray(saddle, dtype=float)
        self.hess = np.asarray(hess, dtype=float)
        self.family = family
        self.params = params

    # -- convenience -------------------------------------------------------

    @property
    def eigvals(self):
        return np.linalg.eigvalsh(self.hess)

    @property
    def anisotropy(self):
        """|lambda_plus / lambda_minus|, how lopsided the saddle is."""
        lam = np.sort(self.eigvals)
        return abs(lam[1] / lam[0]) if abs(lam[0]) > 1e-12 else np.inf

    def domain_bounds(self):
        """((xmin, xmax), (ymin, ymax)) of the square domain."""
        sx, sy = self.saddle
        return ((sx - DOMAIN_HALF, sx + DOMAIN_HALF),
                (sy - DOMAIN_HALF, sy + DOMAIN_HALF))

    def sample_start(self, rng):
        """Draw an episode start from the annulus around the saddle."""
        r = rng.uniform(START_R_MIN, START_R_MAX)
        a = rng.uniform(0.0, 2.0 * np.pi)
        return self.saddle + r * np.array([np.cos(a), np.sin(a)])

    def __repr__(self):
        lam = np.sort(self.eigvals)
        return (f"SaddleField({self.family}, s=({self.saddle[0]:+.3f}, "
                f"{self.saddle[1]:+.3f}), eig=({lam[0]:+.3f}, {lam[1]:+.3f}))")


# --------------------------------------------------------------------------
# Numerical helpers, used for self-checks and for amplitude normalization
# --------------------------------------------------------------------------

def fd_gradient(phi, x, y, h=1e-5):
    return np.array([(phi(x + h, y) - phi(x - h, y)) / (2 * h),
                     (phi(x, y + h) - phi(x, y - h)) / (2 * h)])


def fd_hessian(phi, x, y, h=1e-4):
    fxx = (phi(x + h, y) - 2 * phi(x, y) + phi(x - h, y)) / h ** 2
    fyy = (phi(x, y + h) - 2 * phi(x, y) + phi(x, y - h)) / h ** 2
    fxy = (phi(x + h, y + h) - phi(x + h, y - h)
           - phi(x - h, y + h) + phi(x - h, y - h)) / (4 * h ** 2)
    return np.array([[fxx, fxy], [fxy, fyy]])


def _rms_ring_gradient(phi, saddle, radius=GRAD_REF_RADIUS, n=32):
    """RMS |grad phi| on a ring around the saddle. Sets the amplitude scale."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = saddle[None, :] + radius * np.column_stack([np.cos(a), np.sin(a)])
    mags = [np.linalg.norm(fd_gradient(phi, p[0], p[1])) for p in pts]
    return float(np.sqrt(np.mean(np.square(mags))))


def _rescale(phi_raw, hess_raw, saddle, grad_scale):
    """Scale the field so RMS ring gradient equals grad_scale.

    Scaling a scalar field by a constant scales its gradient and Hessian by the
    same constant, so the saddle location is untouched and the Hessian is just
    multiplied through.
    """
    rms = _rms_ring_gradient(phi_raw, saddle)
    if not np.isfinite(rms) or rms < 1e-12:
        k = 1.0
    else:
        k = grad_scale / rms
    return (lambda x, y: k * phi_raw(x, y)), hess_raw * k, k


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _sample_saddle(rng):
    """Saddle location. The domain follows it, so this only sets world coords."""
    return rng.uniform(-1.5, 1.5, size=2)


# --------------------------------------------------------------------------
# Family 1: pure quadratic saddle
# --------------------------------------------------------------------------

def make_quadratic(rng):
    """phi = 1/2 (r-s)' M (r-s),  M = R diag(l_pos, l_neg) R'.

    The estimator's best case: the field IS a quadratic, so the plane-fit
    gradient is exact and the recovered Hessian is exactly the deviatoric part
    of M, scaled by cos(2*theta_err).
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi)
    l_pos = rng.uniform(0.5, 3.0)
    l_neg = -rng.uniform(0.5, 3.0)
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    M = R @ np.diag([l_pos, l_neg]) @ R.T

    def phi_raw(x, y, s=s, M=M):
        dx, dy = x - s[0], y - s[1]
        return 0.5 * (M[0, 0] * dx * dx + 2.0 * M[0, 1] * dx * dy
                      + M[1, 1] * dy * dy)

    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "quadratic",
                       dict(theta=theta, l_pos=l_pos, l_neg=l_neg,
                            grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 2: log-sum-exp of two wells
# --------------------------------------------------------------------------

def make_log_sum_exp(rng):
    """phi = log(exp(g1) + exp(g2)),  g_i = -k/2 |r - c_i|^2.

    Generalization of the field in saddle_point_4_robot.ipynb (which is k=1,
    separation 2, saddle at the origin).  Wells sit symmetrically about s, so
    the softmax weights are both 1/2 there and the gradients cancel exactly.

    Hessian at s works out to  -k I + k^2 d^2 u u',  giving eigenvalue
    (k^2 d^2 - k) along the separation axis and -k across it.  Saddle requires
    k d^2 > 1, enforced below with margin.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    k_well = rng.uniform(0.6, 1.6)
    # k*d^2 > 1 is the saddle condition; take d comfortably past the boundary.
    d_min = np.sqrt(1.6 / k_well)
    d = rng.uniform(d_min, d_min + 1.4)
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    u = np.array([np.cos(theta), np.sin(theta)])
    c1 = s - d * u
    c2 = s + d * u

    def phi_raw(x, y, c1=c1, c2=c2, k=k_well):
        g1 = -0.5 * k * ((x - c1[0]) ** 2 + (y - c1[1]) ** 2)
        g2 = -0.5 * k * ((x - c2[0]) ** 2 + (y - c2[1]) ** 2)
        m = max(g1, g2)
        return m + np.log(np.exp(g1 - m) + np.exp(g2 - m))

    M = -k_well * np.eye(2) + (k_well ** 2) * (d ** 2) * np.outer(u, u)
    phi, hess, kk = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "log_sum_exp",
                       dict(theta=theta, k_well=k_well, d=d,
                            grad_scale=grad_scale, k=kk))


# --------------------------------------------------------------------------
# Family 3: col between two Gaussian hills
# --------------------------------------------------------------------------

def make_gaussian_pair(rng):
    """phi = A[G(r - c1) + G(r - c2)],  G Gaussian of width sigma.

    Saddle at the midpoint by symmetry.  Unlike family 2 the gradient decays to
    zero far from the hills, so the far field is nearly flat and a policy that
    relies on a strong gradient signal has nothing to work with out there.

    Hessian at s is  2 A G0 [d^2 u u'/sigma^4 - I/sigma^2],  a saddle when
    d > sigma.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    sigma = rng.uniform(0.6, 1.2)
    d = rng.uniform(1.25 * sigma, 2.2 * sigma)   # d > sigma for a saddle
    A = 1.0
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    u = np.array([np.cos(theta), np.sin(theta)])
    c1 = s - d * u
    c2 = s + d * u
    two_s2 = 2.0 * sigma ** 2

    def phi_raw(x, y, c1=c1, c2=c2, A=A, two_s2=two_s2):
        r1 = (x - c1[0]) ** 2 + (y - c1[1]) ** 2
        r2 = (x - c2[0]) ** 2 + (y - c2[1]) ** 2
        return A * (np.exp(-r1 / two_s2) + np.exp(-r2 / two_s2))

    G0 = np.exp(-d ** 2 / two_s2)
    M = 2.0 * A * G0 * (d ** 2 * np.outer(u, u) / sigma ** 4
                        - np.eye(2) / sigma ** 2)
    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "gaussian_pair",
                       dict(theta=theta, sigma=sigma, d=d,
                            grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 4: quadratic plus a homogeneous cubic
# --------------------------------------------------------------------------

def make_cubic_perturbed(rng):
    """phi = 1/2 d' M d + eps * p3(d),  d = r - s,  p3 homogeneous cubic.

    grad(p3) is homogeneous of degree 2 and Hess(p3) of degree 1, so both
    vanish at d = 0.  The saddle location AND the Hessian there are therefore
    exactly those of the quadratic part, while the field is decidedly not
    quadratic anywhere else.  This is the model-mismatch stressor.

    eps is bounded so the cubic's gradient stays below half the quadratic's at
    the far edge of the domain, which keeps spurious critical points out.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi)
    l_pos = rng.uniform(0.8, 3.0)
    l_neg = -rng.uniform(0.8, 3.0)
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    M = R @ np.diag([l_pos, l_neg]) @ R.T

    # Random unit-norm homogeneous cubic in (dx, dy).
    coef = rng.normal(size=4)
    coef /= np.linalg.norm(coef)

    lam_min = min(abs(l_pos), abs(l_neg))
    eps_max = 0.5 * lam_min / (3.0 * DOMAIN_REACH)
    eps = rng.uniform(0.3, 1.0) * eps_max

    def phi_raw(x, y, s=s, M=M, c=coef, eps=eps):
        dx, dy = x - s[0], y - s[1]
        quad = 0.5 * (M[0, 0] * dx * dx + 2.0 * M[0, 1] * dx * dy
                      + M[1, 1] * dy * dy)
        cubic = (c[0] * dx ** 3 + c[1] * dx * dx * dy
                 + c[2] * dx * dy * dy + c[3] * dy ** 3)
        return quad + eps * cubic

    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "cubic_perturbed",
                       dict(theta=theta, l_pos=l_pos, l_neg=l_neg, eps=eps,
                            cubic_coef=coef.tolist(),
                            grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 5: checkerboard four-Gaussian streamfunction
# --------------------------------------------------------------------------

def make_streamfunction_quad(rng):
    """psi = sum_i s_i G(r - c_i),  four centres, checkerboard signs (+,-,-,+).

    The sign pattern makes psi odd in each of the two configuration axes about
    the centre, so psi_xx = psi_yy = 0 there while psi_xy does not vanish.  The
    Hessian is exactly [[0, c], [c, 0]]: a genuinely traceless saddle with
    eigenvalues +/- |c| and eigenvectors at 45 degrees to the configuration axes.

    Off-diagonal magnitude at the centre, with corners at (+/-a, +/-a) in the
    rotated frame:  psi_xy = -4 (a^2 / sigma^4) exp(-a^2 / sigma^2).
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi / 2.0)
    a = rng.uniform(0.7, 1.4)          # corner half-offset along each axis
    sigma = rng.uniform(0.55, 1.0)
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    local = np.array([[-a, a], [a, a], [-a, -a], [a, -a]])   # TL TR BL BR
    signs = np.array([1.0, -1.0, -1.0, 1.0])                 # checkerboard
    centers = s[None, :] + local @ R.T
    two_s2 = 2.0 * sigma ** 2

    def phi_raw(x, y, centers=centers, signs=signs, two_s2=two_s2):
        total = 0.0
        for (cx, cy), sg in zip(centers, signs):
            total += sg * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / two_s2)
        return total

    psi_xy = -4.0 * (a ** 2 / sigma ** 4) * np.exp(-a ** 2 / sigma ** 2)
    M_local = np.array([[0.0, psi_xy], [psi_xy, 0.0]])
    M = R @ M_local @ R.T

    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "streamfunction_quad",
                       dict(theta=theta, a=a, sigma=sigma,
                            grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 6: periodic double-gyre streamfunction
# --------------------------------------------------------------------------

def make_double_gyre_psi(rng):
    """psi = A sin(pi xi / L) sin(pi eta / L),  (xi, eta) = R'(r - s).

    Critical points: saddles on the lattice spaced L apart (psi_xx = psi_yy = 0,
    psi_xy = A (pi/L)^2 nonzero), and extrema at the half-lattice.  Wavelength
    is held at L >= 5 so the nearest OTHER saddle sits at distance L, beyond the
    domain reach of 3*sqrt(2) = 4.24.  The nearest extrema, at L/sqrt(2), do
    fall inside the domain, which is the point: this family supplies genuine
    non-saddle critical points as distractors.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi / 2.0)
    L = rng.uniform(5.0, 8.0)
    A = 1.0
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    w = np.pi / L

    def phi_raw(x, y, s=s, R=R, A=A, w=w):
        dx, dy = x - s[0], y - s[1]
        xi = R[0, 0] * dx + R[1, 0] * dy     # R' @ d
        eta = R[0, 1] * dx + R[1, 1] * dy
        return A * np.sin(w * xi) * np.sin(w * eta)

    M_local = np.array([[0.0, A * w * w], [A * w * w, 0.0]])
    M = R @ M_local @ R.T

    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "double_gyre_psi",
                       dict(theta=theta, L=L, grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 7: quadratic saddle plus a quartic double-well on the weak axis
# --------------------------------------------------------------------------

def make_quartic_wells(rng):
    """phi = 1/2 d'Md + eps*(u2^2 - a^2)^2,  d = r - s,  u2 = weak-axis coord.

    The quartic term has no linear-in-u2 part (it is even, minimum at
    u2=+/-a), so its gradient vanishes at d=0 regardless of homogeneity, and
    its only contribution to the Hessian at d=0 is a single scalar on the
    weak-axis entry: Hess(quartic term)(0) = diag(0, -4*eps*a^2) in local
    (u1,u2) coordinates (verified symbolically).  The saddle location is
    therefore exact and the saddle's Hessian is exactly the base M with that
    one known shift folded in, both closed form, no approximation.

    At u2 = +/-a (world coords s +/- a * v_weak, v_weak the weak eigenvector)
    the field has two further exact stationary points.  There,
    Hess(quartic)(0,+/-a) = diag(0, +8*eps*a^2), so the full local Hessian is
    diag(l_pos, 8*eps*a^2): both positive, genuine local MINIMA, at a location
    known in closed form.  These are the "wrong answer" a policy can converge
    to, analogous to the wells in gaussian_pair but built from a completely
    different, purely polynomial mechanism.

    eps is bounded so the weak-axis correction stays a fraction of l_neg,
    which keeps det(M_total) < 0 (a genuine saddle) comfortably satisfied.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi)
    l_pos = rng.uniform(0.6, 3.0)
    l_neg = -rng.uniform(0.6, 3.0)
    a = rng.uniform(0.6, 1.6)          # offset of the extra wells, metres
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    eps = rng.uniform(0.1, 0.4) * abs(l_neg) / (4.0 * a * a)

    def phi_raw(x, y, s=s, R=R, l_pos=l_pos, l_neg=l_neg, a=a, eps=eps):
        dx, dy = x - s[0], y - s[1]
        u1 = R[0, 0] * dx + R[1, 0] * dy
        u2 = R[0, 1] * dx + R[1, 1] * dy
        quad = 0.5 * (l_pos * u1 * u1 + l_neg * u2 * u2)
        quartic = eps * (u2 * u2 - a * a) ** 2
        return quad + quartic

    M_local = np.diag([l_pos, l_neg - 4.0 * eps * a * a])
    M = R @ M_local @ R.T

    phi, hess, k = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "quartic_wells",
                       dict(theta=theta, l_pos=l_pos, l_neg=l_neg, a=a,
                            eps=eps, grad_scale=grad_scale, k=k))


# --------------------------------------------------------------------------
# Family 8: quadratic saddle under a Lorentzian (rational) amplitude decay
# --------------------------------------------------------------------------

def make_rational_envelope(rng):
    """phi = q(d) * w(d),  q(d) = 1/2 d'Md,  w(d) = 1 / (1 + k|d|^2).

    q vanishes to second order at d=0 and w(0)=1, so at the saddle:
    grad(phi)(0) = grad(q)(0)*w(0) = 0, and
    Hess(phi)(0)  = Hess(q)(0)*w(0) = M
    exactly (the cross terms and the q*Hess(w) term both carry a factor of
    q or grad(q), which are 0 at d=0; verified symbolically).

    Far from the saddle, w decays as 1/|d|^2 while q grows as |d|^2, so
    phi = q*w approaches a finite, direction-dependent limit
    (1/2 u'Mu)/k as |d| -> infinity (u = d/|d|).  Because M is indefinite,
    that limit is positive along the attracting axis and negative along the
    repelling one, which forces phi to be non-monotonic along most radial
    directions: it rises from 0, peaks or dips, then relaxes toward the
    asymptote.  That interior extremum is the "wrong answer" this family
    contributes, and unlike quartic_wells (built by direct polynomial
    construction) it comes from a multiplicative envelope, a mechanism nothing
    else in this module uses.
    """
    s = _sample_saddle(rng)
    theta = rng.uniform(0.0, np.pi)
    l_pos = rng.uniform(0.6, 3.0)
    l_neg = -rng.uniform(0.6, 3.0)
    k_decay = rng.uniform(0.15, 0.5)   # sets the envelope's decay radius
    grad_scale = rng.uniform(*GRAD_SCALE_RANGE)

    R = _rot(theta)
    M = R @ np.diag([l_pos, l_neg]) @ R.T

    def phi_raw(x, y, s=s, M=M, k=k_decay):
        dx, dy = x - s[0], y - s[1]
        q = 0.5 * (M[0, 0] * dx * dx + 2.0 * M[0, 1] * dx * dy
                  + M[1, 1] * dy * dy)
        w = 1.0 / (1.0 + k * (dx * dx + dy * dy))
        return q * w

    phi, hess, kk = _rescale(phi_raw, M, s, grad_scale)
    return SaddleField(phi, s, hess, "rational_envelope",
                       dict(theta=theta, l_pos=l_pos, l_neg=l_neg,
                            k_decay=k_decay, grad_scale=grad_scale, k=kk))


# --------------------------------------------------------------------------
# Registry and sampling
# --------------------------------------------------------------------------

FAMILIES = {
    "quadratic":           make_quadratic,
    "log_sum_exp":         make_log_sum_exp,
    "gaussian_pair":       make_gaussian_pair,
    "cubic_perturbed":     make_cubic_perturbed,
    "streamfunction_quad": make_streamfunction_quad,
    "double_gyre_psi":     make_double_gyre_psi,
    "quartic_wells":       make_quartic_wells,
    "rational_envelope":   make_rational_envelope,
}


def sample_field(rng, families=None):
    """Draw one random field.

    Args:
        rng:      np.random.Generator
        families: list of family names to draw from, or None for all six.

    Returns:
        SaddleField
    """
    names = list(FAMILIES) if families is None else list(families)
    name = names[rng.integers(len(names))]
    return FAMILIES[name](rng)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_field(fld, tol_grad=2e-4, tol_hess=2e-2):
    """Check the advertised saddle and Hessian against finite differences.

    Returns (ok, report_dict).  Raises nothing, so callers can aggregate.
    """
    sx, sy = fld.saddle
    g = fd_gradient(fld.phi, sx, sy)
    H = fd_hessian(fld.phi, sx, sy)

    g_err = float(np.linalg.norm(g))
    h_err = float(np.linalg.norm(H - fld.hess, "fro")
                  / max(np.linalg.norm(fld.hess, "fro"), 1e-12))
    det = float(np.linalg.det(fld.hess))

    ok = (g_err < tol_grad) and (h_err < tol_hess) and (det < 0.0)
    return ok, dict(grad_norm=g_err, hess_rel_err=h_err, det=det,
                    eig=np.sort(fld.eigvals).tolist())


def _import_time_self_check():
    """One draw per family, asserted at import.

    Mirrors the convention in
    Paper_Writing/Separatrix_and_OW_Paper/scripts/verify_estimator_bias.py,
    where the closed-form derivatives are asserted against central differences
    at import so a broken formula cannot silently reach a figure.  Costs well
    under a millisecond per family.
    """
    rng = np.random.default_rng(0)
    for name, maker in FAMILIES.items():
        fld = maker(rng)
        ok, rep = verify_field(fld)
        if not ok:
            raise AssertionError(
                f"saddle_fields: {name} failed its self-check: {rep}")


_import_time_self_check()


# --------------------------------------------------------------------------
# Standalone verification report
# --------------------------------------------------------------------------

def _scan_for_extra_critical_points(fld, n=61):
    """Count near-critical grid points that are not the advertised saddle.

    Coarse and deliberately cheap.  Used only by the __main__ report, never in
    the RL loop.  Flags a field whose domain contains other stationary points,
    which for the reward's purposes are distractors.
    """
    (x0, x1), (y0, y1) = fld.domain_bounds()
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    ref = _rms_ring_gradient(fld.phi, fld.saddle)
    thresh = 0.02 * max(ref, 1e-9)
    hits = []
    for x in xs:
        for y in ys:
            if np.linalg.norm(fd_gradient(fld.phi, x, y)) < thresh:
                if np.hypot(x - fld.saddle[0], y - fld.saddle[1]) > 0.4:
                    hits.append((x, y))
    return hits


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=200,
                   help="random draws per family for the statistical check")
    p.add_argument("--scan", action="store_true",
                   help="also grid-scan for extra critical points (slow)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print(f"saddle_fields verification, {args.draws} draws per family")
    print("=" * 78)
    print(f"{'family':22s} {'ok':>5s} {'max|grad(s)|':>13s} "
          f"{'max Hess rel err':>17s} {'anisotropy p50/p95':>20s}")
    print("-" * 78)

    all_ok = True
    for name, maker in FAMILIES.items():
        g_worst, h_worst, aniso = 0.0, 0.0, []
        n_ok = 0
        for _ in range(args.draws):
            fld = maker(rng)
            ok, rep = verify_field(fld)
            n_ok += int(ok)
            g_worst = max(g_worst, rep["grad_norm"])
            h_worst = max(h_worst, rep["hess_rel_err"])
            aniso.append(fld.anisotropy)
        ok_all = (n_ok == args.draws)
        all_ok &= ok_all
        a50, a95 = np.percentile(aniso, [50, 95])
        print(f"{name:22s} {str(ok_all):>5s} {g_worst:13.3e} "
              f"{h_worst:17.3e} {a50:9.2f} /{a95:8.2f}")

    print("-" * 78)
    print(f"ALL FAMILIES PASS: {all_ok}")

    if args.scan:
        print()
        print("=" * 78)
        print("Extra-critical-point scan (5 draws per family, 61x61 grid)")
        print("=" * 78)
        rng2 = np.random.default_rng(args.seed + 1)
        for name, maker in FAMILIES.items():
            counts = []
            for _ in range(5):
                counts.append(len(_scan_for_extra_critical_points(maker(rng2))))
            print(f"  {name:22s} extra near-critical grid points: {counts}")
        print()
        print("  Nonzero counts on double_gyre_psi are expected and intended:")
        print("  the periodic family carries extrema as distractors.")

    print()
    print("=" * 78)
    print("Example draws")
    print("=" * 78)
    rng3 = np.random.default_rng(args.seed + 2)
    for name, maker in FAMILIES.items():
        print(f"  {maker(rng3)!r}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
