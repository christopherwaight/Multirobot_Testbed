"""Reviewer 3, check T1: symbolic and numerical checks of Draft_12 Section II.

Independent of the repo code. Each check prints PASS/FAIL or the numbers the
paper quotes next to the numbers computed here.
"""
import numpy as np
import sympy as sp

np.set_printoptions(precision=5, suppress=True)

# ---------------------------------------------------------------------------
# 1. Eqs. (8)-(13): Jacobian, det gradient/Hessian, strain gradient
# ---------------------------------------------------------------------------
x, y = sp.symbols('x y', real=True)
a = sp.symbols('a1:7', real=True)
b = sp.symbols('b1:7', real=True)
u = a[0] + a[1]*x + a[2]*y + a[3]*x*y + a[4]*x**2/2 + a[5]*y**2/2
v = b[0] + b[1]*x + b[2]*y + b[3]*x*y + b[4]*x**2/2 + b[5]*y**2/2
J = sp.Matrix([[sp.diff(u, x), sp.diff(u, y)], [sp.diff(v, x), sp.diff(v, y)]])
J_paper = sp.Matrix([[a[1] + a[4]*x + a[3]*y, a[2] + a[3]*x + a[5]*y],
                     [b[1] + b[4]*x + b[3]*y, b[2] + b[3]*x + b[5]*y]])
print("Eq.(8) Jhat:", "PASS" if sp.simplify(J - J_paper) == sp.zeros(2) else "FAIL")

D = J.det()
g0 = sp.Matrix([sp.diff(D, x), sp.diff(D, y)]).subs({x: 0, y: 0})
a1, a2, a3, a4, a5, a6 = a
b1, b2, b3, b4, b5, b6 = b
g_paper = sp.Matrix([a5*b3 + a2*b4 - a4*b2 - a3*b5,
                     a4*b3 + a2*b6 - a6*b2 - a3*b4])
print("Eq.(9) grad D:", "PASS" if sp.simplify(g0 - g_paper) == sp.zeros(2, 1) else "FAIL")
H = sp.hessian(D, (x, y))
H_paper = sp.Matrix([[2*(a5*b4 - a4*b5), a5*b6 - a6*b5],
                     [a5*b6 - a6*b5, 2*(a4*b6 - a6*b4)]])
print("Eq.(10) Hess D:", "PASS" if sp.simplify(H - H_paper) == sp.zeros(2) else "FAIL",
      "| constant over footprint:", all(sp.diff(h, s) == 0 for h in H for s in (x, y)))

mu = (J[0, 0] + J[1, 1])/2
sn = (J[0, 0] - J[1, 1])/2
ss = (J[0, 1] + J[1, 0])/2
gmu = [sp.diff(mu, s) for s in (x, y)]
gsn = [sp.diff(sn, s) for s in (x, y)]
gss = [sp.diff(ss, s) for s in (x, y)]
ok13 = (sp.simplify(gmu[0] - (a5 + b4)/2) == 0 and sp.simplify(gmu[1] - (a4 + b6)/2) == 0 and
        sp.simplify(gsn[0] - (a5 - b4)/2) == 0 and sp.simplify(gsn[1] - (a4 - b6)/2) == 0 and
        sp.simplify(gss[0] - (a4 + b5)/2) == 0 and sp.simplify(gss[1] - (a6 + b4)/2) == 0)
print("Eq.(13) strain gradients:", "PASS" if ok13 else "FAIL")
r = sp.sqrt(sn**2 + ss**2)
s1 = mu - r
gs1 = [sp.diff(s1, s) for s in (x, y)]
gs1_paper = [gmu[i] - (sn*gsn[i] + ss*gss[i])/r for i in range(2)]
print("Eq.(12) grad s1:", "PASS" if all(sp.simplify(gs1[i] - gs1_paper[i]) == 0 for i in range(2)) else "FAIL")

# ---------------------------------------------------------------------------
# 2. Double gyre, Eqs. (15)-(16) and the vorticity line
# ---------------------------------------------------------------------------
A, xf, yf = sp.symbols('A x_f y_f', positive=True)
U = -sp.pi*A*sp.sin(sp.pi*xf)*sp.cos(sp.pi*yf)
V = sp.pi*A*sp.cos(sp.pi*xf)*sp.sin(sp.pi*yf)
Jdg = sp.Matrix([[sp.diff(U, xf), sp.diff(U, yf)], [sp.diff(V, xf), sp.diff(V, yf)]])
Ddg = sp.simplify(Jdg.det())
D_paper = -sp.Rational(1, 2)*sp.pi**4*A**2*(sp.cos(2*sp.pi*xf) + sp.cos(2*sp.pi*yf))
print("Eq.(15) D:", "PASS" if sp.simplify(sp.expand_trig(Ddg - D_paper)) == 0 else "FAIL")
Hdg = sp.hessian(Ddg, (xf, yf))
H_paper_dg = 2*sp.pi**6*A**2*sp.diag(sp.cos(2*sp.pi*xf), sp.cos(2*sp.pi*yf))
print("Eq.(15) H_D:", "PASS" if sp.simplify(sp.expand_trig(Hdg - H_paper_dg)) == sp.zeros(2) else "FAIL")
om = sp.diff(V, xf) - sp.diff(U, yf)
print("omega = -2 pi^2 A sin sin:", "PASS" if sp.simplify(om + 2*sp.pi**2*A*sp.sin(sp.pi*xf)*sp.sin(sp.pi*yf)) == 0 else "FAIL")
print("shear strain identically zero:", sp.simplify(Jdg[0, 1] + Jdg[1, 0]) == 0)

# ---------------------------------------------------------------------------
# 3. Eq. (14): D' = D + Omega*omega + Omega^2 under a rotating observer
# ---------------------------------------------------------------------------
m11, m12, m21, m22, Om = sp.symbols('m11 m12 m21 m22 Omega', real=True)
M = sp.Matrix([[m11, m12], [m21, m22]])
E = sp.Matrix([[0, -1], [1, 0]])          # Jacobian of Omega*(-y, x)
Dp = (M + Om*E).det()
print("Eq.(14) D' = D + Omega*omega + Omega^2:",
      "PASS" if sp.simplify(Dp - (M.det() + Om*(m21 - m12) + Om**2)) == 0 else "FAIL")

# ---------------------------------------------------------------------------
# 4. Formation conditioning: kappa(Phi) = 8.26 at every radius and heading
# ---------------------------------------------------------------------------
def basis(px, py):
    return np.array([1.0, px, py, px*py, 0.5*px**2, 0.5*py**2])


def pentagon(rho, heading=0.0, center=(0.0, 0.0)):
    ang = heading + np.pi/2 + 2*np.pi*np.arange(5)/5
    ring = np.c_[rho*np.cos(ang), rho*np.sin(ang)]
    return np.vstack([np.array(center), ring])


def phi_matrix(P, rho):
    Phi = np.array([basis(*p) for p in P])
    scale = np.array([1, rho, rho, rho**2, rho**2, rho**2])   # radius-normalized
    return Phi, Phi/scale


for rho in (0.01, 0.075, 0.098, 1.0):
    ks = [np.linalg.cond(phi_matrix(pentagon(rho, h), rho)[1]) for h in np.linspace(0, 2*np.pi, 37)]
    print(f"kappa(Phi_norm) rho={rho:<6} min={min(ks):.3f} max={max(ks):.3f}  (paper 8.26)")

P = pentagon(1.0)
P[0] = [0.99, 0.0]     # center robot moved out toward the ring
# 'moved out to 0.99 rho' read as radial position 0.99 rho; try two readings
k1 = np.linalg.cond(phi_matrix(P, 1.0)[1])
# Distance of six points from best-fit circle
def circle_fit_resid(P):
    A_ = np.c_[2*P[:, 0], 2*P[:, 1], np.ones(len(P))]
    bb = (P**2).sum(1)
    c = np.linalg.lstsq(A_, bb, rcond=None)[0]
    R = np.sqrt(c[2] + c[0]**2 + c[1]**2)
    d = np.abs(np.hypot(P[:, 0]-c[0], P[:, 1]-c[1]) - R)
    return d.max()/R
print(f"center at 0.99 rho (toward a ring vertex gap): kappa={k1:.1f} (paper 564); "
      f"max circle deviation {100*circle_fit_resid(P):.2f}% (paper 0.6%)")
for ang in np.linspace(0, 2*np.pi/5, 5):
    P = pentagon(1.0); P[0] = [0.99*np.cos(ang + np.pi/2), 0.99*np.sin(ang + np.pi/2)]
    print(f"   direction {np.degrees(ang):5.1f} deg from a vertex: kappa={np.linalg.cond(phi_matrix(P,1.0)[1]):7.1f}, "
          f"circle dev {100*circle_fit_resid(P):.2f}%")

# ---------------------------------------------------------------------------
# 5. Noise gains: heading independence and per-coefficient std
# ---------------------------------------------------------------------------
rho = 0.075
for h in (0.0, 0.3, 1.0):
    Phi, _ = phi_matrix(pentagon(rho, h), rho)
    Pinv = np.linalg.inv(Phi)
    std = np.sqrt((Pinv**2).sum(1))          # std per coefficient for unit-variance noise
    norm_std = std*np.array([1, rho, rho, rho**2, rho**2, rho**2])
    print(f"heading {h:.1f}: normalized noise gains [c, x, y, xy, xx, yy] = {norm_std}")
print("   reference 8/sqrt(10) =", 8/np.sqrt(10), " 4/sqrt(10) =", 4/np.sqrt(10))

# Covariance of the full Hessian-of-u estimate in world frame vs heading
def hess_cov(h):
    Phi, _ = phi_matrix(pentagon(rho, h), rho)
    Pinv = np.linalg.inv(Phi)
    C = Pinv @ Pinv.T
    return C[3:, 3:]*rho**4
print("second-order covariance invariant under heading:",
      np.allclose(hess_cov(0.0), hess_cov(0.7), atol=1e-9))

# ---------------------------------------------------------------------------
# 6. Fitted vs true H_D on the double-gyre separatrix
# ---------------------------------------------------------------------------
Aval = 0.1
def dg(xw, yw, A=Aval):
    X, Y = np.pi*(xw + 1), np.pi*(yw + 0.5)
    return -np.pi*A*np.sin(X)*np.cos(Y), np.pi*A*np.cos(X)*np.sin(Y)

def fit_at(field, c, rho=0.075, heading=0.0):
    P = pentagon(rho, heading)
    Phi = np.array([basis(*p) for p in P])
    uu = np.array([field(c[0]+p[0], c[1]+p[1])[0] for p in P])
    vv = np.array([field(c[0]+p[0], c[1]+p[1])[1] for p in P])
    return np.linalg.solve(Phi, uu), np.linalg.solve(Phi, vv)

def D_parts(tu, tv):
    _, A2, A3, A4, A5, A6 = tu
    _, B2, B3, B4, B5, B6 = tv
    D0 = A2*B3 - A3*B2
    g = np.array([A5*B3 + A2*B4 - A4*B2 - A3*B5, A4*B3 + A2*B6 - A6*B2 - A3*B4])
    H = np.array([[2*(A5*B4 - A4*B5), A5*B6 - A6*B5], [A5*B6 - A6*B5, 2*(A4*B6 - A6*B4)]])
    return D0, g, H

def H_true(xw, yw, A=Aval):
    X, Y = np.pi*(xw + 1), np.pi*(yw + 0.5)
    return 2*np.pi**6*A**2*np.diag([np.cos(2*X), np.cos(2*Y)])

print("\nFitted vs true H_D eigenvalues on the separatrix x=0 (rho=0.075, and rho->0):")
for yw in (0.45, 0.4, 0.35, 0.3, 0.25, 0.15, 0.05, 0.0):
    D0, g, Hf = D_parts(*fit_at(dg, (0.0, yw)))
    _, _, Hs = D_parts(*fit_at(dg, (0.0, yw), rho=0.005))
    lt = np.linalg.eigvalsh(H_true(0.0, yw))
    lf = np.linalg.eigvalsh(Hf)
    ls = np.linalg.eigvalsh(Hs)
    ratio = abs(D0)/np.linalg.norm(Hf, 'fro')
    print(f"  y={yw:5.2f}  true {lt}  fit(0.075) {lf}  fit(0.005) {ls}  "
          f"band ratio |D|/||H||={ratio:.4f} ({'IN' if ratio < 0.025 or abs(D0) < 1e-3 else 'out'})")

# band extent along the separatrix with the fitted values
ys = np.linspace(-0.49, 0.49, 981)
inb = []
for yw in ys:
    D0, g, Hf = D_parts(*fit_at(dg, (0.0, yw)))
    inb.append(abs(D0) < 1e-3 or abs(D0)/np.linalg.norm(Hf, 'fro') < 0.025)
inb = np.array(inb)
print(f"FLOW band on the separatrix (fitted, heading 0): |y| < {ys[inb].max():.3f}")

# ---------------------------------------------------------------------------
# 7. Bias of D-hat and s1-hat under measurement noise (heading 0, y=0.35)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
P = pentagon(rho, 0.0)
Phi = np.array([basis(*p) for p in P]); Pinv = np.linalg.inv(Phi)
c = (0.0, 0.35)
uu = np.array([dg(c[0]+p[0], c[1]+p[1])[0] for p in P])
vv = np.array([dg(c[0]+p[0], c[1]+p[1])[1] for p in P])
tu0, tv0 = Pinv @ uu, Pinv @ vv
def s1_of(tu, tv):
    mu = 0.5*(tu[1] + tv[2]); a_ = 0.5*(tu[1] - tv[2]); b_ = 0.5*(tu[2] + tv[1])
    return mu - np.hypot(a_, b_)
N = 200000
for sig in (0.002, 0.0077):
    eu = rng.standard_normal((N, 6))*sig; ev = rng.standard_normal((N, 6))*sig
    TU = tu0 + eu @ Pinv.T; TV = tv0 + ev @ Pinv.T
    Dh = TU[:, 1]*TV[:, 2] - TU[:, 2]*TV[:, 1]
    S1h = 0.5*(TU[:, 1] + TV[:, 2]) - np.hypot(0.5*(TU[:, 1] - TV[:, 2]), 0.5*(TU[:, 2] + TV[:, 1]))
    D0 = tu0[1]*tv0[2] - tu0[2]*tv0[1]
    print(f"sigma_uv={sig}: D bias {Dh.mean()-D0:+.2e} (SE {Dh.std()/np.sqrt(N):.1e}); "
          f"s1 bias {S1h.mean()-s1_of(tu0, tv0):+.2e} (SE {S1h.std()/np.sqrt(N):.1e})")

# ---------------------------------------------------------------------------
# 8. Rotating observer: trench shift and predicted straddle-loss onset
# ---------------------------------------------------------------------------
Om = 0.23
print(f"\nPredicted D'-trench shift at the origin, Omega=0.23: {Om/(np.pi**3*Aval):.4f} (paper observed 0.084)")
print(f"Predicted Omega at which the shift equals rho=0.075: {0.075*np.pi**3*Aval:.4f} (paper first loss at 0.21)")
print(f"Omega/(pi^2 A) = {Om/(np.pi**2*Aval):.3f}; k*c_max = {3.0*0.04:.3f}; max Omega*|p| on path = {Om*0.5:.3f}")

# Units sanity of the D law: Newton across-term magnitude at n = 0.05 off the trench
D0, g, Hf = D_parts(*fit_at(dg, (0.05, 0.0)))
lam, W = np.linalg.eigh(Hf)
print(f"across Newton term -w2.g/lambda2 at n=0.05, y=0: {-(W[:,1]@g)/lam[1]:+.4f} (a length; it is fed to sat() with c_max in length/time)")
