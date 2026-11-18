"""Closed forms for the 2026-11-16 robot model, checked against simulation.

1. Attraction stability. Linear loop, dt-sampled, robot = gain g, delay L = (d + f) dt, lag alpha:
       v[j]   = alpha v[j-1] + (1 - alpha) g u_L[j],   u_L[j] = -k ((1-f) e[j-d] + f e[j-d-1])
       e[j+1] = e[j] + dt v[j]
   Stability limit k_max from the eigenvalues of this linear system (bisection). For alpha ~ 0 and
   L < dt (d = 0): characteristic z^2 - (1 - a(1-f)) z + a f, a = k g dt, stable iff a f < 1 and
   a (1 - 2f) < 2, so k_max = 1 / (g L) when f >= 1/4.
   Exact sampling of the continuous robot (tau v' = -v + g u(t - L), p' = v, command held over dt)
   keeps the lag's position delay inside each period, which the model above drops. Its limit is
   lower (16.1 at L 0.055, tau 0.02; paper Sec. III-D), and over every split of the fitted
   L + tau = 0.075 s it is at least 1 / (g (L + tau)) = 15.5 (reached when tau = 0).
2. Orbit floor. The orbit command turns at V / r; the robot follows it T late, which moves the
   centroid outward at about T V^2 / r; the radial correction g k_r (r - r_d) balances it:
       r* = (r_d + sqrt(r_d^2 + 4 C)) / 2,   C = V^2 T / (g k_r),   V = g k_t.
   T is the effective lag; candidates L + tau and L + tau + dt/2 are compared with simulation.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rigidsim as rs  # noqa: E402
import vecsim as vs  # noqa: E402


def k_max(g=0.86, L=0.055, tau=0.02, dt=0.1):
    a = np.exp(-dt / tau) if tau > 0 else 0.0
    s = L / dt
    d = int(np.floor(s + 1e-9))
    f = s - d

    def rho(k):
        n = d + 2
        A = np.zeros((n + 1, n + 1))
        c = np.zeros(n + 1)
        c[d] -= (1 - f) * k * g * (1 - a)
        c[d + 1] -= f * k * g * (1 - a)
        c[n] += a
        A[0] = np.eye(n + 1)[0] + dt * c
        for i in range(1, n):
            A[i, i - 1] = 1
        A[n] = c
        return np.max(np.abs(np.linalg.eigvals(A)))
    lo, hi = 0.01, 1000.0
    for _ in range(80):
        m = np.sqrt(lo * hi)
        lo, hi = (m, hi) if rho(m) < 1 else (lo, m)
    return lo


def k_max_exact(g=0.86, L=0.055, tau=0.02, dt=0.1):
    """Stability limit of the exactly sampled continuous robot, for 0 <= L < dt. State (p, v, u_prev)."""
    def step(t):   # state transition and input response of (p, v) over time t with a held command
        if tau == 0:
            return np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([g * t, g])
        e = np.exp(-t / tau)
        return np.array([[1.0, tau * (1 - e)], [0.0, e]]), g * np.array([t - tau * (1 - e), 1 - e])

    P1, G1 = step(L)
    P2, G2 = step(dt - L)

    def rho(k):
        A = np.zeros((3, 3))
        A[:2, :2] = P2 @ P1
        A[:2, 0] -= k * G2
        A[:2, 2] = P2 @ G1
        A[2, 0] = -k
        return np.max(np.abs(np.linalg.eigvals(A)))
    lo, hi = 0.01, 1000.0
    for _ in range(80):
        m = np.sqrt(lo * hi)
        lo, hi = (m, hi) if rho(m) < 1 else (lo, m)
    return lo


def r_star(rd, T, g=0.86, kt=0.15, kr=1.0):
    V = g * kt
    C = V * V * T / (g * kr)
    return (rd + np.sqrt(rd * rd + 4 * C)) / 2


def main():
    print("1. attraction stability limit k_max (1/s)")
    print(f"   eigenvalues, new model: {k_max():.2f};  1/(g L) = {1 / (0.86 * 0.055):.2f};  "
          f"old model (tau 0.28, no delay, g 1): {k_max(g=1.0, L=0.0, tau=0.28):.1f} (paper: 113)")
    print(f"   exact sampling of the continuous robot: {k_max_exact():.2f} (paper: 16.1)")
    splits = [k_max_exact(L=L, tau=0.075 - L) for L in np.linspace(0, 0.075, 16)]
    print(f"   over splits of L + tau = 0.075 s: {min(splits):.2f} to {max(splits):.2f};  "
          f"1/(g (L + tau)) = {1 / (0.86 * 0.075):.2f} (paper: 15.5)")
    F = vs.make_field("vortex")
    for k in (15.0, 20.0, 21.0, 22.0, 25.0):
        P = vs.params(200, k=k, vcmd_max=1e9, vaxis=1e9, vs=0.0, vmax=1e9)
        rng = np.random.default_rng(0)
        st = rng.uniform(-0.05, 0.05, (200, 2))
        C, _ = rs.simulate(F, "attract", "fit", P, 30.0, rs.triangle(np.zeros(200), np.full(200, 0.3)), st)
        grow = np.nanmedian(np.hypot(*C[:, -1].T) / np.hypot(*C[:, 0].T))
        print(f"   simulation k = {k:5.1f}: median |e(30 s)| / |e(0)| = {grow:.2e}  ({'stable' if grow < 1 else 'UNSTABLE'})")
    print("\n2. orbit floor r* (m): simulation (linear vortex, 60 s, mean radius after 20 s) vs closed form")
    rds = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    n = len(rds)
    P = vs.params(n, rd=0.0)
    P["rd"] = rds.copy()
    C, _ = rs.simulate(F, "orbit", "fit", P, 60.0, rs.triangle(np.zeros(n), np.full(n, 0.3)),
                       np.stack([rds + 0.05, np.zeros(n)], 1))
    r_sim = np.nanmean(np.hypot(C[:, 200:, 0], C[:, 200:, 1]), 1)
    T1, T2 = 0.055 + 0.02, 0.055 + 0.02 + 0.05
    for rd, rs_ in zip(rds, r_sim):
        print(f"   r_d {rd:.2f}: sim {rs_:.4f}   T = L + tau: {r_star(rd, T1):.4f}   T = L + tau + dt/2: {r_star(rd, T2):.4f}")
    # best-fit T
    Ts = np.linspace(0.05, 0.2, 301)
    err = [np.max(np.abs(r_star(rds, T) - r_sim)) for T in Ts]
    print(f"   best single T = {Ts[int(np.argmin(err))]:.3f} s (max error {1000 * min(err):.1f} mm)")


if __name__ == "__main__":
    main()
