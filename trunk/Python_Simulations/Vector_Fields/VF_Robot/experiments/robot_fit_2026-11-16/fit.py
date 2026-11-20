"""Robot response fits on the 2026-11 campaign (Python port of fit_coulomb_delay.m plus checks).

Model (per robot, dt 0.1 s), delay D = d + f steps (f fractional, linear blend):
    uu(k)  = (1-f) u(k-d) + f u(k-d-1)
    free   = a v(k-1) + (1-a) g uu(k)
    v(k)   = free (1 - (1-a) vk / |free|)   (0 if that goes negative), |v| <= vmax
"""
import pickle
import sys

import numpy as np
from scipy.optimize import minimize

R_ALL = pickle.load(open(sys.argv[1], "rb"))
NAMES = ["celeste", "tidal", "pacific-blue", "redwood"]
VS = 0.025


def ratio(r):
    um = np.linalg.norm(r["u"], axis=1)
    vm = np.linalg.norm(r["v"], axis=1)
    k = (um > 0.05) & np.isfinite(vm)
    return np.median(vm[k] / um[k]) if k.any() else np.nan


def pack(runs, key="v"):
    T = max(len(r["u"]) for r in runs)
    N = len(runs)
    U = np.zeros((N, T, 2))
    V = np.full((N, T, 2), np.nan)
    for i, r in enumerate(runs):
        n = len(r["u"])
        U[i, :n] = r["u"]
        V[i, :n] = r[key]
    return U, V


def delayed(U, d, f):
    N, T, _ = U.shape
    Z = np.zeros((N, d + 1, 2))
    P = np.concatenate([Z, U], axis=1)          # P[:, j] = U[:, j - d - 1]
    a = P[:, 1:T + 1]                            # u(k - d)
    b = P[:, 0:T]                                # u(k - d - 1)
    return (1 - f) * a + f * b


def simulate(U, a, g, vk, d=1, f=0.0, vmax=np.inf, vs=None):
    UU = delayed(U, d, f)
    N, T, _ = U.shape
    v = np.zeros((N, 2))
    stuck = np.ones(N, bool) if vs is not None else np.zeros(N, bool)
    out = np.zeros((N, T, 2))
    drop = (1 - a) * vk
    for k in range(T):
        u = UU[:, k]
        um = np.linalg.norm(u, axis=1)
        if vs is not None:
            stuck &= ~np.any(v != 0, axis=1)
            hold = stuck & (um <= vs)
            stuck &= hold
        else:
            hold = np.zeros(N, bool)
        free = a * v + (1 - a) * g * u
        fm = np.linalg.norm(free, axis=1)
        scale = np.where(fm > drop, 1 - drop / np.maximum(fm, 1e-12), 0.0)
        v = free * scale[:, None]
        if vs is not None:
            stuck |= (fm <= drop) & (um <= vs)
        v[hold] = 0.0
        sp = np.linalg.norm(v, axis=1)
        over = sp > vmax
        v[over] *= (vmax / sp[over])[:, None]
        out[:, k] = v
    return out


def onestep(U, V, a, g, vk, d=1, f=0.0):
    UU = delayed(U, d, f)
    Vp = np.concatenate([np.full_like(V[:, :1], np.nan), V[:, :-1]], axis=1)
    free = a * Vp + (1 - a) * g * UU
    fm = np.linalg.norm(free, axis=2, keepdims=True)
    drop = (1 - a) * vk
    pred = np.where(fm > drop, free * (1 - drop / np.maximum(fm, 1e-12)), 0.0)
    pred[:, : 1 + d + (f > 0)] = np.nan          # MATLAB starts at k = 2 + d
    return pred


def rmse(P, V):
    e = (P - V) ** 2
    m = np.isfinite(e).all(axis=2)
    return np.sqrt(e[m].sum() / m.sum())


def fit(U, V, kind, d=1, f=0.0, x0=(0.3, 0.85, 0.01), vmax=np.inf, free_vmax=False):
    def cost(x):
        a, g, vk = x[0], x[1], max(x[2], 0.0)
        vm = x[3] if free_vmax else vmax
        if not (0 <= a < 1) or g <= 0 or vm <= 0.05:
            return 1e3
        P = onestep(U, V, a, g, vk, d, f) if kind == "eq" else simulate(U, a, g, vk, d, f, vm)
        return rmse(P, V)
    x0 = list(x0) + ([0.35] if free_vmax else [])
    r = minimize(cost, x0, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-6, maxfev=3000))
    x = r.x.copy()
    x[2] = max(x[2], 0.0)
    return x, r.fun


def tau(a):
    return -0.1 / np.log(a) if 0 < a < 1 else 0.0


def show(tag, x, U, V, d=1, f=0.0, vmax=np.inf):
    a, g, vk = x[:3]
    vm = x[3] if len(x) > 3 else vmax
    os_ = rmse(onestep(U, V, a, g, vk, d, f), V)
    fr = rmse(simulate(U, a, g, vk, d, f, vm), V)
    print(f"  {tag:<34} a {a:.3f} (tau {tau(a):.3f} s)  g {g:.3f}  vk {vk:.4f}"
          + (f"  vmax {vm:.3f}" if np.isfinite(vm) else "")
          + f" | 1-step {os_:.4f}  free-run {fr:.4f}")
    return fr


if __name__ == "__main__":
    healthy = [r for r in R_ALL if ratio(r) >= 0.75]
    old = [r for r in healthy if not (r["field"] == "Saddle" and r["nr"] == 4)
           and "0_point_5_run10" not in r["file"]]
    print(f"robot-runs: all {len(R_ALL)}, healthy {len(healthy)}, old subset {len(old)}")

    print("\n[1] Replicate fit_coulomb_delay.m (one-step / equation-error fit) on the old subset")
    U, V = pack(old)
    for d in (0, 1, 2):
        x, _ = fit(U, V, "eq", d)
        show(f"one-step fit, delay {d}", x, U, V, d)
    print("  references, free-run:")
    print(f"    python OmniCluster now: {rmse(simulate(U, 0.70, 1.0, 0.025, 0, 0, 0.4), V):.4f}")

    print("\n[2] Same one-step fit on ALL healthy data")
    U, V = pack(healthy)
    for d in (0, 1, 2):
        x, _ = fit(U, V, "eq", d)
        show(f"one-step fit, delay {d}", x, U, V, d)

    print("\n[3] Output-error fit (minimise free-run error) on ALL healthy data")
    for d in (0, 1, 2):
        x, _ = fit(U, V, "oe", d)
        show(f"free-run fit, delay {d}", x, U, V, d)

    print("\n[4] Fractional delay scan, free-run fit")
    for D in (0.5, 0.75, 1.0, 1.25, 1.5):
        d, f = int(np.floor(D)), D - np.floor(D)
        x, _ = fit(U, V, "oe", d, f)
        show(f"free-run fit, delay {D:.2f} steps", x, U, V, d, f)

    print("\n[5] Velocity from nominal 0.1 s instead of wall-clock dt (free-run fit, delay 1)")
    Un, Vn = pack(healthy, "v_nom")
    x, _ = fit(Un, Vn, "oe", 1)
    show("nominal dt", x, Un, Vn, 1)

    print("\n[6] Cross-validation (free-run fit on A, scored on B), delay 1")
    groups = {
        "vortex": [r for r in healthy if r["field"] == "Vortex"],
        "saddle": [r for r in healthy if r["field"] == "Saddle"],
        "3R": [r for r in healthy if r["nr"] == 3],
        "4R": [r for r in healthy if r["nr"] == 4],
    }
    for A, B in (("vortex", "saddle"), ("saddle", "vortex"), ("3R", "4R"), ("4R", "3R")):
        Ua, Va = pack(groups[A])
        Ub, Vb = pack(groups[B])
        x, fa = fit(Ua, Va, "oe", 1)
        xb, fb = fit(Ub, Vb, "oe", 1)
        cross = rmse(simulate(Ub, *x, d=1), Vb)
        print(f"  fit {A:<6} a {x[0]:.3f} g {x[1]:.3f} vk {x[2]:.4f} | on {B}: {cross:.4f} "
              f"(own fit {fb:.4f}: a {xb[0]:.3f} g {xb[1]:.3f} vk {xb[2]:.4f})")

    print("\n[7] Per robot, free-run fit, delay 1")
    for nm in NAMES:
        Ur, Vr = pack([r for r in healthy if r["robot"] == nm])
        x, _ = fit(Ur, Vr, "oe", 1, x0=(0.2, 0.85, 0.005))
        show(f"{nm} (n={Ur.shape[0]})", x, Ur, Vr, 1)

    print("\n[8] Speed cap: free vmax in the free-run fit, delay 1")
    x, _ = fit(U, V, "oe", 1, free_vmax=True, x0=(0.2, 0.85, 0.005))
    show("free vmax", x, U, V, 1)
    for vm in (0.25, 0.3, 0.35, 0.4):
        xx, _ = fit(U, V, "oe", 1, vmax=vm, x0=(0.2, 0.85, 0.005))
        show(f"vmax fixed {vm}", xx, U, V, 1, vmax=vm)

    with open(sys.argv[1].replace(".pkl", "_healthy_idx.pkl"), "wb") as fh:
        pickle.dump([R_ALL.index(r) for r in healthy], fh)
