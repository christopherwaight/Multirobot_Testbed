"""Reviewer 3, check T4: both primitives, unchanged, on fields other than the
steady double gyre.

  (a) linear saddle  v = (x, -y): the gradient of the hyperbolic paraboloid
      (x^2 - y^2)/2.  Separatrices are the axes.  D and s1 are uniform.
  (b) an exactly quadratic, incompressible flow whose D is itself a
      hyperbolic paraboloid (a mountain pass with no end minima), with the
      crest inside the FLOW band (D_c = -0.3) and outside it (D_c = -1.0).
  (c) frozen snapshots of the forced double gyre (Shadden), eps = 0.1, 0.25,
      phase pi/2: the separatrix is still the straight line f(x) = 1, but the
      shear strain and vorticity on it no longer vanish.
"""
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
import json
import numpy as np
from harness import (prim_D, prim_s1, prim_flow, make_linear_saddle, make_hp_pass,
                     make_dg_frozen, run)

np.set_printoptions(precision=4, suppress=True)
results = {}


def jac_fd(fn, x, y, h=1e-5):
    u1, v1 = fn(x + h, y); u0, v0 = fn(x - h, y)
    u3, v3 = fn(x, y + h); u2, v2 = fn(x, y - h)
    return np.array([[(u1-u0)/(2*h), (u3-u2)/(2*h)], [(v1-v0)/(2*h), (v3-v2)/(2*h)]])


def surrogates(fn, x, y):
    J = jac_fd(fn, x, y)
    S = 0.5*(J + J.T)
    return np.linalg.det(J), np.linalg.eigvalsh(S)[0], J[1, 0] - J[0, 1]


# ---------------------------------------------------------------- (a)
print("(a) linear saddle v = (x, -y)")
fa = make_linear_saddle(1.0)
res_a = {}
for name, prim in (("D", prim_D), ("s1", prim_s1), ("flow", prim_flow)):
    for sig in (0.0, 0.002):
        h, _, cl = run(fa, prim, 0.15, 0.20, steps=300, sigma_uv=sig, seed=1)
        disp = np.linalg.norm(h[-1] - h[0])
        path = np.sum(np.linalg.norm(np.diff(h, axis=0), axis=1))
        modes = {}
        for d in cl.diagnostics:
            modes[d['mode']] = modes.get(d['mode'], 0) + 1
        res_a[f"{name}_s{sig}"] = dict(end=h[-1].tolist(), disp=float(disp), path=float(path),
                                        min_abs_x=float(np.abs(h[:, 0]).min()),
                                        min_abs_y=float(np.abs(h[:, 1]).min()), modes=modes)
        print(f"  {name:4s} sigma_uv={sig}: end={h[-1]}, net displacement {disp:.3f}, "
              f"path length {path:.3f}, min|x|={np.abs(h[:,0]).min():.3f}, "
              f"min|y|={np.abs(h[:,1]).min():.3f}, modes={modes}")
results["a"] = res_a

# ---------------------------------------------------------------- (b)
print("\n(b) quadratic flow with a hyperbolic-paraboloid D (trench y=0, crest at origin)")
beta = np.pi**3*0.1
res_b = {}
for c2 in (0.3, 1.0):
    fb = make_hp_pass(beta, np.sqrt(c2), 0.3)
    Dc, s1c, _ = surrogates(fb, 0.0, 0.0)
    print(f"  D_c = {Dc:+.3f}, s1 at crest = {s1c:+.3f}, |D_c|/||H_D||_F = {abs(Dc)/(2*np.sqrt(2)*beta**2):.4f}")
    for name, prim in (("D", prim_D), ("s1", prim_s1), ("flow", prim_flow)):
        h, _, cl = run(fb, prim, -0.30, 0.08, steps=500, seed=1,
                       stop=lambda k, cx, cy: abs(cx) > 1.0 or abs(cy) > 1.0)
        modes = {}
        for d in cl.diagnostics:
            modes[d['mode']] = modes.get(d['mode'], 0) + 1
        crossed = bool((h[:, 0] > 0.02).any())
        k_cross = int(np.argmax(h[:, 0] > 0.0)) if (h[:, 0] > 0).any() else -1
        res_b[f"c2={c2}_{name}"] = dict(end=h[-1].tolist(), crossed=crossed, k_cross=k_cross,
                                        steps=len(h), final_abs_y=float(abs(h[-1, 1])),
                                        modes=modes, xmax=float(h[:, 0].max()),
                                        path=h[::5].tolist())
        print(f"    {name:4s}: steps={len(h):3d} end=({h[-1,0]:+.3f},{h[-1,1]:+.3f}) "
              f"crossed crest={crossed} (first x>0 at step {k_cross}) max x={h[:,0].max():+.3f} modes={modes}")
results["b"] = res_b

# ---------------------------------------------------------------- (c)
print("\n(c) frozen forced double gyre")
res_c = {}
for eps in (0.1, 0.25):
    fc = make_dg_frozen(eps)
    xs = fc.separatrix_x
    print(f"  eps={eps}: separatrix at x = {xs:+.4f}")
    # where do the D and s1 trenches sit relative to the separatrix?
    xx = np.linspace(xs - 0.25, xs + 0.25, 2001)
    tr = []
    for yy in (0.4, 0.3, 0.15, 0.05, -0.05, -0.15, -0.3, -0.4):
        Ds = np.array([surrogates(fc, X, yy)[0] for X in xx])
        Ss = np.array([surrogates(fc, X, yy)[1] for X in xx])
        om = surrogates(fc, xs, yy)[2]
        J = jac_fd(fc, xs, yy); S = 0.5*(J+J.T)
        w, V = np.linalg.eigh(S)
        tilt = np.degrees(np.arctan2(abs(V[0, 1]), abs(V[1, 1])))   # e2 angle from the y axis
        tr.append(dict(y=yy, D_trench=float(xx[np.argmin(Ds)] - xs),
                       s1_trench=float(xx[np.argmin(Ss)] - xs), omega=float(om),
                       e2_tilt_deg=float(min(tilt, 90 - tilt))))
        print(f"    y={yy:+.2f}: D-trench offset {xx[np.argmin(Ds)]-xs:+.4f}, "
              f"s1-trench offset {xx[np.argmin(Ss)]-xs:+.4f}, omega on sep {om:+.4f}, "
              f"strain-eigvec tilt {min(tilt, 90-tilt):5.1f} deg")
    Dcr, s1cr, _ = surrogates(fc, xs, 0.0)
    print(f"    at (x_s, 0): D={Dcr:+.4f}, s1={s1cr:+.4f}")
    starts = [("S1", -0.15, 0.30), ("S2", 0.05, 0.40), ("S3", 0.0, 0.0), ("S4", 0.10, -0.20),
              ("S5", 0.15, 0.25), ("S6", -0.20, -0.30), ("N0", 0.0, 0.35)]
    far = np.array([xs, -0.5])
    for name, prim in (("D", prim_D), ("s1", prim_s1)):
        rows = []
        for lab, sx, sy in starts:
            st = {"c": None}
            def stop(k, cx, cy, st=st):
                if abs(cx - xs) > 1.0 or abs(cy) > 0.52:
                    return True
                if st["c"] is None and np.hypot(cx - far[0], cy - far[1]) < 0.06:
                    st["c"] = k
                return st["c"] is not None and k - st["c"] >= 150
            h, rh, cl = run(fc, prim, xs + sx, sy, steps=600, stop=stop)
            dx = h[:, 0] - xs
            inb = np.abs(dx) < 0.05
            acq = next((k for k in range(len(inb) - 10) if inb[k:k+10].all()), -1)
            ride = slice(acq, None) if acq >= 0 else slice(0, 0)
            straddle = (rh[:, :, 0].min(1) < xs) & (rh[:, :, 0].max(1) > xs)
            lost = int((~straddle[acq:]).sum()) if acq >= 0 else -1
            dfar = np.linalg.norm(h - far, axis=1)
            rows.append(dict(start=lab, acq=acq, mean_off=float(np.abs(dx[ride]).mean()) if acq >= 0 else None,
                             max_off=float(np.abs(dx[ride]).max()) if acq >= 0 else None,
                             min_far=float(dfar.min()), end=(h[-1] - [xs, 0]).tolist(),
                             straddle_lost_steps=lost))
            print(f"    {name:2s} {lab}: acq={acq:3d} ride |x-xs| mean={rows[-1]['mean_off'] if acq>=0 else float('nan'):.4f} "
                  f"max={rows[-1]['max_off'] if acq>=0 else float('nan'):.4f} min_far={dfar.min():.3f} "
                  f"end(rel)=({h[-1,0]-xs:+.3f},{h[-1,1]:+.3f}) straddle-lost steps={lost}")
        res_c[f"eps={eps}_{name}"] = rows
    res_c[f"eps={eps}_trenches"] = tr

results["c"] = res_c
with open(__import__("os").path.join(HERE, "t4_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
