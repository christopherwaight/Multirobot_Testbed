"""Reviewer 3, check T4b: (i) does the D tracker cross an out-of-band crest
only by momentum overshoot?  (ii) separatrix-phase metrics on the frozen
forced double gyre (acquisition to first far-saddle contact only)."""
import numpy as np
from harness import prim_D, prim_s1, make_hp_pass, make_dg_frozen, run, dg_static

beta = np.pi**3*0.1
print("(i) hyperbolic-paraboloid pass, D tracker, start (-0.30, 0.08)")
for c2 in (0.3, 1.0, 3.0):
    fb = make_hp_pass(beta, np.sqrt(c2), 0.3)
    for alpha in (0.7, 0.0):
        h, _, cl = run(fb, prim_D, -0.30, 0.08, steps=400, alpha=alpha,
                       stop=lambda k, cx, cy: abs(cx) > 1.0 or abs(cy) > 1.0)
        modes = {}
        for d in cl.diagnostics:
            modes[d['mode']] = modes.get(d['mode'], 0) + 1
        print(f"  D_c={-c2:+.1f} alpha_mom={alpha}: steps={len(h)} max x={h[:,0].max():+.4f} "
              f"x at step 100={h[min(100,len(h)-1),0]:+.4f} end=({h[-1,0]:+.3f},{h[-1,1]:+.3f}) modes={modes}")

print("\n(ii) frozen forced double gyre, separatrix phase only")
starts = [("S1", -0.15, 0.30), ("S2", 0.05, 0.40), ("S3", 0.0, 0.0), ("S4", 0.10, -0.20),
          ("S5", 0.15, 0.25), ("S6", -0.20, -0.30), ("N0", 0.0, 0.35)]
for eps in (0.0, 0.1, 0.25):
    fc = make_dg_frozen(eps) if eps > 0 else dg_static
    xs = fc.separatrix_x if eps > 0 else 0.0
    far, near = np.array([xs, -0.5]), np.array([xs, 0.5])
    for name, prim in (("D", prim_D), ("s1", prim_s1)):
        line = []
        for lab, sx, sy in starts:
            h, rh, cl = run(fc, prim, xs + sx, sy, steps=600,
                            stop=lambda k, cx, cy: abs(cx - xs) > 1.0 or abs(cy) > 0.52)
            dfar = np.linalg.norm(h - far, axis=1)
            kc = int(np.argmax(dfar < 0.06)) if (dfar < 0.06).any() else -1
            dx = h[:, 0] - xs
            inb = np.abs(dx) < 0.05
            acq = next((k for k in range(len(inb) - 10) if inb[k:k+10].all()), -1)
            if acq >= 0 and kc > acq:
                seg = slice(acq, kc)
                straddle = (rh[seg, :, 0].min(1) < xs) & (rh[seg, :, 0].max(1) > xs)
                line.append(f"{lab}: acq {acq}, contact {kc}, mean|dx| {np.abs(dx[seg]).mean():.3f}, "
                            f"max|dx| {np.abs(dx[seg]).max():.3f}, lost {int((~straddle).sum())}")
            else:
                end = h[-1]
                which = "near" if np.linalg.norm(end - near) < 0.1 else ("far" if np.linalg.norm(end - far) < 0.1 else "other")
                line.append(f"{lab}: acq {acq}, no far contact after acquisition, ended {which} ({end[0]-xs:+.2f},{end[1]:+.2f})")
        print(f"  eps={eps} {name}:")
        for l in line:
            print("     " + l)
