"""Check the critical point lists of vecsim's nonlinear fields: v = 0 at each listed point, the listed
Jacobian at the target and each point's index sign match finite differences, and a Newton search from a
dense grid finds no zero inside the box that the list misses."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hyperparameter_study"))
import vecsim as vs

H = 1e-6
def jac(F, x, y):
    ux = (np.array(F.fn(x + H, y)) - np.array(F.fn(x - H, y))) / (2 * H)
    uy = (np.array(F.fn(x, y + H)) - np.array(F.fn(x, y - H))) / (2 * H)
    return np.array([[ux[0], uy[0]], [ux[1], uy[1]]])

def newton_zeros(F, box, ng=41):
    found = []
    for x0 in np.linspace(-box, box, ng):
        for y0 in np.linspace(-box, box, ng):
            p = np.array([x0, y0])
            for _ in range(60):
                v = np.array(F.fn(p[0], p[1]), float)
                J = jac(F, *p)
                if abs(np.linalg.det(J)) < 1e-12 or not np.isfinite(v).all():
                    break
                p = p - np.linalg.solve(J, v)
                if np.abs(p).max() > 5 * box:
                    break
            v = np.array(F.fn(p[0], p[1]), float)
            if np.isfinite(v).all() and np.hypot(*v) < 1e-9 and np.abs(p).max() <= box:
                if not any(np.hypot(*(p - q)) < 1e-6 for q in found):
                    found.append(p)
    return np.array(found)

def check(label, F, box):
    bad = 0
    for x, y, idx in F.crit:
        v = np.hypot(*np.array(F.fn(x, y), float)); d = np.linalg.det(jac(F, x, y))
        if v > 1e-9 or np.sign(d) != np.sign(idx):
            bad += 1; print(f"  {label}: listed ({x:.3f},{y:.3f}) idx {idx:+.0f}  |v|={v:.1e} det={d:+.3e}  MISMATCH")
    J0 = jac(F, *F.target); dJ = np.abs(J0 - F.J0).max()
    inbox = [(x, y) for x, y, _ in F.crit if max(abs(x), abs(y)) <= box]
    found = newton_zeros(F, box)
    missing = [tuple(np.round(p, 4)) for p in found if not any(np.hypot(p[0]-a, p[1]-b) < 1e-4 for a, b in inbox)]
    extra = [t for t in inbox if not any(np.hypot(t[0]-p[0], t[1]-p[1]) < 1e-4 for p in found)]
    print(f"{label:34s} listed {len(F.crit):3d} (in box {len(inbox):3d})  Newton found {len(found):3d}  "
          f"bad {bad}  J0 diff {dJ:.1e}  missing {len(missing)}  unreached-by-search {len(extra)}")
    if missing: print("   missing:", missing[:6])

check("gyre", vs.make_field("gyre"), 1.5)
for fam in ("potential", "cellular"):
    for e in (-1.0, -0.5, 0.5, 1.0):
        check(f"curved_saddle {fam} eps={e:+g}", vs.make_field("curved_saddle", eps=e, family=fam), 1.5)
