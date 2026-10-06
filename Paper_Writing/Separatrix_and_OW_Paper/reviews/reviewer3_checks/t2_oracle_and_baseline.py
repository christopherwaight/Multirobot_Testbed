"""Reviewer 3, check T2: clean benchmark runs from the paper's six starts and
the noise-sweep start, for the D tracker (fitted), the D tracker with the
TRUE H_D substituted, the s1 tracker, and a pure flow-following baseline."""
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
import json
import numpy as np
from harness import (dg_static, prim_D, prim_s1, prim_flow, make_prim_D_oracle,
                     H_true_dg, run)

STARTS = [("S1", -0.15, 0.30), ("S2", 0.05, 0.40), ("S3", 0.00, 0.00),
          ("S4", 0.10, -0.20), ("S5", 0.15, 0.25), ("S6", -0.20, -0.30),
          ("N0", 0.00, 0.35)]
FAR, NEAR = np.array([0.0, -0.5]), np.array([0.0, 0.5])
PRIMS = {"D_fit": prim_D, "D_oracleH": make_prim_D_oracle(H_true_dg),
         "s1": prim_s1, "flow": prim_flow}


def stopper():
    st = {"contact": None}
    def stop(k, cx, cy):
        if abs(cx) > 1.0 or abs(cy) > 0.52:
            return True
        if st["contact"] is None and np.hypot(cx - FAR[0], cy - FAR[1]) < 0.06:
            st["contact"] = k
        return st["contact"] is not None and k - st["contact"] >= 150
    return stop


def acquire_step(hist):
    inb = np.abs(hist[:, 0]) < 0.05
    for k in range(len(inb) - 10):
        if inb[k:k+10].all():
            return k
    return -1


out = {}
for name, prim in PRIMS.items():
    print(f"\n== {name} ==")
    rows = []
    for lab, x0, y0 in STARTS:
        h, rh, _ = run(dg_static, prim, x0, y0, steps=600, stop=stopper())
        dfar = np.linalg.norm(h - FAR, axis=1); dnear = np.linalg.norm(h - NEAR, axis=1)
        end = h[-1]
        # where did it spend its last 50 steps
        tail = h[-50:]
        row = dict(start=lab, steps=len(h), acq=acquire_step(h),
                   min_far=float(dfar.min()), min_near=float(dnear.min()),
                   end=[float(end[0]), float(end[1])],
                   end_far=float(np.linalg.norm(end - FAR)), end_near=float(np.linalg.norm(end - NEAR)),
                   tail_spread=float(np.linalg.norm(tail.max(0) - tail.min(0))))
        rows.append(row)
        print(f"  {lab}: steps={row['steps']:3d} acq={row['acq']:3d} min_far={row['min_far']:.3f} "
              f"min_near={row['min_near']:.3f} end=({end[0]:+.3f},{end[1]:+.3f}) "
              f"tail_spread={row['tail_spread']:.3f}")
    out[name] = rows

with open(__import__("os").path.join(HERE, "t2_results.json"), "w") as f:
    json.dump(out, f, indent=1)
