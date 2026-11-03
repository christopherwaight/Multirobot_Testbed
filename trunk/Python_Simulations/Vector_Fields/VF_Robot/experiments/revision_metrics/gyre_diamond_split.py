"""Gyre attraction split by start: inside the det J = 0 diamond |x|+|y| < 0.5 m or outside it.
Coulomb model, side 0.33 m, 10 s, starts uniform in the +-0.45 m box (vecsim default for the gyre)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import triangle_rescore as tr  # noqa: E402
import sim  # noqa: E402
import metrics as mt  # noqa: E402
import vecsim as vs  # noqa: E402

N = 5000
F = vs.make_field("gyre")
P = vs.params(N)
out, _ = vs.simulate(F, "attract", "fit", P, 10.0, seed=0, stiction_model=sim.MODEL, record=True)
rec = out["rec"]
c0 = rec["c"][:, 0]
inside_d = (np.abs(c0[:, 0]) + np.abs(c0[:, 1])) < 0.5
rows = []
for i in range(N):
    r = tr.attraction_row(rec["pos"][i], F.target)
    r["tri_final"] = r["tri_final"] and not bool(out["escaped"][i])
    rt, idx, dist = tr.reached(F, rec["c"][i][-1])
    r.update(reached_target=rt, reached_index=idx, start_r=float(np.hypot(*c0[i])), inside=bool(inside_d[i]),
             cls=mt.classification_scores(rec["J"][i], 1, start=int(0.8 * rec["J"].shape[1]))["cls_acc"],
             det_sign_start=float(np.sign(rec["J"][i, 0, 0] * rec["J"][i, 0, 3] - rec["J"][i, 0, 1] * rec["J"][i, 0, 2])))
    rows.append(r)
import pandas as pd
d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, "results", "triangle_gyre_diamond_split.csv"), index=False)
for name, m in (("inside diamond", d.inside), ("outside diamond", ~d.inside)):
    s = d[m]
    print(f"{name:16s} n={len(s):5d}  center in triangle {100*s.tri_final.mean():5.1f}%  reached a saddle {100*(s.reached_index<0).mean():5.1f}%  "
          f"start det(Jhat)>0 {100*(s.det_sign_start>0).mean():5.1f}%  median settle {s.tri_settle_s.median():.2f} s")
print("by start radius (m):")
b = np.arange(0, 0.65, 0.1)
d["bin"] = pd.cut(d.start_r, b)
print(d.groupby("bin", observed=True).agg(n=("tri_final", "size"), center_pct=("tri_final", lambda x: round(100*x.mean(), 1))).to_string())
