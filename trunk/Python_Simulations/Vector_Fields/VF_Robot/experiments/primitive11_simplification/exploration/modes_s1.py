import os, sys
VF = os.getcwd()
sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
import main_separatrix_traverse as M
from itertools import groupby

for name, sx, sy in M.STARTS:
    cl = M.run_traverser(sx, sy)
    modes = [d for d in cl.diagnostics if isinstance(d, dict) and 'mode' in d]
    hist = np.asarray(cl.get_center_history())
    print(f"\n{name} start ({sx}, {sy}), {len(modes)} steps")
    k = 0
    for m, grp in groupby(modes, key=lambda d: (d['mode'], d.get('tangent_id'))):
        g = list(grp)
        c0, c1 = hist[k], hist[min(k + len(g), len(hist) - 1)]
        tag = m[0] + (f" ({m[1]})" if m[1] else "")
        print(f"  steps {k:4d}-{k+len(g)-1:4d}  {tag:12s} centroid ({c0[0]:+.3f},{c0[1]:+.3f}) -> ({c1[0]:+.3f},{c1[1]:+.3f})  s1 {g[0]['s1']:+.3f} -> {g[-1]['s1']:+.3f}  r {g[0]['r']:.3f} -> {g[-1]['r']:.3f}")
        k += len(g)
