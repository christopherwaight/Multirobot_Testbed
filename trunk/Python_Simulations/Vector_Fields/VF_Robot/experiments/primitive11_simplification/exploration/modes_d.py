import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
from itertools import groupby
from collections import Counter
import main_separatrix_traverse as M
MODES = ["FLOW", "FLOW_DRIFT", "SLIDE", "ATTRACT", "ATTRACT_FALLBACK"]
tot = Counter()
for name, sx, sy in M.STARTS:
    with contextlib.redirect_stdout(io.StringIO()):
        cl = M.run_logic_c(sx, sy)
    modes = [d['mode'] for d in cl.diagnostics if 'mode' in d]
    h = np.asarray(cl.get_center_history())
    c = Counter(modes); tot.update(c)
    runs = [(m, len(list(g))) for m, g in groupby(modes)]
    print(f"{name} ({sx:+.2f},{sy:+.2f}) {len(modes)} steps, end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}): "
          + ", ".join(f"{m} {c[m]}" for m in MODES) + f" | {len(runs)} mode switches")
    # where ATTRACT_FALLBACK fires
    idx = [k for k, m in enumerate(modes) if m == 'ATTRACT_FALLBACK']
    if idx:
        ys = h[np.minimum(idx, len(h)-1)]
        print(f"      ATTRACT_FALLBACK at y in [{ys[:,1].min():+.3f}, {ys[:,1].max():+.3f}], x in [{ys[:,0].min():+.3f}, {ys[:,0].max():+.3f}], first at step {idx[0]}")
print("TOTAL", ", ".join(f"{m} {tot[m]}" for m in MODES))
