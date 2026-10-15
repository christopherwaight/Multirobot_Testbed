"""Clean double-gyre screen: seven starts, paper stop rule (20-step hold).
Also checks that 'merge only' reproduces the repo D tracker bit for bit."""
import time
from collections import Counter
import numpy as np
from screen_common import M, run_held, VARIANTS

t0 = time.time()
hist = {}
for var in ("repo", "merge only") + VARIANTS[1:]:
    print(f"\n{var}")
    for name, sx, sy in M.STARTS:
        cl, modes, out = run_held(var, sx, sy)
        h = np.asarray(cl.get_center_history())
        hist[(var, name)] = h
        c = Counter(modes)
        print(f"  {name} {len(modes):3d} steps, {out:7s} end ({h[-1,0]:+.3f},{h[-1,1]:+.3f}) | "
              + ", ".join(f"{m} {n}" for m, n in sorted(c.items())))
same = [np.array_equal(hist[("repo", n)], hist[("merge only", n)]) for n, _, _ in M.STARTS]
print(f"\nmerge only identical to repo on {sum(same)}/7 starts")
for var in VARIANTS[1:]:
    gap = [np.max(np.linalg.norm(hist[(var, n)][:min(len(hist[(var, n)]), len(hist[("repo", n)]))]
                                 - hist[("repo", n)][:min(len(hist[(var, n)]), len(hist[("repo", n)]))], axis=1))
           for n, _, _ in M.STARTS]
    print(f"{var}: max path gap to repo over common steps " + ", ".join(f"{g:.4f}" for g in gap))
print(f"[{time.time()-t0:.0f} s]")
