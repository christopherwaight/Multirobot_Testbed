import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from itertools import groupby
import src.control.pentagon_primitives as P
import main_separatrix_traverse as M
import experiments._mc_common as mc
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from variant_q import oecs_separatrix_step_q

N, STEPS = 50, 300
SADDLES = [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5)]

def trial(fn, sx, sy, s_uv, s_p, seed):
    np.random.seed(seed); rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(mc.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
    cl.reset(sx, sy, heading_offset=rng.uniform(0, 2*np.pi))
    cl.measurement_noise_std = s_uv; cl.position_noise_std = s_p
    def prim(c):
        vx, vy = fn(c, v_max=mc.V_MAX, g_perp=1.0, s_trim=0.05, r_band=0.05,
                    g_capture=0.15, s_capture=None)
        return vx * mc.GAIN, vy * mc.GAIN
    exited = False
    for _ in range(STEPS):
        cl.move(prim); cx, cy = cl.get_centroid()
        if abs(cx) > 1.0 or abs(cy) > 0.60: exited = True; break
    h = np.asarray(cl.get_center_history()); end = h[-1]
    d = [x for x in cl.diagnostics if 'mode' in x]
    modes = [x['mode'] for x in d]
    if np.hypot(end[0], end[1] + 0.5) < 0.06: cls = "far saddle"
    elif np.hypot(end[0], end[1] - 0.5) < 0.06: cls = "near saddle"
    elif exited: cls = "exited"
    else: cls = "elsewhere"
    # capture entries away from every saddle
    bad = 0
    for k in range(1, len(modes)):
        if modes[k] == 'CAPTURE' and modes[k-1] != 'CAPTURE':
            p = h[min(k, len(h)-1)]
            if min(np.hypot(p[0]-a, p[1]-b) for a, b in SADDLES) > 0.1: bad += 1
    rel = getattr(cl, '_q_releases', None)
    if rel is None:  # original: count CAPTURE -> other transitions
        rel = sum(1 for k in range(1, len(modes)) if modes[k-1] == 'CAPTURE' and modes[k] != 'CAPTURE')
    pattern = " > ".join(m for m, _ in groupby(modes))
    return cls, rel, bad, pattern, ('CROSS' in modes), ('ACQUIRE' in modes)

for axis, s_uv, s_p in (("sigma_uv = 0.002", 0.002, 0.0), ("sigma_p = 0.002", 0.0, 0.002)):
    print(f"\n===== {axis}, {N} trials per start, {STEPS} steps")
    tot = {"original": Counter(), "variant": Counter()}
    for name, sx, sy in M.STARTS:
        line = f"{name}"
        for label, fn in (("original", P.oecs_separatrix_step), ("variant", oecs_separatrix_step_q)):
            cls_c = Counter(); rels = bads = cross = acq = 0; pats = Counter()
            for t in range(N):
                cls, rel, bad, pat, cr, aq = trial(fn, sx, sy, s_uv, s_p, 1000 + 7919 * t)
                cls_c[cls] += 1; rels += rel > 0; bads += bad > 0; cross += cr; acq += aq; pats[pat] += 1
            tot[label] += cls_c
            line += (f" | {label}: far {cls_c['far saddle']:2d} near {cls_c['near saddle']:2d} "
                     f"exit {cls_c['exited']:2d} else {cls_c['elsewhere']:2d}; "
                     f"released {rels:2d}, off-saddle capture {bads:2d}, CROSS {cross:2d}, ACQ {acq:2d}")
            if label == "variant":
                line += "\n      variant patterns: " + "; ".join(f"{n}x {p}" for p, n in pats.most_common(3))
        print(line)
    for k, c in tot.items():
        print(f"  TOTAL {k}: " + ", ".join(f"{a} {b}" for a, b in c.most_common()))
