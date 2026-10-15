import os, sys, io, contextlib
VF = os.getcwd()
sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
import numpy as np
from itertools import groupby
from collections import Counter
from src.control.pentagon_primitives import oecs_separatrix_step

LOG = []
def logged(call):
    def prim(c):
        if getattr(c, 'diagnostics', None) is None:
            c.diagnostics = []
        n0 = len(c.diagnostics)
        out = call(c)
        new = [d for d in c.diagnostics[n0:] if isinstance(d, dict) and 'mode' in d]
        LOG.append(new[-1]['mode'] if new else '?')
        return out
    return prim

def runs_of(modes):
    out, k = [], 0
    for m, g in groupby(modes):
        n = len(list(g)); out.append((m, k, k + n - 1)); k += n
    return out

which = sys.argv[1]

if which == "noise":
    import experiments._mc_common as mc
    import experiments.mc_sweep_noise_both_trackers as NB
    start = (-0.10, 0.30)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    for axis in ("uv", "p"):
        for sig in (0.002, 0.005, 0.01, 0.02):
            specs = NB.cell_specs("s1", sig if axis == "uv" else 0.0,
                                  sig if axis == "p" else 0.0, N, start)
            ever = Counter(); steps = Counter(); patterns = Counter(); succ = 0
            for s in specs:
                LOG.clear()
                s.pop("tracker"); s["primitive"] = logged(NB._prim_s1)
                with contextlib.redirect_stdout(io.StringIO()):
                    row = mc.run_trial(s)
                steps.update(LOG)
                for m in set(LOG): ever[m] += 1
                patterns[" > ".join(m for m, _, _ in runs_of(LOG))] += 1
            tot = sum(steps.values())
            print(f"\nsigma_{axis} = {sig}: {N} trials, modes ever fired: " +
                  ", ".join(f"{m} {ever[m]}" for m in ("ACQUIRE","PARK","CAPTURE","CROSS","TRACK")))
            print("   share of steps: " + ", ".join(f"{m} {100*steps[m]/tot:.1f}%" for m in steps))
            for p, n in patterns.most_common(4):
                print(f"   {n:4d}x  {p}")

elif which == "rot":
    import experiments.traverse_objectivity_demo as T
    from src.robot.pentagon_cluster import PentagonCluster
    from src.fields.field_types import AnalyticalField
    from src.fields.environments.Double_Gyre import double_gyre_static
    from src.fields.environments.Rotating_Frame import make_rotating_frame
    for label, fn, rot in (("inertial", double_gyre_static, False),
                           ("rotating", make_rotating_frame(double_gyre_static, T.OMEGA_ROT), True)):
        LOG.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            field = AnalyticalField(fn); cl = PentagonCluster(T.FORMATION_CONFIG, field)
        cl.reset(*T.START); field.reset_clock()
        prim = logged(T.traverser_primitive)
        for k in range(T.SIM_STEPS):
            cl.move(prim); field.step(cl.timestep)
            cx, cy = cl.get_centroid()
            if rot:
                th = T.OMEGA_ROT * (k + 1) * T.DT
                cx, cy = (np.cos(th)*cx + np.sin(th)*cy, -np.sin(th)*cx + np.cos(th)*cy)
            if abs(cx) > 1.05 or abs(cy) > 0.60: break
        diag = [d for d in cl.diagnostics if 'mode' in d]
        print(f"\n{label} (Omega = {T.OMEGA_ROT if rot else 0}), start {T.START}, {len(LOG)} steps")
        for m, a, b in runs_of(LOG):
            tid = Counter(d.get('tangent_id') for d in diag[a:b+1] if d.get('tangent_id'))
            print(f"   steps {a:4d}-{b:4d}  {m:8s} s1 {diag[a]['s1']:+.3f} -> {diag[b]['s1']:+.3f}  r {diag[a]['r']:.3f} -> {diag[b]['r']:.3f}  {dict(tid) if tid else ''}")

elif which == "ocean":
    import main_ocean_hfr_2km_shared_start as S
    from src.robot.pentagon_cluster import PentagonCluster
    from src.fields.field_types import AnalyticalField
    from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                             stiction_threshold=S.STICTION_THRESHOLD)
    def p(c):
        vx, vy = oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP, s_trim=S.S_TRIM,
                                      r_band=S.R_BAND, g_capture=S.G_CAPTURE, s_capture=None)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN
    LOG.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        path = S.run_traj(field, cl, logged(p), *S.START)
    diag = [d for d in cl.diagnostics if 'mode' in d][-len(LOG):]
    print(f"params: s_trim {S.S_TRIM}, r_band {S.R_BAND}, g_capture {S.G_CAPTURE}, {len(LOG)} steps (10 min each)")
    for m, a, b in runs_of(LOG):
        tid = Counter(d.get('tangent_id') for d in diag[a:b+1] if d.get('tangent_id'))
        print(f"   steps {a:4d}-{b:4d} ({a/6:5.1f}-{(b+1)/6:5.1f} h) {m:8s} s1 {diag[a]['s1']:+.4f} -> {diag[b]['s1']:+.4f}  r {diag[a]['r']:.4f} -> {diag[b]['r']:.4f} {dict(tid) if tid else ''}")
    print("ends at", path[-1])
    t = [d.get("tangent_id") for d in diag]
    sw = [i for i in range(1, len(t)) if t[i] != t[i-1]]
    print("tangent switches:", len(sw), "at steps", sw[:40])
    print("min r:", min(d["r"] for d in diag), "max s1:", max(d["s1"] for d in diag))
