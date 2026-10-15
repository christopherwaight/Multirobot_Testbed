import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from multiprocessing import Pool
import main_separatrix_traverse as M
import main_ocean_hfr_2km_shared_start as S
import experiments._mc_common as mc
import experiments.mc_sweep_noise_both_trackers as sw
from experiments.rescore_single_target import SADDLE_FAR, SADDLE_CONTACT_D
from experiments.mc_mechanism_starts import STARTS
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step
from dvar import d_step
CQS = (0.5, 0.25, 0.1)
def work(spec):
    spec = dict(spec); cq = spec.pop("cq")
    spec["primitive"] = lambda c: tuple(mc.GAIN * x for x in d_step(c, mc.V_MAX, 'crest', cq, [], False, 'abs'))
    with contextlib.redirect_stdout(io.StringIO()):
        r = mc.run_trial(spec)
    far = np.hypot(r["final_x"] - SADDLE_FAR[0], r["final_y"] - SADDLE_FAR[1]) < SADDLE_CONTACT_D and not r["collapsed"]
    return far, far and bool(r["success_straddle"])
if __name__ == "__main__":
    # 1. clean gyre, run past the saddle (exit 0.60) to see whether it now holds
    for cq in CQS:
        out = []
        for name, sx, sy in M.STARTS:
            with contextlib.redirect_stdout(io.StringIO()):
                cl = PentagonCluster(M.FORMATION_CONFIG, AnalyticalField(M.double_gyre_static))
            cl.reset(sx, sy); log = []
            prim = lambda c: tuple(M.GAIN * x for x in d_step(c, M.V_MAX, 'crest', cq, log, False, 'abs'))
            for k in range(M.SIM_STEPS):
                cl.move(prim); cx, cy = cl.get_centroid()
                if abs(cx) > 1.0 or abs(cy) > 0.60: break
            h = np.asarray(cl.get_center_history()); far = float(np.min(np.hypot(h[:,0], h[:,1] + 0.5)))
            held = np.hypot(h[-1,0], h[-1,1] + 0.5) < 0.06
            out.append(f"{name} {'ok' if far < 0.06 else 'MISS'}{', holds' if held else ', leaves'} (FLOW {Counter(log)['FLOW']}, FB {Counter(log)['CAPTURE']})")
        print(f"gyre clean, crest {cq} rho, |lambda| fallback: " + "; ".join(out), flush=True)
    # 2. SBC
    def run(p):
        with contextlib.redirect_stdout(io.StringIO()):
            field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
            field.config["isotropic_map"] = S.ISOTROPIC_MAP
            c = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
            return S.run_traj(field, c, p, *S.START)
    for cq in CQS:
        log = []; path = run(lambda c: tuple(S.CONTROL_GAIN * x for x in d_step(c, S.V_MAX, 'crest', cq, log, False, 'abs')))
        np.save(f"{os.path.dirname(os.path.abspath(__file__))}/sbc_abs_{cq}.npy", path)
        print(f"SBC crest {cq} rho, |lambda| fallback: end ({path[-1,0]:.3f}N, {path[-1,1]:.3f}), at 24 h ({path[143,0]:.3f}N, {path[143,1]:.3f}) | {dict(Counter(log))}", flush=True)
    # 3. noise, sigma_uv 0.005 and 0.002, 50 per start
    for sig in (0.005, 0.002):
        for cq in CQS:
            specs = []
            for name, (x, y) in STARTS.items():
                for s in sw.cell_specs("D", sig, 0.0, 50, (x, y)):
                    s.pop("tracker"); s["cq"] = cq; specs.append(s)
            with Pool(4) as pool:
                res = pool.map(work, specs, chunksize=5)
            print(f"noise sigma_uv {sig}, crest {cq} rho, |lambda| fallback: success {sum(r[0] for r in res)}/350 retain {sum(r[1] for r in res)}  (current D: {'196/187' if sig == 0.005 else '320/318'})", flush=True)
