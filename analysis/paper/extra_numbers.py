"""Paper numbers not already in the revision_metrics summaries, computed directly (new model, hardware
controller settings). Writes analysis/paper/extra_numbers.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import paths  # noqa: E402
import rigidsim as rs  # noqa: E402
import vecsim as vs  # noqa: E402
from theory_check import k_max, r_star  # noqa: E402

RES = os.path.join(paths.OUT_SIM, "revision_metrics", "results")
IDEAL = dict(alpha=0.0, vs=0.0, vmax=1e9, gain=1.0, delay=0.0)


def noise_effect():
    """Reconstructed fields, robot lag/delay/deadband/cap off: change of mean final error with white noise."""
    out = {}
    for name in ("vortex_nn_2026", "saddle_nn_2026"):
        F = vs.make_field(name)
        base = None
        for sg in (0.0, 0.001, 0.003, 0.01, 0.03):
            M, _ = vs.simulate(F, "attract", "fit", vs.params(1000, sigma_n=sg, **IDEAL), 10.0, seed=0)
            e = float(np.mean(M["final_err"]))
            base = e if base is None else base
            out[f"{name}_{sg}"] = e
        out[f"{name}_max_change_mm"] = 1000 * max(abs(out[f"{name}_{s}"] - base) for s in (0.001, 0.003, 0.01, 0.03))
    return out


def r_min(**over):
    """Orbit radius for r_d = 0 on the linear vortex, mean after 20 s of a 60 s run."""
    P = vs.params(4, rd=0.0, **over)
    st = np.array([[0.05, 0.0], [0.0, 0.05], [-0.1, 0.0], [0.0, -0.1]])
    C, _ = rs.simulate(vs.make_field("vortex"), "orbit", "fit", P, 60.0, rs.triangle(np.zeros(4), np.full(4, 0.3)), st)
    return float(np.nanmean(np.hypot(C[:, 200:, 0], C[:, 200:, 1])))


def poles(k=1.0, g=0.86, L=0.055, dt=0.1):
    """Closed-loop poles of the linear attraction loop (delay under one step, lag ~ 0)."""
    f = L / dt
    a = k * g * dt
    z = np.roots([1, -(1 - a * (1 - f)), a * f])
    return [float(abs(x)) for x in z]


def main():
    X = {}
    X["k_max"] = k_max()
    X["k_max_dt"] = k_max() * 0.1
    X["poles_k1"] = poles()
    X["r_min_default"] = r_min()
    X["r_min_kt0.1"] = r_min(kt=0.1)
    X["r_min_no_delay_no_lag"] = r_min(delay=0.0, alpha=0.0)
    X["r_min_kr2"] = r_min(kr=2.0)
    X["r_star_formula_T0.125"] = float(r_star(0.0, 0.125))
    X["noise"] = noise_effect()
    # linear-field orbit spread across trials (six noise-free fields identical)
    S = json.load(open(os.path.join(RES, "summary.json")))
    lin = ("sink", "source", "vortex", "sinking_vortex", "spewing_vortex", "saddle")
    X["linear_orbit_between_sd_max"] = max(S["sim"]["orbit"][f][rd]["between_sd_w1"] for f in lin for rd in S["sim"]["orbit"][f])
    X["recon_orbit_rd050"] = {f: dict(bias=S["sim"]["orbit"][f]["0.50"]["bias_w1"], ctrl=S["sim"]["orbit"][f]["0.50"]["ctrl_bias_w1"])
                              for f in ("vortex_nn_2026", "saddle_nn_2026")}
    # classification with noise: center (vortex), star nodes (sink, source), saddle
    C = S["classification"]["sim"]
    noisy = [k for k in C if k != "noise-free"][0]
    X["cls_noisy"] = {f: C[noisy][f]["attract"]["acc"] for f in C[noisy]}
    X["cls_noise_free"] = {f: C["noise-free"][f]["attract"]["acc"] for f in C["noise-free"]}
    # nonlinear
    N = json.load(open(os.path.join(RES, "nonlinear_summary.json")))
    X["gyre_loc_mm"] = 1000 * N["attraction"]["gyre"]["all"]["est_ss_mean"]
    X["gyre_size_loc_mm"] = {s: 1000 * v["loc_mean"] for s, v in N["size"]["gyre"].items()}
    X["nonlinear_orbit_bias_rd030"] = {f: N["orbit"][f]["0.30"]["bias_kept"] for f in N["orbit"]}
    sw = N["sweep"]
    rot = [np.mean([sw[k][rd]["bias_kept"] for rd in sw[k] if sw[k][rd].get("bias_kept") is not None]) for k in sw if "rot_chi" in k and "irrot" not in k]
    X["sweep_keys"] = list(sw)[:30]
    X["nonlinear_n_attraction"] = {f: N["attraction"][f]["all"]["n"] for f in N["attraction"]}
    # anisotropic node: V2S failures, median distance from p*
    A = pd.read_csv(os.path.join(RES, "baselines_v2s_attraction.csv"))
    sel = (A.field == "node_aniso") & (A.variant == "v2s")
    X["node_aniso_v2s_median_e"] = float(A[sel].e_final.median())
    X["node_aniso_v2s_median_e_failures"] = float(A[sel & (A.e_final > 0.05)].e_final.median())
    json.dump(X, open(os.path.join(HERE, "extra_numbers.json"), "w"), indent=1, default=float)
    for k, v in X.items():
        print(k, ":", v if not isinstance(v, float) else round(v, 4))


if __name__ == "__main__":
    main()
