"""
hop_rate_instrumentation.py

Regenerates the Draft_6a sec:disc_noise numbers that had no committed
generator (revision/items.yaml HOP_RATE_PROVENANCE): the channel-wise
clean-value injection ladder (42.7% -> 45.0% -> 100% / 95.7%) and the
tangent hop rate (1.1% at sigma_uv=0.001, 16.5% at 0.002), plus the start
relocation to (0, 0.15).

Mechanism: pentagon_primitives.oecs_separatrix_step now takes optional
override_seed / override_argmax kwargs (default None, no effect on any
existing call site). This script supplies the TRUE analytic double-gyre
flow / grad(s1) / eigenframe at the cluster's current centroid for
whichever channel is being isolated, while the other channel is left to
the noisy fit -- exactly "supplying one channel at a time from noise-free
values" (Draft_6a sec:disc_noise).

Scoring follows rescore_single_target.py / mc_sweep_flip_resolution.py:
the trial runs to EITHER saddle (mc.run_trial's own stop condition), and
success is scored AFTERWARD against the single FAR saddle (0, -0.5) from
final_x/final_y -- the near saddle is not an easier success, it is the
tangent-sign-flip failure mode this whole experiment is about.

Hop rate is read from the tangent_id field _log('TRACK', tangent_id=...)
now records in cluster.diagnostics (added alongside this script): a hop is
a TRACK-to-TRACK step where tangent_id flips between e1 and e2.
"""
import os
import sys
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VFR = os.path.dirname(_HERE)
_PAPER_SCRIPTS = os.path.abspath(os.path.join(_VFR, "..", "..", "..", "..",
                                              "Paper_Writing", "Separatrix_and_OW_Paper", "scripts"))
sys.path.insert(0, _VFR)
sys.path.insert(0, _PAPER_SCRIPTS)
os.chdir(_VFR)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import oecs_separatrix_step
import experiments._mc_common as mc

from verify_estimator_bias import field_derivs  # single source of analytic truth

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
TARGET_BOTH = [(0.0, 0.5), (0.0, -0.5)]
SADDLE_FAR = (0.0, -0.5)
SADDLE_CONTACT_D = 0.06
Y_EXIT = 0.60


def true_flow(x, y):
    d = field_derivs(x, y)
    return np.array([d["u"], d["v"]])


def true_argmax(x, y):
    d = field_derivs(x, y)
    ux, uy, vx, vy = d["ux"], d["uy"], d["vx"], d["vy"]
    uxx, uxy, uyy = d["uxx"], d["uxy"], d["uyy"]
    vxx, vxy, vyy = d["vxx"], d["vxy"], d["vyy"]
    mu = 0.5 * (ux + vy)
    a = 0.5 * (ux - vy)
    b = 0.5 * (uy + vx)
    r = float(np.hypot(a, b))
    grad_mu = 0.5 * np.array([uxx + vxy, uxy + vyy])
    grad_a = 0.5 * np.array([uxx - vxy, uxy - vyy])
    grad_b = 0.5 * np.array([uxy + vxx, uyy + vxy])
    grad_s1 = grad_mu - (a * grad_a + b * grad_b) / max(r, 1e-12)
    S = np.array([[ux, b], [b, vy]])
    _, V = np.linalg.eigh(S)
    return grad_s1, V[:, 0], V[:, 1]


def make_prim(clean_seed=False, clean_argmax=False, capture_diag=None):
    def prim(c):
        override_seed = override_argmax = None
        if clean_seed or clean_argmax:
            cx, cy = c.get_centroid()
            if clean_seed:
                override_seed = true_flow(cx, cy)
            if clean_argmax:
                override_argmax = true_argmax(cx, cy)
        if capture_diag is not None and c.diagnostics is not capture_diag:
            # PentagonCluster.__init__/reset already set c.diagnostics = [];
            # replace it with OUR list (same object) so appends land here.
            c.diagnostics = capture_diag
        vx, vy = oecs_separatrix_step(c, v_max=mc.V_MAX, g_perp=G_PERP,
                                      s_trim=S_TRIM, r_band=R_BAND,
                                      g_capture=G_CAPTURE, s_capture=None,
                                      override_seed=override_seed,
                                      override_argmax=override_argmax)
        return vx * mc.GAIN, vy * mc.GAIN
    return prim


def run_scored(seed, sigma_uv, start, clean_seed=False, clean_argmax=False,
               want_hops=False):
    diag = [] if want_hops else None
    spec = {"sigma_uv": sigma_uv, "sigma_p": 0.0, "start": start,
           "target": TARGET_BOTH, "y_exit": Y_EXIT, "seed": seed,
           "primitive": make_prim(clean_seed, clean_argmax, capture_diag=diag)}
    row = mc.run_trial(spec)
    fx, fy = row["final_x"], row["final_y"]
    d_far = np.hypot(fx - SADDLE_FAR[0], fy - SADDLE_FAR[1])
    reached_far = bool(d_far < SADDLE_CONTACT_D and not row["collapsed"])
    hops = None
    if want_hops and diag:
        hops, prev_id, n_track = 0, None, 0
        for entry in diag:
            if entry.get("mode") != "TRACK":
                continue
            n_track += 1
            tid = entry.get("tangent_id")
            if prev_id is not None and tid != prev_id:
                hops += 1
            prev_id = tid
        hops = (hops, n_track)
    return reached_far, hops


def sweep_success(sigma_uv_vals, n_trials, start, clean_seed=False, clean_argmax=False,
                  seed0=0):
    out = {}
    for sigma_uv in sigma_uv_vals:
        n_success = 0
        for i in range(n_trials):
            reached, _ = run_scored(seed0 + i, sigma_uv, start, clean_seed, clean_argmax)
            n_success += int(reached)
        out[sigma_uv] = n_success / n_trials
    return out


def sweep_hop_rate(sigma_uv_vals, n_trials, start, seed0=100000):
    out = {}
    for sigma_uv in sigma_uv_vals:
        total_hops, total_track = 0, 0
        for i in range(n_trials):
            _, hops = run_scored(seed0 + i, sigma_uv, start, want_hops=True)
            if hops:
                total_hops += hops[0]
                total_track += hops[1]
        out[sigma_uv] = total_hops / max(total_track, 1)
    return out


def main():
    n_trials = int(os.environ.get("HOP_RATE_N_TRIALS", 2000))
    print(f"n_trials = {n_trials} (set HOP_RATE_N_TRIALS to change; "
          f"Draft_6a's own acquisition sweeps use 10,000)")

    print("\n=== Channel-wise clean-value injection, start (0, 0.35) ===")
    baseline_002 = sweep_success([0.002], n_trials, mc.FIXED_START)[0.002]
    clean_seed_002 = sweep_success([0.002], n_trials, mc.FIXED_START, clean_seed=True)[0.002]
    clean_argmax_002 = sweep_success([0.002], n_trials, mc.FIXED_START, clean_argmax=True)[0.002]
    clean_argmax_005 = sweep_success([0.005], n_trials, mc.FIXED_START, clean_argmax=True)[0.005]
    print(f"  baseline @ 0.002:            {baseline_002:.1%}  (paper: 42.7%)")
    print(f"  clean seed @ 0.002:          {clean_seed_002:.1%}  (paper: 45.0%)")
    print(f"  clean argmax @ 0.002:        {clean_argmax_002:.1%}  (paper: 100%)")
    print(f"  clean argmax @ 0.005:        {clean_argmax_005:.1%}  (paper: 95.7%)")

    print("\n=== Hop rate, start (0, 0.35) ===")
    hop_rates = sweep_hop_rate([0.001, 0.002], n_trials, mc.FIXED_START)
    for sigma, rate in hop_rates.items():
        print(f"  sigma_uv={sigma}: hop rate {rate:.1%}")
    print("  (paper: 1.1% at 0.001, 16.5% at 0.002)")

    print("\n=== Start relocation to (0, 0.15): 50% crossing shift ===")
    sigma_grid_orig = [0.0015, 0.002, 0.0025, 0.003]
    sigma_grid_new = [0.0025, 0.003, 0.0035, 0.004]
    orig = sweep_success(sigma_grid_orig, n_trials, mc.FIXED_START)
    relocated = sweep_success(sigma_grid_new, n_trials, (0.0, 0.15))
    for sigma, rate in orig.items():
        print(f"  (0,0.35) sigma_uv={sigma}: {rate:.1%}")
    for sigma, rate in relocated.items():
        print(f"  (0,0.15) sigma_uv={sigma}: {rate:.1%}")

    out = {
        "n_trials": n_trials,
        "clean_channel_injection": {
            "baseline_0.002": baseline_002,
            "clean_seed_0.002": clean_seed_002,
            "clean_argmax_0.002": clean_argmax_002,
            "clean_argmax_0.005": clean_argmax_005,
        },
        "hop_rate": {f"sigma_{str(k).replace('.', '_')}": v for k, v in hop_rates.items()},
        "start_relocation": {
            "orig_start_0_35": {str(k): v for k, v in orig.items()},
            "relocated_start_0_15": {str(k): v for k, v in relocated.items()},
        },
    }
    out_dir = os.path.join(_HERE, "outputs", "mc_oecs_traverse")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hop_rate.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
