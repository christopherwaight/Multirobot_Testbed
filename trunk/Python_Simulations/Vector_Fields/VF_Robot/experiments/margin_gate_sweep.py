"""
margin_gate_sweep.py

Implements and sweeps the s1 margin gate from Draft_6a's future-work list
(sec:limitations): hold the TRACK tangent identity when
||grad(s1).e1| - |grad(s1).e2|| falls below gamma_2*sigma_eff/rho^2 (the
noise floor of eq:gain_ladder on the discriminating coefficient), instead
of re-resolving argmax every cycle. Both reviews rank this the single
highest-value addition available: the noise floor is already in closed
form, so this is a few lines of code (added as an opt-in margin_gate=True
kwarg on oecs_separatrix_step, default off, no effect on any existing
call site) plus one sweep.

Scoring matches hop_rate_instrumentation.py / mc_sweep_flip_resolution.py:
single far-saddle target (0, -0.5) from the straddling start (0, 0.35),
success = reached the far saddle without collapsing.

Writes outputs/mc_oecs_traverse/margin_gate_sweep.csv.
"""
import os
import sys
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VFR = os.path.dirname(_HERE)
sys.path.insert(0, _VFR)
os.chdir(_VFR)

from src.control.pentagon_primitives import oecs_separatrix_step, GAMMA2
import experiments._mc_common as mc

G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.05, 0.05, 0.15
RHO = 0.075  # config/formations/pentagon_small.yaml, L_2
TARGET_BOTH = [(0.0, 0.5), (0.0, -0.5)]
SADDLE_FAR = (0.0, -0.5)
SADDLE_CONTACT_D = 0.06
Y_EXIT = 0.60

SIGMA_UV_VALS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01]
N_TRIALS = int(os.environ.get("MARGIN_GATE_N_TRIALS", 10000))


def make_prim(margin_gate, sigma_uv):
    def prim(c):
        vx, vy = oecs_separatrix_step(
            c, v_max=mc.V_MAX, g_perp=G_PERP, s_trim=S_TRIM,
            r_band=R_BAND, g_capture=G_CAPTURE, s_capture=None,
            margin_gate=margin_gate, sigma_eff=sigma_uv, rho=RHO)
        return vx * mc.GAIN, vy * mc.GAIN
    return prim


def sweep(margin_gate):
    out = {}
    for sigma_uv in SIGMA_UV_VALS:
        n_success = 0
        for i in range(N_TRIALS):
            spec = {"sigma_uv": sigma_uv, "sigma_p": 0.0, "start": mc.FIXED_START,
                   "target": TARGET_BOTH, "y_exit": Y_EXIT, "seed": i,
                   "primitive": make_prim(margin_gate, sigma_uv)}
            row = mc.run_trial(spec)
            fx, fy = row["final_x"], row["final_y"]
            d_far = np.hypot(fx - SADDLE_FAR[0], fy - SADDLE_FAR[1])
            reached_far = bool(d_far < SADDLE_CONTACT_D and not row["collapsed"])
            n_success += int(reached_far)
        out[sigma_uv] = n_success / N_TRIALS
    return out


def main():
    print(f"N_TRIALS = {N_TRIALS}, GAMMA2 = {GAMMA2:.4f}, rho = {RHO}")
    print("\n=== Baseline (margin_gate=False) ===")
    baseline = sweep(margin_gate=False)
    for sigma, rate in baseline.items():
        print(f"  sigma_uv={sigma}: {rate:.1%}")

    print("\n=== Margin-gated (margin_gate=True) ===")
    gated = sweep(margin_gate=True)
    for sigma, rate in gated.items():
        print(f"  sigma_uv={sigma}: {rate:.1%}")

    print("\n=== Delta (gated - baseline) ===")
    for sigma in SIGMA_UV_VALS:
        print(f"  sigma_uv={sigma}: {(gated[sigma] - baseline[sigma])*100:+.1f} pts")

    out_dir = os.path.join(_HERE, "outputs", "mc_oecs_traverse")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "margin_gate_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("sigma_uv,success_baseline,success_margin_gate,delta\n")
        for sigma in SIGMA_UV_VALS:
            f.write(f"{sigma},{baseline[sigma]},{gated[sigma]},"
                   f"{gated[sigma]-baseline[sigma]}\n")
    print(f"\nWrote {csv_path}")

    summary = {
        "n_trials": N_TRIALS, "gamma2": GAMMA2, "rho": RHO,
        "baseline": {f"sigma_{str(k).replace('.', '_')}": v for k, v in baseline.items()},
        "margin_gate": {f"sigma_{str(k).replace('.', '_')}": v for k, v in gated.items()},
    }
    json_path = os.path.join(out_dir, "margin_gate_sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
