"""
verify_estimator_separatrix.py

Open-loop estimator thresholds ALONG THE SEPARATRIX, for the D-tracker cliff sentence
in the Behavior Under Noise subsection. Writes NO figure and NO .tex. Prints a table
for manual review; numbers enter the paper only by hand.

Why this exists: fig_estimator_accuracy_vs_noise sweeps one point, (-0.3, 0.1), which
is rotation-dominated (D = +0.545, inside the Okubo-Weiss diamond). The D tracker's
noise cliff (sigma_uv ~ 0.0079, from (0, 0.35)) is a separatrix result, so comparing it
with that point's threshold was like against unlike. This script repeats the same sweep
(same formation, seed, and draw count) at points on x = 0 that the tracker travels
through, and keeps (-0.3, 0.1) as a control that must reproduce the figure's value.

Per point it reports:
  thr_D, thr_gradD   sigma_uv at which the median relative error of D-hat or
                     grad D-hat first reaches 1 (log-linear interpolation)
  ang_at_cliff       mean H_D-hat principal-eigenvector angle error at the cliff
  relgrad_at_cliff   median relative error of grad D-hat at the cliff

Run from this directory with the VF_Robot venv:
  python3 verify_estimator_separatrix.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_DIR, _git_commit, _git_dirty  # noqa: E402

import numpy as np  # noqa: E402
import fig_estimator_accuracy_vs_noise as F  # noqa: E402
from verify_estimator_bias import true_D, true_grad_D  # noqa: E402

PARAMS = {
    "seed": F.PARAMS["seed"],
    "n_trials": F.PARAMS["n_trials"],
    "rho": F.PARAMS["rho_nominal"],
    "ring_phase_deg": F.PARAMS["ring_phase_deg"],
    "cliff_sigma_uv": F.PARAMS["closed_loop_50pct_sigma_uv"],
    "sigma_uv_vals": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.0075, 0.008,
                      0.009, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1],
    "points": {
        "control_rotation_dominated": [-0.3, 0.1],
        "separatrix_y+0.35_noise_sweep_start": [0.0, 0.35],
        "separatrix_y+0.20": [0.0, 0.2],
        "separatrix_y-0.20": [0.0, -0.2],
        "separatrix_y-0.35": [0.0, -0.35],
    },
}


def _crossing(xs, ys, level=1.0):
    for i in range(1, len(xs)):
        if ys[i - 1] < level <= ys[i]:
            t = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(np.exp(np.log(xs[i - 1])
                                + t * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return None


def main():
    p = PARAMS
    sig = p["sigma_uv_vals"]
    cliff = p["cliff_sigma_uv"]
    rel = F.pentagon_rel_positions(p["rho"], p["ring_phase_deg"])
    results = {}
    print(f"verify_estimator_separatrix  (N = {p['n_trials']} per level, "
          f"seed = {p['seed']}, cliff = {cliff})")
    for name, (x, y) in p["points"].items():
        rng = np.random.default_rng(p["seed"])
        rows = [F.sweep(rel, x, y, s, p["n_trials"], rng) for s in sig]
        eD = [r["D"] for r in rows]
        eG = [r["gD"] for r in rows]
        ang = [r["ang"] for r in rows]
        res = {
            "xy": [x, y],
            "D_true": float(true_D(x, y)),
            "grad_D_true_norm": float(np.linalg.norm(true_grad_D(x, y))),
            "thr_D": _crossing(sig, eD),
            "thr_gradD": _crossing(sig, eG),
            "ang_at_cliff": float(np.interp(cliff, sig, ang)),
            "relgrad_at_cliff": float(np.interp(cliff, sig, eG)),
            "rel_err_D": eD, "rel_err_grad_D": eG, "H_angle_deg": ang,
        }
        results[name] = res
        print(f"  {name:38s} D={res['D_true']:+.4f} |gradD|={res['grad_D_true_norm']:.3f} "
              f"thr_D={res['thr_D']:.4f} thr_gradD={res['thr_gradD']:.4f} "
              f"ang@cliff={res['ang_at_cliff']:.1f} relgrad@cliff={res['relgrad_at_cliff']:.3f}")

    sep = [v for k, v in results.items() if k.startswith("separatrix")]
    summary = {
        "separatrix_thr_gradD_range": [min(v["thr_gradD"] for v in sep),
                                       max(v["thr_gradD"] for v in sep)],
        "separatrix_ang_at_cliff_range": [min(v["ang_at_cliff"] for v in sep),
                                          max(v["ang_at_cliff"] for v in sep)],
    }
    print(f"  separatrix grad-D threshold range: "
          f"{summary['separatrix_thr_gradD_range'][0]:.4f} to "
          f"{summary['separatrix_thr_gradD_range'][1]:.4f}")
    print(f"  separatrix eigenframe error at cliff: "
          f"{summary['separatrix_ang_at_cliff_range'][0]:.1f} to "
          f"{summary['separatrix_ang_at_cliff_range'][1]:.1f} deg")

    out = PAPER_DIR / "scripts" / "verify_estimator_separatrix.json"
    out.write_text(json.dumps({
        "git_commit": _git_commit(), "git_dirty": _git_dirty(),
        "params": p, "results": results, "summary": summary,
    }, indent=2))
    print(f"  json -> {out.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
