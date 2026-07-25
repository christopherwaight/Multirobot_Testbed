"""
baseline_fixed_weights.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  the fixed-weight baseline comparison of Section III / VII
          (SIM-3 of the T-RO hardening plan) and the data for the
          deformation-bias figure.
  Reads:  nothing. Writes experiments/outputs/baseline_fixed_weights/
          results.csv and prints the PASS gate evaluation.

EXPERIMENT
  Implements the fixed symmetric-weight gradient and Hessian recovery
  of Brinon-Arranz, Renzaglia, and Schenato (T-RO 2019, reference [15])
  on the pentagon-plus-center formation, applied per channel to u and
  v, then assembles D, grad D, and the deviatoric part of H_D through
  the same product-rule code the paper's estimator uses.  Their
  construction, specialized to N = 5 robots on a ring of radius D_r
  plus one robot at the center c:
    gradient (their Theorem 1):
        grad_hat = (2 / (N D_r^2)) sum_i sigma(r_i) (r_i - c)
    Hessian (their Theorem 2):
        K = (16 / (N D_r^4)) sum_i (sigma(r_i) - sigma(c))
                                    (r_i - c)(r_i - c)^T
        solve  3 H + R_{pi/2} H R_{pi/2}^T = K
        i.e.   H11 = (3 K11 - K22)/8,  H22 = (3 K22 - K11)/8,
               H12 = K12 / 2.
  The fixed weights use the NOMINAL formation geometry; the robots
  sample the field at their TRUE (perturbed) positions.  The paper's
  estimator solves the quadratic fit at the measured positions instead.

  Grid: two test points (near-saddle trench point, generic strain
  point) x per-robot displacement std delta in {0, 0.02, 0.05, 0.10,
  0.20} rho (per axis) x sigma_uv in {0, 0.01}, N_TRIALS draws each.
  Ground truth from high-order finite differences of the analytic
  field, so both estimators carry the same O(M3 rho) truncation.

  PASS gate (plan SIM-3): at delta = 0.10 rho, near the saddle, at
  sigma_uv = 0.01, the fixed-weight deviatoric-Hessian bias exceeds
  3x its own noise std while the measured-position LS bias stays
  below 1x.

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/baseline_fixed_weights.py
"""
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (_fit_quadratic, _det_value,
                                             _det_gradient, _det_hessian)

RHO = 0.075
N_RING = 5
PHI0 = np.pi / 2
TEST_POINTS = {
    "near_saddle": (0.0, -0.45),
    "generic_strain": (-0.20, -0.30),
}
DELTAS = [0.0, 0.02, 0.05, 0.10, 0.20]      # per-axis std, units of rho
SIGMAS = [0.0, 0.01]
N_TRIALS = 2000
SEED = 20260718
FD_H = 1e-5


def field(x, y):
    u, v = double_gyre_static(x, y, 0.0)
    return np.array([u, v])


def true_theta(cx, cy):
    """[f, fx, fy, fxx, fxy, fyy] per channel by central differences."""
    h = FD_H
    f0 = field(cx, cy)
    fx = (field(cx + h, cy) - field(cx - h, cy)) / (2 * h)
    fy = (field(cx, cy + h) - field(cx, cy - h)) / (2 * h)
    fxx = (field(cx + h, cy) - 2 * f0 + field(cx - h, cy)) / h**2
    fyy = (field(cx, cy + h) - 2 * f0 + field(cx, cy - h)) / h**2
    fxy = (field(cx + h, cy + h) - field(cx + h, cy - h)
           - field(cx - h, cy + h) + field(cx - h, cy - h)) / (4 * h**2)
    tu = [f0[0], fx[0], fy[0], fxx[0], fxy[0], fyy[0]]
    tv = [f0[1], fx[1], fy[1], fxx[1], fxy[1], fyy[1]]
    return np.array(tu), np.array(tv)


NOMINAL_REL = np.array(
    [[0.0, 0.0]] + [[RHO * np.cos(PHI0 + 2 * np.pi * k / N_RING),
                     RHO * np.sin(PHI0 + 2 * np.pi * k / N_RING)]
                    for k in range(N_RING)])


def fixed_weight_theta(readings_ring, reading_center):
    """[15]'s recovery -> theta = [f_c, grad, H] with nominal weights."""
    rel = NOMINAL_REL[1:]                       # ring only, ideal positions
    grad = (2.0 / (N_RING * RHO**2)) * (readings_ring[:, None] * rel).sum(0)
    dev = readings_ring - reading_center
    K = (16.0 / (N_RING * RHO**4)) * sum(
        d * np.outer(r, r) for d, r in zip(dev, rel))
    h11 = (3 * K[0, 0] - K[1, 1]) / 8.0
    h22 = (3 * K[1, 1] - K[0, 0]) / 8.0
    h12 = K[0, 1] / 2.0
    return np.array([reading_center, grad[0], grad[1], h11, h12, h22])


def ls_theta(rel_positions, readings):
    """The paper's estimator: quadratic LS at the measured positions."""
    return np.array(_fit_quadratic(rel_positions, readings))


def dev_vec(H):
    """Deviatoric part of a symmetric 2x2 as (a, b) = ((H11-H22)/2, H12)."""
    return np.array([(H[0, 0] - H[1, 1]) / 2.0, H[0, 1]])


def eig_angle(H):
    """Orientation of the eigenvector of the smaller eigenvalue, mod pi."""
    w, V = np.linalg.eigh(H)
    v = V[:, 0]
    return np.arctan2(v[1], v[0]) % np.pi


def angle_err(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def run_cell(point, delta, sigma, rng):
    cx, cy = TEST_POINTS[point]
    tu, tv = true_theta(cx, cy)
    D_true = _det_value(tu, tv)
    g_true = np.array(_det_gradient(tu, tv))
    H_true = np.array(_det_hessian(tu, tv))
    dev_true = dev_vec(H_true)
    ang_true = eig_angle(H_true)

    est = {"fixed": {"D": [], "g": [], "dev": [], "ang": []},
           "ls": {"D": [], "g": [], "dev": [], "ang": []}}
    for _ in range(N_TRIALS):
        pert = rng.normal(0.0, delta * RHO, size=(6, 2))
        pos = NOMINAL_REL + pert
        readings = np.array([field(cx + px, cy + py)
                             + rng.normal(0.0, sigma, size=2)
                             for px, py in pos])

        for name in ("fixed", "ls"):
            if name == "fixed":
                theta_u = fixed_weight_theta(readings[1:, 0], readings[0, 0])
                theta_v = fixed_weight_theta(readings[1:, 1], readings[0, 1])
            else:
                theta_u = ls_theta(pos, readings[:, 0])
                theta_v = ls_theta(pos, readings[:, 1])
            H = np.array(_det_hessian(theta_u, theta_v))
            est[name]["D"].append(_det_value(theta_u, theta_v))
            est[name]["g"].append(np.array(_det_gradient(theta_u, theta_v)))
            est[name]["dev"].append(dev_vec(H))
            est[name]["ang"].append(angle_err(eig_angle(H), ang_true))

    rows = []
    for name in ("fixed", "ls"):
        D = np.array(est[name]["D"])
        g = np.stack(est[name]["g"])
        dv = np.stack(est[name]["dev"])
        ang = np.array(est[name]["ang"])
        bias_dev = np.linalg.norm(dv.mean(0) - dev_true)
        std_dev = float(np.sqrt(dv.var(0).sum()))
        rows.append({
            "point": point, "delta": delta, "sigma_uv": sigma,
            "estimator": name,
            "bias_D": float(D.mean() - D_true),
            "rmse_D": float(np.sqrt(np.mean((D - D_true)**2))),
            "bias_grad": float(np.linalg.norm(g.mean(0) - g_true)),
            "rmse_grad": float(np.sqrt(np.mean(
                np.sum((g - g_true)**2, axis=1)))),
            "bias_dev_H": float(bias_dev),
            "std_dev_H": std_dev,
            "bias_over_std": float(bias_dev / std_dev) if std_dev > 0
            else float("inf"),
            "mean_eig_angle_err_deg": float(np.degrees(ang.mean())),
        })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    out_dir = os.path.join(project_root, "experiments", "outputs",
                           "baseline_fixed_weights")
    os.makedirs(out_dir, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    all_rows = []
    for point in TEST_POINTS:
        for delta in DELTAS:
            for sigma in SIGMAS:
                rows = run_cell(point, delta, sigma, rng)
                all_rows.extend(rows)
                for r in rows:
                    print(f"{point} delta={delta:g}rho sigma={sigma:g} "
                          f"{r['estimator']:5s}: bias_dev_H="
                          f"{r['bias_dev_H']:.4g} std={r['std_dev_H']:.4g} "
                          f"ratio={r['bias_over_std']:.2f} "
                          f"ang_err={r['mean_eig_angle_err_deg']:.2f} deg",
                          flush=True)

    cols = list(all_rows[0].keys())
    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        f.write("# generated_by: experiments/baseline_fixed_weights.py\n")
        f.write(f"# git_commit: {commit}\n")
        f.write(f"# date: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# trials_per_cell: {N_TRIALS}  seed: {SEED}"
                f"  rho: {RHO}  per_axis_pert_std: delta*rho\n")
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # PASS gate
    def find(point, delta, sigma, est):
        for r in all_rows:
            if (r["point"] == point and r["delta"] == delta
                    and r["sigma_uv"] == sigma and r["estimator"] == est):
                return r
        return None

    fx = find("near_saddle", 0.10, 0.01, "fixed")
    ls = find("near_saddle", 0.10, 0.01, "ls")
    print("\nPASS gate (near_saddle, delta=0.10rho, sigma_uv=0.01):",
          flush=True)
    print(f"  fixed-weight bias/std = {fx['bias_over_std']:.2f} "
          f"(gate: > 3)", flush=True)
    print(f"  measured-LS  bias/std = {ls['bias_over_std']:.2f} "
          f"(gate: < 1)", flush=True)
    verdict = "PASS" if (fx["bias_over_std"] > 3.0
                         and ls["bias_over_std"] < 1.0) else "FAIL"
    print(f"  verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
