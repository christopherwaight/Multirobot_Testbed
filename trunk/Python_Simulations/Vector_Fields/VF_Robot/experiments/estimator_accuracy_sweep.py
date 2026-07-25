"""
estimator_accuracy_sweep.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes the data behind:
    - Fig. \\ref{fig:est_accuracy} (figures/estimator_accuracy_vs_noise.png),
      plotted from these CSVs by plot_estimator_accuracy.py.
    - The empirical verification of Lemma 1 (\\ref{lem:gains}) quoted in the
      "Estimator Accuracy versus Noise" subsection of Results.
    - The formation-radius trade-off claim of Eq. \\ref{eq:rho_star}.
  Outputs: experiments/outputs/estimator_accuracy/
    lemma_gains.csv        empirical vs predicted coefficient noise stds
    noise_sweep.csv        derived-quantity errors vs sigma_uv (nominal scale)
    radius_sweep.csv       derived-quantity errors vs formation scale
    posnoise_coupling.csv  position-noise -> effective measurement noise check

EXPERIMENT
  Static pentagon-plus-center formation (pentagon_small.yaml geometry, no
  control loop). At fixed field locations in the steady double gyre, draw
  Monte Carlo measurement noise per Eq. (meas_noise), fit the quadratic
  model through the same _quadratic_basis / solve path the controllers use,
  and compare:
    (a) per-coefficient noise stds against Lemma 1's closed forms
        sigma_u0 = sigma_uv,
        sigma_ux = sigma_uy   = (2/sqrt(10)) sigma_uv / rho,
        sigma_uxx = sigma_uyy = (8/sqrt(10)) sigma_uv / rho^2,
        sigma_uxy             = (4/sqrt(10)) sigma_uv / rho^2,
        and against the general row-norm prediction sigma * ||row_k(Phi^-1)||;
    (b) errors of the derived navigation quantities D, grad D, H_D against
        the analytic truth (Appendix A closed forms), including the
        noise-free truncation floor;
    (c) the first-order position-noise coupling J(p_i) xi_i of
        Eq. (pos_noise).

Run:
  cd trunk/Python_Simulations/Vector_Fields/VF_Robot
  venv/bin/python3 experiments/estimator_accuracy_sweep.py
"""
import os
import sys
import subprocess
from datetime import datetime, timezone

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (
    _quadratic_basis,
    _get_relative_positions,
    _det_value,
    _det_gradient,
    _det_hessian,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
A = 0.1                       # double gyre amplitude (config/fields/double_gyre.yaml)
SEED = 20260702
N_DRAWS = 200_000             # Monte Carlo draws per cell
N_DRAWS_POS = 20_000          # draws for the position-noise coupling check

# Field locations (world frame): generic strain-region point and a point on
# the separatrix trench. Gains are formation-only; derived-quantity errors
# depend on the local truth magnitudes, hence two locations.
LOCATIONS = {
    "strain_region": (-0.35, -0.20),
    "separatrix":    (0.00, -0.25),
}

SIGMA_UV_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
SIGMA_UV_FIXED = 0.01         # for the radius sweep
SCALE_LEVELS = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
SIGMA_P_LEVELS = [0.005, 0.01]

OUT_DIR = os.path.join(project_root, "experiments", "outputs",
                       "estimator_accuracy")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# ANALYTIC TRUTH (Appendix A of Paper_Draft_2A)
# ============================================================================

def truth_D(x, y):
    xf, yf = x + 1.0, y + 0.5
    return -(np.pi**4 * A**2 / 2.0) * (np.cos(2*np.pi*xf) + np.cos(2*np.pi*yf))

def truth_grad(x, y):
    xf, yf = x + 1.0, y + 0.5
    return np.pi**5 * A**2 * np.array([np.sin(2*np.pi*xf), np.sin(2*np.pi*yf)])

def truth_hess(x, y):
    xf, yf = x + 1.0, y + 0.5
    return 2*np.pi**6 * A**2 * np.diag([np.cos(2*np.pi*xf), np.cos(2*np.pi*yf)])

def truth_jacobian(x, y):
    xf, yf = x + 1.0, y + 0.5
    sx, cx = np.sin(np.pi*xf), np.cos(np.pi*xf)
    sy, cy = np.sin(np.pi*yf), np.cos(np.pi*yf)
    return np.pi**2 * A * np.array([[-cx*cy,  sx*sy],
                                    [-sx*sy,  cx*cy]])

# ============================================================================
# VECTORIZED DERIVED QUANTITIES
# (array forms of _det_value/_det_gradient/_det_hessian; cross-checked
#  against those functions on a subsample below)
# ============================================================================

def derived_batch(Tu, Tv):
    """Tu, Tv: (N, 6) coefficient arrays -> D (N,), G (N,2), H (N,2,2)."""
    ux, uy, uxx, uxy, uyy = Tu[:, 1], Tu[:, 2], Tu[:, 3], Tu[:, 4], Tu[:, 5]
    vx, vy, vxx, vxy, vyy = Tv[:, 1], Tv[:, 2], Tv[:, 3], Tv[:, 4], Tv[:, 5]
    D = ux*vy - uy*vx
    G = np.stack([uxx*vy + ux*vxy - uxy*vx - uy*vxx,
                  uxy*vy + ux*vyy - uyy*vx - uy*vxy], axis=1)
    H = np.empty((Tu.shape[0], 2, 2))
    H[:, 0, 0] = 2.0*(uxx*vxy - uxy*vxx)
    H[:, 0, 1] = uxx*vyy - uyy*vxx
    H[:, 1, 0] = H[:, 0, 1]
    H[:, 1, 1] = 2.0*(uxy*vyy - uyy*vxy)
    return D, G, H

# ============================================================================
# HELPERS
# ============================================================================

def metadata_lines(extra=""):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root, text=True).strip()
    except Exception:
        commit = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"# generated_by: experiments/estimator_accuracy_sweep.py\n"
            f"# git_commit: {commit}\n# date: {stamp}\n"
            f"# seed: {SEED}  n_draws: {N_DRAWS}\n"
            f"# formation: {FORMATION_CONFIG}\n{extra}")

def clean_readings(centroid, rel):
    """Noise-free (u, v) at centroid + rel positions."""
    u = np.empty(6)
    v = np.empty(6)
    for i in range(6):
        u[i], v[i] = double_gyre_static(centroid[0] + rel[i, 0],
                                        centroid[1] + rel[i, 1])
    return u, v

def fit_stats(centroid, rel, sigma_uv, rng):
    """Monte Carlo error stats of D, grad D, H_D at one (location, geometry).

    Returns dict of stats plus the truncation-only (noise-free) errors.
    """
    Phi = np.array([_quadratic_basis(rel[j, 0], rel[j, 1]) for j in range(6)])
    Phi_inv = np.linalg.inv(Phi)
    u_c, v_c = clean_readings(centroid, rel)
    tu_clean = Phi_inv @ u_c
    tv_clean = Phi_inv @ v_c

    Tu = tu_clean + (rng.standard_normal((N_DRAWS, 6)) * sigma_uv) @ Phi_inv.T
    Tv = tv_clean + (rng.standard_normal((N_DRAWS, 6)) * sigma_uv) @ Phi_inv.T

    D, G, H = derived_batch(Tu, Tv)

    # Cross-check the vectorized forms against the controller's own functions
    for k in range(50):
        assert np.isclose(D[k], _det_value(Tu[k], Tv[k]))
        assert np.allclose(G[k], _det_gradient(Tu[k], Tv[k]))
        assert np.allclose(H[k], _det_hessian(Tu[k], Tv[k]))

    x, y = centroid
    eD = np.abs(D - truth_D(x, y))
    eG = np.linalg.norm(G - truth_grad(x, y), axis=1)
    eH = np.linalg.norm(H - truth_hess(x, y), axis=(1, 2))

    D0, G0, H0 = derived_batch(tu_clean[None, :], tv_clean[None, :])
    trunc = {
        "detJ":   abs(D0[0] - truth_D(x, y)),
        "gradD":  np.linalg.norm(G0[0] - truth_grad(x, y)),
        "hessD":  np.linalg.norm(H0[0] - truth_hess(x, y)),
    }
    truth_mag = {
        "detJ":   abs(truth_D(x, y)),
        "gradD":  np.linalg.norm(truth_grad(x, y)),
        "hessD":  np.linalg.norm(truth_hess(x, y)),
    }
    out = []
    for name, e in (("detJ", eD), ("gradD", eG), ("hessD", eH)):
        out.append({
            "quantity": name,
            "mean": e.mean(), "median": np.median(e),
            "q25": np.percentile(e, 25), "q75": np.percentile(e, 75),
            "p95": np.percentile(e, 95),
            "truncation_only": trunc[name],
            "truth_magnitude": truth_mag[name],
        })
    return out

# ============================================================================
# MAIN
# ============================================================================

def main():
    rng = np.random.default_rng(SEED)

    field = AnalyticalField(double_gyre_static)
    cluster = PentagonCluster(FORMATION_CONFIG, field)
    loc0 = LOCATIONS["strain_region"]
    cluster.reset(loc0[0], loc0[1])
    rel = _get_relative_positions(cluster)
    centroid = cluster.get_centroid()

    # ---------------- geometry report -----------------------------------
    dists = np.linalg.norm(rel, axis=1)
    center_idx = int(np.argmin(dists))
    ring = np.delete(dists, center_idx)
    rho = ring.mean()
    print("Formation geometry (pentagon_small.yaml):")
    print(f"  robot distances from centroid: {dists.round(4)}")
    print(f"  center robot index {center_idx}, dist {dists[center_idx]:.2e}")
    print(f"  ring radius rho = {rho:.4f}  (spread {np.ptp(ring):.2e})")

    # ---------------- 1. Lemma gains ------------------------------------
    Phi = np.array([_quadratic_basis(rel[j, 0], rel[j, 1]) for j in range(6)])
    Phi_inv = np.linalg.inv(Phi)
    sigma = SIGMA_UV_FIXED
    coef_err = (rng.standard_normal((N_DRAWS, 6)) * sigma) @ Phi_inv.T
    emp_std = coef_err.std(axis=0)
    rownorm_pred = sigma * np.linalg.norm(Phi_inv, axis=1)
    lemma_pred = sigma * np.array([
        1.0,
        2/np.sqrt(10)/rho, 2/np.sqrt(10)/rho,
        8/np.sqrt(10)/rho**2, 4/np.sqrt(10)/rho**2, 8/np.sqrt(10)/rho**2,
    ])
    names = ["u0", "ux", "uy", "uxx", "uxy", "uyy"]
    path = os.path.join(OUT_DIR, "lemma_gains.csv")
    with open(path, "w") as f:
        f.write(metadata_lines(f"# sigma_uv: {sigma}  rho: {rho:.6f}\n"))
        f.write("coeff,empirical_std,rownorm_pred,lemma_pred\n")
        for n, e, r, l in zip(names, emp_std, rownorm_pred, lemma_pred):
            f.write(f"{n},{e:.6e},{r:.6e},{l:.6e}\n")
    print(f"\nLemma gain check (sigma_uv={sigma}):")
    for n, e, r, l in zip(names, emp_std, rownorm_pred, lemma_pred):
        print(f"  {n:>4}: empirical {e:.5f}  rownorm {r:.5f}  lemma {l:.5f}"
              f"  (emp/lemma = {e/l:.3f})")

    # ---------------- 2. noise sweep at nominal scale -------------------
    path = os.path.join(OUT_DIR, "noise_sweep.csv")
    with open(path, "w") as f:
        f.write(metadata_lines())
        f.write("location,sigma_uv,quantity,mean,median,q25,q75,p95,"
                "truncation_only,truth_magnitude\n")
        for loc_name, loc in LOCATIONS.items():
            cluster.reset(loc[0], loc[1])
            rel_l = _get_relative_positions(cluster)
            cen_l = cluster.get_centroid()
            for s_uv in SIGMA_UV_LEVELS:
                for row in fit_stats(cen_l, rel_l, s_uv, rng):
                    f.write(f"{loc_name},{s_uv},{row['quantity']},"
                            f"{row['mean']:.6e},{row['median']:.6e},"
                            f"{row['q25']:.6e},{row['q75']:.6e},"
                            f"{row['p95']:.6e},{row['truncation_only']:.6e},"
                            f"{row['truth_magnitude']:.6e}\n")
    print(f"\nWrote {path}")

    # ---------------- 3. radius sweep at fixed sigma --------------------
    cluster.reset(loc0[0], loc0[1])
    rel_base = _get_relative_positions(cluster)
    cen = cluster.get_centroid()
    path = os.path.join(OUT_DIR, "radius_sweep.csv")
    with open(path, "w") as f:
        f.write(metadata_lines(f"# sigma_uv fixed: {SIGMA_UV_FIXED}\n"
                               f"# location: strain_region {loc0}\n"))
        f.write("scale,rho,sigma_uv,quantity,mean,median,q25,q75,p95,"
                "truncation_only,truth_magnitude\n")
        for s in SCALE_LEVELS:
            rel_s = rel_base * s
            for row in fit_stats(cen, rel_s, SIGMA_UV_FIXED, rng):
                f.write(f"{s},{rho*s:.6f},{SIGMA_UV_FIXED},{row['quantity']},"
                        f"{row['mean']:.6e},{row['median']:.6e},"
                        f"{row['q25']:.6e},{row['q75']:.6e},"
                        f"{row['p95']:.6e},{row['truncation_only']:.6e},"
                        f"{row['truth_magnitude']:.6e}\n")
    print(f"Wrote {path}")

    # ---------------- 4. position-noise coupling ------------------------
    path = os.path.join(OUT_DIR, "posnoise_coupling.csv")
    with open(path, "w") as f:
        f.write(metadata_lines(f"# location: strain_region {loc0}\n"
                               f"# n_draws: {N_DRAWS_POS}\n"))
        f.write("robot,sigma_p,empirical_std_u,empirical_std_v,"
                "pred_std_u,pred_std_v\n")
        for s_p in SIGMA_P_LEVELS:
            for i in range(6):
                px, py = cen[0] + rel_base[i, 0], cen[1] + rel_base[i, 1]
                u0, v0 = double_gyre_static(px, py)
                du = np.empty(N_DRAWS_POS)
                dv = np.empty(N_DRAWS_POS)
                for k in range(N_DRAWS_POS):
                    xi = rng.standard_normal(2) * s_p
                    uu, vv = double_gyre_static(px + xi[0], py + xi[1])
                    du[k], dv[k] = uu - u0, vv - v0
                J = truth_jacobian(px, py)
                pred_u = s_p * np.linalg.norm(J[0])
                pred_v = s_p * np.linalg.norm(J[1])
                f.write(f"{i},{s_p},{du.std():.6e},{dv.std():.6e},"
                        f"{pred_u:.6e},{pred_v:.6e}\n")
    print(f"Wrote {path}")

    # ---------------- 5. Hessian eigenvector survival --------------------
    # Data behind Remark (rem:eigvec): noise-free fits; angle between the
    # eigenvectors of H_hat and H_true, and both eigenvalue pairs.
    eig_points = [
        ("sep_mid",        0.00, -0.25),
        ("sep_off_0.05",   0.05, -0.25),
        ("sep_near_crest", 0.00, -0.05),
        ("sep_near_well",  0.00, -0.45),
        ("generic_strain", -0.35, -0.20),
    ]
    path = os.path.join(OUT_DIR, "eigvec_check.csv")
    with open(path, "w") as f:
        f.write(metadata_lines("# noise-free fits at nominal scale\n"))
        f.write("point,x,y,ang_w1_deg,ang_w2_deg,"
                "lam1_hat,lam2_hat,lam1_true,lam2_true\n")
        for name, px, py in eig_points:
            cluster.reset(px, py)
            rel_p = _get_relative_positions(cluster)
            cen_p = cluster.get_centroid()
            Phi_p = np.array([_quadratic_basis(rel_p[j, 0], rel_p[j, 1])
                              for j in range(6)])
            u_p, v_p = clean_readings(cen_p, rel_p)
            tu = np.linalg.solve(Phi_p, u_p)
            tv = np.linalg.solve(Phi_p, v_p)
            H_hat = _det_hessian(tu, tv)
            H_tru = truth_hess(cen_p[0], cen_p[1])
            wh, Vh = np.linalg.eigh(H_hat)
            wt, Vt = np.linalg.eigh(H_tru)
            angs = [np.degrees(np.arccos(min(abs(Vh[:, k] @ Vt[:, k]), 1.0)))
                    for k in range(2)]
            f.write(f"{name},{px},{py},{angs[0]:.4f},{angs[1]:.4f},"
                    f"{wh[0]:.4f},{wh[1]:.4f},{wt[0]:.4f},{wt[1]:.4f}\n")
    print(f"Wrote {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
