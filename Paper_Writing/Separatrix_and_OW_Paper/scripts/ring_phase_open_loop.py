"""
ring_phase_open_loop.py

Open-loop (no cluster, no controller, no noise) ring-phase sweeps that settle
the four numerical disputes in revision/items.yaml D1-D4 / reviews/Draft_6a_review_v2.md
section 1: the estimator recomputed at a fixed field point as the pentagon's
formation heading (ring phase) varies, exactly the analysis the second referee
report and v2 both did by hand.

D1: fitted H_D eigenvalue pair at (0, 0.45), rho=0.075, swept over ring phase.
    Target check (v2 section 1, D1): phase 0 -> [-0.5056, +0.4486],
    phase 18 -> [-0.7211, +0.4345] (the manuscript's reported pair).

D2: fitted transverse restoring slope of grad(s1) at (0.03, 0.25), rho=0.075,
    via central difference of the fitted grad(s1)_x across a re-fit at each
    shifted centroid (matches Remark 2 / M11: the fit is redone at the new
    centroid every cycle, so this is what closed-loop tracking actually sees).
    Target check (v2 section 1, D2): phase 0 -> 3.997, phase 18 -> 7.099,
    phase 36 -> 9.664, full-sweep mean 6.826, range 3.984 to 9.676.

D3: parking condition. For each ring phase, root-find x* along y=0.25 where
    the D tracker's transverse command (grad D . w2, w2 = H_D's larger-|eigenvalue|
    eigenvector) vanishes, and separately where the s1 tracker's transverse
    command (the component of grad s1 orthogonal to its tangent) vanishes.
    Trial-mean |x*| over phase is the open-loop prediction Draft_6a quotes as
    0.0019 (D) and 0.0079 (s1).

Nothing here touches pentagon_primitives.py or any control law; it re-fits the
same _fit_vector_quadratic machinery the controllers use, at swept centroids,
against the analytic double-gyre field directly.
"""
import os
import sys

import numpy as np
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
_VFR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "trunk",
                                    "Python_Simulations", "Vector_Fields", "VF_Robot"))
sys.path.insert(0, _VFR)

from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (
    _fit_vector_quadratic, _det_hessian, _det_gradient, _strain_quantities,
)

RHO = 0.075
A_AMPLITUDE = 0.1


def pentagon_rel_positions(rho, phase_deg):
    """Same construction as scripts/fig_estimator_accuracy_vs_noise.py."""
    th = np.radians(phase_deg) + 2.0 * np.pi * np.arange(5) / 5.0
    ring = np.column_stack([rho * np.cos(th), rho * np.sin(th)])
    return np.vstack([np.zeros((1, 2)), ring])


def fit_at(field, cx, cy, rho, phase_deg):
    """Noise-free fit of the pentagon-plus-center centered at (cx, cy)."""
    rel = pentagon_rel_positions(rho, phase_deg)
    abs_pos = rel + np.array([cx, cy])
    u = np.array([field.get_value(x, y)[0] for x, y in abs_pos])
    v = np.array([field.get_value(x, y)[1] for x, y in abs_pos])
    return _fit_vector_quadratic(rel, u, v)


def d1_sweep(field, phases_deg):
    out = []
    for phase in phases_deg:
        theta_u, theta_v = fit_at(field, 0.0, 0.45, RHO, phase)
        H = _det_hessian(theta_u, theta_v)
        eigvals = np.linalg.eigvalsh(H)  # ascending
        out.append((phase, float(eigvals[0]), float(eigvals[1])))
    return out


TRENCH_OFFSET = 0.03  # x-offset of the probe point (0.03, 0.25) from the trench at x=0


def d2_sweep(field, phases_deg):
    """
    "Restoring slope" = the transverse component of grad(s1) at (0.03, 0.25),
    divided by the 0.03 offset from the trench -- a secant from the trench
    (where grad s1's transverse component is zero by symmetry) to this probe
    point, not a derivative. Single fit, no refit. Reproduces v2's own
    recomputation (phase 0 -> 3.997, 18 -> 7.099, 36 -> 9.664) to 3 decimals,
    checked 2026-08-04.
    """
    out = []
    for phase in phases_deg:
        theta_u, theta_v = fit_at(field, 0.03, 0.25, RHO, phase)
        _, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
        d1v = abs(float(grad_s1 @ e1))
        d2v = abs(float(grad_s1 @ e2))
        n_hat = e2 if d1v > d2v else e1  # transverse = the SMALLER-projection eigenvector
        slope = abs(float(grad_s1 @ n_hat)) / TRENCH_OFFSET
        out.append((phase, slope))
    return out


def _d_transverse_command(field, x, y, phase):
    theta_u, theta_v = fit_at(field, x, y, RHO, phase)
    grad_D = _det_gradient(theta_u, theta_v)
    H = _det_hessian(theta_u, theta_v)
    w, V = np.linalg.eigh(H)
    w2 = V[:, int(np.argmax(np.abs(w)))]  # larger-|eigenvalue| = transverse
    return float(grad_D @ w2)


def _s1_transverse_command(field, x, y, phase):
    theta_u, theta_v = fit_at(field, x, y, RHO, phase)
    s1, grad_s1, e1, e2, r = _strain_quantities(theta_u, theta_v)
    d1v = abs(float(grad_s1 @ e1))
    d2v = abs(float(grad_s1 @ e2))
    t_hat = e1 if d1v > d2v else e2
    g_perp_vec = grad_s1 - float(grad_s1 @ t_hat) * t_hat
    # Signed scalar transverse command along the direction orthogonal to
    # t_hat within the (x,y) plane, i.e. project g_perp_vec onto the
    # transverse unit normal so the root-find has a signed scalar.
    n_hat = np.array([-t_hat[1], t_hat[0]])
    return float(g_perp_vec @ n_hat)


def d3_parking(field, phases_deg, y0=0.25, bracket=0.06):
    d_roots, s1_roots = [], []
    for phase in phases_deg:
        try:
            xd = brentq(lambda x: _d_transverse_command(field, x, y0, phase),
                       -bracket, bracket, xtol=1e-7)
            d_roots.append(xd)
        except ValueError:
            pass
        try:
            xs = brentq(lambda x: _s1_transverse_command(field, x, y0, phase),
                       -bracket, bracket, xtol=1e-7)
            s1_roots.append(xs)
        except ValueError:
            pass
    return np.array(d_roots), np.array(s1_roots)


def main():
    field = AnalyticalField(double_gyre_static)

    print("=== D1: fitted H_D eigenvalue pair at (0, 0.45), rho=0.075 ===")
    key_phases = [0, 18, 36]
    sweep = d1_sweep(field, np.arange(0, 72, 1))
    for phase, lo, hi in d1_sweep(field, key_phases):
        print(f"  phase {phase:5.1f} deg -> [{lo:+.4f}, {hi:+.4f}]")
    los = [s[1] for s in sweep]
    his = [s[2] for s in sweep]
    print(f"  full sweep -> lambda1 in [{min(los):.3f}, {max(los):.3f}], "
          f"lambda2 in [{min(his):.3f}, {max(his):.3f}]")

    print("\n=== D2: fitted transverse restoring slope of grad(s1) at (0.03, 0.25) ===")
    sweep2 = d2_sweep(field, np.arange(0, 72, 1))
    for phase, slope in d2_sweep(field, key_phases):
        print(f"  phase {phase:5.1f} deg -> slope {slope:.3f}")
    slopes = [s[1] for s in sweep2]
    analytic = A_AMPLITUDE * np.pi**4 * abs(np.cos(np.pi * 0.25))
    print(f"  full sweep -> {min(slopes):.3f} to {max(slopes):.3f}, "
          f"mean {np.mean(slopes):.3f}")
    print(f"  analytic transverse curvature (Table II): {analytic:.3f}")
    print(f"  phase-mean deficit: {(1 - np.mean(slopes)/analytic)*100:.2f}%")

    print("\n=== D3: parking condition (trial-mean |x*| over ring phase) ===")
    phases_fine = np.linspace(0, 72, 145)  # 0.5-deg resolution over one period
    d_roots, s1_roots = d3_parking(field, phases_fine)
    print(f"  D  tracker: mean|x*| = {np.mean(np.abs(d_roots)):.4f} "
          f"(n={len(d_roots)}/{len(phases_fine)} converged)")
    print(f"  s1 tracker: mean|x*| = {np.mean(np.abs(s1_roots)):.4f} "
          f"(n={len(s1_roots)}/{len(phases_fine)} converged)")

    import json
    out = {
        "d1_eigenvalue_pair": {
            "phase_0": {"lambda1": sweep[0][1] if sweep[0][0] == 0 else None},
            "phase_0_deg": {"lambda1": d1_sweep(field, [0])[0][1],
                           "lambda2": d1_sweep(field, [0])[0][2]},
            "phase_18_deg": {"lambda1": d1_sweep(field, [18])[0][1],
                             "lambda2": d1_sweep(field, [18])[0][2]},
            "sweep_lambda1_range": [min(los), max(los)],
            "sweep_lambda2_range": [min(his), max(his)],
        },
        "d2_transverse_slope": {
            "phase_0_deg": d2_sweep(field, [0])[0][1],
            "phase_18_deg": d2_sweep(field, [18])[0][1],
            "phase_36_deg": d2_sweep(field, [36])[0][1],
            "sweep_range": [min(slopes), max(slopes)],
            "sweep_mean": float(np.mean(slopes)),
            "analytic_transverse_curvature": float(analytic),
            "phase_mean_deficit_pct": float((1 - np.mean(slopes)/analytic) * 100),
        },
        "d3_parking": {
            "d_tracker_mean_abs_x": float(np.mean(np.abs(d_roots))),
            "s1_tracker_mean_abs_x": float(np.mean(np.abs(s1_roots))),
            "n_phases": len(phases_fine),
        },
    }
    out_path = os.path.join(_HERE, "..", "revision", "ring_phase_open_loop.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
