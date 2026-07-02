"""
fig_estimator_accuracy_vs_noise.py

Figure: estimation error of det(J) and ||grad det(J)|| as a function of
measurement noise std sigma_uv, at the fixed nominal formation.

Runs a Monte Carlo: at each sigma_uv level, perturbs the (u,v) readings at
the six robot positions and computes the estimation error versus the analytic
ground truth.

Canonical output: figures/estimator_accuracy_vs_noise.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    PAPER_DIR, FIGURES_DIR, VFR_ROOT,
    write_sidecar, compile_paper, make_parser,
)

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(VFR_ROOT))
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (
    _fit_vector_quadratic, _det_value, _det_gradient,
    _get_relative_positions,
)

FIGURE_NAME = "estimator_accuracy_vs_noise"

PARAMS = {
    "seed":          42,
    "sigma_uv_vals": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
    "n_trials":      300,
    "centroid":      [-0.3, 0.1],
    "formation_config": "config/formations/pentagon_small.yaml",
    "dpi":           220,
}


def _analytic_det(x, y, A=0.1):
    xf = x + 1.0
    yf = y + 0.5
    dudx = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    dudy =  np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdy =  np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    return dudx * dvdy - dudy * dvdx


def _analytic_grad_det(x, y, A=0.1):
    xf = x + 1.0
    yf = y + 0.5
    s = np.pi
    ux = -s**2 * A * np.cos(s*xf) * np.cos(s*yf)
    uy =  s**2 * A * np.sin(s*xf) * np.sin(s*yf)
    vx = -s**2 * A * np.sin(s*xf) * np.sin(s*yf)
    vy =  s**2 * A * np.cos(s*xf) * np.cos(s*yf)
    uxx =  s**3 * A * np.sin(s*xf) * np.cos(s*yf)
    uxy = -s**3 * A * np.cos(s*xf) * np.sin(s*yf)
    vxx = -s**3 * A * np.cos(s*xf) * np.sin(s*yf)
    vxy = -s**3 * A * np.sin(s*xf) * np.cos(s*yf)
    uyy = -s**3 * A * np.sin(s*xf) * np.cos(s*yf)
    vyy =  s**3 * A * np.sin(s*xf) * np.cos(s*yf)
    Dx = uxx*vy + ux*vxy - uxy*vx - uy*vxx
    Dy = uxy*vy + ux*vyy - uyy*vx - uy*vxy
    return np.array([Dx, Dy])


def main(args):
    p = PARAMS.copy()
    np.random.seed(p["seed"])

    xc, yc = p["centroid"]
    field   = AnalyticalField(double_gyre_static, noise_std=0.0)
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.reset(xc, yc)

    coords   = cluster.get_robot_positions()
    rel_pos  = _get_relative_positions(cluster)
    true_u   = np.array([field.field_function(coords[2*i], coords[2*i+1])[0] for i in range(6)])
    true_v   = np.array([field.field_function(coords[2*i], coords[2*i+1])[1] for i in range(6)])

    D_true   = _analytic_det(xc, yc)
    gD_true  = _analytic_grad_det(xc, yc)

    sigma_vals = p["sigma_uv_vals"]
    n_trials   = p["n_trials"]

    det_errors  = []
    grad_errors = []

    for sigma in sigma_vals:
        d_errs  = []
        gd_errs = []
        for _ in range(n_trials):
            u_noisy = true_u + np.random.randn(6) * sigma
            v_noisy = true_v + np.random.randn(6) * sigma
            theta_u, theta_v = _fit_vector_quadratic(rel_pos, u_noisy, v_noisy)
            d_hat  = _det_value(theta_u, theta_v)
            gd_hat = _det_gradient(theta_u, theta_v)
            d_errs.append(abs(d_hat - D_true))
            gd_errs.append(np.linalg.norm(gd_hat - gD_true))
        det_errors.append(d_errs)
        grad_errors.append(gd_errs)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

    for ax, data, ylabel, title in zip(
        axes,
        [det_errors, grad_errors],
        [r"$|\hat{D} - D_\mathrm{true}|$",
         r"$\|\nabla\hat{D} - \nabla D_\mathrm{true}\|$"],
        [r"$\det(\hat{\mathbf{J}})$ estimation error",
         r"$\nabla\det(\hat{\mathbf{J}})$ estimation error"],
    ):
        bp = ax.boxplot(data, labels=[str(s) for s in sigma_vals],
                        patch_artist=True, medianprops={"color":"black"})
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec6e8")
        ax.set_xlabel(r"$\sigma_{uv}$ (m/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Estimator accuracy vs measurement noise\n"
                 f"centroid = ({xc}, {yc}), N={n_trials} trials per level",
                 fontsize=9)
    fig.tight_layout()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"], bbox_inches="tight")
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    medians_det  = [float(np.median(d)) for d in det_errors]
    medians_grad = [float(np.median(g)) for g in grad_errors]
    write_sidecar(
        out, figure_name=FIGURE_NAME, params=p,
        source_script=f"scripts/{FIGURE_NAME}.py",
        extra={
            "sigma_uv_vals":          sigma_vals,
            "median_det_error":       medians_det,
            "median_grad_det_error":  medians_grad,
            "D_true":                 float(D_true),
            "grad_D_true_norm":       float(np.linalg.norm(gD_true)),
        },
    )

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    if args.n_trials is not None:
        PARAMS["n_trials"] = args.n_trials
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
