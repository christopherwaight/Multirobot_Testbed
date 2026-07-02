"""
fig_ow_trajectories.py

Figure: Newton-step Okubo-Weiss boundary tracker -- centroid trajectories
from a set of starting positions, overlaid on the det(J)=0 boundary.

Canonical output: figures/ow_trajectories.png
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
import matplotlib.ticker as ticker

sys.path.insert(0, str(VFR_ROOT))
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP,
)
from src.control.pentagon_primitives import logic_g_newton_contour_pentagon

FIGURE_NAME = "ow_trajectories"

PARAMS = {
    "seed":          0,
    "sim_steps":     300,
    "sigma_uv":      0.0,
    "sigma_p":       0.0,
    "control_gain":  3.0,
    "v_max":         0.04,
    "eps_grad":      1e-6,
    "formation_config": "config/formations/pentagon_small.yaml",
    "start_points": [
        [-0.50,  0.25],
        [-0.70,  0.00],
        [-0.30,  0.00],
        [ 0.50,  0.25],
        [ 0.30,  0.00],
    ],
    "dpi": 220,
}

GYRE_LEFT  = (-0.5, 0.0)
GYRE_RIGHT = ( 0.5, 0.0)


def _det_j_grid(X, Y, A=0.1):
    xf = X + 1.0
    yf = Y + 0.5
    dudx = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    dudy =  np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdy =  np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    return dudx * dvdy - dudy * dvdx


def _run_trial(p, start_x, start_y):
    np.random.seed(p["seed"])
    field   = AnalyticalField(double_gyre_static, noise_std=p["sigma_uv"])
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = p["sigma_p"]
    cluster.reset(start_x, start_y)

    gain     = p["control_gain"]
    v_max    = p["v_max"]
    eps_grad = p["eps_grad"]

    def primitive(c):
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=v_max,
                                                  eps_grad=eps_grad)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(primitive)

    return cluster.get_center_history()


def main(args):
    p = PARAMS.copy()
    if args.seed is not None:
        p["seed"] = args.seed

    # Pre-compute det(J)=0 boundary
    n = 150
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-0.55, 0.55, n)
    X, Y = np.meshgrid(xs, ys)
    D = _det_j_grid(X, Y)

    starts = p["start_points"]
    n_starts = len(starts)
    fig, axes = plt.subplots(1, n_starts, figsize=(3.8 * n_starts, 4.2), sharey=True)
    if n_starts == 1:
        axes = [axes]

    traj_colors = plt.cm.tab10(np.linspace(0, 0.9, n_starts))
    final_det_vals = []

    for ax, (sx, sy), col in zip(axes, starts, traj_colors):
        # Background: det(J)=0 boundary
        ax.contour(X, Y, D, levels=[0.0], colors=["#e07b00"],
                   linewidths=2.0, zorder=2)

        hist = _run_trial(p, sx, sy)

        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], "-", color=col,
                    linewidth=1.6, zorder=6)
            ax.plot(hist[0, 0], hist[0, 1], "o", color=col,
                    markersize=6, zorder=7)
            ax.plot(hist[-1, 0], hist[-1, 1], "s", color=col,
                    markersize=6, zorder=7)
            xf, yf = hist[-1]
            xf_field = xf + 1.0
            yf_field = yf + 0.5
            import numpy as _np
            A = 0.1
            dudx = -_np.pi**2 * A * _np.cos(_np.pi*xf_field) * _np.cos(_np.pi*yf_field)
            dudy =  _np.pi**2 * A * _np.sin(_np.pi*xf_field) * _np.sin(_np.pi*yf_field)
            dvdx = -_np.pi**2 * A * _np.sin(_np.pi*xf_field) * _np.sin(_np.pi*yf_field)
            dvdy =  _np.pi**2 * A * _np.cos(_np.pi*xf_field) * _np.cos(_np.pi*yf_field)
            det_final = dudx*dvdy - dudy*dvdx
            final_det_vals.append(float(det_final))
        else:
            final_det_vals.append(float("nan"))

        ax.plot(*GYRE_LEFT, "+", color="black", markersize=10,
                markeredgewidth=2, zorder=8)
        ax.plot(*GYRE_RIGHT, "+", color="black", markersize=10,
                markeredgewidth=2, zorder=8)
        ax.plot(*SADDLE_BOTTOM, "x", color="grey", markersize=8,
                markeredgewidth=1.5, zorder=8)
        ax.plot(*SADDLE_TOP, "x", color="grey", markersize=8,
                markeredgewidth=1.5, zorder=8)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.55, 0.55)
        det_str = f"{final_det_vals[-1]:.4f}" if not np.isnan(final_det_vals[-1]) else "n/a"
        ax.set_xlabel(f"start ({sx:.2f},{sy:.2f})\n$\\det(J)_f={det_str}$", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    axes[0].set_ylabel(r"$y$")
    fig.suptitle(
        "Newton-step OW boundary tracker -- centroid trajectories\n"
        r"($\sigma_{uv}=" + f"{p['sigma_uv']:.4f}" + r"$, "
        r"$\sigma_p=" + f"{p['sigma_p']:.4f}" + r"$ m, "
        r"$\det(J)=0$ boundary in orange)",
        fontsize=10,
    )
    fig.tight_layout()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"], bbox_inches="tight")
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(
        out, figure_name=FIGURE_NAME, params=p,
        source_script=f"scripts/{FIGURE_NAME}.py",
        primitive_name="logic_g_newton_contour_pentagon",
        primitive_file=(
            "trunk/Python_Simulations/Vector_Fields/VF_Robot/"
            "src/control/pentagon_primitives.py"
        ),
        extra={"final_det_val": final_det_vals},
    )

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--sigma-uv", type=float, default=None)
    parser.add_argument("--sigma-p", type=float, default=None)
    parser.add_argument("--sim-steps", type=int, default=None)
    args = parser.parse_args()
    if args.sigma_uv is not None:
        PARAMS["sigma_uv"] = args.sigma_uv
    if args.sigma_p is not None:
        PARAMS["sigma_p"] = args.sigma_p
    if args.sim_steps is not None:
        PARAMS["sim_steps"] = args.sim_steps
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
