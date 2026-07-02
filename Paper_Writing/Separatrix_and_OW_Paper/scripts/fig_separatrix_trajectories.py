"""
fig_separatrix_trajectories.py

Figure: Logic C separatrix tracker -- centroid trajectories from a set of
starting positions, overlaid on double-gyre streamlines.

Canonical output: figures/separatrix_trajectories.png
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
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP, SEPARATRIX_X,
)
from src.control.pentagon_primitives import separatrix_logic_c_step
from src.simulation.runner import execute_omni_simulation

FIGURE_NAME = "separatrix_trajectories"

PARAMS = {
    "seed":          0,
    "sim_steps":     200,
    "sigma_uv":      0.0,
    "sigma_p":       0.0,
    "control_gain":  3.0,
    "v_max":         0.04,
    "eps_raw":       1e-3,
    "eps_dim":       0.025,
    "formation_config": "config/formations/pentagon_small.yaml",
    "start_points": [
        [-0.45,  0.30],
        [ 0.05,  0.40],
        [ 0.00,  0.00],
        [ 0.10, -0.20],
        [ 0.25,  0.42],
    ],
    "dpi": 220,
}


def _run_trial(p, start_x, start_y):
    np.random.seed(p["seed"])
    field   = AnalyticalField(double_gyre_static, noise_std=p["sigma_uv"])
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = p["sigma_p"]
    cluster.reset(start_x, start_y)

    gain = p["control_gain"]
    vmax = p["v_max"]
    eps_raw = p["eps_raw"]
    eps_dim = p["eps_dim"]

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=vmax,
                                          eps_raw=eps_raw, eps_dim=eps_dim)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(primitive)

    history = cluster.get_center_history()
    return history


def _background_stream(ax, n=50, A=0.1):
    x0, x1, y0, y1 = -1.0, 1.0, -0.55, 0.55
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(n):
        for j in range(n):
            u, v = double_gyre_static(X[i, j], Y[i, j], A=A)
            U[i, j] = u
            V[i, j] = v
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(X, Y, U, V, color=speed, cmap="Greys",
                  density=0.9, linewidth=0.6, arrowsize=0.7)


def main(args):
    p = PARAMS.copy()
    if args.seed is not None:
        p["seed"] = args.seed

    starts = p["start_points"]
    n = len(starts)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    final_dists = []
    traj_colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    for ax, (sx, sy), col in zip(axes, starts, traj_colors):
        _background_stream(ax)
        hist = _run_trial(p, sx, sy)

        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], "-", color=col,
                    linewidth=1.6, zorder=6)
            ax.plot(hist[0, 0], hist[0, 1], "o", color=col,
                    markersize=6, zorder=7)
            ax.plot(hist[-1, 0], hist[-1, 1], "s", color=col,
                    markersize=6, zorder=7)
            d_bot = np.linalg.norm(hist[-1] - np.array(SADDLE_BOTTOM))
            d_top = np.linalg.norm(hist[-1] - np.array(SADDLE_TOP))
            final_dists.append(min(d_bot, d_top))
        else:
            final_dists.append(float("nan"))

        ax.axvline(x=SEPARATRIX_X, color="crimson", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=5)
        ax.plot(*SADDLE_BOTTOM, "x", color="crimson", markersize=9,
                markeredgewidth=2, zorder=8)
        ax.plot(*SADDLE_TOP, "x", color="crimson", markersize=9,
                markeredgewidth=2, zorder=8)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.55, 0.55)
        ax.set_xlabel(f"start ({sx:.2f}, {sy:.2f})\n"
                      + (f"dist={final_dists[-1]:.3f} m"
                         if not np.isnan(final_dists[-1]) else "no history"),
                      fontsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    axes[0].set_ylabel(r"$y$")
    fig.suptitle(
        "Logic C separatrix tracker -- centroid trajectories\n"
        r"($\sigma_{uv}=" + f"{p['sigma_uv']:.4f}" + r"$, "
        r"$\sigma_p=" + f"{p['sigma_p']:.4f}" + r"$ m)",
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
        primitive_name="separatrix_logic_c_step",
        primitive_file=(
            "trunk/Python_Simulations/Vector_Fields/VF_Robot/"
            "src/control/pentagon_primitives.py"
        ),
        extra={"final_dist_to_saddle": final_dists},
    )

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--sigma-uv", type=float, default=None,
                        help="Override measurement noise std.")
    parser.add_argument("--sigma-p", type=float, default=None,
                        help="Override position noise std.")
    parser.add_argument("--sim-steps", type=int, default=None,
                        help="Override number of simulation steps.")
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
