"""
fig_controller_comparison.py

Figure: side-by-side comparison of the separatrix tracker (Logic C) and
the OW boundary tracker, run from the same initial conditions and noise level.

Shows basin-of-attraction: each IC coloured by whether the trial succeeded.

Canonical output: figures/controller_comparison.png
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
from src.control.pentagon_primitives import (
    separatrix_logic_c_step,
    logic_g_newton_contour_pentagon,
)

FIGURE_NAME = "controller_comparison"

PARAMS = {
    "seed":          0,
    "sim_steps":     200,
    "sigma_uv":      0.005,
    "sigma_p":       0.0,
    "control_gain":  3.0,
    "v_max":         0.04,
    "eps_raw":       1e-3,
    "eps_dim":       0.025,
    "eps_grad":      1e-6,
    "formation_config": "config/formations/pentagon_small.yaml",
    "success_dist_sep":  0.08,
    "success_det_ow":    2e-3,
    "ic_grid_n":     7,
    "ic_range":      [-0.75, 0.75, -0.40, 0.40],
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


def _run_sep(p, sx, sy):
    np.random.seed(p["seed"])
    field   = AnalyticalField(double_gyre_static, noise_std=p["sigma_uv"])
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = p["sigma_p"]
    cluster.reset(sx, sy)
    gain = p["control_gain"]; vmax = p["v_max"]
    eps_raw = p["eps_raw"]; eps_dim = p["eps_dim"]

    def prim(c):
        vx, vy = separatrix_logic_c_step(c, v_max=vmax, eps_raw=eps_raw, eps_dim=eps_dim)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(prim)

    hist = cluster.get_center_history()
    if len(hist) == 0:
        return False
    final = hist[-1]
    db = np.linalg.norm(final - np.array(SADDLE_BOTTOM))
    dt = np.linalg.norm(final - np.array(SADDLE_TOP))
    return min(db, dt) < p["success_dist_sep"]


def _run_ow(p, sx, sy):
    np.random.seed(p["seed"])
    field   = AnalyticalField(double_gyre_static, noise_std=p["sigma_uv"])
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = p["sigma_p"]
    cluster.reset(sx, sy)
    gain = p["control_gain"]; vmax = p["v_max"]; eps_grad = p["eps_grad"]

    def prim(c):
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=vmax, eps_grad=eps_grad)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(prim)

    hist = cluster.get_center_history()
    if len(hist) == 0:
        return False
    xf, yf = hist[-1]
    return abs(_analytic_det(xf, yf)) < p["success_det_ow"]


def main(args):
    p = PARAMS.copy()
    if args.seed is not None:
        p["seed"] = args.seed
    if args.sigma_uv is not None:
        p["sigma_uv"] = args.sigma_uv

    n = p["ic_grid_n"]
    x0, x1, y0, y1 = p["ic_range"]
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    ICs = [(x, y) for x in xs for y in ys]

    print("  running separatrix controller ...")
    sep_results = [_run_sep(p, sx, sy) for sx, sy in ICs]
    print("  running OW controller ...")
    ow_results  = [_run_ow(p, sx, sy) for sx, sy in ICs]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)

    for ax, results, title in zip(
        axes,
        [sep_results, ow_results],
        ["Logic C separatrix tracker", "Newton-step OW tracker"],
    ):
        for (sx, sy), ok in zip(ICs, results):
            ax.scatter(sx, sy,
                       color="#2ca02c" if ok else "#d62728",
                       s=38, zorder=4, edgecolors="none")

        ax.axvline(x=0.0, color="crimson", linewidth=1.2, linestyle="--",
                   alpha=0.5, zorder=2)
        ax.plot(*SADDLE_BOTTOM, "x", color="black", markersize=9,
                markeredgewidth=2, zorder=5)
        ax.plot(*SADDLE_TOP, "x", color="black", markersize=9,
                markeredgewidth=2, zorder=5)

        n_ok = sum(results)
        ax.set_title(f"{title}\n{n_ok}/{len(results)} succeeded", fontsize=10)
        ax.set_xlim(x0 - 0.05, x1 + 0.05)
        ax.set_ylim(y0 - 0.05, y1 + 0.05)
        ax.set_xlabel(r"$x$")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_aspect("equal")

    axes[0].set_ylabel(r"$y$")
    fig.suptitle(
        r"Basin-of-attraction comparison  ($\sigma_{uv}=" +
        f"{p['sigma_uv']:.4f}" + r"$, " +
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
        extra={
            "sep_success_rate": sum(sep_results) / len(sep_results),
            "ow_success_rate":  sum(ow_results)  / len(ow_results),
            "n_ics":            len(ICs),
        },
    )

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--sigma-uv", type=float, default=None)
    parser.add_argument("--sigma-p", type=float, default=None)
    args = parser.parse_args()
    if args.sigma_p is not None:
        PARAMS["sigma_p"] = args.sigma_p
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
