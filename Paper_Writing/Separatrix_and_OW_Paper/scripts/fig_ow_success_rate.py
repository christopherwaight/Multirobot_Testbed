"""
fig_ow_success_rate.py

Figure: success-rate heatmap of the Newton-step OW boundary tracker over a
(sigma_uv, sigma_p) grid.

Success criterion: |det(J(p_c))| at the end of the trial is below the threshold.

Slow. Results cached to figures/ow_success_rate.cache.npz.

Canonical output: figures/ow_success_rate.png
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
from pathlib import Path

sys.path.insert(0, str(VFR_ROOT))
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import logic_g_newton_contour_pentagon

FIGURE_NAME = "ow_success_rate"
CACHE_FILE  = FIGURES_DIR / f"{FIGURE_NAME}.cache.npz"

PARAMS = {
    "seed_base":          0,
    "n_trials_per_cell":  200,
    "sim_steps":          300,
    "sigma_uv_vals":      [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
    "sigma_p_vals":       [0.0, 0.005, 0.01, 0.02],
    "control_gain":       3.0,
    "v_max":              0.04,
    "eps_grad":           1e-6,
    "formation_config":   "config/formations/pentagon_small.yaml",
    "success_det_thresh": 2e-3,
    "dpi":                220,
}

_IC_GRID = [
    (x, y)
    for x in np.linspace(-0.80, 0.80, 5)
    for y in np.linspace(-0.40, 0.40, 5)
]


def _analytic_det(x, y, A=0.1):
    xf = x + 1.0
    yf = y + 0.5
    dudx = -np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    dudy =  np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdx = -np.pi**2 * A * np.sin(np.pi * xf) * np.sin(np.pi * yf)
    dvdy =  np.pi**2 * A * np.cos(np.pi * xf) * np.cos(np.pi * yf)
    return dudx * dvdy - dudy * dvdx


def _run_trial(p, start_x, start_y, sigma_uv, sigma_p, seed):
    np.random.seed(seed)
    field   = AnalyticalField(double_gyre_static, noise_std=sigma_uv)
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = sigma_p
    cluster.reset(start_x, start_y)

    gain     = p["control_gain"]
    v_max    = p["v_max"]
    eps_grad = p["eps_grad"]

    def primitive(c):
        vx, vy = logic_g_newton_contour_pentagon(c, v_max=v_max, eps_grad=eps_grad)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(primitive)

    hist = cluster.get_center_history()
    if len(hist) == 0:
        return False
    xf, yf = hist[-1]
    return abs(_analytic_det(xf, yf)) < p["success_det_thresh"]


def _run_sweep(p):
    sigma_uv_vals = p["sigma_uv_vals"]
    sigma_p_vals  = p["sigma_p_vals"]
    n             = p["n_trials_per_cell"]
    seed_base     = p["seed_base"]
    ics           = _IC_GRID
    n_ic          = len(ics)

    rates = np.zeros((len(sigma_uv_vals), len(sigma_p_vals)))
    for i, su in enumerate(sigma_uv_vals):
        for j, sp in enumerate(sigma_p_vals):
            successes = 0
            for k in range(n):
                sx, sy = ics[k % n_ic]
                seed = seed_base + i * 10000 + j * 1000 + k
                if _run_trial(p, sx, sy, su, sp, seed):
                    successes += 1
            rates[i, j] = successes / n
            print(f"  sigma_uv={su:.4f} sigma_p={sp:.4f}  rate={rates[i,j]:.2f}")
    return rates


def main(args):
    p = PARAMS.copy()
    if args.seed is not None:
        p["seed_base"] = args.seed
    if args.n_trials_per_cell is not None:
        p["n_trials_per_cell"] = args.n_trials_per_cell

    if args.use_cache and CACHE_FILE.exists():
        print(f"  loading cache from {CACHE_FILE.name}")
        cache = np.load(CACHE_FILE)
        rates = cache["rates"]
    else:
        print(f"  running sweep ({p['n_trials_per_cell']} trials/cell) ...")
        rates = _run_sweep(p)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_FILE, rates=rates,
                 sigma_uv_vals=p["sigma_uv_vals"],
                 sigma_p_vals=p["sigma_p_vals"])
        print(f"  cache -> {CACHE_FILE.name}")

    fig, ax = plt.subplots(figsize=(5, 3.8))
    im = ax.imshow(rates, vmin=0.0, vmax=1.0, cmap="RdYlGn",
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(p["sigma_p_vals"])))
    ax.set_xticklabels([str(v) for v in p["sigma_p_vals"]], fontsize=8)
    ax.set_yticks(range(len(p["sigma_uv_vals"])))
    ax.set_yticklabels([str(v) for v in p["sigma_uv_vals"]], fontsize=8)
    ax.set_xlabel(r"$\sigma_p$ (m)", fontsize=9)
    ax.set_ylabel(r"$\sigma_{uv}$ (m/s)", fontsize=9)
    ax.set_title("OW boundary tracker -- success rate", fontsize=10)
    for ii in range(len(p["sigma_uv_vals"])):
        for jj in range(len(p["sigma_p_vals"])):
            ax.text(jj, ii, f"{rates[ii,jj]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="black" if 0.3 < rates[ii, jj] < 0.8 else "white")
    fig.colorbar(im, ax=ax, label="Success rate")
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
        extra={"success_rate_grid": rates.tolist()},
    )

    if not args.no_compile:
        compile_paper()


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--n-trials-per-cell", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true", default=False)
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
