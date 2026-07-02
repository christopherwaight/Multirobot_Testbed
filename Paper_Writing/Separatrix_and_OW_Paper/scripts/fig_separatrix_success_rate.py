"""
fig_separatrix_success_rate.py

Figure: success-rate heatmap of the Logic C separatrix tracker over a
(sigma_uv, sigma_p) grid.

Slow: N trials per cell. Results are cached to figures/separatrix_success_rate.cache.npz.
Rerun with --use-cache to skip re-simulation and only redraw.

Canonical output: figures/separatrix_success_rate.png
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
from pathlib import Path

sys.path.insert(0, str(VFR_ROOT))
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import (
    double_gyre_static, SADDLE_BOTTOM, SADDLE_TOP,
)
from src.control.pentagon_primitives import separatrix_logic_c_step

FIGURE_NAME  = "separatrix_success_rate"
CACHE_FILE   = FIGURES_DIR / f"{FIGURE_NAME}.cache.npz"

PARAMS = {
    "seed_base":           0,
    "n_trials_per_cell":   200,
    "sim_steps":           200,
    "sigma_uv_vals":       [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
    "sigma_p_vals":        [0.0, 0.005, 0.01, 0.02],
    "control_gain":        3.0,
    "v_max":               0.04,
    "eps_raw":             1e-3,
    "eps_dim":             0.025,
    "formation_config":    "config/formations/pentagon_small.yaml",
    "success_dist_thresh": 0.08,
    "dpi":                 220,
}

# 5x5 grid + 25 random per cell = 50 ICs per cell, repeated n_trials_per_cell times.
# For the canonical figure the 200 trials are from the grid.
_IC_GRID = [
    (x, y)
    for x in np.linspace(-0.75, 0.75, 5)
    for y in np.linspace(-0.40, 0.40, 5)
]


def _run_trial(p, start_x, start_y, sigma_uv, sigma_p, seed):
    np.random.seed(seed)
    field   = AnalyticalField(double_gyre_static, noise_std=sigma_uv)
    cluster = PentagonCluster(p["formation_config"], field)
    cluster.position_noise_std = sigma_p
    cluster.reset(start_x, start_y)

    gain    = p["control_gain"]
    v_max   = p["v_max"]
    eps_raw = p["eps_raw"]
    eps_dim = p["eps_dim"]

    def primitive(c):
        vx, vy = separatrix_logic_c_step(c, v_max=v_max,
                                          eps_raw=eps_raw, eps_dim=eps_dim)
        return vx * gain, vy * gain

    for _ in range(p["sim_steps"]):
        cluster.move(primitive)

    hist = cluster.get_center_history()
    if len(hist) == 0:
        return False
    final = hist[-1]
    d_bot = np.linalg.norm(final - np.array(SADDLE_BOTTOM))
    d_top = np.linalg.norm(final - np.array(SADDLE_TOP))
    return min(d_bot, d_top) < p["success_dist_thresh"]


def _run_sweep(p):
    sigma_uv_vals = p["sigma_uv_vals"]
    sigma_p_vals  = p["sigma_p_vals"]
    n             = p["n_trials_per_cell"]
    seed_base     = p["seed_base"]

    n_su = len(sigma_uv_vals)
    n_sp = len(sigma_p_vals)
    rates = np.zeros((n_su, n_sp))

    ics = _IC_GRID
    n_ic = len(ics)

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


def _draw_heatmap(ax, rates, sigma_uv_vals, sigma_p_vals, title):
    im = ax.imshow(
        rates,
        vmin=0.0, vmax=1.0,
        cmap="RdYlGn",
        aspect="auto",
        origin="lower",
    )
    ax.set_xticks(range(len(sigma_p_vals)))
    ax.set_xticklabels([str(v) for v in sigma_p_vals], fontsize=8)
    ax.set_yticks(range(len(sigma_uv_vals)))
    ax.set_yticklabels([str(v) for v in sigma_uv_vals], fontsize=8)
    ax.set_xlabel(r"$\sigma_p$ (m)", fontsize=9)
    ax.set_ylabel(r"$\sigma_{uv}$ (m/s)", fontsize=9)
    ax.set_title(title, fontsize=10)
    for ii in range(len(sigma_uv_vals)):
        for jj in range(len(sigma_p_vals)):
            ax.text(jj, ii, f"{rates[ii,jj]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="black" if 0.3 < rates[ii, jj] < 0.8 else "white")
    return im


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
    im = _draw_heatmap(ax, rates, p["sigma_uv_vals"], p["sigma_p_vals"],
                       "Separatrix tracker -- success rate")
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
        primitive_name="separatrix_logic_c_step",
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
    parser.add_argument("--use-cache", action="store_true", default=False,
                        help="Skip re-simulation; redraw from cached .npz.")
    args = parser.parse_args()
    if args.show_params:
        import json; print(json.dumps(PARAMS, indent=2)); sys.exit(0)
    main(args)
