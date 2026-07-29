"""
fig_estimator_accuracy_vs_noise.py

Estimator accuracy on the steady double gyre, for the Results section.

REWRITTEN 2026-07-29. The previous version was unreproducible: its sidecar recorded
300 draws against a caption claiming 2e5, it produced two boxplot panels rather than
the log-log panels the caption described, it no longer ran (AnalyticalField had since
dropped the noise_std kwarg it passed), and its `_analytic_grad_det` had three of six
second derivatives wrong, so the gradient threshold the paper leans on had been
measured against a bad reference.

Ground truth is imported from verify_estimator_bias, where the closed forms are
asserted against finite differences at import time. One source of truth.

Panels:
  (a) median relative error of D, grad D and H_D versus sensor noise at rho = 0.075
  (b) mean angle between the fitted and true principal eigenvector of H_D versus
      sensor noise, against the 45 deg value for a uniformly random axis
  (c) median relative error versus formation radius at sigma_uv = 0.01, with the
      noise-free truncation floors dotted

Canonical output: figures/estimator_accuracy_vs_noise.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_DIR, FIGURES_DIR, write_sidecar, make_parser  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from verify_estimator_bias import (  # noqa: E402
    true_D, true_grad_D, true_hess_D, _angle_deg, _self_check,
)
from src.control.pentagon_primitives import (  # noqa: E402
    _fit_vector_quadratic, _det_value, _det_gradient, _det_hessian,
)
from src.fields.environments.Double_Gyre import double_gyre_static  # noqa: E402

FIGURE_NAME = "estimator_accuracy_vs_noise"

PARAMS = {
    "seed": 20260729,
    "n_trials": 10000,           # matches the Monte Carlo cell size used elsewhere
    "sigma_uv_vals": [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01,
                      0.02, 0.05, 0.1],
    "rho_vals": [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
    "rho_nominal": 0.075,        # config/formations/pentagon_small.yaml, L_2
    "sigma_uv_for_radius_panel": 0.01,
    "centroid": [-0.3, 0.1],
    "ring_phase_deg": -90.0,     # matches the built cluster's ring phase
    # Traverse-success 50% crossing for the D tracker at sigma_p = 0, interpolated
    # log-linearly between 62.4% at 0.005 and 43.7% at 0.010 in Table tab:mc_success(a).
    "closed_loop_50pct_sigma_uv": 0.0079,
    "dpi": 220,
}


def pentagon_rel_positions(rho, phase_deg=-90.0):
    """Pentagon-plus-center relative positions: 5 on a ring of radius rho, 1 centered.

    Built analytically so the radius can be swept. Validated against the geometry
    that PentagonCluster produces from pentagon_small.yaml (see _validate_geometry).
    """
    th = np.radians(phase_deg) + 2.0 * np.pi * np.arange(5) / 5.0
    ring = np.column_stack([rho * np.cos(th), rho * np.sin(th)])
    return np.vstack([np.zeros((1, 2)), ring])


def _validate_geometry():
    """Confirm the analytic formation matches the one the cluster actually builds."""
    from src.robot.pentagon_cluster import PentagonCluster
    from src.fields.field_types import AnalyticalField
    from src.control.pentagon_primitives import _get_relative_positions

    cluster = PentagonCluster("config/formations/pentagon_small.yaml",
                              AnalyticalField(double_gyre_static))
    cluster.reset(*PARAMS["centroid"])
    built = np.asarray(_get_relative_positions(cluster))
    mine = pentagon_rel_positions(PARAMS["rho_nominal"], PARAMS["ring_phase_deg"])
    # compare as point sets, order is not guaranteed
    d = max(min(np.linalg.norm(b - m) for m in mine) for b in built)
    if d > 2e-3:
        raise SystemExit(f"geometry mismatch: max point distance {d:.5f}")
    print(f"  geometry check: analytic pentagon matches the built cluster "
          f"to {d:.2e} m")


def _readings(rel, xc, yc):
    pts = rel + np.array([xc, yc])
    u = np.array([double_gyre_static(px, py)[0] for px, py in pts])
    v = np.array([double_gyre_static(px, py)[1] for px, py in pts])
    return u, v


def sweep(rel, xc, yc, sigma_uv, n_trials, rng):
    """Median relative errors and mean H_D eigendirection angle at one setting."""
    u0, v0 = _readings(rel, xc, yc)
    D_t, gD_t, H_t = true_D(xc, yc), true_grad_D(xc, yc), true_hess_D(xc, yc)
    nD, ngD, nH = abs(D_t), np.linalg.norm(gD_t), np.linalg.norm(H_t, "fro")
    eH_t = np.linalg.eigh(H_t)[1][:, 0]

    eD, eG, eH, ang = [], [], [], []
    for _ in range(n_trials):
        um = u0 + (rng.normal(0.0, sigma_uv, 6) if sigma_uv > 0 else 0.0)
        vm = v0 + (rng.normal(0.0, sigma_uv, 6) if sigma_uv > 0 else 0.0)
        tu, tv = _fit_vector_quadratic(rel, um, vm)
        H = _det_hessian(tu, tv)
        eD.append(abs(_det_value(tu, tv) - D_t) / nD)
        eG.append(np.linalg.norm(_det_gradient(tu, tv) - gD_t) / ngD)
        eH.append(np.linalg.norm(H - H_t, "fro") / nH)
        ang.append(_angle_deg(np.linalg.eigh(H)[1][:, 0], eH_t))
        if sigma_uv == 0.0:
            break  # deterministic
    return {
        "D": float(np.median(eD)), "gD": float(np.median(eG)),
        "H": float(np.median(eH)), "ang": float(np.mean(ang)),
    }


def main(args):
    p = PARAMS.copy()
    if args.seed is not None:
        p["seed"] = args.seed
    rng = np.random.default_rng(p["seed"])
    xc, yc = p["centroid"]
    N = p["n_trials"]

    print(f"\n{FIGURE_NAME}  (N = {N} per point, seed = {p['seed']})")
    _self_check(verbose=True)
    _validate_geometry()

    rel_nom = pentagon_rel_positions(p["rho_nominal"], p["ring_phase_deg"])

    # ---- panels (a) and (b): sweep sigma_uv at nominal radius ----
    noise = {k: [] for k in ("D", "gD", "H", "ang")}
    for s in p["sigma_uv_vals"]:
        r = sweep(rel_nom, xc, yc, s, N, rng)
        for k in noise:
            noise[k].append(r[k])
        print(f"  sigma_uv={s:<7g} relD={r['D']:.4f} relGrad={r['gD']:.4f} "
              f"relH={r['H']:.4f} Hangle={r['ang']:.2f} deg")
    ang0 = sweep(rel_nom, xc, yc, 0.0, 1, rng)["ang"]
    print(f"  truncation-only H_D eigendirection error: {ang0:.2f} deg")

    # ---- panel (c): sweep radius at fixed noise, plus noise-free floors ----
    srad = p["sigma_uv_for_radius_panel"]
    rad = {k: [] for k in ("D", "gD", "H")}
    floor = {k: [] for k in ("D", "gD", "H")}
    for rho in p["rho_vals"]:
        rel = pentagon_rel_positions(rho, p["ring_phase_deg"])
        r = sweep(rel, xc, yc, srad, N, rng)
        f = sweep(rel, xc, yc, 0.0, 1, rng)
        for k in rad:
            rad[k].append(r[k])
            floor[k].append(f[k])
        print(f"  rho={rho:<6g} relD={r['D']:.4f} relGrad={r['gD']:.4f} "
              f"relH={r['H']:.4f}   floors: {f['D']:.4f} {f['gD']:.4f} {f['H']:.4f}")

    # ---------------------------- plot ----------------------------
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "axes.titlesize": 8})
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.25))
    sv = p["sigma_uv_vals"]
    cliff = p["closed_loop_50pct_sigma_uv"]
    style = {"D": ("#1f77b4", "o", r"$\hat{D}$"),
             "gD": ("#d62728", "s", r"$\nabla\hat{D}$"),
             "H": ("#2ca02c", "^", r"$\hat{\mathbf{H}}_D$")}

    for k, (c, m, lab) in style.items():
        ax[0].loglog(sv, noise[k], color=c, marker=m, ms=3, lw=1.2, label=lab)
    ax[0].axhline(1.0, ls="--", c="0.4", lw=0.9)
    ax[0].axvline(cliff, ls="-.", c="0.4", lw=0.9)
    ax[0].text(1.05, 1.15, "error = signal", transform=ax[0].get_yaxis_transform(),
               ha="right", va="bottom", fontsize=6, color="0.35")
    ax[0].set_xlabel(r"$\sigma_{uv}$")
    ax[0].set_ylabel("median relative error")
    ax[0].set_title("(a) accuracy vs sensor noise")
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].grid(alpha=0.25, which="both", lw=0.4)

    ax[1].semilogx(sv, noise["ang"], color="#2ca02c", marker="^", ms=3, lw=1.2)
    ax[1].axhline(45.0, ls="--", c="0.4", lw=0.9)
    ax[1].axvline(cliff, ls="-.", c="0.4", lw=0.9)
    ax[1].plot([sv[0]], [ang0], marker="*", ms=8, color="k", ls="none", zorder=5)
    ax[1].annotate("truncation\nalone", xy=(sv[0], ang0),
                   xytext=(sv[0] * 1.6, ang0 + 11), fontsize=6, color="0.25",
                   arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
    ax[1].text(0.97, 45.0, "random axis", transform=ax[1].get_yaxis_transform(),
               ha="right", va="bottom", fontsize=6, color="0.35")
    ax[1].text(cliff * 1.15, 4, "closed-loop\n50% cliff", fontsize=6, color="0.35")
    ax[1].set_xlabel(r"$\sigma_{uv}$")
    ax[1].set_ylabel(r"$\hat{\mathbf{H}}_D$ eigendirection error (deg)")
    ax[1].set_title("(b) eigendirection vs sensor noise")
    ax[1].set_ylim(0, 50)
    ax[1].grid(alpha=0.25, which="both", lw=0.4)

    for k, (c, m, lab) in style.items():
        ax[2].loglog(p["rho_vals"], rad[k], color=c, marker=m, ms=3, lw=1.2, label=lab)
        ax[2].loglog(p["rho_vals"], floor[k], color=c, ls=":", lw=1.0)
    ax[2].axhline(1.0, ls="--", c="0.4", lw=0.9)
    ax[2].axvline(p["rho_nominal"], ls="-.", c="0.4", lw=0.9)
    ax[2].set_xlabel(r"formation radius $\rho$")
    ax[2].set_ylabel("median relative error")
    ax[2].set_title(rf"(c) accuracy vs radius, $\sigma_{{uv}} = {srad}$")
    ax[2].grid(alpha=0.25, which="both", lw=0.4)

    fig.tight_layout()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=p["dpi"], bbox_inches="tight")
    print(f"  figure -> {out.relative_to(PAPER_DIR)}")
    plt.close(fig)

    write_sidecar(
        out, figure_name=FIGURE_NAME, params=p,
        source_script=f"scripts/{FIGURE_NAME}.py",
        primitive_file="trunk/Python_Simulations/Vector_Fields/VF_Robot/"
                       "src/control/pentagon_primitives.py",
        extra={
            "sigma_uv_vals": p["sigma_uv_vals"],
            "rel_err_D": noise["D"], "rel_err_grad_D": noise["gD"],
            "rel_err_H_D": noise["H"], "H_eigendirection_deg": noise["ang"],
            "H_eigendirection_truncation_only_deg": ang0,
            "rho_vals": p["rho_vals"],
            "rel_err_D_vs_rho": rad["D"], "rel_err_grad_D_vs_rho": rad["gD"],
            "rel_err_H_D_vs_rho": rad["H"],
            "floor_D_vs_rho": floor["D"], "floor_grad_D_vs_rho": floor["gD"],
            "floor_H_D_vs_rho": floor["H"],
            "D_true": float(true_D(xc, yc)),
            "grad_D_true_norm": float(np.linalg.norm(true_grad_D(xc, yc))),
            "H_D_true_fro": float(np.linalg.norm(true_hess_D(xc, yc), "fro")),
        },
    )
    print("  NOTE: no pdflatex recompile; _common.compile_paper points at "
          "Paper_Draft_1A.tex, not Draft_5d.tex.\n")


if __name__ == "__main__":
    parser = make_parser(FIGURE_NAME)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    if args.n_trials is not None:
        PARAMS["n_trials"] = args.n_trials
    if args.show_params:
        import json
        print(json.dumps(PARAMS, indent=2))
        sys.exit(0)
    main(args)
