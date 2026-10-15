"""
plot_estimator_accuracy.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex
  Makes:  Fig. \\ref{fig:est_accuracy}, file figures/estimator_accuracy_vs_noise.png
          (review copy written to experiments/outputs/estimator_accuracy/;
          copied into the paper's figures/ folder only at integration time).
  Reads:  experiments/outputs/estimator_accuracy/noise_sweep.csv
          experiments/outputs/estimator_accuracy/radius_sweep.csv
          (produced by estimator_accuracy_sweep.py; regenerate with that
          script first if the CSVs are missing).

  Panel (a): relative estimation error of D, grad D, H_D versus measurement
  noise sigma_uv at the nominal formation scale (rho = 0.075), strain-region
  location. Median with IQR band, log-log. The horizontal line marks
  error = signal.
  Panel (b): the same relative errors versus formation radius rho at fixed
  sigma_uv = 0.01, with each quantity's noise-free truncation floor dashed.
  Shows the radius trade-off (U-shape) for D and grad D and the structural
  floor of the H_D estimate.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "estimator_accuracy")

# Validated categorical palette (dataviz skill, light surface):
COLORS = {"detJ": "#2a78d6", "gradD": "#1baf7a", "hessD": "#4a3aa7"}
MARKERS = {"detJ": "o", "gradD": "s", "hessD": "^"}
LABELS = {"detJ": r"$\hat{D}$", "gradD": r"$\nabla\hat{D}$",
          "hessD": r"$\hat{\mathbf{H}}_D$"}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def main():
    noise = pd.read_csv(os.path.join(OUT_DIR, "noise_sweep.csv"), comment="#")
    radius = pd.read_csv(os.path.join(OUT_DIR, "radius_sweep.csv"), comment="#")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 5.2))

    # ---------------- panel (a): error vs sigma_uv ----------------------
    df = noise[noise.location == "strain_region"]
    for q in ["detJ", "gradD", "hessD"]:
        d = df[df.quantity == q].sort_values("sigma_uv")
        rel_med = d["median"] / d["truth_magnitude"]
        rel_q25 = d["q25"] / d["truth_magnitude"]
        rel_q75 = d["q75"] / d["truth_magnitude"]
        ax_a.fill_between(d.sigma_uv, rel_q25, rel_q75,
                          color=COLORS[q], alpha=0.18, linewidth=0)
        ax_a.loglog(d.sigma_uv, rel_med, color=COLORS[q],
                    marker=MARKERS[q], ms=4.5, lw=1.6, label=LABELS[q])
        ax_a.annotate(LABELS[q], xy=(d.sigma_uv.iloc[-1], rel_med.iloc[-1]),
                      xytext=(4, 0), textcoords="offset points",
                      color=COLORS[q], fontsize=8, va="center")
    ax_a.axhline(1.0, color="#52514e", lw=0.9, ls="--")
    ax_a.text(0.0011, 0.62, "error = signal", color="#52514e", fontsize=7)

    # V-2 annotation: the noise level where CLOSED-LOOP traverse success
    # crosses 50% (Table II), log-interpolated between the bracketing
    # cells of the archived 10k sweep. Ties the open-loop estimator
    # curves to the closed-loop cliff in one glance.
    arch = os.path.join(HERE, "outputs", "mc_final_1000_ARCHIVE",
                        "separatrix", "summary_fixed_10000.csv")
    s = pd.read_csv(arch, comment="#")
    s0 = s[s.sigma_p == 0].sort_values("sigma_uv")
    above = s0[s0.success_traverse >= 0.5].iloc[-1]
    below = s0[(s0.success_traverse < 0.5)
               & (s0.sigma_uv > above.sigma_uv)].iloc[0]
    frac = ((above.success_traverse - 0.5)
            / (above.success_traverse - below.success_traverse))
    sig50 = np.exp(np.log(above.sigma_uv)
                   + frac * (np.log(below.sigma_uv)
                             - np.log(above.sigma_uv)))
    print(f"closed-loop 50% success at sigma_uv = {sig50:.4f} "
          f"(between {above.sigma_uv:g} and {below.sigma_uv:g})")
    ax_a.axvline(sig50, color="#52514e", lw=1.0, ls="-.")
    ax_a.text(sig50 * 0.90, 1.7e-2, "closed-loop\n50% success",
              color="#52514e", fontsize=7, ha="right")

    ax_a.set_xlabel(r"measurement noise $\sigma_{uv}$ (m/s)")
    ax_a.set_ylabel("relative error (median, IQR)")
    ax_a.set_title(r"(a) vs. noise, $\rho = 0.075$", loc="left")
    ax_a.set_xlim(8e-4, 0.30)
    ax_a.legend(loc="lower right", frameon=False)

    # ---------------- panel (b): error vs rho ---------------------------
    for q in ["detJ", "gradD", "hessD"]:
        d = radius[radius.quantity == q].sort_values("rho")
        rel_med = d["median"] / d["truth_magnitude"]
        rel_flr = d["truncation_only"] / d["truth_magnitude"]
        ax_b.loglog(d.rho, rel_med, color=COLORS[q],
                    marker=MARKERS[q], ms=4.5, lw=1.6, label=LABELS[q])
        ax_b.loglog(d.rho, rel_flr, color=COLORS[q], lw=1.1, ls=":")
    ax_b.axhline(1.0, color="#52514e", lw=0.9, ls="--")
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.set_xlabel(r"formation radius $\rho$ (m)")
    ax_b.set_ylabel("relative error (median)")
    ax_b.set_title(r"(b) vs. radius, $\sigma_{uv} = 0.01$", loc="left")
    ax_b.text(0.020, 0.032, "dotted: noise-free\ntruncation floor",
              fontsize=7, color="#52514e")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "estimator_accuracy_vs_noise.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
