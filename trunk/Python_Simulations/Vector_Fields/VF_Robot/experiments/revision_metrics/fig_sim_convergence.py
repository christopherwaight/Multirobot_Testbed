"""
Fig. 3 of the vector field revision, redrawn 2026-11-04: centroid distance to the
critical point against time in simulated attraction, mean and +-1 standard deviation
over ten trials per field that start 0.4 m from the critical point, evenly spaced
around the circle (Coulomb model and gains of sim.py, random headings, seed 0). Replaces the untraced six-trajectory fig_sim_convergence.pdf. Review output
only, written to figures/fig_sim_convergence_band.{pdf,png}.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/fig_sim_convergence.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim  # noqa: E402
import vecsim as vs  # noqa: E402

OUT = os.path.join(HERE, "figures")
FIELDS = [("vortex", "Analytical vortex", "#2468c8", "-"),
          ("vortex_nn", "Vortex (recon.)", "#e8692c", (0, (5, 2))),
          ("saddle_nn", "Saddle (recon.)", "#2a9d5c", (0, (1.2, 1.4)))]
R0, N_START = 0.4, 10
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})


def main(n=N_START, r0=R0):
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ph = 2 * np.pi * np.arange(n) / n
    for name, label, col, ls in FIELDS:
        F = vs.make_field(name)
        starts = F.target + r0 * np.stack([np.cos(ph), np.sin(ph)], 1)
        out, _ = vs.simulate(F, "attract", "fit", vs.params(n), 10.0, seed=0, starts=starts,
                             stiction_model=sim.MODEL, record=True)
        d = np.linalg.norm(out["rec"]["c"] - F.target, axis=2)
        t = np.arange(d.shape[1]) * 0.1
        m, s = d.mean(0), d.std(0)
        ax.fill_between(t, np.maximum(m - s, 1e-3), m + s, color=col, alpha=0.18, lw=0)
        ax.plot(t, m, color=col, lw=1.4, ls=ls, label=label)
        print(f"{name:10s} final mean {m[-1]:.4f} m, std {s[-1]:.4f} m")
    ax.axhline(0.025, color="0.3", lw=0.6, ls=(0, (4, 2)))
    ax.text(0.15, 0.027, "$v_c/k$", fontsize=7, va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(5e-3, 1)
    ax.set_xlabel("Time (s)", labelpad=1)
    ax.set_ylabel(r"$\|\mathbf{p}_c - \mathbf{p}^*\|$ (m)", labelpad=1)
    ax.grid(True, which="major", lw=0.3, color="0.85")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_sim_convergence_band.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("wrote", os.path.join(OUT, "fig_sim_convergence_band.pdf"))


if __name__ == "__main__":
    main()
