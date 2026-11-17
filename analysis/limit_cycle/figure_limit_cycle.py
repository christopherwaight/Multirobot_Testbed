"""Limit-cycle figure: vortex, 3 robots, r_d = 0.10 m (chosen in limit_cycle.py).

(a) Simulation on the 2026 reconstructed vortex: paths from 8 starts inside (0.04 m) and 8 outside
    (0.30 m) the cycle, and the cycle r(theta) they converge to.
(b) Hardware: the 10 runs at r_d = 0.10 m, mapped into the same field frame and overlaid
    (concatenated), with their mean cycle and the simulated cycle for comparison.
(c) Distance to p* against time: simulation (inside and outside starts) and hardware runs.
Run from the repo root after limit_cycle.py.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD, NR, RD = "vortex", 3, 0.10
BINS = np.radians(np.arange(-180, 181, 10))
MID = 0.5 * (BINS[:-1] + BINS[1:])


def closed(curve):
    th = np.r_[MID, MID[0] + 2 * np.pi]
    r = np.r_[curve, curve[0]]
    return r * np.cos(th), r * np.sin(th)


def main():
    D = pickle.load(open(os.path.join(HERE, "outputs", f"paths_{FIELD}.pkl"), "rb"))
    sim, hw = D["sim"][RD], D["hw"][(NR, RD)]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for C, tag in zip(sim["C"], sim["tags"]):
        ax[0].plot(C[:, 0], C[:, 1], lw=0.6, alpha=0.6, color="#1f5fa8" if tag == "in" else "#d1495b")
    ax[0].plot(*closed(sim["curve"]), "k", lw=1.6)
    for C in hw["C"]:
        ax[1].plot(C[:, 0], C[:, 1], lw=0.6, alpha=0.7, color="#555555")
    ax[1].plot(*closed(hw["curve"]), "k", lw=1.6, label="hardware cycle")
    ax[1].plot(*closed(sim["curve"]), "--", color="#2a9d5c", lw=1.4, label="simulated cycle")
    for a, title in zip(ax[:2], ("(a) simulation, reconstructed vortex", "(b) hardware, 10 runs overlaid")):
        circ = np.linspace(0, 2 * np.pi, 200)
        a.plot(RD * np.cos(circ), RD * np.sin(circ), ":", color="#e8692c", lw=1.2)
        a.plot(0, 0, "k+", ms=8)
        a.set_aspect("equal"); a.set_xlim(-0.22, 0.22); a.set_ylim(-0.22, 0.22)
        a.set_title(title, fontsize=9); a.grid(alpha=0.3)
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")
    t = np.arange(sim["C"].shape[1]) * 0.2
    for C, tag in zip(sim["C"], sim["tags"]):
        ax[2].plot(t, np.hypot(*C.T), lw=0.6, alpha=0.6, color="#1f5fa8" if tag == "in" else "#d1495b")
    for C in hw["C"]:
        ax[2].plot(np.arange(len(C)) * 0.1, np.hypot(*C.T), lw=0.8, color="#555555", alpha=0.8)
    ax[2].axhline(RD, ls=":", color="#e8692c", lw=1.2)
    ax[2].set_xlim(0, 40); ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("distance to p* (m)")
    ax[2].set_title("(c) sim inside (blue), outside (red); hardware (grey)", fontsize=9); ax[2].grid(alpha=0.3)
    for a in ax[:2]:
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_limit_cycle.{ext}"), dpi=200)
    ok = np.isfinite(sim["curve"]) & np.isfinite(hw["curve"])
    rho = np.corrcoef(sim["curve"][ok], hw["curve"][ok])[0, 1]
    gap = np.median(np.abs(sim["curve"][ok] - hw["curve"][ok]))
    print(f"r(theta) of the simulated vs hardware cycle: correlation {rho:.2f}, median gap {1000 * gap:.1f} mm; "
          f"mean radius sim {np.nanmean(sim['curve']):.3f} m, hw {np.nanmean(hw['curve']):.3f} m")


if __name__ == "__main__":
    main()
