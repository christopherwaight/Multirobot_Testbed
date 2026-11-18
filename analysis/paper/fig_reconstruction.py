"""fig_reconstruction_3x2 for the paper, in the style of the submitted figure: arrows colored by direction.

Rows: vortex, saddle. Columns:
  analytical      the linear field v = J p that best fits the readings (the printed design in the frame of the
                  2026 nets, with its critical point at the origin, black dot)
  reconstructed   the readings the 2026 nets were trained on (1 cm cell means of hue and saturation, decoded)
  error           reconstructed minus analytical, at fewer points, drawn ERR_GAIN times longer (in the title)
Each panel draws arrows at real cell positions, thinned to one per block (block size sets the density).
The field panels share one length scale. Color is the arrow's direction (same wheel as the submitted figure).
Writes analysis/paper/fig_reconstruction_3x2.png and .pdf; caches the cells in fig_reconstruction_cells.npz
(delete it to reload the readings).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
CACHE = os.path.join(HERE, "fig_reconstruction_cells.npz")
L = 0.5                                                    # half-width of each panel (m)
BLOCK = {"analytical": 0.075, "reconstructed": 0.05, "error": 0.10}
SCALE = 6.0                                                # field units per metre of arrow
ERR_GAIN = 5                                               # error arrows drawn this many times longer


def load_cells():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {f: (z[f"{f}_P"], z[f"{f}_F"]) for f in ("vortex", "saddle")}
    import map_readings as mr
    out, save = {}, {}
    for f in ("vortex", "saddle"):
        d = mr.prepare(f)
        X, Yh, Ys = mr.cells(d, d["trn26"])
        ang = np.arctan2(Yh[:, 0], Yh[:, 1]) + np.pi / 2
        mag = ((Ys[:, 0] + 1) / 2 - 0.3) / 0.7
        F = np.stack([mag * np.cos(ang), mag * np.sin(ang)], 1)
        out[f] = (X, F)
        save[f"{f}_P"], save[f"{f}_F"] = X, F
    np.savez(CACHE, **save)
    return out


def thin(P, block, seed=0):
    """One cell per block x block square, chosen at random, inside the panel."""
    inside = np.where((np.abs(P) < L - 0.02).all(1))[0]
    i = np.random.default_rng(seed).permutation(inside)
    _, first = np.unique(np.floor(P[i] / block).astype(int), axis=0, return_index=True)
    return np.sort(i[first])


def colors(F):
    hue = (np.arctan2(F[:, 1], F[:, 0]) / (2 * np.pi) + 0.5) % 1
    return mcolors.hsv_to_rgb(np.stack([hue, np.ones_like(hue), 0.95 * np.ones_like(hue)], 1))


def arrows(ax, P, F):
    ax.quiver(P[:, 0], P[:, 1], F[:, 0], F[:, 1], color=colors(F), angles="xy", scale_units="xy",
              scale=SCALE, width=0.009, headwidth=3.5, headlength=4, headaxislength=3.5)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 8})
    data = load_cells()
    fig, ax = plt.subplots(2, 3, figsize=(3.5, 2.45), sharex=True, sharey=True)
    for row, (field, label) in enumerate((("vortex", "Vortex"), ("saddle", "Saddle"))):
        P, F = data[field]
        r = np.hypot(*P.T)
        fit = (np.abs(P) <= L).all(1) & (r > 0.05)
        J = np.linalg.lstsq(P[fit], F[fit], rcond=None)[0].T          # analytical: v = J p
        Fa = P @ J.T
        err = np.degrees(np.angle(np.exp(1j * (np.arctan2(F[:, 1], F[:, 0]) - np.arctan2(Fa[:, 1], Fa[:, 0])))))
        print(f"{label}: J = {J.round(3).tolist()}, median direction error {np.median(np.abs(err[fit])):.1f} deg, "
              f"median |error vector| {np.median(np.hypot(*(F - Fa)[fit].T)):.3f}, "
              f"median |field| {np.median(np.hypot(*Fa[fit].T)):.3f}")
        k = thin(P, BLOCK["analytical"], seed=1)
        arrows(ax[row, 0], P[k], Fa[k])
        ax[row, 0].plot(0, 0, "o", color="k", ms=4.5, zorder=5)
        k = thin(P, BLOCK["reconstructed"], seed=2)
        arrows(ax[row, 1], P[k], F[k])
        k = thin(P, BLOCK["error"], seed=3)
        arrows(ax[row, 2], P[k], ERR_GAIN * (F[k] - Fa[k]))
        ax[row, 0].set_ylabel(f"{label}, $y$ (m)")
    for c, t in enumerate(("Analytical", "Reconstructed", f"Error ($\\times${ERR_GAIN})")):
        ax[0, c].set_title(t, fontsize=8, pad=3)
        ax[1, c].set_xlabel("$x$ (m)")
    for a in ax.ravel():
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal")
        a.set_xticks([-0.4, 0, 0.4]); a.set_yticks([-0.4, 0, 0.4])
        a.set_xticklabels(["-0.4", "0", "0.4"]); a.set_yticklabels(["-0.4", "0", "0.4"])
        a.tick_params(direction="out", length=2.5, width=0.8, labelsize=7, colors="0.25")
        for s in a.spines.values():
            s.set_color("0.3"); s.set_linewidth(0.8)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.93, wspace=0.06, hspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_reconstruction_3x2.{ext}"), dpi=600)


if __name__ == "__main__":
    main()
