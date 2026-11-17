"""fig_reconstruction_3x2 for the paper, from the 2026 reconstructed fields (vortex_nn_2026, saddle_nn_2026).

Rows: vortex, saddle. Columns:
  analytical       the linear field that best fits the reconstruction over |x|, |y| <= 0.6 m (the printed
                   design in the reconstruction's frame, with its zero at p* = (0, 0))
  reconstructed    the MLP fitted to the robots' readings (as used in simulation)
  error            direction error, reconstructed minus analytical (deg)
Arrows: direction and relative magnitude; background: hue = direction, as on the printed maps.
Writes analysis/paper/fig_reconstruction_3x2.png and .pdf.
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
import paths  # noqa: E402,F401
import vecsim as vs  # noqa: E402

L, N = 0.6, 121


def main():
    x = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9))
    stats = {}
    for row, (name, label) in enumerate((("vortex_nn_2026", "Vortex"), ("saddle_nn_2026", "Saddle"))):
        U, V = vs.make_field(name).fn(X, Y)
        A = np.c_[X.ravel(), Y.ravel()]
        J = np.linalg.lstsq(A, np.c_[U.ravel(), V.ravel()], rcond=None)[0].T        # analytical: v = J p
        Ua, Va = (J @ A.T).reshape(2, N, N)
        err = np.degrees(np.angle(np.exp(1j * (np.arctan2(V, U) - np.arctan2(Va, Ua)))))
        r = np.hypot(X, Y)
        stats[label] = dict(J=J.round(3).tolist(), median_abs_err=float(np.median(np.abs(err[r > 0.05]))),
                            p90=float(np.percentile(np.abs(err[r > 0.05]), 90)))
        for col, (u, v, title) in enumerate(((Ua, Va, "analytical"), (U, V, "reconstructed"))):
            a = ax[row, col]
            hue = (np.arctan2(v, u) % (2 * np.pi)) / (2 * np.pi)
            mag = np.hypot(u, v)
            hsv = np.dstack([hue, np.clip(mag / np.percentile(mag, 99), 0.15, 1), np.ones_like(hue)])
            a.imshow(mcolors.hsv_to_rgb(hsv), extent=[-L, L, -L, L], origin="lower")
            s = 10
            a.quiver(X[::s, ::s], Y[::s, ::s], u[::s, ::s], v[::s, ::s], color="k", scale=8, width=0.004)
            a.set_title(f"{label}, {title}", fontsize=8)
        im = ax[row, 2].imshow(err, extent=[-L, L, -L, L], origin="lower", cmap="RdBu_r", vmin=-30, vmax=30)
        ax[row, 2].set_title(f"{label}, direction error (deg)", fontsize=8)
        cb = fig.colorbar(im, ax=ax[row, 2], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    for a in ax.ravel():
        a.plot(0, 0, "k+", ms=6)
        a.set_xticks([-0.5, 0, 0.5]); a.set_yticks([-0.5, 0, 0.5]); a.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_reconstruction_3x2.{ext}"), dpi=220)
    print(stats)


if __name__ == "__main__":
    main()
