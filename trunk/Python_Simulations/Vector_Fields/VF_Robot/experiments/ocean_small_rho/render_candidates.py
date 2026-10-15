"""
render_candidates.py

Renders Fig. 9 (main_ocean_hfr_2km_progression.py, unchanged) for chosen
operating points, by overriding the constants it imports from
main_ocean_hfr_2km_shared_start.py. Writes to review/<name>/, never into the
paper's figures/. Also writes review/<name>_ridge.png with the extracted main
ridge drawn over the final panel, for checking the score by eye.

Running (from VF_Robot):
    venv/bin/python3 experiments/ocean_small_rho/render_candidates.py \
        name lat lon scale heading_deg gain v_max [name lat lon ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.dirname(EXP))

import main_ocean_hfr_2km_shared_start as S
import main_ocean_hfr_2km_progression as P
import matplotlib.pyplot as plt

_real = P.compute_ftle_field
_memo = {}


def _cached(*a, **k):
    if "v" not in _memo:
        _memo["v"] = _real(*a, **k)
    return _memo["v"]


P.compute_ftle_field = _cached


def render(name, lat, lon, scale, hdg, gain, vmax):
    S.START = (lat, lon)
    S.FORMATION_SCALE, S.HEADING_DEG = scale, hdg
    S.CONTROL_GAIN, S.V_MAX = gain, vmax
    out = os.path.join(HERE, "review", name)
    os.makedirs(out, exist_ok=True)
    S.OUT_DIR = out
    print(f"--- {name}: start ({lat:.5f}, {lon:.5f}) scale {scale:.3f} hdg {hdg:.1f} "
          f"k {gain:.3f} c {vmax:.4f}")
    P.main()
    fig = plt.gcf()
    r = np.load(os.path.join(HERE, "main_ridge.npy"))
    fig.axes[3].plot(r[:, 1], r[:, 0], color="cyan", lw=0.6, ls=":", zorder=20)
    fig.savefig(os.path.join(HERE, "review", f"{name}_ridge.png"), dpi=250)
    plt.close("all")


if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 7):
        render(a[i], *map(float, a[i + 1:i + 7]))
