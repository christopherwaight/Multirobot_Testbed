"""
main_ocean_hfr_2km_progression.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_11.tex
  Makes:  fig:ocean_ftle, four snapshots of the shared-start Santa Barbara
          Channel trial, each showing both tracker paths up to that time.
          Output: ocean_data/det_jacobian_plots/ocean_progression_2km.png,
          installed by hand as figures/ocean_progression_2km.png.

The trial is the one in main_ocean_hfr_2km_shared_start.py (same start, gains,
field config, and ISOTROPIC_MAP, imported from it), so the final panel matches
the former single-panel figure. The FTLE background is the same 24-h forward
field anchored at the record start in every panel: a field anchored at a later
snapshot would need data past the 28-h record.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/main_ocean_hfr_2km_progression.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.ticker

import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
from _ftle_common import compute_ftle_field

SNAPSHOT_HOURS = [7, 14, 21, 28]
STEP_MINUTES = 10  # one control step = 0.1 s * TIME_WARP 6000 = 600 s
# Zoomed to the western channel, where both rides happen; keeps the channel
# entrance, the bifurcation, and the middle island named in Section V.
VIEW_LON = (-120.65, -119.95)
VIEW_LAT = (33.92, 34.52)
COLOR_PCT = (34.0, 98.8)          # linear FTLE colour limits, percentiles
LAND_GREEN = (0.0, 0.5, 0.0)      # MATLAB default green, as in [26]


def main():
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    cluster = PentagonCluster(S.FORMATION_CONFIG, field,
                              momentum_alpha=S.MOMENTUM_ALPHA,
                              stiction_threshold=S.STICTION_THRESHOLD)

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW,
                                         eps_dim=S.EPS_DIM)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP,
                                      s_trim=S.S_TRIM, r_band=S.R_BAND,
                                      g_capture=S.G_CAPTURE, s_capture=None)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN

    def robots_latlon():
        """(steps, robots, 2) lat/lon of every robot over the last run."""
        rob = cluster.get_robot_history()
        return np.array([[S._world_to_latlon(x, y, field.config) for x, y in step]
                         for step in rob])

    d_path = S.run_traj(field, cluster, prim_d, *S.START)
    d_rob = robots_latlon()
    s1_path = S.run_traj(field, cluster, prim_s1, *S.START)
    s1_rob = robots_latlon()
    print(f"D end ({d_path[-1, 0]:.4f}N, {d_path[-1, 1]:.4f}), "
          f"s1 end ({s1_path[-1, 0]:.4f}N, {s1_path[-1, 1]:.4f})")

    lat_f, lon_f, f_val, land_f, coast_polys, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP,
        S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR,
        seed_upsample=S.SEED_UPSAMPLE)
    # A run ends at landfall: the first step at which any robot is over land
    # (the same fine-grid land mask the figure draws). Later steps are dropped.
    def landfall_step(rob):
        i = np.clip(np.searchsorted(lat_f, rob[..., 0]), 0, len(lat_f) - 1)
        j = np.clip(np.searchsorted(lon_f, rob[..., 1]), 0, len(lon_f) - 1)
        hit = land_f[i, j].any(axis=1)
        return int(np.argmax(hit)) if hit.any() else None

    for name in ("d", "s1"):
        path, rob = (d_path, d_rob) if name == "d" else (s1_path, s1_rob)
        k_land = landfall_step(rob)
        if k_land is not None:
            print(f"{name} tracker: landfall at step {k_land} ({k_land / 6:.1f} h), run stopped")
            path, rob = path[:k_land + 1], rob[:k_land + 1]
        if name == "d":
            d_path, d_rob = path, rob
        else:
            s1_path, s1_rob = path, rob

    f_plot = np.ma.array(f_val, mask=land_f)
    L, LA = np.meshgrid(lon_f, lat_f)
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_GREEN)
    # Linear scale clipped from the COLOR_PCT percentiles of the in-view water
    # FTLE. Matched by inverting the jet colours of Michini et al. (2014)
    # Fig. 11: with (34, 98.8) about 64% of the water renders navy, against
    # their 68%, so the dominant ridge and its bifurcation carry the figure.
    # The former PowerNorm(0.35) lit about 80% of the water cyan or warmer.
    inview = ((L >= VIEW_LON[0]) & (L <= VIEW_LON[1]) &
              (LA >= VIEW_LAT[0]) & (LA <= VIEW_LAT[1]))
    water = f_val[inview & ~land_f & np.isfinite(f_val)]
    norm = Normalize(vmin=float(np.percentile(water, COLOR_PCT[0])),
                     vmax=float(np.percentile(water, COLOR_PCT[1])))
    outline = [pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()]

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    # Colourbar sits below the panels with the legend, so the four maps take
    # the full text width.
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.08), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.279, top=0.986, wspace=0.03)

    for ax, hours, tag in zip(axes, SNAPSHOT_HOURS, "abcd"):
        n_t = hours * 60 // STEP_MINUTES
        nD, nS = min(n_t, len(d_path)), min(n_t, len(s1_path))
        im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto",
                           rasterized=True)
        # Land from the coastline polygons, so the islands have smooth edges
        # rather than the FTLE grid's cell steps.
        for poly in coast_polys:
            ax.fill(poly[:, 0], poly[:, 1], color=LAND_GREEN, lw=0, zorder=2)
        ax.plot(d_path[:nD, 1], d_path[:nD, 0], color="white", lw=0.8,
                path_effects=outline)
        ax.plot(s1_path[:nS, 1], s1_path[:nS, 0], color="magenta", lw=0.8,
                ls=(0, (3, 1.5)), path_effects=outline)
        ax.plot(S.START[1], S.START[0], marker="*", color="lime", ms=7,
                mec="black", mew=0.6, zorder=11)
        # formation at time t: pentagon through the five ring robots, one dot
        # per robot (the sixth sits at the centroid)
        for rob, col, n in ((d_rob, "white", nD), (s1_rob, "magenta", nS)):
            P = rob[n - 1]
            cen = P.mean(axis=0)
            ring = P[np.argsort(np.linalg.norm(P - cen, axis=1))[1:]]
            ring = ring[np.argsort(np.arctan2(ring[:, 0] - cen[0], ring[:, 1] - cen[1]))]
            ring = np.vstack([ring, ring[:1]])
            ax.plot(ring[:, 1], ring[:, 0], color=col, lw=0.7, zorder=11,
                    path_effects=outline)
            ax.plot(P[:, 1], P[:, 0], "o", color=col, ms=2.0, mec="black",
                    mew=0.4, ls="none", zorder=12)
        ax.text(0.04, 0.96, f"({tag}) $t$ = {hours} h", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", lw=0.4))
        ax.set_xlim(*VIEW_LON)
        ax.set_ylim(*VIEW_LAT)
        ax.set_aspect("equal")
        ax.set_xticks([-120.5, -120.3, -120.1])
        ax.set_yticks([34.0, 34.2, 34.4])
        ax.set_xlabel("Longitude (deg)", labelpad=1)
        ax.tick_params(length=2, pad=1)
    axes[0].set_ylabel("Latitude (deg)", labelpad=1)

    # Horizontal colourbar drops matplotlib's offset text, so the power of
    # ten goes in the label and the ticks are scaled by hand.
    expo = int(np.floor(np.log10(norm.vmax)))
    cax = fig.add_axes([0.745, 0.062, 0.245, 0.036])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="min")
    cb.ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 10**expo:g}"))
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.text(0.735, 0.080, rf"FTLE [$10^{{{expo}}}$ s$^{{-1}}$]", ha="right",
             va="center", fontsize=7.5)

    proxies = [
        Line2D([], [], color="white", lw=0.8, label="$D$ tracker",
               path_effects=outline),
        Line2D([], [], color="magenta", lw=0.8, ls=(0, (3, 1.5)),
               label="$s_1$ tracker", path_effects=outline),
        Line2D([], [], color="lime", marker="*", ls="none", ms=7, mec="black",
               mew=0.6, label="Shared start"),
        Line2D([], [], color="0.55", marker="p", ls="none", ms=5, mfc="none",
               mec="black", mew=0.7, label="Formation at $t$"),
    ]
    fig.legend(handles=proxies, loc="lower left", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.03, -0.03), columnspacing=1.2,
               handletextpad=0.4)

    out_path = os.path.join(S.OUT_DIR, "ocean_progression_2km.png")
    fig.savefig(out_path, dpi=400)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
