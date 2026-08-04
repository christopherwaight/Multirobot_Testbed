"""
ocean_candidate_overlays.py

Renders each shortlisted start from ocean_rank_candidates.py in the style of
the paper's ocean figure: D and s1 tracker paths over the 24-h forward FTLE
field, with the published (pre-fix) paths drawn faintly underneath so the
change from the map fix is visible in one look.

Also renders the published start under the new square map as a "do nothing"
panel, and writes a contact sheet of all panels.

The FTLE background is identical for every panel and is expensive, so it is
computed once and cached to an .npz. Delete the cache to force a recompute.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_candidate_overlays.py
"""
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C
from _ftle_common import compute_ftle_field

FTLE_HOURS, SUBSTEPS_HR, SEED_UPSAMPLE = 24, 6, 4
LAND_RGB = (0.30, 0.55, 0.25)

SHORTLIST_JSON = os.path.join(C.OUT_DIR, "square_map_candidates.json")
FTLE_CACHE = os.path.join(C.OUT_DIR, f"ftle_cache_{FTLE_HOURS}h_u{SEED_UPSAMPLE}.npz")
OUT_DIR = os.path.join(C.PLOT_DIR, "square_map_candidates")


def load_ftle():
    """24-h forward FTLE background, cached. Returns (lat, lon, ftle, land, coasts, t0)."""
    if os.path.exists(FTLE_CACHE):
        z = np.load(FTLE_CACHE, allow_pickle=True)
        print(f"FTLE background from cache: {FTLE_CACHE}")
        return (z["lat"], z["lon"], z["ftle"], z["land"],
                list(z["coasts"]), float(z["t0"]))

    print("Computing 24-h forward FTLE background (2km native), one time...")
    lat_f, lon_f, f_val, land_f, coast_polys, t_list = compute_ftle_field(
        C.DATA_DIR, C.FRAME_GLOB, C.COAST_SHP,
        C.LAT_MIN, C.LAT_MAX, C.LON_MIN, C.LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR, seed_upsample=SEED_UPSAMPLE,
    )
    np.savez_compressed(FTLE_CACHE, lat=lat_f, lon=lon_f, ftle=f_val, land=land_f,
                        coasts=np.array(coast_polys, dtype=object), t0=t_list[0])
    print(f"Cached: {FTLE_CACHE}")
    return lat_f, lon_f, f_val, land_f, coast_polys, float(t_list[0])


def draw_panel(ax, ftle, d_path, s1_path, start, legacy_d, legacy_s1,
               title, show_legend=False):
    lat_f, lon_f, f_val, land_f, coast_polys = ftle
    f_plot = np.ma.array(f_val, mask=land_f)
    L, LA = np.meshgrid(lon_f, lat_f)
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=float(np.nanpercentile(f_val, 99)))
    im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    ax.plot(legacy_d[:, 1], legacy_d[:, 0], color="white", linewidth=3.0,
            alpha=0.45, zorder=5, label="published D (old map)")
    ax.plot(legacy_s1[:, 1], legacy_s1[:, 0], color="white", linewidth=3.0,
            alpha=0.45, linestyle=":", zorder=5, label="published $s_1$ (old map)")

    ax.plot(d_path[:, 1], d_path[:, 0], color="black", linewidth=2.2,
            alpha=0.95, zorder=8, label="D tracker")
    ax.plot(s1_path[:, 1], s1_path[:, 0], color="deepskyblue", linewidth=2.2,
            alpha=0.95, linestyle="--", zorder=8, label="$s_1$ tracker")

    ax.plot(start[1], start[0], marker="*", color="lime", markersize=15,
            markeredgecolor="black", markeredgewidth=1.4, zorder=11, label="shared start")
    ax.plot(C.E0[1], C.E0[0], marker="o", color="magenta", markersize=8,
            markeredgecolor="black", markeredgewidth=1.0, zorder=11,
            label="published landfall $E_0$")
    ax.plot(d_path[-1, 1], d_path[-1, 0], marker="X", color="black", markersize=10,
            markeredgecolor="white", markeredgewidth=1.1, zorder=11)
    ax.plot(s1_path[-1, 1], s1_path[-1, 0], marker="X", color="deepskyblue",
            markersize=10, markeredgecolor="black", markeredgewidth=1.1, zorder=11)

    for poly in coast_polys:
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=0.6, alpha=0.85)

    ax.set_xlim(C.LON_MIN, C.LON_MAX)
    ax.set_ylim(C.LAT_MIN, C.LAT_MAX)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    return im


def panel_title(name, start, stats):
    def g(k, default=float("nan")):
        return stats.get(k, default)
    return (f"{name}\n({start[0]:.4f}N, {-start[1]:.4f}W)  J={stats['J_km']:.1f} km\n"
            f"D {stats['dD_closest_km']:.1f} km@{stats['dD_frac']:.0%}   "
            f"$s_1$ {stats['dS1_closest_km']:.1f} km@{stats['dS1_frac']:.0%}\n"
            f"corridor dev {stats['corrD_mean_km']:.1f}/{stats['corrS1_mean_km']:.1f}  "
            f"cover {g('coverD_km'):.1f}/{g('coverS1_km'):.1f}  "
            f"ridge {g('ridgeD_mean_km'):.1f}/{g('ridgeS1_mean_km'):.1f} km")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SHORTLIST_JSON) as f:
        short = json.load(f)

    _, legacy_d, legacy_s1 = C.load_legacy_reference()
    lat_f, lon_f, f_val, land_f, coasts, t0 = load_ftle()
    ftle = (lat_f, lon_f, f_val, land_f, coasts)

    field, cluster, prim_d, prim_s1 = C.build_trial()

    panels = []
    base = short.get("baseline_published_start_under_square_map")
    if base:
        panels.append(("published start, square map", tuple(C.LEGACY_START), base))
    for c in short["candidates"]:
        panels.append((c["criterion"], (c["start_lat"], c["start_lon"]), c))

    rendered = []
    for name, start, stats in panels:
        print(f"Rendering {name} at ({start[0]:.6f},{start[1]:.6f})...")
        d_path = C.run_traj(field, cluster, prim_d, *start)
        s1_path = C.run_traj(field, cluster, prim_s1, *start)

        fig, ax = plt.subplots(figsize=(7.5, 6.6))
        im = draw_panel(ax, ftle, d_path, s1_path, start, legacy_d, legacy_s1,
                        panel_title(name, start, stats), show_legend=True)
        ax.set_xlabel("Longitude (deg)", fontsize=10)
        ax.set_ylabel("Latitude (deg)", fontsize=10)
        plt.colorbar(im, ax=ax, label=r"FTLE [s$^{-1}$]", shrink=0.85)
        plt.tight_layout()
        slug = name.replace(" ", "_").replace(",", "")
        out = os.path.join(OUT_DIR, f"candidate_{slug}.png")
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rendered.append({"name": name, "start": list(start), "png": out, "stats": stats})
        print(f"  saved {out}")

    n = len(rendered)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, start, stats) in zip(axes, panels):
        d_path = C.run_traj(field, cluster, prim_d, *start)
        s1_path = C.run_traj(field, cluster, prim_s1, *start)
        draw_panel(ax, ftle, d_path, s1_path, start, legacy_d, legacy_s1,
                   panel_title(name, start, stats),
                   show_legend=(ax is axes[0]))
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Shared-start candidates under the square (isotropic) world map\n"
                 "white: published paths under the old anisotropic map",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    sheet = os.path.join(OUT_DIR, "candidate_contact_sheet.png")
    plt.savefig(sheet, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nContact sheet: {sheet}")

    C.atomic_write_json(os.path.join(OUT_DIR, "rendered.json"), {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_candidate_overlays.py",
        "ftle_hours": FTLE_HOURS,
        "contact_sheet": sheet,
        "panels": rendered,
    })


if __name__ == "__main__":
    main()
