"""
ocean_candidate_time_panels.py

Four-panel time progression for a shared-start candidate: the same run shown
against the FTLE field at four anchor times across the record, so the field's
evolution and the tracker's progress can be read together instead of being
collapsed onto one static background.

Anchors are 0, 5, 10 and 16 h with a 12-h forward horizon, the same four the
paper's persistence check uses (ocean_hfr_2km_ftle_snapshots.py), which is the
longest horizon the 29-frame record supports at four anchors.

In each panel the paths are drawn in three weights:

  faint      the whole 28-h path, for context
  bold       the segment the displayed FTLE actually governs, from the anchor
             to 12 h later. This is the part that should be read against that
             panel's ridges.
  markers    open circle at the anchor time, cross 12 h later

Drawing the bold segment over the FTLE's own window rather than "everything so
far" is deliberate: an FTLE field anchored at hour h describes transport over
[h, h+12], so that is the only stretch of trajectory it can explain.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_candidate_time_panels.py
    venv/bin/python3 experiments/ocean_candidate_time_panels.py --which balanced
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C
from _ftle_common import compute_ftle_field

ANCHOR_HOURS = [0, 5, 10, 16]
FTLE_HOURS, SUBSTEPS_HR, SEED_UPSAMPLE = 12, 6, 4

# Styling follows the FTLE panels of Michini et al., IEEE T-RO 30(3), 2014,
# Figs. 10(b) and 11, the closest published comparator on this exact record.
# Their look: jet on a deep blue ground so only genuine ridges light up, land
# in flat green, vehicle tracks in plain white, and panels packed tight with no
# axis furniture between them.
#
# The gamma matters more than anything else here. An earlier version used
# PowerNorm(gamma=0.35), which lifts low FTLE into green and cyan and turns the
# background into visual noise. gamma > 1 pushes the low end down instead, so
# the field reads as dark blue with sharp red ridges, as theirs does.
LAND_RGB = (0.30, 0.55, 0.25)
FTLE_GAMMA = 1.35
FTLE_VMAX_PCT = 98.5
# Michini draws a single track in white. With two trackers one of them needs a
# second colour, and it has to sit off the jet ramp or it vanishes into the
# field. White and magenta are the only two that qualify.
TRACK_D = "white"
TRACK_S1 = (1.0, 0.30, 0.90)

HOURS_TOTAL = 28.0
STEPS_PER_HOUR = C.SIM_STEPS / HOURS_TOTAL      # 168 steps / 28 h = 6

SHORTLIST_JSON = os.path.join(C.OUT_DIR, "square_map_candidates.json")
OUT_DIR = os.path.join(C.PLOT_DIR, "square_map_candidates")


def ftle_at(anchor_h):
    """12-h forward FTLE anchored at `anchor_h` hours into the record, cached."""
    cache = os.path.join(C.OUT_DIR,
                         f"ftle_cache_{FTLE_HOURS}h_a{anchor_h}_u{SEED_UPSAMPLE}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["lat"], z["lon"], z["ftle"], z["land"], list(z["coasts"])

    print(f"  computing {FTLE_HOURS}-h FTLE anchored at {anchor_h} h...")
    lat_f, lon_f, f_val, land_f, coasts, _ = compute_ftle_field(
        C.DATA_DIR, C.FRAME_GLOB, C.COAST_SHP,
        C.LAT_MIN, C.LAT_MAX, C.LON_MIN, C.LON_MAX,
        ftle_hours=FTLE_HOURS, substeps_hr=SUBSTEPS_HR,
        seed_upsample=SEED_UPSAMPLE, file_offset=anchor_h, verbose=False,
    )
    np.savez_compressed(cache, lat=lat_f, lon=lon_f, ftle=f_val, land=land_f,
                        coasts=np.array(coasts, dtype=object))
    return lat_f, lon_f, f_val, land_f, coasts


def idx(hour, n):
    """Path index for a time in hours, clamped to the recorded history length."""
    return int(np.clip(round(hour * STEPS_PER_HOUR), 0, n - 1))


def draw(ax, ftle, d_path, s1_path, start, anchor_h, label_panel=""):
    lat_f, lon_f, f_val, land_f, coasts = ftle
    f_plot = np.ma.array(f_val, mask=land_f)
    L, LA = np.meshgrid(lon_f, lat_f)
    jet = plt.get_cmap("jet").copy()
    jet.set_bad(color=LAND_RGB)
    norm = PowerNorm(gamma=FTLE_GAMMA, vmin=0.0,
                     vmax=float(np.nanpercentile(f_val, FTLE_VMAX_PCT)))
    im = ax.pcolormesh(L, LA, f_plot, cmap=jet, norm=norm, shading="auto")

    # A thin dark halo keeps both tracks readable where they cross a red ridge.
    halo = [pe.withStroke(linewidth=3.2, foreground=(0, 0, 0, 0.65))]

    for path, colour, style in ((d_path, TRACK_D, "-"),
                                (s1_path, TRACK_S1, "--")):
        i0, i1 = idx(anchor_h, len(path)), idx(anchor_h + FTLE_HOURS, len(path))
        ax.plot(path[:, 1], path[:, 0], color=colour, linewidth=0.9,
                alpha=0.40, linestyle=style, zorder=6)
        seg = path[i0:i1 + 1]
        ax.plot(seg[:, 1], seg[:, 0], color=colour, linewidth=2.3,
                linestyle=style, zorder=8, path_effects=halo, solid_capstyle="round")
        ax.plot(path[i0, 1], path[i0, 0], marker="o", color=colour, markersize=7,
                markerfacecolor="none", markeredgewidth=1.8, zorder=10,
                path_effects=halo)
        ax.plot(path[i1, 1], path[i1, 0], marker="o", color=colour, markersize=6,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    ax.plot(start[1], start[0], marker="*", color="white", markersize=13,
            markeredgecolor="black", markeredgewidth=1.0, zorder=11)

    for poly in coasts:
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=0.5, alpha=0.9)

    ax.set_xlim(C.LON_MIN, C.LON_MAX)
    ax.set_ylim(C.LAT_MIN, C.LAT_MAX)
    ax.set_aspect("equal")
    # Panel identity goes inside the frame, as in their figures, so the panels
    # can be packed with no vertical gap for titles.
    ax.text(0.028, 0.955, label_panel, transform=ax.transAxes,
            fontsize=9.5, color="white", va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    return im


def render(name, start, stats, out_dir, scale=1.0, heading_deg=0.0,
           gain=C.CONTROL_GAIN, v_max=C.V_MAX):
    """
    Render one candidate. scale/heading/gain/v_max default to the paper's Ocean
    operating point, so a start-only candidate renders unchanged; the
    operating-point search passes its own values.
    """
    field, cluster, prim_d, prim_s1 = C.build_trial(v_max=v_max, gain=gain)
    C.set_formation_scale(cluster, scale)
    hdg = np.radians(heading_deg)
    d_path = C.run_traj(field, cluster, prim_d, *start, heading_offset=hdg)
    s1_path = C.run_traj(field, cluster, prim_s1, *start, heading_offset=hdg)

    letters = "abcdefgh"
    fig, axes = plt.subplots(1, len(ANCHOR_HOURS),
                             figsize=(3.5 * len(ANCHOR_HOURS), 3.5), sharey=True)
    im = None
    for k, (ax, h) in enumerate(zip(axes, ANCHOR_HOURS)):
        im = draw(ax, ftle_at(h), d_path, s1_path, start, h,
                  label_panel=f"({letters[k]}) $t$ = {h:.0f} h")

    handles = [Line2D([], [], color=TRACK_D, lw=2.2, ls="-", label="$D$ tracker"),
               Line2D([], [], color=TRACK_S1, lw=2.2, ls="--", label="$s_1$ tracker"),
               Line2D([], [], color="white", marker="*", ls="none", markersize=10,
                      markeredgecolor="black", label="shared start")]
    leg = axes[0].legend(handles=handles, loc="lower left", fontsize=8,
                         framealpha=0.82, borderpad=0.4, handlelength=1.8)
    leg.get_frame().set_edgecolor("none")

    fig.subplots_adjust(left=0.004, right=0.93, top=0.995, bottom=0.005, wspace=0.012)
    cax = fig.add_axes([0.940, 0.09, 0.009, 0.82])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"FTLE  [s$^{-1}$]", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = os.path.join(out_dir, f"timepanels_{name}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="balanced,corridor",
                    help="comma-separated criterion names from square_map_candidates.json")
    ap.add_argument("--from-params", type=str, default=None,
                    help="render from ocean_param_candidates.json instead: "
                         "'<key>:<index>,<key>:<index>' where key is best_ridge, "
                         "best_ridge_with_reach or pareto")
    args = ap.parse_args()

    if args.from_params:
        with open(os.path.join(C.OUT_DIR, "ocean_param_candidates.json")) as f:
            pc = json.load(f)
        os.makedirs(OUT_DIR, exist_ok=True)
        for spec in args.from_params.split(","):
            key, i = spec.split(":")
            r = pc[key.strip()][int(i)]
            tag = f"{key.strip()}{i}"
            print(f"Rendering {tag}: ({r['start_lat']:.5f},{r['start_lon']:.5f}) "
                  f"rho={r['rho_km']:.2f} km hdg={r['heading_deg']:.0f} "
                  f"k={r['gain']:.1f} c_max={r['v_max']:.3f}")
            render(tag, (r["start_lat"], r["start_lon"]), r, OUT_DIR,
                   scale=r["scale"], heading_deg=r["heading_deg"],
                   gain=r["gain"], v_max=r["v_max"])
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SHORTLIST_JSON) as f:
        short = json.load(f)
    by_name = {c["criterion"]: c for c in short["candidates"]}

    wanted = [w.strip() for w in args.which.split(",")]
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise SystemExit(f"--which: no such criterion {missing}. Have: {sorted(by_name)}")

    print(f"Anchors {ANCHOR_HOURS} h, {FTLE_HOURS}-h forward horizon each.")
    for w in wanted:
        c = by_name[w]
        print(f"Rendering {w} at ({c['start_lat']:.6f},{c['start_lon']:.6f})...")
        render(w, (c["start_lat"], c["start_lon"]), c, OUT_DIR)


if __name__ == "__main__":
    main()
