"""Santa Barbara screen: repo D tracker (current Fig. 9) against the dnew.py
variants (|lambda2| and signed lambda2), with the s1 path for reference.
Writes maps/d_tracker_newton_sbc.png."""
import io, contextlib
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from screen_common import VF
import os
import main_ocean_hfr_2km_shared_start as S
import main_ocean_hfr_2km_progression as G
from _ftle_common import compute_ftle_field
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
import dnew

K = S.CONTROL_GAIN


def run(step):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                             stiction_threshold=S.STICTION_THRESHOLD)
        cl.diagnostics = []
        path = S.run_traj(field, cl, lambda c: tuple(K * x for x in step(c)), *S.START)
    rob = np.array([[S._world_to_latlon(x, y, field.config) for x, y in st] for st in cl.get_robot_history()])
    return path, rob, cl


def km(a, b):
    dlat = (a[..., 0] - b[..., 0]) * 111.0
    dlon = (a[..., 1] - b[..., 1]) * 111.0 * np.cos(np.radians(34.2))
    return np.hypot(dlat, dlon)


runs = {}
repo_path, repo_rob, repo_cl = run(lambda c: separatrix_logic_c_step(c, v_max=S.V_MAX))
runs["repo D (current Fig. 9)"] = (repo_path, repo_rob, Counter(d['mode'] for d in repo_cl.diagnostics), [])
for lab, absl2 in (("new D, |lambda2|", True), ("new D, signed lambda2", False)):
    log = []
    p, r, _ = run(lambda c, absl2=absl2, log=log: dnew.d_step(c, S.V_MAX, abs_l2=absl2, log=log))
    runs[lab] = (p, r, Counter(log), [k for k, m in enumerate(log) if m == 'TRACK_DEFINITE'])
s1, s1_rob, _ = run(lambda c: oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP))

with contextlib.redirect_stdout(io.StringIO()):
    lat_f, lon_f, f_val, land_f, coast_polys, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE)


def landfall(rob):
    i = np.clip(np.searchsorted(lat_f, rob[..., 0]), 0, len(lat_f) - 1)
    j = np.clip(np.searchsorted(lon_f, rob[..., 1]), 0, len(lon_f) - 1)
    hit = land_f[i, j].any(axis=1)
    return int(np.argmax(hit)) if hit.any() else len(rob) - 1


def ring(P):
    cen = P.mean(axis=0); r = P[np.argsort(np.linalg.norm(P - cen, axis=1))[1:]]
    r = r[np.argsort(np.arctan2(r[:, 0] - cen[0], r[:, 1] - cen[1]))]
    return np.vstack([r, r[:1]])


L, LA = np.meshgrid(lon_f, lat_f)
inview = (L >= G.VIEW_LON[0]) & (L <= G.VIEW_LON[1]) & (LA >= G.VIEW_LAT[0]) & (LA <= G.VIEW_LAT[1])
water = f_val[inview & ~land_f & np.isfinite(f_val)]
norm = Normalize(vmin=np.percentile(water, G.COLOR_PCT[0]), vmax=np.percentile(water, G.COLOR_PCT[1]))
jet = plt.get_cmap("jet").copy(); jet.set_bad(color=G.LAND_GREEN)
ol = [pe.Stroke(linewidth=2.6, foreground="black"), pe.Normal()]
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.08, top=0.90, wspace=0.03)
ks = landfall(s1_rob); kr = landfall(repo_rob)
for ax, (lab, (path, rob, cnt, dfn)) in zip(axes, runs.items()):
    k = landfall(rob)
    ax.pcolormesh(L, LA, np.ma.array(f_val, mask=land_f), cmap=jet, norm=norm, shading="auto", rasterized=True)
    for poly in coast_polys:
        ax.fill(poly[:, 0], poly[:, 1], color=G.LAND_GREEN, lw=0, zorder=2)
    ax.plot(s1[:ks+1, 1], s1[:ks+1, 0], color="magenta", lw=1.0, ls=(0, (3, 1.5)), zorder=4, path_effects=ol, label="s1 tracker")
    if lab != "repo D (current Fig. 9)":
        ax.plot(repo_path[:kr+1, 1], repo_path[:kr+1, 0], color="black", lw=5, alpha=0.35, zorder=3, label="repo D (current Fig. 9)")
    end = f"landfall {k/6:.1f} h" if k < len(rob) - 1 else f"at sea, {k/6:.0f} h"
    ax.plot(path[:k+1, 1], path[:k+1, 0], color="white", lw=1.3, zorder=5, path_effects=ol, label=f"{lab}, {end}")
    dd = [i for i in dfn if i <= k]
    if dd:
        ax.plot(path[dd, 1], path[dd, 0], "o", ms=3.5, color="yellow", mec="black", mew=0.5, zorder=6,
                label=f"definite H_D steps ({len(dd)})")
    for t in sorted(set([0, 42, 84, 126, k])):
        if t <= k:
            r = ring(rob[t]); ax.plot(r[:, 1], r[:, 0], color="white", lw=0.9, zorder=6, path_effects=ol)
    ax.plot(S.START[1], S.START[0], marker="*", color="lime", ms=11, mec="black", zorder=7)
    ax.set_xlim(-120.55, -120.15); ax.set_ylim(33.98, 34.45); ax.set_aspect(1 / np.cos(np.radians(34.2)))
    ax.set_title(f"{lab}\n" + ", ".join(f"{m} {n}" for m, n in sorted(cnt.items())), fontsize=8.5)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85); ax.set_xlabel("longitude", fontsize=8)
axes[0].set_ylabel("latitude", fontsize=8)
out = os.path.join(VF, "experiments", "primitive11_simplification", "maps", "d_tracker_newton_sbc.png")
fig.savefig(out, dpi=200)
print("Saved", out)
for lab, (path, rob, cnt, dfn) in runs.items():
    k = landfall(rob)
    n = min(len(path), len(repo_path))
    gap = km(path[:n], repo_path[:n]); gs = km(path[:n], s1[:n])
    first = next((i for i, g in enumerate(gap) if g > 1.0), None)
    print(f"{lab}: end ({path[k,0]:.3f}N, {path[k,1]:.3f}) at {k/6:.1f} h"
          + (" (landfall)" if k < len(rob) - 1 else " (at sea)")
          + f" | modes {dict(cnt)} | gap to repo D: max {gap.max():.1f} km, first >1 km at {first/6 if first is not None else '-'} h"
          + f" | gap to s1: max {gs.max():.1f} km, at end {gs[min(k, n-1)]:.1f} km")
print(f"s1: end ({s1[ks,0]:.3f}N, {s1[ks,1]:.3f}) at {ks/6:.1f} h")
