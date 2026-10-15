import os, sys, io, contextlib
VF = os.getcwd(); sys.path.insert(0, VF); sys.path.insert(0, os.path.join(VF, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import main_ocean_hfr_2km_shared_start as S
import main_ocean_hfr_2km_progression as G
from _ftle_common import compute_ftle_field
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
from dvar import d_step

def run(p):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
        path = S.run_traj(field, cl, p, *S.START)
    rob = np.array([[S._world_to_latlon(x, y, field.config) for x, y in st] for st in cl.get_robot_history()])
    return path, rob
G_ = S.CONTROL_GAIN
mk = lambda band, cq, drift: (lambda c: tuple(G_ * x for x in d_step(c, S.V_MAX, band, cq, [], drift)))
variants = [("current band", 'orig', None), ("q_D < 1", 'qD', 1.0), ("q_D < 0.3", 'qD', 0.3), ("q_D < 0.03", 'qD', 0.03)]
colors = {"current band": "white", "q_D < 1": "lime", "q_D < 0.3": "yellow", "q_D < 0.03": "magenta"}
runs = {}
for drift in (True, False):
    for lab, band, cq in variants:
        runs[(drift, lab)] = run(mk(band, cq, drift))
pub, pub_rob = run(lambda c: tuple(G_ * x for x in separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)))
s1, s1_rob = run(lambda c: tuple(G_ * x for x in oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP)))
with contextlib.redirect_stdout(io.StringIO()):
    lat_f, lon_f, f_val, land_f, coast_polys, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE)
def cut(path, rob):
    i = np.clip(np.searchsorted(lat_f, rob[..., 0]), 0, len(lat_f) - 1)
    j = np.clip(np.searchsorted(lon_f, rob[..., 1]), 0, len(lon_f) - 1)
    hit = land_f[i, j].any(axis=1)
    k = int(np.argmax(hit)) if hit.any() else None
    return (path[:k + 1], k) if k is not None else (path, None)
L, LA = np.meshgrid(lon_f, lat_f)
inview = (L >= G.VIEW_LON[0]) & (L <= G.VIEW_LON[1]) & (LA >= G.VIEW_LAT[0]) & (LA <= G.VIEW_LAT[1])
water = f_val[inview & ~land_f & np.isfinite(f_val)]
norm = Normalize(vmin=np.percentile(water, G.COLOR_PCT[0]), vmax=np.percentile(water, G.COLOR_PCT[1]))
jet = plt.get_cmap("jet").copy(); jet.set_bad(color=G.LAND_GREEN)
outline = [pe.Stroke(linewidth=2.4, foreground="black"), pe.Normal()]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.93, wspace=0.04)
pub_c, kp = cut(pub, pub_rob); s1_c, ks = cut(s1, s1_rob)
for ax, drift in zip(axes, (True, False)):
    ax.pcolormesh(L, LA, np.ma.array(f_val, mask=land_f), cmap=jet, norm=norm, shading="auto", rasterized=True)
    for poly in coast_polys: ax.fill(poly[:, 0], poly[:, 1], color=G.LAND_GREEN, lw=0, zorder=2)
    ax.plot(pub_c[:, 1], pub_c[:, 0], color="black", lw=4.5, alpha=0.45, zorder=3, label="published D (Fig. 9)")
    ax.plot(s1_c[:, 1], s1_c[:, 0], color="cyan", lw=1.0, ls=(0, (2, 1.5)), zorder=4, path_effects=outline, label="s1 tracker")
    for lab, _, _ in variants:
        path, rob = runs[(drift, lab)]; pc, k = cut(path, rob)
        ax.plot(pc[:, 1], pc[:, 0], color=colors[lab], lw=1.3, zorder=5, path_effects=outline,
                label=f"{lab}" + (f" (landfall {k/6:.1f} h)" if k is not None else " (28 h, no landfall)"))
        ax.plot(pc[-1, 1], pc[-1, 0], "o", color=colors[lab], ms=5, mec="black", zorder=6)
    ax.plot(S.START[1], S.START[0], marker="*", color="lime", ms=12, mec="black", zorder=7)
    ax.set_xlim(*G.VIEW_LON); ax.set_ylim(*G.VIEW_LAT); ax.set_aspect(1 / np.cos(np.radians(34.2)))
    ax.set_title(f"D tracker variants, FLOW_DRIFT {'kept' if drift else 'disabled'}", fontsize=10)
    ax.set_xlabel("longitude"); ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
axes[0].set_ylabel("latitude")
out = os.path.join(VF, "experiments", "primitive11_simplification", "maps", "d_tracker_qD_sbc.png")
fig.savefig(out, dpi=200); print("Saved", out, "| published D landfall step", kp, "| s1 landfall", ks)
