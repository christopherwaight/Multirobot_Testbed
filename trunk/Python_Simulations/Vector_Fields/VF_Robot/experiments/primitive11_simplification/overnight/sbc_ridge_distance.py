"""Mean distance from each tracker's SBC centroid path to the nearest FTLE ridge
point (Draft_12 Sec. V-B). Ridge points: water points at or above the 95th
percentile of the 24-h forward FTLE field (same field as Fig. 9). Paths are cut
at landfall. Usage: sbc_ridge_distance.py [old]  ('old' = pre-rewrite primitives)."""
import os, sys, io, contextlib, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
import main_ocean_hfr_2km_shared_start as S
from _ftle_common import compute_ftle_field
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
if len(sys.argv) > 1 and sys.argv[1] == "old":
    spec = importlib.util.spec_from_file_location("p_old", os.path.join(here, "pentagon_primitives_pre_rewrite.py"))
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
else:
    import src.control.pentagon_primitives as P
def run(prim):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA, stiction_threshold=S.STICTION_THRESHOLD)
        path = S.run_traj(field, cl, prim, *S.START)
    rob = np.array([[S._world_to_latlon(x, y, field.config) for x, y in st] for st in cl.get_robot_history()])
    return path, rob
K = S.CONTROL_GAIN
d_path, d_rob = run(lambda c: tuple(K * x for x in P.separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)))
s_path, s_rob = run(lambda c: tuple(K * x for x in P.oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP, s_trim=S.S_TRIM,
                                                                        r_band=S.R_BAND, g_capture=S.G_CAPTURE, s_capture=None)))
with contextlib.redirect_stdout(io.StringIO()):
    lat_f, lon_f, f_val, land_f, _, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE)
L, LA = np.meshgrid(lon_f, lat_f)
water = ~land_f & np.isfinite(f_val)
ridge = water & (f_val >= np.percentile(f_val[water], 95))
R = np.column_stack([LA[ridge], L[ridge]])
cosl = np.cos(np.radians(34.2))
def landfall(rob):
    i = np.clip(np.searchsorted(lat_f, rob[..., 0]), 0, len(lat_f) - 1)
    j = np.clip(np.searchsorted(lon_f, rob[..., 1]), 0, len(lon_f) - 1)
    hit = land_f[i, j].any(axis=1)
    return int(np.argmax(hit)) if hit.any() else len(rob) - 1
for name, path, rob in (("D", d_path, d_rob), ("s1", s_path, s_rob)):
    k = landfall(rob); p = path[:k + 1]
    d = np.array([np.min(np.hypot((R[:, 0] - a) * 111.32, (R[:, 1] - b) * 111.32 * cosl)) for a, b in p])
    print(f"{name}: {len(p)} steps ({k/6:.1f} h{', landfall' if k < len(rob) - 1 else ''}), end ({p[-1,0]:.4f}N, {p[-1,1]:.4f}), "
          f"mean distance to nearest ridge point {d.mean():.2f} km, median {np.median(d):.2f} km")
