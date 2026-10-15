"""Ridge fidelity on Santa Barbara for |lambda2| vs signed lambda2 (dnew.py),
with the repo D, the pre-rewrite (published) D, and s1 for reference. Metric
of sbc_ridge_distance.py: distance from each centroid to the nearest 95th-pct
FTLE ridge point, paths cut at landfall. Part 2 repeats |l2| and signed from
16 starts within 1 km of the paper's start to see whether the difference holds."""
import io, contextlib, os, importlib.util
from multiprocessing import Pool
import numpy as np
from screen_common import VF
import main_ocean_hfr_2km_shared_start as S
from _ftle_common import compute_ftle_field
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
import src.control.pentagon_primitives as P
import dnew

spec = importlib.util.spec_from_file_location(
    "p_old", os.path.join(VF, "experiments", "primitive11_simplification", "overnight", "pentagon_primitives_pre_rewrite.py"))
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
K = S.CONTROL_GAIN
COSL = np.cos(np.radians(34.2))
STEPS = {
    "repo D": lambda c: P.separatrix_logic_c_step(c, v_max=S.V_MAX),
    "pre-rewrite D (published)": lambda c: OLD.separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM),
    "new |l2|": lambda c: dnew.d_step(c, S.V_MAX, abs_l2=True),
    "new signed l2": lambda c: dnew.d_step(c, S.V_MAX, abs_l2=False),
    "s1": lambda c: P.oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP),
}
with contextlib.redirect_stdout(io.StringIO()):
    lat_f, lon_f, f_val, land_f, _, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE)
L, LA = np.meshgrid(lon_f, lat_f)
water = ~land_f & np.isfinite(f_val)
ridge = water & (f_val >= np.percentile(f_val[water], 95))
R = np.column_stack([LA[ridge], L[ridge]])


def run(name, lat, lon):
    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
        field.config["isotropic_map"] = S.ISOTROPIC_MAP
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                             stiction_threshold=S.STICTION_THRESHOLD)
        path = S.run_traj(field, cl, lambda c: tuple(K * x for x in STEPS[name](c)), lat, lon)
    rob = np.array([[S._world_to_latlon(x, y, field.config) for x, y in st] for st in cl.get_robot_history()])
    i = np.clip(np.searchsorted(lat_f, rob[..., 0]), 0, len(lat_f) - 1)
    j = np.clip(np.searchsorted(lon_f, rob[..., 1]), 0, len(lon_f) - 1)
    hit = land_f[i, j].any(axis=1)
    k = int(np.argmax(hit)) if hit.any() else len(rob) - 1
    p = path[:k + 1]
    d = np.array([np.min(np.hypot((R[:, 0] - a) * 111.32, (R[:, 1] - b) * 111.32 * COSL)) for a, b in p])
    return dict(hours=k / 6, landfall=k < len(rob) - 1, end=p[-1], south=p[:, 0].min(),
                mean=d.mean(), median=np.median(d), within1=(d < 1.0).mean())


def job(a):
    name, lat, lon = a
    return name, lat, lon, run(name, lat, lon)


if __name__ == "__main__":
    print("1. Paper start: distance to nearest FTLE ridge point (95th pct), cut at landfall")
    with Pool(5) as pool:
        res = pool.map(job, [(n, *S.START) for n in STEPS])
    for n, _, _, r in res:
        print(f"  {n:26s} {r['hours']:4.1f} h {'landfall' if r['landfall'] else 'at sea  '} end ({r['end'][0]:.3f}N, {r['end'][1]:.3f})"
              f" | ridge distance mean {r['mean']:.2f} km, median {r['median']:.2f} km, within 1 km {100*r['within1']:.0f}%"
              f" | southmost {r['south']:.3f}N", flush=True)
    print("\n2. 16 starts within 1 km of the paper's start (rings of 0.5 and 1 km, 8 bearings)")
    starts = [(S.START[0] + rk * np.cos(t) / 111.32, S.START[1] + rk * np.sin(t) / (111.32 * COSL))
              for rk in (0.5, 1.0) for t in np.arange(8) * np.pi / 4]
    with Pool(7) as pool:
        res = pool.map(job, [(n, la, lo) for n in ("new |l2|", "new signed l2") for la, lo in starts])
    for n in ("new |l2|", "new signed l2"):
        rr = [r for m, _, _, r in res if m == n]
        print(f"  {n:14s} landfall {sum(r['landfall'] for r in rr):2d}/16 (median {np.median([r['hours'] for r in rr if r['landfall']]) if any(r['landfall'] for r in rr) else '-'} h)"
              f" | ridge distance mean over starts {np.mean([r['mean'] for r in rr]):.2f} km"
              f" | within 1 km {100*np.mean([r['within1'] for r in rr]):.0f}%"
              f" | southmost: median {np.median([r['south'] for r in rr]):.3f}N, reached south of 34.12N {sum(r['south'] < 34.12 for r in rr)}/16")
        print("     per start (mean km / southmost / end): " + "; ".join(
            f"{r['mean']:.2f}/{r['south']:.3f}/{'L' if r['landfall'] else 'S'}" for r in rr))
