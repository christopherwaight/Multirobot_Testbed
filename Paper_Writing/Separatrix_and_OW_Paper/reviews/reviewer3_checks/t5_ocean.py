"""Reviewer 3, check T5: null baselines for the Santa Barbara Channel ridge
metric.  Same start, same 168-step budget, same field and formation as the
paper's shared-start trial, scored with the paper's metric (mean distance
from the centroid path to the nearest FTLE grid point over water at or above
the 95th percentile, 24-h forward FTLE from the repo cache).

  D, s1     the published runs (reproduction check: paper 1.5 and 2.1 km)
  flow      follows the fitted centroid flow, k * sat(v0_hat), same k, c_max
  drifter   centroid velocity = fitted centroid flow, no gain, no cap
"""
import contextlib
import io
import os
import sys

import numpy as np

VFR = "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot"
sys.path.insert(0, VFR); sys.path.insert(0, os.path.join(VFR, "experiments"))
os.chdir(VFR)
with contextlib.redirect_stdout(io.StringIO()):
    import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
import src.control.pentagon_primitives as pp

z = np.load(os.path.join(VFR, "experiments/outputs/oecs/ftle_cache_24h_u4.npz"), allow_pickle=True)
lat, lon, ftle, land = z["lat"], z["lon"], z["ftle"], z["land"]
water = ~land & np.isfinite(ftle)
thr = np.percentile(ftle[water], 95)
LA, LO = np.meshgrid(lat, lon, indexing="ij")
ridge = np.c_[LA[water & (ftle >= thr)], LO[water & (ftle >= thr)]]
KM_LAT = 111.32
KM_LON = 111.32*np.cos(np.radians(34.2))


def mean_ridge_km(path):
    d = []
    for la, lo in path:
        dd = np.hypot((ridge[:, 0] - la)*KM_LAT, (ridge[:, 1] - lo)*KM_LON)
        d.append(dd.min())
    return float(np.mean(d))


with contextlib.redirect_stdout(io.StringIO()):
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    cluster = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                              stiction_threshold=S.STICTION_THRESHOLD)


def prim_d(c):
    vx, vy = pp.separatrix_logic_c_step(c, v_max=S.V_MAX, eps_raw=S.EPS_RAW, eps_dim=S.EPS_DIM)
    return vx*S.CONTROL_GAIN, vy*S.CONTROL_GAIN


def prim_s1(c):
    vx, vy = pp.oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP, s_trim=S.S_TRIM,
                                     r_band=S.R_BAND, g_capture=S.G_CAPTURE, s_capture=None)
    return vx*S.CONTROL_GAIN, vy*S.CONTROL_GAIN


def _flow(c):
    u_arr, v_arr = pp._sample_vector_at_robots(c)
    tu, tv = pp._fit_vector_quadratic(pp._get_relative_positions(c), u_arr, v_arr)
    return np.array([tu[0], tv[0]])


def prim_flow(c):
    f = _flow(c); n = np.linalg.norm(f)
    if n < 1e-12:
        return 0.0, 0.0
    sc = S.V_MAX*np.tanh(n/S.V_MAX)/n
    return f[0]*sc*S.CONTROL_GAIN, f[1]*sc*S.CONTROL_GAIN


def prim_drift(c):
    f = _flow(c)
    return float(f[0]), float(f[1])


out = {}
for name, prim in (("D", prim_d), ("s1", prim_s1), ("flow", prim_flow), ("drifter", prim_drift)):
    with contextlib.redirect_stdout(io.StringIO()):
        path = S.run_traj(field, cluster, prim, *S.START)
    km = mean_ridge_km(path)
    length = np.sum(np.hypot(np.diff(path[:, 0])*KM_LAT, np.diff(path[:, 1])*KM_LON))
    out[name] = (km, length, path[-1])
    print(f"{name:8s}: mean distance to ridge {km:.2f} km, path length {length:.1f} km, "
          f"end ({path[-1,0]:.3f}N, {-path[-1,1]:.3f}W)")

# Reference: distance from random water points to the ridge set
rng = np.random.default_rng(0)
wpts = np.c_[LA[water], LO[water]]
box = (wpts[:, 0] > 34.0) & (wpts[:, 0] < 34.45) & (wpts[:, 1] > -120.55) & (wpts[:, 1] < -120.2)
samp = wpts[box][rng.choice(box.sum(), 400, replace=False)]
print(f"random water points in the channel box: mean distance to ridge {mean_ridge_km(samp):.2f} km")
print(f"ridge threshold FTLE >= {thr:.3e}; {len(ridge)} ridge points")
