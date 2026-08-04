"""
_ocean_run_common.py

One definition of the 2km Santa Barbara Channel trial: the operating point, the
two controllers, the 168-step run loop, and the km-distance scoring helpers.

Extracted 2026-08-03 when the world map was made isotropic. Before that, the
same run_traj/km/closest_approach block was copy-pasted across
ocean_shared_start_search.py, ocean_shared_start_closest_approach.py,
ocean_candidateA_finemc_{search,v2,v3}.py, ocean_shared_start_visualize_top2.py
and main_ocean_hfr_2km_shared_start.py. Those copies are left as-is (they are
the provenance record for the pre-fix candidate A); everything written after
the map fix imports from here instead.

Operating point is the Ocean column of the paper's parameter table and is
identical to the pre-fix scripts. Nothing here changed with the map fix except
that the distance constants now match the field map's tangent plane
(111.32 km/deg lat) rather than the 111.0 the older search scripts used.
"""
import os
import sys
import json
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _HERE)

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step

from _coords_common import latlon_to_world, world_to_latlon

# ---------------------------------------------------------------------------
# Operating point (paper's Ocean column)
# ---------------------------------------------------------------------------

FORMATION_CONFIG  = "config/formations/pentagon_small_2km.yaml"
FIELD_CONFIG_NAME = "ocean_hfr_2km_timevarying"

V_MAX, SIM_STEPS, CONTROL_GAIN = 0.04, 168, 1.8
MOMENTUM_ALPHA, STICTION_THRESHOLD, TIME_WARP = 0.0, 0.002, 6000.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
G_PERP, S_TRIM, R_BAND, G_CAPTURE = 1.0, 0.3, 0.05, 0.15

# Published start of the pre-fix (anisotropic-map) figure, and the middle-island
# landfall the D tracker reached there. Both are targets, not settings.
LEGACY_START = (34.411906, -120.392016)
E0           = (34.0412, -120.2617)

# Landfall branch split used by ocean_hfr_2km_branch_sensitivity.py.
BRANCH_LON_SPLIT = -120.35

# Tangent-plane constants, same as Ocean_HFR.world_scales and _ftle_common.
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * float(np.cos(np.radians(34.2)))

# Plot/FTLE extent shared by every ocean figure.
LON_MIN, LON_MAX = -120.7, -119.7
LAT_MIN, LAT_MAX = 33.8, 34.7

_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "trunk", "Python_Simulations", "Vector_Fields",
                        "ocean_data", "hfr_uswc_2012may_2km")
FRAME_GLOB = "*_2km_rtv_uwls_SIO.nc"
PLOT_DIR = os.path.join(_REPO_ROOT, "trunk", "Python_Simulations", "Vector_Fields",
                        "ocean_data", "det_jacobian_plots")
COAST_SHP = os.path.join(_REPO_ROOT, "trunk", "Python_Simulations", "Vector_Fields",
                         "ocean_data", "coastlines", "ne_10m_land", "ne_10m_land.shp")
OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "outputs", "oecs")

LEGACY_REFERENCE_JSON = os.path.join(OUT_DIR, "legacy_corridor_reference.json")


# ---------------------------------------------------------------------------
# Trial setup
# ---------------------------------------------------------------------------

def build_trial(isotropic_map=None, v_max=V_MAX, gain=CONTROL_GAIN):
    """
    Build (field, cluster, prim_d, prim_s1).

    Defaults are the paper's Ocean operating point. v_max and gain are exposed
    so a search can vary them; formation scale and heading are varied through
    set_formation_scale and run_traj(heading_offset=...) instead, because those
    live on the cluster rather than on the primitives.

    Args:
        isotropic_map: None to take the config's value (true), or an explicit
                       bool to override it. False reproduces the pre-2026-08-03
                       map, which is how the legacy corridor reference is
                       regenerated.
    """
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=FIELD_CONFIG_NAME)
    if isotropic_map is not None:
        field.config = dict(field.config)
        field.config["isotropic_map"] = bool(isotropic_map)

    cluster = PentagonCluster(FORMATION_CONFIG, field,
                              momentum_alpha=MOMENTUM_ALPHA,
                              stiction_threshold=STICTION_THRESHOLD)

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=v_max, eps_raw=EPS_RAW, eps_dim=EPS_DIM)
        return vx * gain, vy * gain

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=v_max, g_perp=G_PERP, s_trim=S_TRIM,
                                      r_band=R_BAND, g_capture=G_CAPTURE, s_capture=None)
        return vx * gain, vy * gain

    return field, cluster, prim_d, prim_s1


_FORMATION_LENGTHS = ("desired_L_2", "desired_L_3", "desired_L_4",
                      "desired_p_1", "desired_q_1")


def set_formation_scale(cluster, scale):
    """
    Rescale the pentagon uniformly about its centroid.

    All five SAS length parameters scale together, so the shape is unchanged
    and only the ring radius rho moves. The yaml's own values are stashed on
    first call and every later call is applied to those, so repeated calls do
    not compound. Angles (theta_*, beta_1) are untouched.

    scale = 1.0 is the config as written, rho = 0.105 world units (5.4 km).
    """
    if not hasattr(cluster, "_formation_base_lengths"):
        cluster._formation_base_lengths = {k: getattr(cluster, k)
                                           for k in _FORMATION_LENGTHS}
    for k, base in cluster._formation_base_lengths.items():
        setattr(cluster, k, base * scale)
    cluster._formation_scale = scale


def rho_km(scale=1.0):
    """Physical ring radius in km for a given formation scale."""
    return 0.105 * scale * (KM_PER_DEG_LAT * (0.3 / 0.65))


def run_traj(field, cluster, prim, lat, lon, sim_steps=SIM_STEPS, heading_offset=0.0):
    """
    One 168-step trial from a lat/lon start. Returns (N, 2) lat/lon centroid path.

    cluster.reset also clears the OECS tangent/mode state, without which a
    reused cluster's second run is contaminated by the first. heading_offset
    rotates the whole formation rigidly about its centroid at t=0.
    """
    sx, sy = latlon_to_world(lat, lon, field.config)
    cluster.reset(sx, sy, heading_offset=heading_offset)
    field.reset_clock()
    for _ in range(sim_steps):
        cluster.move(prim)
        field.step(cluster.timestep * TIME_WARP)
    center = cluster.get_center_history()
    return np.array([world_to_latlon(p[0], p[1], field.config) for p in center])


# ---------------------------------------------------------------------------
# Scoring helpers (all distances in km on the local tangent plane)
# ---------------------------------------------------------------------------

def km(p, q):
    """Distance in km between two (lat, lon) points."""
    return float(np.hypot((p[0] - q[0]) * KM_PER_DEG_LAT,
                          (p[1] - q[1]) * KM_PER_DEG_LON))


def to_km_xy(path_latlon):
    """(N,2) lat/lon path -> (N,2) east/north km, for distance work."""
    return np.column_stack([path_latlon[:, 1] * KM_PER_DEG_LON,
                            path_latlon[:, 0] * KM_PER_DEG_LAT])


def closest_approach(path_latlon, target):
    """
    (min_dist_km, frac_of_run) from any point on the path to `target`.

    frac 0 = start, 1 = final step. Near 1.0 means still closing at the end;
    a lower value means the path swung close and continued through.
    """
    d = np.hypot((path_latlon[:, 0] - target[0]) * KM_PER_DEG_LAT,
                 (path_latlon[:, 1] - target[1]) * KM_PER_DEG_LON)
    i = int(np.argmin(d))
    return float(d[i]), i / max(len(d) - 1, 1)


def path_dev_km(path_ref, path_test):
    """
    Mean over points of path_test of the distance to the nearest point of
    path_ref, in km: how far the test path strays off the reference corridor.

    Deviation alone cannot say whether a corridor was followed. A path that
    stalls near the reference corridor's start never strays and scores near
    zero. Always read this together with path_cover_km, which penalizes
    exactly that.
    """
    a = to_km_xy(path_ref)
    b = to_km_xy(path_test)
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def path_dev_max_km(path_ref, path_test):
    """Worst-case counterpart of path_dev_km."""
    a = to_km_xy(path_ref)
    b = to_km_xy(path_test)
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=1).max())


def path_cover_km(path_ref, path_test):
    """
    Mean over points of path_ref of the distance to the nearest point of
    path_test, in km: how much of the reference corridor the test path
    actually covered. The reverse direction of path_dev_km.

    A stalled path scores badly here because the far end of the reference
    corridor has nothing near it. Low deviation AND low coverage distance
    together mean the test path traversed the reference corridor.
    """
    a = to_km_xy(path_ref)
    b = to_km_xy(path_test)
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return float(d.min(axis=0).mean())


def ridge_tree(ftle_npz_path, percentile=95.0):
    """
    KD-tree over the FTLE ridge, for measuring how closely a path tracks the
    transport skeleton independently of the published paths.

    The ridge is the set of grid points at or above `percentile` of the FTLE
    field, the same definition ocean_hfr_2km_branch_sensitivity.py uses.
    Tree coordinates are km east/north, so query distances are km.

    Returns (tree, n_ridge_points), or (None, 0) if the cache is absent.
    """
    if not os.path.exists(ftle_npz_path):
        return None, 0
    from scipy.spatial import cKDTree
    z = np.load(ftle_npz_path, allow_pickle=True)
    lat_f, lon_f, f_val, land = z["lat"], z["lon"], z["ftle"], z["land"]
    LON, LAT = np.meshgrid(lon_f, lat_f)
    ok = np.isfinite(f_val) & (~land)
    thr = np.percentile(f_val[ok], percentile)
    sel = ok & (f_val >= thr)
    pts = np.column_stack([LON[sel] * KM_PER_DEG_LON, LAT[sel] * KM_PER_DEG_LAT])
    return cKDTree(pts), int(sel.sum())


def dist_to_ridge_km(tree, path_latlon):
    """(mean, max) distance in km from a path to the FTLE ridge."""
    if tree is None:
        return float("nan"), float("nan")
    d, _ = tree.query(to_km_xy(path_latlon))
    return float(d.mean()), float(d.max())


def path_length_km(path_latlon):
    xy = to_km_xy(path_latlon)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def mean_turning_angle(path_latlon):
    """
    Mean absolute turning angle (radians) between consecutive centroid steps.
    0 = straight, pi = reversing every step. Near-zero-displacement steps are
    skipped so a parked formation does not dominate the average.
    """
    xy = to_km_xy(path_latlon)
    seg = np.diff(xy, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    seg = seg[lens > 1e-4]
    if len(seg) < 3:
        return 0.0
    v1, v2 = seg[:-1], seg[1:]
    cos_a = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
    return float(np.mean(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def branch_of(end_latlon):
    """Landfall branch classification, same split as the branch-sensitivity script."""
    return "south/island" if end_latlon[1] >= BRANCH_LON_SPLIT else "west/offshore"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def atomic_write_json(path, obj):
    """Write JSON via a temp file and os.replace, so a kill mid-write cannot truncate."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_legacy_reference():
    """
    Load the frozen pre-fix D and s1 paths.

    Returns (start, d_path, s1_path) with the paths as (N,2) lat/lon arrays.
    Regenerate with ocean_legacy_corridor_reference.py.
    """
    with open(LEGACY_REFERENCE_JSON) as f:
        d = json.load(f)
    return (tuple(d["start"]),
            np.array(d["d_path"], dtype=float),
            np.array(d["s1_path"], dtype=float))
