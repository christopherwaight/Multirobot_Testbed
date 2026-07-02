"""
Ocean HF radar vector field, Santa Barbara Channel, May 2012.

Two public field functions:

  ocean_hfr_socal           -- static (first frame only).
                               Reads config/fields/ocean_hfr.yaml.

  ocean_hfr_socal_timevarying -- linear temporal interpolation across all frames.
                               Reads config/fields/ocean_hfr_timevarying.yaml.

Both use an affine coordinate map: cluster world coordinates in
[-world_half, world_half] -> lat/lon within +/- roi_half_deg of
(center_lat, center_lon). The cluster simulates in world coords;
this module converts to lat/lon internally for the NetCDF lookup.

Time handling:
  enable_time_interp: false -> load only files[0]; field.t is ignored.
  enable_time_interp: true  -> load all matching files; field.t is used to
                               linearly interpolate between bracketing frames.
  TIME_WARP in the experiment file scales the effective dt passed to
  field.step() so the field clock can advance faster than wall time.

Data source: trunk/Python_Simulations/Vector_Fields/ocean_data/hfr_uswc_2012may/
Format: NetCDF-4, variables: lat, lon, u (m/s), v (m/s), time (Unix epoch s).
Fill sentinel: -32767.0 (replaced with NaN before interpolation).
"""
import os
import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FILL     = -32767.0

# Default affine-map parameters (overridden by YAML config).
_DEFAULT_CENTER_LAT = 34.2
_DEFAULT_CENTER_LON = -120.4
_DEFAULT_ROI_HALF   = 0.3    # degrees
_DEFAULT_WORLD_HALF = 0.65   # matches runner.py vector-field plot bounds

# Per-config-name dataset cache.
# Keyed by the "cache_key" value in each YAML (or a fallback derived from the
# config dict). Two field functions with different config names therefore load
# and cache their frames independently.
_DATASET_CACHE = {}   # {cache_key_str: dataset_dict}


# ---------------------------------------------------------------------------
# Internal I/O helpers (adapted from ocean_data/plot_det_jacobian.py)
# ---------------------------------------------------------------------------

def _load_frame(fp):
    """Load one NetCDF frame. Returns (lat, lon, u, v, t)."""
    import netCDF4 as nc
    with nc.Dataset(fp) as f:
        lat = np.array(f.variables["lat"][:], dtype=float)
        lon = np.array(f.variables["lon"][:], dtype=float)
        u   = np.array(f.variables["u"][0, :, :], dtype=float)
        v   = np.array(f.variables["v"][0, :, :], dtype=float)
        t   = float(f.variables["time"][0])
    u[np.abs(u - _FILL) < 1.0] = np.nan
    v[np.abs(v - _FILL) < 1.0] = np.nan
    return lat, lon, u, v, t


def _gap_fill(lat, lon, u, v):
    """Fill NaN gaps via linear griddata, then zero-fill remaining edge NaN."""
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    uf, vf = u.ravel(), v.ravel()
    ok = np.isfinite(uf) & np.isfinite(vf)
    if ok.sum() < 10:
        return np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)
    uf2 = griddata(pts[ok], uf[ok], pts, method="linear").reshape(u.shape)
    vf2 = griddata(pts[ok], vf[ok], pts, method="linear").reshape(v.shape)
    return (np.where(np.isfinite(uf2), uf2, 0.0),
            np.where(np.isfinite(vf2), vf2, 0.0))


def _make_interp(lat, lon, u, v):
    """Build (u_interp, v_interp) as RegularGridInterpolators (linear, fill=0)."""
    ui = RegularGridInterpolator((lat, lon), u,
                                  method="linear",
                                  bounds_error=False,
                                  fill_value=0.0)
    vi = RegularGridInterpolator((lat, lon), v,
                                  method="linear",
                                  bounds_error=False,
                                  fill_value=0.0)
    return ui, vi


def _cache_key(cfg):
    """Derive a stable string key from the config dict."""
    explicit = cfg.get("cache_key")
    if explicit:
        return str(explicit)
    # Fallback: data_dir + enable_time_interp.
    this_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", "..", "..", ".."))
    default_data = os.path.join(
        repo_root,
        "trunk", "Python_Simulations", "Vector_Fields",
        "ocean_data", "hfr_uswc_2012may"
    )
    data_dir = cfg.get("data_dir", default_data)
    return f"{data_dir}|{bool(cfg.get('enable_time_interp', False))}"


def _load_dataset(cfg):
    """
    Load and cache HFR frames for the given config dict.

    Skips if the cache key is already populated.
    """
    key = _cache_key(cfg)
    if key in _DATASET_CACHE:
        return

    enable_time_interp = bool(cfg.get("enable_time_interp", False))
    frame_glob         = cfg.get("frame_glob", "*_6km_rtv_uwls_SIO.nc")

    this_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", "..", "..", ".."))
    default_data = os.path.join(
        repo_root,
        "trunk", "Python_Simulations", "Vector_Fields",
        "ocean_data", "hfr_uswc_2012may"
    )
    data_dir = cfg.get("data_dir", default_data)

    files = sorted(glob.glob(os.path.join(data_dir, frame_glob)))
    if not files:
        raise FileNotFoundError(
            f"Ocean_HFR: no files matching '{frame_glob}' in {data_dir}"
        )

    if not enable_time_interp:
        files = files[:1]

    u_interps, v_interps, t_frames = [], [], []
    for fp in files:
        lat, lon, u, v, t = _load_frame(fp)
        u, v = _gap_fill(lat, lon, u, v)
        ui, vi = _make_interp(lat, lon, u, v)
        u_interps.append(ui)
        v_interps.append(vi)
        t_frames.append(t)

    _DATASET_CACHE[key] = {
        "u_interps":          u_interps,
        "v_interps":          v_interps,
        "t_frames":           np.array(t_frames, dtype=float),
        "enable_time_interp": enable_time_interp,
    }


# ---------------------------------------------------------------------------
# Shared evaluation kernel
# ---------------------------------------------------------------------------

def _evaluate(x, y, t, config):
    """
    Evaluate (u, v) at world position (x, y) and field time t.

    Loads dataset on first call for this config. Subsequent calls use the cache.
    """
    cfg = config or {}
    _load_dataset(cfg)
    key = _cache_key(cfg)
    ds  = _DATASET_CACHE[key]

    center_lat = cfg.get("center_lat",   _DEFAULT_CENTER_LAT)
    center_lon = cfg.get("center_lon",   _DEFAULT_CENTER_LON)
    roi_half   = cfg.get("roi_half_deg", _DEFAULT_ROI_HALF)
    world_half = cfg.get("world_half",   _DEFAULT_WORLD_HALF)

    scale = roi_half / world_half
    lat_q = center_lat + y * scale
    lon_q = center_lon + x * scale
    pt    = np.array([[lat_q, lon_q]])

    if not ds["enable_time_interp"] or len(ds["t_frames"]) == 1:
        return float(ds["u_interps"][0](pt)), float(ds["v_interps"][0](pt))

    # Linear temporal interpolation between bracketing frames.
    # NOTE (bug fix): callers pass `t` as seconds elapsed since simulation
    # start (field.reset_clock() sets self.t = 0.0), but t_frames holds
    # absolute Unix epoch seconds (~1.3e9). Previously t was clipped
    # directly against t_arr, which clamped every simulation to frame 0
    # for the entire run regardless of TIME_WARP -- the field never
    # actually varied in time. Convert relative t to absolute by adding
    # the first frame's epoch before clipping.
    t_arr = ds["t_frames"]
    t_abs = t_arr[0] + t
    t_c   = float(np.clip(t_abs, t_arr[0], t_arr[-1]))
    i     = int(np.searchsorted(t_arr, t_c, side="right")) - 1
    i     = int(np.clip(i, 0, len(t_arr) - 2))
    dt    = t_arr[i + 1] - t_arr[i]
    alpha = (t_c - t_arr[i]) / dt if dt > 0.0 else 0.0

    u0 = float(ds["u_interps"][i](pt))
    v0 = float(ds["v_interps"][i](pt))
    u1 = float(ds["u_interps"][i + 1](pt))
    v1 = float(ds["v_interps"][i + 1](pt))
    return (1.0 - alpha) * u0 + alpha * u1, (1.0 - alpha) * v0 + alpha * v1


# ---------------------------------------------------------------------------
# Public field functions
# ---------------------------------------------------------------------------

def ocean_hfr_socal(x, y, t=0.0, config=None):
    """
    Ocean HFR surface current field -- static variant (first frame only).

    Args:
        x: World x coordinate (scalar).
        y: World y coordinate (scalar).
        t: Field time (seconds). Ignored when enable_time_interp is False.
        config: Dict from config/fields/ocean_hfr.yaml, or None for defaults.

    Returns:
        (u, v): Surface current components (m/s).
    """
    return _evaluate(x, y, t, config)


ocean_hfr_socal.config_name = "ocean_hfr"


def ocean_hfr_socal_timevarying(x, y, t=0.0, config=None):
    """
    Ocean HFR surface current field -- time-varying variant (all frames, linear interp).

    Args:
        x: World x coordinate (scalar).
        y: World y coordinate (scalar).
        t: Field time (seconds). Linearly interpolated across hourly frames.
           Advance with field.step(cluster.timestep * TIME_WARP) in the experiment.
        config: Dict from config/fields/ocean_hfr_timevarying.yaml, or None.

    Returns:
        (u, v): Surface current components (m/s) interpolated at time t.
    """
    return _evaluate(x, y, t, config)


ocean_hfr_socal_timevarying.config_name = "ocean_hfr_timevarying"
