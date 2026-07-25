"""
_mc_common.py

Shared Monte Carlo trial runner for the noise sweeps of
Paper_Writing/Separatrix_and_OW_Paper/Draft_5c.tex (Testing Plan /
Results sections). Used by mc_sweep_separatrix.py (and later
mc_sweep_ow.py). Not an entry point.

One trial = one (controller, sigma_uv, sigma_p, seed) tuple:
  - start drawn uniformly over START_BOX (basin structure recoverable by
    binning start_x/start_y from the CSV rows)
  - random initial heading via reset(heading_offset=U[0, 2pi))
  - measurement/position noise via cluster.measurement_noise_std /
    cluster.position_noise_std (Eqs. meas_noise / pos_noise of the paper)
  - metrics per the merged test plan (plan.md Phase 4).
"""
import io
import contextlib

import numpy as np

from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Double_Gyre import double_gyre_static

FORMATION_CONFIG = "config/formations/pentagon_small.yaml"
V_MAX, GAIN = 0.04, 3.0
EPS_RAW, EPS_DIM = 1e-3, 0.025
SIM_STEPS = 500
# Paper experiment (user 2026-07-02): ONE consistent straddling start on
# the separatrix, noise dialed up; success then measures pure noise
# robustness of the traverse, Michini-style. Random starts (START_BOX)
# remain available for basin evidence only; the paper does not claim to
# characterize the basin of attraction.
FIXED_START = (0.0, 0.35)
START_BOX = (-0.8, 0.8, -0.4, 0.4)   # x_min, x_max, y_min, y_max
BAND_X, BAND_HOLD = 0.05, 10
# Trials STOP at bottom-saddle contact or domain exit (same rule as
# separatrix_clean_runs.py): the controller's intended behavior is to
# continue past the saddle onto the wall trench, which would otherwise
# contaminate the straddle and tracking metrics (a successful traverse
# stops straddling x=0 once it turns onto the wall).
SADDLE = (0.0, -0.5)
SADDLE_CONTACT_D = 0.06
# Formation collapse: RMS error of the 15 inter-robot distances vs their
# nominal values exceeding 0.30, i.e. 4x the ring radius / nominal pair
# length L_2 = 0.075 of the ACTIVE pentagon_small.yaml block. (An earlier
# comment said "2x L_2 = 0.15"; that anchors to the full-scale formation,
# which no experiment uses. The threshold value itself is unchanged.)
COLLAPSE_RMS = 0.30

_NOMINAL_PDIST = None


def _pairdists(pts):
    d = []
    for i in range(6):
        for j in range(i + 1, 6):
            d.append(np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
    return np.array(d)


def run_trial(spec):
    """spec: dict(sigma_uv, sigma_p, seed, primitive). Returns row dict."""
    global _NOMINAL_PDIST
    np.random.seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])

    with contextlib.redirect_stdout(io.StringIO()):
        field = AnalyticalField(double_gyre_static)
        cluster = PentagonCluster(FORMATION_CONFIG, field)

    if spec.get("start") is not None:
        x0, y0 = spec["start"]
    else:
        x0 = rng.uniform(START_BOX[0], START_BOX[1])
        y0 = rng.uniform(START_BOX[2], START_BOX[3])
    heading = spec.get("heading")
    if heading is None:
        heading = rng.uniform(0.0, 2 * np.pi)
    cluster.reset(x0, y0, heading_offset=heading)
    cluster.measurement_noise_std = spec["sigma_uv"]
    cluster.position_noise_std = spec["sigma_p"]

    if _NOMINAL_PDIST is None:
        pts0 = np.array(cluster.get_robot_positions()).reshape(6, 2)
        _NOMINAL_PDIST = _pairdists(pts0)

    prim = spec["primitive"]
    effort = 0.0

    def wrapped(c):
        nonlocal effort
        vx, vy = prim(c)
        effort += np.hypot(vx, vy) * c.timestep
        return vx, vy

    # Terminal target defaults to the bottom saddle (Logic C traverse);
    # the OECS core-seek sweep passes the TOP saddle, its segment's TRAP
    # core (mc_sweep_oecs.py). A list of (x, y) tuples is also accepted
    # (mc_sweep_oecs_traverse.py): the traverser's CAPTURE branch holds
    # whichever saddle its flow-seeded tangent orientation leads to, not
    # a pre-committed one, so success is contact with ANY listed target.
    target = spec.get("target", SADDLE)
    targets = [target] if isinstance(target[0], (int, float)) else list(target)
    max_steps = spec.get("max_steps", SIM_STEPS)
    x_exit = spec.get("x_exit", 1.0)
    y_exit = spec.get("y_exit", 0.52)
    reached_saddle = False
    for _ in range(max_steps):
        cluster.move(wrapped)
        cx, cy = cluster.get_centroid()
        if min(np.hypot(cx - tx, cy - ty) for tx, ty in targets) < SADDLE_CONTACT_D:
            reached_saddle = True
            break
        if abs(cx) > x_exit or abs(cy) > y_exit:
            break

    hist = cluster.get_center_history()
    rhist = cluster.get_robot_history()
    xs = hist[:, 0]

    # time-to-band
    inband = np.abs(xs) < BAND_X
    t_band = -1
    for k in range(max(1, len(inband) - BAND_HOLD)):
        if inband[k:k + BAND_HOLD].all():
            t_band = k
            break

    # formation collapse (max over trial)
    rms_max = 0.0
    for k in range(0, len(rhist), 10):
        err = _pairdists(rhist[k]) - _NOMINAL_PDIST
        rms_max = max(rms_max, float(np.sqrt(np.mean(err ** 2))))
    collapsed = rms_max > COLLAPSE_RMS

    # straddle retention over the trench-tracking phase (trial already
    # stops at saddle contact): straddling from first straddle to stop
    rx = rhist[:, :, 0]
    straddle = (rx.min(axis=1) < 0.0) & (rx.max(axis=1) > 0.0)
    first = int(np.argmax(straddle)) if straddle.any() else -1
    straddle_ok = bool(straddle.any() and straddle[first:].all())

    # tracking error |x_c| over the on-trench phase (band entry to stop)
    if t_band >= 0 and len(xs) - t_band > 5:
        track = np.abs(xs[t_band:])
    else:
        track = np.abs(xs[-50:])
    return {
        "sigma_uv": spec["sigma_uv"], "sigma_p": spec["sigma_p"],
        "seed": spec["seed"], "start_x": x0, "start_y": y0,
        "heading": heading, "t_band": t_band, "steps": len(xs),
        "success_traverse": int(reached_saddle and not collapsed),
        "success_band": int(t_band >= 0 and not collapsed),
        "success_straddle": int(straddle_ok and reached_saddle
                                and not collapsed),
        "first_straddle": first, "collapsed": int(collapsed),
        "track_mean": float(track.mean()), "track_p95":
            float(np.percentile(track, 95)),
        "shape_rms_max": rms_max, "effort": effort,
        "final_x": float(hist[-1, 0]), "final_y": float(hist[-1, 1]),
    }
