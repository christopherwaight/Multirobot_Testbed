"""Ocean unit chain + decomposition of the s1 tracker's commanded speed.

Re-runs the paper's 168-step ocean trial from _ocean_run_common at the canonical
operating point, logging the tangential and transverse channels of
oecs_separatrix_step separately, plus the realized centroid speed.
"""
import os
import sys

import numpy as np

EXP = ("/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations"
       "/Vector_Fields/VF_Robot/experiments")
sys.path.insert(0, os.path.dirname(EXP))
sys.path.insert(0, EXP)

from _ocean_run_common import (build_trial, LEGACY_START, SIM_STEPS, TIME_WARP,
                               V_MAX, CONTROL_GAIN, KM_PER_DEG_LAT, KM_PER_DEG_LON)
from _coords_common import latlon_to_world, world_to_latlon

# ---- unit chain, from the constants themselves ----------------------------
DT = 0.1                     # PentagonCluster default timestep, world time units
dt_s = DT * TIME_WARP        # seconds of field time per step
tu_s = TIME_WARP             # seconds per world time unit
L_w_km = 0.6 * KM_PER_DEG_LAT / 1.3   # world span [-0.65,0.65] <-> +/-0.300 deg lat
vel_unit = L_w_km * 1000.0 / tu_s     # m/s per (world unit / world time unit)

print("--- unit chain ---")
print(f"L_w              = {L_w_km:.3f} km per world unit")
print(f"dt               = {DT} time units = {dt_s:.0f} s of field time per step")
print(f"1 time unit      = {tu_s:.0f} s")
print(f"velocity unit    = {vel_unit:.4f} m/s per (world unit / time unit)")
print(f"c_max            = {V_MAX} -> {V_MAX*vel_unit:.4f} m/s")
print(f"k c_max          = {CONTROL_GAIN*V_MAX*vel_unit:.4f} m/s")
print(f"k c_max tanh(1)  = {CONTROL_GAIN*V_MAX*np.tanh(1)*vel_unit:.4f} m/s")
print(f"sqrt2 k c_max    = {np.sqrt(2)*CONTROL_GAIN*V_MAX*vel_unit:.4f} m/s"
      f"  ({np.sqrt(2)*CONTROL_GAIN*V_MAX*vel_unit*1.94384:.2f} knots)")
print(f"strain-rate unit = 1/{tu_s:.0f} s = {1.0/tu_s:.3e} 1/s")

# ---- run, logging the two channels ---------------------------------------
import src.control.pentagon_primitives as pp

field, cluster, prim_d, prim_s1 = build_trial()

log = {"tan": [], "perp": [], "cmd": []}
_orig = pp.oecs_separatrix_step


def traced(c, **kw):
    vx, vy = _orig(c, **kw)
    # recompute the two channels from the same fit the primitive just used
    u_arr, v_arr = pp._sample_vector_at_robots(c)
    rel = pp._get_relative_positions(c)
    tu_, tv_ = pp._fit_vector_quadratic(rel, u_arr, v_arr)
    s1, g, e1, e2, r = pp._strain_quantities(tu_, tv_)
    t_hat = c._oecs_prev_tangent
    gp = g - float(g @ t_hat) * t_hat
    gpn = float(np.linalg.norm(gp))
    v_max, g_perp = kw.get("v_max", V_MAX), kw.get("g_perp", 1.0)
    perp = v_max * abs(np.tanh(-g_perp * gpn / v_max))
    log["tan"].append(v_max * np.tanh(1.0))
    log["perp"].append(perp)
    log["cmd"].append(float(np.hypot(vx, vy)))
    return vx, vy


def prim_s1_traced(c):
    vx, vy = traced(c, v_max=V_MAX, g_perp=1.0, s_trim=0.3, r_band=0.05,
                    g_capture=0.15, s_capture=None)
    return vx * CONTROL_GAIN, vy * CONTROL_GAIN


lat0, lon0 = LEGACY_START
sx, sy = latlon_to_world(lat0, lon0, field.config)
cluster.reset(sx, sy)
field.reset_clock()
for _ in range(SIM_STEPS):
    cluster.move(prim_s1_traced)
    field.step(cluster.timestep * TIME_WARP)
center = cluster.get_center_history()
path = np.array([world_to_latlon(p[0], p[1], field.config) for p in center])

xy_km = np.column_stack([path[:, 1] * KM_PER_DEG_LON, path[:, 0] * KM_PER_DEG_LAT])
speed = np.linalg.norm(np.diff(xy_km, axis=0), axis=1) * 1000.0 / dt_s

tan = np.array(log["tan"]) * CONTROL_GAIN * vel_unit
perp = np.array(log["perp"]) * CONTROL_GAIN * vel_unit
cmd = np.array(log["cmd"]) * CONTROL_GAIN * vel_unit

print("\n--- s1 tracker, 168 steps ---")
print(f"tangential channel   mean {tan.mean():.4f}  min {tan.min():.4f}  max {tan.max():.4f} m/s")
print(f"transverse channel   mean {perp.mean():.4f}  min {perp.min():.4f}  max {perp.max():.4f} m/s")
print(f"commanded |delta|    mean {cmd.mean():.4f}  min {cmd.min():.4f}  max {cmd.max():.4f} m/s")
print(f"realized speed       mean {speed.mean():.4f}  min {speed.min():.4f}  max {speed.max():.4f} m/s")
print(f"fraction of the sqrt2 k c_max cap: mean "
      f"{speed.mean()/(np.sqrt(2)*CONTROL_GAIN*V_MAX*vel_unit)*100:.1f}%  max "
      f"{speed.max()/(np.sqrt(2)*CONTROL_GAIN*V_MAX*vel_unit)*100:.1f}%")
print(f"transverse saturated (>99% of k c_max) on "
      f"{100*np.mean(perp > 0.99*CONTROL_GAIN*V_MAX*vel_unit):.1f}% of steps")
