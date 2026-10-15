"""
ocean_section_v_metrics.py

PAPER TRACEABILITY
  Paper:  Paper_Writing/Separatrix_and_OW_Paper/Draft_13.tex, Section V-B
  Makes:  review numbers only (stdout, tee to ocean_section_v_metrics.log).
          Nothing is written into the paper or figures/.

Reruns the Fig. 9 trial exactly as main_ocean_hfr_2km_progression.py does
(same start, gains, field config, landfall rule) and measures, per tracker:
  1. Why the two trackers agree. From the cluster's own fit at each step:
     |omega| / (2 r) and |mu| / r, with mu = (u_x + v_y)/2 and r the deviatoric
     strain. det J = (mu^2 - r^2) + omega^2/4 and s1 s2 = mu^2 - r^2, so when
     both ratios are small D is close to -s1^2. Also the sign of D, and the
     separation between the two centroid paths.
  2. Which eigenvector the s1 tracker rode (e1 compression / e2 stretching),
     and the transverse slopes of s1 and s2 across its path (a trench of s1
     is a ridge of s2 only where mu is uniform across it).
  3. Path length, mean ground speed, and the measured current speed, plus the
     D tracker's mode history.
  4. Mean distance to the 95th-percentile ridge of the 24-h forward FTLE, per
     7-h window (the Fig. 9 panels) and overall (the paper's 1.6 / 2.1 km).

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_section_v_metrics.py | tee experiments/ocean_section_v_metrics.log
"""
import os
import sys
import io
import contextlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial import cKDTree

import main_ocean_hfr_2km_shared_start as S
from src.robot.pentagon_cluster import PentagonCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
from src.control.pentagon_primitives import (
    separatrix_logic_c_step, oecs_separatrix_step,
    _sample_vector_at_robots, _get_relative_positions, _fit_vector_quadratic,
    _strain_quantities,
)
from _ftle_common import compute_ftle_field

KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * float(np.cos(np.radians(34.2)))
STEP_H = 1.0 / 6.0
WINDOWS = [(0, 7), (7, 14), (14, 21), (21, 28.01)]


def to_km(latlon):
    latlon = np.asarray(latlon)
    return np.column_stack([latlon[:, 1] * KM_PER_DEG_LON, latlon[:, 0] * KM_PER_DEG_LAT])


def fit_record(c):
    """Strain, vorticity, divergence and gradients from the cluster's own fit."""
    u, v = _sample_vector_at_robots(c)
    rel = _get_relative_positions(c)
    tu, tv = _fit_vector_quadratic(rel, u, v)
    a, b = tu, tv                     # a[0..5] = a1..a6, b likewise
    ux, uy, vx, vy = a[1], a[2], b[1], b[2]
    mu = 0.5 * (ux + vy)
    sn = 0.5 * (ux - vy)
    ss = 0.5 * (uy + vx)
    r = float(np.hypot(sn, ss))
    omega = vx - uy
    grad_mu = 0.5 * np.array([a[4] + b[3], a[3] + b[5]])
    s1, grad_s1, e1, e2, _ = _strain_quantities(tu, tv)
    grad_s2 = 2.0 * grad_mu - grad_s1
    return dict(mu=float(mu), r=r, omega=float(omega), D=float(ux * vy - uy * vx),
                flow=np.array([a[0], b[0]]), grad_s1=np.asarray(grad_s1),
                grad_s2=grad_s2, e1=np.asarray(e1), e2=np.asarray(e2))


def run(field, prim):
    with contextlib.redirect_stdout(io.StringIO()):
        cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                             stiction_threshold=S.STICTION_THRESHOLD)
    rec = []

    def wrapped(c):
        rec.append(fit_record(c))
        return prim(c)

    path = S.run_traj(field, cl, wrapped, *S.START)
    rob = np.array([[S._world_to_latlon(x, y, field.config) for x, y in step]
                    for step in cl.get_robot_history()])
    return path, rob, rec, list(cl.diagnostics)


def main():
    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    lw_km = None

    def prim_d(c):
        vx, vy = separatrix_logic_c_step(c, v_max=S.V_MAX)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN

    def prim_s1(c):
        vx, vy = oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP)
        return vx * S.CONTROL_GAIN, vy * S.CONTROL_GAIN

    runs = {"D": run(field, prim_d), "s1": run(field, prim_s1)}

    with contextlib.redirect_stdout(io.StringIO()):
        lat_f, lon_f, f_val, land_f, _, _ = compute_ftle_field(
            S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
            ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE)
    LON, LAT = np.meshgrid(lon_f, lat_f)
    ok = np.isfinite(f_val) & (~land_f)
    thr = np.percentile(f_val[ok], 95.0)
    sel = ok & (f_val >= thr)
    tree = cKDTree(np.column_stack([LON[sel] * KM_PER_DEG_LON, LAT[sel] * KM_PER_DEG_LAT]))

    def landfall_step(rob):
        i = np.clip(np.searchsorted(lat_f, rob[:, :, 0]), 0, len(lat_f) - 1)
        j = np.clip(np.searchsorted(lon_f, rob[:, :, 1]), 0, len(lon_f) - 1)
        hit = land_f[i, j].any(axis=1)
        return int(np.argmax(hit)) if hit.any() else None

    # world unit -> km, from the map (one world unit spans L_w km)
    p0 = S._world_to_latlon(0.0, 0.0, field.config)
    p1 = S._world_to_latlon(0.1, 0.0, field.config)
    lw_km = 10.0 * abs(p1[1] - p0[1]) * KM_PER_DEG_LON
    print(f"L_w = {lw_km:.2f} km; ride speed k c tanh1 = "
          f"{S.CONTROL_GAIN * S.V_MAX * np.tanh(1) * lw_km * 1000 / S.TIME_WARP:.3f} m/s; "
          f"per-channel cap k c = {S.CONTROL_GAIN * S.V_MAX * lw_km * 1000 / S.TIME_WARP:.3f} m/s")

    trimmed = {}
    for name, (path, rob, rec, diag) in runs.items():
        k = landfall_step(rob)
        n = (k + 1) if k is not None else len(path)
        trimmed[name] = (path[:n], rec[:n], diag[:n], k)
        print(f"\n==== {name} tracker: {len(path)} steps recorded, "
              f"{'landfall at step %d (%.1f h)' % (k, k * STEP_H) if k is not None else 'at sea at 28 h'}")

    for name, (path, rec, diag, k) in trimmed.items():
        xy = to_km(path)
        t = np.arange(len(path)) * STEP_H
        om = np.array([q["omega"] for q in rec]); mu = np.array([q["mu"] for q in rec])
        r = np.array([q["r"] for q in rec]); D = np.array([q["D"] for q in rec])
        wr = np.abs(om) / (2 * r); mr = np.abs(mu) / r
        print(f"\n---- {name}: item 1, fitted kinematics along the path ({len(rec)} steps)")
        for lab, x in (("|omega|/(2r)", wr), ("|mu|/r", mr)):
            q = np.percentile(x, [25, 50, 75, 90])
            print(f"  {lab:13s} median {q[1]:.3f}  IQR {q[0]:.3f}-{q[2]:.3f}  p90 {q[3]:.3f}  "
                  f"frac<0.25 {np.mean(x < 0.25):.2f}  frac<0.5 {np.mean(x < 0.5):.2f}")
        rel = np.abs((om ** 2 / 4 + mu ** 2) / r ** 2)
        print(f"  (omega^2/4 + mu^2)/r^2 = |D + s1^2|... median {np.median(rel):.3f}; "
              f"D<0 on {np.mean(D < 0):.2f} of steps")
        print(f"  r median {np.median(r) / (lw_km * 1000.0) * 3600:.3f} 1/h")
        for (a_, b_) in WINDOWS:
            m = (t >= a_) & (t < b_)
            if m.any():
                print(f"    window {a_:>4.0f}-{min(b_, 28):>2.0f} h: |omega|/2r median {np.median(wr[m]):.3f}, "
                      f"|mu|/r median {np.median(mr[m]):.3f}")

        print(f"---- {name}: item 3, speeds")
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        L = seg.sum()
        # Ocean_HFR returns the current in m/s, so the fitted a1, b1 are m/s.
        flow_ms = np.array([np.linalg.norm(q["flow"]) for q in rec])
        print(f"  path {L:.1f} km in {t[-1]:.1f} h, mean ground speed {L / (t[-1] * 3.6):.3f} m/s; "
              f"measured current at centroid median {np.median(flow_ms):.3f} m/s "
              f"(IQR {np.percentile(flow_ms, 25):.3f}-{np.percentile(flow_ms, 75):.3f})")
        modes = [d.get("mode") for d in diag]
        cnt = Counter(modes)
        print(f"  modes {dict(cnt)}")
        first = {m: next((i for i, x in enumerate(modes) if x == m), None) for m in cnt}
        print("  first step of each mode: " + ", ".join(f"{m} {i} ({i * STEP_H:.1f} h)" for m, i in first.items()))
        for m in cnt:
            idx = np.array([i for i, x in enumerate(modes) if x == m and i < len(seg)])
            if len(idx):
                print(f"    mean ground speed in {m}: {seg[idx].mean() / (STEP_H * 3.6):.3f} m/s")

        if name == "s1":
            print("---- s1: item 2, tangent eigenvector and slopes")
            tid = [d.get("tangent_id") for d in diag if d.get("mode") == "TRACK"]
            c = Counter(tid)
            print(f"  TRACK steps {len(tid)}: {dict(c)}")
            tr_idx = [i for i, d in enumerate(diag) if d.get("mode") == "TRACK"]
            for (a_, b_) in WINDOWS:
                ids = [diag[i].get("tangent_id") for i in tr_idx if a_ <= i * STEP_H < b_]
                if ids:
                    print(f"    window {a_:>4.0f}-{min(b_, 28):>2.0f} h: {dict(Counter(ids))}")
            # transverse slopes across the ride direction
            ratios = []
            for i in tr_idx:
                q = rec[i]
                e = q["e1"] if diag[i].get("tangent_id") == "e1" else q["e2"]
                tp = np.array([-e[1], e[0]])
                g1, g2 = float(tp @ q["grad_s1"]), float(tp @ q["grad_s2"])
                ratios.append((g1, g2))
            ratios = np.array(ratios)
            same = np.mean(np.sign(ratios[:, 0]) == -np.sign(ratios[:, 1]))
            print(f"  across the ride, slope of s2 opposes slope of s1 on {same:.2f} of TRACK steps "
                  f"(a trench of s1 is then a ridge of s2)")
            # how closely grad s1 lines up with the chosen eigenvector (1 = along it, 0.71 = 45 deg)
            al = []
            for i in tr_idx:
                q = rec[i]; g = q["grad_s1"]; gn = np.linalg.norm(g)
                if gn > 0:
                    al.append(max(abs(g @ q["e1"]), abs(g @ q["e2"])) / gn)
            al = np.array(al)
            print(f"  |cos| between grad s1 and chosen eigenvector: median {np.median(al):.3f}, "
                  f"frac > 0.95 {np.mean(al > 0.95):.2f}, frac < 0.8 {np.mean(al < 0.8):.2f}")
            # switch count
            sw = sum(1 for i in range(1, len(tid)) if tid[i] != tid[i - 1])
            print(f"  eigenvector switches during TRACK: {sw}")

        print(f"---- {name}: item 4, distance to FTLE ridge (95th pct, {int(sel.sum())} points)")
        d, _ = tree.query(xy)
        print(f"  overall mean {d.mean():.2f} km, median {np.median(d):.2f}, max {d.max():.2f}")
        for (a_, b_) in WINDOWS:
            m = (t >= a_) & (t < b_)
            if m.any():
                seg_m = np.linalg.norm(np.diff(xy[m], axis=0), axis=1).sum()
                net = np.linalg.norm(xy[m][-1] - xy[m][0])
                print(f"    window {a_:>4.0f}-{min(b_, 28):>2.0f} h: path {seg_m:.1f} km, net {net:.1f} km")
        for (a_, b_) in WINDOWS:
            m = (t >= a_) & (t < b_)
            if m.any():
                print(f"    window {a_:>4.0f}-{min(b_, 28):>2.0f} h: mean {d[m].mean():.2f} km, "
                      f"max {d[m].max():.2f} km, lat {path[m][0, 0]:.3f}->{path[m][-1, 0]:.3f}")

    pD, pS = trimmed["D"][0], trimmed["s1"][0]
    xD, xS = to_km(pD), to_km(pS)
    n = min(len(xD), len(xS))
    same_t = np.linalg.norm(xD[:n] - xS[:n], axis=1)
    near = np.linalg.norm(xD[:, None, :] - xS[None, :, :], axis=2).min(axis=1)
    print(f"\n==== path agreement (D path vs s1 path, first {n} steps)")
    print(f"  same-time separation: mean {same_t.mean():.2f} km, max {same_t.max():.2f} km")
    for (a_, b_) in WINDOWS:
        tt = np.arange(n) * STEP_H
        m = (tt >= a_) & (tt < b_)
        if m.any():
            print(f"    window {a_:>4.0f}-{min(b_, 28):>2.0f} h: same-time sep mean {same_t[m].mean():.2f} km")
    print(f"  nearest-point distance from D path to s1 path: mean {near.mean():.2f} km, max {near.max():.2f} km")


if __name__ == "__main__":
    main()
