"""
ocean_backward_ftle.py  (Draft_14, Section V-B; numbers only, Fig. 9 unchanged)

Backward FTLE marks attracting structures, which the forward field of Fig. 9
cannot show. Computes a 12-h backward FTLE field anchored at 14 h (frames 14
back to 2, velocities negated, same pipeline and grid as Fig. 9's forward
field) and measures how close the Fig. 9 rides pass to its top-5% band over
7 to 14 h, when the s1 tangent followed e2 (an attracting OECS), and to the
forward field's top-5% band over the same window.

Run from VF_Robot: venv/bin/python3 experiments/draft14/ocean_backward_ftle.py
Writes experiments/draft14/outputs/ocean_backward_ftle.txt and .npz (the field).
"""
import glob
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
sys.path.insert(0, EXP)

import main_ocean_hfr_2km_shared_start as S
from _ftle_common import (load_frame, load_coastline_polygons, build_land_mask,
                          fill_ocean_only, crop_stack, ftle_linear_time, compute_ftle_field)

ANCHOR_H, SPAN_H = 14, 12
KLA = 111.32
KLO = 111.32 * np.cos(np.radians(34.2))


def backward_ftle():
    files = sorted(glob.glob(f"{S.DATA_DIR}/{S.FRAME_GLOB}"))
    sel = files[ANCHOR_H - SPAN_H:ANCHOR_H + 1][::-1]        # 14, 13, ..., 2
    raw = [load_frame(fp, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX) for fp in sel]
    lat0, lon0 = raw[0][0], raw[0][1]
    polys = load_coastline_polygons(S.COAST_SHP, S.LON_MIN - .5, S.LON_MAX + .5,
                                    S.LAT_MIN - .5, S.LAT_MAX + .5)
    land = build_land_mask(lat0, lon0, polys)
    us, vs = [], []
    for _, _, u, v, _ in raw:
        uf, vf = fill_ocean_only(u, v, land)
        us.append(-uf)
        vs.append(-vf)
    lat_c, lon_c, us = crop_stack(lat0, lon0, us, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX)
    _, _, vs = crop_stack(lat0, lon0, vs, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX)
    lat_f, lon_f, f = ftle_linear_time(lat_c, lon_c, us, vs, n_substeps_per_hour=S.SUBSTEPS_HR,
                                       seed_upsample=S.SEED_UPSAMPLE)
    return lat_f, lon_f, f, build_land_mask(lat_f, lon_f, polys)


def band_tree(lat_f, lon_f, f, land, pct=95.0):
    water = ~land & np.isfinite(f)
    thr = np.percentile(f[water], pct)
    ii, jj = np.nonzero(water & (f >= thr))
    return cKDTree(np.column_stack([lon_f[jj] * KLO, lat_f[ii] * KLA]))


def main():
    from src.fields.field_types import AnalyticalField
    from src.fields.environments.Ocean_HFR import ocean_hfr_socal_timevarying
    from src.robot.pentagon_cluster import PentagonCluster
    from src.control.pentagon_primitives import separatrix_logic_c_step, oecs_separatrix_step
    out = []
    lat_b, lon_b, f_b, land_b = backward_ftle()
    np.savez_compressed(os.path.join(HERE, "outputs", "ocean_backward_ftle.npz"),
                        lat=lat_b, lon=lon_b, ftle=f_b, land=land_b)
    lat_f, lon_f, f_f, land_f, _, _ = compute_ftle_field(
        S.DATA_DIR, S.FRAME_GLOB, S.COAST_SHP, S.LAT_MIN, S.LAT_MAX, S.LON_MIN, S.LON_MAX,
        ftle_hours=S.FTLE_HOURS, substeps_hr=S.SUBSTEPS_HR, seed_upsample=S.SEED_UPSAMPLE,
        verbose=False)
    tb, tf = band_tree(lat_b, lon_b, f_b, land_b), band_tree(lat_f, lon_f, f_f, land_f)

    field = AnalyticalField(ocean_hfr_socal_timevarying, config_name=S.FIELD_CONFIG_NAME)
    field.config["isotropic_map"] = S.ISOTROPIC_MAP
    cl = PentagonCluster(S.FORMATION_CONFIG, field, momentum_alpha=S.MOMENTUM_ALPHA,
                         stiction_threshold=S.STICTION_THRESHOLD)
    prims = {"D": lambda c: tuple(g * S.CONTROL_GAIN for g in separatrix_logic_c_step(c, v_max=S.V_MAX)),
             "s1": lambda c: tuple(g * S.CONTROL_GAIN for g in
                                   oecs_separatrix_step(c, v_max=S.V_MAX, g_perp=S.G_PERP))}
    out.append(f"Backward FTLE: {SPAN_H} h anchored at {ANCHOR_H} h; forward: 24 h anchored at 0 h; "
               "bands are the top 5% of water values.")
    for name, prim in prims.items():
        path = S.run_traj(field, cl, prim, *S.START)
        for h0, h1 in ((0, 7), (7, 14), (14, 21)):
            p = path[h0 * 6:h1 * 6 + 1]
            xy = np.column_stack([p[:, 1] * KLO, p[:, 0] * KLA])
            db, _ = tb.query(xy)
            df, _ = tf.query(xy)
            out.append(f"  {name:2s} {h0:2d}-{h1:2d} h: mean distance to backward band {db.mean():.2f} km, "
                       f"to forward band {df.mean():.2f} km")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "outputs", "ocean_backward_ftle.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
