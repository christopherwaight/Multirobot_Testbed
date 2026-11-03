"""
Triangle success rule, scored on simulation (Coulomb model) and on the three-robot hardware logs.

ATTRACTION. A trial succeeds if the true critical point lies inside the triangle of the three robots at
the last sample, and the centroid never left the map. Also reported: the fraction of the last 20 % of the
trial spent inside, and the first time after which the point stays inside.

ORBIT. At each sample after the 10 s transient the triangle (an area) either touches the commanded circle
of radius r_d about the critical point or it does not. The triangle is convex, so it touches the circle
exactly when d_min <= r_d <= d_max, with d_min the distance from the point to the nearest point of the
triangle (0 if inside) and d_max the distance to the farthest vertex. Reported per run: the fraction of
scored samples that touch, and success = touches on every scored sample and the centroid never left the map.

Simulation reuses sim.py's seeds and settings (attraction 1000 trials, 10 s; orbit 100 trials per field and
r_d, 60 s), so trials match the existing CSVs. The cm-threshold and radial-bias columns are not recomputed
here; they stay in results/*.csv.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/triangle_rescore.py            # full
    venv/bin/python3 experiments/revision_metrics/triangle_rescore.py --quick    # 50 attraction / 10 orbit
    venv/bin/python3 experiments/revision_metrics/triangle_rescore.py --skip-sim # hardware only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hardware as hw  # noqa: E402
import metrics as mt  # noqa: E402
import sim  # noqa: E402

RES = os.path.join(HERE, "results")


def inside(tri, pt):
    """tri (..., 3, 2), pt (..., 2) or (2,). True where pt is in the closed triangle."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]

    def cross(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])

    s1, s2, s3 = cross(a, b, pt), cross(b, c, pt), cross(c, a, pt)
    neg = (s1 < 0) | (s2 < 0) | (s3 < 0)
    pos = (s1 > 0) | (s2 > 0) | (s3 > 0)
    return ~(neg & pos)


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def dist_range(tri, pt):
    """Nearest and farthest distance from pt to the triangle region."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    dmax = np.max(np.linalg.norm(tri - pt[..., None, :], axis=-1), axis=-1)
    dmin = np.minimum(np.minimum(seg_dist(pt, a, b), seg_dist(pt, b, c)), seg_dist(pt, c, a))
    return np.where(inside(tri, pt), 0.0, dmin), dmax


def attraction_row(pos, target, dt=mt.DT):
    """pos (T+1, 3, 2)."""
    tgt = np.asarray(target, float)
    ins = inside(pos, np.broadcast_to(tgt, (len(pos), 2)))
    last = ins[int(0.8 * len(ins)):]
    stay = np.where(~ins)[0]
    t_in = 0.0 if len(stay) == 0 else ((stay[-1] + 1) * dt if stay[-1] + 1 < len(ins) else np.nan)
    c = pos.mean(1)
    left = bool(mt.left_workspace(c))
    return dict(tri_final=bool(ins[-1]) and not left, tri_frac_last20=float(last.mean()),
                tri_settle_s=t_in, left_map=left)


def orbit_row(pos, rd, target, transient=mt.TRANSIENT_STEPS):
    tgt = np.asarray(target, float)
    tri = pos[transient + 1:]
    dmin, dmax = dist_range(tri, np.broadcast_to(tgt, (len(tri), 2)))
    touch = (dmin <= rd) & (rd <= dmax)
    left = bool(mt.left_workspace(pos.mean(1)))
    return dict(touch_frac=float(touch.mean()), tri_orbit_success=bool(touch.all()) and not left,
                touch_ge95=bool(touch.mean() >= 0.95) and not left, left_map=left,
                dmin_mean=float(dmin.mean()), dmax_mean=float(dmax.mean()))


def configs():
    """(label, make_field kwargs, side, group). Group 'base' is the 8 table fields at the paper side."""
    out = [(f, dict(name=f), 0.33, "base") for f in sim.FIELDS]
    out.append(("gyre", dict(name="gyre"), 0.33, "nonlinear"))
    for fam in ("potential", "cellular"):
        for e in (-1.0, -0.5, 0.0, 0.5, 1.0):
            out.append((f"curved_saddle_{fam}_{e:+g}", dict(name="curved_saddle", eps=e, family=fam), 0.33, "nonlinear"))
    for side in (0.20, 0.45):
        out.append((f"gyre_side{side:.2f}", dict(name="gyre"), side, "ablation"))
        for fam in ("potential", "cellular"):
            for e in (-1.0, 0.0, 1.0):
                out.append((f"curved_saddle_{fam}_{e:+g}_side{side:.2f}",
                            dict(name="curved_saddle", eps=e, family=fam), side, "ablation"))
    return out


def reached(F, c_end):
    """Which listed critical point is nearest to the final centroid, and its index sign."""
    d = np.linalg.norm(F.crit[:, :2] - c_end, axis=1)
    j = int(np.argmin(d))
    return j == 0, float(F.crit[j, 2]), float(d[j])


def sim_all(n_att, n_orb):
    import vecsim as vs
    att, orb = [], []
    for label, kw, side, group in configs():
        t0 = time.time()
        F = vs.make_field(**kw)
        truth = int(mt.classify(F.J0))
        P = vs.params(n_att, side=side)
        out, _ = vs.simulate(F, "attract", "fit", P, 10.0, seed=0, stiction_model=sim.MODEL, record=True)
        rec = out["rec"]
        for i in range(n_att):
            r = attraction_row(rec["pos"][i], F.target)
            r.update(field=label, group=group, side=side, trial=i, escaped=bool(out["escaped"][i]))
            r["tri_final"] = r["tri_final"] and not r["escaped"]
            r["reached_target"], r["reached_index"], r["reached_dist"] = reached(F, rec["c"][i][-1])
            r.update(mt.classification_scores(rec["J"][i], truth, start=int(0.8 * rec["J"].shape[1])))
            r["true_type"] = truth
            att.append(r)
        for rd in sim.PAPER_RDS:
            P = vs.params(n_orb, rd=rd, side=side)
            out, _ = vs.simulate(F, "orbit", "fit", P, 60.0, seed=0, stiction_model=sim.MODEL, record=True)
            rec = out["rec"]
            for i in range(n_orb):
                r = orbit_row(rec["pos"][i], rd, F.target)
                r.update(field=label, group=group, side=side, trial=i, rd=rd, escaped=bool(out["escaped"][i]))
                r["tri_orbit_success"] = r["tri_orbit_success"] and not r["escaped"]
                r["touch_ge95"] = r["touch_ge95"] and not r["escaped"]
                r.update(mt.classification_scores(rec["J"][i], truth, start=mt.TRANSIENT_STEPS))
                r["true_type"] = truth
                orb.append(r)
        print(f"  sim {label:42s} {time.time() - t0:6.1f} s", flush=True)
    return pd.DataFrame(att), pd.DataFrame(orb)


def hw_all():
    att, orb = [], []
    for field in ("saddle", "vortex"):
        for r in hw.attraction_runs(field):
            if r["n_robots"] != 3:
                continue
            row = attraction_row(r["robots"][:, :3], (0.0, 0.0))
            row.update(field=field, run=r["run"], file=r["file"])
            att.append(row)
    for r in hw.orbit_runs():
        if r["n_robots"] != 3:
            continue
        row = orbit_row(r["robots"][:, :3], r["rd"], (0.0, 0.0))
        row.update(field=r["field"], series=r["series"], name=r["name"], rd=r["rd"])
        orb.append(row)
    return pd.DataFrame(att), pd.DataFrame(orb)


def summarize(att, orb, tag):
    out = {}
    if len(att):
        out["attraction"] = {f: dict(n=int(len(d)), tri_success_pct=float(100 * d.tri_final.mean()),
                                     tri_frac_last20_mean=float(d.tri_frac_last20.mean()),
                                     tri_settle_median_s=float(np.nanmedian(d.tri_settle_s)))
                             for f, d in att.groupby("field")}
    if len(orb):
        out["orbit"] = {f"{f} r_d={rd:.2f}": dict(n=int(len(d)), all_touch_pct=float(100 * d.tri_orbit_success.mean()),
                                                  touch95_pct=float(100 * d.touch_ge95.mean()),
                                                  touch_frac_mean=float(d.touch_frac.mean()))
                        for (f, rd), d in orb.groupby(["field", "rd"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    summ = {}
    h_att, h_orb = hw_all()
    h_att.to_csv(os.path.join(RES, "triangle_hw_attraction.csv"), index=False)
    h_orb.to_csv(os.path.join(RES, "triangle_hw_orbit.csv"), index=False)
    summ["hardware"] = summarize(h_att, h_orb, "hw")
    print("hardware done", flush=True)
    if not a.skip_sim:
        n_att, n_orb = (50, 10) if a.quick else (1000, 100)
        s_att, s_orb = sim_all(n_att, n_orb)
        s_att.to_csv(os.path.join(RES, "triangle_sim_attraction.csv"), index=False)
        s_orb.to_csv(os.path.join(RES, "triangle_sim_orbit.csv"), index=False)
        summ["simulation"] = summarize(s_att, s_orb, "sim")
        summ["simulation"]["n"] = dict(attraction=n_att, orbit=n_orb, model=sim.MODEL)
    with open(os.path.join(RES, "triangle_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
