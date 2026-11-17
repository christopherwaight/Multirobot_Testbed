"""
Rebuild the vortex reconstructed field (vortex_nn) from the 2025 and 2026 vortex readings, the same
way as saddle_predictors_2026 (whose functions this reuses), with two differences:
  - the critical point is the zero of the direction field (plane fit over a formation-sized ring),
    because the 2025 vortex saturation centre and direction centre differ by about 2 cm;
  - geometry scale is 1. The data cannot fix it: the 2025 and 2026 saturation slopes differ by 4.5%
    (0.657 vs 0.628), but hue cannot see scale on a vortex (its directions do not change with radius),
    and registering the print pattern (hue minus the ideal vortex) is flat from 0.90 to 1.10. The same
    robots read the saddle mat at 0.997, both years put the centre in the same place, and it is the
    same mat, so the 4.5% is taken as a saturation calibration difference: the 2025 readings used for
    the corners are mapped onto the 2026 saturation line, S -> a26 + (S - a25) b26 / b25.

Run from VF_Robot:  venv/bin/python3 vortex_predictors_2026/nn_trainer_vortex_2026.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
VF = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VF, "saddle_predictors_2026"))
import nn_trainer_saddle_2026 as S  # noqa: E402  (shared helpers)

ROOT = S.ROOT
CSV25 = os.path.join(VF, "vortex_predictors", "all_vortex_data_possible.csv")   # the old net's training file
RUNS26 = os.path.join(ROOT, "New_Hardware_Data", "data_hardware_campaign_11_11_2026", "Vortex")
FLIP25 = np.array([-1, -1])            # the old trainer's frame: (-x, -y)


def load25():
    d = pd.read_csv(CSV25)
    P = np.vstack([d.iloc[:, [5 * i, 5 * i + 1]].values * FLIP25 for i in range(3)])
    H = np.concatenate([d.iloc[:, 5 * i + 3].values for i in range(3)])
    Sa = np.concatenate([d.iloc[:, 5 * i + 4].values for i in range(3)])
    row = np.concatenate([np.arange(len(d))] * 3)
    ok = np.isfinite(P).all(1) & np.isfinite(H) & np.isfinite(Sa)
    return P[ok], H[ok], Sa[ok], row[ok], len(d)


def load26():
    P, H, Sa, run, robot = [], [], [], [], []
    files = sorted(glob.glob(os.path.join(RUNS26, "**", "*.mat"), recursive=True))
    files = [f for f in files if "off_the_mat" not in f and not ("three_robot" in f and "4robots" in f)]
    for j, f in enumerate(files):
        M = sio.loadmat(f, squeeze_me=True, struct_as_record=False,
                        variable_names=["sensor_hue", "sensor_sat", "cmd_robots_selected", "run_config",
                                        "pose_r1", "pose_r2", "pose_r3", "pose_r4"])
        names = dict((str(a), str(b)) for a, b in M["run_config"].robots)
        for r in range(M["cmd_robots_selected"].shape[1] // 3):
            p = M[f"pose_r{r + 1}"][:, :2]
            h = np.atleast_2d(M["sensor_hue"])[:, r]
            s = np.atleast_2d(M["sensor_sat"])[:, r]
            ok = np.isfinite(p).all(1) & np.isfinite(h) & np.isfinite(s)
            P.append(p[ok]); H.append(h[ok]); Sa.append(s[ok])
            run += [j] * ok.sum(); robot += [names[f"r{r + 1}"]] * ok.sum()
    return np.vstack(P), np.concatenate(H), np.concatenate(Sa), np.array(run), np.array(robot)


def decode(H, Sa):
    ang = 2 * np.pi * H + np.pi / 2
    m = (Sa - 0.3) / 0.7
    return np.stack([m * np.cos(ang), m * np.sin(ang)], 1)


def direction_centre(P, F, c0, ring=(0.08, 0.35), iters=6):
    """Zero of a linear fit of the decoded field over a ring around the current centre."""
    c = np.asarray(c0, float)
    for _ in range(iters):
        r = np.hypot(*(P - c).T)
        b = (r > ring[0]) & (r < ring[1])
        coef = np.linalg.lstsq(np.c_[P[b], np.ones(b.sum())], F[b], rcond=None)[0]
        J, h0 = coef[:2].T, coef[2]
        c = -np.linalg.solve(J, h0)
    return c, J


def register(P26, H26, c26, P25, H25, c25, free_scale):
    """Rotation, mirror, translation (and scale) taking 2026 positions into the 2025 frame,
    matching each 2026 hue to the 2025 hue map (2 cm cells) at the same mat point."""
    lo, w = P25.min(0) - 0.05, 0.02
    ij = np.floor((P25 - lo) / w).astype(int)
    shape = tuple(ij.max(0) + 2)
    cnt = np.zeros(shape); sn = np.zeros(shape); cs = np.zeros(shape)
    np.add.at(cnt, tuple(ij.T), 1)
    np.add.at(sn, tuple(ij.T), np.sin(2 * np.pi * H25))
    np.add.at(cs, tuple(ij.T), np.cos(2 * np.pi * H25))
    ref = np.where(cnt >= 5, np.arctan2(sn, cs), np.nan)
    rng = np.random.default_rng(0)
    i = rng.choice(np.where(np.hypot(*(P26 - c26).T) < 0.65)[0], 10000, replace=False)

    def place(x, m, P):
        k = x[3] if len(x) > 3 else 1.0
        Q = (P - c26) * k
        if m:
            Q = Q * [1, -1]
        c, s = np.cos(x[0]), np.sin(x[0])
        return Q @ np.array([[c, s], [-s, c]]) + x[1:3]

    def loss(x, m):
        q = np.floor((place(x, m, P26[i]) - lo) / w).astype(int)
        inside = ((q >= 0) & (q < shape)).all(1)
        r = np.full(len(i), np.nan)
        r[inside] = ref[tuple(q[inside].T)]
        ok = np.isfinite(r)
        if not ok.any():
            return 1.0
        return np.minimum(1 - np.cos(2 * np.pi * H26[i][ok] - r[ok]), 0.5).mean() + 0.5 * (1 - ok.mean())

    _, m, th = min(((loss([th, *c25], m), m, th) for m in (False, True)
                    for th in np.radians(np.arange(0, 360, 5))), key=lambda t: t[0])
    x0 = [th, *c25] + ([1.0] if free_scale else [])
    x = minimize(loss, x0, args=(m,), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-9, maxiter=3000)).x
    profile = {}
    for k in (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06):
        xk = minimize(lambda y: loss([*y, k], m), x[:3], method="Nelder-Mead",
                      options=dict(xatol=1e-5, fatol=1e-9, maxiter=800)).x
        profile[k] = loss([*xk, k], m)
    return x, m, place, profile


def main():
    out = {}
    P25, H25, S25, row25, n25 = load25()
    P26, H26, S26, run26, robot26 = load26()
    tr = np.isin(robot26, list(S.TRAINED))

    # centres: direction field zero (start from the saturation centre)
    f25 = S.sat_fit(P25, S25, (0.0, 0.0))
    f26 = S.sat_fit(P26[tr], S26[tr], (0.84, 0.86))
    c25, _ = direction_centre(P25, decode(H25, S25), f25[:2])
    c26, _ = direction_centre(P26[tr], decode(H26[tr], S26[tr]), f26[:2])
    print(f"2025 (old trainer frame): saturation centre ({f25[0]:+.4f}, {f25[1]:+.4f}), "
          f"direction centre ({c25[0]:+.4f}, {c25[1]:+.4f})")
    print(f"2026 (Motive): saturation centre ({f26[0]:+.4f}, {f26[1]:+.4f}), direction centre ({c26[0]:+.4f}, {c26[1]:+.4f})")

    # geometry at scale 1; 2025 saturation mapped onto the 2026 calibration line
    k_sat = f26[3] / f25[3]
    x, m, place, prof = register(P26, H26, c26, P25, H25, c25, free_scale=False)
    k_hue = 1.0
    S25 = f26[2] + (S25 - f25[2]) * k_sat
    print(f"saturation slopes 2025 {f25[3]:.4f}, 2026 {f26[3]:.4f} (ratio {k_sat:.4f}): 2025 saturation mapped "
          f"onto the 2026 line; geometry scale 1")
    out["centres"] = dict(sat_2025=f25[:2].tolist(), dir_2025=c25.tolist(), sat_2026=f26[:2].tolist(),
                          dir_2026=c26.tolist())
    out["scale"] = dict(saturation=k_sat, hue_registration=k_hue, profile=prof,
                        sat_slopes=dict(y2025=f25[3], y2026=f26[3]))
    out["align_2026_to_2025"] = dict(rotation_deg=float(np.degrees(x[0]) % 360), mirror=bool(m),
                                     translation=x[1:3].tolist(), scale=k_hue, centre_2026_motive=c26.tolist())
    print(f"2026 -> 2025 frame: rotation {np.degrees(x[0]) % 360:.2f} deg, mirror {m}, "
          f"2026 direction centre lands at ({x[1]:+.4f}, {x[2]:+.4f}), scale {k_hue:.4f}")
    Q26 = place(x, m, P26)

    # centre the pooled data on the direction centre of the pooled training readings
    test25 = row25 >= int(0.8 * n25)
    test26 = tr & (run26 % 5 == 0)
    trn26 = tr & (run26 % 5 != 0)
    red26 = robot26 == "redwood"
    pool_P = np.vstack([P25[~test25], Q26[trn26]])
    pool_F = decode(np.concatenate([H25[~test25], H26[trn26]]), np.concatenate([S25[~test25], S26[trn26]]))
    c, _ = direction_centre(pool_P, pool_F, c25)
    out["critical_point_in_old_frame"] = c.tolist()
    print(f"critical point (pooled direction centre) in the old vortex_nn frame: ({c[0]:+.4f}, {c[1]:+.4f})")
    P25c, Q26c = P25 - c, Q26 - c

    X, Yh, Ys = S.grid(Q26c[trn26], H26[trn26], S26[trn26], P25c[~test25], H25[~test25], S25[~test25])
    hm, sm = S.train(X, Yh, S.HUE_ARCH), S.train(X, Ys, S.SAT_ARCH)
    Ptr = np.vstack([P25c[~test25], Q26c[trn26]])
    Htr = np.concatenate([H25[~test25], H26[trn26]]); Str = np.concatenate([S25[~test25], S26[trn26]])

    sys.path.insert(0, VF)
    from src.fields.field_types import NNField
    old = NNField(os.path.join(VF, "vortex_predictors"))
    rep = {}
    for name, Pt, Ht, St in (("2026 held-out runs", Q26c[test26], H26[test26], S26[test26]),
                             ("2026 redwood (unseen robot)", Q26c[red26], H26[red26], S26[red26]),
                             ("2025 held-out last 20%", P25c[test25], H25[test25], S25[test25])):
        hn, sn = S.predict(hm, sm, Pt)
        ho, so = S.predict(old.hue_model, old.sat_model, Pt + c)
        fh, fs, keep = S.floor(Pt, Ht, St, Ptr, Htr, Str)
        en, eo = S.dang(hn, Ht), S.dang(ho, Ht)
        rad = np.hypot(*Pt.T)
        print(f"\n{name}: {len(Pt):,} readings. Median |direction error| deg (bias); saturation |error|")
        rows = []
        for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .65), (.65, 1.5)):
            b = (rad >= lo) & (rad < hi)
            bk = b[keep]
            if b.sum() < 50:
                continue
            r = dict(r=f"{lo:.2f}-{hi:.2f}", n=int(b.sum()),
                     new=float(np.median(np.abs(en[b]))), new_bias=float(np.mean(en[b])),
                     old=float(np.median(np.abs(eo[b]))), old_bias=float(np.mean(eo[b])),
                     floor=float(np.median(np.abs(fh[bk]))) if bk.sum() > 30 else float("nan"),
                     sat_new=float(np.median(np.abs(sn[b] - St[b]))), sat_old=float(np.median(np.abs(so[b] - St[b]))))
            rows.append(r)
            print(f"  r {r['r']} m  n {r['n']:6d}   new {r['new']:5.2f} ({r['new_bias']:+5.2f})   "
                  f"old {r['old']:5.2f} ({r['old_bias']:+5.2f})   floor {r['floor']:5.2f}   "
                  f"sat new {r['sat_new']:.3f} old {r['sat_old']:.3f}")
        print(f"  all          new {np.median(np.abs(en)):5.2f}   old {np.median(np.abs(eo)):5.2f}   "
              f"floor {np.median(np.abs(fh)):5.2f}")
        rep[name] = dict(bins=rows, new=float(np.median(np.abs(en))), old=float(np.median(np.abs(eo))),
                         floor=float(np.median(np.abs(fh))))
    out["validation"] = rep

    # what the plane fit sees near the centre, new and old, plus the data's own answer
    g = np.stack(np.meshgrid(np.linspace(-0.35, 0.35, 71), np.linspace(-0.35, 0.35, 71)), -1).reshape(-1, 2)
    g = g[(np.hypot(*g.T) > 0.08) & (np.hypot(*g.T) < 0.35)]
    A = np.c_[g, np.ones(len(g))]
    for tag, hmod, smod, shift in (("new", hm, sm, 0.0), ("old", old.hue_model, old.sat_model, c)):
        h, s = S.predict(hmod, smod, g + shift)
        coef = np.linalg.lstsq(A, decode(h, s), rcond=None)[0]
        J, h0 = coef[:2].T, coef[2]
        p = -np.linalg.solve(J, h0)
        s_c = S.predict(hmod, smod, np.array([[0.0, 0.0]]) + shift)[1][0]
        out[f"plane_fit_{tag}"] = dict(p_star=p.tolist(), det=float(np.linalg.det(J)), trace=float(np.trace(J)),
                                       sat_at_centre=float(s_c))
        print(f"plane fit, {tag} net: p* = ({p[0]:+.4f}, {p[1]:+.4f}) m, det J {np.linalg.det(J):+.3f}, "
              f"trace {np.trace(J):+.3f}, saturation at the centre {s_c:.3f}")
    held = np.vstack([Q26c[test26]])
    pd_, Jd = direction_centre(held, decode(H26[test26], S26[test26]), (0.0, 0.0))
    r = np.hypot(*Q26c[test26].T)
    print(f"held-out 2026 readings themselves: p* = ({pd_[0]:+.4f}, {pd_[1]:+.4f}) m, det J {np.linalg.det(Jd):+.3f}, "
          f"median saturation within 3 cm {np.median(S26[test26][r < 0.03]):.3f}")
    out["held_out_data_plane_fit"] = dict(p_star=pd_.tolist(), det=float(np.linalg.det(Jd)))

    torch.save(hm.state_dict(), os.path.join(HERE, "hue_model.pth"))
    torch.save(sm.state_dict(), os.path.join(HERE, "sat_model.pth"))
    with open(os.path.join(HERE, "model_info.pkl"), "wb") as fh:
        pickle.dump({"hue_architecture": S.HUE_ARCH, "sat_architecture": S.SAT_ARCH}, fh)
    with open(os.path.join(HERE, "scaling_info.pkl"), "wb") as fh:
        pickle.dump({"input_range": (-S.EXTENT, S.EXTENT), "hue_representation": "circular_sin_cos",
                     "sat_scaling": "[0,1] -> [-1,1]", "frame": "old vortex_nn frame shifted so the "
                     "direction-field critical point is at (0, 0)"}, fh)
    with open(os.path.join(HERE, "build_report.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("saved to vortex_predictors_2026/")


if __name__ == "__main__":
    main()
