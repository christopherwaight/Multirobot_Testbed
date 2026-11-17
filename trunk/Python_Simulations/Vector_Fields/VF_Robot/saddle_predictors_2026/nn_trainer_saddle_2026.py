"""
Rebuild the saddle reconstructed field (saddle_nn) from the 2025 and 2026 saddle readings.

Why (2026-11-16): the old saddle_predictors net was trained on 4,000 points and misses its own
training data by 4 to 5 deg beyond 0.35 m, with a -3 to -10 deg bias (see SIM_RERUN_PLAN).

Steps
  1. Load readings. 2025: saddle_predictors/saddle_data.csv in the old trainer's frame (x, -y).
     2026: every saddle run of data_hardware_campaign_11_11_2026 (logged variables only).
  2. Scale: from saturation, S = a + b d, the 2026-to-2025 length ratio is b26 / b25 (computed, not assumed).
  3. Align 2026 to the 2025 frame: rotation, mirror and translation that best match each 2026 hue
     to the 2025 readings within 2 cm (data to data; no net involved).
  4. Centre the pooled data on the critical point (centre of the saturation fit), so it sits at (0, 0).
  5. Split: hold out every 5th 2026 run, the last 20% of the 2025 rows, and redwood (never trained on).
  6. Grid into 1 cm cells (circular mean hue, mean saturation; readings more than 20 deg from their
     cell's mean dropped), so dense orbit rings and long 2025 dwell do not dominate. A cell takes the
     2026 readings where 2026 measured it and the 2025 readings only where 2026 did not (mainly the
     corners): the 2025 file mixes sessions about 3 deg apart and has blocks of outliers, and the
     paper's hardware is 2026. Then fit the hue (sin, cos) and saturation nets.
  7. Validate on the held-out readings against the noise floor and the old net; save.

Run from VF_Robot:  venv/bin/python3 saddle_predictors_2026/nn_trainer_saddle_2026.py
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
import torch.nn as nn
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
VF = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(VF, "..", "..", "..", ".."))
CSV25 = os.path.join(VF, "saddle_predictors", "saddle_data.csv")
RUNS26 = os.path.join(ROOT, "New_Hardware_Data", "data_hardware_campaign_11_11_2026", "Saddle")
TRAINED = {"celeste", "tidal", "pacific-blue"}
HUE_ARCH = {"hidden_sizes": [32, 32, 16], "activation": "tanh", "output_neurons": 2}
SAT_ARCH = {"hidden_sizes": [32, 16], "activation": "tanh", "output_neurons": 1}
EXTENT = 0.85          # train on cells within this half-width (m) of the critical point
CELL = 0.01            # grid cell (m)


def dang(a, b):
    """Hue difference (turns) as a wrapped angle in degrees."""
    return (((a - b) * 360 + 180) % 360) - 180


def circ_mean(h):
    a = 2 * np.pi * np.asarray(h)
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi)) / (2 * np.pi)


# ---------------------------------------------------------------- 1. load
def load25():
    d = pd.read_csv(CSV25)
    P = np.vstack([d.iloc[:, [5 * i, 5 * i + 1]].values * [1, -1] for i in range(3)])
    H = np.concatenate([d.iloc[:, 5 * i + 3].values for i in range(3)])
    S = np.concatenate([d.iloc[:, 5 * i + 4].values for i in range(3)])
    row = np.concatenate([np.arange(len(d))] * 3)
    ok = np.isfinite(P).all(1) & np.isfinite(H) & np.isfinite(S)
    return P[ok], H[ok], S[ok], row[ok], len(d)


def load26():
    P, H, S, run, robot = [], [], [], [], []
    files = sorted(glob.glob(os.path.join(RUNS26, "**", "*.mat"), recursive=True))
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
            P.append(p[ok]); H.append(h[ok]); S.append(s[ok])
            run += [j] * ok.sum(); robot += [names[f"r{r + 1}"]] * ok.sum()
    return np.vstack(P), np.concatenate(H), np.concatenate(S), np.array(run), np.array(robot), files


# ---------------------------------------------------------------- 2. saturation fit (centre and slope)
def sat_fit(P, S, c0, rmax=0.65):
    def res(x):
        d = np.hypot(*(P - x[:2]).T)
        return np.where((d > 0.08) & (d < rmax), x[2] + x[3] * d - S, 0.0)
    return least_squares(res, [c0[0], c0[1], 0.3, 0.6], loss="soft_l1", f_scale=0.03).x


# ---------------------------------------------------------------- 3. align 2026 to the 2025 frame
def align(P26, H26, c26, k, P25, H25, c25):
    """Rotation, mirror and translation taking 2026 positions into the 2025 frame, chosen so each
    2026 hue matches the 2025 hue map (2 cm cells, circular mean) at the same mat point."""
    lo, w = P25.min(0) - 0.05, 0.02
    ij = np.floor((P25 - lo) / w).astype(int)
    shape = tuple(ij.max(0) + 2)
    cnt = np.zeros(shape); sn = np.zeros(shape); cs = np.zeros(shape)
    np.add.at(cnt, tuple(ij.T), 1)
    np.add.at(sn, tuple(ij.T), np.sin(2 * np.pi * H25))
    np.add.at(cs, tuple(ij.T), np.cos(2 * np.pi * H25))
    ref = np.where(cnt >= 5, np.arctan2(sn, cs), np.nan)          # 2025 hue angle per cell
    rng = np.random.default_rng(0)
    near = np.hypot(*(P26 - c26).T) < 0.6
    i = rng.choice(np.where(near)[0], 8000, replace=False)

    def place(x, m, P):
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
        e = 1 - np.cos(2 * np.pi * H26[i][ok] - r[ok])
        return np.minimum(e, 0.5).mean() + 0.5 * (1 - ok.mean())

    best = min(((loss([th, *c25], m), m, th) for m in (False, True)
                for th in np.radians(np.arange(0, 360, 5))), key=lambda t: t[0])
    _, m, th = best
    x = minimize(loss, [th, *c25], args=(m,), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-8, maxiter=2000)).x
    return x, m, place


# ---------------------------------------------------------------- 6. grid and train
def cells(P, H, S):
    """1 cm cells: circular mean hue and mean saturation, after dropping readings more than
    20 deg from their cell's first-pass mean."""
    key = np.floor(P / CELL).astype(int)
    df = pd.DataFrame({"kx": key[:, 0], "ky": key[:, 1], "x": P[:, 0], "y": P[:, 1],
                       "sn": np.sin(2 * np.pi * H), "cs": np.cos(2 * np.pi * H), "s": S})
    g = df.groupby(["kx", "ky"])
    first = np.arctan2(g["sn"].transform("mean"), g["cs"].transform("mean"))
    dev = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(df["sn"], df["cs"]) - first)))))
    df = df[dev <= 20]
    g = df.groupby(["kx", "ky"]).agg(x=("x", "mean"), y=("y", "mean"), sn=("sn", "mean"),
                                     cs=("cs", "mean"), s=("s", "mean"), n=("s", "size")).reset_index()
    return g[(g["n"] >= 3) & (g[["x", "y"]].abs().max(axis=1) < EXTENT)]


def grid(P26, H26, S26, P25, H25, S25):
    g26 = cells(P26, H26, S26)
    g25 = cells(P25, H25, S25)
    have = set(zip(g26["kx"], g26["ky"]))
    g25 = g25[[k not in have for k in zip(g25["kx"], g25["ky"])]]
    print(f"cells: {len(g26):,} from 2026, {len(g25):,} from 2025 where 2026 has none")
    g = pd.concat([g26, g25], ignore_index=True)
    ang = np.arctan2(g["sn"], g["cs"])          # circular mean direction per cell
    X = g[["x", "y"]].values.astype(np.float32)
    Yh = np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)
    Ys = (2 * g["s"].values - 1)[:, None].astype(np.float32)
    return X, Yh, Ys


def net(arch):
    layers, n_in = [], 2
    for h in arch["hidden_sizes"]:
        layers += [nn.Linear(n_in, h), nn.Tanh()]
        n_in = h
    layers.append(nn.Linear(n_in, arch["output_neurons"]))
    return nn.Sequential(*layers)


class Wrapped(nn.Module):        # same state_dict layout as NNField's FieldPredictor (.network.*)
    def __init__(self, arch):
        super().__init__()
        self.network = net(arch)

    def forward(self, x):
        return self.network(x)


def train(X, Y, arch, epochs=6000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    m = Wrapped(arch)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xt) - Yt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m.eval()


def predict(hm, sm, P):
    with torch.no_grad():
        t = torch.as_tensor(P, dtype=torch.float32)
        h = hm(t).numpy()
        s = (sm(t).numpy()[:, 0] + 1) / 2
    return (np.arctan2(h[:, 0], h[:, 1]) % (2 * np.pi)) / (2 * np.pi), s


def floor(Ptest, Htest, Stest, Pref, Href, Sref):
    """Noise floor: each test reading predicted by other readings within 1.5 cm."""
    tree = cKDTree(Pref)
    eh, es, keep = [], [], []
    for k, q in enumerate(Ptest):
        nb = tree.query_ball_point(q, 0.015)
        if len(nb) >= 5:
            eh.append(dang(circ_mean(Href[nb]), Htest[k])); es.append(Sref[nb].mean() - Stest[k]); keep.append(k)
    return np.array(eh), np.array(es), np.array(keep, int)


def main():
    out = {}
    P25, H25, S25, row25, n25 = load25()
    P26, H26, S26, run26, robot26, files26 = load26()

    # 2. scale and centres from saturation (trained robots only for 2026)
    tr = np.isin(robot26, list(TRAINED))
    f25 = sat_fit(P25, S25, (0.0, 0.0))
    f26 = sat_fit(P26[tr], S26[tr], (0.83, 0.80))
    k = f26[3] / f25[3]
    out["saturation_fit"] = {"2025": dict(centre=f25[:2].tolist(), a=f25[2], b=f25[3]),
                             "2026": dict(centre=f26[:2].tolist(), a=f26[2], b=f26[3]),
                             "length_ratio_2026_to_2025": k}
    print(f"saturation slope 2025 {f25[3]:.4f}, 2026 {f26[3]:.4f}: 2026 lengths x {k:.4f} in the 2025 frame")

    # 3. align 2026 into the 2025 frame
    x, m, place = align(P26, H26, f26[:2], k, P25, H25, f25[:2])
    Q26 = place(x, m, P26)
    out["align_2026_to_2025"] = dict(rotation_deg=float(np.degrees(x[0]) % 360), mirror=bool(m),
                                     translation=x[1:3].tolist(), centre_2026_motive=f26[:2].tolist(), scale=k)
    print(f"2026 -> 2025 frame: rotation {np.degrees(x[0]) % 360:.2f} deg, mirror {m}, "
          f"2026 saturation centre lands at ({x[1]:+.4f}, {x[2]:+.4f})")

    # 4. centre on the pooled critical point
    trn26 = tr & (run26 % 5 != 0)
    pool_P = np.vstack([P25, Q26[trn26]]); pool_S = np.concatenate([S25, S26[trn26]])
    c = sat_fit(pool_P, pool_S, f25[:2])[:2]
    P25c, Q26c = P25 - c, Q26 - c
    out["critical_point_in_old_frame"] = c.tolist()
    print(f"critical point (pooled saturation centre) in the old saddle_nn frame: ({c[0]:+.4f}, {c[1]:+.4f})")

    # 5. split
    test25 = row25 >= int(0.8 * n25)
    test26 = tr & (run26 % 5 == 0)
    red26 = robot26 == "redwood"
    Ptr = np.vstack([P25c[~test25], Q26c[trn26]])
    Htr = np.concatenate([H25[~test25], H26[trn26]]); Str = np.concatenate([S25[~test25], S26[trn26]])

    # 6. grid and train
    X, Yh, Ys = grid(Q26c[trn26], H26[trn26], S26[trn26], P25c[~test25], H25[~test25], S25[~test25])
    print(f"training cells: {len(X):,} (1 cm, at least 3 readings, within +-{EXTENT} m) from {len(Ptr):,} readings")
    hm, sm = train(X, Yh, HUE_ARCH), train(X, Ys, SAT_ARCH)

    # 7. validate
    sys.path.insert(0, VF)
    from src.fields.field_types import NNField
    old = NNField(os.path.join(VF, "saddle_predictors"))
    rep = {}
    for name, Pt, Ht, St in (("2026 held-out runs", Q26c[test26], H26[test26], S26[test26]),
                             ("2026 redwood (unseen robot)", Q26c[red26], H26[red26], S26[red26]),
                             ("2025 held-out last 20%", P25c[test25], H25[test25], S25[test25])):
        hn, sn = predict(hm, sm, Pt)
        ho, so = predict(old.hue_model, old.sat_model, Pt + c)
        fh, fs, keep = floor(Pt, Ht, St, Ptr, Htr, Str)
        en, eo = dang(hn, Ht), dang(ho, Ht)
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

    # what the plane fit sees near the centre: linear fit of the decoded field over a ring the size
    # of the 0.30 m formation (0.08 to 0.35 m), estimate p* = -J^-1 h, and the field type
    def decoded(hmod, smod, P):
        h, s = predict(hmod, smod, P)
        ang = 2 * np.pi * h + np.pi / 2
        mag = (s - 0.3) / 0.7
        return np.stack([mag * np.cos(ang), mag * np.sin(ang)], 1)
    g = np.stack(np.meshgrid(np.linspace(-0.35, 0.35, 71), np.linspace(-0.35, 0.35, 71)), -1).reshape(-1, 2)
    g = g[(np.hypot(*g.T) > 0.08) & (np.hypot(*g.T) < 0.35)]
    A = np.c_[g, np.ones(len(g))]
    for tag, hmod, smod, shift in (("new", hm, sm, 0.0), ("old", old.hue_model, old.sat_model, c)):
        F = decoded(hmod, smod, g + shift)
        coef = np.linalg.lstsq(A, F, rcond=None)[0]
        J, h0 = coef[:2].T, coef[2]
        p = -np.linalg.solve(J, h0)
        r0 = np.hypot(*g.T)
        s_c = predict(hmod, smod, np.array([[0.0, 0.0]]) + shift)[1][0]
        out[f"plane_fit_{tag}"] = dict(p_star=p.tolist(), J=J.tolist(), det=float(np.linalg.det(J)),
                                       trace=float(np.trace(J)), sat_at_centre=float(s_c))
        print(f"plane fit, {tag} net: p* = ({p[0]:+.4f}, {p[1]:+.4f}) m, det J {np.linalg.det(J):+.3f}, "
              f"trace {np.trace(J):+.3f} ({'saddle' if np.linalg.det(J) < 0 else 'NOT a saddle'}), "
              f"saturation at the centre {s_c:.3f}")

    # save in NNField's format
    torch.save(hm.state_dict(), os.path.join(HERE, "hue_model.pth"))
    torch.save(sm.state_dict(), os.path.join(HERE, "sat_model.pth"))
    with open(os.path.join(HERE, "model_info.pkl"), "wb") as fh:
        pickle.dump({"hue_architecture": HUE_ARCH, "sat_architecture": SAT_ARCH}, fh)
    with open(os.path.join(HERE, "scaling_info.pkl"), "wb") as fh:
        pickle.dump({"input_range": (-EXTENT, EXTENT), "hue_representation": "circular_sin_cos",
                     "sat_scaling": "[0,1] -> [-1,1]", "frame": "old saddle_nn frame shifted so the "
                     "critical point is at (0, 0)"}, fh)
    with open(os.path.join(HERE, "build_report.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("saved hue_model.pth, sat_model.pth, model_info.pkl, scaling_info.pkl, build_report.json")


if __name__ == "__main__":
    main()
