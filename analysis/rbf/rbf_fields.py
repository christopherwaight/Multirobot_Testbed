"""Thin-plate-spline (RBF) reconstruction of the vortex and saddle maps, on the same data as the 2026 nets.

Same readings, frame, 1 cm cells and held-out runs as vortex_predictors_2026 and saddle_predictors_2026 (frame
read from their build_report.json; cells from the trainers' own grid()). Hue is fit as (sin, cos), saturation
as 2S - 1, exactly as the nets are. Two RBFs per field:
  exact    smoothing 0, as in the 2025 rbf_trainer (passes through every cell)
  smooth   smoothing chosen on validation runs (training runs with run % 5 == 1), then refit on all training cells
Compared with the nets on the held-out runs, the plane fit over a formation-sized ring, and the centre a
0.30 m triangle would estimate from each position.
Writes analysis/rbf/outputs/{vortex,saddle}_rbf.pkl, rbf_build.json and analysis/rbf/figures/.
Run from the repo root with the VF_Robot venv.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402
from map_readings import NETDIR, SAD, cells, prepare  # noqa: E402,F401
from src.fields.field_types import NNField  # noqa: E402

OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
LAMBDAS = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
BINS = ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .65), (.65, 1.5))


def fit(X, Y, lam):
    return RBFInterpolator(X, Y, kernel="thin_plate_spline", degree=1, smoothing=lam)


def predict(model, P):
    if "nn" in model:
        return SAD.predict(model["nn"].hue_model, model["nn"].sat_model, P)
    hs = model["hue"](P)
    s = (model["sat"](P)[:, 0] + 1) / 2
    return (np.arctan2(hs[:, 0], hs[:, 1]) % (2 * np.pi)) / (2 * np.pi), s


def decoded(model, P):
    h, s = predict(model, P)
    ang = 2 * np.pi * h + np.pi / 2
    m = (s - 0.3) / 0.7
    return np.stack([m * np.cos(ang), m * np.sin(ang)], 1)


def choose_lambda(d):
    X, Yh, Ys = cells(d, d["trn26"] & ~d["val26"])
    Pv, Hv, Sv = d["Q26"][d["val26"]], d["H26"][d["val26"]], d["S26"][d["val26"]]
    scan = []
    for lam in LAMBDAS:
        hs = fit(X, Yh, lam)(Pv)
        sv = (fit(X, Ys, lam)(Pv)[:, 0] + 1) / 2
        h = (np.arctan2(hs[:, 0], hs[:, 1]) % (2 * np.pi)) / (2 * np.pi)
        scan.append(dict(lam=lam, dir_deg=float(np.median(np.abs(SAD.dang(h, Hv)))),
                         sat=float(np.median(np.abs(sv - Sv)))))
        print(f"  smoothing {lam:7.0e}: validation direction {scan[-1]['dir_deg']:.3f} deg, saturation {scan[-1]['sat']:.4f}")
    lam_h = min(scan, key=lambda r: r["dir_deg"])["lam"]
    lam_s = min(scan, key=lambda r: r["sat"])["lam"]
    return lam_h, lam_s, scan


def heldout(models, d):
    rep = {}
    floor = {b["r"]: b["floor"] for b in d["rep"]["validation"]["2026 held-out runs"]["bins"]}
    for name, m in (("2026 held-out runs", d["test26"]), ("2026 redwood (unseen robot)", d["red26"])):
        P, H, S = d["Q26"][m], d["H26"][m], d["S26"][m]
        rad = np.hypot(*P.T)
        rows = []
        for lo, hi in BINS:
            b = (rad >= lo) & (rad < hi)
            if b.sum() < 50:
                continue
            r = dict(r=f"{lo:.2f}-{hi:.2f}", n=int(b.sum()), floor=floor.get(f"{lo:.2f}-{hi:.2f}", float("nan")))
            for tag, mod in models.items():
                h, s = predict(mod, P[b])
                e = SAD.dang(h, H[b])
                r[tag] = float(np.median(np.abs(e)))
                r[f"{tag}_bias"] = float(np.mean(e))
                r[f"{tag}_sat"] = float(np.median(np.abs(s - S[b])))
            rows.append(r)
        allr = {}
        for tag, mod in models.items():
            h, s = predict(mod, P)
            allr[tag] = float(np.median(np.abs(SAD.dang(h, H))))
            allr[f"{tag}_sat"] = float(np.median(np.abs(s - S)))
        rep[name] = dict(bins=rows, all=allr, n=int(m.sum()))
    return rep


def plane_fit(model):
    g = np.stack(np.meshgrid(np.linspace(-0.35, 0.35, 71), np.linspace(-0.35, 0.35, 71)), -1).reshape(-1, 2)
    g = g[(np.hypot(*g.T) > 0.08) & (np.hypot(*g.T) < 0.35)]
    coef = np.linalg.lstsq(np.c_[g, np.ones(len(g))], decoded(model, g), rcond=None)[0]
    J, h0 = coef[:2].T, coef[2]
    p = -np.linalg.solve(J, h0)
    return dict(p_star=p.tolist(), det=float(np.linalg.det(J)), trace=float(np.trace(J)))


def triangle_estimates(model, field, rmax=0.40, step=0.02, side=0.30, n_head=6):
    """Centre a 0.30 m triangle estimates (3-point plane fit) at each centroid position; distance from the
    true centre (0, 0) and whether det J has the right sign."""
    g = np.stack(np.meshgrid(np.arange(-rmax, rmax + 1e-9, step), np.arange(-rmax, rmax + 1e-9, step)), -1).reshape(-1, 2)
    g = g[np.hypot(*g.T) <= rmax]
    R = side / np.sqrt(3)
    err, ok = [], []
    for k in range(n_head):
        a0 = 2 * np.pi * k / (3 * n_head)
        offs = R * np.stack([np.cos(a0 + 2 * np.pi * np.arange(3) / 3), np.sin(a0 + 2 * np.pi * np.arange(3) / 3)], 1)
        pts = (g[:, None, :] + offs[None]).reshape(-1, 2)
        F = decoded(model, pts).reshape(len(g), 3, 2)
        A = np.concatenate([g[:, None, :] + offs[None], np.ones((len(g), 3, 1))], 2)
        coef = np.linalg.solve(A, F)                       # rows: d/dx, d/dy, constant
        J = np.swapaxes(coef[:, :2, :], 1, 2)
        h0 = coef[:, 2, :]
        p = -np.linalg.solve(J, h0[..., None])[..., 0]
        err.append(np.hypot(*p.T))
        ok.append(np.sign(np.linalg.det(J)) == (1 if field == "vortex" else -1))
    err, ok = np.array(err), np.array(ok)
    r = np.hypot(*g.T)
    near = r <= 0.20
    return dict(median_err_all=float(np.median(err)), p90_err_all=float(np.quantile(err, 0.9)),
                median_err_near=float(np.median(err[:, near])), p90_err_near=float(np.quantile(err[:, near], 0.9)),
                wrong_type_pct=float(100 * (1 - ok.mean()))), (g, np.median(err, 0))


def figures(fields_models, tri_maps, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    os.makedirs(FIG, exist_ok=True)
    tags = ["nn", "rbf_exact", "rbf_smooth"]
    titles = {"cells": "Training cells", "nn": "NN (2026)", "rbf_exact": "RBF, exact", "rbf_smooth": "RBF, smoothed"}
    L = 0.6
    x = np.linspace(-L, L, 241)
    G = np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for i, field in enumerate(("vortex", "saddle")):
        X, Yh, Ys = data[field]
        k = np.abs(X).max(1) < L
        hue = (np.arctan2(Yh[k, 0], Yh[k, 1]) % (2 * np.pi)) / (2 * np.pi)
        sat = np.clip((Ys[k, 0] + 1) / 2, 0, 1)
        ax[i, 0].scatter(X[k, 0], X[k, 1], c=hsv_to_rgb(np.stack([hue, sat, np.ones_like(hue)], 1)), s=1.2, marker="s")
        ax[i, 0].set_title(f"{field}: {titles['cells']}", fontsize=9)
        for j, t in enumerate(tags):
            h, s = predict(fields_models[field][t], G)
            img = hsv_to_rgb(np.stack([h, np.clip(s, 0, 1), np.ones_like(h)], 1)).reshape(len(x), len(x), 3)
            ax[i, j + 1].imshow(img, origin="lower", extent=(-L, L, -L, L))
            sub = G[::1].reshape(len(x), len(x), 2)[::16, ::16].reshape(-1, 2)
            F = decoded(fields_models[field][t], sub)
            ax[i, j + 1].quiver(sub[:, 0], sub[:, 1], F[:, 0], F[:, 1], color="k", width=0.003, scale=8)
            ax[i, j + 1].set_title(f"{field}: {titles[t]}", fontsize=9)
    for a in ax.flat:
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal"); a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rbf_vs_nn_maps.png"), dpi=150)

    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    for i, field in enumerate(("vortex", "saddle")):
        for j, t in enumerate(tags):
            g, e = tri_maps[field][t]
            sc = ax[i, j].scatter(g[:, 0], g[:, 1], c=100 * e, s=14, marker="s", cmap="viridis", vmin=0, vmax=10)
            ax[i, j].set_title(f"{field}: {titles[t]}", fontsize=9)
            ax[i, j].set_aspect("equal"); ax[i, j].tick_params(labelsize=7)
    fig.colorbar(sc, ax=ax, label="centre estimate error of a 0.30 m triangle (cm), median over headings")
    fig.savefig(os.path.join(FIG, "rbf_vs_nn_triangle_error.png"), dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    res, models_all, tri_maps, data = {}, {}, {}, {}
    for field in ("vortex", "saddle"):
        print(f"\n=== {field}")
        d = prepare(field)
        print("choosing smoothing on validation runs:")
        lam_h, lam_s, scan = choose_lambda(d)
        X, Yh, Ys = cells(d, d["trn26"])
        data[field] = (X, Yh, Ys)
        models = {"nn": {"nn": NNField(os.path.join(paths.VF, NETDIR[field]))},
                  "rbf_exact": {"hue": fit(X, Yh, 0.0), "sat": fit(X, Ys, 0.0)},
                  "rbf_smooth": {"hue": fit(X, Yh, lam_h), "sat": fit(X, Ys, lam_s)}}
        models_all[field] = models
        r = dict(lambda_hue=lam_h, lambda_sat=lam_s, scan=scan, n_cells=len(X), heldout=heldout(models, d))
        r["plane_fit"] = {t: plane_fit(m) for t, m in models.items()}
        r["triangle"], tri_maps[field] = {}, {}
        for t, m in models.items():
            r["triangle"][t], tri_maps[field][t] = triangle_estimates(m, field)
        res[field] = r
        pickle.dump({t: m for t, m in models.items() if t != "nn"}, open(os.path.join(OUT, f"{field}_rbf.pkl"), "wb"))

        print(f"cells {len(X):,}; smoothing hue {lam_h:.0e}, saturation {lam_s:.0e}")
        for name, v in r["heldout"].items():
            print(f"{name} (n {v['n']:,}): median direction error deg (bias)")
            for b in v["bins"]:
                print(f"  r {b['r']}  n {b['n']:6d}  floor {b['floor']:5.2f}  " + "  ".join(
                    f"{t} {b[t]:5.2f} ({b[t + '_bias']:+5.2f})" for t in models))
            print("  all  " + "  ".join(f"{t} {v['all'][t]:.2f} deg, sat {v['all'][t + '_sat']:.4f}" for t in models))
        for t in models:
            pf, tri = r["plane_fit"][t], r["triangle"][t]
            print(f"{t:10s} plane fit p* ({pf['p_star'][0]:+.4f}, {pf['p_star'][1]:+.4f}) det {pf['det']:+.3f} "
                  f"trace {pf['trace']:+.3f} | triangle estimate error median {100 * tri['median_err_all']:.2f} cm "
                  f"(within 0.2 m {100 * tri['median_err_near']:.2f}, p90 {100 * tri['p90_err_near']:.2f}), "
                  f"wrong type {tri['wrong_type_pct']:.2f}%")
    json.dump(res, open(os.path.join(OUT, "rbf_build.json"), "w"), indent=1, default=float)
    figures(models_all, tri_maps, data)
    print("\nwrote", OUT, "and", FIG)


if __name__ == "__main__":
    main()
