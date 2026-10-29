"""Load study results: manifest rows plus per-trial arrays, and per-cell summaries."""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOAT_COLS = ["eps", "N", "dt", "T", "window", "side", "alpha", "vs", "vmax", "k", "vcmd_max",
              "kt", "kr", "rclamp", "rd", "sigma_n", "kf", "kb", "r0"]

CONV = 0.05          # convergence radius (m) for attraction
ORBIT_FAIL_REVS = 0.25


def manifest(res="results"):
    rows = list(csv.DictReader(open(os.path.join(HERE, res, "manifest.csv"))))
    for r in rows:
        for k in FLOAT_COLS:
            r[k] = float(r.get(k, "nan"))
        r["random"] = r["random"] == "1"
        r["stiction"] = r.get("stiction") or "legacy"
        r["hw"] = r.get("hw") or ""
        r["path"] = os.path.join(HERE, res, "cells", r["file"])
    return rows


def trials(row):
    with np.load(row["path"]) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def attract_summary(row, M):
    ok = (M["escaped"] == 0) & (M["max_side_err"] < 0.1)
    conv = ok & (M["final_err"] < CONV)
    err = M["final_err"][conv]
    fx, fy = M["final_x"][conv], M["final_y"][conv]
    return dict(n=len(M["final_err"]), conv=conv.mean(), escaped=(M["escaped"] > 0).mean(),
                other_cp=(ok & (M["nearest"] > 0) & (M["nearest_dist"] < CONV)).mean(),
                stalled=(ok & (M["nearest_dist"] >= CONV)).mean(),
                err_mean=err.mean() if len(err) else np.nan,
                err_med=np.median(err) if len(err) else np.nan,
                err_p95=np.percentile(err, 95) if len(err) else np.nan,
                err_max=err.max() if len(err) else np.nan,
                bias=np.hypot(fx.mean(), fy.mean()) if len(err) else np.nan,
                prec=np.sqrt(fx.var() + fy.var()) if len(err) else np.nan,
                t5_med=np.nanmedian(M["t5"][conv]) if conv.any() else np.nan,
                over_med=np.median(M["overshoot"][ok]) if ok.any() else np.nan,
                over_p95=np.percentile(M["overshoot"][ok], 95) if ok.any() else np.nan,
                path_med=np.median(M["path"][ok]) if ok.any() else np.nan,
                sat=np.mean(M["sat"][ok]) if ok.any() else np.nan,
                side_err=M["max_side_err"].max())


def orbit_summary(row, M):
    ok = (M["escaped"] == 0) & (M["max_side_err"] < 0.1) & (M["revs"] >= ORBIT_FAIL_REVS) \
        & np.isfinite(M["mean_r"])
    e = M["mean_r"][ok] - row["rd"]
    rms = np.sqrt((M["std_r"][ok] ** 2 + e ** 2))
    return dict(n=len(M["mean_r"]), ok=ok.mean(), escaped=(M["escaped"] > 0).mean(),
                nostart=(M["path"] < 1e-9).mean(),
                err_mean=e.mean() if ok.any() else np.nan,
                err_between_sd=e.std() if ok.any() else np.nan,
                within_sd=np.mean(M["std_r"][ok]) if ok.any() else np.nan,
                rms=np.sqrt(np.mean(rms ** 2)) if ok.any() else np.nan,
                mean_r=np.mean(M["mean_r"][ok]) if ok.any() else np.nan,
                a2=np.mean(M["a2"][ok]) if ok.any() else np.nan,
                ctr_off=np.mean(M["ctr_off"][ok]) if ok.any() else np.nan,
                est_err=np.mean(M["est_err"][ok]) if ok.any() else np.nan,
                revs=np.median(M["revs"]),
                path_rate=np.median(M["path"][ok]) if ok.any() else np.nan,
                sat=np.mean(M["sat"][ok]) if ok.any() else np.nan,
                side_err=M["max_side_err"].max())


def summary(row, M=None):
    M = trials(row) if M is None else M
    return (attract_summary if row["task"] == "attract" else orbit_summary)(row, M)
