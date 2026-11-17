"""Find an orbit radius where the cluster falls into a limit cycle, in simulation and on hardware.

A limit cycle here is a closed orbit that the cluster converges to from both sides, and that is not
the commanded circle. Test, per r_d:
  - Simulation, 2026 reconstructed fields: 8 bearings x 2 starts (inside: 0.4 r_d; outside: r_d +
    0.2 m), 90 s. Over the last 30 s, bin r against the bearing about p* (10 deg bins): the cycle's
    curve r(theta). Converged when the inside and outside curves agree (median gap D) and each
    repeats lap to lap. A cycle stands out when its mean radius differs from r_d, or its shape
    is not a circle (amplitude A = max - min of r(theta)).
  - Hardware, 2026 runs mapped into the same field frame: the 10 runs at one r_d, concatenated.
    Same r(theta) curves after the 3 s transient; across-run agreement, and whether runs starting
    inside and outside the cycle meet on it.
Run from the repo root. Writes analysis/limit_cycle/outputs/ and figures/.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import paths  # noqa: E402
import rigidsim as rs  # noqa: E402
import vecsim as vs  # noqa: E402
import frames  # noqa: E402

OUT = os.path.join(HERE, "outputs")
RECON = {"vortex": "vortex_nn_2026", "saddle": "saddle_nn_2026"}
BINS = np.radians(np.arange(-180, 181, 10))
RDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def curve(c, t0=0):
    """r(theta) in 10 deg bins of the bearing about the origin, from samples t0 on."""
    p = c[t0:]
    th = np.arctan2(p[:, 1], p[:, 0])
    r = np.hypot(p[:, 0], p[:, 1])
    idx = np.digitize(th, BINS) - 1
    return np.array([np.mean(r[idx == b]) if np.any(idx == b) else np.nan for b in range(len(BINS) - 1)])


def turn_of(field):
    fr = frames.frame_2026(field, np.zeros(2))
    return -1.0 if fr.mirror else 1.0


def sim_scan(field, nr=3):
    rows, curves = [], {}
    bear = np.radians(np.arange(0, 360, 45))
    for rd in RDS:
        starts, tags = [], []
        for r0, tag in ((0.4 * rd if rd > 0.05 else 0.02, "in"), (rd + 0.2, "out")):
            starts += [r0 * np.array([np.cos(b), np.sin(b)]) for b in bear]
            tags += [tag] * len(bear)
        starts = np.array(starts)
        n = len(starts)
        th = np.zeros(n)
        off = rs.triangle(th, np.full(n, 0.30)) if nr == 3 else rs.square(th)
        P = vs.params(n, rd=rd, turn=turn_of(field))
        C, _ = rs.simulate(vs.make_field(RECON[field]), "orbit", "fit", P, 90.0, off, starts)
        K = np.array([curve(C[i], 600) for i in range(n)])
        tags = np.array(tags)
        cin, cout = np.nanmean(K[tags == "in"], 0), np.nanmean(K[tags == "out"], 0)
        late = [curve(C[i], 800) for i in range(n)]
        mid = [curve(C[i], 600)[:] for i in range(n)]
        rep = np.nanmedian(np.abs(np.array(late) - np.array(mid)))
        cm = np.nanmean(K, 0)
        rows.append(dict(field=field, nr=nr, rd=rd, mean_r=float(np.nanmean(cm)), offset=float(np.nanmean(cm) - rd),
                         amplitude=float(np.nanmax(cm) - np.nanmin(cm)),
                         D_in_out=float(np.nanmedian(np.abs(cin - cout))), repeat=float(rep),
                         lost=int(np.sum(np.hypot(*C[:, -1].T) > 1.0))))
        curves[rd] = dict(C=C[:, ::2], tags=tags, curve=cm)
    return pd.DataFrame(rows), curves


def hw_scan(field):
    runs = pickle.load(open(os.path.join(paths.OUT_HW, "outputs", "runs2026.pkl"), "rb"))
    fits = json.load(open(os.path.join(paths.OUT_HW, "outputs", "centres_2026.json")))
    ps = np.array(fits[f"{field}_pooled"]["p_star"])
    fr = frames.frame_2026(field, ps)
    rows, paths_ = [], {}
    for nr in (3, 4):
        for rd in sorted({r["rd"] for r in runs if r["field"] == field and r["nr"] == nr}):
            g = [r for r in runs if r["field"] == field and r["nr"] == nr and r["rd"] == rd]
            Cs = [fr(r["c"] - ps) for r in g]
            K = np.array([curve(C, 30) for C in Cs])
            cm = np.nanmean(K, 0)
            r0 = np.array([np.hypot(*C[0]) for C in Cs])
            rl = np.array([np.nanmean(np.hypot(*C[-50:].T)) for C in Cs])
            rows.append(dict(field=field, nr=nr, rd=rd, n=len(g), mean_r=float(np.nanmean(cm)), offset=float(np.nanmean(cm) - rd),
                             amplitude=float(np.nanmax(cm) - np.nanmin(cm)), across_run=float(np.nanmedian(np.nanstd(K, 0))),
                             starts_inside=int(np.sum(r0 < np.nanmean(cm))), starts_outside=int(np.sum(r0 >= np.nanmean(cm))),
                             r0_min=float(r0.min()), r0_max=float(r0.max()), late_r_sd=float(rl.std())))
            paths_[(nr, rd)] = dict(C=Cs, curve=cm, r0=r0)
    return pd.DataFrame(rows), paths_


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for field in ("vortex", "saddle"):
        S, sc = sim_scan(field)
        H, hp = hw_scan(field)
        S.to_csv(os.path.join(OUT, f"sim_{field}.csv"), index=False)
        H.to_csv(os.path.join(OUT, f"hw_{field}.csv"), index=False)
        pickle.dump(dict(sim=sc, hw=hp), open(os.path.join(OUT, f"paths_{field}.pkl"), "wb"))
        print(f"\n=== {field}: simulation, 2026 reconstructed field (3R)")
        print(S.round(4).to_string())
        print(f"=== {field}: hardware 2026, in the same frame")
        print(H.round(4).to_string())
        res[field] = dict(sim=S.to_dict("records"), hw=H.to_dict("records"))
    json.dump(res, open(os.path.join(OUT, "limit_cycle_scan.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
