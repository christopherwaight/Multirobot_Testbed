"""Hardware-matched runs on the RBF maps, against the same runs on the 2026 nets and on the hardware.

Re-runs analysis/simulation/paired_runs.py (same starts, durations, controller and robot model) with the
reconstructed field swapped for each RBF from rbf_fields.py. The net results come from paired/*.csv.
Score per (field, robots, r_d) group: mean orbit bias, within-run spread and centre-estimate error, sim minus
hardware; per attraction set: mean final error and steady centre-estimate error.
Writes analysis/rbf/outputs/paired_rbf_*.csv. Run from the repo root after rbf_fields.py.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "simulation"))
import paired_runs as pr  # noqa: E402
import vecsim as vs  # noqa: E402
import paths  # noqa: E402

OUT = os.path.join(HERE, "outputs")
VARIANTS = ("rbf_exact", "rbf_smooth")


def rbf_field(name, model, kind):
    def fn(x, y):
        shp = np.shape(x)
        P = np.stack([np.ravel(x), np.ravel(y)], 1)
        hs = model["hue"](P)
        s = (model["sat"](P)[:, 0] + 1) / 2
        ang = np.arctan2(hs[:, 0], hs[:, 1]) + np.pi / 2
        m = (s - 0.3) / 0.7
        return (m * np.cos(ang)).reshape(shp), (m * np.sin(ang)).reshape(shp)
    return vs.Field(name, "reconstructed", fn, [(0.0, 0.0, 1.0 if kind == "vortex" else -1.0)],
                    vs.LINEAR_J[kind], 0.5, 0.5, "RBF fit to testbed color-sensor data")


def main():
    extra = {}
    for kind in ("vortex", "saddle"):
        M = pickle.load(open(os.path.join(OUT, f"{kind}_rbf.pkl"), "rb"))
        for v in VARIANTS:
            extra[f"{kind}_{v}"] = rbf_field(f"{kind}_{v}", M[v], kind)
    orig = vs.make_field
    vs.make_field = lambda name, *a, **k: extra[name] if name in extra else orig(name, *a, **k)

    keys_o, keys_a = ["field", "nr", "rd", "run"], ["field", "nr", "run"]
    O = pd.read_csv(os.path.join(paths.OUT_SIM, "paired", "paired_orbits.csv"))
    A = pd.read_csv(os.path.join(paths.OUT_SIM, "paired", "paired_attraction.csv"))
    O = O.rename(columns=lambda c: c.replace("recon_", "nn_"))
    A = A.rename(columns=lambda c: c.replace("recon_", "nn_"))
    for v in VARIANTS:
        pr.RECON = {"vortex": f"vortex_{v}", "saddle": f"saddle_{v}"}
        print(f"running {v} ...", flush=True)
        o = pr.orbits()
        O = O.merge(o[keys_o + [c for c in o if c.startswith("recon_")]].rename(
            columns=lambda c: c.replace("recon_", f"{v}_")), on=keys_o)
        a = pr.attraction()
        A = A.merge(a[keys_a + [c for c in a if c.startswith("recon_")]].rename(
            columns=lambda c: c.replace("recon_", f"{v}_")), on=keys_a)

    HO = pd.read_csv(os.path.join(paths.OUT_HW, "outputs", "orbit_per_run.csv"))
    HA = pd.read_csv(os.path.join(paths.OUT_HW, "outputs", "attraction_per_trial.csv"))
    O = O.merge(HO[keys_o + ["bias_w1", "sd_within_w1", "est_err_w1"]].rename(
        columns=lambda c: c if c in keys_o else f"hw_{c}"), on=keys_o)
    A = A.merge(HA[keys_a + ["e_final", "est_ss", "t_settle_5cm"]].rename(
        columns=lambda c: c if c in keys_a else f"hw_{c}"), on=keys_a)
    O.to_csv(os.path.join(OUT, "paired_rbf_orbits.csv"), index=False)
    A.to_csv(os.path.join(OUT, "paired_rbf_attraction.csv"), index=False)

    tags = ["linear", "nn"] + list(VARIANTS)
    G = O.groupby(["field", "nr", "rd"]).mean(numeric_only=True).reset_index()
    rows = []
    for (field, nr), g in G.groupby(["field", "nr"]):
        for q in ("bias_w1", "sd_within_w1", "est_err_w1"):
            r = dict(field=field, nr=nr, metric=q, hw_mean=g[f"hw_{q}"].mean())
            for t in tags:
                r[f"{t}_rms_gap"] = float(np.sqrt(((g[f"{t}_{q}"] - g[f"hw_{q}"]) ** 2).mean()))
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "paired_rbf_orbit_gaps.csv"), index=False)
    lost = O.groupby(["field", "nr"])[[f"{t}_lost" for t in tags[1:]]].sum().reset_index()

    SA = A.groupby(["field", "nr"]).agg(
        **{"hw_e": ("hw_e_final", "mean"), "hw_loc": ("hw_est_ss", "mean"), "hw_t5": ("hw_t_settle_5cm", "median")},
        **{f"{t}_e": (f"{t}_e_final", "mean") for t in tags},
        **{f"{t}_loc": (f"{t}_est_ss", "mean") for t in tags[1:]},
        **{f"{t}_t5": (f"{t}_t_settle_5cm", "median") for t in tags}).reset_index()
    SA.to_csv(os.path.join(OUT, "paired_rbf_attraction_summary.csv"), index=False)
    G.to_csv(os.path.join(OUT, "paired_rbf_orbit_groups.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nOrbit: RMS over r_d of (sim group mean - hardware group mean), metres")
    print(S.round(4).to_string(index=False))
    print("\nOrbit runs lost:"); print(lost.to_string(index=False))
    print("\nAttraction (means; t5 medians):"); print(SA.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
