"""
Score every run listed in Synthetic_data/manifest.csv with revision_metrics/metrics.py and
write the per-trial CSVs, the summary JSON, and the LaTeX table bodies the draft \input s.

SYNTHETIC: the orbit runs and every four-robot run are fabricated (see Synthetic_data/README.md).
The three-robot attraction runs are real copies of the old logs, which the 2026-11 campaign
replaces. When the campaign data arrive, point DATA at them (same layout) and rerun.

Run with the VF_Robot venv:
    trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3 \
        "Paper_Writing/Systems Paper Revision/Queued_Edits/V_VIA_synthetic_draft/scripts/analyze.py"
"""
import csv
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(DRAFT, "..", "..", "..", ".."))
DATA = os.path.join(REPO, "Synthetic_data")
RM = os.path.join(REPO, "trunk", "Python_Simulations", "Vector_Fields", "VF_Robot",
                  "experiments", "revision_metrics")
OUT = os.path.join(DRAFT, "results")
sys.path.insert(0, RM)
import hardware  # noqa: E402  (loader for the Simulink .mat format)
import metrics  # noqa: E402  (one set of metric definitions for hardware and simulation)

FIELDS = ("vortex", "saddle")
TEAMS = (3, 4)
EPS = metrics.EPS_CONV
# Practical radius range: the smallest r_d whose mean radial error is under this fraction of r_d.
RANGE_FRAC = 0.25


def manifest():
    with open(os.path.join(DATA, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def load(row):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return hardware.load(os.path.join(DATA, row["file"]))


def score():
    att, orb = [], []
    for row in manifest():
        kind = row["kind"]
        if kind.startswith("attraction"):
            d = load(row)
            m = metrics.attraction_trial(d["c"], d["phat"], robots=d["robots"])
            m.update(field=row["field"], n_robots=int(row["n_robots"]), origin=row["origin"],
                     file=row["file"], start=(int(os.path.basename(row["file"])[3:-4]) - 1) // 10 + 1)
            att.append(m)
        elif kind.startswith("orbit") and row["origin"] == "SYNTHETIC":
            d = load(row)
            m, _ = metrics.orbit_run(d["c"], d["phat"], float(row["rd"]), robots=d["robots"])
            m.update(field=row["field"], n_robots=int(row["n_robots"]), origin=row["origin"],
                     file=row["file"], trial=int(row["trial"]))
            orb.append(m)
    return att, orb


def write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize_attraction(att):
    s = {}
    for f in FIELDS:
        for n in TEAMS:
            g = [r for r in att if r["field"] == f and r["n_robots"] == n]
            if not g:
                continue
            e = np.array([r["e_final"] for r in g])
            fin = np.array([[r["final_x"], r["final_y"]] for r in g])
            mu = fin.mean(0)
            s[f"{f}_{n}R"] = dict(
                n=len(g), origin=g[0]["origin"],
                success={f"{round(x * 100)}cm": float(np.mean([not r[f"fail_{round(x * 100)}cm"] for r in g]))
                         for x in EPS},
                e_mean=float(e.mean()), e_sd=float(e.std(ddof=1)), e_median=float(np.median(e)),
                e_p95=float(np.percentile(e, 95)), e_max=float(e.max()),
                bias=float(np.hypot(*mu)), sigma=float(np.sqrt(np.mean(np.sum((fin - mu) ** 2, 1)))),
                t_settle_5cm_median=float(np.nanmedian([r["t_settle_5cm"] for r in g])),
                est_ss_mean=float(np.nanmean([r["est_ss"] for r in g])),
                L_path_mean=float(np.mean([r["L_path"] for r in g])),
                E_real_mean=float(np.mean([r["E_real"] for r in g])),
                left_ws=int(sum(r["left_ws"] for r in g)))
    return s


def summarize_orbit(orb):
    s = {}
    for f in FIELDS:
        for n in TEAMS:
            g = [r for r in orb if r["field"] == f and r["n_robots"] == n]
            rds = sorted({r["rd"] for r in g})
            rows = []
            for rd in rds:
                h = [r for r in g if r["rd"] == rd]
                b = np.array([r["bias_w1"] for r in h])
                rows.append(dict(rd=rd, n=len(h), bias=float(b.mean()), bias_between_sd=float(b.std(ddof=1)),
                                 sd_within=float(np.mean([r["sd_within_w1"] for r in h])),
                                 rmse=float(np.mean([r["rmse_w1"] for r in h])),
                                 ctrl_bias=float(np.nanmean([r["ctrl_bias_w1"] for r in h])),
                                 est_err=float(np.nanmean([r["est_err_w1"] for r in h])),
                                 revs=float(np.mean([r["revs_w1"] for r in h])),
                                 E_per_orbit=float(np.nanmean([r.get("E_real_per_orbit", np.nan) for r in h]))))
            ok = [r["rd"] for r in rows if abs(r["bias"]) <= RANGE_FRAC * r["rd"]]
            fails = [r["rd"] for r in g if r["left_ws"]]
            s[f"{f}_{n}R"] = dict(rows=rows, min_practical_rd=min(ok) if ok else None, n_runs=len(g),
                                  n_fail=len(fails), min_fail_rd=min(fails) if fails else None)
    return s


def tex_tables(sa, so):
    """Table bodies only. Captions and headers live in the draft so the prose stays editable."""
    lines = []
    for f in FIELDS:
        for n in TEAMS:
            k = f"{f}_{n}R"
            if k not in sa:
                continue
            a = sa[k]
            tag = "" if a["origin"] == "real" else "$^\\dagger$"
            lines.append(f"{f.capitalize()} {n}R{tag} & {a['n']} & "
                         + " & ".join(f"{100 * a['success'][t]:.0f}" for t in ("3cm", "4cm", "5cm"))
                         + f" & {1000 * a['e_mean']:.0f}$\\pm${1000 * a['e_sd']:.0f} & {1000 * a['e_p95']:.0f}"
                         + f" & {1000 * a['bias']:.0f} & {1000 * a['sigma']:.0f}"
                         + f" & {a['t_settle_5cm_median']:.1f} & {1000 * a['est_ss_mean']:.0f}"
                         + f" & {a['L_path_mean']:.2f} \\\\")
    with open(os.path.join(OUT, "tab_attraction_body.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = []
    rds = [r["rd"] for r in so["vortex_3R"]["rows"]]
    for i, rd in enumerate(rds):
        cells = [f"{rd:.2f}"]
        for f in FIELDS:
            r = so[f"{f}_3R"]["rows"][i]
            cells.append(f"{r['bias']:+.3f} & {r['bias_between_sd']:.3f} & {r['sd_within']:.3f}")
        r4 = [so[f"{f}_4R"]["rows"][i]["bias"] for f in FIELDS]
        cells.append(f"{r4[0]:+.3f} & {r4[1]:+.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    with open(os.path.join(OUT, "tab_orbit_body.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def tex_numbers(sa, so):
    """Every number the draft prose quotes, as \\newcommand macros, so the text updates with the data.
    Names: field (Vtx, Sad) + team (Three, Four) + quantity. Lengths in mm for attraction, m for orbit."""
    F = {"vortex": "Vtx", "saddle": "Sad"}
    T = {3: "Three", 4: "Four"}
    m = {}
    for f in FIELDS:
        for n in TEAMS:
            a = sa.get(f"{f}_{n}R")
            if a is None:
                continue
            p = F[f] + T[n]
            m[p + "N"] = f"{a['n']}"
            for t, w in (("3cm", "SuccThree"), ("4cm", "SuccFour"), ("5cm", "SuccFive")):
                m[p + w] = f"{100 * a['success'][t]:.0f}"
            m[p + "Emean"] = f"{1000 * a['e_mean']:.0f}"
            m[p + "Esd"] = f"{1000 * a['e_sd']:.0f}"
            m[p + "Epct"] = f"{1000 * a['e_p95']:.0f}"
            m[p + "Bias"] = f"{1000 * a['bias']:.0f}"
            m[p + "Sigma"] = f"{1000 * a['sigma']:.0f}"
            m[p + "Tset"] = f"{a['t_settle_5cm_median']:.1f}"
            m[p + "Loc"] = f"{1000 * a['est_ss_mean']:.0f}"
            m[p + "Path"] = f"{a['L_path_mean']:.2f}"
            o = so[f"{f}_{n}R"]["rows"]
            q = F[f] + "Orb" + T[n]
            m[q + "BiasSmall"] = f"{o[0]['bias']:.3f}"
            m[q + "BiasLarge"] = f"{o[-1]['bias']:.3f}"
            m[q + "RdSmall"] = f"{o[0]['rd']:.2f}"
            m[q + "RdLarge"] = f"{o[-1]['rd']:.2f}"
            m[q + "CtrlSmall"] = f"{o[0]['ctrl_bias']:.3f}"
            m[q + "BetweenMin"] = f"{min(r['bias_between_sd'] for r in o):.3f}"
            m[q + "BetweenMax"] = f"{max(r['bias_between_sd'] for r in o):.3f}"
            m[q + "WithinMin"] = f"{min(r['sd_within'] for r in o):.3f}"
            m[q + "WithinMax"] = f"{max(r['sd_within'] for r in o):.3f}"
            m[q + "LocSmall"] = f"{o[0]['est_err']:.3f}"
            m[q + "LocLarge"] = f"{o[-1]['est_err']:.3f}"
            m[q + "Range"] = f"{so[f'{f}_{n}R']['min_practical_rd']:.2f}"
            m[q + "Trials"] = f"{o[0]['n']}"
            m[q + "Radii"] = f"{len(o)}"
            m[q + "Fail"] = f"{so[f'{f}_{n}R']['n_fail']}"
            m[q + "Runs"] = f"{so[f'{f}_{n}R']['n_runs']}"
            mf = so[f"{f}_{n}R"]["min_fail_rd"]
            m[q + "FailRd"] = "--" if mf is None else f"{mf:.2f}"
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("% Written by scripts/analyze.py. Do not edit by hand.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{\\sn{{{v}}}}}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    att, orb = score()
    write_csv(att, os.path.join(OUT, "attraction_trials.csv"))
    write_csv(orb, os.path.join(OUT, "orbit_runs.csv"))
    sa, so = summarize_attraction(att), summarize_orbit(orb)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(dict(attraction=sa, orbit=so, eps=EPS, range_frac=RANGE_FRAC), fh, indent=1)
    tex_tables(sa, so)
    tex_numbers(sa, so)
    for k, a in sa.items():
        print(f"{k:10s} n={a['n']:3d} ({a['origin']}) success 3/4/5cm "
              + "/".join(f"{100 * v:.0f}" for v in a["success"].values())
              + f"  e={1000 * a['e_mean']:.0f}+-{1000 * a['e_sd']:.0f} mm  p95={1000 * a['e_p95']:.0f}"
              + f"  bias={1000 * a['bias']:.0f} sigma={1000 * a['sigma']:.0f}  t5={a['t_settle_5cm_median']:.1f}s"
              + f"  est={1000 * a['est_ss_mean']:.0f} mm  L={a['L_path_mean']:.2f} m")
    for k, o in so.items():
        print(k, "min practical r_d", o["min_practical_rd"], " ".join(
            f"[{r['rd']:.2f}: {r['bias']:+.3f}+-{r['bias_between_sd']:.3f} w{r['sd_within']:.3f} "
            f"cb{r['ctrl_bias']:+.3f} est{r['est_err']:.3f}]" for r in o["rows"]))


if __name__ == "__main__":
    main()
