"""
Simulation Table II of the vector field revision (Section IV restructure, 2026-11-04):
ten fields as columns in one single-column table, blocks (a) attraction, (b) orbit
radial bias, (c) classification, laid out like the hardware Table IV.
Writes tables/table_sim_merged.tex (the full table environment). The paper copy is
pasted from it.

Sources
    eight linear and reconstructed fields: results/summary.json (run_all.py) and
        results/triangle_summary.json (triangle_rescore.py), the numbers of the
        revision's former Tables II and III
    three nonlinear fields: results/nonlinear_summary.json (nonlinear.py)
Nonlinear cells are wrapped in \\newnum{} until reviewed.

Attraction accuracy rows use the trials that succeeded (all of them in the eight
single-critical-point fields). The success row's brackets are the starts with all
three robots inside the det J = 0 diamond. Orbit cells are the radial bias over the
runs kept; a dash where fewer than half were kept, superscript a where 50-99 %.
The within-run std and the controller-only bias (Ctrl.) are no longer tabulated
(layout experiment, 2026-11-04); they stay in summary.json and nonlinear_summary.json.

Run from VF_Robot:
    venv/bin/python3 experiments/revision_metrics/table_merged.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables", "table_sim_merged.tex")
LIN = ["sink", "source", "vortex", "sinking_vortex", "spewing_vortex", "saddle"]
NL = ["gyre", "curved_saddle_irrot", "curved_saddle_rot"]
OLD = ["sink", "vortex_nn", "saddle_nn"]


def num(x, nd=4):
    if x is None or not np.isfinite(x):
        return "--"
    if 0 <= x < 0.5 * 10 ** -nd:
        return "$<$0." + "0" * (nd - 1) + "1"
    return f"{x:.{nd}f}"


def sgn(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def pct(x):
    if x is None:
        return "--"
    return f"{x:.1f}" if 99.5 <= x < 100 else f"{x:.0f}"


def new(s):
    return f"\\newnum{{{s}}}"


def build():
    S = json.load(open(os.path.join(RES, "summary.json")))
    T = json.load(open(os.path.join(RES, "triangle_summary.json")))["simulation"]["attraction"]
    N = json.load(open(os.path.join(RES, "nonlinear_summary.json")))
    A, O, C = S["sim"]["attraction"], S["sim"]["orbit"], S["classification"]["sim"]
    NA = {f: N["attraction"][f] for f in NL}
    acc = [A[f] for f in OLD] + [NA[f]["success"] for f in NL]
    isnew = [False] * 3 + [True] * 3

    def cells(fn):
        return [new(fn(s)) if n else fn(s) for s, n in zip(acc, isnew)]

    att = {}
    att["Succ.\\ triangle (\\%)"] = [pct(T[f]["tri_success_pct"]) for f in OLD] + [
        new(pct(NA[f]["all"]["tri_success"]) if NA[f]["out_region"]["n"] == 0 else
            f"{pct(NA[f]['all']['tri_success'])} [{pct(NA[f]['formation_in_region']['tri_success'])}]") for f in NL]
    att["Succ.\\ any c.p.\\ (\\%)"] = [pct(T[f]["tri_success_pct"]) for f in OLD] + [new(pct(NA[f]["all"]["any_tri_success"])) for f in NL]
    att["Bias (m)"] = cells(lambda s: num(s["bias"]))
    att["$\\sigma$ (m)"] = cells(lambda s: num(s["scatter"]))
    att["$\\bar{e}_f$ (m)"] = cells(lambda s: num(s["e_mean"]))
    att["$s_e$ (m)"] = cells(lambda s: num(s["e_sd"]))
    att["Median (m)"] = cells(lambda s: num(s["e_med"]))
    att["p95 (m)"] = cells(lambda s: num(s["e_p95"]))
    att["Loc.\\ error (m)"] = cells(lambda s: num(s["est_ss_mean"]))
    att["$T_{\\mathrm{conv}}$ (s)"] = cells(lambda s: f"{s['t_settle_med_5cm']:.1f}")
    att["$E_u$ (m$^2$/s)"] = cells(lambda s: f"{s['E_real_mean']:.3f}")
    for t in (3, 4, 5):
        att[f"Succ.\\ {t}~cm (\\%)"] = cells(lambda s, t=t: f"{s[f'success_{t}cm']:.0f}")

    orb, partial = {}, []
    for rd in sorted(O["sink"], key=float):
        c = [sgn(np.mean([O[f][rd]["bias_w1"] for f in LIN]))] + [sgn(O[f][rd]["bias_w1"]) for f in ("vortex_nn", "saddle_nn")]
        for f in NL:
            o = N["orbit"][f][rd]
            if o["kept_pct"] < 50:
                c.append(new("--"))
            else:
                if o["kept_pct"] < 100:
                    partial.append(o["kept_pct"])
                c.append(new(sgn(o["bias_kept"]) + ("$^{a}$" if o["kept_pct"] < 100 else "")))
        orb[f"\\quad {float(rd):.2f}"] = c
    assert len(set(partial)) <= 1, partial

    nf, ns = list(C)
    lin_n = [100 * C[ns][f]["attract"]["acc"] for f in LIN]
    cls = {"No noise": ["100"] + [pct(100 * C[nf][f]["attract"]["acc"]) for f in ("vortex_nn", "saddle_nn")]
           + [new(pct(NA[f]["success"]["cls_target"])) for f in NL],
           f"Noise {S['classification']['sigma_noise']}~m/s": [f"{pct(min(lin_n))}--{pct(max(lin_n))}"]
           + [pct(100 * C[ns][f]["attract"]["acc"]) for f in ("vortex_nn", "saddle_nn")]
           + [new(pct(NA[f]["success"]["cls_target_noise"])) for f in NL]}

    def rows(d):
        return "\n".join(f"{k} & " + " & ".join(v) + r" \\" for k, v in d.items())

    def blk(t):
        return f"\\multicolumn{{7}}{{l}}{{\\textit{{{t}}}}} \\\\\n\\hline\n"

    lost = f"{100 - partial[0]:.0f}" if partial else ""
    return ("\\begin{table}[!t]\n\\scriptsize\n\\centering\n"
            "\\caption{Simulation results on ten fields. Brackets give starts with all three robots inside the "
            "$\\det\\mathbf{J} = 0$ diamond, and the accuracy rows use the trials that succeeded. A dash marks a radius "
            f"at which most orbits were lost, and $^{{a}}$ one at which \\newnum{{{lost}}}\\% were.}}\n"
            "\\label{tab:simulation_results}\n\\setlength{\\tabcolsep}{1.5pt}\n"
            "\\rev{\\begin{tabular}{l|c|cc|ccc}\n\\hline\n"
            r" & \textbf{Anal.} & \multicolumn{2}{c|}{\textbf{Recon.}} & \multicolumn{3}{c}{\textbf{Nonlinear}} \\" "\n"
            r" & \textbf{(6)} & \textbf{Vort.} & \textbf{Sadd.} & \textbf{Gyre} & \textbf{Irrot.} & \textbf{Rot.} \\" "\n\\hline\n"
            + blk("(a) Attraction") + rows(att) + "\n\\hline\n"
            + blk(r"(b) Orbit radial bias (m), by $r_d$ (m)") + rows(orb) + "\n\\hline\n"
            + blk(r"(c) Classification (\%)") + rows(cls) + "\n\\hline\n\\end{tabular}}"
            " % [REVIEW] experiments/revision_metrics/table_merged.py -> tables/table_sim_merged.tex. Eight-field cells: "
            "summary.json and triangle_summary.json (former Tables II and III). Nonlinear cells: nonlinear_summary.json "
            "(nonlinear.py, Coulomb model, seed 0; noise seed 1)\n\\end{table}\n")


def main():
    tex = build()
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
