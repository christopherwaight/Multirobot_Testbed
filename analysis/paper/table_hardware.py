"""Hardware table for the paper (tab:estimation_results), 3R and 4R, from analysis/hardware/outputs/.

(a) Attraction (2025 runs; vortex 4R = Oct 2025 least squares; no saddle 4R attraction data), with the
    mean final error of the matched simulations (each run re-simulated from its start, linear/reconstructed
    field; analysis/simulation/paired/).
(b) Orbit radial bias e_r +- between-run spread s_b, (c) within-run spread s_w (2026 runs, 3 s
transient excluded). (d) Classification of J-hat on the orbit runs (% of cycles).
Writes analysis/paper/table_hardware.tex and prints it.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402

HW = os.path.join(paths.OUT_HW, "outputs")
CLASSIFICATION_ROWS = False      # (d) dropped 2026-11-17 for length; the numbers are in Sec. V-D


def f(x, nd=3, signed=False):
    if x is None or pd.isna(x):
        return "--"
    if signed and round(x, nd) == 0:
        signed = False
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if signed else s


def main():
    H = json.load(open(os.path.join(HW, "hardware_summary.json")))
    R, CL = H["attraction"], H["classification_2026_orbits"]
    PA = pd.read_csv(os.path.join(paths.OUT_SIM, "paired", "paired_attraction_summary.csv")).set_index(["field", "nr"])
    SO = pd.read_csv(os.path.join(HW, "orbit_summary.csv"))
    cols = [("vortex_3R", "V 3R"), ("vortex_4R", "V 4R"), ("saddle_3R", "S 3R")]
    A = [R[k] for k, _ in cols]
    lines = [r"\begin{tabular}{l|cc|cc}", r"\hline",
             r" & \multicolumn{2}{c|}{\textbf{Vortex}} & \multicolumn{2}{c}{\textbf{Saddle}} \\",
             r" & \textbf{3R} & \textbf{4R} & \textbf{3R} & \textbf{4R} \\", r"\hline",
             r"\multicolumn{5}{l}{\textit{(a) Attraction}} \\", r"\hline"]
    rows = [("Runs", lambda s: f"{s['n']:.0f}"), ("Bias (m)", lambda s: f(s["bias"])),
            (r"$\sigma$ (m)", lambda s: f(s["scatter"])),
            (r"$\bar{e}_f \pm s_e$ (m)", lambda s: f"{f(s['e_mean'])}$\\pm${f(s['e_sd'])}"),
            ("Median [p95] (m)", lambda s: f"{f(s['e_med'])} [{f(s['e_p95'])}]"),
            (r"Loc.\ error (m)", lambda s: f(s["est_ss_mean"])),
            (r"$T_{\mathrm{conv}}$ (s)", lambda s: f"{s['t_conv_med']:.1f}"),
            (r"$E_u$ (m$^2$/s)", lambda s: f(s["E_real_mean"])),
            (r"Succ.\ 3/4/5 cm (\%)", lambda s: f"{s['success_3cm']:.0f}/{s['success_4cm']:.0f}/{s['success_5cm']:.0f}")]
    for name, fn in rows:
        v = [fn(s) for s in A]
        lines.append(f"{name} & {v[0]} & {v[1]} & {v[2]} & -- \\\\")
    sim = [PA.loc[(fld, nr)] for fld, nr in (("vortex", 3), ("vortex", 4), ("saddle", 3))]
    v = [f"{f(p.lin_e)}/{f(p.rec_e)}" for p in sim]
    lines.append(f"Sim.\\ $\\bar{{e}}_f$ anal./rec.\\ (m) & {v[0]} & {v[1]} & {v[2]} & -- \\\\")
    lines += [r"\hline", r"\multicolumn{5}{l}{\textit{(b) Orbit radial bias $\bar e_r \pm s_b$ (m), by $r_d$ (m)}} \\", r"\hline"]
    order = [("vortex", 3), ("vortex", 4), ("saddle", 3), ("saddle", 4)]
    for rd in sorted(SO.rd.unique()):
        cells = []
        for fld, nr in order:
            g = SO[(SO.field == fld) & (SO.nr == nr) & (SO.rd == rd)]
            cells.append("--" if g.empty else f"{f(g.bias.iloc[0], signed=True)}$\\pm${f(g.s_b.iloc[0])}")
        lines.append(f"\\quad {rd:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\multicolumn{5}{l}{\textit{(c) Orbit within-run spread $s_w$ (m)}} \\", r"\hline"]
    for rd in sorted(SO.rd.unique()):
        cells = []
        for fld, nr in order:
            g = SO[(SO.field == fld) & (SO.nr == nr) & (SO.rd == rd)]
            cells.append("--" if g.empty else f(g.s_w.iloc[0]))
        lines.append(f"\\quad {rd:.2f} & " + " & ".join(cells) + r" \\")
    if not CLASSIFICATION_ROWS:
        lines += [r"\hline", r"\end{tabular}"]
        tex = "\n".join(lines)
        open(os.path.join(HERE, "table_hardware.tex"), "w").write(tex + "\n")
        print(tex)
        return
    lines += [r"\hline", r"\multicolumn{5}{l}{\textit{(d) Orbit classification (\%)}} \\", r"\hline"]
    keys = ("vortex_3R", "vortex_4R", "saddle_3R", "saddle_4R")
    lines.append("Type correct & " + " & ".join(f"{CL[k]['strict']:.1f}" for k in keys) + r" \\")
    lines.append("Rotational & " + " & ".join(f"{CL[k]['rotational']:.1f}" if k.startswith("vortex") else "--" for k in keys) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    tex = "\n".join(lines)
    open(os.path.join(HERE, "table_hardware.tex"), "w").write(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
