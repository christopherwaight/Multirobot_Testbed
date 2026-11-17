"""Apply the numeric edits to the revision tex, and write the numbers map.

Scope (2026-11-16 answer): tables, figures and numbers only. A number whose sentence still holds is
replaced and marked \\rev{} (plan answer, 2026-11-16). A sentence whose claim changed is left as is and
listed in queued_prose_batch.md with its new numbers. Every edit is matched on exact text first;
any mismatch aborts the whole batch and nothing is written.

Run from the repo root:  python3 analysis/paper/apply_paper_edits.py [--dry-run]
"""
import csv
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402

TEX = os.path.join(paths.PAPER, "cwaight_systems_paper_with_revisions2.tex")
SIMTAB = os.path.join(paths.OUT_SIM, "revision_metrics", "tables")
HWOUT = "analysis/hardware/outputs (score_hardware.py)"
SIMRES = "analysis/simulation/revision_metrics/results"

# marker index (document order, extract_markers.py) -> (expected old value, new value, source); None = keep
EDITS = {
    1: ("100", "99", f"{HWOUT}: saddle 3R success 5 cm, 80 runs incl. extras (100 without)"),
    2: ("86", "91", f"{HWOUT}: vortex 3R success 5 cm"),
    3: ("158", "160", f"{HWOUT}: 3R attraction runs (80 vortex + 80 saddle)"),
    4: ("0.037", "0.034", f"{HWOUT}: largest 3R mean final error (vortex 0.034)"),
    5: ("36", "120", f"{HWOUT}: 2026 3R orbit runs (abstract reports 3R; 4R sentence queued)"),
    6: ("six", "six", "2026 orbit radii (3R)"),
    7: ("0.40", "0.20", f"{HWOUT}: vortex 3R practical range (|bias| <= 0.25 r_d)"),
    8: ("0.50", "0.20", f"{HWOUT}: saddle 3R practical range"),
    9: ("campaign", "459", "239 attraction (2025 3R 160 + 4R 79) + 220 orbit (2026) runs"),
    14: ("1.7", "3.0", "extra_numbers.json noise: reconstructed fields, robot ideal, sigma <= 0.03 m/s"),
    78: ("5.7", "4.3", f"{SIMRES}/nonlinear_summary.json: gyre loc. error, successful trials (Table II)"),
    79: ("0.16", "0.15", "extra_numbers.json gyre_size_loc_mm 0.10"),
    86: ("0.044", "0.064", f"{SIMRES}/nonlinear_summary.json: gyre orbit bias at r_d 0.3 (-0.064)"),
    88: ("0.02", "0.01", "nonlinear sweep: irrotational saddle spread across chi, r_d <= 0.3"),
    89: ("90", "86", "summary.json classification, vortex (center), noise 0.02 m/s"),
    90: ("95", "94", "summary.json classification, sink and source, noise 0.02 m/s"),
    92: ("86", "83", "table_sim_merged.tex classification, gyre, noise"),
    93: ("20", "10", "2025 attraction runs: 101 samples at 10 Hz"),
    94: ("0.33 or 0.30", "0.30", "run_config / 2025 logged p, q"),
    95: ("six", "six", "2026 orbit radii (3R; 4R five)"),
    96: ("three", "ten", "2026 orbit runs per radius"),
    97: ("1800", "200 to 300", "2026 orbit runs: 20 to 30 s"),
    98: ("80", "80", f"{HWOUT}"),
    99: ("78", "80", f"{HWOUT}: 78 runs of 2025 + extra1, extra2 (2026, Extras/HANDLING_NOTES.txt)"),
    100: ("100", "99", f"{HWOUT}: saddle 3R success 5 cm"),
    101: ("86", "91", f"{HWOUT}: vortex 3R success 5 cm"),
    102: ("77", "80", f"{HWOUT}: saddle 3R success 3 cm"),
    103: ("34", "39", f"{HWOUT}: vortex 3R success 3 cm"),
    104: ("100", "100", f"{HWOUT}: p* inside the final triangle"),
    105: ("0.018", "0.007", f"{HWOUT}: vortex 3R bias"),
    106: ("0.036", "0.036", f"{HWOUT}: vortex 3R precision"),
    107: ("0.007", "0.002", f"{HWOUT}: saddle 3R bias"),
    108: ("0.024", "0.025", f"{HWOUT}: saddle 3R precision"),
    110: ("campaign", "100", f"{HWOUT}: 2026 saddle orbits, cycles labeled saddle"),
    116: ("1800", "200 to 300", "2026 orbit runs: 20 to 30 s"),
    117: ("0.166", "0.060", f"{HWOUT}: vortex 3R bias r_d 0.01"),
    118: ("0.018", "0.020", f"{HWOUT}: vortex 3R bias r_d 0.50"),
    119: ("0.027", "0.014", f"{HWOUT}: vortex 3R s_w min"),
    120: ("0.119", "0.076", f"{HWOUT}: vortex 3R s_w max"),
    121: ("0.111", "0.066", f"{HWOUT}: saddle 3R bias r_d 0.01"),
    122: ("0.088", "0.005", f"{HWOUT}: saddle 3R bias r_d 0.50"),
    123: ("0.018", "0.012", f"{HWOUT}: saddle 3R s_w min"),
    124: ("0.095", "0.083", f"{HWOUT}: saddle 3R s_w max"),
    125: ("0.10", "0.10", f"{HWOUT}: s_w > s_b from r_d 0.10 in every column"),
    126: ("0.40", "0.20", f"{HWOUT}: vortex 3R practical range"),
    127: ("0.50", "0.20", f"{HWOUT}: saddle 3R practical range"),
    137: ("$2\\times10^{-5}$", "$2\\times10^{-4}$", f"{HWOUT}/parked_noise.csv: extras, final 5 s (mean 2.15e-4 m/s)"),
    138: ("campaign block E", None, "marker removed; source in REPORT.md"),
    139: ("0.177", "0.071", f"{HWOUT}: vortex 3R controller-only bias r_d 0.01"),
    140: ("0.166", "0.060", f"{HWOUT}: vortex 3R bias r_d 0.01"),
    145: ("0.031", "0.003", f"{SIMRES}/baselines_v2s_summary.json linear v2s overshoot_med"),
    146: ("1.9", "1.6", "baselines_v2s: E_u V2S / Jacobian = 0.106 / 0.067"),
    147: ("0.002", "0.004", "baselines_v2s: max |RMSE(V2S + oracle) - RMSE(Jacobian)| over r_d"),
    148: ("0.12", "0.21", "baselines_v2s linear v2s_est final_speed"),
    149: ("7", "6", "baselines_v2s node_aniso v2s success"),
    150: ("0.67", "0.64", "extra_numbers.json node_aniso v2s failures, median distance"),
    151: ("0.46", "0.50", "baselines_v2s node_aniso v2s_est est_orb"),
    152: ("96", "97", "baselines_v2s gyre_same_point_as_jacobian v2s"),
    156: ("158", "160", f"{HWOUT}"),
    157: ("100", "99", f"{HWOUT}"),
    158: ("86", "91", f"{HWOUT}"),
    159: ("0.037", "0.034", f"{HWOUT}"),
    160: ("0.40", "0.20", f"{HWOUT}"),
    161: ("0.50", "0.20", f"{HWOUT}"),
}
PROSE = {   # marker index -> why the sentence goes to the queued batch (new numbers in queued_prose_batch.md)
    10: "Jury limit changes form under the delay model (k_max = 1/(g L) = 21.1 /s; k dt limit 2.11)",
    11: "pole magnitude at k = 1: 0.91 and 0.05 (was sqrt(alpha) = 0.84)",
    13: "every reconstructed-field trial now ends within 0.05 m ('at least 95%' would read 'at least 100%')",
    109: "vortex: strict center label 45% (the center sits on the class boundary), rotational class 99.9%; needs a sentence",
    15: "covariance prediction (error note) not regenerated with the delay model",
    84: "practical range is r_d >= 0.10 m on every simulated field; the sentence's split collapses",
    85: "as 84",
    128: "no orbit was lost (none left the map); the 'One vortex and three saddle' sentence no longer holds",
    129: "as 128",
    130: "as 128",
    131: "orbit closed form changes (V = g k_t, T = L + tau + dt/2: within 3 mm; fitted T 0.111 s: 0.6 mm)",
    132: "r_min = 0.047 m (sim; formula 0.049)",
    133: "error < 0.05 m at every r_d now",
    134: "k_t 0.1 -> r_min 0.031 m",
    135: "removing delay and lag -> 0.032 m",
    136: "k_r 2 -> 0.034 m",
    141: "claim reversed: reconstructed vortex bias +0.027 now exceeds controller-only +0.013 at r_d 0.5",
    142: "as 141",
    143: "saddle: bias +0.019 vs controller-only -0.003",
    144: "as 143",
}

STRINGS = [   # unmarked protocol numbers that changed: (exact old text, new text, source)
    ("in an equilateral triangular formation with a 0.33~m side length",
     "in an equilateral triangular formation with a \\rev{0.30}~m side length", "D8: 2026 hardware p = q = 0.30 m"),
    ("and $k_t = 0.2$~m/s, $k_r = 0.5$~s$^{-1}$ in (\\ref{eq:11})",
     "and $k_t = \\rev{0.15}$~m/s, $k_r = \\rev{1}$~s$^{-1}$ in (\\ref{eq:11})", "D2: hardware orbit gains"),
    ("and the commanded centroid speed was limited to 0.5~m/s.",
     "and the commanded centroid velocity was limited to \\rev{0.3}~m/s on each axis.", "2026 hardware per-axis clip"),
    ("its radial error $\\|\\mathbf{p}_c - \\mathbf{p}^*\\| - r_d$ after the first 10~s.",
     "its radial error $\\|\\mathbf{p}_c - \\mathbf{p}^*\\| - r_d$ after the first \\rev{3}~s.", "D11: transient trim 3 s"),
    ("turns less than a quarter revolution about $\\mathbf{p}^*$ after the first 10~s,",
     "turns less than a quarter revolution about $\\mathbf{p}^*$ after the first \\rev{3}~s,", "D11"),
    ("then the orbital controller (\\ref{eq:11}) was used for 600 time steps.",
     "then the orbital controller (\\ref{eq:11}) was used for \\rev{300} time steps.", "sim.py run_orbit T = 30 s"),
    ("after discarding the first 100 time steps (10~s) as a settling transient.",
     "after discarding the first \\rev{30} time steps (\\rev{3}~s) as a settling transient.", "D11"),
    ("were recorded after the first 10~s, as in simulation",
     "were recorded after the first \\rev{3}~s, as in simulation", "D11"),
    ("\\caption{\\syn{[Placeholder]} Convergence trajectories", "\\caption{Convergence trajectories", "hardware figure"),
    ("placeholder_hw_trajectories_a.png", "fig_hw_trajectories_vortex.pdf", "analysis/hardware/figures"),
    ("placeholder_hw_trajectories_b.png", "fig_hw_trajectories_saddle.pdf", "analysis/hardware/figures"),
    ("placeholder_hw_distance.png", "fig_hw_distance.pdf", "analysis/hardware/figures"),
    ("{fig_reconstruction_3x2.png}", "{fig_reconstruction_3x2_2026.png}", "analysis/paper/fig_reconstruction.py"),
    ("{fig_sim_convergence_band.pdf}", "{fig_sim_convergence_band_2026.pdf}", "revision_metrics/fig_sim_convergence.py"),
    ("{fig_nonlinear_triangles.pdf}", "{fig_nonlinear_triangles_2026.pdf}", "revision_metrics/fig_nonlinear.py"),
    ("{fig_v2s_effort.pdf}", "{fig_v2s_effort_2026.pdf}", "revision_metrics/fig_v2s_effort.py (draft A)"),
    ("with the solid line showing the mean path and the shaded region one standard deviation across the 10 repetitions. (c) Distance to center over time, with mean and standard deviation bounds.}",
     "with thin lines showing each run and the thick line the mean path from each start. (c) Distance to $\\mathbf{p}^*$ over time, mean and one standard deviation across runs, for three robots in both fields and four robots in the vortex.}",
     "figures_hardware.py: no shaded band; (c) adds vortex 4R"),
    (", $^{a}$ one where \\newnum{48}\\% were.}", ".}", "no radius now has about half its orbits lost below the dash threshold"),
]
FIGS = {   # source -> name in the paper folder
    os.path.join(paths.OUT_HW, "figures", "fig_hw_trajectories_vortex.pdf"): "fig_hw_trajectories_vortex.pdf",
    os.path.join(paths.OUT_HW, "figures", "fig_hw_trajectories_saddle.pdf"): "fig_hw_trajectories_saddle.pdf",
    os.path.join(paths.OUT_HW, "figures", "fig_hw_distance.pdf"): "fig_hw_distance.pdf",
    os.path.join(HERE, "fig_reconstruction_3x2.png"): "fig_reconstruction_3x2_2026.png",
    os.path.join(paths.OUT_SIM, "revision_metrics", "figures", "fig_sim_convergence_band.pdf"): "fig_sim_convergence_band_2026.pdf",
    os.path.join(paths.OUT_SIM, "revision_metrics", "figures", "fig_nonlinear_triangles.pdf"): "fig_nonlinear_triangles_2026.pdf",
    os.path.join(paths.OUT_SIM, "revision_metrics", "figures", "fig_v2s_effort_A.pdf"): "fig_v2s_effort_2026.pdf",
}


def balanced(s, i):
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError("unbalanced")


def tabular_body(path):
    s = open(path).read()
    i = s.index("\\begin{tabular}")
    j = s.index("\\end{tabular}") + len("\\end{tabular}")
    return s[i:j]


def sim_table():
    """table_sim_merged.tex body in the paper's layout: inner \\newnum removed, 3/4/5 cm rows merged."""
    b = tabular_body(os.path.join(SIMTAB, "table_sim_merged.tex"))
    b = re.sub(r"\\newnum\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", b)
    rows = {k: re.search(rf"Succ\.\\ {k}~cm \(\\%\) & (.*?) \\\\", b).group(1).split(" & ") for k in (3, 4, 5)}
    merged = " & ".join("100" if a == b_ == c == "100" else f"{a}/{b_}/{c}" for a, b_, c in zip(rows[3], rows[4], rows[5]))
    b = re.sub(r"Succ\.\\ 3~cm \(\\%\) & .*? \\\\\nSucc\.\\ 4~cm \(\\%\) & .*? \\\\\nSucc\.\\ 5~cm \(\\%\) & .*? \\\\",
               lambda m: f"Succ.\\ 3/4/5~cm (\\%) & {merged} \\\\", b)
    assert "3/4/5" in b
    return b


def main(dry):
    s = open(TEX).read()
    original = s
    # 1. numeric markers, matched by index and old value
    spans = []
    for k, m in enumerate(re.finditer(r"\\(syn|rev|newnum|open)\{", s), 1):
        end = balanced(s, m.end() - 1)
        spans.append((k, m.group(1), m.start(), end, s[m.end():end - 1]))
    out, last, log = [], 0, []
    for k, kind, a, b, body in spans:
        if k in EDITS:
            old, new, src = EDITS[k]
            if body != old:
                sys.exit(f"ABORT: marker {k} reads {body!r}, expected {old!r}; file changed since extract_markers.py")
            out.append(s[last:a])
            out.append("" if new is None else f"\\rev{{{new}}}")
            last = b
            log.append(dict(n=k, marker=kind, old=old, new="" if new is None else new, action="edit", source=src))
        elif k in PROSE:
            log.append(dict(n=k, marker=kind, old=body, new="", action="queued prose", source=PROSE[k]))
        elif not body.lstrip().startswith("\\begin{tabular}"):
            log.append(dict(n=k, marker=kind, old=body if len(body) < 60 else body[:57] + "...", new="", action="unchanged", source=""))
    out.append(s[last:])
    s = "".join(out)
    if "\\open{campaign block E}" not in original:
        sys.exit("ABORT: expected the campaign block E marker")
    s = s.replace(" between cycles , while", " between cycles, while")
    # 2. tables
    for label, new_body, wrapper in (("tab:simulation_results", sim_table(), "rev"),
                                     ("tab:estimation_results", tabular_body(os.path.join(HERE, "table_hardware.tex")), "rev"),
                                     ("tab:baselines", tabular_body(os.path.join(SIMTAB, "table_v2s.tex")), "rev")):
        i = s.index(f"\\label{{{label}}}")
        j = re.compile(r"\\(syn|rev|newnum)\{\\begin\{tabular\}").search(s, i)
        end = balanced(s, j.end() - len("\\begin{tabular}") - 1)
        s = s[:j.start()] + f"\\{wrapper}{{{new_body}}}" + s[end:]
        log.append(dict(n="table", marker=label, old="(table)", new="(regenerated)", action="edit", source=label))
    # hardware table caption
    cap_old = re.search(r"\\caption\{Hardware experimental results \\syn\{\(placeholder data\)\}\. \(a\) Attraction, vortex: \\syn\{80\} runs, saddle: \\syn\{78\} runs\. Bias measures mean offset from origin;", s)
    if not cap_old:
        sys.exit("ABORT: hardware table caption not found")
    s = s[:cap_old.start()] + ("\\caption{Hardware experimental results, three robots (3R) and four (4R). (a) Attraction (2025 runs): "
                               "vortex 3R \\rev{80}, vortex 4R \\rev{79}, saddle 3R \\rev{80} runs, two of them 2026 runs filling "
                               "missing slots; no four-robot saddle attraction was run. Bias measures mean offset from $\\mathbf{p}^*$;") + s[cap_old.end():]
    s = s.replace("(b) Orbital control, radial errors in metres after the first 10~s. $\\bar e_r$ is the radial bias, $s_b$ the between-trial spread, and $s_w$ the within-run standard deviation.",
                  "(b), (c) Orbital control (2026 runs, ten per radius), radial errors in metres after the first \\rev{3}~s. $\\bar e_r$ is the radial bias, $s_b$ the between-trial spread, and $s_w$ the within-run standard deviation.")
    if "(b), (c) Orbital control" not in s:
        sys.exit("ABORT: hardware caption part (b) not found")
    # 3. exact strings
    for old, new, src in STRINGS:
        c = s.count(old)
        if c != 1:
            sys.exit(f"ABORT: {old[:60]!r} found {c} times")
        s = s.replace(old, new)
        log.append(dict(n="text", marker="", old=old[:80], new=new[:80], action="edit", source=src))
    # 4. write
    with open(os.path.join(HERE, "numbers_map.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "marker", "old", "new", "action", "source"])
        w.writeheader()
        w.writerows(log)
    if dry:
        print("dry run: all", len(log), "entries matched; nothing written")
        return
    shutil.copy2(TEX, os.path.join(HERE, "backup_cwaight_systems_paper_with_revisions2_before_2026-11-16.tex"))
    for src, name in FIGS.items():
        shutil.copy2(src, os.path.join(paths.PAPER, name))
    open(TEX, "w").write(s)
    print(f"applied: {sum(1 for r in log if r['action'] == 'edit')} edits; queued prose: "
          f"{sum(1 for r in log if r['action'] == 'queued prose')}; backup and numbers_map.csv in analysis/paper/")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
