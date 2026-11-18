"""Round 2 of the revision edits (2026-11-17): four-robot results in the prose, matched simulations
(new Fig. sim_vs_hw and Table III rows), the limit cycle moved to its own Sec. VI subsection, the
robot-model fit quality, abstract, introduction and conclusion counts, restyled figures, and the letter.

Every edit matches exact text and the run aborts before writing anything if one fails.
Backups: analysis/paper/backup_*_before_round2.tex (made once).
Usage: python3 analysis/paper/apply_round2.py [--dry-run]
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REV = os.path.join(ROOT, "Paper_Writing", "Systems Paper Revision")
PAPER = os.path.join(REV, "cwaight_systems_paper")
TEX = os.path.join(PAPER, "cwaight_systems_paper_with_revisions2.tex")
LETTER = os.path.join(REV, "Response_to_Reviewers.tex")

FIGS = {os.path.join(ROOT, "analysis", "hardware", "figures", "fig_hw_attraction.pdf"): "fig_hw_attraction_2026.pdf",
        os.path.join(ROOT, "analysis", "simulation", "figures", "fig_sim_vs_hw.pdf"): "fig_sim_vs_hw_2026.pdf",
        os.path.join(ROOT, "analysis", "limit_cycle", "figures", "fig_limit_cycle.pdf"): "fig_limit_cycle_2026.pdf"}

LIMIT_CYCLE_OLD_PARA = r"""At $r_d = 0.10$~m in the vortex the cluster does not hold the commanded circle but settles onto a closed curve of its own (Fig.~\ref{fig:limit_cycle}). Ten hardware runs, four started inside the curve and six outside, converged to the same curve, with a mean radius of \rev{0.127}~m that varies by \rev{0.043}~m around the loop and \rev{2.9}~mm between runs. On the reconstructed vortex in simulation, starts inside and outside meet on a curve of mean radius \rev{0.111}~m.

"""
LIMIT_CYCLE_FIG = r"""\begin{figure*}[!t]
    \centering
    \includegraphics[width=0.95\textwidth]{fig_limit_cycle_2026.pdf}
    \caption{Limit cycle in the vortex at $r_d = 0.10$~m (dotted circle). (a) Simulation on the reconstructed vortex from starts inside (blue) and outside (red) the cycle, which meet on one closed curve (black). (b) Ten hardware runs overlaid, with the hardware cycle (black) and the simulated cycle (dashed). (c) Distance to $\mathbf{p}^*$ over time for the simulated starts and the hardware runs.}
    \label{fig:limit_cycle}
\end{figure*}

"""

SIM_VS_HW_PARA = r"""The simulations matched to each hardware run (Section~V-B) show how much of the hardware error the model and the reconstructed fields reproduce (Fig.~\ref{fig:sim_vs_hw}). In attraction they reproduce the saddle, a mean final error of \rev{0.026}~m against \rev{0.023}~m on hardware, while the hardware vortex ends \rev{0.4} to \rev{1.5}~cm farther from $\mathbf{p}^*$ than its simulations (Table~\ref{tab:estimation_results}(a)). In orbit the hardware bias at $r_d = 0.01$~m exceeds the reconstructed-field simulations by \rev{14} to \rev{19}~mm. In these runs the live estimate ran \rev{1.5} to \rev{2}~cm ahead of the cluster along its motion, a lead whose cause is not identified (sensor delay was ruled out) and which the model omits; adding a 1.5~cm lead to the model brings its vortex orbits at $r_d = 0.01$ and 0.10~m to within \rev{7}~mm of the hardware. From $r_d = 0.30$~m the reconstructed fields produce \rev{27} to \rev{63}\% of the hardware within-run spread, and the analytical fields at most \rev{1}~mm, so the fixed distortion of the printed maps accounts for about half of that spread. The model also treats every robot alike, yet the fitted gains of the four robots range from \rev{0.80} to \rev{0.89}, and a robot whose readings were held out of the field reconstruction read the printed map with a bias of up to \rev{5}$^\circ$.

\begin{figure}[!t]
    \centering
    \includegraphics[width=\columnwidth]{fig_sim_vs_hw_2026.pdf}
    \caption{Hardware orbits against simulations matched to each run (Section~V-B), three robots (blue) and four (orange). (a) Radial bias, hardware mean and $\pm 1$ between-trial spread (markers), with the matched simulations on the reconstructed (solid) and analytical (dashed) field. (b) Within-run spread.}
    \label{fig:sim_vs_hw}
\end{figure}

"""

LIMIT_CYCLE_SECTION = r"""\subsection{The Orbit as a Limit Cycle}

Below the practical range the orbit does not converge to the commanded circle, yet it still converges to a closed curve of its own, a limit cycle of the closed loop. At $r_d = 0.10$~m in the vortex, ten three-robot hardware runs, four started inside the curve and six outside, converged to the same curve (Fig.~\ref{fig:limit_cycle}), with a mean radius of \rev{0.127}~m and a spread of \rev{2.9}~mm between runs. On the reconstructed vortex in simulation, starts inside and outside meet on a curve of mean radius \rev{0.111}~m. The closed form (\ref{eq:orbit_cf}) sets the size of this cycle: at $r_d = 0.10$~m it gives $r^* = \rev{0.120}$~m, and on the analytical fields the cluster holds that radius on a circle. The fixed distortion of the printed map bends the cycle out of round, so its radius varies around the loop by \rev{0.029}~m in simulation and \rev{0.043}~m on hardware.

""" + LIMIT_CYCLE_FIG

ABSTRACT_OLD = r"""    Simulation verifies both attraction and orbital control laws across six canonical linear vector fields and three nonlinear fields using three omnidirectional robots. Hardware validation within a lab-based experimental testbed demonstrates a \rev{99}\% (saddle) and \rev{91}\% (vortex) attraction convergence success rate within 0.05~m across \rev{160} trials, with mean position errors of \rev{0.034}~m or less from the true critical points. An additional \rev{120} orbital trials demonstrated bounded orbits at \rev{six} commanded radii across two fields, with radial bias below a quarter of the commanded radius at radii of \rev{0.20}~m and above in both fields. A four-robot formation converged within 0.05~m in all \rev{79} vortex trials.
    These findings establish this estimation method as a viable approach to navigation in vector fields such as oceanic and atmospheric flows, though the printed lab fields do not yet advect the robots as a physical flow would."""
ABSTRACT_NEW = r"""    Simulation verifies both laws on six canonical linear fields and three nonlinear fields. In \rev{459} hardware runs with three and four robots, the cluster converged within 0.05~m in \rev{99}\% (saddle) and \rev{91}\% (vortex) of three-robot trials and in all four-robot trials, with mean errors of \rev{0.034}~m or less. Three-robot orbits held radial bias below a quarter of the commanded radius from \rev{0.20}~m upward; at 0.10~m in the vortex they settled onto a limit cycle sized by the actuator delay and lag.
    These findings establish this estimation method as a viable approach to navigation in vector fields such as oceanic and atmospheric flows, though the printed fields do not yet advect the robots as a physical flow would."""

TEX_EDITS = [
    # abstract
    ("The sensing is spatially distributed across the robots, while the estimate",
     "The sensing is distributed across the robots, while the estimate"),
    ("the attraction error is shown to stay bounded under actuator lag, sampling, and friction.",
     "the attraction error is shown to stay bounded under actuator delay and lag, sampling, and friction."),
    (ABSTRACT_OLD, ABSTRACT_NEW),
    # introduction
    (r"in \rev{459} hardware experiments that demonstrate repeatable critical point estimation, attraction, and orbit control.",
     r"in \rev{459} hardware experiments with three- and four-robot formations that demonstrate repeatable critical point estimation, attraction, and orbit control, and it compares each run with a simulation from the same start."),
    ("Section VI compares performance against alternative navigation strategies and discusses limitations and future work.",
     "Section VI traces the simulation and hardware errors to their sources, compares performance against an alternative navigation strategy, and discusses limitations and future work."),
    # III-A fit quality
    (r"fixed the sum $L + \tau = \rev{0.075}$~s more tightly than either term.",
     r"fixed the sum $L + \tau = \rev{0.075}$~s more tightly than either term. The fitted model reproduces the measured robot velocities with an RMS error of \rev{0.029}~m/s over \rev{718} robot runs, and a fit to either field or either formation predicts the other within \rev{0.0004}~m/s."),
    # III-B four-robot square
    (r"and $q$ is the distance between robots 2 and 3.",
     r"and $q$ is the distance between robots 2 and 3. The four-robot square is parameterized by its diagonals $d_1$ and $d_2$, the fractions $r_1$ and $r_2$ of each diagonal at which they cross, and the angle $\phi$ between them, with $r_1 = r_2 = 0.5$ and $\phi = 90^\circ$ \cite{50}."),
    # IV
    (r"by up to \newnum{0.17}~m", r"by up to \rev{0.17}~m"),
    (r"& +0.018 & +0.006 & $-$0.000 & +0.018", r"& +0.018 & +0.006 & 0.000 & +0.018"),
    # V-A
    ("Four Decabots are shown; the experiments here used three of this fleet.",
     "The four Decabots shown are the fleet used in this work."),
    ("The experimental platform used three omnidirectional Decabot rovers",
     "The experimental platform used three or four omnidirectional Decabot rovers"),
    # V-B matched simulations
    (r"and the four-robot orbits used five radii, from 0.01 to 0.40~m.",
     r"and the four-robot orbits used five radii, from 0.01 to 0.40~m. Each hardware run was also repeated in simulation from its own start, with the formation it held over the run, the same commanded radius and duration, and the model of Section~III-A, on both the analytical and the reconstructed field."),
    # V-C
    (r"Following the test plan above, \rev{80} vortex trials and \rev{80} saddle trials were analyzed.",
     r"Following the test plan above, \rev{80} vortex and \rev{80} saddle trials with three robots and \rev{79} vortex trials with four robots were analyzed."),
    (r"The cluster converged to the critical point in \rev{99}\%",
     r"The three-robot cluster converged to the critical point in \rev{99}\%"),
    (r"and the estimate ended closer to $\mathbf{p}^*$ than the cluster.",
     r"and the estimate ended closer to $\mathbf{p}^*$ than the cluster." + "\n\n" +
     r"With four robots in the vortex, every trial ended within 0.05~m and \rev{65}\% within 0.03~m. The final positions were tighter than with three, with a precision of \rev{0.014}~m against \rev{0.036}~m, but their bias of \rev{0.024}~m matched the four-robot localization error of \rev{0.025}~m, so the square came to rest consistently to one side of $\mathbf{p}^*$, where its estimate placed the critical point. These trials came from a separate session with a larger formation (Section~V-B), so the two formations are not compared run for run."),
    # Fig. 7
    (r"""    \begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{fig_hw_trajectories_vortex.pdf}\end{minipage}\hspace{0.04\columnwidth}%
    \begin{minipage}[b]{0.47\columnwidth}\centering\includegraphics[width=\textwidth]{fig_hw_trajectories_saddle.pdf}\end{minipage}\\[2pt]
    \includegraphics[width=0.71\columnwidth]{fig_hw_distance.pdf}\\[-1pt]
    {\footnotesize (c)}
""", "    \\includegraphics[width=\\columnwidth]{fig_hw_attraction_2026.pdf}\n"),
    ("Convergence trajectories from eight starting positions to the center of the vortex (a) and saddle (b), with thin lines showing each run and the thick line the mean path from each start.",
     "Convergence trajectories from eight starting positions to the critical point of the vortex (a) and saddle (b), with thin lines showing each run (dashed for the two refilled saddle trials) and the thick line the mean path from each start."),
    # Table III caption
    (r"two of them 2026 runs filling missing slots; no four-robot saddle attraction was run. Bias measures mean offset from $\mathbf{p}^*$; precision ($\sigma$) measures repeatability.",
     r"two of them 2026 runs filling missing slots; no four-robot saddle attraction was run. Bias measures mean offset from $\mathbf{p}^*$; precision ($\sigma$) measures repeatability. The last row gives the mean final error of the simulations matched to each run (Section~V-B) on the analytical and reconstructed field."),
    (r"Negative error indicates the cluster orbited inside the commanded radius.}",
     r"Negative error indicates the cluster orbited inside the commanded radius. (d) Share of orbit cycles on which $\hat{\mathbf{J}}$ had the type of the printed field, and in the vortex the share labeled rotational (complex eigenvalues).}"),
    # V-D
    ("Orbital control was evaluated on both fields at commanded radii from 0.01~m to 0.50~m. For each radius,",
     "Orbital control was evaluated on both fields at commanded radii from 0.01~m to 0.50~m with three robots and to 0.40~m with four. For each radius,"),
    (" In the vortex field, mean radial errors decreased from",
     " With three robots in the vortex field, mean radial errors decreased from"),
    (r"Below these radii the controller holds a bounded orbit about the critical point but does not track the commanded radius. No orbit run left the map.",
     r"Below these radii the controller holds a bounded orbit about the critical point but does not track the commanded radius." + "\n\n" +
     r"The four-robot square behaved like the triangle at small radii, with a radial bias of \rev{+0.057}~m at $r_d = 0.01$~m in both fields, and its within-run standard deviations ranged from \rev{0.017} to \rev{0.057}~m in the vortex and from \rev{0.022} to \rev{0.076}~m in the saddle. In the saddle its practical range also begins at \rev{0.20}~m. In the vortex it met the $0.25\,r_d$ criterion at 0.20 and 0.40~m but not at 0.30~m, where its bias reached \rev{+0.098}~m, as at 0.40~m. No orbit run with either formation left the map."),
    (LIMIT_CYCLE_OLD_PARA + LIMIT_CYCLE_FIG, ""),
    (r"The classifier labeled $\hat{\mathbf{J}}$ on every cycle of the orbit runs. It matched the printed saddle on \rev{100}\% of cycles. In the vortex, \rev{99.9}\% of cycles were labeled rotational, but only \rev{45}\% as a center,",
     r"The classifier labeled $\hat{\mathbf{J}}$ on every cycle of the orbit runs (Table~\ref{tab:estimation_results}(d)). It matched the printed saddle on \rev{100}\% of cycles with three robots and \rev{99.6}\% with four. In the vortex, \rev{99.9}\% and \rev{99.6}\% of cycles were labeled rotational, but only \rev{45}\% and \rev{47}\% as a center,"),
    # VI-A matched simulations, after the field-distortion paragraph
    (r"In the final error of Section~III-D, friction sets the spread of final positions through $v_c/k$, and the distortion sets their bias." + "\n\n",
     r"In the final error of Section~III-D, friction sets the spread of final positions through $v_c/k$, and the distortion sets their bias." + "\n\n" + SIM_VS_HW_PARA),
    # VI-B limit cycle, before the V2S subsection
    ("\\subsection{Comparison with the Vector-to-Scalar Primitive}",
     LIMIT_CYCLE_SECTION + "\\subsection{Comparison with the Vector-to-Scalar Primitive}"),
    # VI Limitations, Third
    ("Third, robot team sizes were limited to three. The framework supports larger teams",
     "Third, the largest team tested was four robots. The framework supports larger teams"),
    ("adds communication and formation-control burden.",
     r"adds communication and formation-control burden. The four-robot square illustrates the trade-off: its final positions in the vortex were less than half as spread as the triangle's, yet they came to rest \rev{0.024}~m to one side of $\mathbf{p}^*$, and its orbits fluctuated as much as the triangle's (Section~V). Teams beyond four, and the loss of a robot during a run, were not tested."),
    # Table IV counts
    (r"\newnum{500} attraction trials of 10~s and \newnum{50} orbit runs",
     r"\rev{500} attraction trials of 10~s and \rev{50} orbit runs"),
    # conclusion
    (r"uses instantaneous measurements from three robots to identify",
     r"uses instantaneous measurements from three or more robots to identify"),
    (r"In \rev{160} hardware convergence trials, the system achieved a \rev{99}\% (saddle) and \rev{91}\% (vortex) success rate within 0.05~m, with mean position errors of \rev{0.034}~m or less. Orbital control experiments maintained commanded radii within a quarter of the radius from \rev{0.20}~m upward in both fields, and below that held bounded orbits that did not track the commanded radius.",
     r"In \rev{459} hardware runs, the cluster converged within 0.05~m in \rev{99}\% of saddle and \rev{91}\% of vortex trials with three robots and in every vortex trial with four, with mean position errors of \rev{0.034}~m or less. Orbital control maintained commanded radii within a quarter of the radius from \rev{0.20}~m upward with three robots in both fields, and below that held bounded orbits that did not track the commanded radius; at a commanded 0.10~m the vortex orbit settled onto a limit cycle whose size follows from the delay and lag of the robots. Simulations matched to each run reproduced the saddle's attraction error and about half of the orbit fluctuation, and the remainder points to effects the model omits, such as differences between individual robots and a lead of the live estimate."),
]

LETTER_SHIFT = [("Section~VI-D", "Section~VI-E", 1), ("Section~VI-C", "Section~VI-D", 6), ("Section~VI-B", "Section~VI-C", 6)]
LETTER_EDITS = [
    ("Fig.~9 plots speed and distance", "Fig.~10 plots speed and distance"),
    ("now say the controller ``holds a bounded orbit about the critical point but does not track the commanded radius''.",
     r"now say the controller ``holds a bounded orbit about the critical point but does not track the commanded radius''. Below the practical range the orbit is a limit cycle whose size follows from the delay and lag: at a commanded 0.10~m in the vortex, hardware runs started inside and outside converge to one closed curve of mean radius \rev{0.127}~m, against \rev{0.120}~m from (16) (Section~VI-B, Fig.~9)."),
    (r"$r_d \ge \rev{0.20}$~m (vortex) and $\rev{0.20}$~m (saddle) on hardware (Section~V-D).",
     r"$r_d \ge \rev{0.20}$~m in both fields with three robots on hardware, and with four robots from \rev{0.20}~m in the saddle and at 0.20 and 0.40~m in the vortex (Section~V-D)."),
    (r"\resp (a, b) We agree. Limitation Three of Section~VI-D discusses larger teams (search ``supports larger teams''): averaging and fault tolerance against a wider footprint, with the trade-off tied to the bound (8).",
     r"\resp (a, b) We agree, and the hardware now tests a larger team. The vortex attraction trials and both fields' orbit trials were repeated with four robots in a square, with $\hat{\mathbf{J}}$ from the least-squares fit (Section~V-B); Table~III reports three- and four-robot columns. Limitation Three of Section~VI-D discusses the benefits and trade-offs (search ``supports larger teams''): averaging and fault tolerance against a wider footprint, with the trade-off tied to the bound (8). The four-robot square shows both, with final positions less than half as spread as the triangle's but a \rev{0.024}~m offset."),
]


def replace_table(s, label, body):
    i = s.index("\\label{" + label + "}")
    a = s.index("\\rev{\\begin{tabular}", i)
    b = s.index("\\end{tabular}}", a) + len("\\end{tabular}}")
    return s[:a] + "\\rev{" + body.strip() + "}" + s[b:]


def apply(s, edits, name):
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT ({name}): expected 1 match, found {n}: {old[:90]!r}")
        s = s.replace(old, new)
    return s


def main(dry):
    tex, letter = open(TEX).read(), open(LETTER).read()
    tex = apply(tex, TEX_EDITS, "paper")
    tex = replace_table(tex, "tab:estimation_results", open(os.path.join(HERE, "table_hardware.tex")).read())
    for old, new, n in LETTER_SHIFT:
        if letter.count(old) != n:
            sys.exit(f"ABORT (letter): {old} appears {letter.count(old)} times, expected {n}")
        letter = letter.replace(old, new)
    letter = apply(letter, LETTER_EDITS, "letter")
    left = re.findall(r"\\newnum\{", tex)
    print(f"paper: {len(TEX_EDITS)} edits and Table III; letter: {sum(n for *_, n in LETTER_SHIFT)} section numbers "
          f"and {len(LETTER_EDITS)} edits; \\newnum left in the paper: {len(left)}")
    if dry:
        return
    for path, tag in ((TEX, "cwaight_systems_paper_with_revisions2"), (LETTER, "Response_to_Reviewers")):
        bak = os.path.join(HERE, f"backup_{tag}_before_round2.tex")
        if not os.path.exists(bak):
            shutil.copy(path, bak)
    open(TEX, "w").write(tex)
    open(LETTER, "w").write(letter)
    for src, dst in FIGS.items():
        shutil.copy(src, os.path.join(PAPER, dst))
    print("written; figures copied:", ", ".join(FIGS.values()))


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
