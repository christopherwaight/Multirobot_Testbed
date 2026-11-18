"""Round 2b (2026-11-17): bring the paper back to 13 pages after round 2.

Tightens the new round-2 prose, shortens the Table III caption, folds the Fig. 7 pointer into the results
sentence, and replaces the full-width limit-cycle figure with a one-column version (panels a and b).
Exact-match edits; aborts before writing if one fails. Run after apply_round2.py.
Usage: python3 analysis/paper/apply_round2b.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apply_round2 as r2  # noqa: E402

OLD_VIA = r2.SIM_VS_HW_PARA.split("\n\n")[0]
NEW_VIA = r"""The simulations matched to each hardware run (Section~V-B) show how much of the hardware error the model reproduces (Fig.~\ref{fig:sim_vs_hw}). They reproduce the saddle's attraction error, \rev{0.026}~m against \rev{0.023}~m, while the hardware vortex ends \rev{0.4} to \rev{1.5}~cm farther from $\mathbf{p}^*$ (Table~\ref{tab:estimation_results}(a)). In orbit at $r_d = 0.01$~m the hardware bias exceeds the reconstructed-field simulations by \rev{14} to \rev{19}~mm. In these runs the live estimate ran \rev{1.5} to \rev{2}~cm ahead of the cluster along its motion, a lead of unidentified cause that the model omits; adding it to the model brings the vortex orbits at $r_d = 0.01$ and 0.10~m within \rev{7}~mm of the hardware. From $r_d = 0.30$~m the reconstructed fields produce \rev{27} to \rev{63}\% of the hardware within-run spread and the analytical fields at most \rev{1}~mm, so the fixed distortion of the printed maps accounts for about half of it. The model also treats every robot alike, yet their fitted gains range from \rev{0.80} to \rev{0.89}, and a robot held out of the field reconstruction read the map with a bias of up to \rev{5}$^\circ$."""

OLD_VIB = r2.LIMIT_CYCLE_SECTION
NEW_VIB = r"""\subsection{The Orbit as a Limit Cycle}

Below the practical range the orbit converges not to the commanded circle but to a closed curve of its own, a limit cycle of the closed loop. At $r_d = 0.10$~m in the vortex, ten three-robot runs, four started inside the curve and six outside, converged to the same curve (Fig.~\ref{fig:limit_cycle}), of mean radius \rev{0.127}~m with \rev{2.9}~mm between runs, and in simulation on the reconstructed vortex the curve has mean radius \rev{0.111}~m. The closed form (\ref{eq:orbit_cf}) sets its size, $r^* = \rev{0.120}$~m, which the cluster holds on a circle in the analytical fields. The fixed distortion of the printed map bends the cycle out of round, so its radius varies around the loop by \rev{0.029}~m in simulation and \rev{0.043}~m on hardware.

\begin{figure}[!t]
    \centering
    \includegraphics[width=\columnwidth]{fig_limit_cycle_2026.pdf}
    \caption{Limit cycle in the vortex at $r_d = 0.10$~m (dotted circle). (a) Simulation on the reconstructed vortex from starts inside (blue) and outside (red) the cycle, which meet on one closed curve (black). (b) Ten hardware runs, with the hardware cycle (black) and the simulated cycle (dashed).}
    \label{fig:limit_cycle}
\end{figure}

"""

EDITS = [
    (OLD_VIA, NEW_VIA),
    (OLD_VIB, NEW_VIB),
    (r"With four robots in the vortex, every trial ended within 0.05~m and \rev{65}\% within 0.03~m. The final positions were tighter than with three, with a precision of \rev{0.014}~m against \rev{0.036}~m, but their bias of \rev{0.024}~m matched the four-robot localization error of \rev{0.025}~m, so the square came to rest consistently to one side of $\mathbf{p}^*$, where its estimate placed the critical point. These trials came from a separate session with a larger formation (Section~V-B), so the two formations are not compared run for run.",
     r"With four robots in the vortex, every trial ended within 0.05~m and \rev{65}\% within 0.03~m. Precision improved to \rev{0.014}~m from \rev{0.036}~m with three, but the bias of \rev{0.024}~m matched the localization error of \rev{0.025}~m: the square came to rest to one side of $\mathbf{p}^*$, where its estimate placed it. These trials came from a separate session with a larger formation (Section~V-B)."),
    (r"The four-robot square behaved like the triangle at small radii, with a radial bias of \rev{+0.057}~m at $r_d = 0.01$~m in both fields, and its within-run standard deviations ranged from \rev{0.017} to \rev{0.057}~m in the vortex and from \rev{0.022} to \rev{0.076}~m in the saddle. In the saddle its practical range also begins at \rev{0.20}~m. In the vortex it met the $0.25\,r_d$ criterion at 0.20 and 0.40~m but not at 0.30~m, where its bias reached \rev{+0.098}~m, as at 0.40~m.",
     r"The four-robot square behaved like the triangle at small radii, with a bias of \rev{+0.057}~m at $r_d = 0.01$~m in both fields, and its within-run standard deviations ranged from \rev{0.017} to \rev{0.057}~m in the vortex and \rev{0.022} to \rev{0.076}~m in the saddle. Its practical range begins at \rev{0.20}~m in the saddle; in the vortex it met the $0.25\,r_d$ criterion at 0.20 and 0.40~m but not at 0.30~m, where its bias was \rev{+0.098}~m."),
    (r" The four-robot square illustrates the trade-off: its final positions in the vortex were less than half as spread as the triangle's, yet they came to rest \rev{0.024}~m to one side of $\mathbf{p}^*$, and its orbits fluctuated as much as the triangle's (Section~V).",
     r" The four-robot square showed this trade-off: its vortex final positions were less than half as spread as the triangle's but rested \rev{0.024}~m to one side of $\mathbf{p}^*$."),
    (r"at a commanded 0.10~m the vortex orbit settled onto a limit cycle whose size follows from the delay and lag of the robots. Simulations matched to each run reproduced the saddle's attraction error and about half of the orbit fluctuation, and the remainder points to effects the model omits, such as differences between individual robots and a lead of the live estimate.",
     r"at a commanded 0.10~m the vortex orbit settled onto a limit cycle sized by the delay and lag of the robots. Simulations matched to each run reproduced the saddle's attraction error and about half of the orbit fluctuation."),
    (r" Each hardware run was also repeated in simulation from its own start, with the formation it held over the run, the same commanded radius and duration, and the model of Section~III-A, on both the analytical and the reconstructed field.",
     r" Each hardware run was also simulated from its own start, with the formation it held and the same commanded radius and duration, on the analytical and the reconstructed field."),
    (r"(a) Attraction (2025 runs): vortex 3R \rev{80}, vortex 4R \rev{79}, saddle 3R \rev{80} runs, two of them 2026 runs filling missing slots; no four-robot saddle attraction was run. Bias measures mean offset from $\mathbf{p}^*$; precision ($\sigma$) measures repeatability. The last row gives the mean final error of the simulations matched to each run (Section~V-B) on the analytical and reconstructed field. (b), (c) Orbital control (2026 runs, ten per radius), radial errors in metres after the first \rev{3}~s. $\bar e_r$ is the radial bias, $s_b$ the between-trial spread, and $s_w$ the within-run standard deviation. Negative error indicates the cluster orbited inside the commanded radius. (d) Share of orbit cycles on which $\hat{\mathbf{J}}$ had the type of the printed field, and in the vortex the share labeled rotational (complex eigenvalues).}",
     r"(a) Attraction, 2025 runs (two 2026 runs fill missing saddle slots; no four-robot saddle attraction was run); the last row is the mean final error of the matched simulations (Section~V-B) on the analytical and reconstructed field. (b), (c) Orbits, 2026 runs, ten per radius, after the first \rev{3}~s: radial bias $\bar e_r$ (negative inside $r_d$), between-trial spread $s_b$, and within-run spread $s_w$, in metres. (d) Share of orbit cycles on which $\hat{\mathbf{J}}$ had the type of the printed field, and in the vortex the share labeled rotational.}"),
    ("Fig.~\\ref{fig:vortextomiddle} shows convergence trajectories from all eight starting positions and mean centroid distance to the critical point over time for the same trials.\n\n", ""),
    (r"The three-robot cluster converged to the critical point in \rev{99}\%",
     r"The three-robot cluster converged to the critical point (Fig.~\ref{fig:vortextomiddle}) in \rev{99}\%"),
]


def main(dry):
    s = r2.apply(open(r2.TEX).read(), EDITS, "paper")
    print(f"paper: {len(EDITS)} trim edits")
    if dry:
        return
    open(r2.TEX, "w").write(s)
    src = os.path.join(r2.ROOT, "analysis", "limit_cycle", "figures", "fig_limit_cycle_column.pdf")
    shutil.copy(src, os.path.join(r2.PAPER, "fig_limit_cycle_2026.pdf"))
    for src, dst in list(r2.FIGS.items())[:2]:          # re-copy the resized Figs. 7 and 8
        shutil.copy(src, os.path.join(r2.PAPER, dst))
    print("written; figures copied")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
