"""Apply the approved prose batch (queued_prose_batch.md, approved 2026-11-16) to the revision tex and letter.

Items F (covariance check) and the sweep sentence of M wait for the v4 sweep. Each edit matches exact text (or
an exact start and end anchor) once; any mismatch aborts the whole batch and nothing is written.
Run from the repo root:  python3 analysis/paper/apply_prose_batch.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402

TEX = os.path.join(paths.PAPER, "cwaight_systems_paper_with_revisions2.tex")
LETTER = os.path.join(os.path.dirname(paths.PAPER), "Response_to_Reviewers.tex")
FIG = (os.path.join(os.path.dirname(HERE), "limit_cycle", "figures", "fig_limit_cycle.pdf"),
       os.path.join(paths.PAPER, "fig_limit_cycle_2026.pdf"))

A = r"""Details of the hardware platform are described in Section~V. In simulation, each robot is modeled as a single omnidirectional point mass that follows its command after a transport delay, through a first order response, with a dead zone and a speed limit. A command reaches the robot a time $L$ after it is computed, and the robot attains the fraction $g$ of it, so its velocity $\mathbf{v}_{r,i} = \dot{\mathbf{p}}_i$ follows
\begin{equation}
    \tau\,\dot{\mathbf{v}}_{r,i}(t) = -\mathbf{v}_{r,i}(t) + g\,\mathbf{u}_i(t - L),
    \label{eq:continuous}
\end{equation}
where $\tau$ is the actuator time constant \cite{4,7}. The command passes through a dead zone that models static friction: $\mathbf{u}_i = \mathbf{v}_{\text{des},i}$ when $\|\mathbf{v}_{\text{des},i}\| \ge v_c$, and $\mathbf{u}_i = \mathbf{0}$ otherwise. Fitting (\ref{eq:continuous}) to the logged commands and tracked positions of the hardware runs of Section~V gave $g = \rev{0.86}$ and fixed the sum $L + \tau = \rev{0.075}$~s more tightly than either term. The robots reach their commanded speed within one cycle, so we take $\tau = \rev{0.02}$~s and $L = \rev{0.055}$~s. The robots are also modeled with a maximum speed of $v_{\max} = \rev{0.3}$~m/s and a dead zone of $v_c = 0.025$~m/s (remeasured for this work; an earlier characterization of the testbed reported 0.05~m/s \cite{22}).

For digital implementation at control period $\Delta t$ and cycle $j$, each command is held over its period. A delay $L = f\Delta t$ with $0 \le f < 1$ means the robot spends the first fraction $f$ of each period on the previous command, so
\begin{equation}\label{eq:momentum}
\begin{split}
&\bar{\mathbf{u}}_i[j] = (1-f)\,\mathbf{u}_i[j] + f\,\mathbf{u}_i[j-1],\\
&\tilde{\mathbf{v}} = \alpha\,\mathbf{v}_{r,i}[j] + (1-\alpha)\,g\,\bar{\mathbf{u}}_i[j],\\
&\mathbf{v}_{r,i}[j+1] = \tilde{\mathbf{v}}\,\min\{1,\ v_{\max}/\|\tilde{\mathbf{v}}\|\},
\end{split}
\end{equation}
where $\alpha = e^{-\Delta t/\tau}$ and the dead zone acts on $\bar{\mathbf{u}}_i$. At a 10~Hz control rate, $f = \rev{0.55}$ and $\alpha \approx \rev{0.007}$, so the robot reaches $g$ times its command within one cycle, and the delay, not the lag, shapes its response. The position advances as $\mathbf{p}_i[j+1] = \mathbf{p}_i[j] + \Delta t\,\mathbf{v}_{r,i}[j+1]$. The sensed field enters only the estimate, not the robot dynamics."""

B = r"""With the formation held, and apart from the dead zone and the cap, the centroid follows (\ref{eq:momentum}) with $\alpha \approx 0$. The $x$ and $y$ axes then decouple, and for each axis $a$
\begin{equation}\label{eq:cl}
e_a[j{+}1] = e_a[j] - gk\Delta t\bigl((1-f)(e_a[j] - \delta_a[j]) + f(e_a[j-1] - \delta_a[j-1])\bigr).
\end{equation}
The Jury test on its characteristic polynomial $z^2 - (1 - gk\Delta t(1-f))z + gk\Delta t f$ gives exponential stability if and only if $gk\Delta t f < 1$ and $gk\Delta t(1-2f) < 2$, with $k > 0$. For $f \ge 1/4$ the first condition binds, and since $f\Delta t = L$,
\begin{equation}\label{eq:kmax}
0 < k < 1/(gL).
\end{equation}
At $g = 0.86$ and $L = 0.055$~s the limit is \rev{21.1}~s$^{-1}$, and at $k = 1$~s$^{-1}$ the poles have magnitudes \rev{0.91} and \rev{0.05}. Including the lag changes the limit by less than 0.01~s$^{-1}$.
A stable linear loop is input-to-state stable, so a bounded estimate error $\boldsymbol{\delta}$, and the dead zone acting as a command disturbance of size at most $v_c$, leave the centroid error bounded."""

D = r"""Even with an exact estimate, the delay and lag leave a predictable offset in the steady orbit radius. The robots reach the fraction $g$ of each command, so the orbit speed is $V = gk_t$ and the radial correction has gain $gk_r$. The cluster follows the turning command a time $T$ late, so in a steady orbit of radius $r^*$ it drifts outward at $TV^2/r^*$. The orbit settles where the inward radial command balances this drift, giving
\begin{equation}\label{eq:orbit_cf}
r^* = \tfrac{1}{2}\bigl(r_d + \sqrt{r_d^2 + 4C}\bigr), \qquad C = TV^2/(gk_r).
\end{equation}
In the sampled loop $T \approx L + \tau + \Delta t/2$: the delay, the lag, and half a period of command hold. Equation (\ref{eq:orbit_cf}) accounts for two behaviors seen in Sections~IV and~V: the error $r^* - r_d$ cannot shrink below the floor $r_{\min} = \sqrt{C}$ as $r_d \to 0$, while at large radii it decays as $C/r_d$."""

G = r"""The reconstructed fields share the bias at $r_d = 0.01$~m and nearly cancel it at 0.10~m. From 0.20~m their bias lies between \rev{0.019} and \rev{0.050}~m, with within-run standard deviations that increase with the commanded radius. By the $0.25\,r_d$ criterion the practical range is $r_d \ge \rev{0.10}$~m on the analytical and reconstructed fields."""

M = r"""Positive radial bias at small orbital radii is also a result of robot dynamics. At small radii, the commanded angular rate $k_t / r_d$ is high relative to the radial correction bandwidth $gk_r$, and the outward drift from the delay and lag sets the bias (\ref{eq:orbit_cf}). The closed form matches simulation within \rev{3}~mm at every commanded radius at the settings of Section~IV. There $r_{\min} = \rev{0.047}$~m, so the error stays below 0.05~m at every commanded radius. Lowering the orbital speed $k_t$ to 0.1~m/s cuts $r_{\min}$ to \rev{0.031}~m, and removing the delay and lag cuts it to \rev{0.032}~m. Raising the radial gain $k_r$ to 2~s$^{-1}$ cuts it to \rev{0.034}~m, at the cost of increased sensitivity to noise in the critical point estimate."""

N = r"""Together they set the localization error at the largest commanded radius. There the orbital bias on both reconstructed fields exceeds the controller-only bias, \rev{+0.027} against \rev{+0.013}~m on the vortex and \rev{+0.019} against \rev{$-$0.003}~m on the saddle, so the estimate lies closer to the cluster than the critical point does, and a cluster holding its estimated radius orbits outside $r_d$."""

PARA_VD = r"""No orbit run left the map.

At $r_d = 0.10$~m in the vortex the cluster does not hold the commanded circle but settles onto a closed curve of its own (Fig.~\ref{fig:limit_cycle}). Ten hardware runs, four started inside the curve and six outside, converged to the same curve, with a mean radius of \rev{0.127}~m that varies by \rev{0.043}~m around the loop and \rev{2.9}~mm between runs. On the reconstructed vortex in simulation, starts inside and outside meet on a curve of mean radius \rev{0.111}~m.

\begin{figure*}[!t]
    \centering
    \includegraphics[width=0.95\textwidth]{fig_limit_cycle_2026.pdf}
    \caption{Limit cycle in the vortex at $r_d = 0.10$~m (dotted circle). (a) Simulation on the reconstructed vortex from starts inside (blue) and outside (red) the cycle, which meet on one closed curve (black). (b) Ten hardware runs overlaid, with the hardware cycle (black) and the simulated cycle (dashed). (c) Distance to $\mathbf{p}^*$ over time for the simulated starts and the hardware runs.}
    \label{fig:limit_cycle}
\end{figure*}

The classifier labeled $\hat{\mathbf{J}}$ on every cycle of the orbit runs. It matched the printed saddle on \rev{100}\% of cycles. In the vortex, \rev{99.9}\% of cycles were labeled rotational, but only \rev{45}\% as a center, because the center's eigenvalues lie on the boundary between the two spiral types (Section~IV-C), so a small real part from distortion or noise moves the label to a spiral."""

TEX_EDITS = [   # (exact old, new) or (("start", "end"), new) for an anchored span
    ("despite actuator lag, sampled updates, and friction", "despite actuator delay and lag, sampled updates, and friction"),
    (("Details of the hardware platform are described in Section~V. In simulation,",
      "The sensed field enters only the estimate, not the robot dynamics."), A),
    ("the centroid responds through the sampled lag and Coulomb friction,", "the centroid responds through the delay, the lag, and the dead zone,"),
    (("With the formation held, and apart from friction and the cap,", "leave the centroid error bounded."), B),
    ("The second effect is friction, and it sets where the cluster finally comes to rest. Coulomb friction stops the cluster wherever its command is at most $v_c$.",
     "The second effect is friction, and it sets where the cluster finally comes to rest. The dead zone stops each robot whenever its command is below $v_c$."),
    (("Even with an exact estimate, lag and friction leave a predictable offset", "while at large radii it decays as $C/r_d$."), D),
    ("while at least \\rev{95}\\% of trials still end within 0.05~m.", "while every trial still ends within 0.05~m."),
    (("The reconstructed fields repeat the trend,", "and $r_d \\ge \\rev{0.4}$~m on the reconstructed vortex."), G),
    ("using the three-robot equilateral triangle formation with a \\rev{0.30}~m side, yielding 80 trials per field. Run order was randomized.",
     "using the three-robot triangle formation, yielding 80 trials per field. The logged positions give a mean side of \\rev{0.30}~m in the saddle trials and \\rev{0.26}~m in the vortex trials. Run order was randomized. Two saddle trials missing from the original set were filled by later runs started within \\rev{9}~cm of the same locations, with their positions rotated into the original frame of the printed map."),
    ("Each orbiting controller experiment was run for \\rev{200 to 300} cycles at 10~Hz, allowing the cluster to complete several orbits around the critical point.",
     "Each orbiting controller experiment was run for \\rev{200 to 300} cycles at 10~Hz, long enough for at least one full orbit at the largest radius and up to eight at the smallest. The vortex attraction trials and both fields' orbit trials were repeated with four robots in a square formation, with $\\hat{\\mathbf{J}}$ from a least squares fit to the four readings. The square's diagonals measured \\rev{0.33}~m in the attraction trials and \\rev{0.31}~m in the orbit trials, and the four-robot orbits used five radii, from 0.01 to 0.40~m."),
    (" The type of $\\hat{\\mathbf{J}}$ matched the printed field on \\open{campaign}\\% of cycles in the vortex and \\rev{100}\\% in the saddle.", ""),
    ("the practical range is $r_d \\ge \\rev{0.20}$~m in the vortex and $r_d \\ge \\rev{0.20}$~m in the saddle.",
     "the practical range is $r_d \\ge \\rev{0.20}$~m in both fields."),
    ("\\syn{One} vortex and \\syn{three} saddle orbits failed by leaving the map, all at $r_d \\ge \\syn{0.40}$~m.", PARA_VD),
    ("at radii of \\rev{0.20}~m (vortex) and \\rev{0.20}~m (saddle) and above.",
     "at radii of \\rev{0.20}~m and above in both fields. A four-robot formation converged within 0.05~m in all \\rev{79} vortex trials."),
    ("from \\rev{0.20}~m (vortex) and \\rev{0.20}~m (saddle) upward,", "from \\rev{0.20}~m upward in both fields,"),
    (("Positive radial bias at small orbital radii is also a result of robot dynamics.",
      "at the cost of increased sensitivity to noise in the critical point estimate."), M),
    (("Together they account for the orbital bias at the largest commanded radius", "and the radial controller pulls the cluster inward."), N),
    ("so centrifugal drift went uncorrected and every orbit spiraled off the map (Fig.~\\ref{fig:effort}(b)).",
     "so centrifugal drift went uncorrected and every orbit drifted outward, with a mean radial error of \\rev{0.14}~m at $r_d = 0.3$~m (Fig.~\\ref{fig:effort}(b))."),
    ("With its own speed estimate the primitive held the radius at a lower RMSE, because the estimate's inward bias offsets the lag drift of Section~III-D; during attraction, however, that estimate left",
     "With its own speed estimate the primitive orbited inside the commanded radius, by \\rev{0.075}~m at $r_d = 0.1$~m, because the estimate's inward bias outweighs the lag drift of Section~III-D; during attraction, that estimate left"),
    ("and even with the oracle most orbits were lost, because the speed contours are ellipses",
     "and even with the oracle \\rev{88}\\% of orbits at $r_d = 0.1$~m and \\rev{30}\\% at 0.5~m were lost, because the speed contours are ellipses"),
    ("The primitive lost every gyre orbit, and with the oracle every orbit from $r_d = 0.3$~m.",
     "The primitive lost every gyre orbit, and with the oracle \\rev{92}\\% at $r_d = 0.3$~m and all at 0.5~m."),
]

LETTER_EDITS = [
    ("\\syn{All hardware experiments were repeated in a new campaign designed around the reviewers' requests, and every hardware result in the revision comes from it.}",
     "The hardware orbit experiments were repeated in a new campaign designed around the reviewers' requests, with ten repeats per radius and with both three and four robots, and every hardware result is re-scored with the metrics of Section~IV."),
    ("\\syn{In re-examining the original logs we found that the vortex rows of the submitted Tables~IV and~V came from four-robot runs; the new campaign uses the three-robot formation throughout.}",
     "In re-examining the original logs we found that the vortex rows of the submitted Tables~IV and~V came from four-robot runs; the revision reports three-robot and four-robot results in separate columns of Table~III."),
    ("Hardware accuracy is in Section~V-C \\open{campaign}.",
     "Hardware accuracy is in Section~V-D: \\rev{100}\\% of cycles in the saddle, and in the vortex \\rev{99.9}\\% rotational and \\rev{45}\\% a center."),
    ("with the same first-\\rev{3}~s exclusion; Table~III(b) reports the between-trial and within-run spreads.",
     "with the same first-\\rev{3}~s exclusion; Table~III(b) reports the between-trial and within-run spreads. A settling analysis of the new orbit runs found no start-up transient on hardware, since each run starts near its circle, and in simulation the radial error falls to 2~mm within 3~s. The submitted 10~s exclusion was therefore more generous than needed, and the revision uses 3~s for simulation and hardware alike; the mean radial bias changes by at most 4~mm between exclusions of 0, 2, 5 and 10~s."),
    ("(search ``matches simulation to a median'')", "(search ``matches simulation within'')"),
]


def apply(s, edits, name):
    for old, new in edits:
        if isinstance(old, tuple):
            a, b = old
            if s.count(a) != 1:
                sys.exit(f"ABORT ({name}): start anchor {a[:60]!r} found {s.count(a)} times")
            i = s.index(a)
            j = s.find(b, i)
            if j < 0 or s.count(b, i) < 1:
                sys.exit(f"ABORT ({name}): end anchor {b[:60]!r} not found after start")
            s = s[:i] + new + s[j + len(b):]
        else:
            if s.count(old) != 1:
                sys.exit(f"ABORT ({name}): {old[:70]!r} found {s.count(old)} times")
            s = s.replace(old, new)
    return s


def main(dry):
    tex = apply(open(TEX).read(), TEX_EDITS, "tex")
    let = apply(open(LETTER).read(), LETTER_EDITS, "letter")
    if dry:
        print(f"dry run: {len(TEX_EDITS)} tex and {len(LETTER_EDITS)} letter edits matched")
        return
    shutil.copy2(TEX, os.path.join(HERE, "backup_cwaight_systems_paper_with_revisions2_before_prose_batch.tex"))
    shutil.copy2(LETTER, os.path.join(HERE, "backup_Response_to_Reviewers_before_prose_batch.tex"))
    shutil.copy2(*FIG)
    open(TEX, "w").write(tex)
    open(LETTER, "w").write(let)
    print(f"applied {len(TEX_EDITS)} tex and {len(LETTER_EDITS)} letter edits; backups in analysis/paper/")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
