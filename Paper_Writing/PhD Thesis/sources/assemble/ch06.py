import port
from port import P, T, write
port.PRE = 'ch6'

# 2026-11-18: rewritten to mirror the revision paper (Systems Paper Revision).
# Structure follows the revision: estimation, control laws, closed-loop
# stability, simulation, hardware, discussion. Dynamics stay in Chapter 2 and
# the architecture in Chapter 4. Limitations and future work are in Chapter 9.
# The revision's labels (sec:sim_setup, eq:bound, ...) gain the ch6 prefix
# automatically, so the section labels below reuse the revision's names.

IEEEPROOF = [(r'\begin{IEEEproof}', r'\begin{proof}'), (r'\end{IEEEproof}', r'\end{proof}')]
STAB = ('Section~III-D', r'Section~\ref{sec:ch6:stability}')

parts = [
T(r"""
% ===================================================================
% CHAPTER 6: FIRST-ORDER PRIMITIVES (critical-points paper)
% ===================================================================
\chapter{First-Order Primitives: Critical Points}
\label{ch:first_order}

\section{Introduction}
\label{sec:ch6:intro}

% New text
Chapter~\ref{ch:zeroth} showed that direction-only primitives reach a
feature only when the field itself carries the cluster there. Orbiting a
fixed vortex with the vector-sum primitive drifted steadily outward,
because nothing in a heading measures the distance to the center. This
chapter moves one rung up the ladder of Section~\ref{sec:ch4:ladder}.
"""),
P('sys', r'Recovering those coordinates is harder than following a heading'),
P('sys', r'The implementation problem is that the local field must be reconstructed'),
P('sys', r'This paper introduces a cooperative estimation strategy',
  subs=[('This paper introduces', 'This chapter develops')]),

T(r"""
\section{Critical Point Estimation from Three Robots}
\label{sec:ch6:estimation}
"""),
P('sys', r'Given a cluster of three robots sampling a planar vector field',
  end='Thus, provided', extra=3),
P('sys', r'The eigenvalues of the Jacobian estimate $\hat{\mathbf{J}}$ can be used to identify the type',
  subs=[(r'(Table~\ref{tab:field_summary})', r'(Table~\ref{tab:ch3:field_summary})')]),
P('sys', r'Equation (\ref{eq:9}) returns the zero of the fitted affine field',
  end='matches the variance of the mean of three independent readings.'),

T(r"""
\subsection{Minimum Robot Requirements}
\label{sec:ch6:minimum}
"""),
P('sys', r'each contain three coefficients. From', end=r'\end{IEEEproof}',
  subs=IEEEPROOF + [('Therefore: three robots are the minimum required for critical point estimation.',
                     'Therefore: three robots are the minimum required to recover an unrestricted affine model, and with it the critical point, from one set of instantaneous measurements. A prior on the field, or measurements over time, could reduce the count.')]),

T(r"""
\section{Control Laws}
\label{sec:ch6:control_laws}
"""),
P('sys', r'We develop two control primitives that drive the cluster centroid'),

T(r"""
\subsection{Critical Point Attraction}
\label{sec:ch6:attraction_def}
"""),
P('sys', r'To navigate the cluster center $\mathbf{p}_c$ to $\hat{\mathbf{p}}^*$, we apply',
  end='provided the estimate is stationary.'),

T(r"""
\subsection{Orbital Navigation}
\label{sec:ch6:orbital_def}
"""),
P('sys', r'To maintain a circular orbit of radius $r_d$ around',
  end='producing steady circular motion.',
  subs=[(r'$\omega = k_t/r_d$', r'$\omega_o = k_t/r_d$')]),

T(r"""
\section{Closed-Loop Stability and Tracking Bounds}
\label{sec:ch6:stability}
"""),
P('sys', r'Section~II-D analyzed the control laws assuming a stationary estimate',
  subs=[('Section~II-D', r'Section~\ref{sec:ch6:control_laws}')]),

T(r"""
\subsection{Sampled Loop}
\label{sec:ch6:sampled_loop}
"""),
P('sys', r'Let $\hat{\mathbf{p}}^*[j] = \mathbf{p}^* + \boldsymbol{\delta}[j]$ be the estimate at cycle',
  end='so the result there is local.',
  subs=[(r'(\ref{eq:momentum})', r'(\ref{eq:ch2:momentum})'),
        (r'(\ref{eq:continuous})', r'(\ref{eq:ch2:continuous})'),
        (r'($\alpha \approx 0$)', r'($\alpha_{\text{mom}} \approx 0$)')]),

T(r"""
\subsection{Friction and Estimate Error}
\label{sec:ch6:friction}
"""),
P('sys', r'Because the attraction command $k\|\mathbf{p}_c - \hat{\mathbf{p}}^*\|$ falls below'),

T(r"""
\subsection{Near-Singular Estimates}
\label{sec:ch6:near_singular}
"""),
P('sys', r'Because the formation keeps the robots far from collinear'),

T(r"""
\subsection{Orbits}
\label{sec:ch6:orbits}
"""),
P('sys', r'The orbital law (\ref{eq:11}) is undefined at $r = 0$',
  end=r'\end{equation}',
  subs=[(r'it drifts outward at $TV^2/r^*$.',
         r'it drifts outward at $TV^2/r^*$. This is the outward drift that the vector-sum primitive showed in Chapter~\ref{ch:zeroth}, where nothing opposed it.')]),

T(r"""
\section{Control Performance in Simulation}
\label{sec:ch6:sim}

\subsection{Simulation Setup}
\label{sec:ch6:sim_setup}
"""),
P('sys', r'A Python-based simulation implementing the three-layer control architecture',
  end='in the rotational form.',
  subs=[(r'$S(s) = \sin(\omega s)/\omega$ and $\omega^2 = 6\chi$', r'$S(s) = \sin(\gamma s)/\gamma$ and $\gamma^2 = 6\chi$'),
        ('described in Section~III', r'described in Chapter~\ref{ch:estimation}'),
        ('control primitives in Section~II.', 'control primitives of this chapter.'),
        ('(Section~V)', r'(Section~\ref{sec:ch6:hw})'),
        ('The robots follow the model of Section~III-C', r'The robots follow the model of Section~\ref{sec:ch2:dynamics}'),
        (r'listed in Table~\ref{tab:field_summary}', r'listed in Table~\ref{tab:ch3:field_summary}')]),

T(r"""
\subsection{Metrics}
\label{sec:ch6:metrics}
"""),
P('sys', r'\emph{Attraction.} A trial of duration $T$ succeeds if',
  end='matched the type of the critical point the trial reached or orbited.'),

T(r"""
\subsection{Critical Point Attraction}
\label{sec:ch6:sim_attraction}
"""),
P('sys', r'For each of the eleven fields, the cluster was initialized at a random position',
  end='the same way at the same place on every pass.',
  subs=[STAB]),
P('sys', r'\caption{Simulation results on eleven fields.', end=r'\end{table}', back=3,
  subs=[(r'A dash marks a radius where most orbits were lost.',
         r'A dash marks a radius where most orbits were lost. In (a), $\bar{e}_f \pm s_e$ is the mean and standard deviation of the final error $e_f$.')]),
P('sys', r'In the nonlinear fields, curvature affected both the localization error'),
P('sys', r'\caption{The gyre cell in simulation, with the', end=r'\end{figure}', back=3,
  subs=[('fig_nonlinear_triangles_2026.pdf', 'ch6_fig_nonlinear_triangles_2026.pdf')],
  width=r'0.8\textwidth'),

T(r"""
\subsection{Orbital Navigation}
\label{sec:ch6:sim_orbit}
"""),
P('sys', r'Orbital control was evaluated on the same eleven fields at different commanded radii',
  end="while the irrotational saddle's opposite shifts"),

T(r"""
\subsection{Classification}
\label{sec:ch6:sim_class}
"""),
P('sys', r'Table~\ref{tab:simulation_results}(c) gives the classifier',
  subs=[(r'Table~\ref{tab:field_summary}', r'Table~\ref{tab:ch3:field_summary}')]),

T(r"""
\section{Hardware Validation}
\label{sec:ch6:hw}

\subsection{Hardware Testbed and Field Representation}
\label{sec:ch6:hw_testbed}

% New text
The testbed, the printed color maps and their sensor calibration, and the
field reconstructions behind the reconstructed simulation fields are
described in Sections~\ref{sec:ch2:testbed}, \ref{sec:ch2:hsv},
\ref{sec:ch2:calibration}, and~\ref{sec:ch2:reconstruction}.
"""),
P('sys', r'The adaptive navigation and cluster space controller layers of Section~III run in MATLAB/Simulink',
  subs=[(r"The adaptive navigation and cluster space controller layers of Section~III run in MATLAB/Simulink on a central computer that communicates with the robots over Wi-Fi at 10~Hz: at each control cycle it receives OptiTrack positions and RGB sensor readings, computes the control laws, and returns velocity commands, which each robot's onboard computer (Intel Edison) converts into drive motor commands. ", ''),
        ('the robot model of Section~III-C', r'the robot model of Section~\ref{sec:ch2:dynamics}')]),

T(r"""
\subsection{Validation Test Plan}
\label{sec:ch6:hw_plan}
"""),
P('sys', r'Hardware validation focused on two representative field types',
  end=r'Every trial is scored with the metrics of Section~\ref{sec:metrics}.'),

T(r"""
\subsection{Critical Point Attraction}
\label{sec:ch6:hw_attraction}
"""),
P('sys', r'Following the test plan,',
  end='matched the localization error',
  subs=[(r'and in 80\% and 39\% at 0.03~m.',
         r'and in 80\% and 39\% at 0.03~m. At the 0.05~m threshold the 95\% Wilson intervals are 93 to 100\% for the saddle and 83 to 96\% for the vortex.'),
        (r'ended within 0.05~m and 65\% within 0.03~m.',
         r'ended within 0.05~m (95 to 100\%, 95\% Wilson interval) and 65\% within 0.03~m.')]),
P('sys', r'\caption{Convergence trajectories from eight starting positions', end=r'\end{figure}', back=3,
  subs=[('fig_hw_attraction_2026.pdf', 'ch6_fig_hw_attraction_2026.pdf'), STAB],
  width=r'0.9\textwidth'),
P('sys', r'\caption{Hardware experimental results, three robots (3R) and four (4R)', end=r'\end{table}', back=3,
  subs=[(r'\begin{tabular*}{\columnwidth}', r'\begin{tabular*}{0.8\textwidth}'),
        (r'with matched simulations of each attraction trial.',
         r'with matched simulations of each attraction trial. The final-error row gives the mean $\bar{e}_f$ and the standard deviation $s_e$ of the final error $e_f$.')]),

T(r"""
\subsection{Orbital Navigation}
\label{sec:ch6:hw_orbit}
"""),
P('sys', r'Orbital control was evaluated on both fields at commanded radii from 0.01~m to 0.50~m',
  end='No orbit run with either formation left the map.'),

T(r"""
\subsection{Classification}
\label{sec:ch6:hw_class}
"""),
P('sys', r'Across the orbit runs, the classifier matched the printed saddle'),

T(r"""
\subsection{Offline Classification of the Attraction Trials}
\label{sec:ch6:classification_offline}

% New text. Analysis: experiments/classify_from_hardware_logs.py, results
% in experiments/outputs/classify_hw/summary.json (VF_Robot), run
% 2026-09-29. Those logs are the three-robot attraction trials of the
% revision (2025 campaign) less saddle runs 43 and 64, which the extras
% later filled. It has not been re-run with the extras.
The same fit that locates the critical point also names its type. The
eigenvalues of $\hat{\mathbf{J}}$ place it in one of the classes of
Table~\ref{tab:ch3:field_summary}, at no extra sampling cost. The
attraction trials were designed to test location, but every control cycle
logged the positions and sensed readings needed to recompute
$\hat{\mathbf{J}}$. Recomputing it offline reproduces the controller's
logged critical point estimates to within $5\times10^{-7}$~m, so the
Jacobians classified here are the ones the controller computed. The
logs hold 78 saddle trials and 80 vortex trials of 101 cycles each. These
are the three-robot attraction trials of
Section~\ref{sec:ch6:hw_attraction}, less saddle runs 43 and 64, which
were filled later by extra runs.

Each estimate was classified as a saddle when $\det\hat{\mathbf{J}} < 0$,
a node when the eigenvalues are real and of one sign, and a rotational
type when they form a complex pair $\alpha \pm i\nu$. A measured center
never has $\alpha$ exactly zero, so a complex pair was labeled a center
when $|\alpha| < 0.2\,\nu$ and a spiral otherwise. That threshold is a
choice, and Table~\ref{tab:ch6:classification_hw} also gives the results
at 0.1 and 0.3.

\begin{table}[htbp]
\centering
\caption{Offline classification of the hardware Jacobian estimates.
The terminal window is the final 2~s of each trial, and the trial vote is
the majority label over that window.}
\label{tab:ch6:classification_hw}
\begin{tabular}{lcc}
\hline
 & Saddle map & Vortex map \\
\hline
Trials & 78 & 80 \\
Class correct (saddle, rotational, node), all cycles & 99.9\% & 99.9\% \\
Class correct, terminal window & 100\% & 100\% \\
Class correct, trial vote & 78/78 & 80/80 \\
Exact type, terminal window, threshold 0.1 & 100\% & 71\% \\
Exact type, terminal window, threshold 0.2 & 100\% & 89\% \\
Exact type, terminal window, threshold 0.3 & 100\% & 99\% \\
\hline
\end{tabular}
\end{table}

The estimate named the class of both fields correctly in effectively
every cycle, including cycles more than 0.8~m from the critical point.
Every vortex error is a stable spiral with a small negative real part,
and none is a saddle or a node. A small damping in the printed map or a
small bias in the fit is enough to turn a center into a weak spiral, and
the data cannot separate the two.

Classification carries one condition that location does not. A
reflection of the sensed vectors reverses the sign of
$\det\hat{\mathbf{J}}$, which exchanges saddles with nodes but leaves the
estimated critical point where it was. The two printed maps were placed
under different frame conventions, including reflections, rotations, and
camera mirroring. The results above use the registration from the field
reconstructions of Section~\ref{sec:ch2:reconstruction}, which negates
the sensed $v$ component for the saddle map only. The controller applied
no registration, since it used the estimate only for location, and its
unregistered Jacobian read the saddle map as a stable node or spiral
throughout the terminal window. A deployment that reports feature type
therefore needs its sensor frame registered to its navigation frame.

\section{Discussion}
\label{sec:ch6:discussion}

\subsection{Error Analysis}
\label{sec:ch6:error}
"""),
P('sys', r'Estimation and control errors arise from four sources',
  end='locates it as well as any other critical point',
  subs=[('Sections~II-B and~III-D', r'Sections~\ref{sec:ch6:estimation} and~\ref{sec:ch6:stability}'),
        ('Sections~IV and~V', r'Sections~\ref{sec:ch6:sim} and~\ref{sec:ch6:hw}'),
        STAB,
        (r'Section~\ref{sec:hw_testbed}', r'Section~\ref{sec:ch2:calibration}')]),

T(r"""
\subsection{The Orbit as a Limit Cycle}
\label{sec:ch6:limit_cycle}
"""),
P('sys', r'The orbit converges not to the commanded circle but to a limit cycle'),
P('sys', r'\caption{Limit cycle in the vortex at $r_d = 0.10$~m', end=r'\end{figure}', back=3,
  subs=[('fig_limit_cycle_2026.pdf', 'ch6_fig_limit_cycle_2026.pdf')],
  width=r'0.9\textwidth'),
T(r"""
% New text. Analysis: VF_Robot experiments/thesis_ch5_orbit_drift_check.py,
% output experiments/outputs/thesis_ch5/orbit_drift_check.txt (2026-11-19).
The same delay and lag account for the outward drift of the vector-sum
primitive in Chapter~\ref{ch:zeroth}, which has no radial term to oppose
it. For a cluster of speed $V$ and no radial correction, $r^2$ grows at
the constant rate $2TV^2$. The ten 2025 vector-sum runs on the fixed
vortex grew linearly in $r^2$, with $R^2$ from 0.79 to 0.93 after the
first second, at mean centroid speeds of 0.37 to 0.40~m/s. Their slopes
give a median $T$ of 0.16~s, with quartiles of 0.15 and 0.18~s, when the
map center is taken as the origin of the motion capture frame, and 0.14~s
when a common center is fitted to the ten runs. The delay and lag of
Section~\ref{sec:ch2:dynamics} give 0.125~s.
"""),

T(r"""
\subsection{Comparison with the Vector-to-Scalar Primitive}
\label{sec:ch6:comparison}
"""),
P('sys', r'The modular architecture allows navigation controllers to be interchanged',
  end='and all at 0.5~m.',
  subs=[STAB]),
P('sys', r'\caption{Comparison with the vector-to-scalar (V2S) primitive in simulation', end=r'\end{table}', back=3),

T(r"""
\section{Summary}
\label{sec:ch6:summary}
"""),
P('sys', r'This paper presents a framework for multirobot navigation in vector fields',
  end='natural next step toward field deployment.',
  subs=[('This paper presents', 'This chapter presented'),
        ('approach uses instantaneous', 'approach used instantaneous'),
        ('These results establish cooperative Jacobian estimation',
         "Offline, the hardware attraction estimates also named the type of both printed fields in every trial by majority vote, once each map's sensor frame was registered to the motion-capture frame. These results establish cooperative Jacobian estimation")]),
]

write('ch06_first_order.tex', parts)
